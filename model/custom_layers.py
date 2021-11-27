#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date:
#   Description :
#
# ================================================================
import paddle
import paddle.nn.functional as F
import paddle.nn as nn
import paddle.fluid as fluid
from paddle import ParamAttr
from paddle.regularizer import L2Decay
from paddle.nn.initializer import Uniform
from paddle.nn.initializer import Constant
from paddle.vision.ops import DeformConv2D




class MyDCNv2(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation=1,
                 groups=1,
                 weight_attr=None,
                 bias_attr=None):
        super(MyDCNv2, self).__init__()
        assert weight_attr is not False, "weight_attr should not be False in Conv."
        self.weight_attr = weight_attr
        self.bias_attr = bias_attr
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups.")
        self.groups = groups

        filter_shape = [out_channels, in_channels // groups, kernel_size, kernel_size]

        self.weight = self.create_parameter(
            shape=filter_shape,
            attr=self.weight_attr)
        self.bias = self.create_parameter(
            attr=self.bias_attr, shape=[out_channels, ], is_bias=True)

    def forward(self, x, offset, mask):
        in_C = self.in_channels
        out_C = self.out_channels
        stride = self.stride
        padding = self.padding
        # dilation = self.dilation
        groups = self.groups
        N, _, H, W = x.shape
        _, w_in, kH, kW = self.weight.shape
        out_W = (W + 2 * padding - (kW - 1)) // stride
        out_H = (H + 2 * padding - (kH - 1)) // stride

        # ================== 1.先对图片x填充得到填充后的图片pad_x ==================
        pad_x_H = H + padding * 2 + 1
        pad_x_W = W + padding * 2 + 1
        pad_x = F.pad(x, pad=[0, 0, 0, 0, padding, padding + 1, padding, padding + 1], value=0.0)

        # ================== 2.求所有采样点的坐标 ==================
        # 卷积核中心点在pad_x中的位置
        y_outer, x_outer = paddle.meshgrid([paddle.arange(out_H), paddle.arange(out_W)])
        y_outer = y_outer * stride + padding
        x_outer = x_outer * stride + padding
        start_pos_yx = paddle.stack((y_outer, x_outer), 2).cast(dtype='float32')  # [out_H, out_W, 2]         仅仅是卷积核中心点在pad_x中的位置
        start_pos_yx = paddle.unsqueeze(start_pos_yx, axis=[0, 3])                # [1, out_H, out_W, 1, 2]   仅仅是卷积核中心点在pad_x中的位置
        start_pos_yx = paddle.tile(start_pos_yx, [N, 1, 1, kH * kW, 1])  # [N, out_H, out_W, kH*kW, 2]   仅仅是卷积核中心点在pad_x中的位置
        start_pos_y = start_pos_yx[:, :, :, :, :1]  # [N, out_H, out_W, kH*kW, 1]   仅仅是卷积核中心点在pad_x中的位置
        start_pos_x = start_pos_yx[:, :, :, :, 1:]  # [N, out_H, out_W, kH*kW, 1]   仅仅是卷积核中心点在pad_x中的位置
        start_pos_y.stop_gradient = True
        start_pos_x.stop_gradient = True

        # 卷积核内部的偏移
        half_W = (kW - 1) // 2
        half_H = (kH - 1) // 2
        y_inner, x_inner = paddle.meshgrid([paddle.arange(kH), paddle.arange(kW)])
        y_inner -= half_H
        x_inner -= half_W
        filter_inner_offset_yx = paddle.stack((y_inner, x_inner), 2).cast(dtype='float32')     # [kH, kW, 2]       卷积核内部的偏移
        filter_inner_offset_yx = paddle.reshape(filter_inner_offset_yx, (1, 1, 1, kH * kW, 2))  # [1, 1, 1, kH*kW, 2]   卷积核内部的偏移
        filter_inner_offset_yx = paddle.tile(filter_inner_offset_yx, [N, out_H, out_W, 1, 1])  # [N, out_H, out_W, kH*kW, 2]   卷积核内部的偏移
        filter_inner_offset_y = filter_inner_offset_yx[:, :, :, :, :1]  # [N, out_H, out_W, kH*kW, 1]   卷积核内部的偏移
        filter_inner_offset_x = filter_inner_offset_yx[:, :, :, :, 1:]  # [N, out_H, out_W, kH*kW, 1]   卷积核内部的偏移
        filter_inner_offset_y.stop_gradient = True
        filter_inner_offset_x.stop_gradient = True

        # 预测的偏移
        offset = paddle.transpose(offset, [0, 2, 3, 1])  # [N, out_H, out_W, kH*kW*2]
        offset_yx = paddle.reshape(offset, (N, out_H, out_W, kH * kW, 2))  # [N, out_H, out_W, kH*kW, 2]
        offset_y = offset_yx[:, :, :, :, :1]  # [N, out_H, out_W, kH*kW, 1]
        offset_x = offset_yx[:, :, :, :, 1:]  # [N, out_H, out_W, kH*kW, 1]

        # 最终采样位置。
        pos_y = start_pos_y + filter_inner_offset_y + offset_y  # [N, out_H, out_W, kH*kW, 1]
        pos_x = start_pos_x + filter_inner_offset_x + offset_x  # [N, out_H, out_W, kH*kW, 1]
        pos_y = paddle.clip(pos_y, 0.0, H + padding * 2 - 1.0)  # 最终采样位置限制在pad_x内
        pos_x = paddle.clip(pos_x, 0.0, W + padding * 2 - 1.0)  # 最终采样位置限制在pad_x内

        # ================== 3.采样。用F.grid_sample()双线性插值采样。 ==================
        pos_x = pos_x / (pad_x_W - 1) * 2.0 - 1.0
        pos_y = pos_y / (pad_x_H - 1) * 2.0 - 1.0
        xtyt = paddle.concat([pos_x, pos_y], -1)  # [N, out_H, out_W, kH*kW, 2]
        xtyt = paddle.reshape(xtyt, (N, out_H, out_W * kH * kW, 2))  # [N, out_H, out_W*kH*kW, 2]
        value = F.grid_sample(pad_x, xtyt, mode='bilinear', padding_mode='zeros', align_corners=True)  # [N, in_C, out_H, out_W*kH*kW]
        value = paddle.reshape(value, (N, in_C, out_H, out_W, kH * kW))    # [N, in_C, out_H, out_W, kH * kW]
        value = value.transpose((0, 1, 4, 2, 3))                           # [N, in_C, kH * kW, out_H, out_W]

        # ================== 4.乘以重要程度 ==================
        # 乘以重要程度
        mask = paddle.unsqueeze(mask, [1])  # [N,    1, kH * kW, out_H, out_W]
        value = value * mask                # [N, in_C, kH * kW, out_H, out_W]
        new_x = paddle.reshape(value, (N, in_C * kH * kW, out_H, out_W))  # [N, in_C * kH * kW, out_H, out_W]

        # ================== 5.乘以本层的权重，加上偏置 ==================
        # 1x1卷积
        rw = paddle.reshape(self.weight, (out_C, w_in * kH * kW, 1, 1))  # [out_C, w_in, kH, kW] -> [out_C, w_in*kH*kW, 1, 1]  变成1x1卷积核
        out = F.conv2d(new_x, rw, bias=self.bias, stride=1, groups=groups)  # [N, out_C, out_H, out_W]
        return out


def get_norm(norm_type):
    bn = 0
    sync_bn = 0
    gn = 0
    af = 0
    if norm_type == 'bn':
        bn = 1
    elif norm_type == 'sync_bn':
        sync_bn = 1
    elif norm_type == 'gn':
        gn = 1
    elif norm_type == 'in':
        gn = 1
    elif norm_type == 'ln':
        gn = 1
    elif norm_type == 'affine_channel':
        af = 1
    return bn, sync_bn, gn, af




class Mish(paddle.nn.Layer):
    def __init__(self):
        super(Mish, self).__init__()

    def __call__(self, x):
        return x * paddle.tanh(F.softplus(x))


class Conv2dUnit(paddle.nn.Layer):
    def __init__(self,
                 input_dim,
                 filters,
                 filter_size,
                 stride=1,
                 bias_attr=False,
                 norm_type=None,
                 groups=1,
                 norm_groups=32,
                 act=None,
                 freeze_norm=False,
                 is_test=False,
                 norm_decay=0.,
                 lr=1.,
                 bias_lr=None,
                 weight_init=None,
                 bias_init=None,
                 use_dcn=False,
                 name=''):
        super(Conv2dUnit, self).__init__()
        self.filters = filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = (filter_size - 1) // 2
        self.act = act
        self.freeze_norm = freeze_norm
        self.is_test = is_test
        self.norm_decay = norm_decay
        self.use_dcn = use_dcn
        self.name = name

        # conv
        conv_name = name
        self.conv_offset = None
        if use_dcn:
            conv_battr = False
            if bias_attr:
                blr = lr
                if bias_lr:
                    blr = bias_lr
                conv_battr = ParamAttr(learning_rate=blr,
                                       initializer=bias_init,
                                       regularizer=L2Decay(0.))   # 不可以加正则化的参数：norm层(比如bn层、affine_channel层、gn层)的scale、offset；卷积层的偏移参数。

            self.offset_channel = 2 * filter_size**2
            self.mask_channel = filter_size**2

            self.conv_offset = nn.Conv2D(
                in_channels=input_dim,
                out_channels=3 * filter_size**2,
                kernel_size=filter_size,
                stride=stride,
                padding=self.padding,
                weight_attr=ParamAttr(initializer=Constant(0.)),
                bias_attr=ParamAttr(initializer=Constant(0.)))
            # 官方的DCNv2
            self.conv = DeformConv2D(
                in_channels=input_dim,
                out_channels=filters,
                kernel_size=filter_size,
                stride=stride,
                padding=self.padding,
                dilation=1,
                groups=groups,
                weight_attr=ParamAttr(learning_rate=lr),
                bias_attr=conv_battr)
            # 自实现的DCNv2
            # self.conv = MyDCNv2(
            #     in_channels=input_dim,
            #     out_channels=filters,
            #     kernel_size=filter_size,
            #     stride=stride,
            #     padding=self.padding,
            #     dilation=1,
            #     groups=groups,
            #     weight_attr=ParamAttr(learning_rate=lr),
            #     bias_attr=conv_battr)
        else:
            conv_battr = False
            if bias_attr:
                blr = lr
                if bias_lr:
                    blr = bias_lr
                conv_battr = ParamAttr(learning_rate=blr,
                                       initializer=bias_init,
                                       regularizer=L2Decay(0.))   # 不可以加正则化的参数：norm层(比如bn层、affine_channel层、gn层)的scale、offset；卷积层的偏移参数。
            self.conv = nn.Conv2D(
                in_channels=input_dim,
                out_channels=filters,
                kernel_size=filter_size,
                stride=stride,
                padding=self.padding,
                groups=groups,
                weight_attr=ParamAttr(learning_rate=lr, initializer=weight_init),
                bias_attr=conv_battr)


        # norm
        assert norm_type in [None, 'bn', 'sync_bn', 'gn', 'affine_channel', 'in', 'ln']
        bn, sync_bn, gn, af = get_norm(norm_type)
        if norm_type == 'in':
            norm_groups = filters
        if norm_type == 'ln':
            norm_groups = 1
        if conv_name == "conv1":
            norm_name = "bn_" + conv_name
            if gn:
                norm_name = "gn_" + conv_name
            if af:
                norm_name = "af_" + conv_name
        else:
            norm_name = "bn" + conv_name[3:]
            if gn:
                norm_name = "gn" + conv_name[3:]
            if af:
                norm_name = "af" + conv_name[3:]
        norm_lr = 0. if freeze_norm else lr
        pattr = ParamAttr(
            learning_rate=norm_lr,
            regularizer=L2Decay(norm_decay),   # 不可以加正则化的参数：norm层(比如bn层、affine_channel层、gn层)的scale、offset；卷积层的偏移参数。
            name=norm_name + "_scale",
            trainable=False if freeze_norm else True)
        battr = ParamAttr(
            learning_rate=norm_lr,
            regularizer=L2Decay(norm_decay),   # 不可以加正则化的参数：norm层(比如bn层、affine_channel层、gn层)的scale、offset；卷积层的偏移参数。
            name=norm_name + "_offset",
            trainable=False if freeze_norm else True)
        self.bn = None
        self.gn = None
        self.af = None
        if bn:
            self.bn = paddle.nn.BatchNorm2D(filters, weight_attr=pattr, bias_attr=battr)
        if sync_bn:
            self.bn = paddle.nn.SyncBatchNorm(filters, weight_attr=pattr, bias_attr=battr)
        if gn:
            self.gn = paddle.nn.GroupNorm(num_groups=norm_groups, num_channels=filters, weight_attr=pattr, bias_attr=battr)
        if af:
            self.af = True
            self.scale = self.create_parameter(
                shape=[filters],
                dtype='float32',
                attr=pattr,
                default_initializer=Constant(1.))
            self.offset = self.create_parameter(
                shape=[filters],
                dtype='float32',
                attr=battr,
                default_initializer=Constant(0.), is_bias=True)

        # act
        self.act = None
        if act == 'relu':
            self.act = paddle.nn.ReLU()
        elif act == 'leaky':
            self.act = paddle.nn.LeakyReLU(0.1)
        elif act == 'mish':
            self.act = Mish()
        elif act is None:
            pass
        else:
            raise NotImplementedError("Activation \'{}\' is not implemented.".format(act))


    def freeze(self):
        if self.conv is not None:
            if self.conv.weight is not None:
                self.conv.weight.stop_gradient = True
            if self.conv.bias is not None:
                self.conv.bias.stop_gradient = True
        if self.conv_offset is not None:
            if self.conv_offset.weight is not None:
                self.conv_offset.weight.stop_gradient = True
            if self.conv_offset.bias is not None:
                self.conv_offset.bias.stop_gradient = True
        if self.bn is not None:
            self.bn.weight.stop_gradient = True
            self.bn.bias.stop_gradient = True
        if self.gn is not None:
            self.gn.weight.stop_gradient = True
            self.gn.bias.stop_gradient = True
        if self.af is not None:
            self.scale.stop_gradient = True
            self.offset.stop_gradient = True

    def fix_bn(self):
        if self.bn is not None:
            self.bn.eval()

    def forward(self, x):
        if self.use_dcn:
            offset_mask = self.conv_offset(x)
            offset, mask = paddle.split(
                offset_mask,
                num_or_sections=[self.offset_channel, self.mask_channel],
                axis=1)
            mask = F.sigmoid(mask)
            x = self.conv(x, offset, mask=mask)
        else:
            x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.gn:
            x = self.gn(x)
        if self.af:
            x = x * self.scale.unsqueeze([0, 2, 3]) + self.offset.unsqueeze([0, 2, 3])
        if self.act:
            x = self.act(x)
        return x


class CoordConv(paddle.nn.Layer):
    def __init__(self, coord_conv=True):
        super(CoordConv, self).__init__()
        self.coord_conv = coord_conv

    def __call__(self, input):
        if not self.coord_conv:
            return input
        b = input.shape[0]
        h = input.shape[2]
        w = input.shape[3]
        eps = 1e-3
        x_range = paddle.arange(0, w - eps, 1., dtype='float32') / (w - 1) * 2.0 - 1
        y_range = paddle.arange(0, h - eps, 1., dtype='float32') / (h - 1) * 2.0 - 1
        # x_range = paddle.to_tensor(x_range, place=input.place)
        # y_range = paddle.to_tensor(y_range, place=input.place)
        x_range = paddle.reshape(x_range, (1, 1, 1, -1))  # [1, 1, 1, w]
        y_range = paddle.reshape(y_range, (1, 1, -1, 1))  # [1, 1, h, 1]
        x_range = paddle.tile(x_range, [b, 1, h, 1])  # [b, 1, h, w]
        y_range = paddle.tile(y_range, [b, 1, 1, w])  # [b, 1, h, w]
        offset = paddle.concat([input, x_range, y_range], axis=1)
        return offset


class SPP(paddle.nn.Layer):
    def __init__(self, seq='asc'):
        super(SPP, self).__init__()
        assert seq in ['desc', 'asc']
        self.seq = seq
        self.max_pool1 = paddle.nn.MaxPool2D(kernel_size=5, stride=1, padding=2)
        self.max_pool2 = paddle.nn.MaxPool2D(kernel_size=9, stride=1, padding=4)
        self.max_pool3 = paddle.nn.MaxPool2D(kernel_size=13, stride=1, padding=6)

    def __call__(self, x):
        x_1 = x
        x_2 = self.max_pool1(x)
        x_3 = self.max_pool2(x)
        x_4 = self.max_pool3(x)
        if self.seq == 'desc':
            out = paddle.concat([x_4, x_3, x_2, x_1], axis=1)
        else:
            out = paddle.concat([x_1, x_2, x_3, x_4], axis=1)
        return out


class DropBlock(paddle.nn.Layer):
    def __init__(self,
                 block_size=3,
                 keep_prob=0.9,
                 is_test=False):
        super(DropBlock, self).__init__()
        self.block_size = block_size
        self.keep_prob = keep_prob
        self.is_test = is_test

    def __call__(self, input):
        if self.is_test:
            return input

        def CalculateGamma(input, block_size, keep_prob):
            input_shape = paddle.shape(input)
            feat_shape_tmp = fluid.layers.slice(input_shape, [0], [3], [4])
            feat_shape_tmp = paddle.cast(feat_shape_tmp, dtype="float32")
            feat_shape_t = paddle.reshape(feat_shape_tmp, [1, 1, 1, 1])
            feat_area = paddle.pow(feat_shape_t, 2)

            block_shape_t = fluid.layers.fill_constant(
                shape=[1, 1, 1, 1], value=block_size, dtype='float32')
            block_area = fluid.layers.pow(block_shape_t, factor=2)

            useful_shape_t = feat_shape_t - block_shape_t + 1
            useful_area = fluid.layers.pow(useful_shape_t, factor=2)

            upper_t = feat_area * (1 - keep_prob)
            bottom_t = block_area * useful_area
            output = upper_t / bottom_t
            return output

        gamma = CalculateGamma(input, block_size=self.block_size, keep_prob=self.keep_prob)
        input_shape = fluid.layers.shape(input)
        p = fluid.layers.expand_as(gamma, input)

        input_shape_tmp = fluid.layers.cast(input_shape, dtype="int64")
        random_matrix = fluid.layers.uniform_random(
            input_shape_tmp, dtype='float32', min=0.0, max=1.0)
        one_zero_m = fluid.layers.less_than(random_matrix, p)
        one_zero_m.stop_gradient = True
        one_zero_m = fluid.layers.cast(one_zero_m, dtype="float32")

        mask_flag = fluid.layers.pool2d(
            one_zero_m,
            pool_size=self.block_size,
            pool_type='max',
            pool_stride=1,
            pool_padding=self.block_size // 2)
        mask = 1.0 - mask_flag

        elem_numel = fluid.layers.reduce_prod(input_shape)
        elem_numel_m = fluid.layers.cast(elem_numel, dtype="float32")
        elem_numel_m.stop_gradient = True

        elem_sum = fluid.layers.reduce_sum(mask)
        elem_sum_m = fluid.layers.cast(elem_sum, dtype="float32")
        elem_sum_m.stop_gradient = True

        output = input * mask * elem_numel_m / elem_sum_m
        return output


class PointGenerator(object):

    def _meshgrid(self, x, y, w, h, row_major=True):
        xx = paddle.tile(paddle.reshape(x, (1, -1)), [h, 1])
        yy = paddle.tile(paddle.reshape(y, (-1, 1)), [1, w])

        xx = paddle.reshape(xx, (-1, ))
        yy = paddle.reshape(yy, (-1, ))
        if row_major:
            return xx, yy
        else:
            return yy, xx

    def grid_points(self, featmap_size, stride=16):
        feat_h, feat_w = featmap_size
        eps = 1e-3
        shift_x = paddle.arange(0., feat_w - eps, 1., dtype='float32') * stride
        shift_y = paddle.arange(0., feat_h - eps, 1., dtype='float32') * stride

        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y, feat_w, feat_h)
        stride = paddle.full(shape=shift_xx.shape, fill_value=stride, dtype='float32')
        all_points = paddle.stack([shift_xx, shift_yy, stride], axis=-1)
        return all_points

    def valid_flags(self, featmap_size, valid_size, device='cuda'):
        # feat_h, feat_w = featmap_size
        # valid_h, valid_w = valid_size
        # assert valid_h <= feat_h and valid_w <= feat_w
        # valid_x = torch.zeros(feat_w, dtype=torch.bool, device=device)
        # valid_y = torch.zeros(feat_h, dtype=torch.bool, device=device)
        # valid_x[:valid_w] = 1
        # valid_y[:valid_h] = 1
        # valid_xx, valid_yy = self._meshgrid(valid_x, valid_y)
        # valid = valid_xx & valid_yy
        # return valid
        pass




