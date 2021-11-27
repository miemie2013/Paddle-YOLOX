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
from paddle import nn
import paddle.nn.functional as F
from model.custom_layers import *


class ResidualBlock(paddle.nn.Layer):
    def __init__(self, input_dim, filters_1, filters_2, norm_type, name=''):
        super(ResidualBlock, self).__init__()
        self.conv1 = Conv2dUnit(input_dim, filters_1, 1, stride=1, bias_attr=False, norm_type=norm_type, act='mish', name=name+'.conv1')
        self.conv2 = Conv2dUnit(filters_1, filters_2, 3, stride=1, bias_attr=False, norm_type=norm_type, act='mish', name=name+'.conv2')

    def forward(self, input):
        residual = input
        x = self.conv1(input)
        x = self.conv2(x)
        x = residual + x
        return x

    def freeze(self):
        self.conv1.freeze()
        self.conv2.freeze()

    def fix_bn(self):
        self.conv1.fix_bn()
        self.conv2.fix_bn()


class StackResidualBlock(paddle.nn.Layer):
    def __init__(self, input_dim, filters_1, filters_2, n, norm_type, name=''):
        super(StackResidualBlock, self).__init__()
        self.sequential = paddle.nn.LayerList()
        for i in range(n):
            residual_block = ResidualBlock(input_dim, filters_1, filters_2, norm_type, name=name+'.block%d' % (i,))
            self.sequential.append(residual_block)

    def forward(self, x):
        for residual_block in self.sequential:
            x = residual_block(x)
        return x

    def freeze(self):
        for residual_block in self.sequential:
            residual_block.freeze()

    def fix_bn(self):
        for residual_block in self.sequential:
            residual_block.fix_bn()



class CSPDarknet53(paddle.nn.Layer):
    def __init__(self, norm_type='bn', feature_maps=[3, 4, 5], freeze_at=0, fix_bn_mean_var_at=0, lr_mult_list=[1., 1., 1., 1.]):
        super(CSPDarknet53, self).__init__()
        self.norm_type = norm_type
        self.feature_maps = feature_maps
        assert freeze_at in [0, 1, 2, 3, 4, 5]
        assert fix_bn_mean_var_at in [0, 1, 2, 3, 4, 5]
        # assert len(lr_mult_list) == 4, "lr_mult_list length must be 4 but got {}".format(len(lr_mult_list))
        # self.lr_mult_list = lr_mult_list
        self.freeze_at = freeze_at
        self.fix_bn_mean_var_at = fix_bn_mean_var_at
        self.conv1 = Conv2dUnit(3,  32, 3, stride=1, norm_type=norm_type, act='mish', name='backbone.conv1')

        # stage1
        self.stage1_conv1 = Conv2dUnit(32, 64, 3, stride=2, norm_type=norm_type, act='mish', name='backbone.stage1_conv1')
        self.stage1_conv2 = Conv2dUnit(64, 64, 1, stride=1, norm_type=norm_type, act='mish', name='backbone.stage1_conv2')
        self.stage1_conv3 = Conv2dUnit(64, 64, 1, stride=1, norm_type=norm_type, act='mish', name='backbone.stage1_conv3')
        self.stage1_blocks = StackResidualBlock(64, 32, 64, n=1, norm_type=norm_type, name='backbone.stage1_blocks')
        self.stage1_conv4 = Conv2dUnit(64, 64, 1, stride=1, norm_type=norm_type, act='mish', name='backbone.stage1_conv4')
        self.stage1_conv5 = Conv2dUnit(128, 64, 1, stride=1, norm_type=norm_type, act='mish', name='backbone.stage1_conv5')

        # stage2
        self.stage2_conv1 = Conv2dUnit(64, 128, 3, stride=2, norm_type=norm_type, act='mish', name='backbone.stage2_conv1')
        self.stage2_conv2 = Conv2dUnit(128, 64, 1, stride=1, norm_type=norm_type, act='mish', name='backbone.stage2_conv2')
        self.stage2_conv3 = Conv2dUnit(128, 64, 1, stride=1, norm_type=norm_type, act='mish', name='backbone.stage2_conv3')
        self.stage2_blocks = StackResidualBlock(64, 64, 64, n=2, norm_type=norm_type, name='backbone.stage2_blocks')
        self.stage2_conv4 = Conv2dUnit(64, 64, 1, stride=1, norm_type=norm_type, act='mish', name='backbone.stage2_conv4')
        self.stage2_conv5 = Conv2dUnit(128, 128, 1, stride=1, norm_type=norm_type, act='mish', name='backbone.stage2_conv5')

        # stage3
        self.stage3_conv1 = Conv2dUnit(128, 256, 3, stride=2, norm_type=norm_type, act='mish', name='backbone.stage3_conv1')
        self.stage3_conv2 = Conv2dUnit(256, 128, 1, stride=1, norm_type=norm_type, act='mish', name='backbone.stage3_conv2')
        self.stage3_conv3 = Conv2dUnit(256, 128, 1, stride=1, norm_type=norm_type, act='mish', name='backbone.stage3_conv3')
        self.stage3_blocks = StackResidualBlock(128, 128, 128, n=8, norm_type=norm_type, name='backbone.stage3_blocks')
        self.stage3_conv4 = Conv2dUnit(128, 128, 1, stride=1, norm_type=norm_type, act='mish', name='backbone.stage3_conv4')
        self.stage3_conv5 = Conv2dUnit(256, 256, 1, stride=1, norm_type=norm_type, act='mish', name='backbone.stage3_conv5')

        # stage4
        self.stage4_conv1 = Conv2dUnit(256, 512, 3, stride=2, norm_type=norm_type, act='mish', name='backbone.stage4_conv1')
        self.stage4_conv2 = Conv2dUnit(512, 256, 1, stride=1, norm_type=norm_type, act='mish', name='backbone.stage4_conv2')
        self.stage4_conv3 = Conv2dUnit(512, 256, 1, stride=1, norm_type=norm_type, act='mish', name='backbone.stage4_conv3')
        self.stage4_blocks = StackResidualBlock(256, 256, 256, n=8, norm_type=norm_type, name='backbone.stage4_blocks')
        self.stage4_conv4 = Conv2dUnit(256, 256, 1, stride=1, norm_type=norm_type, act='mish', name='backbone.stage4_conv4')
        self.stage4_conv5 = Conv2dUnit(512, 512, 1, stride=1, norm_type=norm_type, act='mish', name='backbone.stage4_conv5')

        # stage5
        self.stage5_conv1 = Conv2dUnit(512, 1024, 3, stride=2, norm_type=norm_type, act='mish', name='backbone.stage5_conv1')
        self.stage5_conv2 = Conv2dUnit(1024, 512, 1, stride=1, norm_type=norm_type, act='mish', name='backbone.stage5_conv2')
        self.stage5_conv3 = Conv2dUnit(1024, 512, 1, stride=1, norm_type=norm_type, act='mish', name='backbone.stage5_conv3')
        self.stage5_blocks = StackResidualBlock(512, 512, 512, n=4, norm_type=norm_type, name='backbone.stage5_blocks')
        self.stage5_conv4 = Conv2dUnit(512, 512, 1, stride=1, norm_type=norm_type, act='mish', name='backbone.stage5_conv4')
        self.stage5_conv5 = Conv2dUnit(1024, 1024, 1, stride=1, norm_type=norm_type, act='mish', name='backbone.stage5_conv5')

    def forward(self, input_tensor):
        x = self.conv1(input_tensor)

        # stage1
        x = self.stage1_conv1(x)
        s2 = self.stage1_conv2(x)
        x = self.stage1_conv3(x)
        x = self.stage1_blocks(x)
        x = self.stage1_conv4(x)
        x = L.concat([x, s2], 1)
        s2 = self.stage1_conv5(x)
        # stage2
        x = self.stage2_conv1(s2)
        s4 = self.stage2_conv2(x)
        x = self.stage2_conv3(x)
        x = self.stage2_blocks(x)
        x = self.stage2_conv4(x)
        x = L.concat([x, s4], 1)
        s4 = self.stage2_conv5(x)
        # stage3
        x = self.stage3_conv1(s4)
        s8 = self.stage3_conv2(x)
        x = self.stage3_conv3(x)
        x = self.stage3_blocks(x)
        x = self.stage3_conv4(x)
        x = L.concat([x, s8], 1)
        s8 = self.stage3_conv5(x)
        # stage4
        x = self.stage4_conv1(s8)
        s16 = self.stage4_conv2(x)
        x = self.stage4_conv3(x)
        x = self.stage4_blocks(x)
        x = self.stage4_conv4(x)
        x = L.concat([x, s16], 1)
        s16 = self.stage4_conv5(x)
        # stage5
        x = self.stage5_conv1(s16)
        s32 = self.stage5_conv2(x)
        x = self.stage5_conv3(x)
        x = self.stage5_blocks(x)
        x = self.stage5_conv4(x)
        x = L.concat([x, s32], 1)
        s32 = self.stage5_conv5(x)

        outs = []
        if 2 in self.feature_maps:
            outs.append(s4)
        if 3 in self.feature_maps:
            outs.append(s8)
        if 4 in self.feature_maps:
            outs.append(s16)
        if 5 in self.feature_maps:
            outs.append(s32)
        return outs

    def get_block(self, name):
        layer = getattr(self, name)
        return layer

    def freeze(self):
        freeze_at = self.freeze_at
        if freeze_at >= 1:
            self.conv1.freeze()
            self.stage1_conv1.freeze()
            self.stage1_conv2.freeze()
            self.stage1_conv3.freeze()
            self.stage1_blocks.freeze()
            self.stage1_conv4.freeze()
            self.stage1_conv5.freeze()
        if freeze_at >= 2:
            self.stage2_conv1.freeze()
            self.stage2_conv2.freeze()
            self.stage2_conv3.freeze()
            self.stage2_blocks.freeze()
            self.stage2_conv4.freeze()
            self.stage2_conv5.freeze()
        if freeze_at >= 3:
            self.stage3_conv1.freeze()
            self.stage3_conv2.freeze()
            self.stage3_conv3.freeze()
            self.stage3_blocks.freeze()
            self.stage3_conv4.freeze()
            self.stage3_conv5.freeze()
        if freeze_at >= 4:
            self.stage4_conv1.freeze()
            self.stage4_conv2.freeze()
            self.stage4_conv3.freeze()
            self.stage4_blocks.freeze()
            self.stage4_conv4.freeze()
            self.stage4_conv5.freeze()
        if freeze_at >= 5:
            self.stage5_conv1.freeze()
            self.stage5_conv2.freeze()
            self.stage5_conv3.freeze()
            self.stage5_blocks.freeze()
            self.stage5_conv4.freeze()
            self.stage5_conv5.freeze()

    def fix_bn(self):
        fix_bn_mean_var_at = self.fix_bn_mean_var_at
        if fix_bn_mean_var_at >= 1:
            self.conv1.fix_bn()
            self.stage1_conv1.fix_bn()
            self.stage1_conv2.fix_bn()
            self.stage1_conv3.fix_bn()
            self.stage1_blocks.fix_bn()
            self.stage1_conv4.fix_bn()
            self.stage1_conv5.fix_bn()
        if fix_bn_mean_var_at >= 2:
            self.stage2_conv1.fix_bn()
            self.stage2_conv2.fix_bn()
            self.stage2_conv3.fix_bn()
            self.stage2_blocks.fix_bn()
            self.stage2_conv4.fix_bn()
            self.stage2_conv5.fix_bn()
        if fix_bn_mean_var_at >= 3:
            self.stage3_conv1.fix_bn()
            self.stage3_conv2.fix_bn()
            self.stage3_conv3.fix_bn()
            self.stage3_blocks.fix_bn()
            self.stage3_conv4.fix_bn()
            self.stage3_conv5.fix_bn()
        if fix_bn_mean_var_at >= 4:
            self.stage4_conv1.fix_bn()
            self.stage4_conv2.fix_bn()
            self.stage4_conv3.fix_bn()
            self.stage4_blocks.fix_bn()
            self.stage4_conv4.fix_bn()
            self.stage4_conv5.fix_bn()
        if fix_bn_mean_var_at >= 5:
            self.stage5_conv1.fix_bn()
            self.stage5_conv2.fix_bn()
            self.stage5_conv3.fix_bn()
            self.stage5_blocks.fix_bn()
            self.stage5_conv4.fix_bn()
            self.stage5_conv5.fix_bn()

def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = nn.Silu()
    elif name == "relu":
        module = nn.ReLU()
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


class BaseConv(nn.Layer):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(
        self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"
    ):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2

        conv_battr = False
        bias_init = None
        if bias:
            blr = 1.0
            conv_battr = ParamAttr(
                                   # name=conv_name + "_bias",
                                   learning_rate=blr,
                                   initializer=bias_init,
                                   regularizer=L2Decay(0.))  # 不可以加正则化的参数：norm层(比如bn层、affine_channel层、gn层)的scale、offset；卷积层的偏移参数。
        self.conv = nn.Conv2D(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias_attr=conv_battr,
        )

        norm_lr = 1.0
        norm_decay = 0.0  # 不可以加正则化的参数：norm层(比如bn层、affine_channel层、gn层)的scale、offset；卷积层的偏移参数。
        pattr = ParamAttr(
            learning_rate=norm_lr,
            regularizer=L2Decay(norm_decay))
        battr = ParamAttr(
            learning_rate=norm_lr,
            regularizer=L2Decay(norm_decay))

        self.bn = nn.BatchNorm2D(out_channels, weight_attr=pattr, bias_attr=battr,
                                 momentum=0.97, epsilon=1e-3)   # YOLOX中momentum和epsilon的值为这两个
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

    def freeze(self):
        if self.conv.weight is not None:
            self.conv.weight.stop_gradient = True
        if self.conv.bias is not None:
            self.conv.bias.stop_gradient = True
        self.bn.weight.stop_gradient = True
        self.bn.bias.stop_gradient = True

    def fix_bn(self):
        self.bn.eval()


class DWConv(nn.Layer):
    """Depthwise Conv + Conv"""

    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu"):
        super().__init__()
        self.dconv = BaseConv(
            in_channels,
            in_channels,
            ksize=ksize,
            stride=stride,
            groups=in_channels,
            act=act,
        )
        self.pconv = BaseConv(
            in_channels, out_channels, ksize=1, stride=1, groups=1, act=act
        )

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)

    def freeze(self):
        self.dconv.freeze()
        self.pconv.freeze()

    def fix_bn(self):
        self.dconv.fix_bn()
        self.pconv.fix_bn()


class Bottleneck(nn.Layer):
    # Standard bottleneck
    def __init__(
        self,
        in_channels,
        out_channels,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = DWConv if depthwise else BaseConv
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, act=act)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        temp1 = self.conv1(x)
        y = self.conv2(temp1)
        if self.use_add:
            y = y + x
        return y

    def freeze(self):
        self.conv1.freeze()
        self.conv2.freeze()

    def fix_bn(self):
        self.conv1.fix_bn()
        self.conv2.fix_bn()



class ResLayer(nn.Layer):
    "Residual layer with `in_channels` inputs."

    def __init__(self, in_channels: int):
        super().__init__()
        mid_channels = in_channels // 2
        self.layer1 = BaseConv(
            in_channels, mid_channels, ksize=1, stride=1, act="lrelu"
        )
        self.layer2 = BaseConv(
            mid_channels, in_channels, ksize=3, stride=1, act="lrelu"
        )

    def forward(self, x):
        out = self.layer2(self.layer1(x))
        return x + out

class SPPBottleneck(nn.Layer):
    """Spatial pyramid pooling layer used in YOLOv3-SPP"""

    def __init__(
        self, in_channels, out_channels, kernel_sizes=(5, 9, 13), activation="silu"
    ):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=activation)
        self.m = nn.LayerList(
            [
                nn.MaxPool2D(kernel_size=ks, stride=1, padding=ks // 2)
                for ks in kernel_sizes
            ]
        )
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = BaseConv(conv2_channels, out_channels, 1, stride=1, act=activation)

    def forward(self, x):
        x = self.conv1(x)
        x = paddle.concat([x] + [m(x) for m in self.m], axis=1)
        x = self.conv2(x)
        return x

    def freeze(self):
        self.conv1.freeze()
        self.conv2.freeze()

    def fix_bn(self):
        self.conv1.fix_bn()
        self.conv2.fix_bn()


class CSPLayer(paddle.nn.Layer):
    """C3 in yolov5, CSP Bottleneck with 3 convolutions"""

    def __init__(
        self,
        in_channels,
        out_channels,
        n=1,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act="silu",
    ):
        """
        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            n (int): number of Bottlenecks. Default value: 1.
        """
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)  # hidden channels
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)
        module_list = [
            Bottleneck(
                hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act
            )
            for _ in range(n)
        ]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = paddle.concat([x_1, x_2], axis=1)
        return self.conv3(x)

    def freeze(self):
        self.conv1.freeze()
        self.conv2.freeze()
        self.conv3.freeze()
        for layer in self.m:
            layer.freeze()

    def fix_bn(self):
        self.conv1.fix_bn()
        self.conv2.fix_bn()
        self.conv3.fix_bn()
        for layer in self.m:
            layer.fix_bn()


class Focus(paddle.nn.Layer):
    """Focus width and height information into channel space."""

    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="silu"):
        super().__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride, act=act)

    def forward(self, x):
        # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
        patch_top_left = x[:, :, ::2, ::2]
        patch_top_right = x[:, :, ::2, 1::2]
        patch_bot_left = x[:, :, 1::2, ::2]
        patch_bot_right = x[:, :, 1::2, 1::2]
        x = paddle.concat(
            [
                patch_top_left,
                patch_bot_left,
                patch_top_right,
                patch_bot_right,
            ],
            axis=1,
        )
        x.stop_gradient = True
        return self.conv(x)

    def freeze(self):
        self.conv.freeze()

    def fix_bn(self):
        self.conv.fix_bn()





class Darknet(paddle.nn.Layer):
    # number of blocks from dark2 to dark5.
    depth2blocks = {21: [1, 2, 2, 1], 53: [2, 8, 8, 4]}

    def __init__(
        self,
        depth,
        in_channels=3,
        stem_out_channels=32,
        out_features=("dark3", "dark4", "dark5"),
    ):
        """
        Args:
            depth (int): depth of darknet used in model, usually use [21, 53] for this param.
            in_channels (int): number of input channels, for example, use 3 for RGB image.
            stem_out_channels (int): number of output chanels of darknet stem.
                It decides channels of darknet layer2 to layer5.
            out_features (Tuple[str]): desired output layer name.
        """
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features
        self.stem = nn.Sequential(
            BaseConv(in_channels, stem_out_channels, ksize=3, stride=1, act="lrelu"),
            *self.make_group_layer(stem_out_channels, num_blocks=1, stride=2),
        )
        in_channels = stem_out_channels * 2  # 64

        num_blocks = Darknet.depth2blocks[depth]
        # create darknet with `stem_out_channels` and `num_blocks` layers.
        # to make model structure more clear, we don't use `for` statement in python.
        self.dark2 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[0], stride=2)
        )
        in_channels *= 2  # 128
        self.dark3 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[1], stride=2)
        )
        in_channels *= 2  # 256
        self.dark4 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[2], stride=2)
        )
        in_channels *= 2  # 512

        self.dark5 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[3], stride=2),
            *self.make_spp_block([in_channels, in_channels * 2], in_channels * 2),
        )

    def make_group_layer(self, in_channels: int, num_blocks: int, stride: int = 1):
        "starts with conv layer then has `num_blocks` `ResLayer`"
        return [
            BaseConv(in_channels, in_channels * 2, ksize=3, stride=stride, act="lrelu"),
            *[(ResLayer(in_channels * 2)) for _ in range(num_blocks)],
        ]

    def make_spp_block(self, filters_list, in_filters):
        m = nn.Sequential(
            *[
                BaseConv(in_filters, filters_list[0], 1, stride=1, act="lrelu"),
                BaseConv(filters_list[0], filters_list[1], 3, stride=1, act="lrelu"),
                SPPBottleneck(
                    in_channels=filters_list[1],
                    out_channels=filters_list[0],
                    activation="lrelu",
                ),
                BaseConv(filters_list[0], filters_list[1], 3, stride=1, act="lrelu"),
                BaseConv(filters_list[1], filters_list[0], 1, stride=1, act="lrelu"),
            ]
        )
        return m

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs["stem"] = x
        x = self.dark2(x)
        outputs["dark2"] = x
        x = self.dark3(x)
        outputs["dark3"] = x
        x = self.dark4(x)
        outputs["dark4"] = x
        x = self.dark5(x)
        outputs["dark5"] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}


class CSPDarknet(paddle.nn.Layer):
    def __init__(
        self,
        dep_mul,
        wid_mul,
        out_features=("dark3", "dark4", "dark5"),
        depthwise=False,
        act="silu",
        freeze_at=0,
        fix_bn_mean_var_at=0,
    ):
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features
        Conv = DWConv if depthwise else BaseConv

        self.freeze_at = freeze_at
        self.fix_bn_mean_var_at = fix_bn_mean_var_at

        base_channels = int(wid_mul * 64)  # 64
        base_depth = max(round(dep_mul * 3), 1)  # 3

        # stem
        self.stem = Focus(3, base_channels, ksize=3, act=act)

        # dark2
        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2, act=act),
            CSPLayer(
                base_channels * 2,
                base_channels * 2,
                n=base_depth,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark3
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2, act=act),
            CSPLayer(
                base_channels * 4,
                base_channels * 4,
                n=base_depth * 3,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark4
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2, act=act),
            CSPLayer(
                base_channels * 8,
                base_channels * 8,
                n=base_depth * 3,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark5
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, 3, 2, act=act),
            SPPBottleneck(base_channels * 16, base_channels * 16, activation=act),
            CSPLayer(
                base_channels * 16,
                base_channels * 16,
                n=base_depth,
                shortcut=False,
                depthwise=depthwise,
                act=act,
            ),
        )

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs["stem"] = x
        # print('bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbackbone')
        # print(self.stem.conv.conv.weight[0][0])
        # print(x)
        x = self.dark2(x)
        outputs["dark2"] = x
        x = self.dark3(x)
        outputs["dark3"] = x
        x = self.dark4(x)
        outputs["dark4"] = x
        x = self.dark5(x)
        outputs["dark5"] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}

    def freeze(self):
        freeze_at = self.freeze_at
        if freeze_at >= 1:
            self.stem.freeze()
        if freeze_at >= 2:
            for layer in self.dark2:
                layer.freeze()
        if freeze_at >= 3:
            for layer in self.dark3:
                layer.freeze()
        if freeze_at >= 4:
            for layer in self.dark4:
                layer.freeze()
        if freeze_at >= 5:
            for layer in self.dark5:
                layer.freeze()

    def fix_bn(self):
        fix_bn_mean_var_at = self.fix_bn_mean_var_at
        if fix_bn_mean_var_at >= 1:
            self.stem.fix_bn()
        if fix_bn_mean_var_at >= 2:
            for layer in self.dark2:
                layer.fix_bn()
        if fix_bn_mean_var_at >= 3:
            for layer in self.dark3:
                layer.fix_bn()
        if fix_bn_mean_var_at >= 4:
            for layer in self.dark4:
                layer.fix_bn()
        if fix_bn_mean_var_at >= 5:
            for layer in self.dark5:
                layer.fix_bn()




