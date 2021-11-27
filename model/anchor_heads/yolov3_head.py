#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-10-23 09:13:23
#   Description : paddle2.0_ppyolo
#
# ================================================================
import numpy as np
import paddle
import paddle.nn.functional as F
import paddle.fluid as fluid
import copy
import math

from model.custom_layers import *
from model.matrix_nms import matrix_nms


def _split_ioup(output, an_num, num_classes):
    """
    Split new output feature map to output, predicted iou
    along channel dimension
    """
    ioup = output[:, :an_num, :, :]
    ioup = F.sigmoid(ioup)

    oriout = output[:, an_num:, :, :]

    return (ioup, oriout)


# sigmoid()函数的反函数。先取倒数再减一，取对数再取相反数。
def _de_sigmoid(x, eps=1e-7):
    # x限制在区间[eps, 1 / eps]内
    x = paddle.clip(x, eps, 1 / eps)

    # 先取倒数再减一
    x = 1.0 / x - 1.0

    # e^(-x)限制在区间[eps, 1 / eps]内
    x = paddle.clip(x, eps, 1 / eps)

    # 取对数再取相反数
    x = -paddle.log(x)
    return x


def _postprocess_output(ioup, output, an_num, num_classes, iou_aware_factor):
    """
    post process output objectness score
    """
    tensors = []
    stride = output.shape[1] // an_num
    for m in range(an_num):
        tensors.append(output[:, stride * m:stride * m + 4, :, :])
        obj = output[:, stride * m + 4:stride * m + 5, :, :]
        obj = F.sigmoid(obj)

        ip = ioup[:, m:m + 1, :, :]

        new_obj = paddle.pow(obj, (1 - iou_aware_factor)) * paddle.pow(ip, iou_aware_factor)
        new_obj = _de_sigmoid(new_obj)   # 置信位未进行sigmoid()激活

        tensors.append(new_obj)

        tensors.append(output[:, stride * m + 5:stride * m + 5 + num_classes, :, :])

    output = paddle.concat(tensors, axis=1)

    return output



def get_iou_aware_score(output, an_num, num_classes, iou_aware_factor):
    ioup, output = _split_ioup(output, an_num, num_classes)
    output = _postprocess_output(ioup, output, an_num, num_classes, iou_aware_factor)
    return output




class DetectionBlock(paddle.nn.Layer):
    def __init__(self,
                 in_c,
                 channel,
                 coord_conv=True,
                 norm_type=None,
                 norm_decay=0.,
                 conv_block_num=2,
                 is_first=False,
                 use_spp=True,
                 drop_block=True,
                 block_size=3,
                 keep_prob=0.9,
                 is_test=True,
                 name=''):
        super(DetectionBlock, self).__init__()
        assert channel % 2 == 0, \
            "channel {} cannot be divided by 2".format(channel)
        self.norm_decay = norm_decay
        self.use_spp = use_spp
        self.coord_conv = coord_conv
        self.is_first = is_first
        self.is_test = is_test
        self.drop_block = drop_block
        self.block_size = block_size
        self.keep_prob = keep_prob

        self.layers = paddle.nn.LayerList()
        self.tip_layers = paddle.nn.LayerList()
        for j in range(conv_block_num):
            coordConv = CoordConv(coord_conv)
            input_c = in_c + 2 if coord_conv else in_c
            conv_unit1 = Conv2dUnit(input_c, channel, 1, stride=1, norm_type=norm_type, act='leaky', norm_decay=self.norm_decay, name='{}.{}.0'.format(name, j))
            self.layers.append(coordConv)
            self.layers.append(conv_unit1)
            if self.use_spp and is_first and j == 1:
                spp = SPP()
                conv_unit2 = Conv2dUnit(channel * 4, 512, 1, stride=1, norm_type=norm_type, act='leaky', norm_decay=self.norm_decay, name='{}.{}.spp.conv'.format(name, j))
                conv_unit3 = Conv2dUnit(512, channel * 2, 3, stride=1, norm_type=norm_type, act='leaky', norm_decay=self.norm_decay, name='{}.{}.1'.format(name, j))
                self.layers.append(spp)
                self.layers.append(conv_unit2)
                self.layers.append(conv_unit3)
            else:
                conv_unit3 = Conv2dUnit(channel, channel * 2, 3, stride=1, norm_type=norm_type, act='leaky', norm_decay=self.norm_decay, name='{}.{}.1'.format(name, j))
                self.layers.append(conv_unit3)

            if self.drop_block and j == 0 and not is_first:
                dropBlock = DropBlock(
                    block_size=self.block_size,
                    keep_prob=self.keep_prob,
                    is_test=is_test)
                self.layers.append(dropBlock)
            in_c = channel * 2

        if self.drop_block and is_first:
            dropBlock = DropBlock(
                block_size=self.block_size,
                keep_prob=self.keep_prob,
                is_test=is_test)
            self.layers.append(dropBlock)
        coordConv = CoordConv(coord_conv)
        if conv_block_num == 0:
            input_c = in_c + 2 if coord_conv else in_c
        else:
            input_c = channel * 2 + 2 if coord_conv else channel * 2
        conv_unit = Conv2dUnit(input_c, channel, 1, stride=1, norm_type=norm_type, act='leaky', norm_decay=self.norm_decay, name='{}.2'.format(name))
        self.layers.append(coordConv)
        self.layers.append(conv_unit)

        coordConv = CoordConv(coord_conv)
        input_c = channel + 2 if coord_conv else channel
        conv_unit = Conv2dUnit(input_c, channel * 2, 3, stride=1, norm_type=norm_type, act='leaky', norm_decay=self.norm_decay, name='{}.tip'.format(name))
        self.tip_layers.append(coordConv)
        self.tip_layers.append(conv_unit)

    def __call__(self, input):
        conv = input
        for ly in self.layers:
            conv = ly(conv)
        route = conv
        tip = conv
        for ly in self.tip_layers:
            tip = ly(tip)
        return route, tip


class YOLOv3Head(paddle.nn.Layer):
    def __init__(self,
                 conv_block_num=2,
                 num_classes=80,
                 anchors=[[10, 13], [16, 30], [33, 23],
                          [30, 61], [62, 45], [59, 119],
                          [116, 90], [156, 198], [373, 326]],
                 anchor_masks=[[6, 7, 8], [3, 4, 5], [0, 1, 2]],
                 norm_type="bn",
                 norm_decay=0.,
                 coord_conv=True,
                 iou_aware=True,
                 iou_aware_factor=0.4,
                 block_size=3,
                 scale_x_y=1.05,
                 spp=True,
                 drop_block=True,
                 keep_prob=0.9,
                 clip_bbox=True,
                 yolo_loss=None,
                 downsample=[32, 16, 8],
                 in_channels=[2048, 1024, 512],
                 nms_cfg=None,
                 focalloss_on_obj=False,
                 prior_prob=0.01,
                 is_train=False
                 ):
        super(YOLOv3Head, self).__init__()
        self.conv_block_num = conv_block_num
        self.num_classes = num_classes
        self.norm_type = norm_type
        self.norm_decay = norm_decay
        self.coord_conv = coord_conv
        self.iou_aware = iou_aware
        self.iou_aware_factor = iou_aware_factor
        self.scale_x_y = scale_x_y
        self.use_spp = spp
        self.drop_block = drop_block
        self.keep_prob = keep_prob
        self.clip_bbox = clip_bbox
        self.anchors = anchors
        self.anchor_masks = anchor_masks
        self.block_size = block_size
        self.downsample = downsample
        self.in_channels = in_channels
        self.yolo_loss = yolo_loss
        self.nms_cfg = nms_cfg
        self.focalloss_on_obj = focalloss_on_obj
        self.prior_prob = prior_prob
        self.is_train = is_train

        _anchors = copy.deepcopy(anchors)
        _anchors = np.array(_anchors)
        _anchors = _anchors.astype(np.float32)
        self._anchors = _anchors   # [9, 2]

        self.mask_anchors = []
        for m in anchor_masks:
            temp = []
            for aid in m:
                temp += anchors[aid]
            self.mask_anchors.append(temp)

        self.detection_blocks = paddle.nn.LayerList()
        self.yolo_output_convs = paddle.nn.LayerList()
        self.upsample_layers = paddle.nn.LayerList()
        out_layer_num = len(downsample)
        for i in range(out_layer_num):
            in_c = self.in_channels[i]
            if i > 0:  # perform concat in first 2 detection_block
                in_c = self.in_channels[i] + 512 // (2**i)
            _detection_block = DetectionBlock(
                in_c=in_c,
                channel=64 * (2**out_layer_num) // (2**i),
                coord_conv=self.coord_conv,
                norm_type=norm_type,
                norm_decay=self.norm_decay,
                is_first=i == 0,
                conv_block_num=self.conv_block_num,
                use_spp=self.use_spp,
                drop_block=self.drop_block,
                block_size=self.block_size,
                keep_prob=self.keep_prob,
                is_test=(not self.is_train),
                name="yolo_block.{}".format(i)
            )
            # out channel number = mask_num * (5 + class_num)
            if self.iou_aware:
                num_filters = len(self.anchor_masks[i]) * (self.num_classes + 6)
            else:
                num_filters = len(self.anchor_masks[i]) * (self.num_classes + 5)
            bias_init = None
            if self.focalloss_on_obj:
                # 设置偏移的初始值使得obj预测概率初始值为self.prior_prob (根据激活函数是sigmoid()时推导出)
                bias_init_value = -math.log((1 - self.prior_prob) / self.prior_prob)
                bias_init_array = np.zeros((num_filters, ), np.float32)
                an_num = len(self.anchor_masks[i])
                start = 0
                stride = (self.num_classes + 5)
                if self.iou_aware:
                    start = an_num
                # 只设置置信位
                for o in range(an_num):
                    bias_init_array[start + o * stride + 4] = bias_init_value
                bias_init = fluid.initializer.NumpyArrayInitializer(bias_init_array)
            yolo_output_conv = Conv2dUnit(64 * (2**out_layer_num) // (2**i) * 2, num_filters, 1, stride=1, bias_attr=True, act=None,
                                          bias_init=bias_init, name="yolo_output.{}.conv".format(i))
            self.detection_blocks.append(_detection_block)
            self.yolo_output_convs.append(yolo_output_conv)


            if i < out_layer_num - 1:
                # do not perform upsample in the last detection_block
                conv_unit = Conv2dUnit(64 * (2**out_layer_num) // (2**i), 256 // (2**i), 1, stride=1, norm_type=norm_type, act='leaky', norm_decay=self.norm_decay, name="yolo_transition.{}".format(i))
                # upsample
                upsample = paddle.nn.Upsample(scale_factor=2, mode='nearest')
                self.upsample_layers.append(conv_unit)
                self.upsample_layers.append(upsample)

    def set_dropblock(self, is_test):
        for detection_block in self.detection_blocks:
            for l in detection_block.layers:
                if isinstance(l, DropBlock):
                    l.is_test = is_test

    def _get_outputs(self, body_feats):
        outputs = []

        # get last out_layer_num blocks in reverse order
        out_layer_num = len(self.anchor_masks)
        blocks = body_feats[-1:-out_layer_num - 1:-1]

        route = None
        for i, block in enumerate(blocks):
            if i > 0:  # perform concat in first 2 detection_block
                block = paddle.concat([route, block], axis=1)
            route, tip = self.detection_blocks[i](block)
            block_out = self.yolo_output_convs[i](tip)
            outputs.append(block_out)
            if i < out_layer_num - 1:
                route = self.upsample_layers[i*2](route)
                route = self.upsample_layers[i*2+1](route)
        return outputs

    def get_loss(self, input, gt_box, gt_label, gt_score, targets):
        """
        Get final loss of network of YOLOv3.

        Args:
            input (list): List of Variables, output of backbone stages
            gt_box (Variable): The ground-truth boudding boxes.
            gt_label (Variable): The ground-truth class labels.
            gt_score (Variable): The ground-truth boudding boxes mixup scores.
            targets ([Variables]): List of Variables, the targets for yolo
                                   loss calculatation.

        Returns:
            loss (Variable): The loss Variable of YOLOv3 network.

        """

        # outputs里为大中小感受野的输出
        outputs = self._get_outputs(input)

        return self.yolo_loss(outputs, gt_box, gt_label, gt_score, targets,
                              self.anchors, self.anchor_masks,
                              self.mask_anchors, self.num_classes)

    def get_prediction(self, body_feats, im_size):
        """
        Get prediction result of YOLOv3 network

        Args:
            input (list): List of Variables, output of backbone stages
            im_size (Variable): Variable of size([h, w]) of each image

        Returns:
            pred (Variable): shape = [bs, keep_top_k, 6]

        """
        # outputs里为大中小感受野的输出
        outputs = self._get_outputs(body_feats)

        boxes = []
        scores = []
        for i, output in enumerate(outputs):
            if self.iou_aware:
                output = get_iou_aware_score(output,
                                             len(self.anchor_masks[i]),
                                             self.num_classes,
                                             self.iou_aware_factor)
            box, score = fluid.layers.yolo_box(
                x=output,
                img_size=im_size,
                anchors=self.mask_anchors[i],
                class_num=self.num_classes,
                conf_thresh=self.nms_cfg['score_threshold'],
                downsample_ratio=self.downsample[i],
                name="yolo_box" + str(i),
                clip_bbox=self.clip_bbox,
                scale_x_y=self.scale_x_y)
            boxes.append(box)
            scores.append(paddle.transpose(score, perm=[0, 2, 1]))
        yolo_boxes = paddle.concat(boxes, axis=1)
        yolo_scores = paddle.concat(scores, axis=2)


        # nms
        preds = []
        nms_cfg = copy.deepcopy(self.nms_cfg)
        nms_type = nms_cfg.pop('nms_type')
        batch_size = yolo_boxes.shape[0]
        if nms_type == 'matrix_nms':
            for i in range(batch_size):
                pred = fluid.layers.matrix_nms(yolo_boxes[i:i+1, :, :], yolo_scores[i:i+1, :, :], background_label=-1, **nms_cfg)
                # pred = matrix_nms(yolo_boxes[i, :, :], yolo_scores[i, :, :], **nms_cfg)
                preds.append(pred)
        elif nms_type == 'multiclass_nms':
            for i in range(batch_size):
                pred = fluid.layers.multiclass_nms(yolo_boxes[i:i+1, :, :], yolo_scores[i:i+1, :, :], background_label=-1, **nms_cfg)
                preds.append(pred)
        return preds





