#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import math
from loguru import logger

import copy
import numpy as np
import paddle
from paddle import nn
import paddle.nn.functional as F

# from .losses import IOUloss
# from .network_blocks import BaseConv, DWConv
from model.backbones.cspdarknet import *


def bboxes_iou(bboxes_a, bboxes_b, xyxy=True):
    """计算两组矩形两两之间的iou
    Args:
        bboxes_a: (tensor) bounding boxes, Shape: [A, 4].
        bboxes_b: (tensor) bounding boxes, Shape: [B, 4].
    Return:
      (tensor) iou, Shape: [A, B].
    """
    A = bboxes_a.shape[0]
    B = bboxes_b.shape[0]
    if xyxy:
        box_a = bboxes_a
        box_b = bboxes_b
    else:  # cxcywh格式
        box_a = paddle.concat([bboxes_a[:, :2] - bboxes_a[:, 2:] * 0.5,
                               bboxes_a[:, :2] + bboxes_a[:, 2:] * 0.5], axis=-1)
        box_b = paddle.concat([bboxes_b[:, :2] - bboxes_b[:, 2:] * 0.5,
                               bboxes_b[:, :2] + bboxes_b[:, 2:] * 0.5], axis=-1)

    box_a_rb = paddle.reshape(box_a[:, 2:], (A, 1, 2))
    box_a_rb = paddle.tile(box_a_rb, [1, B, 1])
    box_b_rb = paddle.reshape(box_b[:, 2:], (1, B, 2))
    box_b_rb = paddle.tile(box_b_rb, [A, 1, 1])
    max_xy = paddle.minimum(box_a_rb, box_b_rb)

    box_a_lu = paddle.reshape(box_a[:, :2], (A, 1, 2))
    box_a_lu = paddle.tile(box_a_lu, [1, B, 1])
    box_b_lu = paddle.reshape(box_b[:, :2], (1, B, 2))
    box_b_lu = paddle.tile(box_b_lu, [A, 1, 1])
    min_xy = paddle.maximum(box_a_lu, box_b_lu)

    inter = F.relu(max_xy - min_xy)
    inter = inter[:, :, 0] * inter[:, :, 1]

    box_a_w = box_a[:, 2]-box_a[:, 0]
    box_a_h = box_a[:, 3]-box_a[:, 1]
    area_a = box_a_h * box_a_w
    area_a = paddle.reshape(area_a, (A, 1))
    area_a = paddle.tile(area_a, [1, B])  # [A, B]

    box_b_w = box_b[:, 2]-box_b[:, 0]
    box_b_h = box_b[:, 3]-box_b[:, 1]
    area_b = box_b_h * box_b_w
    area_b = paddle.reshape(area_b, (1, B))
    area_b = paddle.tile(area_b, [A, 1])  # [A, B]

    union = area_a + area_b - inter + 1e-9
    return inter / union  # [A, B]

def bboxes_iou_batch(bboxes_a, bboxes_b, xyxy=True):
    """计算两组矩形两两之间的iou
    Args:
        bboxes_a: (tensor) bounding boxes, Shape: [N, A, 4].
        bboxes_b: (tensor) bounding boxes, Shape: [N, B, 4].
    Return:
      (tensor) iou, Shape: [N, A, B].
    """
    N = bboxes_a.shape[0]
    A = bboxes_a.shape[1]
    B = bboxes_b.shape[1]
    if xyxy:
        box_a = bboxes_a
        box_b = bboxes_b
    else:  # cxcywh格式
        box_a = paddle.concat([bboxes_a[:, :, :2] - bboxes_a[:, :, 2:] * 0.5,
                               bboxes_a[:, :, :2] + bboxes_a[:, :, 2:] * 0.5], axis=-1)
        box_b = paddle.concat([bboxes_b[:, :, :2] - bboxes_b[:, :, 2:] * 0.5,
                               bboxes_b[:, :, :2] + bboxes_b[:, :, 2:] * 0.5], axis=-1)

    box_a_rb = paddle.reshape(box_a[:, :, 2:], (N, A, 1, 2))
    box_a_rb = paddle.tile(box_a_rb, [1, 1, B, 1])
    box_b_rb = paddle.reshape(box_b[:, :, 2:], (N, 1, B, 2))
    box_b_rb = paddle.tile(box_b_rb, [1, A, 1, 1])
    max_xy = paddle.minimum(box_a_rb, box_b_rb)

    box_a_lu = paddle.reshape(box_a[:, :, :2], (N, A, 1, 2))
    box_a_lu = paddle.tile(box_a_lu, [1, 1, B, 1])
    box_b_lu = paddle.reshape(box_b[:, :, :2], (N, 1, B, 2))
    box_b_lu = paddle.tile(box_b_lu, [1, A, 1, 1])
    min_xy = paddle.maximum(box_a_lu, box_b_lu)

    inter = F.relu(max_xy - min_xy)
    inter = inter[:, :, :, 0] * inter[:, :, :, 1]

    box_a_w = box_a[:, :, 2]-box_a[:, :, 0]
    box_a_h = box_a[:, :, 3]-box_a[:, :, 1]
    area_a = box_a_h * box_a_w
    area_a = paddle.reshape(area_a, (N, A, 1))
    area_a = paddle.tile(area_a, [1, 1, B])  # [N, A, B]

    box_b_w = box_b[:, :, 2]-box_b[:, :, 0]
    box_b_h = box_b[:, :, 3]-box_b[:, :, 1]
    area_b = box_b_h * box_b_w
    area_b = paddle.reshape(area_b, (N, 1, B))
    area_b = paddle.tile(area_b, [1, A, 1])  # [N, A, B]

    union = area_a + area_b - inter + 1e-9
    return inter / union  # [N, A, B]





class YOLOXHead(nn.Layer):
    def __init__(
        self,
        num_classes,
        width=1.0,
        strides=[8, 16, 32],
        in_channels=[256, 512, 1024],
        act="silu",
        depthwise=False,
        iou_loss=None,
        yolo_loss=None,
        nms_cfg=None,
        prior_prob=0.01,
        is_train=False,
    ):
        """
        Args:
            act (str): activation type of conv. Defalut value: "silu".
            depthwise (bool): whether apply depthwise conv in conv branch. Defalut value: False.
        """
        super().__init__()

        self.n_anchors = 1
        # self.is_train = is_train
        self.num_classes = num_classes
        self.decode_in_inference = True  # for deploy, set to False

        self.cls_convs = nn.LayerList()
        self.reg_convs = nn.LayerList()
        self.cls_preds = nn.LayerList()
        self.reg_preds = nn.LayerList()
        self.obj_preds = nn.LayerList()
        self.stems = nn.LayerList()
        Conv = DWConv if depthwise else BaseConv

        self.prior_prob = prior_prob
        # 类别分支最后的卷积。设置偏移的初始值使得各类别预测概率初始值为self.prior_prob (根据激活函数是sigmoid()时推导出，和RetinaNet中一样)
        bias_init_value = -math.log((1 - self.prior_prob) / self.prior_prob)

        for i in range(len(in_channels)):
            self.stems.append(
                BaseConv(
                    in_channels=int(in_channels[i] * width),
                    out_channels=int(256 * width),
                    ksize=1,
                    stride=1,
                    act=act,
                )
            )
            self.cls_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.reg_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            battr1 = ParamAttr(initializer=Constant(bias_init_value),
                               regularizer=L2Decay(0.))   # 不可以加正则化的参数：norm层(比如bn层、affine_channel层、gn层)的scale、offset；卷积层的偏移参数。
            cls_pred_conv = nn.Conv2D(in_channels=int(256 * width),
                                      out_channels=self.n_anchors * self.num_classes,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0,
                                      bias_attr=battr1)
            self.cls_preds.append(cls_pred_conv)
            battr2 = ParamAttr(initializer=Constant(bias_init_value),
                               regularizer=L2Decay(0.))   # 不可以加正则化的参数：norm层(比如bn层、affine_channel层、gn层)的scale、offset；卷积层的偏移参数。
            reg_pred_conv = nn.Conv2D(in_channels=int(256 * width),
                                      out_channels=4,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0,
                                      bias_attr=battr2)
            self.reg_preds.append(reg_pred_conv)
            self.obj_preds.append(
                nn.Conv2D(
                    in_channels=int(256 * width),
                    out_channels=self.n_anchors * 1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )

        self.use_l1 = False
        self.l1_loss = nn.L1Loss(reduction="none")
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss = iou_loss
        self.strides = strides
        self.grids = [paddle.zeros((1, ))] * len(in_channels)

        self.yolo_loss = yolo_loss
        self.nms_cfg = nms_cfg

    def get_outputs(self, xin):
        outputs = []
        origin_preds = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []

        for k, (cls_conv, reg_conv, stride_this_level, x) in enumerate(
            zip(self.cls_convs, self.reg_convs, self.strides, xin)
        ):
            x = self.stems[k](x)
            cls_x = x
            reg_x = x

            cls_feat = cls_conv(cls_x)
            cls_output = self.cls_preds[k](cls_feat)

            reg_feat = reg_conv(reg_x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)

            if self.training:
                output = paddle.concat([reg_output, obj_output, cls_output], 1)
                # output.shape=[N, 1 * 80 * 80, 85]  output里面有解码后的xywh
                # grid.shape=  [1, 1 * 80 * 80, 2]   格子左上角xy坐标，单位是这个特征图的stride
                output, grid = self.get_output_and_grid(
                    output, k, stride_this_level
                )
                x_shifts.append(grid[:, :, 0])   # [1, 1 * 80 * 80]  格子左上角x坐标，单位是这个特征图的stride
                y_shifts.append(grid[:, :, 1])   # [1, 1 * 80 * 80]  格子左上角y坐标，单位是这个特征图的stride
                # [1, 1 * 80 * 80]  这个特征图的stride复制hsize*wsize份，意思是每个格子持有一个stride
                expanded_stride = paddle.ones((1, grid.shape[1]), dtype=xin[0].dtype) * stride_this_level
                expanded_strides.append(expanded_stride)
                if self.use_l1:  # 如果使用L1损失。不使用马赛克增强后会使用L1损失。
                    batch_size = reg_output.shape[0]      # 批大小
                    hsize, wsize = reg_output.shape[-2:]  # 格子行数、列数
                    # L1损失监督的是未解码的xywh
                    reg_output = reg_output.reshape(
                        (batch_size, self.n_anchors, 4, hsize, wsize)
                    )
                    reg_output = reg_output.transpose((0, 1, 3, 4, 2)).reshape(
                        (batch_size, -1, 4)
                    )
                    origin_preds.append(reg_output.clone())

            else:
                output = paddle.concat(
                    [reg_output, F.sigmoid(obj_output), F.sigmoid(cls_output)], 1
                )

            outputs.append(output)

        return outputs, x_shifts, y_shifts, expanded_strides, origin_preds

    def get_prediction(self, xin, im_info):
        outputs, x_shifts, y_shifts, expanded_strides, origin_preds = self.get_outputs(xin)

        # 设N=批大小
        # 设A=n_anchors_all=每张图片输出的预测框数，当输入图片分辨率是640*640时，A=8400

        self.hw = [x.shape[-2:] for x in outputs]
        # [batch, 85, A]
        outputs = paddle.concat(
            [paddle.reshape(x, (x.shape[0], x.shape[1], -1)) for x in outputs], axis=2
        )
        # [batch, A, 85]
        outputs = paddle.transpose(outputs, [0, 2, 1])

        if self.decode_in_inference:
            outputs = self.decode_outputs(outputs)

        # 自加的后处理
        yolo_boxes = outputs[:, :, :4]  # [N, A, 4]  cxcywh格式

        # [N, A, 4]  左上角坐标+右下角坐标格式，即x0y0x1y1格式
        yolo_boxes = paddle.concat([yolo_boxes[:, :, :2] - yolo_boxes[:, :, 2:] * 0.5,
                                    yolo_boxes[:, :, :2] + yolo_boxes[:, :, 2:] * 0.5], axis=-1)

        # 上面的坐标是对输入的640x640分辨率图片的预测，需要把坐标缩放成原图中的坐标
        im_scale = im_info[:, 2:3]  # [N, 1]
        im_scale = im_scale.unsqueeze(2)  # [N, 1, 1]
        yolo_boxes /= im_scale

        yolo_scores = outputs[:, :, 4:5] * outputs[:, :, 5:]
        yolo_scores = paddle.transpose(yolo_scores, [0, 2, 1])  # [N, 80, A]

        # nms
        preds = []
        nms_cfg = copy.deepcopy(self.nms_cfg)
        nms_type = nms_cfg.pop('nms_type')
        batch_size = yolo_boxes.shape[0]
        if nms_type == 'matrix_nms':
            for i in range(batch_size):
                pred = fluid.layers.matrix_nms(yolo_boxes[i:i + 1, :, :], yolo_scores[i:i + 1, :, :],
                                               background_label=-1, **nms_cfg)
                # pred = matrix_nms(yolo_boxes[i, :, :], yolo_scores[i, :, :], **nms_cfg)
                preds.append(pred)
        elif nms_type == 'multiclass_nms':
            for i in range(batch_size):
                pred = fluid.layers.multiclass_nms(yolo_boxes[i:i + 1, :, :], yolo_scores[i:i + 1, :, :],
                                                   background_label=-1, **nms_cfg)
                preds.append(pred)
        return preds

    def get_loss(self, xin, labels=None):
        outputs, x_shifts, y_shifts, expanded_strides, origin_preds = self.get_outputs(xin)
        # aaaaaaaaaa = labels.numpy()
        outputs = paddle.concat(outputs, 1)
        # dic = {}
        # dic['outputs'] = outputs.numpy()
        # dic['labels'] = labels.numpy()

        # dic2 = np.load('data2.npz')
        # outputs2 = dic2['outputs'].astype(np.float32)
        # labels2 = dic2['labels'].astype(np.float32)
        # loss_iou2 = dic2['loss_iou']
        # loss_obj2 = dic2['loss_obj']
        # loss_cls2 = dic2['loss_cls']
        # labels2 = paddle.to_tensor(labels2)
        # outputs2 = paddle.to_tensor(outputs2)
        # labels2 = paddle.cast(labels2, 'float32')
        # outputs2 = paddle.cast(outputs2, 'float32')

        losses = self.get_losses(
            x_shifts,
            y_shifts,
            expanded_strides,
            labels,
            outputs,
            origin_preds,
            dtype=xin[0].dtype,
        )
        # dic['loss_iou'] = losses['loss_iou'].numpy()
        # dic['loss_obj'] = losses['loss_obj'].numpy()
        # dic['loss_cls'] = losses['loss_cls'].numpy()
        # np.savez('data2', **dic)
        # print()

        # losses = self.get_losses(
        #     x_shifts,
        #     y_shifts,
        #     expanded_strides,
        #     labels2,
        #     outputs2,
        #     origin_preds,
        #     dtype=xin[0].dtype,
        # )
        return losses

    def set_dropblock(self, is_test):
        # for detection_block in self.detection_blocks:
        #     for l in detection_block.layers:
        #         if isinstance(l, DropBlock):
        #             l.is_test = is_test
        pass

    def get_output_and_grid(self, output, k, stride):
        grid = self.grids[k]

        batch_size = output.shape[0]
        n_ch = 5 + self.num_classes
        hsize, wsize = output.shape[-2:]
        if grid.shape[2:4] != output.shape[2:4]:
            yv, xv = paddle.meshgrid([paddle.arange(hsize), paddle.arange(wsize)])
            grid = paddle.stack((xv, yv), 2)
            grid = paddle.reshape(grid, (1, 1, hsize, wsize, 2))
            grid = paddle.cast(grid, dtype=output.dtype)
            self.grids[k] = grid

        output = paddle.reshape(output, (batch_size, self.n_anchors, n_ch, hsize, wsize))
        output = paddle.transpose(output, [0, 1, 3, 4, 2])
        output = paddle.reshape(output, (batch_size, self.n_anchors * hsize * wsize, -1))   # [N, 1 * 80 * 80, 85]
        grid = paddle.reshape(grid, (1, -1, 2))   # [1, 1 * 80 * 80, 2]

        xy = (output[:, :, :2] + grid) * stride       # [N, 1 * 80 * 80, 2]  xy解码
        wh = paddle.exp(output[:, :, 2:4]) * stride   # [N, 1 * 80 * 80, 2]  wh解码
        output = paddle.concat([xy, wh, output[:, :, 4:]], 2)   # [N, 1 * 80 * 80, 85]   解码后的xywh放回output里面
        return output, grid

    def decode_outputs(self, outputs):
        grids = []
        strides = []
        for (hsize, wsize), stride in zip(self.hw, self.strides):
            yv, xv = paddle.meshgrid([paddle.arange(hsize), paddle.arange(wsize)])
            grid = paddle.reshape(paddle.stack((xv, yv), 2), (1, -1, 2))
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(paddle.full((*shape, 1), stride))

        grids = paddle.concat(grids, axis=1)
        strides = paddle.concat(strides, axis=1)
        grids = paddle.cast(grids, outputs.dtype)
        strides = paddle.cast(strides, outputs.dtype)

        outputs[:, :, :2] = (outputs[:, :, :2] + grids) * strides
        outputs[:, :, 2:4] = paddle.exp(outputs[:, :, 2:4]) * strides
        return outputs

    def get_losses(
        self,
        x_shifts,
        y_shifts,
        expanded_strides,
        labels,
        outputs,
        origin_preds,
        dtype,
    ):
        # 设N=批大小
        # 设A=n_anchors_all=每张图片输出的预测框数，当输入图片分辨率是640*640时，A=8400
        N = outputs.shape[0]
        A = outputs.shape[1]
        # labels_numpy = labels.numpy()

        # 1.把网络输出切分成预测框、置信度、类别概率
        bbox_preds = outputs[:, :, :4]              # [N, A, 4]   这是解码后的xywh。
        obj_preds = outputs[:, :, 4].unsqueeze(-1)  # [N, A, 1]   置信度。没有经过sigmoid()激活。
        cls_preds = outputs[:, :, 5:]               # [N, A, n_cls]   n_cls个类别的条件概率。没有经过sigmoid()激活。

        # 2.计算gt数目
        labels = paddle.cast(labels, 'float32')  # [N, 120, 5]
        if_gt = labels.sum([2])    # [N, 120]
        if_gt = paddle.cast(if_gt > 0, 'float32')  # [N, 120] 是gt处为1
        nlabel = if_gt.sum([1])    # [N, ] 每张图片gt数量
        nlabel = paddle.cast(nlabel, 'int32')
        nlabel.stop_gradient = True
        G = nlabel.max()  # 每张图片最多的gt数

        if G == 0:  # 所有图片都没有gt时
            obj_targets = paddle.zeros((N, A, 1), 'float32')
            num_fg = 1  # 所有图片都没有gt时，设为1
            loss_obj = self.bcewithlog_loss(obj_preds, obj_targets)
            loss_obj = loss_obj.sum() / num_fg
            losses = {
                "loss_obj": loss_obj,
            }
            return losses

        labels = labels[:, :G, :]  # [N, G, 5] 从最多处截取
        # labels_numpy = labels.numpy()  # [N, G, 5] 从最多处截取

        is_gt = if_gt[:, :G]  # [N, G] 是gt处为1。从最多处截取

        # 3.拼接用到的常量张量x_shifts、y_shifts、expanded_strides
        A = outputs.shape[1]  # 一张图片出8400个anchor
        x_shifts = paddle.concat(x_shifts, 1)  # [1, A]  每个格子左上角的x坐标。单位是下采样步长。比如，第0个特征图的1代表的是8个像素，第2个特征图的1代表的是32个像素。
        y_shifts = paddle.concat(y_shifts, 1)  # [1, A]  每个格子左上角的y坐标。单位是下采样步长。
        expanded_strides = paddle.concat(expanded_strides, 1)  # [1, A]  每个anchor对应的下采样倍率。依次是8, 16, 32
        if self.use_l1:
            origin_preds = paddle.concat(origin_preds, 1)  # [N, A, 4]

        # 4.对于每张图片，决定哪些样本作为正样本

        # 4-1.将每张图片的gt的坐标宽高、类别id提取出来。
        gt_bboxes = labels[:, :, 1:5]  # [N, G, 4]
        gt_classes = labels[:, :, 0]   # [N, G]

        # 4-2.get_assignments()确定正负样本，里面的张量不需要梯度。
        (
            gt_matched_classes,
            fg_mask,
            pred_ious_this_matching,
            matched_gt_inds,
            num_fg,
        ) = self.get_assignments(  # noqa
            N,
            A,
            G,
            gt_bboxes,
            gt_classes,
            bbox_preds,
            expanded_strides,
            x_shifts,
            y_shifts,
            cls_preds,
            obj_preds,
            is_gt,
        )
        # self.get_assignments()返回的结果：
        # num_fg。                  [1, ]       所有图片前景（最终正样本）个数
        # gt_matched_classes。      [num_fg, ]  最终正样本需要学习的类别id
        # pred_ious_this_matching。 [num_fg, ]  最终正样本和所学gt的iou
        # matched_gt_inds。         [num_fg, ]  最终正样本是匹配到了第几个gt（在gt_classes.shape=[N*G, ]中的坐标）
        # fg_mask。                 [N, A]      最终正样本处为1

        # 5.准备监督信息
        # 第一步，准备 各类别概率 需要的监督信息。
        eye = paddle.eye(self.num_classes, dtype='float32')  # [80, 80]  对角线位置全是1，其余位置全为0的矩阵。
        one_hot = paddle.gather(eye, gt_matched_classes)  # [num_fg, 80]  num_fg个最终正样本需要学习的类别one_hot向量。
        cls_targets = one_hot * pred_ious_this_matching.unsqueeze(-1)  # [num_fg, 80]  num_fg个最终正样本需要学习的类别one_hot向量需乘以其与所学gt的iou。

        # 第二步，准备 置信度obj-ness 需要的监督信息。
        obj_targets = fg_mask.reshape((N*A, 1))  # [N*A, 1]   每个anchor obj-ness处需要学习的目标。
        # 第三步，准备 回归分支 需要的监督信息。
        reg_targets = paddle.gather(gt_bboxes.reshape((N*G, 4)), matched_gt_inds)  # [num_fg, 4]  num_fg个最终正样本需要学习的预测框xywh。
        # 第四步，如果使用L1损失，准备 L1损失 需要的监督信息。
        l1_targets = []
        if self.use_l1:
            pos_index_ = paddle.nonzero(fg_mask > 0)   # [num_fg, 2]   最终正样本在fg_mask.shape=[N, A]中的坐标。
            pos_index2 = pos_index_[:, 1]              # [num_fg, ]    最终正样本在fg_mask.shape=[N, A]中第1维（A那维）的坐标。
            # L1损失监督的是最终正样本未解码的xywh，所以把reg_targets编码成未解码的状态，得到l1_targets。
            l1_targets = self.get_l1_target(
                paddle.zeros((num_fg, 4), 'float32'),
                reg_targets,
                paddle.gather(expanded_strides[0], pos_index2),
                x_shifts=paddle.gather(x_shifts[0], pos_index2),
                y_shifts=paddle.gather(y_shifts[0], pos_index2),
            )

        # 监督信息停止梯度
        fg_masks = fg_mask.reshape((N*A, ))   # [N*A, ]
        cls_targets.stop_gradient = True  # [num_fg, 80]
        reg_targets.stop_gradient = True  # [num_fg, 4]
        obj_targets.stop_gradient = True  # [N*A, 1]
        fg_masks.stop_gradient = True     # [N*A, ]


        # 6.计算损失
        num_fg = max(num_fg, 1)
        bbox_preds = paddle.reshape(bbox_preds, [-1, 4])  # [N*A, 4]
        pos_index = paddle.nonzero(fg_masks > 0)[:, 0]    # [num_fg, ]
        pos_bbox_preds = paddle.gather(bbox_preds, pos_index)  # [num_fg, 4]   每个最终正样本预测的xywh
        loss_iou = (
            self.iou_loss(pos_bbox_preds, reg_targets)
        ).sum() / num_fg
        loss_obj = (
            self.bcewithlog_loss(paddle.reshape(obj_preds, [-1, 1]), obj_targets)
        ).sum() / num_fg

        cls_preds = paddle.reshape(cls_preds, [-1, self.num_classes])  # [N*A, 80]
        pos_cls_preds = paddle.gather(cls_preds, pos_index)            # [num_fg, 80]
        loss_cls = (
            self.bcewithlog_loss(pos_cls_preds, cls_targets)
        ).sum() / num_fg
        if self.use_l1:
            origin_preds = paddle.reshape(origin_preds, [-1, 4])       # [N*A, 4]
            pos_origin_preds = paddle.gather(origin_preds, pos_index)  # [num_fg, 4]
            loss_l1 = (
                self.l1_loss(pos_origin_preds, l1_targets)
            ).sum() / num_fg

        reg_weight = 5.0
        losses = {
            "loss_iou": reg_weight * loss_iou,
            "loss_obj": loss_obj,
            "loss_cls": loss_cls,
        }
        if self.use_l1:
            losses["loss_l1"] = loss_l1
        return losses

    def get_l1_target(self, l1_target, gt, stride, x_shifts, y_shifts, eps=1e-8):
        l1_target[:, 0] = gt[:, 0] / stride - x_shifts
        l1_target[:, 1] = gt[:, 1] / stride - y_shifts
        l1_target[:, 2] = paddle.log(gt[:, 2] / stride + eps)
        l1_target[:, 3] = paddle.log(gt[:, 3] / stride + eps)
        l1_target.stop_gradient = True
        return l1_target

    @paddle.no_grad()
    def get_assignments(
        self,
        N,
        A,
        G,
        gt_bboxes,
        gt_classes,
        bbox_preds,
        expanded_strides,
        x_shifts,
        y_shifts,
        cls_preds,
        obj_preds,
        is_gt,
    ):
        # 4-2.get_assignments()确定正负样本，里面的张量不需要梯度。
        # 4-2-1.确定 候选正样本。

        # is_in_boxes_or_center。  [N, A] 每个格子是否是在 任意gt内部 或 任意gt的镜像gt内部（不要求同一个gt）
        # 值为1处的格子可以叫做“候选正样本”
        # is_in_boxes_and_center。 [N, G, A] 每个格子是否是在 某个gt内部 且 这个gt的镜像gt内部（要求同一个gt）
        # 每个格子持有G个值，G个值中若至少有1个值为1，不难证明，这个格子其实也是“候选正样本”中的某个。
        # is_in_boxes_and_center的作用是 用来帮助确定 某些高质量的候选正样本 成为最终正样本。
        # 因为若某个格子既在gt内又在这个gt的镜像gt内时，它就更应该负责去学习这个gt。
        is_in_boxes_or_center, is_in_boxes_and_center = self.get_in_boxes_info(
            gt_bboxes,
            expanded_strides,
            x_shifts,
            y_shifts,
            A,
            G,
        )

        '''
        gt_bboxes     [N, G, 4]
        bbox_preds    [N, A, 4]
        '''
        # 4-2-2.计算每张图片 所有gt 和 所有预测框 两两之间的iou 的cost，iou越大cost越小，越有可能成为最终正样本。
        pair_wise_ious = bboxes_iou_batch(gt_bboxes, bbox_preds, False)  # [N, G, A]  两两之间的iou。
        # 假gt 和 任意预测框 的iou置为0
        pair_wise_ious *= is_gt.unsqueeze(2)
        # 非候选正样本 和 任意gt 的iou置为0。因为只有候选正样本才有资格成为最终的正样本。
        pair_wise_ious *= is_in_boxes_or_center.unsqueeze(1)
        pair_wise_ious_loss = -paddle.log(pair_wise_ious + 1e-8)  # [N, G, A]  iou取对数再取相反数。
        # 假gt 和 任意预测框 的ious_cost放大
        pair_wise_ious_loss += (1.0 - is_gt.unsqueeze(2)) * 100000.0
        # 非候选正样本 和 任意gt 的ious_cost放大
        pair_wise_ious_loss += (1.0 - is_in_boxes_or_center.unsqueeze(1)) * 100000.0

        # 4-2-3.计算每张图片 所有gt 和 所有预测框 两两之间的cls 的cost，cost越小，越有可能成为最终正样本。
        p1 = cls_preds.unsqueeze(1)  # [N, 1, A, 80]
        p2 = obj_preds.unsqueeze(1)  # [N, 1, A, 1]
        p = F.sigmoid(p1) * F.sigmoid(p2)  # [N, 1, A, 80]  各类别分数
        p = paddle.tile(p, [1, G, 1, 1])   # [N, G, A, 80]  各类别分数
        p = paddle.sqrt(p)                 # [N, G, A, 80]  各类别分数开根号求平均
        # 获得N*G个gt的one_hot类别向量，每个候选正样本持有一个。
        gt_classes = paddle.reshape(gt_classes, (N*G, ))  # [N*G, ]
        gt_classes = paddle.cast(gt_classes, 'int32')     # [N*G, ]
        one_hots = F.one_hot(gt_classes, num_classes=self.num_classes)  # [N*G, 80]
        one_hots = paddle.reshape(one_hots, (N, G, 1, self.num_classes))  # [N, G, 1, 80]
        one_hots = paddle.tile(one_hots, [1, 1, A, 1])  # [N, G, A, 80]
        gt_clss = one_hots
        # 二值交叉熵
        # pos_loss = gt_clss * (0 - paddle.log(p + 1e-9))              # [N, G, A, 80]
        # neg_loss = (1.0 - gt_clss) * (0 - paddle.log(1 - p + 1e-9))  # [N, G, A, 80]
        # pair_wise_cls_loss = pos_loss + neg_loss                     # [N, G, A, 80]
        # del pos_loss, neg_loss, p, gt_clss, one_hots
        # 二值交叉熵
        pair_wise_cls_loss = F.binary_cross_entropy(p, gt_clss, reduction='none')       # [N, G, A, 80]
        del p, gt_clss, one_hots

        pair_wise_cls_loss = pair_wise_cls_loss.sum(-1)    # [N, G, A]  cost越小，越有可能成为最终正样本。
        # 假gt 和 任意预测框 的cls_cost放大
        pair_wise_cls_loss += (1.0 - is_gt.unsqueeze(2)) * 100000.0
        # 非候选正样本 和 任意gt 的cls_cost放大
        pair_wise_cls_loss += (1.0 - is_in_boxes_or_center.unsqueeze(1)) * 100000.0

        # 4-2-4.计算每张图片 所有gt 和 所有预测框 两两之间的 总的cost，cost越小，越有可能成为最终正样本。
        # is_in_boxes_and_center的作用是 用来帮助确定 某些高质量的候选正样本 成为最终正样本。
        # 因为若某个格子既在gt内又在这个gt的镜像gt内时，它就更应该负责去学习这个gt。
        # is_in_boxes_and_center是1，cost越小，对应格子越有可能成为最终正样本，学习的是为1处的那个gt。
        # is_in_boxes_and_center是0，cost越大，对应格子越不可能成为最终正样本。
        cost = (
            pair_wise_cls_loss
            + 3.0 * pair_wise_ious_loss
            + 100000.0 * (1.0 - is_in_boxes_and_center)
        )  # [N, G, A]

        # 4-2-5.根据cost从 候选正样本 中 确定 最终正样本。
        (
            num_fg,
            gt_matched_classes,
            pred_ious_this_matching,
            matched_gt_inds,
            fg_mask,
        ) = self.dynamic_k_matching(cost, pair_wise_ious, gt_classes, N, G, A, is_in_boxes_or_center, is_gt)
        del cost, pair_wise_cls_loss, pair_wise_ious_loss, is_in_boxes_and_center
        return (
            gt_matched_classes,
            fg_mask,
            pred_ious_this_matching,
            matched_gt_inds,
            num_fg,
        )

    def get_in_boxes_info(
        self,
        gt_bboxes,
        expanded_strides,
        x_shifts,
        y_shifts,
        A,
        G,
    ):
        # gt_bboxes.shape=[N, G, 4]  格式是cxcywh
        N = gt_bboxes.shape[0]
        # 先把每张图片每个格子的中心点x坐标、y坐标计算出来。
        expanded_strides_per_image = expanded_strides[0]  # [1, 8400] -> [8400, ]   每个格子的格子边长。
        x_shifts = x_shifts[0] * expanded_strides_per_image  # [8400, ]   每个格子左上角的x坐标。单位是1像素。[0, 8, 16, ..., 544, 576, 608]
        y_shifts = y_shifts[0] * expanded_strides_per_image  # [8400, ]   每个格子左上角的y坐标。单位是1像素。[0, 0, 0, ...,  608, 608, 608]
        x_centers = (x_shifts + 0.5 * expanded_strides_per_image).unsqueeze([0, 1])   # [1, 1, A]   每个格子中心点的x坐标。单位是1像素。
        x_centers = paddle.tile(x_centers, [N, G, 1])  # [N, G, A]  每张图片每个格子中心点的x坐标。单位是1像素。重复G次是为了让每个格子和每个gt比较。
        y_centers = (y_shifts + 0.5 * expanded_strides_per_image).unsqueeze([0, 1])   # [1, 1, A]   每个格子中心点的y坐标。单位是1像素。
        y_centers = paddle.tile(y_centers, [N, G, 1])  # [N, G, A]  每张图片每个格子中心点的y坐标。单位是1像素。重复G次是为了让每个格子和每个gt比较。

        gt_bboxes_l = (gt_bboxes[:, :, 0] - 0.5 * gt_bboxes[:, :, 2]).unsqueeze(2)   # [N, G, 1]   cx - w/2   gt左上角x坐标
        gt_bboxes_l = paddle.tile(gt_bboxes_l, [1, 1, A])  # [N, G, A]   重复A次

        gt_bboxes_r = (gt_bboxes[:, :, 0] + 0.5 * gt_bboxes[:, :, 2]).unsqueeze(2)   # [N, G, 1]   cx + w/2   gt右下角x坐标
        gt_bboxes_r = paddle.tile(gt_bboxes_r, [1, 1, A])  # [N, G, A]   重复A次

        gt_bboxes_t = (gt_bboxes[:, :, 1] - 0.5 * gt_bboxes[:, :, 3]).unsqueeze(2)   # [N, G, 1]   cy - h/2   gt左上角y坐标
        gt_bboxes_t = paddle.tile(gt_bboxes_t, [1, 1, A])  # [N, G, A]   重复A次

        gt_bboxes_b = (gt_bboxes[:, :, 1] + 0.5 * gt_bboxes[:, :, 3]).unsqueeze(2)   # [N, G, 1]   cy + h/2   gt右下角y坐标
        gt_bboxes_b = paddle.tile(gt_bboxes_b, [1, 1, A])  # [N, G, A]   重复A次

        # 每个格子的中心点是否在gt内部。
        b_l = x_centers - gt_bboxes_l  # [N, G, A]  格子的中心点x - gt左上角x坐标
        b_r = gt_bboxes_r - x_centers  # [N, G, A]  gt右下角x坐标 - 格子的中心点x
        b_t = y_centers - gt_bboxes_t  # [N, G, A]  格子的中心点y - gt左上角y坐标
        b_b = gt_bboxes_b - y_centers  # [N, G, A]  gt右下角y坐标 - 格子的中心点y
        bbox_deltas = paddle.stack([b_l, b_t, b_r, b_b], 3)  # [N, G, A, 4]  若在某个gt内部，则第3维值全>0
        is_in_boxes = paddle.min(bbox_deltas, axis=-1) > 0   # [N, G, A]  N*A个格子，若在某个gt内部，则为True
        is_in_boxes = paddle.cast(is_in_boxes, 'float32')   # [N, G, A]   N*A个格子，若在某个gt内部，则为1
        is_in_boxes_all = paddle.sum(is_in_boxes, axis=1)   # [N, A]      N*A个格子，在几个gt内部
        is_in_boxes_all = paddle.cast(is_in_boxes_all > 0, 'float32')   # [N, A]  N*A个格子，若在任何一个gt内部，则为1


        # gt中心点处再画一个的正方形镜像gt框。边长是2*center_radius*stride(3个特征图分别是8、16、32)
        center_radius = 2.5

        gt_bboxes_l = paddle.tile(gt_bboxes[:, :, 0:1], [1, 1, A]) \
                      - center_radius * expanded_strides_per_image.unsqueeze([0, 1])   # [N, G, A]   cx - r*s
        gt_bboxes_r = paddle.tile(gt_bboxes[:, :, 0:1], [1, 1, A]) \
                      + center_radius * expanded_strides_per_image.unsqueeze([0, 1])   # [N, G, A]   cx + r*s
        gt_bboxes_t = paddle.tile(gt_bboxes[:, :, 1:2], [1, 1, A]) \
                      - center_radius * expanded_strides_per_image.unsqueeze([0, 1])   # [N, G, A]   cy - r*s
        gt_bboxes_b = paddle.tile(gt_bboxes[:, :, 1:2], [1, 1, A]) \
                      + center_radius * expanded_strides_per_image.unsqueeze([0, 1])   # [N, G, A]   cy + r*s

        # 每个格子的中心点是否在镜像gt内部（即原gt中心附近）。
        c_l = x_centers - gt_bboxes_l  # [N, G, A]  格子的中心点x - 镜像gt左上角x坐标
        c_r = gt_bboxes_r - x_centers  # [N, G, A]  镜像gt右下角x坐标 - 格子的中心点x
        c_t = y_centers - gt_bboxes_t  # [N, G, A]  格子的中心点y - 镜像gt左上角y坐标
        c_b = gt_bboxes_b - y_centers  # [N, G, A]  镜像gt右下角y坐标 - 格子的中心点y
        center_deltas = paddle.stack([c_l, c_t, c_r, c_b], 3)    # [N, G, A, 4]  若在某个镜像gt内部，则第3维值全>0
        is_in_centers = paddle.min(center_deltas, axis=-1) > 0   # [N, G, A]  N*A个格子，若在某个镜像gt内部，则为True
        is_in_centers = paddle.cast(is_in_centers, 'float32')   # [N, G, A]   N*A个格子，若在某个镜像gt内部，则为1
        is_in_centers_all = paddle.sum(is_in_centers, axis=1)   # [N, A]      N*A个格子，在几个镜像gt内部
        is_in_centers_all = paddle.cast(is_in_centers_all > 0, 'float32')   # [N, A]  N*A个格子，若在任何一个镜像gt内部，则为1

        # 逻辑或运算。 [N, A] 每个格子是否是在 任意gt内部 或 任意gt的镜像gt内部（不要求同一个gt）
        # 值为1处的格子可以叫做“候选正样本”
        is_in_boxes_or_center = paddle.cast(is_in_boxes_all + is_in_centers_all > 0, 'float32')

        # 逻辑与运算。 [N, G, A] 每个格子是否是在 某个gt内部 且 这个gt的镜像gt内部（要求同一个gt）
        # 每个格子持有G个值，G个值中若至少有1个值为1，不难证明，这个格子其实也是“候选正样本”中的某个。
        # is_in_boxes_and_center的作用是 用来帮助确定 某些高质量的候选正样本 成为最终正样本。
        # 因为若某个格子既在gt内又在这个gt的镜像gt内时，它就更应该负责去学习这个gt。
        is_in_boxes_and_center = paddle.cast(is_in_boxes + is_in_centers > 1, 'float32')
        return is_in_boxes_or_center, is_in_boxes_and_center

    def dynamic_k_matching(self, cost, pair_wise_ious, gt_classes, N, G, A, is_in_boxes_or_center, is_gt):
        # Dynamic K
        # ---------------------------------------------------------------
        # cost.shape = [N, G, A]  每张图片 所有gt 和 所有预测框 两两之间的cost。
        # pair_wise_ious.shape = [N, G, A]  每张图片 所有gt 和 所有预测框 两两之间的iou。
        # gt_classes.shape = [N*G, ]  每张图片所有gt的类别id。
        # is_in_boxes_or_center.shape = [N, A]  每个格子是否是在 任意gt内部 或 任意gt的镜像gt内部（不要求同一个gt）。候选正样本处为1。
        # is_gt.shape = [N, G]   是真gt处为1。

        # 4-2-5-1.每个gt应该分配给几个预测框（格子）。
        # 表示最多只抽 与每个gt iou最高的10个预测框（格子）。
        n_candidate_k = 10
        # [N, G, n_candidate_k] 表示对于每个gt，选出前n_candidate_k个与它iou最高的预测框。
        topk_ious, _ = paddle.topk(pair_wise_ious, n_candidate_k, axis=-1)

        # [N, G]  最匹配当前gt的前n_candidate_k个的预测框iou求和。
        dynamic_ks = topk_ious.sum(-1)
        dynamic_ks = paddle.clip(dynamic_ks, 1.0, np.inf)  # [N, G]   dynamic_ks限制在区间[1.0, np.inf]内
        dynamic_ks = paddle.cast(dynamic_ks, 'int32')      # [N, G]   取整。表示每个gt应分配给了几个预测框。最少1个。
        max_dynamic_ks = dynamic_ks.max(-1)  # [N, ]  每张图片所有gt的dynamic_ks的最大值
        max_k = max_dynamic_ks.max()         # [1, ]  所有图片所有gt的dynamic_ks的最大值

        # 4-2-5-2.根据4-2-5-1步，构造一个形状为[N, G, A]的matching_matrix，
        # 每个gt前dynamic_ks个cost最小的预测框处填入1，代表gt分配给了这个预测框。
        # 不放心的话，再次将假gt的cost增大。因为不能用假gt确定最终正样本。
        cost += (1.0 - is_gt.unsqueeze(2)) * 100000.0
        # 不放心的话，再次将非候选正样本的cost增大。因为非候选正样本没有资格成为最终正样本。
        cost += (1.0 - is_in_boxes_or_center.unsqueeze(1)) * 100000.0
        # min_cost。       [N, G, max_k] 每个gt，取前max_k个cost最小的cost
        # min_cost_index。 [N, G, max_k] 每个gt，取前max_k个cost最小的cost的坐标。即哪些预测框（格子）与这个gt的cost最小。
        min_cost, min_cost_index = paddle.topk(cost, k=max_k, axis=2, largest=False, sorted=True)

        matching_matrix = paddle.zeros([N * G * A, ], 'float32')  # [N*G*A, ]
        gt_ind = paddle.arange(end=N * G, dtype='int32').unsqueeze(-1)  # [N*G, 1]  每个gt在matching_matrix中的下标。
        min_cost_index = min_cost_index.reshape((N * G, max_k))  # [N*G, max_k]
        min_cost_index = gt_ind * A + min_cost_index  # [N*G, max_k]
        min_cost_index = min_cost_index.flatten()     # [N*G*max_k, ]

        # 下三角全是1的矩阵
        topk_mask = paddle.ones((max_k, max_k), 'float32')  # [max_k, max_k]
        topk_mask = paddle.tril(topk_mask, diagonal=0)      # [max_k, max_k]
        fill_value = paddle.gather(topk_mask, dynamic_ks.reshape((-1,)) - 1)  # [N*G, max_k]   填入matching_matrix
        fill_value *= is_gt.reshape((-1, 1))    # [N*G, max_k]  还要处理假gt，假gt处全部填0
        fill_value = fill_value.reshape((-1,))  # [N*G*max_k, ]   填入matching_matrix

        # 填入matching_matrix
        matching_matrix = paddle.scatter(matching_matrix, min_cost_index, fill_value, overwrite=True)
        matching_matrix = matching_matrix.reshape((N, G, A))  # [N, G, A]

        # 4-2-5-3.如果有预测框anchor（花心大萝卜）匹配到了1个以上的gt时，做特殊处理。
        # 因为不可能让1个预测框学习多个gt，它只有85位信息，做不到；做法是让预测框学习与其具有最小cost的gt。
        # [N, A]  每个预测框（格子）匹配到了几个gt？
        anchor_matching_gt = matching_matrix.sum(1)

        # 如果有预测框（花心大萝卜）匹配到了1个以上的gt时，做特殊处理。
        if paddle.cast(anchor_matching_gt > 1, 'float32').sum() > 0:
            # 首先，找到与花心大萝卜具有最小cost的gt。
            # 找到 花心大萝卜 的下标（这是在anchor_matching_gt.shape[N, A]中的下标）。假设有R个花心大萝卜。
            index = paddle.nonzero(anchor_matching_gt > 1)  # [R, 2]  每个花心大萝卜2个坐标。第0个坐标表示第几张图片，第1个坐标表示第几个格子。
            cost_t = cost.transpose((0, 2, 1))              # [N, G, A] -> [N, A, G]  转置好提取其cost
            cost2 = paddle.gather_nd(cost_t, index)         # [R, G]  抽出 R个花心大萝卜 与 gt 两两之间的cost。
            cost2 = cost2.transpose((1, 0))                 # [G, R]  gt 与 R个花心大萝卜 两两之间的cost。
            cost_argmin = cost2.argmin(axis=0)              # [R, ]  为 每个花心大萝卜 找到 与其cost最小的gt 的下标

            # 准备one_hot
            one_hots = F.one_hot(cost_argmin, num_classes=G)  # [R, G]
            # 花心大萝卜 处 填入one_hot
            matching_matrix = matching_matrix.transpose((0, 2, 1))  # [N, G, A] -> [N, A, G]  转置好以让scatter()填入
            matching_matrix = matching_matrix.reshape((N * A, G))  # [N*A, G]  reshape好以让scatter()填入
            index = index[:, 0] * A + index[:, 1]
            matching_matrix = paddle.scatter(matching_matrix, index, one_hots, overwrite=True)  # [N*A, G]  scatter()填入

            # matching_matrix变回原来的形状
            matching_matrix = matching_matrix.reshape((N, A, G))  # [N, A, G]
            matching_matrix = matching_matrix.transpose((0, 2, 1))  # [N, A, G] -> [N, G, A]

        # 4-2-5-4.收尾工作，准备监督信息以计算损失。
        # 第一步，准备 置信度obj-ness 需要的监督信息。
        # [N, A]  是否是前景（最终正样本）
        fg_mask = matching_matrix.sum(1) > 0.0     # [N, A]
        fg_mask = paddle.cast(fg_mask, 'float32')  # [N, A]   fg_mask作用是监督置信度，计算置信度损失。是最终正样本处为1。
        num_fg = fg_mask.sum()    # 所有图片前景个数

        # 第二步，准备 各类别概率 需要的监督信息。确定最终正样本需要学习的类别id。
        # 最终正样本在fg_mask.shape=[N, A]中的坐标
        pos_index = paddle.nonzero(fg_mask > 0)  # [num_fg, 2]
        image_id = pos_index[:, 0]               # [num_fg, ]  最终正样本是第几张图片的最终正样本。

        matching_matrix_t = matching_matrix.transpose((0, 2, 1))  # [N, G, A] -> [N, A, G]  转置好以便gather_nd()
        matched_gt_inds = paddle.gather_nd(matching_matrix_t, pos_index)  # [num_fg, G]
        matched_gt_inds = matched_gt_inds.argmax(1)  # [num_fg, ]  最终正样本是匹配到了第几个gt（每张图片在[G, ]中的坐标）
        matched_gt_inds += image_id * G              # [num_fg, ]  最终正样本是匹配到了第几个gt（在gt_classes.shape=[N*G, ]中的坐标）
        # 最终正样本需要学习的类别id
        gt_matched_classes = paddle.gather(gt_classes, matched_gt_inds)  # [num_fg, ]

        # 第三步，取出最终正样本和所学gt的iou。
        # [N, G, A]    所有gt 和 所有预测框 两两之间的iou。matching_matrix第1维其实最多只有1个值非0，所以变成了最终正样本和所学gt的iou。
        ious = (matching_matrix * pair_wise_ious)
        # [N, A]       最终正样本和所学gt的iou。
        ious = ious.sum(1)
        # [num_fg, ]   取出最终正样本和所学gt的iou。
        pred_ious_this_matching = paddle.gather_nd(ious, pos_index)
        # 返回这些：
        # num_fg。                  [1, ]       所有图片前景（最终正样本）个数
        # gt_matched_classes。      [num_fg, ]  最终正样本需要学习的类别id
        # pred_ious_this_matching。 [num_fg, ]  最终正样本和所学gt的iou
        # matched_gt_inds。         [num_fg, ]  最终正样本是匹配到了第几个gt（在gt_classes.shape=[N*G, ]中的坐标）
        # fg_mask。                 [N, A]      最终正样本处为1
        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds, fg_mask

