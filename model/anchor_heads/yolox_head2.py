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
import paddle.fluid.layers as L

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
        box_a = L.concat([bboxes_a[:, :2] - bboxes_a[:, 2:] * 0.5,
                          bboxes_a[:, :2] + bboxes_a[:, 2:] * 0.5], axis=-1)
        box_b = L.concat([bboxes_b[:, :2] - bboxes_b[:, 2:] * 0.5,
                          bboxes_b[:, :2] + bboxes_b[:, 2:] * 0.5], axis=-1)

    box_a_rb = L.reshape(box_a[:, 2:], (A, 1, 2))
    box_a_rb = L.expand(box_a_rb, [1, B, 1])
    box_b_rb = L.reshape(box_b[:, 2:], (1, B, 2))
    box_b_rb = L.expand(box_b_rb, [A, 1, 1])
    max_xy = L.elementwise_min(box_a_rb, box_b_rb)

    box_a_lu = L.reshape(box_a[:, :2], (A, 1, 2))
    box_a_lu = L.expand(box_a_lu, [1, B, 1])
    box_b_lu = L.reshape(box_b[:, :2], (1, B, 2))
    box_b_lu = L.expand(box_b_lu, [A, 1, 1])
    min_xy = L.elementwise_max(box_a_lu, box_b_lu)

    inter = L.relu(max_xy - min_xy)
    inter = inter[:, :, 0] * inter[:, :, 1]

    box_a_w = box_a[:, 2]-box_a[:, 0]
    box_a_h = box_a[:, 3]-box_a[:, 1]
    area_a = box_a_h * box_a_w
    area_a = L.reshape(area_a, (A, 1))
    area_a = L.expand(area_a, [1, B])  # [A, B]

    box_b_w = box_b[:, 2]-box_b[:, 0]
    box_b_h = box_b[:, 3]-box_b[:, 1]
    area_b = box_b_h * box_b_w
    area_b = L.reshape(area_b, (1, B))
    area_b = L.expand(area_b, [A, 1])  # [A, B]

    union = area_a + area_b - inter + 1e-9
    return inter / union  # [A, B]





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
            self.cls_preds.append(
                nn.Conv2D(
                    in_channels=int(256 * width),
                    out_channels=self.n_anchors * self.num_classes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.reg_preds.append(
                nn.Conv2D(
                    in_channels=int(256 * width),
                    out_channels=4,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
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
        # self.iou_loss = IOUloss(reduction="none")
        self.iou_loss = iou_loss
        self.strides = strides
        # self.grids = [torch.zeros(1)] * len(in_channels)
        self.grids = [paddle.zeros((1, ))] * len(in_channels)

        self.yolo_loss = yolo_loss
        self.nms_cfg = nms_cfg

    def initialize_biases(self, prior_prob):
        for conv in self.cls_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        for conv in self.obj_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

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
                output, grid = self.get_output_and_grid(
                    output, k, stride_this_level
                )
                x_shifts.append(grid[:, :, 0])
                y_shifts.append(grid[:, :, 1])
                expanded_stride = paddle.ones((1, grid.shape[1]), dtype=xin[0].dtype) * stride_this_level
                expanded_strides.append(expanded_stride)
                if self.use_l1:
                    batch_size = reg_output.shape[0]
                    hsize, wsize = reg_output.shape[-2:]
                    reg_output = reg_output.view(
                        batch_size, self.n_anchors, 4, hsize, wsize
                    )
                    reg_output = reg_output.permute(0, 1, 3, 4, 2).reshape(
                        batch_size, -1, 4
                    )
                    origin_preds.append(reg_output.clone())

            else:
                output = paddle.concat(
                    [reg_output, L.sigmoid(obj_output), L.sigmoid(cls_output)], 1
                )

            outputs.append(output)

        return outputs, x_shifts, y_shifts, expanded_strides, origin_preds

    def get_prediction(self, xin, im_info):
        outputs, x_shifts, y_shifts, expanded_strides, origin_preds = self.get_outputs(xin)


        self.hw = [x.shape[-2:] for x in outputs]
        # [batch, 85, n_anchors_all]
        outputs = paddle.concat(
            [L.reshape(x, (x.shape[0], x.shape[1], -1)) for x in outputs], axis=2
        )
        # [batch, n_anchors_all, 85]
        outputs = paddle.transpose(outputs, [0, 2, 1])

        if self.decode_in_inference:
            outputs = self.decode_outputs(outputs)

        # 自加的后处理
        yolo_boxes = outputs[:, :, :4]  # [N, n_anchors_all, 4]  cxcywh格式

        # [N, n_anchors_all, 4]  左上角坐标+右下角坐标格式，即x0y0x1y1格式
        yolo_boxes = L.concat([yolo_boxes[:, :, :2] - yolo_boxes[:, :, 2:] * 0.5,
                               yolo_boxes[:, :, :2] + yolo_boxes[:, :, 2:] * 0.5], axis=-1)

        # 上面的坐标是对输入的640x640分辨率图片的预测，需要把坐标缩放成原图中的坐标
        im_scale = im_info[:, 2:3]  # [N, 1]
        im_scale = L.unsqueeze(im_scale, 1)  # [N, 1, 1]
        yolo_boxes /= im_scale

        yolo_scores = outputs[:, :, 4:5] * outputs[:, :, 5:]
        yolo_scores = paddle.transpose(yolo_scores, [0, 2, 1])  # [N, 80, n_anchors_all]

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
        outputs = paddle.concat(outputs, 1)

        dic2 = np.load('data.npz')
        imgs2 = dic2['imgs'].astype(np.float32)
        labels2 = dic2['labels'].astype(np.float32)
        # 若anchor在假的gt内部（或中心附近），则anchor会被视为候选正样本。所以假的gt必须设置得很远。
        not_gt = np.array([-10.0, -9999.0, -9999.0, 10.0, 10.0])
        not_gt = not_gt.astype(np.float32)
        for p1 in range(2):
            for p2 in range(120):
                aaaaa = labels2[p1, p2, :]
                sum1 = np.sum(aaaaa)
                if sum1 < 0.0001:
                    labels2[p1, p2, :] = not_gt
        outputs2 = dic2['outputs'].astype(np.float32)
        losses02 = dic2['losses0']
        losses12 = dic2['losses1']
        losses22 = dic2['losses2']
        losses32 = dic2['losses3']
        losses42 = dic2['losses4']
        losses52 = dic2['losses5']
        images2 = paddle.to_tensor(imgs2)
        gt_class_bbox2 = paddle.to_tensor(labels2)
        outputs2 = paddle.to_tensor(outputs2)
        images2 = paddle.cast(images2, 'float32')
        gt_class_bbox2 = paddle.cast(gt_class_bbox2, 'float32')
        outputs2 = paddle.cast(outputs2, 'float32')

        # losses = self.get_losses(
        #     x_shifts,
        #     y_shifts,
        #     expanded_strides,
        #     labels,
        #     outputs,
        #     origin_preds,
        #     dtype=xin[0].dtype,
        # )
        losses = self.get_losses(
            x_shifts,
            y_shifts,
            expanded_strides,
            gt_class_bbox2,
            outputs2,
            origin_preds,
            dtype=xin[0].dtype,
        )
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
        output = paddle.reshape(output, (batch_size, self.n_anchors * hsize * wsize, -1))
        grid = paddle.reshape(grid, (1, -1, 2))
        # output[:, :, :2] = (output[:, :, :2] + grid) * stride
        # output[:, :, 2:4] = paddle.exp(output[:, :, 2:4]) * stride
        xy = (output[:, :, :2] + grid) * stride
        wh = paddle.exp(output[:, :, 2:4]) * stride
        output = paddle.concat([xy, wh, output[:, :, 4:]], 2)
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
        # 1.把网络输出切分成预测框、置信度、类别概率
        bbox_preds = outputs[:, :, :4]  # [batch, n_anchors_all, 4]
        obj_preds = outputs[:, :, 4].unsqueeze(-1)  # [batch, n_anchors_all, 1]
        cls_preds = outputs[:, :, 5:]  # [batch, n_anchors_all, n_cls]

        # 2.计算gt数目
        labels = L.cast(labels, 'float32')  # [N, 120, 5]
        if_gt = paddle.sum(labels, axis=2)  # [N, 120]
        if_gt = L.cast(if_gt > 0, 'float32')  # [N, 120] 是gt处为1
        nlabel = paddle.sum(if_gt, axis=1)    # [N, ] 每张图片gt数量
        nlabel = L.cast(nlabel, 'int32')
        nlabel.stop_gradient = True

        # 3.拼接用到的常量张量x_shifts、y_shifts、expanded_strides
        total_num_anchors = outputs.shape[1]  # 一张图片出8400个anchor
        x_shifts = paddle.concat(x_shifts, 1)  # [1, n_anchors_all]  每个格子左上角的x坐标。单位是下采样步长。比如，第0个特征图的1代表的是8个像素，第2个特征图的1代表的是32个像素。
        y_shifts = paddle.concat(y_shifts, 1)  # [1, n_anchors_all]  每个格子左上角的y坐标。单位是下采样步长。
        expanded_strides = paddle.concat(expanded_strides, 1)  # [1, n_anchors_all]  每个anchor对应的下采样倍率。依次是8, 16, 32
        if self.use_l1:
            origin_preds = paddle.concat(origin_preds, 1)

        # 4.遍历每张图片，决定哪些样本作为正样本
        cls_targets = []
        reg_targets = []
        l1_targets = []
        obj_targets = []
        fg_masks = []

        num_fg = 0.0
        # num_gts = 0.0

        # print('nlabel44444444444444444444444444444')
        # print(nlabel)
        for batch_idx in range(outputs.shape[0]):
            # print(batch_idx)
            num_gt = int(nlabel[batch_idx])
            # num_gts += num_gt
            if num_gt == 0:
                cls_target = paddle.zeros((0, self.num_classes), 'float32')
                reg_target = paddle.zeros((0, 4), 'float32')
                l1_target = paddle.zeros((0, 4), 'float32')
                obj_target = paddle.zeros((total_num_anchors, 1), 'float32')
                fg_mask = paddle.zeros((total_num_anchors, ), 'float32')
            else:
                # 4-1.将这张图片的gt的坐标宽高、类别id提取出来。
                gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:5]  # [num_gt, 4]
                gt_classes = labels[batch_idx, :num_gt, 0]       # [num_gt, ]
                bboxes_preds_per_image = bbox_preds[batch_idx]   # [8400, 4]

                try:
                    # 4-2.get_assignments()确定正负样本，里面的张量不需要梯度。
                    # num_fg_img type=float   最终前景的个数，即T。
                    # gt_matched_classes shape=[T, ]       T个最终正样本需要学习的类别id。
                    # pred_ious_this_matching shape=[T, ]  T个最终正样本和匹配的gt的iou。
                    # matched_gt_inds shape=[T, ]          T个最终正样本 匹配到的gt 的下标。
                    # fg_mask shape=[8400, ]               最终正样本处为1。
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        total_num_anchors,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        bbox_preds,
                        obj_preds,
                        labels,
                    )
                except RuntimeError:
                    logger.error(
                        "OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                           CPU mode is applied in this batch. If you want to avoid this issue, \
                           try to reduce the batch size or image size."
                    )
                    # torch.cuda.empty_cache()
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        total_num_anchors,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        bbox_preds,
                        obj_preds,
                        labels,
                        "cpu",
                    )

                # torch.cuda.empty_cache()
                # 增加前景数
                num_fg += num_fg_img

                eye = paddle.eye(self.num_classes, dtype='float32')  # [80, 80]
                one_hot = L.gather(eye, gt_matched_classes)  # [T, 80]  T个最终正样本需要学习的类别one_hot向量。
                # pred_ious_this_matching shape=[T, ]  T个最终正样本和匹配的gt的iou。
                cls_target = one_hot * pred_ious_this_matching.unsqueeze(-1)  # [T, 80]  T个最终正样本需要学习的类别one_hot向量需乘以匹配到的gt的iou。


                obj_target = fg_mask.unsqueeze(-1)  # [8400, 1]   每个格子objness处需要学习的目标。
                reg_target = L.gather(gt_bboxes_per_image, matched_gt_inds)  # [T, 4]  T个最终正样本需要学习的预测框xywh。
                if self.use_l1:
                    l1_target = self.get_l1_target(
                        outputs.new_zeros((num_fg_img, 4)),
                        gt_bboxes_per_image[matched_gt_inds],
                        expanded_strides[0][fg_mask],
                        x_shifts=x_shifts[0][fg_mask],
                        y_shifts=y_shifts[0][fg_mask],
                    )

            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target)


            fg_masks.append(fg_mask)
            if self.use_l1:
                l1_targets.append(l1_target)

        cls_targets = paddle.concat(cls_targets, 0)
        reg_targets = paddle.concat(reg_targets, 0)
        obj_targets = paddle.concat(obj_targets, 0)
        fg_masks = paddle.concat(fg_masks, 0)


        dic2 = np.load('targets.npz')
        cls_targets2 = dic2['reg_targets']
        cls_targets3 = reg_targets.numpy()
        ddd = np.sum((cls_targets2 - cls_targets3) ** 2)
        print('ddd=%.6f' % ddd)
        cls_targets2 = dic2['obj_targets']
        cls_targets3 = obj_targets.numpy()
        ddd = np.sum((cls_targets2 - cls_targets3) ** 2)
        print('ddd=%.6f' % ddd)
        cls_targets2 = dic2['fg_masks']
        cls_targets3 = fg_masks.numpy()
        ddd = np.sum((cls_targets2 - cls_targets3) ** 2)
        print('ddd=%.6f' % ddd)
        ddd = 0.0
        # 第一张图片
        # if num_gt == 8:
        #     print('ddd=%.6f' % ddd)
        # 第二张图片
        # if num_gt == 16:
        #     print('ddd=%.6f' % ddd)
        # 经过校验，输出一样

        if self.use_l1:
            l1_targets = paddle.concat(l1_targets, 0)

        num_fg = max(num_fg, 1)
        bbox_preds = paddle.reshape(bbox_preds, [-1, 4])  # [N*8400, 4]
        pos_index = L.where(fg_masks > 0)[:, 0]  # [?, ]
        pos_bbox_preds = L.gather(bbox_preds, pos_index)  # [?, 4]
        loss_iou = (
            self.iou_loss(pos_bbox_preds, reg_targets)
        ).sum() / num_fg
        # obj_preds = paddle.cast(obj_preds, 'float32')
        # obj_targets = paddle.cast(obj_targets, 'float32')
        loss_obj = (
            self.bcewithlog_loss(paddle.reshape(obj_preds, [-1, 1]), obj_targets)
        ).sum() / num_fg

        cls_preds = paddle.reshape(cls_preds, [-1, self.num_classes])  # [N*8400, 80]
        pos_cls_preds = L.gather(cls_preds, pos_index)  # [?, 80]
        # pos_cls_preds = paddle.cast(pos_cls_preds, 'float32')
        loss_cls = (
            self.bcewithlog_loss(pos_cls_preds, cls_targets)
        ).sum() / num_fg
        if self.use_l1:
            loss_l1 = (
                self.l1_loss(origin_preds.view(-1, 4)[fg_masks], l1_targets)
            ).sum() / num_fg
        else:
            loss_l1 = 0.0

        reg_weight = 5.0
        # loss = reg_weight * loss_iou + loss_obj + loss_cls + loss_l1

        # losses = (
        #     loss,
        #     reg_weight * loss_iou,
        #     loss_obj,
        #     loss_cls,
        #     loss_l1,
        #     num_fg / max(num_gts, 1),
        # )
        # losses = {
        #     "loss_iou": reg_weight * loss_iou,
        #     "loss_obj": loss_obj,
        #     "loss_cls": loss_cls,
        # }
        losses = {
            "loss_iou": reg_weight * loss_iou,
            "loss_obj": loss_obj,
            "loss_cls": loss_cls,
        }
        # print('lossssssssssssssssssssssssssssssssssssssssss')
        # print(loss_iou)
        # print(loss_obj)
        # print(loss_cls)
        if self.use_l1:
            losses["loss_l1"] = loss_l1
        return losses

    def get_l1_target(self, l1_target, gt, stride, x_shifts, y_shifts, eps=1e-8):
        l1_target[:, 0] = gt[:, 0] / stride - x_shifts
        l1_target[:, 1] = gt[:, 1] / stride - y_shifts
        l1_target[:, 2] = torch.log(gt[:, 2] / stride + eps)
        l1_target[:, 3] = torch.log(gt[:, 3] / stride + eps)
        return l1_target

    def get_assignments(
        self,
        batch_idx,
        num_gt,
        total_num_anchors,
        gt_bboxes_per_image,
        gt_classes,
        bboxes_preds_per_image,
        expanded_strides,
        x_shifts,
        y_shifts,
        cls_preds,
        bbox_preds,
        obj_preds,
        labels,
        mode="gpu",
    ):
        # 4-2.get_assignments()确定正负样本，里面的张量不需要梯度。
        # 4-2-1.只根据每张图片的gt_bboxes_per_image shp=(M, 4)确定哪些作为 候选正样本。
        # fg_mask shp=(8400, ) 为1处表示是 候选正样本。假设有M个位置处为1。
        # is_in_boxes_and_center shp=(num_gt, M) M个候选正样本 与num_gt个gt的关系，是否在gt内部和中心。
        fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(
            gt_bboxes_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
            total_num_anchors,
            num_gt,
        )
        dic2 = np.load('is_in_boxes_or_center.npz')
        is_in_boxes_or_center2 = dic2['is_in_boxes_or_center']
        is_in_boxes_or_center3 = fg_mask.numpy()
        print()
        # 第一张图片
        if num_gt == 8:
            is_in_boxes_or_center4 = is_in_boxes_or_center2[0, :]
            ddd = np.sum((is_in_boxes_or_center4 - is_in_boxes_or_center3) ** 2)
            print('ddd=%.6f' % ddd)
        # 第二张图片
        if num_gt == 16:
            is_in_boxes_or_center4 = is_in_boxes_or_center2[1, :]
            ddd = np.sum((is_in_boxes_or_center4 - is_in_boxes_or_center3) ** 2)
            print('ddd=%.6f' % ddd)

        dic2 = np.load('is_in_boxes_and_center.npz')
        is_in_boxes_and_center2 = dic2['is_in_boxes_and_center']
        is_in_boxes_and_center3 = is_in_boxes_and_center.numpy()
        indexxx = L.where(fg_mask > 0)[:, 0]  # [M, ]
        indexx3 = indexxx.numpy()
        no_indexx3 = []
        for kkk in range(8400):
            if kkk not in indexx3:
                no_indexx3.append(kkk)
        no_indexx3 = np.array(no_indexx3).astype(np.int64)
        print()
        # 第一张图片
        if num_gt == 8:
            is_in_boxes_and_center4 = is_in_boxes_and_center2[0, :8, :]
            is_in_boxes_and_center4 = is_in_boxes_and_center4.transpose(1, 0)
            is_in_boxes_and_center444 = is_in_boxes_and_center4[no_indexx3]
            ddddd = np.sum(is_in_boxes_and_center444)
            is_in_boxes_and_center4 = is_in_boxes_and_center4[indexx3]
            is_in_boxes_and_center4 = is_in_boxes_and_center4.transpose(1, 0)
            ddd = np.sum((is_in_boxes_and_center4 - is_in_boxes_and_center3) ** 2)
            print('ddd=%.6f' % ddd)
        # 第二张图片
        if num_gt == 16:
            is_in_boxes_and_center4 = is_in_boxes_and_center2[1, :, :]
            is_in_boxes_and_center4 = is_in_boxes_and_center4.transpose(1, 0)
            is_in_boxes_and_center4 = is_in_boxes_and_center4[indexx3]
            is_in_boxes_and_center4 = is_in_boxes_and_center4.transpose(1, 0)
            ddd = np.sum((is_in_boxes_and_center4 - is_in_boxes_and_center3) ** 2)
            print('ddd=%.6f' % ddd)



        # 4-2-2.抽出候选正样本预测的bbox、obj和cls。
        index = L.where(fg_mask > 0)[:, 0]  # [M, ]
        bboxes_preds_per_image = L.gather(bboxes_preds_per_image, index)            # [M, 4]
        cls_preds_ = L.gather(cls_preds[batch_idx], index)            # [M, 80]
        obj_preds_ = L.gather(obj_preds[batch_idx], index)            # [M, 1]
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]

        # 4-2-3.计算 gt 和 候选正样本框 两两之间的iou 的cost，iou越大cost越小，越有可能成为最终正样本。
        bboxes_preds_per_image = paddle.cast(bboxes_preds_per_image, 'float32')
        pair_wise_ious = bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, False)  # [num_gt, M]  两两之间的iou。


        # dic2 = np.load('pair_wise_ious.npz')
        # pair_wise_ious2 = dic2['pair_wise_ious']
        # index3 = index.numpy()
        # # 第一张图片
        # if num_gt == 8:
        #     pair_wise_ious3 = pair_wise_ious.numpy()
        #     pair_wise_ious4 = pair_wise_ious2[0, :8, :]
        #     pair_wise_ious4 = pair_wise_ious4.transpose(1, 0)
        #     pair_wise_ious4 = pair_wise_ious4[index3]
        #     pair_wise_ious4 = pair_wise_ious4.transpose(1, 0)
        #     ddd = np.sum((pair_wise_ious4 - pair_wise_ious3) ** 2)
        #     print('ddd=%.6f' % ddd)
        # # 第二张图片
        # if num_gt == 16:
        #     pair_wise_ious3 = pair_wise_ious.numpy()
        #     pair_wise_ious4 = pair_wise_ious2[1, :, :]
        #     pair_wise_ious4 = pair_wise_ious4.transpose(1, 0)
        #     pair_wise_ious4 = pair_wise_ious4[index3]
        #     pair_wise_ious4 = pair_wise_ious4.transpose(1, 0)
        #     ddd = np.sum((pair_wise_ious4 - pair_wise_ious3) ** 2)
        #     print('ddd=%.6f' % ddd)
        # 经过校验，输出一样

        # print('yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy')
        # print(pair_wise_ious)
        pair_wise_ious_loss = -paddle.log(pair_wise_ious + 1e-8)  # [num_gt, M]  iou取对数再取相反数。
        pair_wise_ious_loss.stop_gradient = True

        # dic2 = np.load('pair_wise_ious_loss.npz')
        # pair_wise_ious_loss2 = dic2['pair_wise_ious_loss']
        # pair_wise_ious_loss3 = pair_wise_ious_loss.numpy()
        # indexxx = L.where(fg_mask > 0)[:, 0]  # [M, ]
        # indexx3 = indexxx.numpy()
        # # 第一张图片
        # if num_gt == 8:
        #     pair_wise_ious_loss4 = pair_wise_ious_loss2[0, :8, :]
        #     pair_wise_ious_loss4 = pair_wise_ious_loss4.transpose(1, 0)
        #     pair_wise_ious_loss4 = pair_wise_ious_loss4[indexx3]
        #     pair_wise_ious_loss4 = pair_wise_ious_loss4.transpose(1, 0)
        #     ddd = np.sum((pair_wise_ious_loss4 - pair_wise_ious_loss3) ** 2)
        #     print('ddd=%.6f' % ddd)
        #
        #     pair_wise_ious_loss5 = pair_wise_ious_loss2[0, 8:, :]
        #     pair_wise_ious_loss6 = pair_wise_ious_loss2[0, 8:, :]
        # # 第二张图片
        # if num_gt == 16:
        #     pair_wise_ious_loss4 = pair_wise_ious_loss2[1, :, :]
        #     pair_wise_ious_loss4 = pair_wise_ious_loss4.transpose(1, 0)
        #     pair_wise_ious_loss4 = pair_wise_ious_loss4[indexx3]
        #     pair_wise_ious_loss4 = pair_wise_ious_loss4.transpose(1, 0)
        #     ddd = np.sum((pair_wise_ious_loss4 - pair_wise_ious_loss3) ** 2)
        #     print('ddd=%.6f' % ddd)

        # 4-2-4.计算 gt 和 候选正样本框 两两之间的cls 的cost，cost越小，越有可能成为最终正样本。
        p1 = paddle.cast(cls_preds_, 'float32').unsqueeze(0)  # [1, M, 80]
        p2 = paddle.cast(obj_preds_, 'float32').unsqueeze(0)  # [1, M, 1]
        p = L.sigmoid(p1) * L.sigmoid(p2)  # [1, M, 80]  各类别分数
        p = L.expand(p, [num_gt, 1, 1])  # [num_gt, M, 80]  各类别分数
        cls_preds_ = p  # 各类别分数
        p = L.sqrt(cls_preds_)  # [num_gt, M, 80]  各类别分数开根号求平均
        # 获得num_gt个gt的one_hot类别向量，每个候选正样本持有一个。
        eye = paddle.eye(self.num_classes, dtype='float32')  # [80, 80]
        gt_classes = paddle.cast(gt_classes, 'int32')  # [num_gt, ]
        one_hot = L.gather(eye, gt_classes)
        one_hot = one_hot.unsqueeze(1)
        one_hot = L.expand(one_hot, [1, num_in_boxes_anchor, 1])  # [num_gt, M, 80]  获得num_gt个gt的one_hot类别向量，每个候选正样本持有一个。
        gt_cls_per_image = one_hot
        # 二值交叉熵
        # pair_wise_cls_loss = F.binary_cross_entropy(
        #     cls_preds_.sqrt_(), gt_cls_per_image, reduction="none"
        # ).sum(-1)
        # 二值交叉熵
        pos_loss = gt_cls_per_image * (0 - L.log(p + 1e-9))              # [num_gt, M, 80]
        neg_loss = (1.0 - gt_cls_per_image) * (0 - L.log(1 - p + 1e-9))  # [num_gt, M, 80]
        pair_wise_cls_loss = pos_loss + neg_loss                         # [num_gt, M, 80]
        pair_wise_cls_loss = L.reduce_sum(pair_wise_cls_loss, dim=-1)    # [num_gt, M]  cost越小，越有可能成为最终正样本。
        pair_wise_cls_loss.stop_gradient = True

        dic2 = np.load('pair_wise_cls_loss.npz')
        pair_wise_cls_loss2 = dic2['pair_wise_cls_loss']
        pair_wise_cls_loss3 = pair_wise_cls_loss.numpy()
        indexxx = L.where(fg_mask > 0)[:, 0]  # [M, ]
        indexx3 = indexxx.numpy()
        # 第一张图片
        if num_gt == 8:
            pair_wise_cls_loss4 = pair_wise_cls_loss2[0, :8, :]
            pair_wise_cls_loss4 = pair_wise_cls_loss4.transpose(1, 0)
            pair_wise_cls_loss4 = pair_wise_cls_loss4[indexx3]
            pair_wise_cls_loss4 = pair_wise_cls_loss4.transpose(1, 0)
            ddd = np.sum((pair_wise_cls_loss4 - pair_wise_cls_loss3) ** 2)
            print('ddd=%.6f' % ddd)

            pair_wise_cls_loss5 = pair_wise_cls_loss2[0, 8:, :]
            pair_wise_cls_loss6 = pair_wise_cls_loss2[0, 8:, :]
        # 第二张图片
        if num_gt == 16:
            pair_wise_cls_loss4 = pair_wise_cls_loss2[1, :, :]
            pair_wise_cls_loss4 = pair_wise_cls_loss4.transpose(1, 0)
            pair_wise_cls_loss4 = pair_wise_cls_loss4[indexx3]
            pair_wise_cls_loss4 = pair_wise_cls_loss4.transpose(1, 0)
            ddd = np.sum((pair_wise_cls_loss4 - pair_wise_cls_loss3) ** 2)
            print('ddd=%.6f' % ddd)

        # 4-2-5.计算 gt 和 候选正样本框 两两之间的 总 的cost，cost越小，越有可能成为最终正样本。
        # is_in_boxes_and_center是1 cost越小，越有可能成为最终正样本。
        # is_in_boxes_and_center是0 cost越大，越不可能成为最终正样本。
        cost = (
            pair_wise_cls_loss
            + 3.0 * pair_wise_ious_loss
            + 100000.0 * (1.0 - is_in_boxes_and_center)
        )  # [num_gt, M]

        dic2 = np.load('cost.npz')
        cost2 = dic2['cost']
        cost3 = cost.numpy()
        indexxx = L.where(fg_mask > 0)[:, 0]  # [M, ]
        indexx3 = indexxx.numpy()
        # 第一张图片
        if num_gt == 8:
            cost4 = cost2[0, :8, :]
            cost4 = cost4.transpose(1, 0)
            cost4 = cost4[indexx3]
            cost4 = cost4.transpose(1, 0)
            ddd = np.sum((cost4 - cost3) ** 2)
            print('ddd=%.6f' % ddd)
        # 第二张图片
        if num_gt == 16:
            cost4 = cost2[1, :, :]
            cost4 = cost4.transpose(1, 0)
            cost4 = cost4[indexx3]
            cost4 = cost4.transpose(1, 0)
            ddd = np.sum((cost4 - cost3) ** 2)
            print('ddd=%.6f' % ddd)


        cost.stop_gradient = True
        pair_wise_ious.stop_gradient = True
        gt_classes.stop_gradient = True
        fg_mask.stop_gradient = True

        # 4-2-6.根据cost从 候选正样本 中 选出 最终正样本。
        # num_fg type=float   最终前景的个数，即T。
        # gt_matched_classes shape=[T, ]       T个最终正样本需要学习的类别id。
        # pred_ious_this_matching shape=[T, ]  T个最终正样本和匹配的gt的iou。
        # matched_gt_inds shape=[T, ]          T个最终正样本 匹配到的gt 的下标。
        # fg_mask shape=[8400, ]               最终正样本处为1。
        (
            num_fg,
            gt_matched_classes,
            pred_ious_this_matching,
            matched_gt_inds,
            fg_mask,
        ) = self.dynamic_k_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask)

        gt_matched_classes.stop_gradient = True
        pred_ious_this_matching.stop_gradient = True
        matched_gt_inds.stop_gradient = True
        fg_mask.stop_gradient = True
        return (
            gt_matched_classes,
            fg_mask,
            pred_ious_this_matching,
            matched_gt_inds,
            num_fg,
        )

    def get_in_boxes_info(
        self,
        gt_bboxes_per_image,
        expanded_strides,
        x_shifts,
        y_shifts,
        total_num_anchors,
        num_gt,
    ):
        # (本操作不需要梯度)
        # 4-2-1-1.只根据每张图片的gt_bboxes_per_image shp=(M, 4)确定哪些作为 候选正样本。
        # fg_mask shp=(8400, ) 为1处表示是 候选正样本。假设有M个位置处为1。
        # is_in_boxes_and_center shp=(num_gt, M) M个候选正样本 与num_gt个gt的关系，是否在gt内部和中心。
        expanded_strides_per_image = expanded_strides[0]  # [1, 8400] -> [8400, ]   每个格子的格子边长。
        x_shifts_per_image = x_shifts[0] * expanded_strides_per_image  # [8400, ]   每个格子左上角的x坐标。单位是1像素。[0, 8, 16, ..., 544, 576, 608]
        y_shifts_per_image = y_shifts[0] * expanded_strides_per_image  # [8400, ]   每个格子左上角的y坐标。单位是1像素。[0, 0, 0, ...,  608, 608, 608]
        x_centers_per_image = (x_shifts_per_image + 0.5 * expanded_strides_per_image).unsqueeze(0)   # [1, n_anchor]   每个格子中心点的x坐标。单位是1像素。
        x_centers_per_image = L.expand(x_centers_per_image, [num_gt, 1])  # [num_gt, n_anchor]
        y_centers_per_image = (y_shifts_per_image + 0.5 * expanded_strides_per_image).unsqueeze(0)   # [1, n_anchor]   每个格子中心点的y坐标。单位是1像素。
        y_centers_per_image = L.expand(y_centers_per_image, [num_gt, 1])  # [num_gt, n_anchor]

        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0] - 0.5 * gt_bboxes_per_image[:, 2]).unsqueeze(1)   # [num_gt, 1]
        gt_bboxes_per_image_l = L.expand(gt_bboxes_per_image_l, [1, total_num_anchors])  # [num_gt, n_anchor]

        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0] + 0.5 * gt_bboxes_per_image[:, 2]).unsqueeze(1)   # [num_gt, 1]
        gt_bboxes_per_image_r = L.expand(gt_bboxes_per_image_r, [1, total_num_anchors])  # [num_gt, n_anchor]

        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1] - 0.5 * gt_bboxes_per_image[:, 3]).unsqueeze(1)   # [num_gt, 1]
        gt_bboxes_per_image_t = L.expand(gt_bboxes_per_image_t, [1, total_num_anchors])  # [num_gt, n_anchor]

        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1] + 0.5 * gt_bboxes_per_image[:, 3]).unsqueeze(1)   # [num_gt, 1]
        gt_bboxes_per_image_b = L.expand(gt_bboxes_per_image_b, [1, total_num_anchors])  # [num_gt, n_anchor]

        b_l = x_centers_per_image - gt_bboxes_per_image_l
        b_r = gt_bboxes_per_image_r - x_centers_per_image
        b_t = y_centers_per_image - gt_bboxes_per_image_t
        b_b = gt_bboxes_per_image_b - y_centers_per_image
        bbox_deltas = paddle.stack([b_l, b_t, b_r, b_b], 2)

        is_in_boxes = paddle.min(bbox_deltas, axis=-1) > 0
        is_in_boxes = paddle.cast(is_in_boxes, 'float32')
        is_in_boxes_all = paddle.sum(is_in_boxes, axis=0)
        is_in_boxes_all = paddle.cast(is_in_boxes_all > 0, 'float32')
        # in fixed center

        center_radius = 2.5

        gt_bboxes_per_image_l = L.expand(gt_bboxes_per_image[:, 0:1], [1, total_num_anchors]) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_r = L.expand(gt_bboxes_per_image[:, 0:1], [1, total_num_anchors]) + center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_t = L.expand(gt_bboxes_per_image[:, 1:2], [1, total_num_anchors]) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_b = L.expand(gt_bboxes_per_image[:, 1:2], [1, total_num_anchors]) + center_radius * expanded_strides_per_image.unsqueeze(0)

        c_l = x_centers_per_image - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image
        center_deltas = paddle.stack([c_l, c_t, c_r, c_b], 2)
        is_in_centers = paddle.min(center_deltas, axis=-1) > 0
        is_in_centers = paddle.cast(is_in_centers, 'float32')
        is_in_centers_all = paddle.sum(is_in_centers, axis=0)
        is_in_centers_all = paddle.cast(is_in_centers_all > 0, 'float32')

        # in boxes and in centers


        # 逻辑或运算
        is_in_boxes_anchor = paddle.cast(is_in_boxes_all + is_in_centers_all > 0, 'float32')


        index = L.where(is_in_boxes_anchor > 0)[:, 0]  # [M, ]

        # dic2 = np.load('datai1.npz')

        # is_in_boxes_all2 = dic2['is_in_boxes_all']
        # is_in_centers_all2 = dic2['is_in_centers_all']
        # bbbbbbbbb = is_in_boxes_all.numpy()
        # bbbbbbbbb = is_in_centers_all.numpy()
        # ddd = np.sum((is_in_centers_all2[0] - bbbbbbbbb) ** 2)
        # print('ddd=%.6f' % ddd)
        # print()

        is_in_boxes_t = L.transpose(is_in_boxes, [1, 0])  # [n_anchor, num_gt]
        cond1 = L.gather(is_in_boxes_t, index)            # [M, num_gt]
        cond1 = L.transpose(cond1, [1, 0])                # [num_gt, M]

        is_in_centers_t = L.transpose(is_in_centers, [1, 0])  # [n_anchor, num_gt]
        cond2 = L.gather(is_in_centers_t, index)              # [M, num_gt]
        cond2 = L.transpose(cond2, [1, 0])                    # [num_gt, M]

        # 逻辑与运算
        is_in_boxes_and_center = (cond1 + cond2)  # [num_gt, M]
        is_in_boxes_and_center = paddle.cast(is_in_boxes_and_center > 1.0, 'float32')  # [num_gt, M]

        is_in_boxes_anchor.stop_gradient = True
        is_in_boxes_and_center.stop_gradient = True
        return is_in_boxes_anchor, is_in_boxes_and_center

    def dynamic_k_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        # Dynamic K
        # ---------------------------------------------------------------
        # matching_matrix = paddle.zeros_like(cost)
        # 4-2-6.根据cost从 候选正样本 中 选出 最终正样本。
        # 4-2-6-1.根据cost从 候选正样本 中 选出 最终正样本。
        # cost.shape = [num_gt, M]  gt 和 候选正样本 两两之间的cost。
        matching_matrix = []
        zeros_matrix = paddle.zeros_like(cost[0])  # [M, ]
        ones_matrix = paddle.ones_like(cost[0])    # [M, ]

        ious_in_boxes_matrix = pair_wise_ious  # [num_gt, M]  gt 和 候选正样本 两两之间的iou。
        # 表示最多只抽10个候选正样本（如果候选正样本个数为0，比如图片没有gt时，n_candidate_k可能为0）。
        n_candidate_k = min(10, ious_in_boxes_matrix.shape[1])
        # [num_gt, n_candidate_k] 表示对于每个gt，选出n_candidate_k个与它iou最高的预测框。
        topk_ious, _ = paddle.topk(ious_in_boxes_matrix, n_candidate_k, axis=1)

        dic2 = np.load('topk_ious.npz')
        topk_ious2 = dic2['topk_ious']
        topk_ious3 = topk_ious.numpy()

        indexxx = L.where(fg_mask > 0)[:, 0]  # [M, ]
        indexx3 = indexxx.numpy()
        # 第一张图片
        if num_gt == 8:
            topk_ious4 = topk_ious2[0, :8, :]
            ddd = np.sum((topk_ious4 - topk_ious3) ** 2)
            print('ddd=%.6f' % ddd)
        # 第二张图片
        if num_gt == 16:
            topk_ious4 = topk_ious2[1, :, :]
            ddd = np.sum((topk_ious4 - topk_ious3) ** 2)
            print('ddd=%.6f' % ddd)
        # 经过校验，输出一样

        # [num_gt, ]  最匹配当前gt的前n_candidate_k个的预测框iou求和。
        dynamic_ks = topk_ious.sum(1)
        dynamic_ks = L.clip(dynamic_ks, 1.0, np.inf)  # dynamic_ks限制在区间[1.0, np.inf]内
        dynamic_ks = L.cast(dynamic_ks, 'int32')      # 取整。表示每个gt分配给了几个预测框。最少1个。

        for gt_idx in range(num_gt):
            # 对于每个gt，取前dynamic_ks[gt_idx]个最小的cost。拥有最小cost的预测框成为最终正样本。
            min_value, pos_idx = paddle.topk(
                cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False
            )
            # min_value, pos_idx = paddle.topk(
            #     cost[gt_idx], k=10, largest=False
            # )
            # matching_matrix[gt_idx][pos_idx[0]] = 1.0
            matching_matrix_ = paddle.where(cost[gt_idx] <= min_value[-1], ones_matrix, zeros_matrix).unsqueeze(0)
            matching_matrix.append(matching_matrix_)
        # [num_gt, M]
        matching_matrix = paddle.concat(matching_matrix, 0)

        # dic2 = np.load('matching_matrix.npz')
        # matching_matrix2 = dic2['matching_matrix']
        # matching_matrix3 = matching_matrix.numpy()
        #
        # indexxx = L.where(fg_mask > 0)[:, 0]  # [M, ]
        # indexx3 = indexxx.numpy()
        # # 第一张图片
        # if num_gt == 8:
        #     matching_matrix4 = matching_matrix2[0, :8, :]
        #     matching_matrix4 = matching_matrix4.transpose(1, 0)
        #     matching_matrix4 = matching_matrix4[indexx3]
        #     matching_matrix4 = matching_matrix4.transpose(1, 0)
        #     ddd = np.sum((matching_matrix4 - matching_matrix3) ** 2)
        #     print('ddd=%.6f' % ddd)
        # # 第二张图片
        # if num_gt == 16:
        #     matching_matrix4 = matching_matrix2[1, :, :]
        #     matching_matrix4 = matching_matrix4.transpose(1, 0)
        #     matching_matrix4 = matching_matrix4[indexx3]
        #     matching_matrix4 = matching_matrix4.transpose(1, 0)
        #     ddd = np.sum((matching_matrix4 - matching_matrix3) ** 2)
        #     print('ddd=%.6f' % ddd)
        # # 经过校验，输出一样

        del topk_ious, dynamic_ks, pos_idx

        # [M, ]  每个anchor匹配到了几个gt？有可能匹配到2个或2个以上。
        anchor_matching_gt = matching_matrix.sum(0)  # [M, ]
        # 如果有anchor（花心大萝卜）匹配到了1个以上的gt时，做特殊处理。
        if paddle.cast(anchor_matching_gt > 1, 'float32').sum() > 0:
            # 找到 花心大萝卜 的下标（这是在M个候选正样本中的下标）。假设有R个花心大萝卜。
            index4 = L.where(anchor_matching_gt > 1)[:, 0]  # [R, ]
            cost_t = L.transpose(cost, [1, 0])  # [M, num_gt]
            cost2 = L.gather(cost_t, index4)    # [R, num_gt]  R个花心大萝卜 与 gt 两两之间的cost。
            cost2 = L.transpose(cost2, [1, 0])  # [num_gt, R]  gt 与 R个花心大萝卜 两两之间的cost。
            cost_argmin = paddle.argmin(cost2, axis=0)  # [R, ]  为 每个花心大萝卜 找到 与其cost最小的gt 的下标

            # 先把 花心大萝卜 匹配到的所有gt处 置为0
            # matching_matrix[:, anchor_matching_gt > 1] *= 0.0
            cond1 = (anchor_matching_gt > 1).unsqueeze(0)  # [1, M] 是否是 花心大萝卜
            cond1 = L.expand(cond1, [matching_matrix.shape[0], 1])  # [num_gt, M] 是否是 花心大萝卜
            matching_matrix = paddle.where(cond1, matching_matrix*0.0, matching_matrix)

            # 再把 花心大萝卜 与gt里 cost最小处置为1，这样就实现了一个anchor最多负责预测一个gt。
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0


        dic2 = np.load('matching_matrix2.npz')
        matching_matrix2 = dic2['matching_matrix']
        matching_matrix3 = matching_matrix.numpy()

        indexxx = L.where(fg_mask > 0)[:, 0]  # [M, ]
        indexx3 = indexxx.numpy()
        # 第一张图片
        if num_gt == 8:
            matching_matrix4 = matching_matrix2[0, :8, :]
            matching_matrix4 = matching_matrix4.transpose(1, 0)
            matching_matrix4 = matching_matrix4[indexx3]
            matching_matrix4 = matching_matrix4.transpose(1, 0)
            ddd = np.sum((matching_matrix4 - matching_matrix3) ** 2)
            print('ddd=%.6f' % ddd)
        # 第二张图片
        if num_gt == 16:
            matching_matrix4 = matching_matrix2[1, :, :]
            matching_matrix4 = matching_matrix4.transpose(1, 0)
            matching_matrix4 = matching_matrix4[indexx3]
            matching_matrix4 = matching_matrix4.transpose(1, 0)
            ddd = np.sum((matching_matrix4 - matching_matrix3) ** 2)
            print('ddd=%.6f' % ddd)
        # 经过校验，输出一样

        # matching_matrix   [num_gt, M]
        # fg_mask shp=(8400, ) 为1处表示是 候选正样本。假设有M个位置处为1。
        # [M, ]  是否是前景（正样本）
        fg_mask_inboxes = matching_matrix.sum(0) > 0.0    # [M, ]
        fg_mask_inboxes = paddle.cast(fg_mask_inboxes, 'float32')   # [M, ]
        num_fg = fg_mask_inboxes.sum().item()  # 前景个数（num_fg>=num_gt）


        # 翻译这句pytorch代码
        # fg_mask[fg_mask.clone()] = fg_mask_inboxes
        # fg_mask_clone = fg_mask.clone()
        # index = L.where(fg_mask_clone > 0)[:, 0]
        # index2 = L.where(fg_mask_inboxes > 0)[:, 0]
        # pos_index = L.gather(index, index2)
        # fg_mask = paddle.zeros_like(fg_mask_clone)
        # for iiiiii in pos_index:
        #     fg_mask[iiiiii] = 1.0
        # fg_mask.stop_gradient = True  # 没有可训练参数来自fg_mask，可以关停梯度？

        # fg_mask shp=(8400, ) 为1处表示是 候选正样本。假设有M个位置处为1。
        fg_mask = fg_mask.astype(paddle.bool)
        fg_mask_inboxes = fg_mask_inboxes.astype(paddle.bool)
        fg_mask[fg_mask.clone()] = fg_mask_inboxes
        fg_mask = fg_mask.astype(paddle.float32)
        fg_mask_inboxes = fg_mask_inboxes.astype(paddle.float32)


        dic2 = np.load('fg_mask.npz')
        fg_mask2 = dic2['fg_mask']
        fg_mask3 = fg_mask.numpy()

        indexxx = L.where(fg_mask > 0)[:, 0]  # [M, ]
        indexx3 = indexxx.numpy()
        # 第一张图片
        if num_gt == 8:
            fg_mask4 = fg_mask2[0, :]
            ddd = np.sum((fg_mask4 - fg_mask3) ** 2)
            print('ddd=%.6f' % ddd)
        # 第二张图片
        if num_gt == 16:
            fg_mask4 = fg_mask2[1, :]
            ddd = np.sum((fg_mask4 - fg_mask3) ** 2)
            print('ddd=%.6f' % ddd)
        # 经过校验，输出一样

        # 翻译这句pytorch代码。即从 候选正样本 中 设置 最终正样本处为1，其余处为0。
        # fg_mask[fg_mask.clone()] = fg_mask_inboxes
        # fg_mask_clone = fg_mask.clone()
        # index = L.where(fg_mask_clone > 0)[:, 0]
        # index2 = L.where(fg_mask_inboxes > 0)[:, 0]
        # pos_index = L.gather(index, index2)
        # pos_index2 = np.copy(pos_index.numpy())
        # fg_mask222 = np.zeros(fg_mask_clone.shape)
        # for iiiiii in pos_index2:
        #     fg_mask222[iiiiii] = 1.0
        # fg_mask222 = paddle.to_tensor(fg_mask222)
        # fg_mask222.stop_gradient = True  # 没有可训练参数来自fg_mask，可以关停梯度？

        # 确定最终正样本需要学习的类别id。假设有T个最终正样本。
        index3 = L.where(fg_mask_inboxes > 0)[:, 0]  # [T, ]
        matching_matrix_t = L.transpose(matching_matrix, [1, 0])  # [M, num_gt]
        matched_gt_inds = L.gather(matching_matrix_t, index3)     # [T, num_gt]
        matched_gt_inds = L.transpose(matched_gt_inds, [1, 0])    # [num_gt, T]
        matched_gt_inds = matched_gt_inds.argmax(0)               # [T, ]  最终正样本是匹配到了第几个gt
        # 最终正样本需要学习的类别id
        gt_matched_classes = L.gather(gt_classes, matched_gt_inds)  # [T, ]
        # [num_gt, M]    gt 和 候选正样本 两两之间的iou。由于一个anchor最多匹配一个gt，可以理解为M个anchor和最匹配的gt的iou。
        ious = (matching_matrix * pair_wise_ious)
        # [M, ]    M个anchor和最匹配的gt的iou。
        ious = ious.sum(0)
        # [T, ]    T个最终正样本和匹配的gt的iou。
        pred_ious_this_matching = L.gather(ious, index3)


        dic2 = np.load('dyk.npz')
        gt_matched_classes2 = dic2['matched_gt_inds']
        gt_matched_classes3 = matched_gt_inds.numpy()
        # 第一张图片
        if num_gt == 8:
            print('ddd=%.6f' % ddd)
        # 第二张图片
        if num_gt == 16:
            print('ddd=%.6f' % ddd)
        # 经过校验，输出一样

        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds, fg_mask

