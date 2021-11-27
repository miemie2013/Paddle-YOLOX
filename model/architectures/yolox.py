#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2021-10-12 11:23:07
#   Description : yolox
#
# ================================================================
import paddle


class YOLOX(paddle.nn.Layer):
    def __init__(self, backbone, fpn, head):
        super(YOLOX, self).__init__()
        self.backbone = backbone
        self.fpn = fpn
        self.head = head

    def forward(self, x, im_info):
        body_feats = self.backbone(x)
        fpn_outs = self.fpn(body_feats)
        out = self.head.get_prediction(fpn_outs, im_info)
        return out

    def train_model(self, x, gt_class_bbox):
        body_feats = self.backbone(x)
        fpn_outs = self.fpn(body_feats)
        out = self.head.get_loss(fpn_outs, gt_class_bbox)
        return out



