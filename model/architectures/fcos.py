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


class FCOS(paddle.nn.Layer):
    def __init__(self, backbone, fpn, head):
        super(FCOS, self).__init__()
        self.backbone = backbone
        self.fpn = fpn
        self.head = head

    def forward(self, x, im_info, get_heatmap=False):
        body_feats = self.backbone(x)
        body_feats, spatial_scale = self.fpn(body_feats)
        if get_heatmap:
            out = self.head.get_heatmap(body_feats, im_info)
        else:
            out = self.head.get_prediction(body_feats, im_info)
        return out

    def train_model(self, x, tag_labels, tag_bboxes, tag_centerness):
        body_feats = self.backbone(x)
        body_feats, spatial_scale = self.fpn(body_feats)
        out = self.head.get_loss(body_feats, tag_labels, tag_bboxes, tag_centerness)
        return out



