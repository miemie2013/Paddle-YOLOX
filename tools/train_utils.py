#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date:
#   Description :
#
# ================================================================
import os
import copy
import paddle
import numpy as np
from collections import OrderedDict

from tools.transform import *




def load_weights(model, model_path):
    _state_dict = model.state_dict()
    pretrained_dict = paddle.load(model_path)
    new_state_dict = OrderedDict()
    for k, v in pretrained_dict.items():
        if k in _state_dict:
            shape_1 = _state_dict[k].shape
            shape_2 = pretrained_dict[k].shape
            shape_2 = list(shape_2)
            if shape_1 == shape_2:
                new_state_dict[k] = v
            else:
                print('shape mismatch in \'%s\'. model_shape=%s, while pretrained_shape=%s.' % (k, shape_1, shape_2))
    _state_dict.update(new_state_dict)
    model.set_state_dict(_state_dict)

def clear_model(save_dir):
    path_dir = os.listdir(save_dir)
    it_ids = []
    for name in path_dir:
        sss = name.split('.')
        if sss[0] == '':
            continue
        if sss[0] == 'best_model':   # 不会删除最优模型
            it_id = 9999999999
        else:
            it_id = int(sss[0])
        it_ids.append(it_id)
    if len(it_ids) >= 11 * 1:
        it_id = min(it_ids)
        pdparams_path = '%s/%d.pdparams' % (save_dir, it_id)
        if os.path.exists(pdparams_path):
            os.remove(pdparams_path)

def calc_lr(iter_id, train_steps, max_iters, cfg):
    base_lr = cfg.learningRate['base_lr']
    piecewiseDecay = cfg.learningRate.get('PiecewiseDecay', None)
    cosineDecay = cfg.learningRate.get('CosineDecay', None)
    linearWarmup = cfg.learningRate.get('LinearWarmup', None)

    cur_lr = base_lr

    linearWarmup_end_iter_id = 0
    skip = False
    if linearWarmup is not None:
        start_factor = linearWarmup['start_factor']
        steps = linearWarmup.get('steps', -1)
        epochs = linearWarmup.get('epochs', -1)

        if steps <= 0 and epochs <= 0:
            raise ValueError("\'steps\' or \'epochs\' should be positive in {}.learningRate[\'LinearWarmup\']".format(cfg))
        if steps > 0 and epochs > 0:
            steps = -1   # steps和epochs都设置为正整数时，优先选择epochs
        if steps <= 0 and epochs > 0:
            steps = epochs * train_steps

        linearWarmup_end_iter_id = steps
        if iter_id < steps:
            k = (1.0 - start_factor) / steps
            factor = start_factor + k * iter_id
            cur_lr = base_lr * factor
            skip = True

    if skip:
        return cur_lr

    if piecewiseDecay is not None:
        gamma = piecewiseDecay['gamma']
        milestones = piecewiseDecay.get('milestones', None)
        milestones_epoch = piecewiseDecay.get('milestones_epoch', None)

        if milestones is not None:
            pass
        elif milestones_epoch is not None:
            milestones = [f * train_steps for f in milestones_epoch]
        n = len(milestones)
        cur_lr = base_lr
        for i in range(n, 0, -1):
            if iter_id >= milestones[i - 1]:
                cur_lr = base_lr * gamma ** i
                break

    if cosineDecay is not None:
        start_iter_id = linearWarmup_end_iter_id
        dx = (iter_id - start_iter_id) / (max_iters - start_iter_id) * math.pi
        cur_lr = base_lr * (1.0 + np.cos(dx)) * 0.5
    return cur_lr

def write(filename, logstats):
    with open(filename, 'a', encoding='utf-8') as f:
        f.writelines(logstats + '\n')
        f.close


def get_transforms(cfg):
    # sample_transforms
    sample_transforms = []
    for preprocess_name in cfg.sample_transforms_seq:
        if preprocess_name == 'decodeImage':
            preprocess = DecodeImage(**cfg.decodeImage)   # 对图片解码。最开始的一步。
        elif preprocess_name == 'mixupImage':
            preprocess = MixupImage(**cfg.mixupImage)      # mixup增强
        elif preprocess_name == 'cutmixImage':
            preprocess = CutmixImage(**cfg.cutmixImage)    # cutmix增强
        elif preprocess_name == 'mosaicImage':
            preprocess = MosaicImage(**cfg.mosaicImage)    # mosaic增强
        elif preprocess_name == 'yOLOXMosaicImage':
            preprocess = YOLOXMosaicImage(**cfg.yOLOXMosaicImage)  # YOLOX mosaic增强
        elif preprocess_name == 'colorDistort':
            preprocess = ColorDistort(**cfg.colorDistort)  # 颜色扰动
        elif preprocess_name == 'randomExpand':
            preprocess = RandomExpand(**cfg.randomExpand)  # 随机填充
        elif preprocess_name == 'randomCrop':
            preprocess = RandomCrop(**cfg.randomCrop)        # 随机裁剪
        elif preprocess_name == 'gridMaskOp':
            preprocess = GridMaskOp(**cfg.gridMaskOp)        # GridMaskOp
        elif preprocess_name == 'poly2Mask':
            preprocess = Poly2Mask(**cfg.poly2Mask)         # 多边形变掩码
        elif preprocess_name == 'resizeImage':
            preprocess = ResizeImage(**cfg.resizeImage)        # 多尺度训练
        elif preprocess_name == 'yOLOXResizeImage':
            preprocess = YOLOXResizeImage(**cfg.yOLOXResizeImage)  # YOLOX多尺度训练
        elif preprocess_name == 'randomFlipImage':
            preprocess = RandomFlipImage(**cfg.randomFlipImage)  # 随机翻转
        elif preprocess_name == 'normalizeImage':
            preprocess = NormalizeImage(**cfg.normalizeImage)     # 图片归一化。
        elif preprocess_name == 'normalizeBox':
            preprocess = NormalizeBox(**cfg.normalizeBox)        # 将物体的左上角坐标、右下角坐标中的横坐标/图片宽、纵坐标/图片高 以归一化坐标。
        elif preprocess_name == 'padBox':
            preprocess = PadBox(**cfg.padBox)         # 如果gt_bboxes的数量少于num_max_boxes，那么填充坐标是0的bboxes以凑够num_max_boxes。
        elif preprocess_name == 'bboxXYXY2XYWH':
            preprocess = BboxXYXY2XYWH(**cfg.bboxXYXY2XYWH)     # sample['gt_bbox']被改写为cx_cy_w_h格式。
        elif preprocess_name == 'permute':
            preprocess = Permute(**cfg.permute)    # 图片从HWC格式变成CHW格式
        sample_transforms.append(preprocess)
    # batch_transforms
    batch_transforms = []
    for preprocess_name in cfg.batch_transforms_seq:
        if preprocess_name == 'randomShape':
            preprocess = RandomShapeSingle(random_inter=cfg.randomShape['random_inter'])     # 多尺度训练。随机选一个尺度。也随机选一种插值方式。
        elif preprocess_name == 'normalizeImage':
            preprocess = NormalizeImage(**cfg.normalizeImage)     # 图片归一化。先除以255归一化，再减均值除以标准差
        elif preprocess_name == 'permute':
            preprocess = Permute(**cfg.permute)    # 图片从HWC格式变成CHW格式
        elif preprocess_name == 'squareImage':
            preprocess = SquareImage(**cfg.squareImage)    # 图片变正方形。
        elif preprocess_name == 'gt2YoloTarget':
            preprocess = Gt2YoloTargetSingle(**cfg.gt2YoloTarget)   # 填写target张量。
        elif preprocess_name == 'padBatchSingle':
            use_padded_im_info = cfg.padBatchSingle['use_padded_im_info'] if 'use_padded_im_info' in cfg.padBatchSingle else True
            preprocess = PadBatchSingle(use_padded_im_info=use_padded_im_info)   # 填充黑边。使这一批图片有相同的大小。
        elif preprocess_name == 'padBatch':
            preprocess = PadBatch(**cfg.padBatch)                         # 填充黑边。使这一批图片有相同的大小。
        elif preprocess_name == 'gt2FCOSTargetSingle':
            preprocess = Gt2FCOSTargetSingle(**cfg.gt2FCOSTargetSingle)   # 填写target张量。
        elif preprocess_name == 'gt2Solov2Target':
            preprocess = Gt2Solov2Target(**cfg.gt2Solov2Target)     # 填写target张量。
        elif preprocess_name == 'gt2RepPointsTargetSingle':
            preprocess = Gt2RepPointsTargetSingle(**cfg.gt2RepPointsTargetSingle)     # 填写target张量。
        batch_transforms.append(preprocess)
    return sample_transforms, batch_transforms








