#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-10-23 09:13:23
#   Description : paddle2.0_ppyolo
#
# ================================================================
import json
import os
from tools.cocotools_fcos import eval
from model.decoders.decode_fcos import *
from model.architectures.fcos import *
from tools.argparser import *

import logging
FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


if __name__ == '__main__':
    parser = FCOSArgParser()
    use_gpu = parser.get_use_gpu()
    cfg = parser.get_cfg()
    print(paddle.__version__)
    paddle.disable_static()   # 开启动态图
    gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
    place = paddle.CUDAPlace(gpu_id) if use_gpu else paddle.CPUPlace()

    # 读取的模型
    model_path = cfg.eval_cfg['model_path']

    # 是否给图片画框。
    draw_image = cfg.eval_cfg['draw_image']
    draw_thresh = cfg.eval_cfg['draw_thresh']

    # 验证时的批大小
    eval_batch_size = cfg.eval_cfg['eval_batch_size']

    # 打印，确认一下使用的配置
    print('\n=============== config message ===============')
    print('config file: %s' % str(type(cfg)))
    print('model_path: %s' % model_path)
    print('target_size: %d' % cfg.eval_cfg['target_size'])
    print('use_gpu: %s' % str(use_gpu))
    print()

    # 验证集图片的相对路径
    eval_pre_path = cfg.val_pre_path
    anno_file = cfg.val_path
    from pycocotools.coco import COCO
    val_dataset = COCO(anno_file)
    val_img_ids = val_dataset.getImgIds()
    images = []   # 只跑有gt的图片，跟随PaddleDetection
    for img_id in val_img_ids:
        ins_anno_ids = val_dataset.getAnnIds(imgIds=img_id, iscrowd=False)   # 读取这张图片所有标注anno的id
        if len(ins_anno_ids) == 0:
            continue
        img_anno = val_dataset.loadImgs(img_id)[0]
        images.append(img_anno)

    # 种类id
    _catid2clsid = {}
    _clsid2catid = {}
    _clsid2cname = {}
    with open(cfg.val_path, 'r', encoding='utf-8') as f2:
        dataset_text = ''
        for line in f2:
            line = line.strip()
            dataset_text += line
        eval_dataset = json.loads(dataset_text)
        categories = eval_dataset['categories']
        for clsid, cate_dic in enumerate(categories):
            catid = cate_dic['id']
            cname = cate_dic['name']
            _catid2clsid[catid] = clsid
            _clsid2catid[clsid] = catid
            _clsid2cname[clsid] = cname
    class_names = []
    num_classes = len(_clsid2cname.keys())
    for clsid in range(num_classes):
        class_names.append(_clsid2cname[clsid])


    # 创建模型
    Backbone = select_backbone(cfg.backbone_type)
    backbone = Backbone(**cfg.backbone)
    FPN = select_fpn(cfg.fpn_type)
    fpn = FPN(**cfg.fpn)
    Head = select_head(cfg.head_type)
    head = Head(fcos_loss=None, nms_cfg=cfg.nms_cfg, **cfg.head)
    model = FCOS(backbone, fpn, head)

    param_state_dict = paddle.load(model_path)
    model.set_state_dict(param_state_dict)
    model.eval()  # 必须调用model.eval()来设置dropout和batch normalization layers在运行推理前，切换到评估模式。
    head.set_dropblock(is_test=True)

    _decode = Decode_FCOS(model, class_names, place, cfg, for_test=False)
    box_ap = eval(_decode, images, eval_pre_path, anno_file, eval_batch_size, _clsid2catid, draw_image, draw_thresh)

