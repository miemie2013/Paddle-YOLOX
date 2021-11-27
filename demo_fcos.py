#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-10-23 09:13:23
#   Description : paddle2.0_ppyolo
#
# ================================================================
from collections import deque
import datetime
import cv2
import os
import time
import threading
import argparse
import textwrap
import paddle

from config import *
from model.decoders.decode_fcos import *
from model.architectures.fcos import *
from tools.argparser import *
from tools.cocotools import get_classes

import logging
FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


def read_test_data(path_dir,
                   _decode,
                   test_dic):
    for k, filename in enumerate(path_dir):
        key_list = list(test_dic.keys())
        key_len = len(key_list)
        while key_len >= 3:
            time.sleep(0.01)
            key_list = list(test_dic.keys())
            key_len = len(key_list)

        image = cv2.imread('images/test/' + filename)
        pimage, im_size = _decode.process_image(np.copy(image))
        dic = {}
        dic['image'] = image
        dic['pimage'] = pimage
        dic['im_size'] = im_size
        test_dic['%.8d' % k] = dic

def save_img(filename, image):
    cv2.imwrite('images/res/' + filename, image)

if __name__ == '__main__':
    parser = FCOSArgParser()
    use_gpu = parser.get_use_gpu()
    cfg = parser.get_cfg()
    # cfg = FCOS_RT_R50_FPN_4x_Config()
    print(paddle.__version__)
    paddle.disable_static()   # 开启动态图
    gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
    place = paddle.CUDAPlace(gpu_id) if use_gpu else paddle.CPUPlace()

    # 读取的模型
    model_path = cfg.test_cfg['model_path']

    # 是否给图片画框。
    draw_image = cfg.test_cfg['draw_image']
    draw_thresh = cfg.test_cfg['draw_thresh']

    # 打印，确认一下使用的配置
    print('\n=============== config message ===============')
    print('config file: %s' % str(type(cfg)))
    print('model_path: %s' % model_path)
    print('target_size: %d' % cfg.test_cfg['target_size'])
    print('use_gpu: %s' % str(use_gpu))
    print()

    class_names = get_classes(cfg.classes_path)
    num_classes = len(class_names)


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

    _decode = Decode_FCOS(model, class_names, place, cfg, for_test=True)

    if not os.path.exists('images/res/'): os.mkdir('images/res/')
    path_dir = os.listdir('images/test')

    # 读数据的线程
    test_dic = {}
    thr = threading.Thread(target=read_test_data,
                           args=(path_dir,
                                 _decode,
                                 test_dic))
    thr.start()

    key_list = list(test_dic.keys())
    key_len = len(key_list)
    while key_len == 0:
        time.sleep(0.01)
        key_list = list(test_dic.keys())
        key_len = len(key_list)
    dic = test_dic['%.8d' % 0]
    image = dic['image']
    pimage = dic['pimage']
    im_size = dic['im_size']


    # warm up
    if use_gpu:
        for k in range(10):
            image, boxes, scores, classes = _decode.detect_image(image, pimage, im_size, draw_image=False)


    time_stat = deque(maxlen=20)
    start_time = time.time()
    end_time = time.time()
    num_imgs = len(path_dir)
    start = time.time()
    for k, filename in enumerate(path_dir):
        key_list = list(test_dic.keys())
        key_len = len(key_list)
        while key_len == 0:
            time.sleep(0.01)
            key_list = list(test_dic.keys())
            key_len = len(key_list)
        dic = test_dic.pop('%.8d' % k)
        image = dic['image']
        pimage = dic['pimage']
        im_size = dic['im_size']

        image, boxes, scores, classes = _decode.detect_image(image, pimage, im_size, draw_image, draw_thresh)

        # 估计剩余时间
        start_time = end_time
        end_time = time.time()
        time_stat.append(end_time - start_time)
        time_cost = np.mean(time_stat)
        eta_sec = (num_imgs - k) * time_cost
        eta = str(datetime.timedelta(seconds=int(eta_sec)))

        logger.info('Infer iter {}, num_imgs={}, eta={}.'.format(k, num_imgs, eta))
        if draw_image:
            t2 = threading.Thread(target=save_img, args=(filename, image))
            t2.start()
            logger.info("Detection bbox results save in images/res/{}".format(filename))
    cost = time.time() - start
    logger.info('total time: {0:.6f}s'.format(cost))
    logger.info('Speed: %.6fs per image,  %.1f FPS.'%((cost / num_imgs), (num_imgs / cost)))


