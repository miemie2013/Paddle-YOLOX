#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2021-10-12 11:23:07
#   Description : yolox
#
# ================================================================
from config import *
from model.architectures.yolox import *
import paddle.fluid as fluid
import paddle
import torch
import os

print(paddle.__version__)
paddle.disable_static()
# 开启动态图



use_gpu = True

gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
place = paddle.CUDAPlace(gpu_id) if use_gpu else paddle.CPUPlace()



cfg = YOLOX_L_Config()
# 该模型是COCO数据集上训练好的，所以强制改类别数为80
cfg.num_classes = 80
cfg.head['num_classes'] = cfg.num_classes
model_path = 'yolox_l.pth'



def load_weights(path):
    state_dict = torch.load(path, map_location=torch.device('cpu'))
    return state_dict

state_dict = load_weights(model_path)
state_dict = state_dict['model']
print('============================================================')

backbone_dic = {}
scale_on_reg_dic = {}
fpn_dic = {}
head_dic = {}
others = {}
for key, value in state_dict.items():
    if 'tracked' in key:
        continue
    if 'backbone' in key:
        name2 = key[9:]
        if 'backbone' in name2:
            backbone_dic[name2] = value.data.numpy()
        else:
            fpn_dic[name2] = value.data.numpy()
    elif 'fpn' in key:
        fpn_dic[key] = value.data.numpy()
    elif 'head' in key:
        head_dic[key] = value.data.numpy()
    else:
        others[key] = value.data.numpy()

print()




# 创建模型
Backbone = select_backbone(cfg.backbone_type)
backbone = Backbone(**cfg.backbone)
FPN = select_fpn(cfg.fpn_type)
fpn = FPN(**cfg.fpn)
Head = select_head(cfg.head_type)
head = Head(iou_loss=None, nms_cfg=cfg.nms_cfg, **cfg.head)
model = YOLOX(backbone, fpn, head)

model.eval()  # 必须调用model.eval()来设置dropout和batch normalization layers在运行推理前，切换到评估模式。
param_state_dict = model.state_dict()

print('\nCopying...')



def copy(name, w):
    value2 = paddle.to_tensor(w, place=place)
    value = param_state_dict[name]
    value = value * 0 + value2
    param_state_dict[name] = value

def copy_conv_bn(conv_unit_name, w, scale, offset, m, v):
    copy(conv_unit_name + '.conv.weight', w)
    copy(conv_unit_name + '.bn.weight', scale)
    copy(conv_unit_name + '.bn.bias', offset)
    copy(conv_unit_name + '.bn._mean', m)
    copy(conv_unit_name + '.bn._variance', v)

def copy_conv_af(conv_unit_name, w, scale, offset):
    copy(conv_unit_name + '.conv.weight', w)
    copy(conv_unit_name + '.scale', scale)
    copy(conv_unit_name + '.offset', offset)


def copy_conv(conv_name, w, b):
    copy(conv_name + '.weight', w)
    copy(conv_name + '.bias', b)


def copy_conv_gn(conv_unit_name, w, b, scale, offset):
    copy(conv_unit_name + '.conv.weight', w)
    copy(conv_unit_name + '.conv.bias', b)
    copy(conv_unit_name + '.gn.weight', scale)
    copy(conv_unit_name + '.gn.bias', offset)



# 骨干网络
for key in backbone_dic.keys():
    name2 = key
    w = backbone_dic[key]
    if 'running_mean' in key:
        name2 = key.replace('running_mean', '_mean')
    if 'running_var' in key:
        name2 = key.replace('running_var', '_variance')
    # print(key)
    copy(name2, w)

# fpn
for key in fpn_dic.keys():
    name2 = 'fpn.'+key
    w = fpn_dic[key]
    if 'running_mean' in key:
        name2 = name2.replace('running_mean', '_mean')
    if 'running_var' in key:
        name2 = name2.replace('running_var', '_variance')
    # print(key)
    copy(name2, w)

# head
for key in head_dic.keys():
    name2 = key
    w = head_dic[key]
    if 'running_mean' in key:
        name2 = name2.replace('running_mean', '_mean')
    if 'running_var' in key:
        name2 = name2.replace('running_var', '_variance')
    # print(key)
    copy(name2, w)



save_name = 'dygraph_yolox_l.pdparams'
paddle.save(param_state_dict, save_name)
print('\nDone.')







