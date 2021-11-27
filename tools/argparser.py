#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date:
#   Description :
#
# ================================================================
import argparse
import textwrap
from config import *


class YOLOArgParser(object):
    def __init__(self):
        parser = argparse.ArgumentParser(description='Script', formatter_class=argparse.RawTextHelpFormatter)
        parser.add_argument('--use_gpu', type=bool, default=True, help='whether to use gpu. True or False')
        parser.add_argument('-c', '--config', type=int, default=0,
                            choices=[0, 1, 2, 3, 4, 5, 6],
                            help=textwrap.dedent('''\
                            select one of these config files:
                            0 -- ppyolo_2x.py
                            1 -- yolov4_2x.py
                            2 -- ppyolo_r18vd.py
                            3 -- ppyolo_mobilenet_v3_large.py
                            4 -- ppyolo_mobilenet_v3_small.py
                            5 -- ppyolo_mdf_2x.py
                            6 -- ppyolo_large_2x.py'''))
        self.args = parser.parse_args()
        self.config_file = self.args.config
        self.use_gpu = self.args.use_gpu

    def get_use_gpu(self):
        return self.use_gpu

    def get_cfg(self):
        config_file = self.config_file
        cfg = None
        if config_file == 0:
            cfg = PPYOLO_2x_Config()
        elif config_file == 1:
            cfg = YOLOv4_2x_Config()
        elif config_file == 2:
            cfg = PPYOLO_r18vd_Config()
        elif config_file == 3:
            cfg = PPYOLO_mobilenet_v3_large_Config()
        elif config_file == 4:
            cfg = PPYOLO_mobilenet_v3_large_Config()
        elif config_file == 5:
            cfg = PPYOLO_mdf_2x_Config()
        elif config_file == 6:
            cfg = PPYOLO_large_2x_Config()
        return cfg


class YOLOXArgParser(object):
    def __init__(self):
        parser = argparse.ArgumentParser(description='Script', formatter_class=argparse.RawTextHelpFormatter)
        parser.add_argument('--use_gpu', type=bool, default=True, help='whether to use gpu. True or False')
        parser.add_argument('-c', '--config', type=int, default=0,
                            choices=[0, 1, 2, 3, 4, 5, 6],
                            help=textwrap.dedent('''\
                            select one of these config files:
                            0 -- yolox_s.py
                            1 -- yolox_m.py
                            2 -- yolox_l.py
                            3 -- yolox_x.py'''))
        self.args = parser.parse_args()
        self.config_file = self.args.config
        self.use_gpu = self.args.use_gpu

    def get_use_gpu(self):
        return self.use_gpu

    def get_cfg(self):
        config_file = self.config_file
        cfg = None
        if config_file == 0:
            cfg = YOLOX_S_Config()
        elif config_file == 1:
            cfg = YOLOX_M_Config()
        elif config_file == 2:
            cfg = YOLOX_L_Config()
        elif config_file == 3:
            cfg = YOLOX_X_Config()
        elif config_file == 4:
            cfg = YOLOX_S_Config()
        elif config_file == 5:
            cfg = YOLOX_S_Config()
        elif config_file == 6:
            cfg = YOLOX_S_Config()
        return cfg


class FCOSArgParser(object):
    def __init__(self):
        parser = argparse.ArgumentParser(description='Script', formatter_class=argparse.RawTextHelpFormatter)
        parser.add_argument('--use_gpu', type=bool, default=True, help='whether to use gpu. True or False')
        parser.add_argument('-c', '--config', type=int, default=0,
                            choices=[0, 1, 2, 3, 4],
                            help=textwrap.dedent('''\
                            select one of these config files:
                            0 -- fcos_r50_fpn_multiscale_2x.py
                            1 -- fcos_rt_r50_fpn_4x.py
                            2 -- fcos_rt_dla34_fpn_4x.py
                            3 -- fcos_rt_r50vd_fpn_dcn_2x.py
                            4 -- fcos_dcn_r50_fpn_1x.py'''))
        self.args = parser.parse_args()
        self.config_file = self.args.config
        self.use_gpu = self.args.use_gpu

    def get_use_gpu(self):
        return self.use_gpu

    def get_cfg(self):
        config_file = self.config_file
        cfg = None
        if config_file == 0:
            cfg = FCOS_R50_FPN_Multiscale_2x_Config()
        elif config_file == 1:
            cfg = FCOS_RT_R50_FPN_4x_Config()
        elif config_file == 2:
            cfg = FCOS_RT_DLA34_FPN_4x_Config()
        elif config_file == 3:
            cfg = FCOS_RT_R50VD_FPN_DCN_2x_Config()
        elif config_file == 4:
            cfg = FCOS_DCN_R50_FPN_1x_Config()
        return cfg


class SOLOArgParser(object):
    def __init__(self):
        parser = argparse.ArgumentParser(description='Script', formatter_class=argparse.RawTextHelpFormatter)
        parser.add_argument('--use_gpu', type=bool, default=True, help='whether to use gpu. True or False')
        parser.add_argument('-c', '--config', type=int, default=0,
                            choices=[0, 1, 2, 3, 4],
                            help=textwrap.dedent('''\
                            select one of these config files:
                            0 -- solov2_r50_fpn_8gpu_3x.py
                            1 -- solov2_light_448_r50_fpn_8gpu_3x.py
                            2 -- solov2_light_r50_vd_fpn_dcn_512_3x.py'''))
        self.args = parser.parse_args()
        self.config_file = self.args.config
        self.use_gpu = self.args.use_gpu

    def get_use_gpu(self):
        return self.use_gpu

    def get_cfg(self):
        config_file = self.config_file
        cfg = None
        if config_file == 0:
            cfg = SOLOv2_r50_fpn_8gpu_3x_Config()
        elif config_file == 1:
            cfg = SOLOv2_light_448_r50_fpn_8gpu_3x_Config()
        elif config_file == 2:
            cfg = SOLOv2_light_r50_vd_fpn_dcn_512_3x_Config()
        return cfg


class RepPointsArgParser(object):
    def __init__(self):
        parser = argparse.ArgumentParser(description='Script', formatter_class=argparse.RawTextHelpFormatter)
        parser.add_argument('--use_gpu', type=bool, default=True, help='whether to use gpu. True or False')
        parser.add_argument('-c', '--config', type=int, default=0,
                            choices=[0, 1, 2, 3, 4],
                            help=textwrap.dedent('''\
                            select one of these config files:
                            0 -- reppoints_moment_r50_fpn_gn_neck_head_1x_coco.py
                            1 -- reppoints_moment_r50_fpn_gn_neck_head_1x_coco.py
                            2 -- reppoints_moment_r50_fpn_gn_neck_head_1x_coco.py'''))
        self.args = parser.parse_args()
        self.config_file = self.args.config
        self.use_gpu = self.args.use_gpu

    def get_use_gpu(self):
        return self.use_gpu

    def get_cfg(self):
        config_file = self.config_file
        cfg = None
        if config_file == 0:
            cfg = RepPoints_moment_r50_fpn_1x_Config()
        elif config_file == 1:
            cfg = RepPoints_moment_r50_fpn_1x_Config()
        elif config_file == 2:
            cfg = RepPoints_moment_r50_fpn_1x_Config()
        return cfg



