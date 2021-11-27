#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-10-23 09:13:23
#   Description : paddle2.0_ppyolo
#
# ================================================================
from . import get_model
from .get_model import *

from .ppyolo.ppyolo_2x import *
from .ppyolo.ppyolo_mdf_2x import *
from .ppyolo.yolov4_2x import *
from .ppyolo.ppyolo_r18vd import *
from .ppyolo.ppyolo_mobilenet_v3_large import *
from .ppyolo.ppyolo_large_2x import *

from .fcos.fcos_r50_fpn_multiscale_2x import *
from .fcos.fcos_rt_r50_fpn_4x import *
from .fcos.fcos_rt_dla34_fpn_4x import *
from .fcos.fcos_rt_r50vd_fpn_dcn_2x import *
from .fcos.fcos_dcn_r50_fpn_1x import *

from .solo.solov2_light_448_r50_fpn_8gpu_3x import *
from .solo.solov2_light_r50_vd_fpn_dcn_512_3x import *
from .solo.solov2_r50_fpn_8gpu_3x import *

from .reppoints.reppoints_moment_r50_fpn_gn_neck_head_1x_coco import *

from .yolox.yolox_s import *
from .yolox.yolox_m import *
from .yolox.yolox_l import *
from .yolox.yolox_x import *

