#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-10-23 09:13:23
#   Description : paddle2.0_ppyolo
#
# ================================================================



class PPYOLO_r18vd_Config(object):
    def __init__(self):
        # 自定义数据集
        # self.train_path = 'annotation_json/voc2012_train.json'
        # self.val_path = 'annotation_json/voc2012_val.json'
        # self.classes_path = 'data/voc_classes.txt'
        # self.train_pre_path = '../data/data4379/pascalvoc/VOCdevkit/VOC2012/JPEGImages/'   # 训练集图片相对路径
        # self.val_pre_path = '../data/data4379/pascalvoc/VOCdevkit/VOC2012/JPEGImages/'     # 验证集图片相对路径
        # self.num_classes = 20                              # 数据集类别数

        # AIStudio下的COCO数据集
        self.train_path = '../data/data7122/annotations/instances_train2017.json'
        self.val_path = '../data/data7122/annotations/instances_val2017.json'
        self.classes_path = 'data/coco_classes.txt'
        self.train_pre_path = '../data/data7122/train2017/'  # 训练集图片相对路径
        self.val_pre_path = '../data/data7122/val2017/'      # 验证集图片相对路径
        self.test_path = '../data/data7122/annotations/image_info_test-dev2017.json'      # test集
        self.test_pre_path = '../data/data7122/test2017/'    # test集图片相对路径
        self.num_classes = 80                                # 数据集类别数

        # Windows下的COCO数据集
        # self.train_path = '../COCO/annotations/instances_train2017.json'
        # self.val_path = '../COCO/annotations/instances_val2017.json'
        # self.classes_path = 'data/coco_classes.txt'
        # self.train_pre_path = '../COCO/train2017/'  # 训练集图片相对路径
        # self.val_pre_path = '../COCO/val2017/'      # 验证集图片相对路径
        # self.test_path = '../COCO/annotations/image_info_test-dev2017.json'      # test集
        # self.test_pre_path = '../COCO/test2017/'    # test集图片相对路径
        # self.num_classes = 80                       # 数据集类别数


        # ========= 一些设置 =========
        self.train_cfg = dict(
            batch_size=8,
            num_workers=5,   # 读数据的进程数
            num_threads=5,   # 读数据的线程数
            max_batch=2,     # 最大读多少个批
            model_path='dygraph_ppyolo_r18vd.pdparams',
            # model_path='./weights/1000.pdparams',
            update_iter=1,    # 每隔几步更新一次参数
            log_iter=20,      # 每隔几步打印一次
            save_iter=1000,   # 每隔几步保存一次模型
            eval_epoch=10,    # 每隔几轮计算一次eval集的mAP。
            max_epoch=270,    # 训练多少轮
            mixup_epoch=10,     # 前几轮进行mixup
            fp16=False,         # 是否用混合精度训练
            fleet=False,        # 是否用分布式训练
            find_unused_parameters=False,   # 是否在模型forward函数的返回值的所有张量中，遍历整个向后图。
        )
        self.learningRate = dict(
            base_lr=0.004 * self.train_cfg['batch_size'] / 128,
            PiecewiseDecay=dict(
                gamma=0.1,
                milestones_epoch=[162, 216],
            ),
            LinearWarmup=dict(
                start_factor=0.,
                steps=4000,
            ),
        )
        self.optimizerBuilder = dict(
            optimizer=dict(
                momentum=0.9,
                type='Momentum',
            ),
            regularizer=dict(
                factor=0.0005,
                type='L2',
            ),
        )


        # 验证。用于train.py、eval.py、test_dev.py
        self.eval_cfg = dict(
            model_path='dygraph_ppyolo_r18vd.pdparams',
            # model_path='./weights/1000.pdparams',
            target_size=416,
            draw_image=False,    # 是否画出验证集图片
            draw_thresh=0.15,    # 如果draw_image==True，那么只画出分数超过draw_thresh的物体的预测框。
            eval_batch_size=8,   # 验证时的批大小。
        )

        # 测试。用于demo.py
        self.test_cfg = dict(
            model_path='dygraph_ppyolo_r18vd.pdparams',
            # model_path='./weights/1000.pdparams',
            target_size=416,
            # target_size=320,
            draw_image=True,
            draw_thresh=0.15,   # 如果draw_image==True，那么只画出分数超过draw_thresh的物体的预测框。
        )


        # ============= 模型相关 =============
        self.use_ema = True
        # self.use_ema = False
        self.ema_decay = 0.9998
        self.ema_iter = 1
        self.backbone_type = 'Resnet18Vd'
        self.backbone = dict(
            norm_type='bn',
            feature_maps=[4, 5],
            dcn_v2_stages=[],
            freeze_at=0,
            fix_bn_mean_var_at=0,
            freeze_norm=False,
            norm_decay=0.,
        )
        self.head_type = 'YOLOv3Head'
        self.head = dict(
            num_classes=self.num_classes,
            conv_block_num=0,
            norm_type='bn',
            anchor_masks=[[3, 4, 5], [0, 1, 2]],
            anchors=[[10, 14], [23, 27], [37, 58],
                     [81, 82], [135, 169], [344, 319]],
            coord_conv=False,
            iou_aware=False,
            iou_aware_factor=0.4,
            scale_x_y=1.05,
            spp=False,
            drop_block=True,
            keep_prob=0.9,
            downsample=[32, 16],
            in_channels=[512, 256],
        )
        self.iou_loss_type = 'IouLoss'
        self.iou_loss = dict(
            loss_weight=2.5,
            max_height=608,
            max_width=608,
            ciou_term=False,
        )
        self.yolo_loss_type = 'YOLOv3Loss'
        self.yolo_loss = dict(
            ignore_thresh=0.7,
            scale_x_y=1.05,
            label_smooth=False,
            use_fine_grained_loss=True,
        )
        self.nms_cfg = dict(
            nms_type='matrix_nms',
            score_threshold=0.01,
            post_threshold=0.01,
            nms_top_k=500,
            keep_top_k=100,
            use_gaussian=False,
            gaussian_sigma=2.,
        )


        # ============= 预处理相关 =============
        self.context = {'fields': ['image', 'gt_bbox', 'gt_class', 'gt_score']}
        # DecodeImage
        self.decodeImage = dict(
            to_rgb=True,
            with_mixup=True,
            with_cutmix=False,
            with_mosaic=False,
        )
        # MixupImage
        self.mixupImage = dict(
            alpha=1.5,
            beta=1.5,
        )
        # CutmixImage
        self.cutmixImage = dict(
            alpha=1.5,
            beta=1.5,
        )
        # MosaicImage
        self.mosaicImage = dict(
            alpha=1.5,
            beta=1.5,
        )
        # ColorDistort
        self.colorDistort = dict()
        # RandomExpand
        self.randomExpand = dict(
            fill_value=[123.675, 116.28, 103.53],
        )
        # RandomCrop
        self.randomCrop = dict()
        # RandomFlipImage
        self.randomFlipImage = dict(
            is_normalized=False,
        )
        # NormalizeBox
        self.normalizeBox = dict()
        # PadBox
        self.padBox = dict(
            num_max_boxes=50,
        )
        # BboxXYXY2XYWH
        self.bboxXYXY2XYWH = dict()
        # RandomShape
        self.randomShape = dict(
            sizes=[320, 352, 384, 416, 448, 480, 512, 544, 576, 608],
            random_inter=True,
        )
        # NormalizeImage
        self.normalizeImage = dict(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            is_scale=True,
            is_channel_first=False,
        )
        # Permute
        self.permute = dict(
            to_bgr=False,
            channel_first=True,
        )
        # Gt2YoloTarget
        self.gt2YoloTarget = dict(
            anchor_masks=[[3, 4, 5], [0, 1, 2]],
            anchors=[[10, 14], [23, 27], [37, 58],
                     [81, 82], [135, 169], [344, 319]],
            downsample_ratios=[32, 16],
            num_classes=self.num_classes,
        )
        # ResizeImage
        self.resizeImage = dict(
            target_size=608,
            interp=2,
        )

        # 预处理顺序。增加一些数据增强时这里也要加上，否则train.py中相当于没加！
        self.sample_transforms_seq = []
        self.sample_transforms_seq.append('decodeImage')
        if self.decodeImage['with_mixup']:
            self.sample_transforms_seq.append('mixupImage')
        elif self.decodeImage['with_cutmix']:
            self.sample_transforms_seq.append('cutmixImage')
        elif self.decodeImage['with_mosaic']:
            self.sample_transforms_seq.append('mosaicImage')
        self.sample_transforms_seq.append('colorDistort')
        self.sample_transforms_seq.append('randomExpand')
        self.sample_transforms_seq.append('randomCrop')
        self.sample_transforms_seq.append('randomFlipImage')
        self.sample_transforms_seq.append('normalizeBox')
        self.sample_transforms_seq.append('padBox')
        self.sample_transforms_seq.append('bboxXYXY2XYWH')
        self.batch_transforms_seq = []
        self.batch_transforms_seq.append('randomShape')
        self.batch_transforms_seq.append('normalizeImage')
        self.batch_transforms_seq.append('permute')
        self.batch_transforms_seq.append('gt2YoloTarget')


