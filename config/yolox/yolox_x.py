#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2021-10-12 11:23:07
#   Description : yolox
#
# ================================================================



class YOLOX_X_Config(object):
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
            # batch_size=128,
            batch_size=6,
            num_workers=4,   # 读数据的进程数
            # model_path='dygraph_yolox_x.pdparams',
            model_path=None,
            # model_path='./weights/1000.pdparams',
            update_iter=1,    # 每隔几步更新一次参数
            log_iter=20,      # 每隔几步打印一次
            save_iter=1000,   # 每隔几步保存一次模型
            eval_epoch=10,    # 每隔几轮计算一次eval集的mAP。
            eval_iter=-1,    # 每隔几步计算一次eval集的mAP。设置了eval_epoch的话会自动计算。
            max_epoch=300,    # 训练多少轮
            max_iters=-1,     # 训练多少步。设置了max_epoch的话会自动计算。
            mosaic_epoch=285,  # 前几轮进行mosaic
            fp16=False,         # 是否用混合精度训练
            fleet=False,        # 是否用分布式训练
            find_unused_parameters=False,   # 是否在模型forward函数的返回值的所有张量中，遍历整个向后图。
        )
        self.learningRate = dict(
            base_lr=0.01 * self.train_cfg['batch_size'] / 64,
            CosineDecay=dict(),
            LinearWarmup=dict(
                start_factor=0.,
                epochs=5,
            ),
        )
        self.optimizerBuilder = dict(
            optimizer=dict(
                momentum=0.9,
                use_nesterov=True,
                type='Momentum',
            ),
            regularizer=dict(
                factor=0.0005,
                type='L2',
            ),
        )


        # 验证。用于train.py、eval.py、test_dev.py
        self.eval_cfg = dict(
            model_path='dygraph_yolox_x.pdparams',
            # model_path='./weights/1000.pdparams',
            target_size=640,
            max_size=640,
            draw_image=False,    # 是否画出验证集图片
            draw_thresh=0.15,    # 如果draw_image==True，那么只画出分数超过draw_thresh的物体的预测框。
            eval_batch_size=1,   # 验证时的批大小。
        )

        # 测试。用于demo.py
        self.test_cfg = dict(
            model_path='dygraph_yolox_x.pdparams',
            # model_path='./weights/1000.pdparams',
            target_size=640,
            max_size=640,
            draw_image=True,
            draw_thresh=0.15,   # 如果draw_image==True，那么只画出分数超过draw_thresh的物体的预测框。
        )


        # ============= 模型相关 =============
        self.use_ema = True
        # self.use_ema = False
        self.ema_decay = 0.9998
        self.ema_iter = 1
        self.backbone_type = 'CSPDarknet'
        self.backbone = dict(
            dep_mul=1.33,
            wid_mul=1.25,
            freeze_at=0,
            fix_bn_mean_var_at=0,
        )
        self.fpn_type = 'YOLOPAFPN'
        self.fpn = dict(
            depth=self.backbone['dep_mul'],
            width=self.backbone['wid_mul'],
            in_features=("dark3", "dark4", "dark5"),
            in_channels=[256, 512, 1024],
            depthwise=False,
            act="silu",
        )
        self.head_type = 'YOLOXHead'
        self.head = dict(
            num_classes=self.num_classes,
            width=self.backbone['wid_mul'],
            strides=[8, 16, 32],
            in_channels=[256, 512, 1024],
            act='silu',
            depthwise=False,
        )
        self.iou_loss_type = 'IOUloss'
        self.iou_loss = dict(
            reduction='none',
            loss_type='iou',
        )
        # self.nms_cfg = dict(
        #     nms_type='matrix_nms',
        #     score_threshold=0.01,
        #     post_threshold=0.01,
        #     nms_top_k=500,
        #     keep_top_k=100,
        #     use_gaussian=False,
        #     gaussian_sigma=2.,
        # )
        self.nms_cfg = dict(
            nms_type='multiclass_nms',
            score_threshold=0.01,
            nms_threshold=0.65,
            nms_top_k=1000,
            keep_top_k=100,
        )


        # ============= 预处理相关 =============
        self.context = {'fields': ['image', 'gt_bbox', 'gt_class', 'gt_score']}
        # DecodeImage
        self.decodeImage = dict(
            to_rgb=False,
            with_mixup=True,
            with_cutmix=False,
            with_mosaic=True,
        )
        # YOLOXMosaicImage
        self.yOLOXMosaicImage = dict(
            prob=1.0,
            degrees=10.0,
            translate=0.1,
            scale=(0.1, 2),
            shear=2.0,
            perspective=0.0,
            input_dim=(640, 640),
        )
        # ColorDistort
        self.colorDistort = dict()
        # RandomFlipImage
        self.randomFlipImage = dict(
            is_normalized=False,
        )
        # BboxXYXY2XYWH
        self.bboxXYXY2XYWH = dict()
        # YOLOXResizeImage
        self.yOLOXResizeImage = dict(
            target_size=[640 - i * 32 for i in range(7)] + [640 + i * 32 for i in range(1, 7)],
            # target_size=[640],
            interp=1,  # cv2.INTER_LINEAR = 1
            use_cv2=True,
            resize_box=True,
        )
        # PadBox
        self.padBox = dict(
            num_max_boxes=120,
            init_bbox=[-9999.0, -9999.0, 10.0, 10.0],
        )
        # SquareImage
        self.squareImage = dict(
            fill_value=114,
            is_channel_first=False,
        )
        # Permute
        self.permute = dict(
            to_bgr=False,
            channel_first=True,
        )

        # 预处理顺序。增加一些数据增强时这里也要加上，否则train.py中相当于没加！
        self.sample_transforms_seq = []
        self.sample_transforms_seq.append('decodeImage')
        self.sample_transforms_seq.append('yOLOXMosaicImage')
        self.sample_transforms_seq.append('colorDistort')
        self.sample_transforms_seq.append('randomFlipImage')
        self.sample_transforms_seq.append('bboxXYXY2XYWH')
        self.sample_transforms_seq.append('yOLOXResizeImage')
        self.sample_transforms_seq.append('padBox')
        self.batch_transforms_seq = []
        self.batch_transforms_seq.append('squareImage')
        self.batch_transforms_seq.append('permute')


