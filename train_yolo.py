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
import time
import datetime
import json
import paddle
import paddle.distributed as dist
from paddle.distributed import fleet
from paddle import amp
from paddle.static import InputSpec

from model.EMA import ExponentialMovingAverage
from model.decoders.decode_yolo import *
from model.architectures.yolo import *
from tools.cocotools_yolo import eval
from tools.argparser import *
from tools.data_process import data_clean, get_samples
from tools.train_utils import *
from pycocotools.coco import COCO

import logging

FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


def multi_thread_op(i, num_threads, batch_size, samples, context, with_mixup, sample_transforms, batch_transforms,
                    shape, images, gt_bbox, gt_score, gt_class, target0, target1, target2, n_layers):
    for k in range(i, batch_size, num_threads):
        for sample_transform in sample_transforms:
            if isinstance(sample_transform, MixupImage):
                if with_mixup:
                    samples[k] = sample_transform(samples[k], context)
            else:
                samples[k] = sample_transform(samples[k], context)

        for batch_transform in batch_transforms:
            if isinstance(batch_transform, RandomShapeSingle):
                samples[k] = batch_transform(shape, samples[k], context)
            else:
                samples[k] = batch_transform(samples[k], context)

        # 整理成ndarray
        images[k] = np.expand_dims(samples[k]['image'].astype(np.float32), 0)
        gt_bbox[k] = np.expand_dims(samples[k]['gt_bbox'].astype(np.float32), 0)
        gt_score[k] = np.expand_dims(samples[k]['gt_score'].astype(np.float32), 0)
        gt_class[k] = np.expand_dims(samples[k]['gt_class'].astype(np.int32), 0)
        target0[k] = np.expand_dims(samples[k]['target0'].astype(np.float32), 0)
        target1[k] = np.expand_dims(samples[k]['target1'].astype(np.float32), 0)
        if n_layers > 2:
            target2[k] = np.expand_dims(samples[k]['target2'].astype(np.float32), 0)


def read_train_data(cfg,
                    train_indexes,
                    train_steps,
                    train_records,
                    batch_size,
                    _iter_id,
                    train_dic,
                    use_gpu,
                    n_layers,
                    context, with_mixup, with_cutmix, with_mosaic, mixup_steps, cutmix_steps, mosaic_steps, sample_transforms, batch_transforms):
    iter_id = _iter_id
    num_threads = cfg.train_cfg['num_threads']
    while True:   # 无限个epoch
        # 每个epoch之前洗乱
        np.random.shuffle(train_indexes)
        for step in range(train_steps):
            iter_id += 1

            key_list = list(train_dic.keys())
            key_len = len(key_list)
            while key_len >= cfg.train_cfg['max_batch']:
                time.sleep(0.01)
                key_list = list(train_dic.keys())
                key_len = len(key_list)

            # ==================== train ====================
            sizes = cfg.randomShape['sizes']
            shape = np.random.choice(sizes)
            images = [None] * batch_size
            gt_bbox = [None] * batch_size
            gt_score = [None] * batch_size
            gt_class = [None] * batch_size
            target0 = [None] * batch_size
            target1 = [None] * batch_size
            target2 = [None] * batch_size

            samples = get_samples(train_records, train_indexes, step, batch_size, iter_id,
                                  with_mixup, with_cutmix, with_mosaic, mixup_steps, cutmix_steps, mosaic_steps)
            # sample_transforms用多线程
            threads = []
            for i in range(num_threads):
                t = threading.Thread(target=multi_thread_op, args=(i, num_threads, batch_size, samples, context, with_mixup, sample_transforms, batch_transforms,
                                                                   shape, images, gt_bbox, gt_score, gt_class, target0, target1, target2, n_layers))
                threads.append(t)
                t.start()
            # 等待所有线程任务结束。
            for t in threads:
                t.join()

            images = np.concatenate(images, 0)
            gt_bbox = np.concatenate(gt_bbox, 0)
            gt_score = np.concatenate(gt_score, 0)
            gt_class = np.concatenate(gt_class, 0)
            target0 = np.concatenate(target0, 0)
            target1 = np.concatenate(target1, 0)
            if n_layers > 2:
                target2 = np.concatenate(target2, 0)

            images = paddle.to_tensor(images, place=place)
            gt_bbox = paddle.to_tensor(gt_bbox, place=place)
            gt_score = paddle.to_tensor(gt_score, place=place)
            gt_class = paddle.to_tensor(gt_class, place=place)
            target0 = paddle.to_tensor(target0, place=place)
            target1 = paddle.to_tensor(target1, place=place)
            if n_layers > 2:
                target2 = paddle.to_tensor(target2, place=place)

            dic = {}
            dic['images'] = images
            dic['gt_bbox'] = gt_bbox
            dic['gt_score'] = gt_score
            dic['gt_class'] = gt_class
            dic['target0'] = target0
            dic['target1'] = target1
            if n_layers > 2:
                dic['target2'] = target2
            train_dic['%.8d'%iter_id] = dic

            # ==================== exit ====================
            if iter_id == cfg.train_cfg['max_iters']:
                return 0


if __name__ == '__main__':
    parser = YOLOArgParser()
    use_gpu = parser.get_use_gpu()
    cfg = parser.get_cfg()
    print(paddle.__version__)
    paddle.disable_static()   # 开启动态图
    gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
    place = paddle.CUDAPlace(gpu_id) if use_gpu else paddle.CPUPlace()

    # 打印，确认一下使用的配置
    print('\n=============== config message ===============')
    print('config file: %s' % str(type(cfg)))
    if cfg.train_cfg['model_path'] is not None:
        print('pretrained_model: %s' % cfg.train_cfg['model_path'])
    else:
        print('pretrained_model: None')
    print('use_gpu: %s' % str(use_gpu))
    print()

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


    # 步id，无需设置，会自动读。
    iter_id = 0

    # 创建模型
    Backbone = select_backbone(cfg.backbone_type)
    backbone = Backbone(**cfg.backbone)
    IouLoss = select_loss(cfg.iou_loss_type)
    iou_loss = IouLoss(**cfg.iou_loss)
    iou_aware_loss = None
    if cfg.head['iou_aware']:
        IouAwareLoss = select_loss(cfg.iou_aware_loss_type)
        iou_aware_loss = IouAwareLoss(**cfg.iou_aware_loss)
    Loss = select_loss(cfg.yolo_loss_type)
    yolo_loss = Loss(iou_loss=iou_loss, iou_aware_loss=iou_aware_loss, **cfg.yolo_loss)
    Head = select_head(cfg.head_type)
    head = Head(yolo_loss=yolo_loss, is_train=True, nms_cfg=cfg.nms_cfg, **cfg.head)
    model = PPYOLO(backbone, head)
    _decode = Decode_YOLO(model, class_names, place, cfg, for_test=False)

    # optimizer
    regularization = None
    if cfg.optimizerBuilder['regularizer'] is not None:
        reg_args = cfg.optimizerBuilder['regularizer'].copy()
        reg_type = reg_args['type'] + 'Decay'   # 正则化类型。L1、L2
        reg_factor = reg_args['factor']
        Regularization = select_regularization(reg_type)
        # 在 优化器 中设置正则化。
        # 不可以加正则化的参数：norm层(比如bn层、affine_channel层、gn层)的scale、offset；卷积层的偏移参数。
        # 如果同时在 可训练参数的ParamAttr 和 优化器optimizer 中设置正则化， 那么在 可训练参数的ParamAttr 中设置的优先级会高于在 optimizer 中的设置。
        # 也就是说，等价于没给    norm层(比如bn层、affine_channel层、gn层)的scale、offset；卷积层的偏移参数    加正则化。
        regularization = Regularization(reg_factor)
    optim_args = cfg.optimizerBuilder['optimizer'].copy()
    optim_type = optim_args['type']   # 使用哪种优化器。Momentum、Adam、SGD、...之类的。
    Optimizer = select_optimizer(optim_type)
    del optim_args['type']
    optimizer = Optimizer(learning_rate=cfg.learningRate['base_lr'],
                          parameters=model.parameters(),
                          weight_decay=regularization,   # 正则化
                          grad_clip=None,   # 梯度裁剪
                          **optim_args)

    # 加载权重
    if cfg.train_cfg['model_path'] is not None:
        # 加载参数, 跳过形状不匹配的。
        load_weights(model, cfg.train_cfg['model_path'])

        strs = cfg.train_cfg['model_path'].split('weights/')
        if len(strs) == 2:
            iter_id = int(strs[1].split('.')[0])

    # 冻结，使得需要的显存减少。低显存的卡建议这样配置。
    backbone.freeze()
    backbone.fix_bn()

    print('\n=============== Model ===============')
    trainable_params = 0
    nontrainable_params = 0
    for name_, param_ in model.named_parameters():
        mul = np.prod(param_.shape)
        if param_.stop_gradient is False:
            trainable_params += mul
        else:
            nontrainable_params += mul
    total_params = trainable_params + nontrainable_params
    print('Total params: %s' % format(total_params, ","))
    print('Trainable params: %s' % format(trainable_params, ","))
    print('Non-trainable params: %s\n' % format(nontrainable_params, ","))

    ema = None
    if cfg.use_ema:
        ema = ExponentialMovingAverage(model, cfg.ema_decay)
        ema.register()

    # 分布式训练与混合精度训练
    # 有疑问请参考文档https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/06_distributed_training/cluster_quick_start_cn.html
    _nranks = dist.get_world_size()
    _local_rank = dist.get_rank()
    use_fleet = cfg.train_cfg.get('fleet', False)
    use_fp16 = cfg.train_cfg.get('fp16', False)
    if use_fleet:
        # 初始化Fleet环境
        fleet.init(is_collective=True)
        # 通过Fleet API获取分布式model，用于支持分布式训练
        model = fleet.distributed_model(model)
        optimizer = fleet.distributed_optimizer(optimizer)
    elif _nranks > 1:
        find_unused_parameters = cfg.train_cfg['find_unused_parameters'] \
            if 'find_unused_parameters' in cfg.train_cfg else False
        model = paddle.DataParallel(model, find_unused_parameters=find_unused_parameters)
    if use_fp16:
        # scaler = amp.GradScaler(enable=use_gpu, init_loss_scaling=2.**16,
        #                         incr_every_n_steps=2000, use_dynamic_loss_scaling=True)
        scaler = amp.GradScaler(enable=use_gpu, init_loss_scaling=1024)

    print('\n=============== fleet and fp16 ===============')
    print('use_fleet: %d' % use_fleet)
    print('use_fp16: %d' % use_fp16)
    print('_nranks: %d' % _nranks)
    print('_local_rank: %d' % _local_rank)
    print()


    # 训练集
    train_dataset = COCO(cfg.train_path)
    train_img_ids = train_dataset.getImgIds()
    train_records = data_clean(train_dataset, train_img_ids, _catid2clsid, cfg.train_pre_path)
    num_train = len(train_records)
    train_indexes = [i for i in range(num_train)]
    # 验证集
    val_dataset = COCO(cfg.val_path)
    val_img_ids = val_dataset.getImgIds()
    val_images = []   # 只跑有gt的图片，跟随PaddleDetection
    for img_id in val_img_ids:
        ins_anno_ids = val_dataset.getAnnIds(imgIds=img_id, iscrowd=False)   # 读取这张图片所有标注anno的id
        if len(ins_anno_ids) == 0:
            continue
        img_anno = val_dataset.loadImgs(img_id)[0]
        val_images.append(img_anno)

    batch_size = cfg.train_cfg['batch_size']
    with_mixup = cfg.decodeImage['with_mixup']
    with_cutmix = cfg.decodeImage['with_cutmix']
    with_mosaic = cfg.decodeImage['with_mosaic']
    mixup_epoch = cfg.train_cfg.get('mixup_epoch', 0)
    cutmix_epoch = cfg.train_cfg.get('cutmix_epoch', 0)
    mosaic_epoch = cfg.train_cfg.get('mosaic_epoch', 0)
    context = cfg.context
    # 预处理
    sample_transforms, batch_transforms = get_transforms(cfg)

    print('\n=============== sample_transforms ===============')
    for trf in sample_transforms:
        print('%s' % str(type(trf)))
    print('\n=============== batch_transforms ===============')
    for trf in batch_transforms:
        print('%s' % str(type(trf)))

    # 输出几个特征图
    n_layers = len(cfg.head['anchor_masks'])

    # 保存模型的目录
    if not os.path.exists('./weights'): os.mkdir('./weights')

    time_stat = deque(maxlen=20)
    start_time = time.time()
    end_time = time.time()

    # 一轮的步数。丢弃最后几个样本。
    train_steps = num_train // batch_size
    mixup_steps = mixup_epoch * train_steps
    cutmix_steps = cutmix_epoch * train_steps
    mosaic_steps = mosaic_epoch * train_steps

    # 修改cfg.train_cfg
    max_iters = cfg.train_cfg.get('max_iters', None)
    max_epoch = cfg.train_cfg.get('max_epoch', None)
    if max_epoch is not None:
        max_iters = train_steps * max_epoch
        cfg.train_cfg['max_iters'] = max_iters
    elif max_epoch is None:
        max_epoch = max_iters // train_steps
        cfg.train_cfg['max_epoch'] = max_epoch
    eval_epoch = cfg.train_cfg.get('eval_epoch', None)
    if eval_epoch is not None:
        eval_iter = train_steps * eval_epoch
        cfg.train_cfg['eval_iter'] = eval_iter

    print('\n=============== mixup and cutmix ===============')
    print('steps_per_epoch: %d' % train_steps)
    if with_mixup:
        print('mixup_steps: %d' % mixup_steps)
    else:
        print('don\'t use mixup.')
    if with_cutmix:
        print('cutmix_steps: %d' % cutmix_steps)
    else:
        print('don\'t use cutmix.')
    if with_mosaic:
        print('mosaic_steps: %d' % mosaic_steps)
    else:
        print('don\'t use mosaic.')

    # 读数据的线程
    train_dic ={}
    thr = threading.Thread(target=read_train_data,
                           args=(cfg,
                                 train_indexes,
                                 train_steps,
                                 train_records,
                                 batch_size,
                                 iter_id,
                                 train_dic,
                                 use_gpu,
                                 n_layers,
                                 context, with_mixup, with_cutmix, with_mosaic, mixup_steps, cutmix_steps, mosaic_steps, sample_transforms, batch_transforms))
    thr.start()


    nowTime = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    log_filename = 'log%s.txt'%nowTime
    best_ap_list = [0.0, 0]  #[map, iter]
    while True:   # 无限个epoch
        for step in range(train_steps):
            iter_id += 1

            key_list = list(train_dic.keys())
            key_len = len(key_list)
            while key_len == 0:
                time.sleep(0.01)
                key_list = list(train_dic.keys())
                key_len = len(key_list)
            dic = train_dic.pop('%.8d'%iter_id)

            # 估计剩余时间
            start_time = end_time
            end_time = time.time()
            time_stat.append(end_time - start_time)
            time_cost = np.mean(time_stat)
            eta_sec = (cfg.train_cfg['max_iters'] - iter_id) * time_cost
            eta = str(datetime.timedelta(seconds=int(eta_sec)))

            # ==================== train ====================
            images = dic['images']
            gt_bbox = dic['gt_bbox']
            gt_score = dic['gt_score']
            gt_class = dic['gt_class']
            target0 = dic['target0']
            target1 = dic['target1']
            if n_layers > 2:
                target2 = dic['target2']
                targets = [target0, target1, target2]
            else:
                targets = [target0, target1]
            losses = model.train_model(images, gt_bbox, gt_class, gt_score, targets)
            all_loss = 0.0
            loss_names = {}
            for loss_name in losses.keys():
                sub_loss = losses[loss_name]
                all_loss += sub_loss
                loss_names[loss_name] = sub_loss.numpy()[0]
            _all_loss = all_loss.numpy()[0]

            # 更新权重
            lr = calc_lr(iter_id, train_steps, cfg.train_cfg['max_iters'], cfg)
            optimizer.set_lr(lr)
            all_loss.backward()
            if iter_id % cfg.train_cfg['update_iter'] == 0:
                optimizer.step()
                optimizer.clear_grad()
            if cfg.use_ema and iter_id % cfg.ema_iter == 0:
                ema.update()   # 更新ema字典

            # ==================== log ====================
            if iter_id % cfg.train_cfg['log_iter'] == 0:
                speed = (1.0 / time_cost)
                speed *= batch_size
                speed_msg = '%.3f imgs/s.' % (speed,)
                lr = optimizer.get_lr()
                each_loss = ''
                for loss_name in loss_names.keys():
                    loss_value = loss_names[loss_name]
                    each_loss += ' %s: %.3f,' % (loss_name, loss_value)
                epoch_id = iter_id // train_steps
                strs = 'Train iter: {}/{}, epoch: {}/{}, lr: {:.9f}, all_loss: {:.3f},{} eta: {}, speed: {}'.format(iter_id, cfg.train_cfg['max_iters'], epoch_id+1, cfg.train_cfg['max_epoch'], lr, _all_loss, each_loss, eta, speed_msg)
                logger.info(strs)
                write(log_filename, strs)

            # ==================== save ====================
            if iter_id % cfg.train_cfg['save_iter'] == 0 or iter_id == cfg.train_cfg['max_iters']:
                if cfg.use_ema:
                    ema.apply()
                save_path = './weights/%d.pdparams' % iter_id
                paddle.save(model.state_dict(), save_path)
                if cfg.use_ema:
                    ema.restore()
                logger.info('Save model to {}'.format(save_path))
                write(log_filename, 'Save model to {}'.format(save_path))
                clear_model('weights')

            # ==================== eval ====================
            if iter_id % cfg.train_cfg['eval_iter'] == 0:
                if cfg.use_ema:
                    ema.apply()
                model.eval()   # 切换到验证模式
                head.set_dropblock(is_test=True)
                box_ap = eval(_decode, val_images, cfg.val_pre_path, cfg.val_path, cfg.eval_cfg['eval_batch_size'], _clsid2catid, cfg.eval_cfg['draw_image'], cfg.eval_cfg['draw_thresh'])
                logger.info("box ap: %.3f" % (box_ap[0], ))
                model.train()  # 切换到训练模式
                backbone.fix_bn()  # model.train()后需要重新固定backbone的bn层。
                head.set_dropblock(is_test=False)

                # 以box_ap作为标准
                ap = box_ap
                write(log_filename, 'Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = %.3f' % (box_ap[0], ))
                write(log_filename, 'Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = %.3f' % (box_ap[1], ))
                write(log_filename, 'Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = %.3f' % (box_ap[2], ))
                write(log_filename, 'Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = %.3f' % (box_ap[3], ))
                write(log_filename, 'Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = %.3f' % (box_ap[4], ))
                write(log_filename, 'Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = %.3f' % (box_ap[5], ))
                write(log_filename, 'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = %.3f' % (box_ap[6], ))
                write(log_filename, 'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = %.3f' % (box_ap[7], ))
                write(log_filename, 'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = %.3f' % (box_ap[8], ))
                write(log_filename, 'Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = %.3f' % (box_ap[9], ))
                write(log_filename, 'Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = %.3f' % (box_ap[10], ))
                write(log_filename, 'Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = %.3f' % (box_ap[11], ))
                if ap[0] > best_ap_list[0]:
                    best_ap_list[0] = ap[0]
                    best_ap_list[1] = iter_id
                    save_path = './weights/best_model.pdparams'
                    paddle.save(model.state_dict(), save_path)
                    logger.info('Save model to {}'.format(save_path))
                    write(log_filename, 'Save model to {}'.format(save_path))
                    clear_model('weights')
                if cfg.use_ema:
                    ema.restore()
                logger.info("Best test ap: {}, in iter: {}".format(best_ap_list[0], best_ap_list[1]))
                write(log_filename, "Best test ap: {}, in iter: {}".format(best_ap_list[0], best_ap_list[1]))

            # ==================== exit ====================
            if iter_id == cfg.train_cfg['max_iters']:
                logger.info('Done.')
                write(log_filename, 'Done.')
                exit(0)

