import paddle
import paddle.nn.functional as F
import paddle.nn as nn
from paddle import ParamAttr
from paddle.regularizer import L2Decay

from paddle.fluid.framework import Variable, in_dygraph_mode
from paddle.fluid import core
from paddle.fluid.layer_helper import LayerHelper

import numpy as np


def iou_similarity(x, y, box_normalized=True, name=None):
    """
    Computes intersection-over-union (IOU) between two box lists.
    Box list 'X' should be a LoDTensor and 'Y' is a common Tensor,
    boxes in 'Y' are shared by all instance of the batched inputs of X.
    Given two boxes A and B, the calculation of IOU is as follows:
    $$
    IOU(A, B) =
    \\frac{area(A\\cap B)}{area(A)+area(B)-area(A\\cap B)}
    $$
    Args:
        x (Tensor): Box list X is a 2-D Tensor with shape [N, 4] holds N
             boxes, each box is represented as [xmin, ymin, xmax, ymax],
             the shape of X is [N, 4]. [xmin, ymin] is the left top
             coordinate of the box if the input is image feature map, they
             are close to the origin of the coordinate system.
             [xmax, ymax] is the right bottom coordinate of the box.
             The data type is float32 or float64.
        y (Tensor): Box list Y holds M boxes, each box is represented as
             [xmin, ymin, xmax, ymax], the shape of X is [N, 4].
             [xmin, ymin] is the left top coordinate of the box if the
             input is image feature map, and [xmax, ymax] is the right
             bottom coordinate of the box. The data type is float32 or float64.
        box_normalized(bool): Whether treat the priorbox as a normalized box.
            Set true by default.
        name(str, optional): For detailed information, please refer
            to :ref:`api_guide_Name`. Usually name is no need to set and
            None by default.
    Returns:
        Tensor: The output of iou_similarity op, a tensor with shape [N, M]
              representing pairwise iou scores. The data type is same with x.
    Examples:
        .. code-block:: python
            import paddle
            from ppdet.modeling import ops
            paddle.enable_static()
            x = paddle.static.data(name='x', shape=[None, 4], dtype='float32')
            y = paddle.static.data(name='y', shape=[None, 4], dtype='float32')
            iou = ops.iou_similarity(x=x, y=y)
    """

    if in_dygraph_mode():
        out = core.ops.iou_similarity(x, y, 'box_normalized', box_normalized)
        return out
    else:
        helper = LayerHelper("iou_similarity", **locals())
        out = helper.create_variable_for_type_inference(dtype=x.dtype)

        helper.append_op(
            type="iou_similarity",
            inputs={"X": x,
                    "Y": y},
            attrs={"box_normalized": box_normalized},
            outputs={"Out": out})
        return out


def bbox2delta(src_boxes, tgt_boxes, weights):
    src_w = src_boxes[:, 2] - src_boxes[:, 0]
    src_h = src_boxes[:, 3] - src_boxes[:, 1]
    src_ctr_x = src_boxes[:, 0] + 0.5 * src_w
    src_ctr_y = src_boxes[:, 1] + 0.5 * src_h

    tgt_w = tgt_boxes[:, 2] - tgt_boxes[:, 0]
    tgt_h = tgt_boxes[:, 3] - tgt_boxes[:, 1]
    tgt_ctr_x = tgt_boxes[:, 0] + 0.5 * tgt_w
    tgt_ctr_y = tgt_boxes[:, 1] + 0.5 * tgt_h

    wx, wy, ww, wh = weights
    dx = wx * (tgt_ctr_x - src_ctr_x) / src_w
    dy = wy * (tgt_ctr_y - src_ctr_y) / src_h
    dw = ww * paddle.log(tgt_w / src_w)
    dh = wh * paddle.log(tgt_h / src_h)

    deltas = paddle.stack((dx, dy, dw, dh), axis=1)
    return deltas


def _bipartite_match_for_batch(gt_bbox, gt_label, prior_boxes,
                               bg_index):
    """
    Args:
        gt_bbox (Tensor): [B, N, 4]
        gt_label (Tensor): [B, N, 1]
        prior_boxes (Tensor): [A, 4]
        bg_index (int): Background class index

    SSD算法中，用先验框与gt的iou确定正样本。A一般有10000以上；
    N是每张图片的gt数，一般只有100左右。
    """
    batch_size, num_priors = gt_bbox.shape[0], prior_boxes.shape[0]
    ious = iou_similarity(gt_bbox.reshape((-1, 4)), prior_boxes).reshape(
        (batch_size, -1, num_priors)) # [B, N, A] 两两之间的iou

    # For each prior box, get the max IoU of all GTs.
    # [B, A] 对于每个先验框，求最匹配的gt
    # prior_argmax_iou.shape = [B, A]
    prior_max_iou, prior_argmax_iou = ious.max(axis=1), ious.argmax(axis=1)


    # For each GT, get the max IoU of all prior boxes.
    # [B, N] 对于每个gt，求最匹配的先验框
    # gt_argmax_iou.shape = [B, N]
    gt_max_iou, gt_argmax_iou = ious.max(axis=2), ious.argmax(axis=2)

    # Gather target bbox and label according to 'prior_argmax_iou' index.
    batch_ind = paddle.arange(end=batch_size, dtype='int64').unsqueeze(-1)  # [B, 1]
    batch_ind_tile = batch_ind.tile([1, num_priors])  # [B, A]
    prior_argmax_iou = paddle.stack([batch_ind_tile, prior_argmax_iou], axis=-1)  # [B, A, 2]

    # [B, A, 4] 把每个先验框(B*A个)需要学习的目标gt抽出来。
    targets_bbox = paddle.gather_nd(gt_bbox, prior_argmax_iou)
    # [B, A, 1] 把每个先验框(B*A个)需要学习的目标gt的类别id抽出来。
    targets_label = paddle.gather_nd(gt_label, prior_argmax_iou)


    # [B, A, 1]  负样本类别id
    bg_index_tensor = paddle.full([batch_size, num_priors, 1], bg_index, 'int64')

    overlap_threshold = 0.4
    # [B, A] -> [B, A, 1] 对于每个先验框，最匹配的gt的iou
    prior_max_iou = prior_max_iou.unsqueeze(-1)

    # [B, A, 1] 每个先验框需要学习的类别id。
    # 对于每个先验框，最匹配的gt的iou若小于overlap_threshold，则作为负样本。
    targets_label = paddle.where(prior_max_iou < overlap_threshold, bg_index_tensor, targets_label)

    # Ensure each GT can match the max IoU prior box.
    # 被选为正样本的anchor 在[B*A, ]中的下标
    pos_anchor_ind = batch_ind * num_priors
    pos_anchor_ind = pos_anchor_ind + gt_argmax_iou
    pos_anchor_ind = pos_anchor_ind.flatten()
    targets_bbox = targets_bbox.reshape([-1, 4])

    targets_bbox = paddle.scatter(targets_bbox, pos_anchor_ind, gt_bbox.reshape([-1, 4])).reshape([batch_size, -1, 4])
    targets_label = targets_label.reshape([-1, 1])
    targets_label = paddle.scatter(targets_label, pos_anchor_ind, gt_label.reshape([-1, 1])).reshape([batch_size, -1, 1])
    targets_label[:, :1] = bg_index

    # Encode box
    prior_box_var = 777
    prior_boxes = prior_boxes.unsqueeze(0).tile([batch_size, 1, 1])
    targets_bbox = bbox2delta(
        prior_boxes.reshape([-1, 4]),
        targets_bbox.reshape([-1, 4]), prior_box_var)
    targets_bbox = targets_bbox.reshape([batch_size, -1, 4])

    return targets_bbox, targets_label




# def _bipartite_match_for_batch(gt_bbox, gt_label, prior_boxes,
#                                bg_index):

# [2, 2, 4]
gt_bbox = np.array([
    [[0, 0, 10, 10], [100, 100, 110, 110]],
    [[0, 0, 10, 10], [100, 100, 110, 110]]
]).astype(np.float32)

# [2, 2, 1]
gt_label = np.array([
    [[16], [59]],
    [[37], [23]]
]).astype(np.int64)

# [3, 4]
prior_boxes = np.array([
    [0, 0, 12, 12], [50, 50, 60, 60], [100, 100, 110, 110]
]).astype(np.float32)


gt_bbox = paddle.to_tensor(gt_bbox)
gt_label = paddle.to_tensor(gt_label)
prior_boxes = paddle.to_tensor(prior_boxes)

aaaaaaa = _bipartite_match_for_batch(gt_bbox, gt_label, prior_boxes, -1)
print()








