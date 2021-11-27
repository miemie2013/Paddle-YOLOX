import paddle
import numpy as np
import paddle.nn.functional as F



'''
假设批大小
N = 2
第0张图片有2个gt，第1张图片有3个gt，所以
G = 3

每张图片输出5个预测框（当然，原来640*640输入时是8400个预测框）
A = 5

cost.shape = [N, G, A]  gt 和 所有预测框 两两之间的cost。

'''

N = 2
G = 3
A = 5

cost = np.ones((N, G, A), np.float32)
cost[0, :, :] = np.array([[2, 99, 102, 3, 108],
                          [4, 100, 103, 2, 109],
                          [9, 100, 103, 1, 109]]).astype(np.float32)
cost[1, :, :] = np.array([[2, 99, 102, 1, 108],
                          [5, 100, 103, 2, 109],
                          [7, 100, 103, 3, 109]]).astype(np.float32)


# [N, G]  表示每个gt分配给了几个预测框。最少1个。
dynamic_ks = np.array([[2, 1, 1], [1, 1, 1]]).astype(np.int32)

# [N, G]  是否是gt。
is_gt = np.array([[1, 1, 0], [1, 1, 1]]).astype(np.float32)

is_in_boxes_or_center = np.array([[3, 100, 103, 2, 109],
                                  [3, 100, 103, 2, 109]]).astype(np.float32)

cost = paddle.to_tensor(cost)
dynamic_ks = paddle.to_tensor(dynamic_ks)
is_gt = paddle.to_tensor(is_gt)

max_dynamic_ks = dynamic_ks.max(-1)  # [N, ]  每张图片所有gt的dynamic_ks的最大值
max_k = max_dynamic_ks.max()  # [1, ]  所有图片所有gt的dynamic_ks的最大值


# 下三角全是1的矩阵
topk_mask = paddle.ones((max_k, max_k), 'float32')  # [max_k, max_k]
topk_mask = paddle.tril(topk_mask, diagonal=0)      # [max_k, max_k]
fill_value = paddle.gather(topk_mask, dynamic_ks.reshape((-1,)) - 1)  # [N*G, max_k]   填入matching_matrix
fill_value *= is_gt.reshape((-1, 1))  # [N*G, max_k]  还要处理假gt，假gt处全部填0
fill_value = fill_value.reshape((-1,))  # [N*G*max_k, ]   填入matching_matrix
# 不放心的话，再次将假gt的cost增大
cost += (1.0 - is_gt.unsqueeze(2)) * 100000.0
min_cost, min_cost_index = paddle.topk(cost, k=max_k, axis=2, largest=False, sorted=True)

matching_matrix = paddle.zeros([N * G * A, ], 'float32')  # [N*G*A, ]



gt_ind = paddle.arange(end=N*G, dtype='int32').unsqueeze(-1)  # [N*G, 1]  每个gt在matching_matrix中的下标。
min_cost_index = min_cost_index.reshape((N * G, max_k))  # [N*G, max_k]
min_cost_index = gt_ind * A + min_cost_index  # [N*G, max_k]
min_cost_index = min_cost_index.flatten()  # [N*G*max_k, ]

matching_matrix = paddle.scatter(matching_matrix, min_cost_index, fill_value, overwrite=True)

matching_matrix = matching_matrix.reshape((N, G, A))  # [N, G, A]

# 看cost[1, :, :]，3个gt框都和第3个anchor有最小cost，即第3个anchor匹配到了3个gt框。不可能1个anchor学习3个gt，
# 所以这时需要改写matching_matrix，让第3个anchor学习与其具有最小cost的那个gt
anchor_matching_gt = matching_matrix.sum(1)  # [N, A]  每个anchor匹配到了几个gt？


# 如果有anchor（花心大萝卜）匹配到了1个以上的gt时，做特殊处理。
if paddle.cast(anchor_matching_gt > 1, 'float32').sum() > 0:
    # 找到 花心大萝卜 的下标（这是在anchor_matching_gt.shape[N, A]中的下标）。假设有R个花心大萝卜。
    index4 = paddle.nonzero(anchor_matching_gt > 1)  # [R, 2]
    cost_t = cost.transpose((0, 2, 1))  # [N, G, A] -> [N, A, G]  转置好提取其cost
    cost2 = paddle.gather_nd(cost_t, index4)  # [R, G]  R个花心大萝卜 与 gt 两两之间的cost。
    cost2 = cost2.transpose((1, 0))        # [G, R]  gt 与 R个花心大萝卜 两两之间的cost。
    cost_argmin = cost2.argmin(axis=0)     # [R, ]  为 每个花心大萝卜 找到 与其cost最小的gt 的下标

    # 准备one_hot
    one_hots = F.one_hot(cost_argmin, num_classes=G)  # [R, G]

    # 花心大萝卜 处 填入one_hot
    matching_matrix = matching_matrix.transpose((0, 2, 1))  # [N, G, A] -> [N, A, G]  转置好以让scatter()填入
    matching_matrix = matching_matrix.reshape((N*A, G))     # [N*A, G]  reshape好以让scatter()填入
    index4 = index4[:, 0] * A + index4[:, 1]
    matching_matrix = paddle.scatter(matching_matrix, index4, one_hots, overwrite=True)   # [N*A, G]  scatter()填入

    # matching_matrix变回原来的形状
    matching_matrix = matching_matrix.reshape((N, A, G))    # [N, A, G]
    matching_matrix = matching_matrix.transpose((0, 2, 1))  # [N, A, G] -> [N, G, A]

# [N, A]  是否是前景（正样本）
fg_mask = matching_matrix.sum(1) > 0.0   # [N, A]
fg_mask = paddle.cast(fg_mask, 'float32')   # [N, A]
num_fg = fg_mask.sum()  # 所有图片前景个数

# 确定最终正样本需要学习的类别id。假设有T个最终正样本。
index3 = paddle.nonzero(fg_mask > 0)  # [T, 2]

matching_matrix_t = matching_matrix.transpose((0, 2, 1))  # [N, G, A] -> [N, A, G]  转置好以便gather_nd()抽取
matched_gt_inds = paddle.gather_nd(matching_matrix_t, index3)  # [T, G]
matched_gt_inds = matched_gt_inds.transpose((1, 0))  # [G, T]
matched_gt_inds = matched_gt_inds.argmax(0)  # [T, ]  最终正样本是匹配到了第几个gt

fg_mask = paddle.argmax(fg_mask, 'float32')   # [N, A]

print()









