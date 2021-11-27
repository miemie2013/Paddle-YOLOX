import paddle
import numpy as np
import paddle.fluid.layers as L


fg_mask = np.array([0, 0, 1, 0, 1, 1]).astype(np.float32)
fg_mask_inboxes = np.array([1, 0, 1]).astype(np.float32)

# import torch
# fg_mask2 = torch.Tensor(fg_mask).cuda().bool()
# fg_mask_inboxes2 = torch.Tensor(fg_mask_inboxes).cuda().bool()

# 这句代码的意思是fg_mask2里是1的地方依次被填入fg_mask_inboxes2的值。
# fg_mask2[fg_mask2.clone()] = fg_mask_inboxes2


'''
也就是这个问题：获取张量里某些下标里的值我们可以用gather()或gather_nd()，
修改张量里某些下标里的值用什么？用scatter()！
scatter()用于修改张量里某些下标里的值。看SSD的正负样本确定源码发现的。
'''


# 方法一
# fg_mask = paddle.to_tensor(fg_mask).astype(paddle.bool)
# fg_mask_inboxes = paddle.to_tensor(fg_mask_inboxes).astype(paddle.bool)
# print(paddle.__version__)
# fg_mask[fg_mask.clone()] = fg_mask_inboxes



# 方法二
# fg_mask = paddle.to_tensor(fg_mask)
# fg_mask_inboxes = paddle.to_tensor(fg_mask_inboxes)
# fg_mask_clone = fg_mask.clone()
# index = L.where(fg_mask_clone > 0)[:, 0]
# index2 = L.where(fg_mask_inboxes > 0)[:, 0]
# pos_index = L.gather(index, index2)
# fg_mask = paddle.zeros_like(fg_mask_clone)
# for iiiiii in pos_index:
#     fg_mask[iiiiii] = 1.0


# 方法三，用scatter()
fg_mask = paddle.to_tensor(fg_mask)
fg_mask_inboxes = paddle.to_tensor(fg_mask_inboxes)
fg_mask_clone = fg_mask.clone()
index = paddle.nonzero(fg_mask_clone > 0)[:, 0]
fg_mask = paddle.scatter(fg_mask, index, fg_mask_inboxes, overwrite=True)

print(fg_mask)









