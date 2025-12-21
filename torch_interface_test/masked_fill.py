import torch

# 假设我们有一个 4x4 的掩码矩阵
seq_len = 4
mask = torch.full((seq_len, seq_len), torch.tensor(torch.finfo(torch.float32).min), dtype=torch.float32)
mask_cond = torch.arange(seq_len)  # [0, 1, 2, 3]

print("初始 mask:")
print(mask)
print("\nmask_cond:", mask_cond)

# 通过广播比较:
#   行方向: [0, 1, 2, 3]
#   列方向: [[1], [2], [3], [4]] (通过 view 实现) (seg_len, 1)
mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)

print("\n应用 masked_fill_ 后:")
print(mask)

"""
Output Log:
初始 mask:
tensor([[-3.4028e+38, -3.4028e+38, -3.4028e+38, -3.4028e+38],
        [-3.4028e+38, -3.4028e+38, -3.4028e+38, -3.4028e+38],
        [-3.4028e+38, -3.4028e+38, -3.4028e+38, -3.4028e+38],
        [-3.4028e+38, -3.4028e+38, -3.4028e+38, -3.4028e+38]])

mask_cond: tensor([0, 1, 2, 3])

应用 masked_fill_ 后:
tensor([[ 0.0000e+00, -3.4028e+38, -3.4028e+38, -3.4028e+38],
        [ 0.0000e+00,  0.0000e+00, -3.4028e+38, -3.4028e+38],
        [ 0.0000e+00,  0.0000e+00,  0.0000e+00, -3.4028e+38],
        [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00]])
"""
