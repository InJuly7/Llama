import torch

# 示例参数
tgt_len = 4
dtype = torch.float32
device = "cpu"

# 方式1：原始代码（使用零维张量作为填充值）
fill_value = torch.tensor(torch.finfo(dtype).min, device=device)
print(f"填充值的形状: {fill_value.shape}")  # torch.Size([])，零维！
print(f"填充值的值: {fill_value}")

mask = torch.full((tgt_len, tgt_len), fill_value, device=device)
print(f"\nmask的形状: {mask.shape}")
print(f"mask:\n{mask}")

# 方式2：更简洁的写法（直接用Python标量）
mask_simple = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
print(f"\n简化版mask:\n{mask_simple}")
print(f"\n两者相等: {torch.equal(mask, mask_simple)}")

"""
Output Log: 
填充值的形状: torch.Size([])
填充值的值: -3.4028234663852886e+38

mask的形状: torch.Size([4, 4])
mask:
tensor([[-3.4028e+38, -3.4028e+38, -3.4028e+38, -3.4028e+38],
        [-3.4028e+38, -3.4028e+38, -3.4028e+38, -3.4028e+38],
        [-3.4028e+38, -3.4028e+38, -3.4028e+38, -3.4028e+38],
        [-3.4028e+38, -3.4028e+38, -3.4028e+38, -3.4028e+38]])

简化版mask:
tensor([[-3.4028e+38, -3.4028e+38, -3.4028e+38, -3.4028e+38],
        [-3.4028e+38, -3.4028e+38, -3.4028e+38, -3.4028e+38],
        [-3.4028e+38, -3.4028e+38, -3.4028e+38, -3.4028e+38],
        [-3.4028e+38, -3.4028e+38, -3.4028e+38, -3.4028e+38]])

两者相等: True
"""
