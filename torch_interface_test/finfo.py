import torch

# ============= 示例 1: 基础用法 =============
print("=== 示例 1: 不同数据类型的最小值 ===")

# float32 的最小值
dtype = torch.float32
device = "cpu"
min_value_tensor = torch.tensor(torch.finfo(dtype).min, device=device)

print(f"float32 最小值: {torch.finfo(torch.float32).min}")
print(f"创建的tensor: {min_value_tensor}")
print(f"tensor数据类型: {min_value_tensor.dtype}\n")

# float16 的最小值
print(f"float16 最小值: {torch.finfo(torch.float16).min}")
print(f"float64 最小值: {torch.finfo(torch.float64).min}\n")


# ============= 示例 2: 完整的 finfo 信息 =============
print("=== 示例 2: finfo 包含的所有信息 ===")
finfo = torch.finfo(torch.float32)
print(f"数据类型: {finfo.dtype}")
print(f"最小值 (min): {finfo.min}")
print(f"最大值 (max): {finfo.max}")
print(f"最小正数 (tiny): {finfo.tiny}")
print(f"精度 (eps): {finfo.eps}")
print(f"位数 (bits): {finfo.bits}\n")

"""
Output Log:
=== 示例 1: 不同数据类型的最小值 ===
float32 最小值: -3.4028234663852886e+38
创建的tensor: -3.4028234663852886e+38
tensor数据类型: torch.float32

float16 最小值: -65504.0
float64 最小值: -1.7976931348623157e+308

=== 示例 2: finfo 包含的所有信息 ===
数据类型: float32
最小值 (min): -3.4028234663852886e+38
最大值 (max): 3.4028234663852886e+38
最小正数 (tiny): 1.1754943508222875e-38
精度 (eps): 1.1920928955078125e-07
位数 (bits): 32
"""
