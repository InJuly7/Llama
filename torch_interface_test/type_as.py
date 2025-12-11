import torch

# 示例1: 基本用法
tensor_a = torch.tensor([1.0, 2.0, 3.0])  # float32
tensor_b = torch.tensor([1, 2, 3], dtype=torch.int64)  # int64

print(f"tensor_a 类型: {tensor_a.dtype}")  # torch.float32
print(f"tensor_b 类型: {tensor_b.dtype}")  # torch.int64

# 将 tensor_a 转换为和 tensor_b 相同的类型
tensor_a_converted = tensor_a.type_as(tensor_b)
print(f"转换后的类型: {tensor_a_converted.dtype}")  # torch.int64
print(f"转换后的值: {tensor_a_converted}")  # tensor([1, 2, 3])

print("\n" + "=" * 50 + "\n")

# 示例2: 在深度学习中的典型应用
model_weight = torch.randn(3, 3).half()  # float16
input_data = torch.randn(3, 3)  # float32

print(f"模型权重类型: {model_weight.dtype}")  # torch.float16
print(f"输入数据类型: {input_data.dtype}")  # torch.float32

# 将输入转换为与模型相同的精度
input_data = input_data.type_as(model_weight)
print(f"转换后输入类型: {input_data.dtype}")  # torch.float16

print("\n" + "=" * 50 + "\n")

# 示例3: 在不同设备间的应用
if torch.cuda.is_available():
    tensor_cpu = torch.tensor([1.0, 2.0, 3.0])
    tensor_gpu = torch.tensor([4.0, 5.0, 6.0]).cuda()

    # type_as 也会处理设备转换
    tensor_cpu_to_gpu = tensor_cpu.type_as(tensor_gpu)
    print(f"原始设备: {tensor_cpu.device}")  # cpu
    print(f"目标设备: {tensor_gpu.device}")  # cuda:0
    print(f"转换后设备: {tensor_cpu_to_gpu.device}")  # cuda:0


"""
Output Log:
tensor_a 类型: torch.float32
tensor_b 类型: torch.int64
转换后的类型: torch.int64
转换后的值: tensor([1, 2, 3])

==================================================

模型权重类型: torch.float16
输入数据类型: torch.float32
转换后输入类型: torch.float16

==================================================

原始设备: cpu
目标设备: cuda:0
转换后设备: cuda:0
"""
