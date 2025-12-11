import torch
import math

# ========== 基础示例 ==========
# torch.polar(abs, angle) 创建复数
# abs: 模长（半径）
# angle: 角度（弧度）
# 返回: abs * e^(i*angle) = abs * (cos(angle) + i*sin(angle))

# 示例1：创建单个复数
abs_val = torch.tensor([1.0])  # 模长为1
angle = torch.tensor([math.pi / 4])  # 角度为45度

complex_num = torch.polar(abs_val, angle)
print(f"极坐标(r=1, θ=π/4)的复数: {complex_num}")
print(f"实部: {complex_num.real}, 虚部: {complex_num.imag}")
# 输出: 0.707 + 0.707i （即 √2/2 + √2/2 i）

# ========== 你的代码含义 ==========
# freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
# 这相当于：cis(θ) = e^(iθ) = cos(θ) + i*sin(θ)

# freqs = torch.randn((4,8))
freqs = torch.tensor([0, math.pi / 2, math.pi, 3 * math.pi / 2])
print(f"freqs: {freqs}")
print(f"freqs shape: {freqs.shape}")

freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
print(f"freqs_cis: {freqs_cis}")
print(f"freqs_cis shape: {freqs_cis.shape}")

print("\n角度 -> 复数转换:")
for i, (freq, cis) in enumerate(zip(freqs, freqs_cis)):
    print(f"θ={freq:.2f} -> {cis.real:.2f} + {cis.imag:.2f}i")

"""
Output Log:
极坐标(r=1, θ=π/4)的复数: tensor([0.7071+0.7071j])
实部: tensor([0.7071]), 虚部: tensor([0.7071])
freqs: tensor([0.0000, 1.5708, 3.1416, 4.7124])
freqs shape: torch.Size([4])
freqs_cis: tensor([ 1.0000e+00+0.0000e+00j, -4.3711e-08+1.0000e+00j,
        -1.0000e+00-8.7423e-08j,  1.1925e-08-1.0000e+00j])
freqs_cis shape: torch.Size([4])

角度 -> 复数转换:
θ=0.00 -> 1.00 + 0.00i
θ=1.57 -> -0.00 + 1.00i
θ=3.14 -> -1.00 + -0.00i
θ=4.71 -> 0.00 + -1.00i
"""
