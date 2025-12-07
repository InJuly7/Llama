import torch

"""
@func: 创建一个 torch Tensor
@param: shape: 形状
@param: dtype: 数据类型
@param: ndim: 维度数量
@param: device: 设备名称
@return: torch.Tensor
"""


def create_tensor(shape, dtype=torch.float32, ndim=2, device="cpu"):
    if ndim == 1:
        return torch.randn((shape[0]), device=device).to(dtype).contiguous()
    elif ndim == 2:
        return torch.randn((shape[0], shape[1]), device=device).to(dtype).contiguous()
    elif ndim == 3:
        return torch.randn((shape[0], shape[1], shape[2]), device=device).to(dtype).contiguous()
    else:
        raise ValueError("Unsupported ndim")


"""
@func: 比较两个tensor是否相等
@param: A : 第一个tensor
@param: B : 第二个tensor
@return: bool
"""


# ∣A − B∣ ≤ atol + (rtol × ∣B∣)
def compare_tensor(A: torch.Tensor, B: torch.Tensor, dtype: torch.dtype, rtol: float, atol: float) -> bool:

    # 检查形状
    if A.shape != B.shape:
        print(f"Shape mismatch: {A.shape} vs {B.shape}")
        return False

    # 使用 torch.isclose 生成逐元素的对比掩码, is_close 是一个全是 True/False 的 Tensor
    if A.dtype == torch.float16:
        is_close = torch.isclose(A, B, rtol=rtol, atol=atol)
    elif A.dtype == torch.float32:
        is_close = torch.isclose(A, B, rtol=rtol, atol=atol)

    # 如果全部都 Close，则通过
    if torch.all(is_close):
        return True

    # 定位错误
    print("Tensor values are not close enough.")

    # 找到不一致的位置 (False 的位置)
    mismatch_indices = torch.nonzero(~is_close, as_tuple=False)
    num_mismatches = mismatch_indices.shape[0]
    total_elements = A.numel()

    print(f"   Mismatched elements: {num_mismatches} / {total_elements} ({(num_mismatches/total_elements)*100:.2f}%)")

    # 计算最大绝对误差
    diff = torch.abs(A - B)
    max_diff = torch.max(diff)
    print(f"   Max absolute difference: {max_diff.item()}")

    # 4. 打印前 N 个具体的错误位置供调试
    print("\n   --- First 5 Mismatches ---")
    for i in range(min(5, num_mismatches)):
        idx = mismatch_indices[i]  # 获取由维度组成的索引，如 [0, 2, 1]

        # 将 tensor 索引转为 tuple 以便用于访问
        idx_tuple = tuple(idx.tolist())

        val_a = A[idx_tuple].item()
        val_b = B[idx_tuple].item()
        abs_err = abs(val_a - val_b)

        print(f"   Index {idx_tuple}:")
        print(f"     A: {val_a}")
        print(f"     B: {val_b}")
        print(f"     Diff: {abs_err}")

    return False
