import sys
import torch
from safetensors import safe_open
from pathlib import Path


def analyze_safetensors(file_path):
    """分析 safetensors 文件"""
    print(f"Analyzing: {file_path}\n")
    total_params = 0
    total_size = 0

    with safe_open(file_path, framework="pt", device="cpu") as f:
        print(f"{'Layer Name':<50} {'Shape':<25} {'Dtype':<10} {'Size (MB)':<12}")
        print("=" * 100)

        for key in f.keys():
            tensor = f.get_tensor(key)
            params = tensor.numel()
            size_mb = tensor.element_size() * params / 1024 / 1024

            total_params += params
            total_size += size_mb

            print(f"{key:<50} {str(tuple(tensor.shape)):<25} {str(tensor.dtype):<10} {size_mb:>10.2f}")

    print("=" * 100)
    print(f"Total Parameters: {total_params:,}")
    print(f"Total Size: {total_size:.2f} MB ({total_size/1024:.2f} GB)")


def analyze_pytorch(file_path):
    """分析 pytorch.bin 文件"""
    print(f"Analyzing: {file_path}\n")
    state_dict = torch.load(file_path, map_location="cpu")

    total_params = 0
    total_size = 0

    print(f"{'Layer Name':<50} {'Shape':<25} {'Dtype':<10} {'Size (MB)':<12}")
    print("=" * 100)

    for key, tensor in state_dict.items():
        params = tensor.numel()
        size_mb = tensor.element_size() * params / 1024 / 1024

        total_params += params
        total_size += size_mb

        print(f"{key:<50} {str(tuple(tensor.shape)):<25} {str(tensor.dtype):<10} {size_mb:>10.2f}")

    print("=" * 100)
    print(f"Total Parameters: {total_params:,}")
    print(f"Total Size: {total_size:.2f} MB ({total_size/1024:.2f} GB)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_model.py <model_file>")
        sys.exit(1)

    file_path = sys.argv[1]

    if file_path.endswith(".safetensors"):
        analyze_safetensors(file_path)
    elif file_path.endswith(".bin"):
        analyze_pytorch(file_path)
    else:
        print("Unsupported file format. Please use .safetensors or .bin files.")
