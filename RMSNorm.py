from config import Llama_7B_Config, Llama2_7B_Config, TinyLlama_Config
import torch
from torch import nn
from typing import Optional, Tuple, List
from dataclasses import dataclass
from torch_lib import create_tensor


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


def test_llama_rmsnorm():
    config = Llama2_7B_Config()
    rms_norm = LlamaRMSNorm(hidden_size=config.hidden_size, eps=config.rms_norm_eps)
    if config.torch_dtype == "float16":
        dtype = torch.float16
    elif config.torch_dtype == "bfloat16":
        dtype = torch.bfloat16

    rms_norm.weight.data = create_tensor((config.hidden_size,), dtype=dtype, ndim=1, device=config.device)
    rms_norm.eval()
    A = create_tensor((1, config.hidden_size), dtype=dtype, ndim=2, device=config.device)
    output = rms_norm(A)
    print(f"Input shape: {A.shape}")
    print(f"Output shape: {output.shape}")


# if __name__ == "__main__":
#     test_llama_rmsnorm()

"""
Output Log
Input shape: torch.Size([1, 4096])
Output shape: torch.Size([1, 4096])
"""
