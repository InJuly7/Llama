from config import Llama_7B_Config, Llama2_7B_Config, TinyLlama_Config
import torch
from torch import nn
from typing import Optional, Tuple, List
from dataclasses import dataclass


class SiLUActivation(nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return nn.functional.silu(input)


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = SiLUActivation()

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def test_llama_mlp():
    config = Llama2_7B_Config()
    mlp = LlamaMLP(config)
    x = torch.randn(2, 4, config.hidden_size)
    output = mlp(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    test_llama_mlp()

"""
Output Log:
Input shape: torch.Size([2, 4, 4096])
Output shape: torch.Size([2, 4, 4096])
"""
