from config import Llama_7B_Config, Llama2_7B_Config, TinyLlama_Config
import torch
from torch import nn
from typing import Optional, Tuple, List, Callable
from dataclasses import dataclass
from torch_lib import create_tensor


class LlamaRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: Llama2_7B_Config, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config

        self.rope_type = self.config.rope_parameters["rope_type"]
        rope_init_fn: Callable = self.compute_default_rope_parameters
        # if self.rope_type != "default":
        #     rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = rope_init_fn(self.config, device)
        # 不会被保存到 checkpoint 中
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # 备份
        self.original_inv_freq = inv_freq

    @staticmethod
    def compute_default_rope_parameters(
        config: Optional[Llama2_7B_Config] = None,
        device: Optional["torch.device"] = None,
        seq_len: Optional[int] = None,
    ) -> tuple["torch.Tensor", float]:

        base = config.rope_parameters["rope_theta"]
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads

        attention_factor = 1.0  # Unused in this type of RoPE

        # Compute the inverse frequencies
        # 偶数索引, 对应RoPE中的维度分组 (每对维度共享一个频率)
        # \theta = (10000)^{-2i/d} i = 0,1,2,...,d/2-1
        # dim = 4096/32 = 128
        # inv_freq: [64]
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
        return inv_freq, attention_factor

    @torch.no_grad()
    # @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        # 扩展 inv_freq 到与输入兼容的形状
        # inv_freq: [64], inv_freq_expanded: [1, 64, 1]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)

        # position_ids: [1, seq_len], position_ids_expanded: [1, 1, seq_len]
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            # freqs: [1, seq_len, 64]
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            # emb: [1, seq_len, 128]
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def test_llama_rope():
    config = Llama2_7B_Config()
    rope = LlamaRotaryEmbedding(config=config, device=config.device)
    if config.torch_dtype == "float16":
        dtype = torch.float16
    elif config.torch_dtype == "bfloat16":
        dtype = torch.bfloat16
    rope.eval()
    position_ids = torch.arange(config.max_position_embeddings, dtype=torch.long, device=config.device).unsqueeze(0)
    x = create_tensor((1, config.max_position_embeddings, config.hidden_size), dtype=dtype, ndim=3, device=config.device)
    cos, sin = rope.forward(x, position_ids)
    print(f"Cos shape: {cos.shape}")
    print(f"Sin shape: {sin.shape}")


if __name__ == "__main__":
    test_llama_rope()
