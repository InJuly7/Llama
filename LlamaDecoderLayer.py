import torch
from torch import nn
from typing import Optional, Tuple, List
from dataclasses import dataclass

from config import Llama_7B_Config, Llama2_7B_Config, TinyLlama_Config
from LlamaMLP import LlamaMLP
from LlamaAttention import LlamaAttention, _prepare_decoder_attention_mask
from LlamaAttention import DynamicCache as Cache
from LlamaRMSNorm import LlamaRMSNorm

class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: TinyLlama_Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states
    

def test_llama_decoder_layer():
    config = TinyLlama_Config()
    layer = LlamaDecoderLayer(config, layer_idx=0)

    # prefill 阶段
    batch_size = 2
    seq_length = 16
    hidden_size = config.hidden_size

    hidden_states = torch.randn(batch_size, seq_length, hidden_size) # [B, S, H]
    attention_mask = torch.ones(batch_size, 1, 1, seq_length)

    output = layer(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
    )

    assert output.shape == (batch_size, seq_length, hidden_size)
    print("LlamaDecoderLayer test passed!")

if __name__ == "__main__":
    test_llama_decoder_layer()