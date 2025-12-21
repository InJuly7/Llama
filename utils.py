import torch
from torch import nn
from typing import Optional, List, Union
from dataclasses import dataclass


class DynamicCache:
    """动态缓存 - 每次都拼接, RoPE 在外部应用"""

    def __init__(self):
        self.key_cache = []
        self.value_cache = []

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if layer_idx >= len(self.key_cache):
            # 第一次，直接存储
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            # 拼接
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx=0):
        """获取当前缓存的序列长度"""
        if len(self.key_cache) <= layer_idx:
            return 0
        return self.key_cache[layer_idx].shape[2]


class SiLUActivation(nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return nn.functional.silu(input)


def _make_causal_mask(input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0):
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min, device=device), device=device)  # [tgt_len, tgt_len], -inf
    mask_cond = torch.arange(mask.size(-1), device=device)  # [tgt_len]
    # 左上顶点下三角矩阵置0
    # 比较时候广播: [tgt_len] , [tgt_len, 1] ==> [tgt_len, tgt_len]
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)  # (通过 view 实现) (tgt_len, 1)
    mask = mask.to(dtype)

    # Prefill 阶段的续写
    # chunk processing, 长文本分块处理
    if past_key_values_length > 0:
        # 构造右下顶点下三角矩阵置0 [tgt_len, past_key_values_length + tgt_len]
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)  # [B, 1, N1, N2]


# 多batch推理时序列长度对齐
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len
    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)  # [B,1,tgt_len,src_len]
    inverted_mask = 1.0 - expanded_mask  # [B,1,tgt_len,src_len]
    # 将padding位置(现在是1)填充为-inf
    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


def _prepare_decoder_attention_mask(
    attention_mask, input_shape, inputs_embeds=None, past_key_values_length=0, dtype=torch.float16, device="cuda"
):

    # Decode 阶段
    if past_key_values_length != 0:
        attention_mask = None
        return attention_mask

    if inputs_embeds is not None:
        dtype = inputs_embeds.dtype
        device = inputs_embeds.device

    # Prefill 阶段 create causal mask
    combined_attention_mask = None
    if input_shape[-1] > 1:
        combined_attention_mask = _make_causal_mask(input_shape, dtype, device, past_key_values_length=past_key_values_length)

    if attention_mask is not None:
        expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(inputs_embeds.device)
        combined_attention_mask = expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask

    return combined_attention_mask


@dataclass
class BaseModelOutputWithPast:
    last_hidden_state: Optional[torch.FloatTensor] = None
    # past_key_values: Optional[Cache] = None
    past_key_values: Optional[DynamicCache] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None


@dataclass
class CausalLMOutputWithPast:
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    # past_key_values: Optional[Cache] = None
    past_key_values: Optional[DynamicCache] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None


class StoppingCriteria:
    def __init__(self, max_length: int, eos_token_id: Union[int, List[int]]):
        self.max_length = max_length
        self.eos_token_id = eos_token_id if isinstance(eos_token_id, list) else [eos_token_id]

    def __call__(self, input_ids: torch.LongTensor, **kwargs) -> torch.BoolTensor:
        batch_size = input_ids.shape[0]
        is_done = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)

        # 检查长度
        length_done = input_ids.shape[1] >= self.max_length

        # 检查 EOS token (最后一个生成的 token)
        last_tokens = input_ids[:, -1]
        eos_done = torch.isin(last_tokens, torch.tensor(self.eos_token_id, device=input_ids.device))

        # 任一条件满足即停止
        is_done = length_done | eos_done

        return is_done
