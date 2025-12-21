import torch
from torch import nn
from typing import Optional, Tuple, List


from config import Llama_7B_Config, Llama2_7B_Config, TinyLlama_Config
from LlamaMLP import LlamaMLP
from LlamaAttention import LlamaAttention
from LlamaRMSNorm import LlamaRMSNorm
from LlamaDecoderLayer import LlamaDecoderLayer
from LlamaRotaryEmbedding import LlamaRotaryEmbedding


from utils import DynamicCache
from utils import _prepare_decoder_attention_mask
from utils import BaseModelOutputWithPast


class LlamaModel(nn.Module):
    # nn.Module，实现了 __call__ 方法来调用 forward 方法。
    def __init__(self, config: TinyLlama_Config):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        # self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        # self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        # past_key_values: Optional[Cache] = None,
        past_key_values: Optional[DynamicCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            # past_key_values = DynamicCache(config=self.config)
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position: torch.Tensor = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # causal_mask = create_causal_mask(
        #     config=self.config,
        #     input_embeds=inputs_embeds,
        #     attention_mask=attention_mask,
        #     cache_position=cache_position,
        #     past_key_values=past_key_values,
        #     position_ids=position_ids,
        # )

        causal_mask = attention_mask

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


def test_llama_model():
    config = TinyLlama_Config()
    model = LlamaModel(config).to(device="cuda", dtype=torch.float16)
    model.eval()
    batch = 1
    seg_len = 5
    # input_ids = torch.randint(0, config.vocab_size, (batch, seg_len)).to(device="cuda")
    input_ids = torch.tensor(
        [
            [
                1,
                529,
                29989,
                5205,
                29989,
                29958,
                13,
                3492,
                526,
                263,
                19780,
                13563,
                7451,
                29889,
                2,
                29871,
                13,
                29966,
                29989,
                1792,
                29989,
                29958,
                13,
                30682,
                30651,
                31999,
                30672,
                235,
                177,
                181,
                30287,
                30502,
                31969,
                30745,
                232,
                147,
                154,
                29973,
                2,
                29871,
                13,
                29966,
                29989,
                465,
                22137,
                29989,
                29958,
                13,
            ]
        ],
        device="cuda:0",
    )
    print(f"Input IDs: {input_ids.shape}")
    attention_mask = None
    position_ids = None
    past_key_values = None
    inputs_embeds = None
    cache_position = None
    use_cache = True
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        cache_position=cache_position,
        use_cache=use_cache,
    )
    past_key_values = output.past_key_values
    last_hidden_state = output.last_hidden_state

    print(f"last_hidden_state shape: {last_hidden_state.shape}")
    print(f"KV Cache keys shape: {past_key_values.key_cache[0].shape}")  # 应该是 [B, num_heads, kv_len ,head_dim]
    print(f"KV Cache values shape: {past_key_values.value_cache[0].shape}")


if __name__ == "__main__":
    test_llama_model()

"""
Output Log:
Input IDs: torch.Size([1, 48])
last_hidden_state shape: torch.Size([1, 48, 2048])
KV Cache keys shape: torch.Size([1, 4, 48, 64])
KV Cache values shape: torch.Size([1, 4, 48, 64])
"""
