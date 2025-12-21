import torch
import torch.nn as nn
from dataclasses import dataclass

from config import TinyLlama_Config
from LlamaModel import LlamaModel
from utils import DynamicCache, StoppingCriteria
from utils import BaseModelOutputWithPast, CausalLMOutputWithPast
from utils import _prepare_decoder_attention_mask
from typing import Callable, Optional, Union, Any


class LlamaForCausalLM(nn.Module):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__()
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

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
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:

        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # logits_to_keep 变量用于指定需要保留/计算的logits数量
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        # 只需要计算最后 logits_to_keep 个 token 的 logits
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        # if labels is not None:
        #     loss = self.loss_function(
        #         logits=logits,
        #         labels=labels,
        #         vocab_size=self.config.vocab_size,
        #         **kwargs,
        #     )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def prepare_inputs_for_generation(
    input_ids: torch.LongTensor,
    # past_key_values: Optional[Cache] = None,
    past_key_values: Optional[DynamicCache] = None,
    attention_mask: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
):
    model_inputs = {}
    past_key_values = DynamicCache()
    model_inputs["past_key_values"] = past_key_values
    model_inputs["input_ids"] = input_ids

    input_shape = input_ids.shape  # [B, S]
    model_inputs["attention_mask"] = _prepare_decoder_attention_mask(attention_mask, input_shape, past_key_values_length=0)

    (batch_size, seq_len) = input_ids.shape
    model_inputs["batch_size"] = batch_size
    model_inputs["seq_len"] = seq_len
    model_inputs["cache_position"] = torch.arange(seq_len, device=input_ids.device)
    model_inputs["position_ids"] = model_inputs["cache_position"].unsqueeze(0).expand(batch_size, -1)
    model_inputs["use_cache"] = True
    model_inputs["is_prefill"] = True
    model_inputs["device"] = "cuda"
    model_inputs["dtype"] = torch.float16
    return model_inputs


def _update_model_kwargs_for_generation(
    outputs: CausalLMOutputWithPast,
    model_inputs: dict[str, Any],
    new_token_ids: torch.LongTensor,  # [B, 1] 传入新生成的 token id
) -> dict[str, Any]:
    model_inputs["seq_len"] = 1
    past_key_value_len = model_inputs["past_key_values"].get_seq_length()
    model_inputs["cache_position"] = torch.arange(
        past_key_value_len, past_key_value_len + 1, dtype=torch.long, device=model_inputs["device"]
    )  # [1]
    model_inputs["position_ids"] = model_inputs["cache_position"].unsqueeze(0).expand(model_inputs["batch_size"], -1)  # [B, 1]
    model_inputs["attention_mask"] = None  # Decode phase 不需要 attention_mask
    model_inputs["input_ids"] = new_token_ids  # [B, 1]
    return model_inputs


def _sample(
    model: LlamaForCausalLM,
    input_ids: torch.LongTensor,
    stopping_criteria: StoppingCriteria,
    config: TinyLlama_Config,
) -> torch.LongTensor:
    pad_token_id = config.pad_token_id  # 填充 token 的 ID
    # 检查停止条件中是否有 EOS token 判断
    has_eos_stopping_criteria = hasattr(stopping_criteria, "eos_token_id")
    do_sample = config.do_sample  # True 表示采样,False 表示贪婪搜索

    batch_size, cur_len = input_ids.shape
    this_peer_finished = False
    # 所有序列都是"未完成"状态 (1:未完成)
    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
    # prepare model inputs
    model_inputs = prepare_inputs_for_generation(input_ids)
    while not this_peer_finished:

        if model_inputs.get("is_prefill", True):
            outputs = model.forward(**model_inputs)
            model_inputs["is_prefill"] = False
        else:
            outputs = model.forward(**model_inputs)

        next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)  # [B, S, H]
        # 选择下一个token
        next_tokens = torch.argmax(next_token_logits, dim=-1)  # [B]
        del next_token_logits

        # 处理已完成序列
        if has_eos_stopping_criteria:
            # 如果某个序列已完成，就用 pad_token 填充，不再生成新token
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # 更新输入序列
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        model_inputs = _update_model_kwargs_for_generation(
            outputs,
            model_inputs,
            next_tokens[:, None],  # [B, 1]
        )

        # 检查终止条件
        # stopping_criteria 0:未终止, 1: 终止, 更新 unfinished_sequences
        unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids)
        # 检查是否所有序列都已完成
        this_peer_finished = unfinished_sequences.max() == 0
        # print(f"Current generated sequence: {input_ids.shape}")
        # 这是为了正确删除 outputs.logits，在第一次迭代时它可能非常大
        # 否则会保留对 outputs 的引用，导致 logits 在下一次迭代中仍然占用内存
        del outputs

    return input_ids


def test_llama_for_causal_lm():
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
    print(f"Input IDs: {input_ids}")
    print(f"Input IDs Shape: {input_ids.shape}")
    batch_size, seq_len = input_ids.shape
    torch.cuda.empty_cache()
    state_dict = torch.load("/root/autodl-tmp/model/TinyLlama-1.1B-Chat-v1.0/model.bin", map_location="cuda:0")

    model = LlamaForCausalLM(TinyLlama_Config()).to(device="cuda:0", dtype=torch.float16)
    model.load_state_dict(state_dict)
    model.eval()
    max_newtoken = 50
    max_length = 40960
    with torch.no_grad():
        output_ids = _sample(
            model=model,
            input_ids=input_ids,
            stopping_criteria=StoppingCriteria(max_length=max_length, eos_token_id=2),
            config=TinyLlama_Config(),
        )
    print(f"Output token {output_ids[:, seq_len+1:]}")
    print(f"Output token Shape {output_ids.shape}")


if __name__ == "__main__":
    test_llama_for_causal_lm()

"""
Output Log:
Input IDs: tensor([[    1,   529, 29989,  5205, 29989, 29958,    13,  3492,   526,   263,
         19780, 13563,  7451, 29889,     2, 29871,    13, 29966, 29989,  1792,
         29989, 29958,    13, 30682, 30651, 31999, 30672,   235,   177,   181,
         30287, 30502, 31969, 30745,   232,   147,   154, 29973,     2, 29871,
            13, 29966, 29989,   465, 22137, 29989, 29958,    13]],
       device='cuda:0')
Input IDs Shape: torch.Size([1, 48])
Output token tensor([[  545, 29892,  1244, 29915, 29879,   263,  5828,   363,   366, 29901,
            13,    13, 26222,  2501,   263,   931, 29892,   727,   471,   263,
          4123,  7826,  4257,   365,  2354, 29889,  2296, 10600,   297,   263,
          2319,  5720,   411,   902, 11825,   322,  1023, 20023, 27767, 18964,
         29889,   365,  2354, 18012,  8743,   411,   902, 27767, 18964]],
       device='cuda:0')
Output token Shape torch.Size([1, 98])
"""
