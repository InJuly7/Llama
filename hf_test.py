import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.cuda.empty_cache()

# 模型路径
model_path = "/root/autodl-tmp/model/TinyLlama-1.1B-Chat-v1.0/"

# 设备
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载模型和 tokenizer
print(f"Loading model... on {device}")
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 设置 pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=torch.float16,
    low_cpu_mem_usage=True,
)
model.to(device)

print(f"Model loaded on {device}")

# 使用正确的对话格式
user_message = "可以给我讲一个故事吗?"

# TinyLlama 的对话模板格式
prompt = f"""<|system|>
You are a friendly chatbot.</s>
<|user|>
{user_message}</s>
<|assistant|>
"""

# Tokenize
inputs = tokenizer(prompt, return_tensors="pt").to(device)
for k, v in inputs.items():
    print(f"{k}: {v.shape}")
print(f"input_ids: {inputs['input_ids']}")
print(f"attention_mask: {inputs['attention_mask']}")
batch_size, seq_len = inputs["input_ids"].shape

# Greedy Decoding 每次选择概率最高的 token
print("Generating...")
outputs = model.generate(
    **inputs,
    max_new_tokens=50,
    eos_token_id=tokenizer.eos_token_id,
)
print(f"Output token {outputs[0].shape}")
print(f"Output IDs: {outputs[0][seq_len+1:]}")

# 解码输出
result = tokenizer.decode(outputs[0], skip_special_tokens=False)
print(result)


"""
Output Log:
Loading model... on cuda
The module name  (originally ) is not a valid Python identifier. Please rename the original module to avoid import issues.
Model loaded on cuda
input_ids: torch.Size([1, 48])
attention_mask: torch.Size([1, 48])
input_ids: tensor([[    1,   529, 29989,  5205, 29989, 29958,    13,  3492,   526,   263,
         19780, 13563,  7451, 29889,     2, 29871,    13, 29966, 29989,  1792,
         29989, 29958,    13, 30682, 30651, 31999, 30672,   235,   177,   181,
         30287, 30502, 31969, 30745,   232,   147,   154, 29973,     2, 29871,
            13, 29966, 29989,   465, 22137, 29989, 29958,    13]],
       device='cuda:0')
attention_mask: tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
       device='cuda:0')
Generating...
Output token torch.Size([98])
Output IDs: tensor([  545, 29892,  1244, 29915, 29879,   263,  5828,   363,   366, 29901,
           13,    13, 26222,  2501,   263,   931, 29892,   727,   471,   263,
         4123,  7826,  4257,   365,  2354, 29889,  2296, 10600,   297,   263,
         2319,  5720,   411,   902, 11825,   322,  1023, 20023, 27767, 18964,
        29889,   365,  2354, 18012,  8743,   411,   902, 27767, 18964],
       device='cuda:0')
<s> <|system|>
You are a friendly chatbot.</s> 
<|user|>
可以给我讲一个故事吗?</s> 
<|assistant|>
Sure, here's a story for you:

Once upon a time, there was a young girl named Lily. She lived in a small village with her parents and two younger siblings. Lily loved playing with her siblings
"""
