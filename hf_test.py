import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.cuda.empty_cache()

# 模型路径
model_path = "/home/song/llm_model/TinyLlama_1.1B_Chat_v1.0/"

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
).to(device)

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

# Greedy Decoding 每次选择概率最高的 token
print("Generating...")
outputs = model.generate(
    **inputs,
    max_new_tokens=50,
    eos_token_id=tokenizer.eos_token_id,
)
print(f"Output token {outputs[0].shape}")

# 解码输出
result = tokenizer.decode(outputs[0], skip_special_tokens=False)
print(result)
