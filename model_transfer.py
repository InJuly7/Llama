import torch
from safetensors.torch import load_file

# 从 safetensors 加载
state_dict = load_file("/root/autodl-tmp/model/TinyLlama-1.1B-Chat-v1.0/model.safetensors")

# 保存为 .bin 格式
torch.save(state_dict, "/root/autodl-tmp/model/TinyLlama-1.1B-Chat-v1.0/model.bin")
print("转换完成！")
