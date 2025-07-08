from peft import PeftModel
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen3-8B", trust_remote_code=True, torch_dtype=torch.float16, device_map="auto"
)

model = PeftModel.from_pretrained(base_model, "medical/output/qwen3_lora", device_map="auto")
model = model.merge_and_unload()  # 合并 LoRA 权重

model.save_pretrained("medical/output/qwen3_lora_merged", safe_serialization=True, max_shard_size="2GB")

# 保存 tokenizer（非常重要）
tokenizer = AutoTokenizer.from_pretrained("Qwen3-8B", local_files_only=True, trust_remote_code=True)
tokenizer.save_pretrained("medical/output/qwen3_lora_merged")