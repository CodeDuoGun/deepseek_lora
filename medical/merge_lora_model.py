from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def merge_model():
    base_model_name = "Qwen3-8B"  # 原始模型名
    lora_path = "medical/output/qwen3_lora"  # LoRA adapter 输出路径

    # 1. 加载基础模型
    model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map="auto", local_files_only=True)

    # 2. 加载 LoRA 权重
    model = PeftModel.from_pretrained(model, lora_path)

    # 3. 合并 LoRA → 原始模型
    model = model.merge_and_unload()  # 🔥 核心一步：LoRA 融合

    # 2. 保存为新的 Huggingface 格式权重（合并后）,现在这个模型就不再依赖 LoRA Adapter，也不再依赖 peft 库加载，可以独立用于推理、部署到 WebUI 或 llama.cpp 之类环境。
    # 保存为 safetensors（部署更快更安全）, model.safetensors
    model.save_pretrained("medical/model/qwen3_8b_merged", safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    tokenizer.save_pretrained("medical/model/qwen3_8b_merged")
    
if __name__=="__main__":
    merge_model()