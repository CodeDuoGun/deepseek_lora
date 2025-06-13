from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

# 步骤 1：加载 PEFT 配置（LoRA 模型保存路径）
peft_model_path = "output/deepseek-mutil-test"
peft_config = PeftConfig.from_pretrained(peft_model_path)

# 步骤 2：加载原始（基础）模型
base_model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path)

# 步骤 3：将 LoRA 适配器加载进基础模型
model = PeftModel.from_pretrained(base_model, peft_model_path)

# 步骤 4：加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)


prompt = "你是谁？"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
