from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
import json

# 加载你本地的数据文件
def load_json_dataset(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Dataset.from_list(data)

# 拼接 prompt 模板
def preprocess(example):
    prompt = f"患者：{example['instruction']}\n{example['input']}\n诊断："
    full_text = prompt + example["output"]
    return tokenizer(
        full_text,
        max_length=2048,
        truncation=True,
        padding="max_length"
    )

# 使用你的 Qwen tokenizer
model_name = "Qwen/Qwen2.5-32B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# 加载和处理数据集
data_path = "medical/data/medical_case.json"
dataset = load_json_dataset(data_path)
tokenized_dataset = dataset.map(preprocess, batched=False)

import torch
from transformers import (
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import get_peft_model, LoraConfig, TaskType

# 加载模型
model_kwargs = {
        "torch_dtype": torch.float16,
        # "use_cache": True,
        "trust_remote_code": True,
        "device_map":"cuda:0" if torch.cuda.is_available() else "cpu",
        "quantization_config": None,
    }

model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

# LoRA 配置
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # 根据Qwen结构定义
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)

# 数据整理器
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 训练参数
training_args = TrainingArguments(
    output_dir="./qwen32b_medical_lora",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=100,
    fp16=True,
    learning_rate=2e-4,
    save_total_limit=2,
    report_to=None,
    remove_unused_columns=False
)
### 5、设置可视化工具
import swanlab
from swanlab.integration.transformers import SwanLabCallback
import os

swanlab.login(api_key="F6n03ZzhpVx69ng6uXJwD", save=True)
os.environ["SWANLAB_API_HOST"] = "https://swanlab.115.zone/api"
os.environ["SWANLAB_WEB_HOST"] = "https://swanlab.115.zone"
swanlab_config = {
        "dataset": data_path,
        "peft":"lora"
    }
swanlab_callback = SwanLabCallback(
    project="tcm-finetune-test",
    experiment_name="0701",
    description="微调中医大模型",
    workspace=None,
    config=swanlab_config,
)

### 6、设置训练器参数+训练
from peft import get_peft_model
from transformers import Trainer

# 用于确保模型的词嵌入层参与训练
model.enable_input_require_grads()
# 应用 PEFT 配置到模型
model = get_peft_model(model,lora_config)
model.print_trainable_parameters()

trainer = Trainer(
    model=model,
    args=training_args,
    device="cuda" if torch.cuda.is_available() else "cpu",
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[swanlab_callback]
)

trainer.train()
trainer.save_model("./qwen32b_medical_lora")
tokenizer.save_pretrained("./qwen32b_medical_lora")

# release memory
del model
del trainer
del tokenizer

torch.cuda.empty_cache()
# 推理
