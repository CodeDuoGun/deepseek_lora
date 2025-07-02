### 示例数据集加载代码：
import json
from datasets import Dataset
### 5、设置可视化工具
from transformers import DataCollatorForSeq2Seq
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
import swanlab
from swanlab.integration.transformers import SwanLabCallback
import os

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

swanlab.login(api_key="F6n03ZzhpVx69ng6uXJwD", save=True)
os.environ["SWANLAB_API_HOST"] = "https://swanlab.115.zone/api"
os.environ["SWANLAB_WEB_HOST"] = "https://swanlab.115.zone"
swanlab_config = {
        "dataset": "medical/data/medical_case.json",
        "peft":"lora"
    }
swanlab_callback = SwanLabCallback(
    project="tcm-finetune-test",
    experiment_name="0701",
    description="微调中医大模型",
    workspace=None,
    config=swanlab_config,
)
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_alpaca_dataset(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    print(type(data), len(data), data[0])
    # import pdb
    # pdb.set_trace()
    return Dataset.from_list(data)

    formatted = []
    for item in data:
        prompt = item["instruction"]
        if item["input"]:
            prompt += "\n\n" + item["input"]
        formatted.append({
            "prompt": prompt,
            "response": item["output"]
        })
    
    return Dataset.from_list(formatted)


## 🔧 三、构造 SFT 格式（用于 Qwen）
def format_for_qwen(example):
    return {
        "text": f"<|im_start|>user\n{example['prompt']}<|im_end|>\n<|im_start|>assistant\n{example['response']}<|im_end|>"
    }

dataset = load_alpaca_dataset("medical/data/medical_case.json")


## 🧩 四、准备 Qwen 模型与 LoRA 微调配置

### ✅ 加载 Qwen-7B 和 LoRA 配置
model_name = "Qwen/Qwen2.5-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name, trust_remote_code=True
)
print("分词器：", tokenizer)

# model = prepare_model_for_kbit_training(model)

# LoRA配置
lora_config = LoraConfig(
        r=64,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=['up_proj', 'gate_proj', 'q_proj', 'o_proj', 'down_proj', 'v_proj', 'k_proj'],
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False  # 训练模式
    )
model = get_peft_model(model, lora_config)
model.enable_input_require_grads()
# 应用 PEFT 配置到模型
model.print_trainable_parameters()

## 📦 五、tokenize 数据集
def process_data(data, tokenizer, max_seq_length):
    input_ids, attention_mask, labels = [], [], []

    # conversations = data["conversation"]
    instruction_text = data["instruction"]
    human_text = data["input"]
    assistant_text = data["output"]

    input_text = f"{tokenizer.bos_token}{instruction_text}\ninput:{human_text}\nResponse:"

    input_tokenizer = tokenizer(
        input_text,
        add_special_tokens=False,
        truncation=True,
        padding=False,
        return_tensors=None,
    )
    output_tokenizer = tokenizer(
        assistant_text,
        add_special_tokens=False,
        truncation=True,
        padding=False,
        return_tensors=None,
    )
    input_ids += (
            input_tokenizer["input_ids"] + output_tokenizer["input_ids"] + [tokenizer.eos_token_id]
    )
    attention_mask += input_tokenizer["attention_mask"] + output_tokenizer["attention_mask"] + [1]
    labels += ([-100] * len(input_tokenizer["input_ids"]) + output_tokenizer["input_ids"] + [tokenizer.eos_token_id]
                )

    if len(input_ids) > max_seq_length:  # 做一个截断
        input_ids = input_ids[:max_seq_length]
        attention_mask = attention_mask[:max_seq_length]
        labels = labels[:max_seq_length]
    if not isinstance(input_ids[0], int) or not isinstance(attention_mask[0], int) or not isinstance(labels[0], int):
    
        print({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }) 
        import pdb
        pdb.set_trace()
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

tokenized_dataset = dataset.map(process_data, fn_kwargs={"tokenizer": tokenizer, "max_seq_length": tokenizer.model_max_length},remove_columns=dataset.column_names, batched=False)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, return_tensors="pt")
## 🚀 七、训练参数与 Trainer 启动
training_args = TrainingArguments(
    output_dir="./qwen7b_lora_output",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    logging_steps=1,
    num_train_epochs=3,
    save_steps=5000,
    learning_rate=2e-5,
    save_on_each_node=True,
    gradient_checkpointing=True,
    report_to=None,
    seed=42,
    optim="adamw_torch",
    fp16=True,
    bf16=False,
    remove_unused_columns=False,
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
    callbacks=[swanlab_callback],
    # tokenizer=tokenizer
)

trainer.train()


## ✅ 八、保存模型
trainer.save_model("./qwen7b_lora_final")
tokenizer.save_pretrained("./qwen7b_lora_final")
## 🧪 九、推理示例

# from transformers import pipeline

# pipe = pipeline("text-generation", model="./qwen7b_lora_final", tokenizer=tokenizer, device=0)
# result = pipe("<|im_start|>user\n请解释什么是人工智能？<|im_end|>\n<|im_start|>assistant\n", max_new_tokens=200)
# print(result[0]["generated_text"])
