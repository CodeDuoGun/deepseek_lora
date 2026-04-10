# Introduction
supported features
- [x] deepseek lora sft 微调多轮对话
- [x] qwen3 lora sft 微调ai病历, 开发中


# ENV Install
```bash
conda create -n lora python=3.10
conda activate lora
pip install -r requirements.txt
```

# Dataset prepare
## TCM data

## conversation data

# Conv finetune

## run
```bash
python finetune-multi-conv.py
```

# TCM Instruction finetune
## 使用 qwen3.5_9b_base 在 `medical/data/tcm_data` 全量中医数据集微调（9:1）

### 1) 数据集准备（详细）

1、准备原始数据目录（已存在）：
- `medical/data/tcm_data/train.json`
- `medical/data/tcm_data/dev.json`
- `medical/data/tcm_data/medical_case.json`
- `medical/data/tcm_data/syndrome_knowledge.json`

2、执行数据构建脚本（自动聚合 + 去重 + 9:1 划分）：
```bash
python medical/prepare_sft_data.py \
  --input_dir medical/data/tcm_data \
  --output_dir medical/train_data \
  --train_ratio 0.9 \
  --seed 42
```

3、生成文件：
- `medical/train_data/tcm_sft_all.json`
- `medical/train_data/tcm_sft_train.json`
- `medical/train_data/tcm_sft_val.json`

4、单条样本格式（SFT）：
```json
{
  "instruction": "你是一名中医问诊助手...",
  "input": "主诉：...\n现病史：...\n四诊摘要：...",
  "output": "{\"reasoning_steps\":[...],\"result\":{...}}",
  "source": "medical_case.json"
}
```

### 2) 训练（参考 `finetune/supervised_finetuning.py` 的 SFT 方式）

1、运行 LoRA SFT：
```bash
python medical/finetune_qwen3.py \
  --model_path qwen3.5_9b_base \
  --train_file medical/train_data/tcm_sft_train.json \
  --val_file medical/train_data/tcm_sft_val.json \
  --output_dir medical/output/qwen3_lora
```

> 默认参数：`epochs=3`，`max_length=2048`，`lr=2e-5`，LoRA target modules 为 `q/k/v/o + mlp(up/down/gate)_proj`。

2、可选：如果模型是本地离线目录，增加：
```bash
--local_files_only
```

3、训练完成产物：
- `medical/output/qwen3_lora`（LoRA 适配器 + tokenizer）

### 3) 推理

1、inference without vllm
```bash
python medical/without_merge_inference.py
```

2、inference with vllm
```bash
# 启动vllm
python3 -m vllm.entrypoints.openai.api_server --model medical/output/qwen3_lora_merged --tokenizer Qwen3-8B --dtype float16
# 注意 tokenizer
python3 -m vllm.entrypoints.openai.api_server --model medical/output/qwen3_lora_merged --tokenizer medical/output/qwen3_lora_merged --trust-remote-code  --dtype float16

# 开始推理
python medical/with_vllm_inference.py
```

# Run openmind
```bash
python finetune-multi-openmind.py
```
# TODO
- [x] peft lora 微调
- [ ] 强化学习
- [ ] 视觉模型微调


# 中医问诊模型微调
一期：

1、微调模型诊断能力、
2、辨证推理能力
3、多轮对话问诊能力
4、中药、方剂等医学理解能力

期望输出数据格式： 
{
    "reasoning_steps": [
      "先什么，后什么，病机解释病因描述+核心症状，最后推出证型、证名"
    ],
    "result": {
      "syndrome": "湿毒蕴肤证",
      "syndrome_name": "玫瑰痤疮",
      "medicine": "推荐处方",
      "advice": ""# 医嘱、饮食运动建议等
    }
}
