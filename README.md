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
## Quickly install
1、 train
```bash
python medical medical/finetune_qwen3.py
```

2、 inference without vllm
```bash
python medical/without_merge_inference.py
```

3、 inference with vllm
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
