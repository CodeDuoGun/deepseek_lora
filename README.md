# Introduction
supported features
- [x] deepseek lora sft 微调多轮对话
- [ ] qwen3 lora sft 微调ai病历, 开发中


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

# Instruct finetune


# Run openmind
```bash
python finetune-multi-openmind.py
```
# TODO
- [x] peft 微调
- [ ] 强化学习
- [ ] 视觉模型微调
