import argparse
from os.path import join

import pandas as pd
from datasets import Dataset
from loguru import logger
from openmind import (
    TrainingArguments,
    AutoModelForCausalLM,
    Trainer,
    # DataCollatorForSeq2Seq,
    AutoTokenizer,
)
import torch
from peft import LoraConfig, get_peft_model, TaskType
from swanlab.integration.transformers import SwanLabCallback
import bitsandbytes as bnb
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os
import json

MODEL_NAME = "deepseek-llm-7b-chat"
DATA_NAME = "medical_multi_data_pro.json"

# 下面是需要修改的
OUTPUT_DIR = "openmind-epoch-10"


# 配置参数
def configuration_parameter():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for deepseek model")

    # 模型路径相关参数
    parser.add_argument("--model_name_or_path", type=str, default=f"./model/deepseek-ai/{MODEL_NAME}",
                        help="Path to the model directory downloaded locally")
    parser.add_argument("--output_dir", type=str,
                        default=f"./deepseek/config-test-output/{OUTPUT_DIR}",
                        help="Directory to save the fine-tuned model and checkpoints")

    # 数据集路径
    parser.add_argument("--train_file", type=str, default=f"./data/{DATA_NAME}",
                        help="Path to the training data file in JSONL format")

    # 训练超参数
    parser.add_argument("--num_train_epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2,
                        help="Batch size per device during training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                        help="Number of updates steps to accumulate before performing a backward/update pass")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="Learning rate for the optimizer")
    parser.add_argument("--max_seq_length", type=int, default=2048,
                        help="Maximum sequence length for the input")
    parser.add_argument("--logging_steps", type=int, default=1,
                        help="Number of steps between logging metrics")
    parser.add_argument("--save_steps", type=int, default=5000,
                        help="Number of steps between saving checkpoints")
    parser.add_argument("--save_total_limit", type=int, default=1,
                        help="Maximum number of checkpoints to keep")
    parser.add_argument("--lr_scheduler_type", type=str, default="polynomial",
                        help="Type of learning rate scheduler")
    parser.add_argument("--warmup_ratio", type=float, default=0.0,
                        help="Fraction of total steps to perform warmup (default: 0.0)")
    parser.add_argument("--warmup_steps", type=int, default=0,
                        help="Number of warmup steps for learning rate scheduler")

    # LoRA 特定参数
    parser.add_argument("--lora_rank", type=int, default=64,
                        help="Rank of LoRA matrices")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="Alpha parameter for LoRA")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="Dropout rate for LoRA")

    # 分布式训练参数
    parser.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", -1)),
                        help="Local rank for distributed training")
    parser.add_argument("--distributed", type=bool, default=True, help="Enable distributed training")

    # 额外优化和硬件相关参数
    parser.add_argument("--gradient_checkpointing", type=bool, default=True,
                        help="Enable gradient checkpointing to save memory")
    parser.add_argument("--optim", type=str, default="adamw_torch",
                        help="Optimizer to use during training")
    parser.add_argument("--train_mode", type=str, default="lora",
                        help="lora or qlora")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--fp16", type=bool, default=True,
                        help="Use mixed precision (FP16) training")
    parser.add_argument("--report_to", type=str, default=None,
                        help="Reporting tool for logging (e.g., tensorboard)")
    parser.add_argument("--dataloader_num_workers", type=int, default=0,
                        help="Number of workers for data loading")
    parser.add_argument("--save_strategy", type=str, default="steps",
                        help="Strategy for saving checkpoints ('steps', 'epoch')")
    parser.add_argument("--weight_decay", type=float, default=0,
                        help="Weight decay for the optimizer")
    parser.add_argument("--max_grad_norm", type=float, default=1,
                        help="Maximum gradient norm for clipping")
    parser.add_argument("--remove_unused_columns", type=bool, default=True,
                        help="Remove unused columns from the dataset")

    args = parser.parse_args()
    return args


def find_all_linear_names(model, train_mode):
    """
    找出所有全连接层，为所有全连接添加adapter
    """
    assert train_mode in ['lora', 'qlora']
    cls = bnb.nn.Linear4bit if train_mode == 'qlora' else nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    lora_module_names = list(lora_module_names)
    logger.info(f'LoRA target module names: {lora_module_names}')
    return lora_module_names


def setup_distributed(args):
    """初始化分布式环境"""
    if args.distributed:
        if args.local_rank == -1:
            raise ValueError("未正确初始化 local_rank，请确保通过分布式启动脚本传递参数，例如 torchrun。")

        # 初始化分布式进程组
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(args.local_rank)
        print(f"分布式训练已启用，Local rank: {args.local_rank}")
    else:
        print("未启用分布式训练，单线程模式。")


# 加载模型
def load_model(args, train_dataset, data_collator):
    # 初始化分布式环境
    setup_distributed(args)
    # 自动分配设备
    # 加载模型
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.float16 if args.fp16 else torch.bfloat16,
        "use_cache": False if args.gradient_checkpointing else True,
        "device_map": "auto" if not args.distributed else None,
    }
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)
    # 用于确保模型的词嵌入层参与训练
    model.enable_input_require_grads()
    # 将模型移动到正确设备
    if args.distributed:
        model.to(args.local_rank)
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

    # 哪些模块需要注入Lora参数
    target_modules = find_all_linear_names(model.module if isinstance(model, DDP) else model, args.train_mode)
    # target_modules=['up_proj', 'gate_proj','down_proj']
    # target_modules = ['q_proj', 'o_proj', 'v_proj', 'k_proj']
    # target_modules = ['q_proj', 'k_proj']

    # lora参数设置
    config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=target_modules,
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False

    )
    use_bfloat16 = torch.cuda.is_bf16_supported()  # 检查设备是否支持 bf16
    # 配置训练参数
    train_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        num_train_epochs=args.num_train_epochs,
        save_steps=args.save_steps,
        learning_rate=args.learning_rate,
        save_on_each_node=True,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to=args.report_to,
        seed=args.seed,
        optim=args.optim,
        local_rank=args.local_rank if args.distributed else -1,
        ddp_find_unused_parameters=False,  # 分布式参数检查优化
        fp16=args.fp16,
        bf16=not args.fp16 and use_bfloat16,
        remove_unused_columns=False,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        warmup_ratio=args.warmup_ratio,
    )
    # 应用 PEFT 配置到模型
    model = get_peft_model(model.module if isinstance(model, DDP) else model, config)  # 确保传递的是原始模型
    model.print_trainable_parameters()

    # 创建一个字典来存储模型参数和设备信息
    device_info = {}

    # 获取模型中每个参数的设备信息
    for name, param in model.named_parameters():
        device_info[name] = str(param.device)  # 将设备信息转为字符串

    # 将设备信息写入 JSON 文件
    with open('model_device_info.json', 'w') as f:
        json.dump(device_info, f, indent=4)

    ### 展示平台
    swanlab_config = {
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "data": DATA_NAME
    }
    swanlab_callback = SwanLabCallback(
        project=f"{MODEL_NAME}-finetune",
        experiment_name=f"{OUTPUT_DIR}",
        description=f"使用{DATA_NAME}数据集微调，重要信息：{OUTPUT_DIR}",
        workspace=None,
        config=swanlab_config,
    )
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        callbacks=[swanlab_callback],
    )
    return trainer


# 处理数据
def process_data(data, tokenizer, max_seq_length):
    input_ids, attention_mask, labels = [], [], []

    conversations = data["conversation"]
    for i, conv in enumerate(conversations):

        if "instruction" in conv:
            instruction_text = conv['instruction']
        else:
            instruction_text = ""
        human_text = conv["input"]
        assistant_text = conv["output"]

        input_text = f"{tokenizer.bos_token}{instruction_text}\n\nUser:{human_text}\n\nAssistant:"

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
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }
# 同transformers的DataCollatorForSeq2Seq
class DataCollatorForSeq2SeqCustom:
    def __init__(self, tokenizer, padding=True, return_tensors="pt"):
        self.tokenizer = tokenizer
        self.padding = padding
        self.return_tensors = return_tensors

    def __call__(self, batch):
        # 提取 input_ids, attention_mask, labels
        input_ids = [example['input_ids'] for example in batch]
        attention_mask = [example['attention_mask'] for example in batch]
        labels = [example['labels'] for example in batch]

        # 对所有 input_ids, attention_mask 和 labels 填充到最大长度
        input_ids = self.pad_sequence(input_ids)
        attention_mask = self.pad_sequence(attention_mask)
        labels = self.pad_sequence(labels)

        # 构造 PyTorch tensor
        if self.return_tensors == "pt":
            input_ids = torch.tensor(input_ids)
            attention_mask = torch.tensor(attention_mask)
            labels = torch.tensor(labels)

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    def pad_sequence(self, sequences):
        # 填充到最长序列的长度
        max_length = max(len(seq) for seq in sequences)
        padded_sequences = [
            seq + [self.tokenizer.pad_token_id] * (max_length - len(seq)) for seq in sequences
        ]
        return padded_sequences


# 训练部分
def main():
    args = configuration_parameter()
    print("*****************加载分词器*************************")
    # 加载分词器
    model_path = args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    print("*****************处理数据*************************")
    # 处理数据
    # 获得数据
    if os.path.splitext(args.train_file)[1].lower() == '.json':
        # json文件
        data = pd.read_json(args.train_file)
    else:
        # jsonl文件
        data = pd.read_json(args.train_file, lines=True)
    train_ds = Dataset.from_pandas(data)

    # 最长长度
    if tokenizer.model_max_length:
        max_seq_length = tokenizer.model_max_length
    else:
        max_seq_length = args.max_seq_length

    train_dataset = train_ds.map(process_data,
                                 fn_kwargs={"tokenizer": tokenizer, "max_seq_length": max_seq_length},
                                 remove_columns=train_ds.column_names)
    data_collator = DataCollatorForSeq2SeqCustom(tokenizer=tokenizer, padding=True, return_tensors="pt")
    print(train_dataset, data_collator)
    # 加载模型
    print("*****************训练*************************")
    trainer = load_model(args, train_dataset, data_collator)
    trainer.train()
    # 训练
    final_save_path = join(args.output_dir)
    trainer.save_model(final_save_path)


if __name__ == "__main__":
    main()
