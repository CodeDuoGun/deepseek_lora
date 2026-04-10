import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset, DatasetDict
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)


def build_chat_sample(record: dict, tokenizer, max_length: int):
    instruction = record.get("instruction", "")
    user_input = record.get("input", "")
    output_text = record.get("output", "")

    if not isinstance(output_text, str):
        output_text = json.dumps(output_text, ensure_ascii=False)

    user_text = f"{instruction}\n\n{user_input}".strip()
    prompt = f"<|im_start|>user\n{user_text}\n<|im_end|>\n<|im_start|>assistant\n"
    full_text = prompt + output_text + "\n<|im_end|>"

    tokenized = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
        return_attention_mask=True,
    )

    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    prefix_len = min(len(prompt_ids), len(input_ids))

    labels = [-100] * prefix_len + input_ids[prefix_len:]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def load_json_dataset(path: str) -> Dataset:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} 不是 JSON 数组")
    data = [x for x in data if isinstance(x, dict)]
    return Dataset.from_list(data)


def main():
    parser = argparse.ArgumentParser(description="Qwen3.5 9B LoRA SFT")
    parser.add_argument("--model_path", type=str, default="qwen3.5_9b_base")
    parser.add_argument("--train_file", type=str, default="medical/train_data/tcm_sft_train.json")
    parser.add_argument("--val_file", type=str, default="medical/train_data/tcm_sft_val.json")
    parser.add_argument("--output_dir", type=str, default="medical/output/qwen3_lora")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_files_only", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        use_fast=False,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto" if torch.cuda.is_available() else None,
        local_files_only=args.local_files_only,
    )

    train_raw = load_json_dataset(args.train_file)
    val_raw = load_json_dataset(args.val_file)
    raw_ds = DatasetDict({"train": train_raw, "validation": val_raw})

    tokenized_ds = raw_ds.map(
        build_chat_sample,
        fn_kwargs={"tokenizer": tokenizer, "max_length": args.max_length},
        remove_columns=raw_ds["train"].column_names,
    )

    lora_config = LoraConfig(
        r=64,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
    )

    model.enable_input_require_grads()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=500,
        save_steps=500,
        logging_steps=20,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        bf16=False,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        report_to="none",
        seed=args.seed,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, return_tensors="pt")

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"],
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
