import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_INSTRUCTION = (
    "你是一名中医问诊助手。请根据病史、症状、四诊信息进行辨证分析，"
    "输出包含推理步骤和最终诊断建议的 JSON。"
)

BAICHUAN_KNOWLEDGE_INSTRUCTION = (
    "你是一名中医药知识助手，擅长中药、方剂、证候与临床应用问答。"
)


def _read_json_or_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    text = file_path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    # 优先尝试完整 JSON（list / dict）
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
        if isinstance(obj, dict):
            return [obj]
    except json.JSONDecodeError:
        pass

    # 回退到 JSONL
    rows: List[Dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
            if isinstance(item, dict):
                rows.append(item)
        except json.JSONDecodeError:
            continue
    return rows


def _build_case_input(record: Dict[str, Any]) -> str:
    fields = [
        ("主诉", record.get("chief_complaint")),
        ("现病史", record.get("description")),
        ("四诊摘要", record.get("detection")),
        ("病种", record.get("lcd_name")),
        ("病种编码", record.get("lcd_id")),
    ]
    parts = [f"{k}：{v}" for k, v in fields if v]
    return "\n".join(parts)


def _convert_record(record: Dict[str, Any], source_file: str) -> List[Dict[str, str]]:
    converted: List[Dict[str, str]] = []

    # 对话数据：按 assistant 回合展开，保留多轮上下文
    if "conversations" in record and isinstance(record["conversations"], list):
        conversations = record["conversations"]
        instruction = record.get("system_prompt") or BAICHUAN_KNOWLEDGE_INSTRUCTION

        history_lines: List[str] = []
        role_map = {"human": "用户", "gpt": "助手", "assistant": "助手", "user": "用户"}

        for turn in conversations:
            if not isinstance(turn, dict):
                continue
            role = str(turn.get("from", "")).strip().lower()
            value = str(turn.get("value", "")).strip()
            if not value:
                continue

            normalized_role = role_map.get(role, role or "未知")
            if normalized_role == "助手":
                if history_lines:
                    converted.append(
                        {
                            "instruction": instruction,
                            "input": "\n".join(history_lines),
                            "output": value,
                            "source": source_file,
                        }
                    )
                history_lines.append(f"助手：{value}")
            else:
                history_lines.append(f"用户：{value}")

        return converted

    # 已经是 SFT 三元组
    if "input" in record and "output" in record:
        instruction = record.get("instruction") or DEFAULT_INSTRUCTION
        output_obj = record["output"]
        output_text = output_obj if isinstance(output_obj, str) else json.dumps(output_obj, ensure_ascii=False)
        converted.append(
            {
                "instruction": instruction,
                "input": str(record["input"]),
                "output": output_text,
                "source": source_file,
            }
        )
        return converted

    # 病案数据（medical_case / train.json 等）
    if any(k in record for k in ["chief_complaint", "description", "detection", "norm_syndrome", "syndrome"]):
        syndrome = record.get("norm_syndrome") or record.get("syndrome") or ""
        syndrome_name = record.get("lcd_name") or ""
        output_obj = {
            "reasoning_steps": [
                "先归纳主诉与现病史，提取核心症状与体征。",
                "结合四诊信息与病机进行辨证分析。",
                "最终给出证型、病名与治法建议。",
            ],
            "result": {
                "syndrome": syndrome,
                "syndrome_name": syndrome_name,
                "medicine": "",
                "advice": "",
            },
        }
        converted.append(
            {
                "instruction": DEFAULT_INSTRUCTION,
                "input": _build_case_input(record),
                "output": json.dumps(output_obj, ensure_ascii=False),
                "source": source_file,
            }
        )
        return converted

    # 证候知识数据（syndrome_knowledge.json）
    if "Name" in record and "Definition" in record:
        input_text = f"证候名称：{record.get('Name', '')}"
        output_obj = {
            "definition": record.get("Definition", ""),
            "typical_performance": record.get("Typical_performance", ""),
            "common_disease": record.get("Common_isease", ""),
        }
        converted.append(
            {
                "instruction": "请根据给定证候名称，输出定义、典型表现和常见病。",
                "input": input_text,
                "output": json.dumps(output_obj, ensure_ascii=False),
                "source": source_file,
            }
        )

    return converted


def prepare_dataset(input_dir: Path, output_dir: Path, train_ratio: float, seed: int) -> None:
    files = sorted(input_dir.glob("*.json"))
    samples: List[Dict[str, str]] = []

    for file_path in files:
        rows = _read_json_or_jsonl(file_path)
        for row in rows:
            converted = _convert_record(row, file_path.name)
            for sample in converted:
                if sample and sample["input"].strip() and sample["output"].strip():
                    samples.append(sample)

    # 增加中医皮肤科多轮问诊数据（随机采样 1000 条）
    tcm_sft_path = input_dir / "tcm_sft.jsonl"
    if tcm_sft_path.exists():
        tcm_rows = _read_json_or_jsonl(tcm_sft_path)
        tcm_samples: List[Dict[str, str]] = []
        for row in tcm_rows:
            converted = _convert_record(row, tcm_sft_path.name)
            for sample in converted:
                if sample and sample["input"].strip() and sample["output"].strip():
                    tcm_samples.append(sample)

        if tcm_samples:
            random.seed(seed)
            random.shuffle(tcm_samples)
            samples.extend(tcm_samples[: min(1000, len(tcm_samples))])

    # 去重
    dedup = {}
    for s in samples:
        key = (s["instruction"], s["input"], s["output"])
        dedup[key] = s
    samples = list(dedup.values())

    random.seed(seed)
    random.shuffle(samples)

    split_idx = int(len(samples) * train_ratio)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    output_dir.mkdir(parents=True, exist_ok=True)
    all_path = output_dir / "tcm_sft_all.json"
    train_path = output_dir / "tcm_sft_train.json"
    val_path = output_dir / "tcm_sft_val.json"

    all_path.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
    train_path.write_text(json.dumps(train_samples, ensure_ascii=False, indent=2), encoding="utf-8")
    val_path.write_text(json.dumps(val_samples, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"总样本数: {len(samples)}")
    print(f"训练集: {len(train_samples)}")
    print(f"验证集: {len(val_samples)}")
    print(f"输出文件:\n- {all_path}\n- {train_path}\n- {val_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="准备中医 SFT 数据（9:1 切分）")
    parser.add_argument("--input_dir", type=str, default="medical/data/tcm_data")
    parser.add_argument("--output_dir", type=str, default="medical/train_data")
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prepare_dataset(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        train_ratio=args.train_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
