import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_INSTRUCTION = "你是一名中医皮肤科问诊助手，请完成辨证分析并输出结构化 JSON。"


def _read_json_or_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    text = file_path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
        if isinstance(obj, dict):
            return [obj]
    except json.JSONDecodeError:
        pass

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
        ("主诉", record.get("chief_complaint", "")),
        ("现病史", record.get("description", "")),
        ("四诊摘要", record.get("detection", "")),
        ("病名", record.get("lcd_name", "")),
        ("病名编码", record.get("lcd_id", "")),
    ]
    return "\n".join([f"{k}：{v}" for k, v in fields])


def _load_syndrome_knowledge_map(input_dir: Path) -> Dict[str, Dict[str, str]]:
    path = input_dir / "syndrome_knowledge.json"
    if not path.exists():
        return {}

    rows = _read_json_or_jsonl(path)
    knowledge_map: Dict[str, Dict[str, str]] = {}
    for row in rows:
        name = str(row.get("Name", "")).strip()
        if not name:
            continue
        knowledge_map[name] = {
            "definition": str(row.get("Definition", "")).strip(),
            "pathogenesis": str(row.get("Pathogenesis", "")).strip(),
            "typical_performance": str(row.get("Typical_performance", "")).strip(),
            "common_disease": str(row.get("Common_isease", "")).strip(),
        }
    return knowledge_map


def _match_knowledge(syndrome: str, syndrome_name: str, knowledge_map: Dict[str, Dict[str, str]]) -> Dict[str, str]:
    syndrome = syndrome.strip()
    syndrome_name = syndrome_name.strip()

    if syndrome in knowledge_map:
        return knowledge_map[syndrome]
    if syndrome_name in knowledge_map:
        return knowledge_map[syndrome_name]

    for key, val in knowledge_map.items():
        if syndrome and (syndrome in key or key in syndrome):
            return val
        if syndrome_name and (syndrome_name in key or key in syndrome_name):
            return val
    return {}


def _empty_output(task_type: str) -> Dict[str, Any]:
    return {
        "task_type": task_type,
        "reasoning_steps": [],
        "result": {
            "syndrome": "",
            "syndrome_name": "",
            "treatment_principle": "",
            "formula": "",
            "medicine": "",
            "advice": "",
        },
        "differential_diagnosis": {
            "candidate_syndrome": "",
            "is_match": "",
            "rejection_reason": "",
            "comparison_points": "",
        },
        "dialogue": {
            "history": "",
            "next_response": "",
            "response_type": "",
        },
    }


def _build_complete_diagnosis_sample(
    record: Dict[str, Any],
    source_file: str,
    knowledge_map: Dict[str, Dict[str, str]],
) -> Dict[str, Any] | None:
    syndrome = str(record.get("norm_syndrome") or record.get("syndrome") or "").strip()
    syndrome_name = str(record.get("lcd_name") or "").strip()
    if not syndrome and not syndrome_name:
        return None

    knowledge = _match_knowledge(syndrome, syndrome_name, knowledge_map)
    case_input = _build_case_input(record)
    desc = str(record.get("description", ""))
    detection = str(record.get("detection", ""))

    reasoning_steps = [
        f"证据提取：主诉与病程信息显示{str(record.get('chief_complaint', ''))[:60]}。",
        f"症候归纳：结合现病史与四诊（{desc[:50]}；{detection[:50]}）提取核心症状。",
        f"病机分析：参考证候定义与病因病机（{knowledge.get('definition', '')[:60]}）。",
        f"辨证结论：判定为{syndrome or ''}，对应病名{syndrome_name or ''}。",
    ]

    output = _empty_output("complete_diagnosis")
    output["reasoning_steps"] = reasoning_steps
    output["result"] = {
        "syndrome": syndrome,
        "syndrome_name": syndrome_name,
        "treatment_principle": knowledge.get("pathogenesis", ""),
        "formula": "",
        "medicine": "",
        "advice": knowledge.get("common_disease", ""),
    }

    return {
        "instruction": "请根据病史、症状和四诊信息完成完整辨证，并输出结构化 JSON。",
        "input": case_input,
        "output": output,
        "source": source_file,
    }


def _build_differential_sample(
    record: Dict[str, Any],
    source_file: str,
    syndrome_list: List[str],
) -> Dict[str, Any] | None:
    true_syndrome = str(record.get("norm_syndrome") or record.get("syndrome") or "").strip()
    syndrome_name = str(record.get("lcd_name") or "").strip()
    if not true_syndrome or len(syndrome_list) < 2:
        return None

    candidates = [s for s in syndrome_list if s != true_syndrome]
    if not candidates:
        return None
    candidate = random.choice(candidates)

    case_input = _build_case_input(record)
    diff_input = f"{case_input}\n\n待鉴别证型：{true_syndrome} vs {candidate}"

    output = _empty_output("differential_diagnosis")
    output["reasoning_steps"] = [
        "先基于主诉与现病史提炼核心症状。",
        f"再比较{true_syndrome}与{candidate}在关键症状与舌脉上的差异。",
        f"最终保留{true_syndrome}，排除{candidate}。",
    ]
    output["result"]["syndrome"] = true_syndrome
    output["result"]["syndrome_name"] = syndrome_name
    output["differential_diagnosis"] = {
        "candidate_syndrome": candidate,
        "is_match": "否",
        "rejection_reason": "候选证型与当前病例关键症状和四诊特征不一致。",
        "comparison_points": "症状组合、舌苔脉象、病机方向存在差异。",
    }

    return {
        "instruction": "请做中医证型鉴别诊断，比较两个证型并给出最终判定。输出结构化 JSON。",
        "input": diff_input,
        "output": output,
        "source": source_file,
    }


def _build_negative_rejection_sample(
    record: Dict[str, Any],
    source_file: str,
    syndrome_list: List[str],
) -> Dict[str, Any] | None:
    true_syndrome = str(record.get("norm_syndrome") or record.get("syndrome") or "").strip()
    if not true_syndrome or len(syndrome_list) < 2:
        return None

    candidates = [s for s in syndrome_list if s != true_syndrome]
    if not candidates:
        return None
    wrong = random.choice(candidates)

    case_input = _build_case_input(record)
    reject_input = f"{case_input}\n\n给定候选证型：{wrong}。请判断是否匹配并说明排除理由。"

    output = _empty_output("negative_rejection")
    output["reasoning_steps"] = [
        "读取病例核心症状与舌脉特征。",
        f"对照候选证型{wrong}的典型表现进行一致性检查。",
        "发现关键证据不支持该证型，作排除判断。",
    ]
    output["result"]["syndrome"] = true_syndrome
    output["differential_diagnosis"] = {
        "candidate_syndrome": wrong,
        "is_match": "否",
        "rejection_reason": "候选证型缺乏病例中的关键支持证据，且与现有四诊信息冲突。",
        "comparison_points": "候选证型与当前病例在病机与症状特征上不一致。",
    }

    return {
        "instruction": "请对给定候选证型进行反例判断（是否匹配），并输出结构化 JSON。",
        "input": reject_input,
        "output": output,
        "source": source_file,
    }


def _detect_response_type(text: str) -> str:
    if any(k in text for k in ["请问", "多久", "是否", "有没有"]):
        return "question"
    if any(k in text for k in ["中医病名", "证型", "辨证", "诊断"]):
        return "diagnosis"
    if any(k in text for k in ["处方", "方", "用法", "医嘱"]):
        return "prescription"
    return "general_reply"


def _build_multiturn_samples(record: Dict[str, Any], source_file: str) -> List[Dict[str, Any]]:
    conversations = record.get("conversations")
    if not isinstance(conversations, list):
        return []

    history: List[str] = []
    role_map = {"human": "用户", "user": "用户", "gpt": "助手", "assistant": "助手"}
    samples: List[Dict[str, Any]] = []

    for turn in conversations:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("from", "")).strip().lower()
        value = str(turn.get("value", "")).strip()
        if not value:
            continue

        role_name = role_map.get(role, "用户")
        if role_name == "助手" and history:
            output = _empty_output("multi_turn_completion")
            output["dialogue"] = {
                "history": "\n".join(history),
                "next_response": value,
                "response_type": _detect_response_type(value),
            }
            output["reasoning_steps"] = [
                "先阅读既往多轮对话，识别当前问诊阶段。",
                "根据历史槽位缺失情况生成下一轮合理回复。",
                "保证回复风格连贯且医学逻辑一致。",
            ]

            samples.append(
                {
                    "instruction": "请根据历史问诊对话补全下一轮助手回复，输出结构化 JSON。",
                    "input": "\n".join(history),
                    "output": output,
                    "source": source_file,
                }
            )
        history.append(f"{role_name}：{value}")

    return samples


def prepare_dataset(input_dir: Path, output_dir: Path, train_ratio: float, seed: int) -> None:
    random.seed(seed)

    files = sorted(input_dir.glob("*.json"))
    knowledge_map = _load_syndrome_knowledge_map(input_dir)
    syndrome_list = sorted(list(knowledge_map.keys()))

    samples: List[Dict[str, Any]] = []

    for file_path in files:
        rows = _read_json_or_jsonl(file_path)
        for row in rows:
            # 四类混任务：完整辨证、鉴别诊断、反例排除、多轮问诊补全
            complete = _build_complete_diagnosis_sample(row, file_path.name, knowledge_map)
            if complete:
                samples.append(complete)

            diff = _build_differential_sample(row, file_path.name, syndrome_list)
            if diff:
                samples.append(diff)

            neg = _build_negative_rejection_sample(row, file_path.name, syndrome_list)
            if neg:
                samples.append(neg)

            samples.extend(_build_multiturn_samples(row, file_path.name))

    # 去重
    dedup: Dict[tuple, Dict[str, Any]] = {}
    for s in samples:
        key = (s.get("instruction", ""), s.get("input", ""), json.dumps(s.get("output", {}), ensure_ascii=False, sort_keys=True))
        dedup[key] = s
    samples = list(dedup.values())

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
    print("输出文件:")
    print(f"- {all_path}")
    print(f"- {train_path}")
    print(f"- {val_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="准备中医混任务 SFT 数据（四类任务 + 9:1 切分）")
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
