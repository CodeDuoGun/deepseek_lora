from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "demo.json"

DIMENSION_TYPES = {
    "age": "2",
    "gender": "3",
    "province": "10",
    "city": "11",
    "fan_flag": "5",
    "profession": "9",
}

GENDER_LABELS = [("男性", "男"), ("女性", "女"), ("未知", "未知")]
FAN_BUCKETS = ["粉丝", "非粉丝", "未知"]
AGE_BUCKETS = [">59", "50-59", "40-49", "30-39", "25-29"]
PROFESSION_BUCKETS = [
    "小镇中老年",
    "都市银发",
    "新锐白领",
    "小镇青年",
    "都市蓝领",
    "资深中产",
    "未知",
]


def _load_demo_json(path: Path = DATA_PATH) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_label(value: Any) -> str:
    text = str(value or "").strip()
    for ch in ("\u3000", " "):
        text = text.replace(ch, "")
    return text


def _last_point(series: Iterable[Mapping[str, Any]]) -> Tuple[int, int]:
    last_ts = -1
    last_value = 0
    for point in series:
        ts = _safe_int(point.get("ts"), -1)
        value = _safe_int(point.get("value"))
        if ts >= last_ts:
            last_ts = ts
            last_value = value
    return last_ts, last_value


def _collect_dimension(
    entries: Iterable[Mapping[str, Any]],
    dim_type: str,
) -> Dict[str, Dict[str, Any]]:
    collected: Dict[str, Dict[str, Any]] = {}
    for entry in entries:
        target_dim = next(
            (dim for dim in entry.get("dimensions", []) if dim.get("type") == dim_type),
            None,
        )
        if not target_dim:
            continue

        label = _normalize_label(target_dim.get("value") or target_dim.get("uxLabel"))
        if not label:
            continue

        ts, value = _last_point(entry.get("data", []))
        if ts < 0:
            continue

        existing = collected.get(label)
        code_str = str(target_dim.get("dimensionValue") or "").strip()
        if not existing or ts >= existing["ts"]:
            collected[label] = {"count": value, "code": code_str, "ts": ts}

    return {
        name: {"count": meta["count"], "code": meta["code"]}
        for name, meta in collected.items()
    }


def _sorted_counts(raw: Mapping[str, int]) -> Dict[str, int]:
    sorted_items = sorted(raw.items(), key=lambda item: (-item[1], item[0]))
    return {name: count for name, count in sorted_items}


def _format_gender(raw: Mapping[str, Dict[str, Any]]) -> Dict[str, int]:
    result: MutableMapping[str, int] = {}
    for source, display in GENDER_LABELS:
        result[display] = raw.get(source, {}).get("count", 0)
    return dict(result)


def _format_fan_split(raw: Mapping[str, Dict[str, Any]]) -> Dict[str, int]:
    return {bucket: raw.get(bucket, {}).get("count", 0) for bucket in FAN_BUCKETS}


def _format_age(raw: Mapping[str, Dict[str, Any]]) -> Dict[str, int]:
    return {bucket: raw.get(bucket, {}).get("count", 0) for bucket in AGE_BUCKETS}


def _format_profession(raw: Mapping[str, Dict[str, Any]]) -> Dict[str, int]:
    return {name: raw.get(name, {}).get("count", 0) for name in PROFESSION_BUCKETS}


def _derive_top_province(
    province_raw: Mapping[str, Dict[str, Any]],
    city_raw: Mapping[str, Dict[str, Any]],
    province_sorted: Mapping[str, int],
) -> Dict[str, Any]:
    if not province_sorted:
        return {"name": "", "count": 0, "cities": {}}

    top_name = next(iter(province_sorted))
    province_code = (province_raw.get(top_name, {}).get("code") or "")[:2]
    if not province_code:
        city_counts: Dict[str, int] = {}
    else:
        city_counts = {
            name: info["count"]
            for name, info in city_raw.items()
            if info["count"] > 0 and (info.get("code") or "").startswith(province_code)
        }

    return {
        "name": top_name,
        "count": province_sorted[top_name],
        "cities": _sorted_counts(city_counts),
    }


def _build_distribution(
    entries: Iterable[Mapping[str, Any]],
    include_fan_split: bool,
) -> Dict[str, Any]:
    normalized_entries = list(entries or [])
    gender_raw = _collect_dimension(normalized_entries, DIMENSION_TYPES["gender"])
    age_raw = _collect_dimension(normalized_entries, DIMENSION_TYPES["age"])
    profession_raw = _collect_dimension(normalized_entries, DIMENSION_TYPES["profession"])
    province_raw = _collect_dimension(normalized_entries, DIMENSION_TYPES["province"])
    city_raw = _collect_dimension(normalized_entries, DIMENSION_TYPES["city"])

    province_positive = {
        name: info["count"] for name, info in province_raw.items() if info["count"] > 0
    }
    province_sorted = _sorted_counts(province_positive)

    result: Dict[str, Any] = {
        "gender": _format_gender(gender_raw),
        "age": _format_age(age_raw),
        "profession": _format_profession(profession_raw),
        "province": province_sorted,
        "top_province": _derive_top_province(province_raw, city_raw, province_sorted),
    }

    if include_fan_split:
        fan_raw = _collect_dimension(normalized_entries, DIMENSION_TYPES["fan_flag"])
        result["fan_type"] = _format_fan_split(fan_raw)

    return result


def compute_user_distribution(portrait: Mapping[str, Any]) -> Dict[str, Any]:
    audience_entries = portrait.get("cumulativeWatchUv") or []
    return _build_distribution(audience_entries, include_fan_split=True)


def compute_fan_distribution(portrait: Mapping[str, Any]) -> Dict[str, Any]:
    fan_entries = portrait.get("followerCumulativeWatchUv") or []
    return _build_distribution(fan_entries, include_fan_split=False)


def main() -> None:
    payload = _load_demo_json()
    portrait = payload.get("data", {}).get("portraitAudience", {})
    user_summary = compute_user_distribution(portrait)
    fan_summary = compute_fan_distribution(portrait)

    print("用户分布：")
    print(json.dumps(user_summary, ensure_ascii=False, indent=2))
    print("\n粉丝分布：")
    print(json.dumps(fan_summary, ensure_ascii=False, indent=2))


def test_user_distribution_from_demo_json() -> None:
    payload = _load_demo_json()
    portrait = payload.get("data", {}).get("portraitAudience", {})
    summary = compute_user_distribution(portrait)

    expected = {
        "gender": {"男": 57, "女": 48, "未知": 1},
        "fan_type": {"粉丝": 1, "非粉丝": 105, "未知": 0},
        "age": {">59": 50, "50-59": 30, "40-49": 17, "30-39": 3, "25-29": 6},
        "profession": {
            "小镇中老年": 54,
            "都市银发": 34,
            "新锐白领": 9,
            "小镇青年": 3,
            "都市蓝领": 3,
            "资深中产": 1,
            "未知": 2,
        },
        "province": {
            "山东省": 13,
            "广东省": 12,
            "河北省": 12,
            "浙江省": 8,
            "内蒙古自治区": 5,
            "江西省": 5,
            "陕西省": 5,
            "新疆维吾尔自治区": 4,
            "江苏省": 4,
            "吉林省": 3,
            "四川省": 3,
            "安徽省": 3,
            "山西省": 3,
            "河南省": 3,
            "海南省": 3,
            "湖北省": 3,
            "湖南省": 3,
            "甘肃省": 3,
            "福建省": 3,
            "北京市": 2,
            "上海市": 1,
            "广西壮族自治区": 1,
            "未知": 1,
            "贵州省": 1,
            "辽宁省": 1,
            "黑龙江省": 1,
        },
        "top_province": {
            "name": "山东省",
            "count": 13,
            "cities": {
                "济南市": 3,
                "日照市": 2,
                "青岛市": 2,
                "临沂市": 1,
                "威海市": 1,
                "泰安市": 1,
                "淄博市": 1,
                "烟台市": 1,
                "菏泽市": 1,
            },
        },
    }

    assert summary == expected


def test_fan_distribution_defaults_when_missing_data() -> None:
    payload = {"data": {"portraitAudience": {"followerCumulativeWatchUv": []}}}
    summary = compute_fan_distribution(payload["data"]["portraitAudience"])

    assert summary == {
        "gender": {"男": 0, "女": 0, "未知": 0},
        "age": {">59": 0, "50-59": 0, "40-49": 0, "30-39": 0, "25-29": 0},
        "profession": {
            "小镇中老年": 0,
            "都市银发": 0,
            "新锐白领": 0,
            "小镇青年": 0,
            "都市蓝领": 0,
            "资深中产": 0,
            "未知": 0,
        },
        "province": {},
        "top_province": {"name": "", "count": 0, "cities": {}},
    }


if __name__ == "__main__":
    main()

