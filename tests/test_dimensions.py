from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "demo.json"


def _load_demo_json(path: Path = DATA_PATH) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _series_last_value(points: Iterable[Mapping[str, Any]]) -> int:
    last_value = 0
    for point in points:
        value = point.get("value")
        try:
            last_value = int(value)
        except (TypeError, ValueError):
            continue
    return last_value


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def aggregate_dimensions(trending_source: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Build nested aggregation based on dimensions:

    - level 1: type 16 (公域流量 / 私域流量)
    - level 2: type 1 (公众号 / 分享 / 关注开播通知 ...)
    - level 3: type 7 (公众号文章 / 聊天页 ...)
    """
    allowed_level2: Dict[str, set[str]] = {
        "公域流量": {"直播推荐", "搜索", "关注开播通知", "短视频引流"},
        "私域流量": {"分享", "短视频引流", "预约开播通知", "加热流量", "广告流量"},
    }

    entries = trending_source.get("newWatchPv", [])
    result: Dict[str, Dict[str, Any]] = {
        level1: {level2: 0 for level2 in sorted(level2s)}
        for level1, level2s in allowed_level2.items()
    }

    latest_end_ts: Dict[Tuple[str, str], int] = {}

    for entry in entries:
        dims = {dim.get("type"): dim.get("value") for dim in entry.get("dimensions", [])}
        level1 = dims.get("16")
        level2 = dims.get("1")
        if not level1 or not level2:
            continue

        allowed_level2s = allowed_level2.get(level1)
        if not allowed_level2s or level2 not in allowed_level2s:
            continue

        end_ts = _safe_int(entry.get("endTs"))
        key = (level1, level2)
        latest_end_ts[key] = max(end_ts, latest_end_ts.get(key, -1))

    for entry in entries:
        dims = {dim.get("type"): dim.get("value") for dim in entry.get("dimensions", [])}
        level1 = dims.get("16")
        level2 = dims.get("1")
        if not level1 or not level2:
            continue

        allowed_level2s = allowed_level2.get(level1)
        if not allowed_level2s or level2 not in allowed_level2s:
            continue

        end_ts = _safe_int(entry.get("endTs"))
        key = (level1, level2)
        if end_ts < latest_end_ts.get(key, -1):
            continue

        level3 = dims.get("7")
        final_value = _series_last_value(entry.get("data", []))
        level1_bucket = result[level1]

        if level3:
            existing = level1_bucket[level2]
            if isinstance(existing, int):
                level1_bucket[level2] = {"__total": existing}

            level2_bucket = level1_bucket.setdefault(level2, {})
            assert isinstance(level2_bucket, dict)
            level2_bucket[level3] = final_value
        else:
            existing = level1_bucket[level2]
            if isinstance(existing, dict):
                existing["__total"] = final_value
            else:
                level1_bucket[level2] = final_value

    return result


def main() -> None:
    data = _load_demo_json()
    trending_source = data.get("data", {}).get("trendingSource", {})
    res_data = aggregate_dimensions(trending_source)
    print(json.dumps(res_data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

