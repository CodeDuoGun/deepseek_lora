from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "demo.json"


def _load_demo_json(path: Path = DATA_PATH) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def cal_num_data(trending_source):
    res = []
    for item in trending_source["newWatchUv"]:
        if item["step"] == "86400":
            res.append(item)
    print(res)
    return res



def main() -> None:
    data = _load_demo_json()
    trending_source = data.get("data", {}).get("trendingSource", {})
    # 统计steps值为86400的所有dict
    res_data = cal_num_data(trending_source)
    print(json.dumps(res_data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
