#!/usr/bin/env python3
"""
Build the class/packing composition table used for the RQ1 dataset summary.

The output CSV reports, for train and test:
- total samples
- unpacked samples
- packed samples

This script reads the weekly Win32 JSONL files directly so the table can be
reconstructed without relying on a previously exported CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict

SCRIPT_DIR = Path(__file__).resolve().parent
RQ1_1_DIR = SCRIPT_DIR.parents[1]
RQ1_DIR = SCRIPT_DIR.parents[2]

import sys

_SHARED_LIB_DIR = SCRIPT_DIR.parents[3] / "artifacts" / "shared"
if str(_SHARED_LIB_DIR) not in sys.path:
    sys.path.insert(0, str(_SHARED_LIB_DIR))

from common import is_packed_row, week_paths  # type: ignore  # noqa: E402


DEFAULT_CONFIG = RQ1_1_DIR / "configs" / "LR" / "win32_all_train_all_test.json"
DEFAULT_OUTPUT = RQ1_1_DIR / "results" / "manuscript_tables" / "table_rq1_class_packing_composition.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_data_root(config_path: Path, config: dict) -> Path:
    data_root = Path(config["data_root"])
    if not data_root.is_absolute():
        data_root = (config_path.parent / data_root).resolve()
    return data_root


def count_split(paths: list[Path]) -> Dict[str, Dict[str, int]]:
    counts: Dict[str, Dict[str, int]] = {
        "benign": {"total": 0, "unpacked": 0, "packed": 0},
        "malicious": {"total": 0, "unpacked": 0, "packed": 0},
    }
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                label = row.get("label")
                if label not in {0, 1}:
                    continue
                class_name = "benign" if int(label) == 0 else "malicious"
                packed = is_packed_row(row)
                counts[class_name]["total"] += 1
                if packed:
                    counts[class_name]["packed"] += 1
                else:
                    counts[class_name]["unpacked"] += 1
    return counts


def write_csv_file(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "class", "total", "unpacked", "packed"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data_root = resolve_data_root(args.config, config)
    platform = config["platform"]

    train_paths = week_paths(data_root, platform, config["train_weeks"], "train")
    test_paths = week_paths(data_root, platform, config["test_weeks"], "test")

    train_counts = count_split(train_paths)
    test_counts = count_split(test_paths)

    rows = [
        {"split": "train", "class": "benign", **train_counts["benign"]},
        {"split": "train", "class": "malicious", **train_counts["malicious"]},
        {"split": "test", "class": "benign", **test_counts["benign"]},
        {"split": "test", "class": "malicious", **test_counts["malicious"]},
    ]
    write_csv_file(args.output_csv, rows)
    print(f"Wrote class/packing composition table to {args.output_csv}")


if __name__ == "__main__":
    main()
