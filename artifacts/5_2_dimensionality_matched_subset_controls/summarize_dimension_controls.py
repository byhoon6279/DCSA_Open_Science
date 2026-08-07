#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize RA-Q4 dimension-control outputs.")
    parser.add_argument("--family-summary", type=Path, required=True)
    parser.add_argument("--paired-summary", type=Path, required=True)
    args = parser.parse_args()

    family_rows = load_rows(args.family_summary)
    paired_rows = load_rows(args.paired_summary)

    print(f"family seed summary rows: {len(family_rows)}")
    print(f"paired summary rows: {len(paired_rows)}")
    print("Next step: convert these summaries into manuscript tables/figures.")


if __name__ == "__main__":
    main()
