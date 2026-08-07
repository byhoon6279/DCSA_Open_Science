#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared import compare_family_against_null, summarize_paired_differences, write_csv


def load_csv_rows(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate family-vs-null paired summaries from existing RA-Q4 CSV outputs."
    )
    parser.add_argument("--family-seed-summary", type=Path, required=True)
    parser.add_argument("--null-seed-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    family_seed_summary = load_csv_rows(args.family_seed_summary)
    null_seed_summary = load_csv_rows(args.null_seed_summary)

    paired_rows = compare_family_against_null(family_seed_summary, null_seed_summary)
    paired_summary = summarize_paired_differences(paired_rows)

    write_csv(args.output_dir / "family_vs_null_paired_rows.csv", paired_rows)
    write_csv(args.output_dir / "family_vs_null_paired_summary.csv", paired_summary)


if __name__ == "__main__":
    main()
