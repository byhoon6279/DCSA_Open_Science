#!/usr/bin/env python3
"""Build the manuscript's actual Table 3 (`measurement_summary.tex`,
`\\label{tab:measurement_summary}`), which is LR-only (4 metric columns).

This script reads the already-bundled `table_rq1_measurement_summary.csv`
and emits the plain LR-only table that the manuscript's Table 3 reports
(a separate wider LR+RF variant, `\\label{tab:measurement_summary_with_rf}`,
is a different table not cited by the manuscript and is not built here).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

FEATURE_ORDER = ["all", "header", "section", "strings", "imports"]
FEATURE_LABELS = {
    "all": "All",
    "header": "Header",
    "section": "Section",
    "strings": "Strings",
    "imports": "Imports",
}


def read_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["feature_subset"]: row for row in csv.DictReader(handle)}


def build_tex(rows: dict[str, dict[str, str]]) -> str:
    lines = [
        r"\begin{center}",
        r"\captionsetup{type=table}",
        r"\captionof{table}{Measurement summary across feature subsets under Wild (B) for LR.",
        r"Decision-level and structural metrics produce inconsistent subset rankings.}",
        r"\label{tab:measurement_summary}",
        r"\scriptsize",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{Feature subset} & \textbf{AUC} & \textbf{Mix@10} & \textbf{Silhouette} & \textbf{JS} \\",
        r"\midrule",
    ]
    for feature in FEATURE_ORDER:
        row = rows[feature]
        label = f"{FEATURE_LABELS[feature]:<7}"
        lines.append(
            f"{label} & {float(row['auc']):.4f} & {float(row['mix_at_10']):.4f} "
            f"& {float(row['silhouette']):.4f} & {float(row['js_divergence']):.4f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{center}"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve()
    section_root = here.parents[2]  # 5_1_representation_level_separability_disagreement/
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=section_root / "results" / "LR" / "table_rq1_measurement_summary.csv",
    )
    parser.add_argument(
        "--output-tex",
        type=Path,
        default=section_root / "results" / "manuscript_tables" / "measurement_summary.tex",
    )
    args = parser.parse_args()

    rows = read_rows(args.input_csv)
    args.output_tex.parent.mkdir(parents=True, exist_ok=True)
    args.output_tex.write_text(build_tex(rows), encoding="utf-8")
    print(f"Wrote {args.output_tex}")


if __name__ == "__main__":
    main()
