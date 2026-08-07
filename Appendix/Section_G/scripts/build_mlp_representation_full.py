#!/usr/bin/env python3
"""
Appendix Table G.1: full MLP representation-level results in Wild (B),
aggregated as unweighted means over 5 seeds and 12 test weeks.

Recomputes the table directly from the raw per-seed-per-week CSV already
bundled at `MLP/results/5_1/mlp_rq1_wild_b_main/metric_rows.csv`.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[3]
METRIC_ROWS_PATH = REPO_ROOT / "artifacts" / "MLP" / "results" / "5_1" / "mlp_rq1_wild_b_main" / "metric_rows.csv"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "results"

FEATURE_GROUPS = ["all", "header", "section", "imports", "strings"]


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def mean(values: List[float]) -> float:
    return sum(values) / len(values)


def compute_table() -> List[Dict[str, object]]:
    rows = load_rows(METRIC_ROWS_PATH)
    rows_out: List[Dict[str, object]] = []
    for group in FEATURE_GROUPS:
        sub = [row for row in rows if row["feature_group"] == group]
        rows_out.append(
            {
                "feature_group": group,
                "auc": mean([float(row["test_auc"]) for row in sub]),
                "mix_at_10": mean([float(row["mix_at_k"]) for row in sub]),
                "js_divergence": mean([float(row["js_divergence"]) for row in sub]),
                "silhouette": mean([float(row["silhouette"]) for row in sub]),
            }
        )
    return rows_out


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["feature_group", "auc", "mix_at_10", "js_divergence", "silhouette"])
        writer.writeheader()
        writer.writerows(rows)


def write_tex(path: Path, rows: List[Dict[str, object]]) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Full MLP representation-level results in Wild (B), aggregated as unweighted means over 5 seeds and 12 test weeks.}",
        r"\label{tab:app_mlp_representation_full}",
        r"\small",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{Feature group} & \textbf{AUC} & \textbf{Mix@10} & \textbf{JS} & \textbf{Silhouette} \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['feature_group']:<7} & {row['auc']:.4f} & {row['mix_at_10']:.4f} & "
            f"{row['js_divergence']:.4f} & {row['silhouette']:.4f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    rows = compute_table()
    write_csv(OUTPUT_DIR / "mlp_representation_full.csv", rows)
    write_tex(OUTPUT_DIR / "appendix_mlp_representation_full.tex", rows)
    for row in rows:
        print(
            f"{row['feature_group']:<8}: AUC={row['auc']:.4f} Mix@10={row['mix_at_10']:.4f} "
            f"JS={row['js_divergence']:.4f} Silhouette={row['silhouette']:.4f}"
        )
    print(f"Wrote {OUTPUT_DIR / 'mlp_representation_full.csv'} and appendix_mlp_representation_full.tex")


if __name__ == "__main__":
    main()
