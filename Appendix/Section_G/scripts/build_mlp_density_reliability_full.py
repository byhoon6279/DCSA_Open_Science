#!/usr/bin/env python3
"""
Appendix Table G.3: density-conditioned reliability results for the MLP
extension in Wild (B).

Recomputes the table directly from the raw per-seed CSV already bundled at
`MLP/results/5_4/mlp_rq3_density_reliability_wild_b_main/metric_rows.csv`
(mean over the 5 seeds per feature group x density bin).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[3]
METRIC_ROWS_PATH = (
    REPO_ROOT / "artifacts" / "MLP" / "results" / "5_4" / "mlp_rq3_density_reliability_wild_b_main" / "metric_rows.csv"
)
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "results"

FEATURE_GROUPS = ["all", "header", "section", "imports", "strings"]
GROUP_LABELS = {"all": "All", "header": "Header", "section": "Section", "imports": "Imports", "strings": "Strings"}
DENSITY_BINS = [("high_density", "High"), ("mid_density", "Mid"), ("low_density", "Low")]


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def mean(values: List[float]) -> float:
    return sum(values) / len(values)


def compute_table() -> List[Dict[str, object]]:
    rows = load_rows(METRIC_ROWS_PATH)
    rows_out: List[Dict[str, object]] = []
    for group in FEATURE_GROUPS:
        for density_bin, _ in DENSITY_BINS:
            sub = [row for row in rows if row["feature_group"] == group and row["density_bin"] == density_bin]
            rows_out.append(
                {
                    "feature_group": group,
                    "density_bin": density_bin,
                    "auc": mean([float(row["auc"]) for row in sub]),
                    "positive_rate": mean([float(row["positive_rate"]) for row in sub]),
                    "silhouette": mean([float(row["silhouette"]) for row in sub]),
                }
            )
    return rows_out


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["feature_group", "density_bin", "auc", "positive_rate", "silhouette"])
        writer.writeheader()
        writer.writerows(rows)


def write_tex(path: Path, rows: List[Dict[str, object]]) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Density-conditioned reliability results for the MLP extension in Wild (B).}",
        r"\label{tab:app_mlp_density_reliability_full}",
        r"",
        r"\resizebox{0.95\columnwidth}{!}{%",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"\textbf{Group} & \textbf{Density bin}",
        r"& \textbf{AUC}",
        r"& \textbf{Positive rate}",
        r"& \textbf{Silhouette} \\",
        r"\midrule",
        r"",
    ]
    for group in FEATURE_GROUPS:
        lines.append(f"\\multirow{{3}}{{*}}{{{GROUP_LABELS[group]}}}")
        low_cell = next(row for row in rows if row["feature_group"] == group and row["density_bin"] == "low_density")
        pad = " " if low_cell["silhouette"] < 0 else ""
        for density_bin, bin_label in DENSITY_BINS:
            cell = next(row for row in rows if row["feature_group"] == group and row["density_bin"] == density_bin)
            silhouette = cell["silhouette"]
            silhouette_str = f"{pad}{silhouette:.4f}" if silhouette >= 0 else f"{silhouette:.4f}"
            lines.append(f"& {bin_label:<4} & {cell['auc']:.4f} & {cell['positive_rate']:.4f} & {silhouette_str} \\\\")
        lines.append(r"\midrule")
        lines.append("")
    lines.pop()  # drop the trailing blank line
    lines.pop()  # drop the trailing \midrule before \bottomrule
    lines.append("")
    lines.extend([r"\bottomrule", r"\end{tabular}%", "}", r"\end{table}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = compute_table()
    write_csv(OUTPUT_DIR / "mlp_density_reliability_full.csv", rows)
    write_tex(OUTPUT_DIR / "appendix_mlp_density_reliability_full.tex", rows)
    for row in rows:
        print(
            f"{row['feature_group']:<8} {row['density_bin']:<12}: AUC={row['auc']:.4f} "
            f"Positive={row['positive_rate']:.4f} Silhouette={row['silhouette']:.4f}"
        )
    print(f"Wrote {OUTPUT_DIR / 'mlp_density_reliability_full.csv'} and appendix_mlp_density_reliability_full.tex")


if __name__ == "__main__":
    main()
