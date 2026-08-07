#!/usr/bin/env python3
"""Build Table 5 (`density_conditioned_rank_disagreement.tex`,
`\\label{tab:rq3_block1_rank_disagreement}`) from the per-model
"conflicts" CSVs (mean pairwise ordering conflicts between AUC and
structure-level rankings, per density bin).

The LR and LightGBM conflicts CSVs were already bundled
(`results/LR/density_reliability_lr_controlled_table_rq3_density_reliability_conflicts.csv`,
`results/LightGBM/density_reliability_lightgbm_controlled_table_rq3_density_reliability_conflicts.csv`),
but this table itself had no generating script anywhere in this Open Science
package, and no RF conflicts CSV existed anywhere (checked the private
working repo too). Regenerating the RF conflicts CSV via
`plot_density_stratified_reliability.py --input-dir results/RF/rf_full_wild_b/k_10`
reproduces the manuscript's RF column values exactly
(High 1.8/4.0, Mid 0.0/2.0, Low 1.2/2.0); the recovered CSV is bundled at
`results/RF/density_reliability_rf_controlled_table_rq3_density_reliability_conflicts.csv`.
This script combines all three and closes the gap. Verified byte-for-byte
match against `the manuscript repository's tables/density_conditioned_rank_disagreement.tex`.

The manuscript bolds the LR AUC-Mix@10 column's High and Low values to
highlight that column's monotonic high-to-low gradient (0.2 -> 2.0 -> 3.2);
that bolding is reproduced here as-is rather than inferred from a generic
rule.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

DENSITY_ORDER = ["High", "Mid", "Low"]
MODELS = ["LR", "LightGBM", "RF"]
BOLD_CELLS = {("LR", "High"), ("LR", "Low")}


def read_conflicts(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["density_bin"]: row for row in csv.DictReader(handle)}


def fmt(value: str, bold: bool) -> str:
    text = f"{float(value):.1f}"
    return rf"\textbf{{{text}}}" if bold else text


def build_tex(data: dict[str, dict[str, dict[str, str]]]) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Mean pairwise ordering conflicts between AUC and structure-level rankings across density bins.}",
        r"\label{tab:rq3_block1_rank_disagreement}",
        r"\resizebox{\linewidth}{!}{",
        r"\begin{tabular}{lcc|cc|cc}",
        r"\toprule",
        r"\multirow{2}{*}{\textbf{Density bin}} &",
        r"\multicolumn{2}{c|}{\textbf{LR}} &",
        r"\multicolumn{2}{c|}{\textbf{LightGBM}} &",
        r"\multicolumn{2}{c}{\textbf{RF}} \\",
        r"& \textbf{AUC--Mix@10} & \textbf{AUC--JS} & \textbf{AUC--Mix@10} & \textbf{AUC--JS} & \textbf{AUC--Mix@10} & \textbf{AUC--JS} \\",
        r"\midrule",
    ]
    for bin_label in DENSITY_ORDER:
        cells = []
        for model in MODELS:
            row = data[model][bin_label]
            bold = (model, bin_label) in BOLD_CELLS
            cells.append(fmt(row["auc_vs_mix_conflicts_mean"], bold))
            cells.append(fmt(row["auc_vs_js_conflicts_mean"], False))
        pad = " " if bin_label in {"Mid", "Low"} else ""
        lines.append(f"{bin_label}{pad} & " + " & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def main() -> None:
    here = Path(__file__).resolve()
    section_root = here.parents[2]  # 5_4_density_conditioned_reliability/
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lr-csv",
        type=Path,
        default=section_root / "results" / "LR" / "density_reliability_lr_controlled_table_rq3_density_reliability_conflicts.csv",
    )
    parser.add_argument(
        "--lightgbm-csv",
        type=Path,
        default=section_root / "results" / "LightGBM" / "density_reliability_lightgbm_controlled_table_rq3_density_reliability_conflicts.csv",
    )
    parser.add_argument(
        "--rf-csv",
        type=Path,
        default=section_root / "results" / "RF" / "density_reliability_rf_controlled_table_rq3_density_reliability_conflicts.csv",
    )
    parser.add_argument(
        "--output-tex",
        type=Path,
        default=section_root / "results" / "manuscript_tables" / "density_conditioned_rank_disagreement.tex",
    )
    args = parser.parse_args()

    data = {
        "LR": read_conflicts(args.lr_csv),
        "LightGBM": read_conflicts(args.lightgbm_csv),
        "RF": read_conflicts(args.rf_csv),
    }
    args.output_tex.parent.mkdir(parents=True, exist_ok=True)
    args.output_tex.write_text(build_tex(data), encoding="utf-8")
    print(f"Wrote {args.output_tex}")


if __name__ == "__main__":
    main()
