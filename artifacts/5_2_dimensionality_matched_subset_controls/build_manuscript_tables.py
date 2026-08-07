#!/usr/bin/env python3
"""Build Table 4 (`dimensionality_matched_subset_controls.tex`), Appendix
Table D.2 (`appendix_random_subspace_auc_controls.tex`), and Appendix Table
D.3 (`appendix_random_subspace_structure_controls.tex`).

Source data: `results/{LR,LightGBM}/wild_b_ra_q4_random_null/family_vs_null_paired_summary.csv`,
which reports, per (model, subset family, dimension, metric), the mean paired
difference between the family-aligned subspace and the matched random-subspace
null, with its 95% CI (`paired_difference_mean/_ci_lo/_ci_hi`). AUC is
model-dependent (LR vs LightGBM columns); mix_at_10, js_divergence, and
same_family_rate_at_10 are structure-level metrics computed from the same
feature subsets independent of which classifier scores them, so LR's and
LightGBM's rows for those three metrics are identical (verified) - this
script reads them from the LR file only.

Table 4 is the d=32 slice of Tables D.2 and D.3, dropping the JS column and
all four metrics' confidence intervals — a single-line-per-subset layout
with point estimates only; Tables D.2/D.3 still report CIs.

Verified byte-for-byte match against the corresponding
`the manuscript repository's tables/*.tex` files.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

SUBSETS = ["header", "section", "imports", "strings"]
SUBSET_LABELS = {"header": "Header", "section": "Section", "imports": "Imports", "strings": "Strings"}
DIMENSIONS = ["16", "32", "64"]


def read_paired_summary(path: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {(row["family"], row["dimension"], row["metric"]): row for row in rows}


def signed(value: str) -> str:
    return f"{float(value):+.3f}"


def ci(row: dict[str, str]) -> str:
    return f"[{signed(row['paired_difference_ci_lo'])}, {signed(row['paired_difference_ci_hi'])}]"


def cell_with_ci(row: dict[str, str]) -> str:
    return f"${signed(row['paired_difference_mean'])}$ [${signed(row['paired_difference_ci_lo'])}$, ${signed(row['paired_difference_ci_hi'])}$]"


def build_table3(lr: dict, lgbm: dict) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Dimensionality-matched controls at $d=32$ under Wild (B).",
        r"Values report semantic-subset minus random-subspace differences.}",
        r"\label{tab:ra_q4_dim32_summary}",
        r"",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}lcccc@{}}",
        r"\toprule",
        r"\multirow{2}{*}{\textbf{Subset}}",
        r"& \multicolumn{2}{c}{\textbf{Classifier performance ($\Delta$AUC $\uparrow$)}}",
        r"& \multicolumn{2}{c}{\textbf{Feature-space structure}} \\",
        r"\cmidrule(lr){2-3}",
        r"\cmidrule(lr){4-5}",
        r"& \textbf{LR}",
        r"& \textbf{LightGBM}",
        r"& \shortstack{\textbf{$\Delta$Mix}\\\textbf{@10 $\downarrow$}}",
        r"& \shortstack{\textbf{$\Delta$Same-family}\\\textbf{@10 $\uparrow$}} \\",
        r"\midrule",
    ]
    for subset in SUBSETS:
        lr_auc = lr[(subset, "32", "auc")]
        lgbm_auc = lgbm[(subset, "32", "auc")]
        mix = lr[(subset, "32", "mix_at_10")]
        fam = lr[(subset, "32", "same_family_rate_at_10")]
        label = f"{SUBSET_LABELS[subset]:<7}"
        lines.append(
            f"{label} & ${signed(lr_auc['paired_difference_mean'])}$"
            f" & ${signed(lgbm_auc['paired_difference_mean'])}$"
            f" & ${signed(mix['paired_difference_mean'])}$"
            f" & ${signed(fam['paired_difference_mean'])}$ \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def build_table_c2(lr: dict, lgbm: dict) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\caption{Subset-minus-random AUC differences for dimensionality-matched controls.}",
        r"\label{tab:appendix_ra_q4_auc}",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"\textbf{Subset} & \textbf{Dimension} & \textbf{LR AUC difference} & \textbf{LightGBM AUC difference} \\",
        r"\midrule",
    ]
    for i, subset in enumerate(SUBSETS):
        lines.append(f"\\multirow{{3}}{{*}}{{{SUBSET_LABELS[subset]}}}")
        for dim in DIMENSIONS:
            lr_row = lr[(subset, dim, "auc")]
            lgbm_row = lgbm[(subset, dim, "auc")]
            lines.append(f" & {dim} & {cell_with_ci(lr_row)} & {cell_with_ci(lgbm_row)} \\\\")
        if i < len(SUBSETS) - 1:
            lines.append(r"\midrule")
            lines.append("")
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\textit{Note.} Brackets show 95\% percentile bootstrap confidence intervals.", r"\end{table}"])
    return "\n".join(lines) + "\n"


def build_table_c3(structure: dict) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\caption{Structure-level departures from same-dimensional random-subspace controls.}",
        r"\label{tab:appendix_ra_q4_structure}",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{1pt}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{Subset} & \textbf{Dimension} & \textbf{$\Delta$Mix@10} & \textbf{$\Delta$JS} & \textbf{$\Delta$Same-family@10} \\",
        r"\midrule",
        "",
    ]
    for i, subset in enumerate(SUBSETS):
        lines.append(f"\\multirow{{3}}{{*}}{{{SUBSET_LABELS[subset]}}}")
        for dim in DIMENSIONS:
            mix = structure[(subset, dim, "mix_at_10")]
            js = structure[(subset, dim, "js_divergence")]
            fam = structure[(subset, dim, "same_family_rate_at_10")]
            sep = " " if subset == "header" else ""
            lines.append(f"{sep} & {dim} & {cell_with_ci(mix)} & {cell_with_ci(js)} & {cell_with_ci(fam)} \\\\")
        if i < len(SUBSETS) - 1:
            lines.append(r"\midrule")
            lines.append("")
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\textit{Note.} Brackets show 95\% percentile bootstrap confidence intervals.", r"\end{table}"])
    return "\n".join(lines) + "\n"


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lr-csv", type=Path, default=root / "results/LR/wild_b_ra_q4_random_null/family_vs_null_paired_summary.csv")
    parser.add_argument("--lightgbm-csv", type=Path, default=root / "results/LightGBM/wild_b_ra_q4_random_null/family_vs_null_paired_summary.csv")
    parser.add_argument("--output-dir", type=Path, default=root / "results/manuscript_tables")
    args = parser.parse_args()

    lr = read_paired_summary(args.lr_csv)
    lgbm = read_paired_summary(args.lightgbm_csv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "dimensionality_matched_subset_controls.tex").write_text(build_table3(lr, lgbm), encoding="utf-8")
    (args.output_dir / "appendix_random_subspace_auc_controls.tex").write_text(build_table_c2(lr, lgbm), encoding="utf-8")
    (args.output_dir / "appendix_random_subspace_structure_controls.tex").write_text(build_table_c3(lr), encoding="utf-8")
    print(f"Wrote 3 manuscript tables to {args.output_dir}")


if __name__ == "__main__":
    main()
