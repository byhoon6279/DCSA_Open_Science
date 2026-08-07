#!/usr/bin/env python3
"""Build Appendix Table D.1 (`appendix_k_value_justification.tex`) from the
already-bundled Wild (B) k-sensitivity CSV.

This script builds the manuscript LaTeX from the precomputed compact CSV
(`results/manuscript_tables/k_sensitivity_wild_b_table.csv`).

The source CSV bundled here (`results/manuscript_tables/k_sensitivity_wild_b_table.csv`)
regenerates byte-for-byte from the raw per-k trial data bundled in this
section's own `results/LR/feature_perturbation_balanced_main/k_{5,10,15,20}/`
and `results/LightGBM/feature_perturbation_lightgbm_{gain,permutation}_balanced_main/k_{5,10,15,20}/`,
via `export_k_sensitivity_table.py --results-dir <section>/results` - confirmed
by regenerating it and diffing against the bundled copy. It matches the
Appendix Table D.1 in the manuscript exactly.
"""
from __future__ import annotations

import csv
from pathlib import Path

HERE = Path(__file__).resolve()
PACKAGE_ROOT = HERE.parents[3]  # Appendix/Section_D/scripts/<file> -> parents[0]=scripts,[1]=Section_D,[2]=Appendix,[3]=package root
SOURCE_SECTION_ROOT = PACKAGE_ROOT / "artifacts" / "5_3_targeted_perturbation_response"
INPUT_CSV = SOURCE_SECTION_ROOT / "results/manuscript_tables/k_sensitivity_wild_b_table.csv"
OUTPUT_TEX = HERE.parents[1] / "results" / "appendix_k_value_justification.tex"

MODEL_LABELS = {
    "LR": "LR",
    "LightGBM-gain": "LGBM-gain",
    "LightGBM-permutation": "LGBM-perm",
}
SUBSET_LABELS = {
    "Header": "Header (Imp.)",
    "Imports": "Imports (Imp.)",
}
MODEL_ORDER = ["LR", "LightGBM-gain", "LightGBM-permutation"]
SUBSET_ORDER = ["Header", "Imports"]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def find_row(rows: list[dict[str, str]], model: str, subset: str) -> dict[str, str]:
    for row in rows:
        if row["Model"] == model and row["Subset"] == subset and row["Masking"] == "Important":
            return row
    raise KeyError(f"No Important row for {model}/{subset}")


def build_tex(rows: list[dict[str, str]]) -> str:
    blocks = []
    for model in MODEL_ORDER:
        lines = [f"\\multirow{{2}}{{*}}{{{MODEL_LABELS[model]}}}"]
        for subset in SUBSET_ORDER:
            row = find_row(rows, model, subset)
            k5, k10, k15, k20 = row["k=5"], row["k=10"], row["k=15"], row["k=20"]
            lines.append(f" & {SUBSET_LABELS[subset]:<15} & {k5} & {k10} & {k15} & {k20} \\\\")
        blocks.append("\n".join(lines))
    body = "\n\n\\midrule\n\n".join(blocks)

    return f"""\\begin{{table}}[t]
\\centering
\\caption{{Sensitivity of $\\Delta$Mix@$k$ to neighborhood size at 10\\% important-feature masking.}}
\\label{{tab:appendix_k_value_justification}}
\\scriptsize
\\setlength{{\\tabcolsep}}{{2pt}}
\\resizebox{{\\columnwidth}}{{!}}{{%
\\begin{{tabular}}{{llcccc}}
\\toprule
Model & Subset & $k=5$ & $k=10$ & $k=15$ & $k=20$ \\\\
\\midrule
{body}

\\bottomrule
\\end{{tabular}}
}}
\\end{{table}}
"""


def main() -> None:
    rows = read_rows(INPUT_CSV)
    OUTPUT_TEX.write_text(build_tex(rows), encoding="utf-8")
    print(f"Wrote {OUTPUT_TEX}")


if __name__ == "__main__":
    main()
