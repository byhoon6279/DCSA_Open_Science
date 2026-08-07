#!/usr/bin/env python3
"""Build Appendix Table A.1 (`appendix_family_distribution.tex`) from the
already-bundled family-composition CSV.

`export_family_composition_csv.py` in this same directory produces the
underlying `table_family_composition_original_vs_unpacked.csv`; this script
builds the manuscript LaTeX from it.

Per family and view (In-the-wild / Unpacked): # Samples = train_effective +
test_effective (both 0 when the family was not selected for that view's
top-20 set, per `original_included`/`unpacked_included` in the source CSV,
regardless of any nonzero raw counts). # Train and # Test are the effective
(capped) counts directly.
"""
from __future__ import annotations

import csv
from pathlib import Path

HERE = Path(__file__).resolve()
PACKAGE_ROOT = HERE.parents[3]  # DCSA_Open_Science/ (Appendix/Section_A/scripts/<file> -> parents[0]=scripts,[1]=Section_A,[2]=Appendix,[3]=package root)
SOURCE_SECTION_ROOT = PACKAGE_ROOT / "artifacts" / "5_1_representation_level_separability_disagreement"
INPUT_CSV = SOURCE_SECTION_ROOT / "results/LR/table_family_composition_original_vs_unpacked.csv"
OUTPUT_TEX = HERE.parents[1] / "results" / "appendix_family_distribution.tex"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def fmt(n: int) -> str:
    return f"{n:,}"


def build_tex(rows: list[dict[str, str]]) -> str:
    body_lines = []
    total_orig_train = total_orig_test = total_unp_train = total_unp_test = 0
    for row in rows:
        orig_train = int(row["original_train_effective"])
        orig_test = int(row["original_test_effective"])
        unp_train = int(row["unpacked_train_effective"])
        unp_test = int(row["unpacked_test_effective"])
        orig_samples = orig_train + orig_test
        unp_samples = unp_train + unp_test
        total_orig_train += orig_train
        total_orig_test += orig_test
        total_unp_train += unp_train
        total_unp_test += unp_test
        body_lines.append(
            f"{row['family']:<12}& {fmt(orig_samples):<6}& {fmt(orig_train):<4}& {fmt(orig_test):<6}"
            f"& {fmt(unp_samples):<6}& {fmt(unp_train):<4}& {fmt(unp_test)} \\\\"
        )
    body = "\n".join(body_lines)

    total_orig_samples = total_orig_train + total_orig_test
    total_unp_samples = total_unp_train + total_unp_test

    return f"""\\begin{{table}}[t]
\\centering
\\caption{{Malware-family composition in the Wild (U) and Unpacked (B) views.
A zero indicates that the family was not selected for that view, not that no raw samples were available.}}
\\label{{tab:appendix_family_distribution}}
\\scriptsize
\\setlength{{\\tabcolsep}}{{2pt}}
\\resizebox{{\\columnwidth}}{{!}}{{%
\\begin{{tabular}}{{lrrr|rrr}}
\\toprule
\\multirow{{2}}{{*}}{{\\textbf{{Family}}}} 
& \\multicolumn{{3}}{{c|}}{{\\textbf{{In-the-wild}}}} 
& \\multicolumn{{3}}{{c}}{{\\textbf{{Unpacked}}}} \\\\
\\cmidrule(lr){{2-4}} \\cmidrule(lr){{5-7}}
& \\textbf{{\\# Samples}} & \\textbf{{\\# Train}} & \\textbf{{\\# Test}}
& \\textbf{{\\# Samples}} & \\textbf{{\\# Train}} & \\textbf{{\\# Test}} \\\\
\\midrule
{body}
\\midrule
\\textbf{{Total}} & \\textbf{{{fmt(total_orig_samples)}}} & \\textbf{{{fmt(total_orig_train)}}} & \\textbf{{{fmt(total_orig_test)}}}
& \\textbf{{{fmt(total_unp_samples)}}} & \\textbf{{{fmt(total_unp_train)}}} & \\textbf{{{fmt(total_unp_test)}}} \\\\
\\bottomrule
\\end{{tabular}}%
}}
\\end{{table}}"""


def main() -> None:
    rows = read_rows(INPUT_CSV)
    OUTPUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_TEX.write_text(build_tex(rows), encoding="utf-8")
    print(f"Wrote {OUTPUT_TEX}")


if __name__ == "__main__":
    main()
