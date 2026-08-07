#!/usr/bin/env python3
"""Build Table 6 (`density_import_intervention.tex`, LightGBM-only) from the
already-bundled block3 import-intervention compact CSV.

This script builds the manuscript LaTeX from
`results/manuscript_tables/density_import_intervention.csv` (itself produced
by `../common/run_density_conditioned_import_intervention_probe.py` +
`../common/merge_density_conditioned_import_intervention_probe_results.py`,
already bundled).

Note on the Delta(Low-High) column: the manuscript computes it from the
*rounded* (3-decimal) High/Low values printed in the table, not from the CSV's
own unrounded `delta_low_high` field — e.g. Important: round(0.4096, 3) -
round(0.1653, 3) = 0.410 - 0.165 = +0.245, not the CSV's raw +0.244. This
script replicates that "round, then subtract" convention to match exactly.
"""
from __future__ import annotations

import csv
from pathlib import Path

HERE = Path(__file__).resolve()
SECTION_ROOT = HERE.parents[2]  # 5_5_density_conditioned_fragility/
INPUT_CSV = SECTION_ROOT / "results/manuscript_tables/density_import_intervention.csv"
OUTPUT_TEX = SECTION_ROOT / "results/manuscript_tables/density_import_intervention.tex"

ROW_ORDER = ["Important", "Random"]


def read_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["intervention"]: row for row in csv.DictReader(handle)}


def fmt3(value: str) -> float:
    return round(float(value), 3)


def build_tex(rows: dict[str, dict[str, str]]) -> str:
    body_lines = []
    for intervention in ROW_ORDER:
        row = rows[intervention]
        pooled = fmt3(row["pooled_malware_to_benign_rate"])
        high = fmt3(row["high_density_malware_to_benign_rate"])
        mid = fmt3(row["mid_density_malware_to_benign_rate"])
        low = fmt3(row["low_density_malware_to_benign_rate"])
        delta = low - high
        if intervention == "Important":
            line = (
                f"{intervention} & {pooled:.3f} & \\textbf{{{high:.3f}}} & {mid:.3f} & "
                f"\\textbf{{{low:.3f}}} & \\textbf{{+{delta:.3f}}} \\\\"
            )
        else:
            line = f"{intervention}    & {pooled:.3f} & {high:.3f} & {mid:.3f} & {low:.3f} & +{delta:.3f} \\\\"
        body_lines.append(line)
    body = "\n".join(body_lines)

    return f"""\\begin{{table}}[t]
\\centering
\\caption{{Density-conditioned malware-to-benign transition rates under the 10\\% hashed import-feature intervention in LightGBM.}}
\\label{{tab:density_import_intervention}}
\\small
\\setlength{{\\tabcolsep}}{{5pt}}
\\begin{{tabular}}{{lccccc}}
\\toprule
\\textbf{{Intervention}} & \\textbf{{Pooled}} & \\textbf{{High}} & \\textbf{{Mid}} & \\textbf{{Low}} & \\textbf{{$\\Delta$(Low-High)}}\\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}

\\begin{{minipage}}{{\\linewidth}}
\\footnotesize
\\textit{{Note.}}
This table reports a separate five-seed density-stratified probe that
deliberately reuses the PE-inspired intervention importance-ranking pipeline
used for Table 7 to isolate the density effect, pooling test weeks and
aggregating across density strata
instead of averaging per-week measurements as in \\autoref{{tab:rq2_pe_import_validation}}.
Both tables therefore select the same important dimensions and report the
same Important row; the Random row differs because each analysis draws its
random dimensions from an independently seeded sample.
\\end{{minipage}}
\\end{{table}}
"""


def main() -> None:
    rows = read_rows(INPUT_CSV)
    OUTPUT_TEX.write_text(build_tex(rows), encoding="utf-8")
    print(f"Wrote {OUTPUT_TEX}")


if __name__ == "__main__":
    main()
