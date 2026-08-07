#!/usr/bin/env python3
"""
Appendix Table G.4: density-conditioned masking amplification in the MLP
extension. `Delta flip` is the low-density flip rate minus the high-density
flip rate.

Recomputes the table directly from the raw per-seed-per-strength CSV already
bundled at
`MLP/results/5_5/mlp_rq3_density_fragility_wild_b_main/amplification_rows.csv`
(mean and positive-case count over the 5 seeds x 3 masking strengths = 15
conditions per feature group).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[3]
AMPLIFICATION_ROWS_PATH = (
    REPO_ROOT / "artifacts" / "MLP" / "results" / "5_5" / "mlp_rq3_density_fragility_wild_b_main" / "amplification_rows.csv"
)
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "results"

FEATURE_GROUPS = ["all", "header", "section", "imports", "strings"]
GROUP_LABELS = {"all": "All", "header": "Header", "section": "Section", "imports": "Imports", "strings": "Strings"}


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def compute_cell(rows: List[Dict[str, str]], group: str, mask_type: str) -> Dict[str, object]:
    sub = [float(row["delta_flip_density"]) for row in rows if row["feature_group"] == group and row["mask_type"] == mask_type]
    positive = sum(1 for value in sub if value > 0)
    return {"mean_delta_flip": sum(sub) / len(sub), "positive_cases": positive, "n": len(sub)}


def compute_table() -> List[Dict[str, object]]:
    rows = load_rows(AMPLIFICATION_ROWS_PATH)
    rows_out: List[Dict[str, object]] = []
    for group in FEATURE_GROUPS:
        important = compute_cell(rows, group, "important")
        random_ = compute_cell(rows, group, "random")
        rows_out.append(
            {
                "feature_group": group,
                "important_mean_delta_flip": important["mean_delta_flip"],
                "important_positive_cases": important["positive_cases"],
                "important_n": important["n"],
                "random_mean_delta_flip": random_["mean_delta_flip"],
                "random_positive_cases": random_["positive_cases"],
                "random_n": random_["n"],
            }
        )
    return rows_out


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "feature_group",
        "important_mean_delta_flip",
        "important_positive_cases",
        "important_n",
        "random_mean_delta_flip",
        "random_positive_cases",
        "random_n",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt_signed(value: float) -> str:
    return f"${value:+.4f}$".replace("+", " ") if value >= 0 else f"${value:.4f}$"


def fmt_plain(value: float) -> str:
    return f"${value:.4f}$"


def write_tex(path: Path, rows: List[Dict[str, object]]) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Density-conditioned masking amplification in the MLP extension.",
        r"$\Delta\mathrm{flip}$ denotes the low-density flip rate minus the high-density flip rate.}",
        r"\label{tab:app_mlp_density_amplification_full}",
        r"",
        r"\resizebox{0.95\columnwidth}{!}{%",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"& \multicolumn{2}{c}{\textbf{Important masking}}",
        r"& \multicolumn{2}{c}{\textbf{Random masking}} \\",
        r"\cmidrule(lr){2-3}",
        r"\cmidrule(lr){4-5}",
        r"\textbf{Group}",
        r"& \textbf{Mean $\Delta\mathrm{flip}$}",
        r"& \textbf{Positive cases}",
        r"& \textbf{Mean $\Delta\mathrm{flip}$}",
        r"& \textbf{Positive cases} \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{GROUP_LABELS[row['feature_group']]:<7} & {fmt_signed(row['important_mean_delta_flip'])} & "
            f"{row['important_positive_cases']:>2} / {row['important_n']} & {fmt_plain(row['random_mean_delta_flip'])} & "
            f"{row['random_positive_cases']:>2} / {row['random_n']} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}%", "}", r"\end{table}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = compute_table()
    write_csv(OUTPUT_DIR / "mlp_density_masking_full.csv", rows)
    write_tex(OUTPUT_DIR / "appendix_mlp_density_masking_full.tex", rows)
    for row in rows:
        print(
            f"{row['feature_group']:<8}: imp mean={row['important_mean_delta_flip']:.4f} "
            f"({row['important_positive_cases']}/{row['important_n']})  "
            f"rnd mean={row['random_mean_delta_flip']:.4f} ({row['random_positive_cases']}/{row['random_n']})"
        )
    print(f"Wrote {OUTPUT_DIR / 'mlp_density_masking_full.csv'} and appendix_mlp_density_masking_full.tex")


if __name__ == "__main__":
    main()
