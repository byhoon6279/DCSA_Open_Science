#!/usr/bin/env python3
"""
Appendix Table G.2: full MLP masking results in Wild (B).

Recomputes the table directly from the raw per-seed(-repeat) CSV already
bundled at `MLP/results/5_3/mlp_rq2_targeted_masking_wild_b_main/masking_rows.csv`.

Important-masking columns (delta_auc, flip_rate, malware_to_benign_flip_rate,
benign_to_malware_flip_rate) are averaged directly over the 5 seeds (one row
per seed; masking is deterministic). Random-masking columns are averaged over
the 3 control repeats within each seed first, then over the 5 seeds, per the
manuscript caption.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[3]
MASKING_ROWS_PATH = (
    REPO_ROOT / "artifacts" / "MLP" / "results" / "5_3" / "mlp_rq2_targeted_masking_wild_b_main" / "masking_rows.csv"
)
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "results"

FEATURE_GROUPS = ["all", "header", "section", "imports", "strings"]
STRENGTHS = ["0.01", "0.05", "0.1"]
STRENGTH_LABELS = {"0.01": "1\\%", "0.05": "5\\%", "0.1": "10\\%"}
GROUP_LABELS = {"all": "All", "header": "Header", "section": "Section", "imports": "Imports", "strings": "Strings"}


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def mean(values: List[float]) -> float:
    return sum(values) / len(values)


def compute_cell(rows: List[Dict[str, str]], group: str, strength: str) -> Dict[str, float]:
    important = [row for row in rows if row["feature_group"] == group and row["mask_type"] == "important" and row["strength"] == strength]
    random_ = [row for row in rows if row["feature_group"] == group and row["mask_type"] == "random" and row["strength"] == strength]

    by_seed: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in random_:
        by_seed[row["seed"]].append(row)
    seed_auc_means = [mean([float(row["delta_auc"]) for row in seed_rows]) for seed_rows in by_seed.values()]
    seed_flip_means = [mean([float(row["flip_rate"]) for row in seed_rows]) for seed_rows in by_seed.values()]

    return {
        "important_delta_auc": mean([float(row["delta_auc"]) for row in important]),
        "important_flip_rate": mean([float(row["flip_rate"]) for row in important]),
        "important_malware_to_benign": mean([float(row["malware_to_benign_flip_rate"]) for row in important]),
        "important_benign_to_malware": mean([float(row["benign_to_malware_flip_rate"]) for row in important]),
        "random_delta_auc": mean(seed_auc_means),
        "random_flip_rate": mean(seed_flip_means),
    }


def compute_table() -> List[Dict[str, object]]:
    rows = load_rows(MASKING_ROWS_PATH)
    rows_out: List[Dict[str, object]] = []
    for group in FEATURE_GROUPS:
        for strength in STRENGTHS:
            cell = compute_cell(rows, group, strength)
            rows_out.append({"feature_group": group, "strength": strength, **cell})
    return rows_out


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "feature_group",
        "strength",
        "important_delta_auc",
        "important_flip_rate",
        "important_malware_to_benign",
        "important_benign_to_malware",
        "random_delta_auc",
        "random_flip_rate",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float) -> str:
    if abs(value) < 1e-9:
        value = 0.0
    # Python's f"{value:.4f}" rounds on the value's binary float64
    # representation, which for several cells here (e.g. a mean of exactly
    # 0.33945) is actually a hair below the decimal midpoint and rounds down
    # (0.3394) instead of up (0.3395, the value reported in the manuscript).
    # Round on the decimal string instead, with the conventional
    # round-half-up tie-break, to match.
    text = str(Decimal(str(value)).quantize(Decimal("1.0000"), rounding=ROUND_HALF_UP))
    return f"${text}$" if value < 0 else text


def write_tex(path: Path, rows: List[Dict[str, object]]) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{MLP important- and random-masking results on Wild (B), averaged across five seeds.}",
        r"\label{tab:app_mlp_masking_full}",
        r"",
        r"\resizebox{0.9\columnwidth}{!}{%",
        r"\begin{tabular}{llcccccc}",
        r"\toprule",
        r"& & \multicolumn{4}{c}{\textbf{Important masking}}",
        r"& \multicolumn{2}{c}{\textbf{Random masking}} \\",
        r"\cmidrule(lr){3-6}",
        r"\cmidrule(lr){7-8}",
        r"\textbf{Group} & \textbf{Ratio}",
        r"& \textbf{AUC degradation} & \textbf{Flip rate}",
        r"& \textbf{M$\rightarrow$B} & \textbf{B$\rightarrow$M}",
        r"& \textbf{AUC degradation} & \textbf{Flip rate} \\",
        r"\midrule",
        r"",
    ]
    for group in FEATURE_GROUPS:
        lines.append(f"\\multirow{{3}}{{*}}{{{GROUP_LABELS[group]}}}")
        for strength in STRENGTHS:
            cell = next(row for row in rows if row["feature_group"] == group and row["strength"] == strength)
            lines.append(
                f"& {STRENGTH_LABELS[strength]:<4} & {fmt(cell['important_delta_auc'])} & {fmt(cell['important_flip_rate'])} "
                f"& {fmt(cell['important_malware_to_benign'])} & {fmt(cell['important_benign_to_malware'])} "
                f"& {fmt(cell['random_delta_auc'])} & {fmt(cell['random_flip_rate'])} \\\\"
            )
        lines.append(r"\midrule")
        lines.append("")
    lines.pop()  # drop the trailing \midrule before \bottomrule
    lines.pop()
    lines.extend([r"\bottomrule", r"\end{tabular}%", "}", r"\end{table}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = compute_table()
    write_csv(OUTPUT_DIR / "mlp_masking_full.csv", rows)
    write_tex(OUTPUT_DIR / "appendix_mlp_masking_full.tex", rows)
    for row in rows:
        print(
            f"{row['feature_group']:<8} {row['strength']:<5}: imp dAUC={row['important_delta_auc']:.4f} "
            f"flip={row['important_flip_rate']:.4f} M2B={row['important_malware_to_benign']:.4f} "
            f"B2M={row['important_benign_to_malware']:.4f}  rnd dAUC={row['random_delta_auc']:.4f} "
            f"flip={row['random_flip_rate']:.4f}"
        )
    print(f"Wrote {OUTPUT_DIR / 'mlp_masking_full.csv'} and appendix_mlp_masking_full.tex")


if __name__ == "__main__":
    main()
