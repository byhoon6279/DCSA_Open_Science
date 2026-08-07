#!/usr/bin/env python3
"""Summarize the D3 (density-conditioned reliability, Section 5.4) class-conditional
re-binning sensitivity check backing Appendix D.1.

Split out of a combined `summarize_class_conditional_sensitivity.py`
(which summarized both D3 and D4 in one file) so that D3 assets live under
5_4_density_conditioned_reliability/ per this package's section-first layout. See the
sibling D4 script under 5_5_density_conditioned_fragility/sensitivity/class_conditional/
for the fragility half.

Reads the per-model run directories under ../results/{LR,LightGBM,RF}/ (already
bundled) and writes ../results/d3_class_conditional_comparison.csv plus the D3 section
of ../results/summary.md.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

HERE = Path(__file__).resolve()
RESULT_ROOT = HERE.parents[1] / "results"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def resolve_result_file(base_dir: Path, filename: str, preferred_subdir: str | None = None) -> Path:
    direct = base_dir / filename
    if direct.exists():
        return direct
    if preferred_subdir is not None:
        preferred = base_dir / preferred_subdir / filename
        if preferred.exists():
            return preferred
    candidates = sorted(base_dir.glob(f"**/{filename}"))
    if not candidates:
        raise FileNotFoundError(f"Could not find {filename} under {base_dir}")
    return candidates[0]


def as_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value == "" or value is None:
        return float("nan")
    return float(value)


def find_row(rows: list[dict[str, str]], **matches: str) -> dict[str, str] | None:
    for row in rows:
        if all(row.get(key) == value for key, value in matches.items()):
            return row
    return None


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def require_count(rows: list[dict[str, object]], expected: int, label: str) -> None:
    if len(rows) != expected:
        raise RuntimeError(
            f"{label} row count mismatch: expected {expected}, found {len(rows)}. "
            "This usually means one or more reruns failed or produced incomplete outputs."
        )


def summarize_d3() -> list[dict[str, object]]:
    jobs = [
        ("LR", "density_reliability_controlled_main"),
        ("LightGBM", "density_reliability_controlled_main_lightgbm"),
        ("RF", "rf_full_wild_b"),
    ]
    out: list[dict[str, object]] = []
    for model, run_name in jobs:
        run_dir = RESULT_ROOT / model / run_name
        preferred_subdir = "k_10" if model == "LightGBM" else None
        pooled = read_csv_rows(
            resolve_result_file(run_dir, "aggregate_metric_rows.csv", preferred_subdir=preferred_subdir)
        )
        class_cond = read_csv_rows(
            resolve_result_file(
                run_dir,
                "aggregate_class_conditional_metric_rows.csv",
                preferred_subdir=preferred_subdir,
            )
        )
        for feature_group in ["all", "header", "imports", "section", "strings"]:
            pooled_low = find_row(pooled, density_bin="low_density", feature_group=feature_group)
            pooled_high = find_row(pooled, density_bin="high_density", feature_group=feature_group)
            cc_low = find_row(
                class_cond,
                density_bin="low_density",
                feature_group=feature_group,
                density_definition="class_conditional",
            )
            cc_high = find_row(
                class_cond,
                density_bin="high_density",
                feature_group=feature_group,
                density_definition="class_conditional",
            )
            if not pooled_low or not pooled_high or not cc_low or not cc_high:
                continue
            out.append(
                {
                    "model": model,
                    "feature_group": feature_group,
                    "pooled_auc_gap_low_minus_high": as_float(pooled_low, "auc_mean") - as_float(pooled_high, "auc_mean"),
                    "class_cond_auc_gap_low_minus_high": as_float(cc_low, "auc_mean") - as_float(cc_high, "auc_mean"),
                    "pooled_mix_gap_low_minus_high": as_float(pooled_low, "mix_at_10_mean") - as_float(pooled_high, "mix_at_10_mean"),
                    "class_cond_mix_gap_low_minus_high": as_float(cc_low, "mix_at_10_mean") - as_float(cc_high, "mix_at_10_mean"),
                    "class_balanced_mix_gap_low_minus_high": as_float(cc_low, "mix_at_10_class_balanced_mean") - as_float(cc_high, "mix_at_10_class_balanced_mean"),
                    "pooled_js_gap_low_minus_high": as_float(pooled_low, "js_divergence_mean") - as_float(pooled_high, "js_divergence_mean"),
                    "class_cond_js_gap_low_minus_high": as_float(cc_low, "js_divergence_mean") - as_float(cc_high, "js_divergence_mean"),
                }
            )
    return out


def direction_label(value: float, positive_meaning: str, negative_meaning: str) -> str:
    if math.isnan(value):
        return "missing"
    if value > 0:
        return positive_meaning
    if value < 0:
        return negative_meaning
    return "flat"


def write_summary_md(d3_rows: list[dict[str, object]]) -> None:
    lines = ["# D3 Density Class-Conditional Sensitivity Summary (Section 5.4 / Appendix D.1)", ""]
    for row in d3_rows:
        lines.append(
            f"- {row['model']} / {row['feature_group']}: "
            f"pooled AUC low-high={row['pooled_auc_gap_low_minus_high']:.4f}, "
            f"class-conditional AUC low-high={row['class_cond_auc_gap_low_minus_high']:.4f}; "
            f"class-balanced Mix low-high={row['class_balanced_mix_gap_low_minus_high']:.4f} "
            f"({direction_label(float(row['class_balanced_mix_gap_low_minus_high']), 'more mixing in low density', 'less mixing in low density')})."
        )
    (RESULT_ROOT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    d3_rows = summarize_d3()
    require_count(d3_rows, 15, "D3 class-conditional summary")
    write_csv(RESULT_ROOT / "d3_class_conditional_comparison.csv", d3_rows)
    write_summary_md(d3_rows)


if __name__ == "__main__":
    main()
