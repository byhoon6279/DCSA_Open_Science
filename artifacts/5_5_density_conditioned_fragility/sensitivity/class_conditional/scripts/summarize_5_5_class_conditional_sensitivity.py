#!/usr/bin/env python3
"""Summarize the D4 (density-conditioned fragility, Section 5.5) class-conditional
re-binning sensitivity check backing Appendix D.1.

Split out of a combined `summarize_class_conditional_sensitivity.py`
(which summarized both D3 and D4 in one file) so that D4 assets live under
5_5_density_conditioned_fragility/ per this package's section-first layout. See the
sibling D3 script under 5_4_density_conditioned_reliability/sensitivity/class_conditional/
for the reliability half.

Reads the per-model run directories under ../results/{LR,LightGBM,RF}/ (already
bundled) and writes ../results/d4_class_conditional_comparison.csv plus the D4 section
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


def summarize_d4() -> list[dict[str, object]]:
    jobs = [
        ("LR", "density_fragility_controlled_main"),
        ("LightGBM", "density_fragility_controlled_main_lightgbm_permutation"),
        ("RF", "rf_full_wild_b"),
    ]
    out: list[dict[str, object]] = []
    for model, run_name in jobs:
        run_dir = RESULT_ROOT / model / run_name
        pooled = read_csv_rows(
            resolve_result_file(run_dir, "aggregate_amplification_rows.csv")
        )
        class_cond = read_csv_rows(
            resolve_result_file(run_dir, "aggregate_class_conditional_amplification_rows.csv")
        )
        detail = read_csv_rows(
            resolve_result_file(run_dir, "aggregate_class_conditional_seed_summary_rows.csv")
        )
        for feature_group in ["header", "imports"]:
            for perturbation_type in ["important", "random"]:
                for strength in ["0.01", "0.05", "0.1"]:
                    pooled_row = find_row(
                        pooled,
                        feature_group=feature_group,
                        perturbation_type=perturbation_type,
                        strength=strength,
                    )
                    cc_row = find_row(
                        class_cond,
                        feature_group=feature_group,
                        density_definition="class_conditional",
                        perturbation_type=perturbation_type,
                        strength=strength,
                    )
                    cc_low = find_row(
                        detail,
                        feature_group=feature_group,
                        density_bin="low_density",
                        density_definition="class_conditional",
                        perturbation_type=perturbation_type,
                        strength=strength,
                    )
                    cc_high = find_row(
                        detail,
                        feature_group=feature_group,
                        density_bin="high_density",
                        density_definition="class_conditional",
                        perturbation_type=perturbation_type,
                        strength=strength,
                    )
                    if not pooled_row or not cc_row or not cc_low or not cc_high:
                        continue
                    out.append(
                        {
                            "model": model,
                            "feature_group": feature_group,
                            "perturbation_type": perturbation_type,
                            "strength": strength,
                            "pooled_auc_drop_density": as_float(pooled_row, "auc_drop_density_mean"),
                            "class_cond_auc_drop_density": as_float(cc_row, "auc_drop_density_mean"),
                            "pooled_flip_density": as_float(pooled_row, "delta_flip_density_mean"),
                            "class_cond_flip_density": as_float(cc_row, "delta_flip_density_mean"),
                            "class_cond_low_m2b": as_float(cc_low, "malware_to_benign_rate_cond_mean_mean"),
                            "class_cond_high_m2b": as_float(cc_high, "malware_to_benign_rate_cond_mean_mean"),
                            "class_cond_low_b2m": as_float(cc_low, "benign_to_malware_rate_cond_mean_mean"),
                            "class_cond_high_b2m": as_float(cc_high, "benign_to_malware_rate_cond_mean_mean"),
                        }
                    )
    return out


def write_summary_md(d4_rows: list[dict[str, object]]) -> None:
    lines = ["# D4 Density Class-Conditional Sensitivity Summary (Section 5.5 / Appendix D.1)", ""]
    for row in d4_rows:
        lines.append(
            f"- {row['model']} / {row['feature_group']} / {row['perturbation_type']} / {row['strength']}: "
            f"pooled low-high AUC-drop gap={row['pooled_auc_drop_density']:.4f}, "
            f"class-conditional low-high AUC-drop gap={row['class_cond_auc_drop_density']:.4f}; "
            f"class-conditional M->B low/high={row['class_cond_low_m2b']:.4f}/{row['class_cond_high_m2b']:.4f}, "
            f"B->M low/high={row['class_cond_low_b2m']:.4f}/{row['class_cond_high_b2m']:.4f}."
        )
    (RESULT_ROOT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    d4_rows = summarize_d4()
    require_count(d4_rows, 36, "D4 class-conditional summary")
    write_csv(RESULT_ROOT / "d4_class_conditional_comparison.csv", d4_rows)
    write_summary_md(d4_rows)


if __name__ == "__main__":
    main()
