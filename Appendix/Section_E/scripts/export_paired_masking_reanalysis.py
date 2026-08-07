#!/usr/bin/env python3
"""Computes the paired seed-level masking reanalysis (raw rows, worst-case
summary, input-availability audit, and a narrative report), and builds the
two manuscript-facing worst-case summary tables via `build_metric_table()`
(Appendix Table E.1: AUC degradation, Table E.2: prediction flip rate; both
span LR, LightGBM, RF, and the MLP extension), written to
`significance_masking_auc.tex` / `significance_masking_flip.tex`."""
from __future__ import annotations

import csv
import itertools
import json
import math
import random
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Sequence

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260805


# This script reads its source data from artifacts/5_3_targeted_perturbation_response
# (and artifacts/MLP), and writes its output into this section's own
# results/paired_reanalysis/, rather than into the artifacts/ tree.
PACKAGE_ROOT = Path(__file__).resolve().parents[3]
BASE_DIR = PACKAGE_ROOT / "artifacts" / "5_3_targeted_perturbation_response"
MLP_BASE_DIR = PACKAGE_ROOT / "artifacts" / "MLP"
OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "paired_reanalysis"


@dataclass(frozen=True)
class MainRun:
    model: str
    outcome: str
    feature_group: str
    family: str
    path: Path
    metric_column: str
    effect_sign: float


@dataclass(frozen=True)
class MLPRun:
    outcome: str
    feature_group: str
    family: str
    path: Path
    metric_column: str
    effect_sign: float


MAIN_RUNS: Sequence[MainRun] = (
    MainRun(
        model="LR",
        outcome="auc_drop",
        feature_group="header",
        family="main_models",
        path=BASE_DIR / "results" / "LR" / "feature_perturbation_balanced_main" / "k_10" / "trial_results.csv",
        metric_column="delta_auc",
        effect_sign=-1.0,
    ),
    MainRun(
        model="LR",
        outcome="auc_drop",
        feature_group="imports",
        family="main_models",
        path=BASE_DIR / "results" / "LR" / "feature_perturbation_balanced_main" / "k_10" / "trial_results.csv",
        metric_column="delta_auc",
        effect_sign=-1.0,
    ),
    MainRun(
        model="LR",
        outcome="flip_rate",
        feature_group="header",
        family="main_models",
        path=BASE_DIR / "results" / "LR" / "prediction_flip_balanced_main" / "trial_results.csv",
        metric_column="flip_rate",
        effect_sign=1.0,
    ),
    MainRun(
        model="LR",
        outcome="flip_rate",
        feature_group="imports",
        family="main_models",
        path=BASE_DIR / "results" / "LR" / "prediction_flip_balanced_main" / "trial_results.csv",
        metric_column="flip_rate",
        effect_sign=1.0,
    ),
    MainRun(
        model="LightGBM",
        outcome="auc_drop",
        feature_group="header",
        family="main_models",
        path=BASE_DIR / "results" / "LightGBM" / "feature_perturbation_lightgbm_permutation_balanced_main" / "k_10" / "trial_results.csv",
        metric_column="delta_auc",
        effect_sign=-1.0,
    ),
    MainRun(
        model="LightGBM",
        outcome="auc_drop",
        feature_group="imports",
        family="main_models",
        path=BASE_DIR / "results" / "LightGBM" / "feature_perturbation_lightgbm_permutation_balanced_main" / "k_10" / "trial_results.csv",
        metric_column="delta_auc",
        effect_sign=-1.0,
    ),
    MainRun(
        model="LightGBM",
        outcome="flip_rate",
        feature_group="header",
        family="main_models",
        path=BASE_DIR / "results" / "LightGBM" / "prediction_flip_lightgbm_balanced_main" / "trial_results.csv",
        metric_column="flip_rate",
        effect_sign=1.0,
    ),
    MainRun(
        model="LightGBM",
        outcome="flip_rate",
        feature_group="imports",
        family="main_models",
        path=BASE_DIR / "results" / "LightGBM" / "prediction_flip_lightgbm_balanced_main" / "trial_results.csv",
        metric_column="flip_rate",
        effect_sign=1.0,
    ),
    MainRun(
        model="RF",
        outcome="auc_drop",
        feature_group="header",
        family="main_models",
        path=BASE_DIR / "results" / "RF" / "rq2_2_rf_full_wild_b" / "trial_results.csv",
        metric_column="delta_auc",
        effect_sign=-1.0,
    ),
    MainRun(
        model="RF",
        outcome="auc_drop",
        feature_group="imports",
        family="main_models",
        path=BASE_DIR / "results" / "RF" / "rq2_2_rf_full_wild_b" / "trial_results.csv",
        metric_column="delta_auc",
        effect_sign=-1.0,
    ),
    MainRun(
        model="RF",
        outcome="flip_rate",
        feature_group="header",
        family="main_models",
        path=BASE_DIR / "results" / "RF" / "rq2_3_rf_full_wild_b" / "trial_results.csv",
        metric_column="flip_rate",
        effect_sign=1.0,
    ),
    MainRun(
        model="RF",
        outcome="flip_rate",
        feature_group="imports",
        family="main_models",
        path=BASE_DIR / "results" / "RF" / "rq2_3_rf_full_wild_b" / "trial_results.csv",
        metric_column="flip_rate",
        effect_sign=1.0,
    ),
)


MLP_RUNS: Sequence[MLPRun] = tuple(
    MLPRun(
        outcome=outcome,
        feature_group=feature_group,
        family="mlp_extension",
        path=(
            MLP_BASE_DIR / "results" / "5_3" / "mlp_rq2_targeted_masking_wild_b_main" / "masking_rows.csv"
        ),
        metric_column=metric_column,
        effect_sign=1.0,
    )
    for outcome, metric_column in (
        ("auc_drop", "delta_auc"),
        ("flip_rate", "flip_rate"),
    )
    for feature_group in ("all", "header", "section", "imports", "strings")
)


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def exact_sign_flip_pvalue(deltas: Sequence[float], direction: str = "greater") -> float:
    if not deltas:
        return float("nan")
    observed = sum(deltas) / len(deltas)
    if direction == "greater" and observed <= 0.0:
        return 1.0
    if direction == "less" and observed >= 0.0:
        return 1.0

    exceed_count = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(deltas)):
        total += 1
        signed_mean = sum(sign * delta for sign, delta in zip(signs, deltas)) / len(deltas)
        if direction == "greater":
            if signed_mean >= observed - 1e-12:
                exceed_count += 1
        else:
            if signed_mean <= observed + 1e-12:
                exceed_count += 1
    return exceed_count / total


def bootstrap_ci(
    deltas: Sequence[float],
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """Percentile bootstrap CI on the mean, resampling seed-level paired
    deltas with replacement (not the underlying seed-week observations,
    which are not independent within a seed)."""
    rng = random.Random(seed)
    n = len(deltas)
    resample_means = []
    for _ in range(n_resamples):
        resample_means.append(sum(deltas[rng.randrange(n)] for _ in range(n)) / n)
    resample_means.sort()
    lower_idx = int(round(0.025 * (n_resamples - 1)))
    upper_idx = int(round(0.975 * (n_resamples - 1)))
    return resample_means[lower_idx], resample_means[upper_idx]


def holm_adjust(values: Sequence[float]) -> List[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    adjusted = [1.0] * len(values)
    running = 0.0
    m = len(values)
    for rank, (original_idx, pvalue) in enumerate(indexed):
        candidate = min(1.0, (m - rank) * pvalue)
        running = max(running, candidate)
        adjusted[original_idx] = running
    return adjusted


def fmt_float(value: float, digits: int = 6) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.{digits}g}"


def seed_summary_fields(deltas: Sequence[float]) -> Dict[str, object]:
    return {
        "paired_mean_delta": sum(deltas) / len(deltas),
        "paired_median_delta": median(deltas),
        "seed_min_delta": min(deltas),
        "seed_max_delta": max(deltas),
        "positive_seed_count": sum(1 for value in deltas if value > 0),
        "nonnegative_seed_count": sum(1 for value in deltas if value >= 0),
        "seed_deltas_json": json.dumps([round(value, 12) for value in deltas]),
    }


def summarize_main_run(run: MainRun) -> tuple[List[Dict[str, object]], Dict[str, object]]:
    if not run.path.exists():
        return [], {
            "family": run.family,
            "model": run.model,
            "outcome": run.outcome,
            "feature_group": run.feature_group,
            "status": "missing_raw_rows",
            "path": str(run.path.relative_to(PACKAGE_ROOT)),
            "note": "Paired seed-week reanalysis requires raw trial rows.",
        }

    rows = read_rows(run.path)
    relevant = [
        row
        for row in rows
        if row["feature_group"] == run.feature_group
        and row["perturbation_type"] in {"important", "random"}
        and float(row["strength"]) in {0.01, 0.05, 0.1}
    ]
    if not relevant:
        return [], {
            "family": run.family,
            "model": run.model,
            "outcome": run.outcome,
            "feature_group": run.feature_group,
            "status": "no_matching_rows",
            "path": str(run.path.relative_to(PACKAGE_ROOT)),
            "note": "Raw file exists but no matching rows were found.",
        }

    records: List[Dict[str, object]] = []
    for strength in (0.01, 0.05, 0.1):
        by_seed_week: Dict[tuple[int, str], Dict[str, List[float]]] = {}
        for row in relevant:
            if not math.isclose(float(row["strength"]), strength, rel_tol=0.0, abs_tol=1e-12):
                continue
            key = (int(row["seed"]), row["test_week"])
            bucket = by_seed_week.setdefault(key, {"important": [], "random": []})
            bucket[row["perturbation_type"]].append(float(row[run.metric_column]))

        seed_to_week_deltas: Dict[int, List[float]] = {}
        for (seed, test_week), bucket in sorted(by_seed_week.items()):
            important_values = bucket["important"]
            random_values = bucket["random"]
            if len(important_values) != 1 or not random_values:
                continue
            cell_delta = run.effect_sign * (important_values[0] - (sum(random_values) / len(random_values)))
            seed_to_week_deltas.setdefault(seed, []).append(cell_delta)

        if len(seed_to_week_deltas) != 5:
            records.append(
                {
                    "family": run.family,
                    "model": run.model,
                    "feature_group": run.feature_group,
                    "outcome": run.outcome,
                    "strength": strength,
                    "status": "incomplete_pairing",
                    "path": str(run.path.relative_to(PACKAGE_ROOT)),
                    "n_seeds": len(seed_to_week_deltas),
                    "n_seed_weeks": sum(len(values) for values in seed_to_week_deltas.values()),
                }
            )
            continue

        seed_level_deltas = [
            sum(week_deltas) / len(week_deltas)
            for _, week_deltas in sorted(seed_to_week_deltas.items())
        ]
        entry: Dict[str, object] = {
            "family": run.family,
            "model": run.model,
            "feature_group": run.feature_group,
            "outcome": run.outcome,
            "strength": strength,
            "status": "ok",
            "path": str(run.path.relative_to(PACKAGE_ROOT)),
            "n_seeds": len(seed_level_deltas),
            "weeks_per_seed": min(len(values) for values in seed_to_week_deltas.values()),
            "direction": "greater",
            "raw_pvalue": exact_sign_flip_pvalue(seed_level_deltas, direction="greater"),
        }
        entry.update(seed_summary_fields(seed_level_deltas))
        entry["ci_low"], entry["ci_high"] = bootstrap_ci(seed_level_deltas)
        records.append(entry)

    return records, {
        "family": run.family,
        "model": run.model,
        "outcome": run.outcome,
        "feature_group": run.feature_group,
        "status": "ok" if any(record["status"] == "ok" for record in records) else "incomplete_pairing",
        "path": str(run.path.relative_to(PACKAGE_ROOT)),
        "note": "Used exact one-sided sign-flip test on 5 seed-level paired deltas.",
    }


def summarize_mlp_run(run: MLPRun) -> tuple[List[Dict[str, object]], Dict[str, object]]:
    if not run.path.exists():
        return [], {
            "family": run.family,
            "model": "MLP",
            "outcome": run.outcome,
            "feature_group": run.feature_group,
            "status": "missing_raw_rows",
            "path": str(run.path.relative_to(PACKAGE_ROOT)),
            "note": "MLP paired reanalysis requires masking_rows.csv.",
        }

    rows = read_rows(run.path)
    relevant = [
        row
        for row in rows
        if row["feature_group"] == run.feature_group
        and row["mask_type"] in {"important", "random"}
        and float(row["strength"]) in {0.01, 0.05, 0.1}
    ]
    records: List[Dict[str, object]] = []
    for strength in (0.01, 0.05, 0.1):
        seed_to_rows: Dict[int, Dict[str, List[float]]] = {}
        for row in relevant:
            if not math.isclose(float(row["strength"]), strength, rel_tol=0.0, abs_tol=1e-12):
                continue
            key = int(row["seed"])
            bucket = seed_to_rows.setdefault(key, {"important": [], "random": []})
            bucket[row["mask_type"]].append(float(row[run.metric_column]))

        if len(seed_to_rows) != 5:
            records.append(
                {
                    "family": run.family,
                    "model": "MLP",
                    "feature_group": run.feature_group,
                    "outcome": run.outcome,
                    "strength": strength,
                    "status": "incomplete_pairing",
                    "path": str(run.path.relative_to(PACKAGE_ROOT)),
                    "n_seeds": len(seed_to_rows),
                }
            )
            continue

        seed_level_deltas: List[float] = []
        random_repeats_seen = 0
        for _, bucket in sorted(seed_to_rows.items()):
            important_values = bucket["important"]
            random_values = bucket["random"]
            if len(important_values) != 1 or not random_values:
                continue
            random_repeats_seen = len(random_values)
            seed_level_deltas.append(
                run.effect_sign * (important_values[0] - (sum(random_values) / len(random_values)))
            )

        if len(seed_level_deltas) != 5:
            records.append(
                {
                    "family": run.family,
                    "model": "MLP",
                    "feature_group": run.feature_group,
                    "outcome": run.outcome,
                    "strength": strength,
                    "status": "incomplete_pairing",
                    "path": str(run.path.relative_to(PACKAGE_ROOT)),
                    "n_seeds": len(seed_level_deltas),
                }
            )
            continue

        entry: Dict[str, object] = {
            "family": run.family,
            "model": "MLP",
            "feature_group": run.feature_group,
            "outcome": run.outcome,
            "strength": strength,
            "status": "ok",
            "path": str(run.path.relative_to(PACKAGE_ROOT)),
            "n_seeds": len(seed_level_deltas),
            "random_repeats_per_seed": random_repeats_seen,
            "direction": "greater",
            "raw_pvalue": exact_sign_flip_pvalue(seed_level_deltas, direction="greater"),
        }
        entry.update(seed_summary_fields(seed_level_deltas))
        entry["ci_low"], entry["ci_high"] = bootstrap_ci(seed_level_deltas)
        records.append(entry)

    return records, {
        "family": run.family,
        "model": "MLP",
        "outcome": run.outcome,
        "feature_group": run.feature_group,
        "status": "ok" if any(record["status"] == "ok" for record in records) else "incomplete_pairing",
        "path": str(run.path.relative_to(PACKAGE_ROOT)),
        "note": "Used exact one-sided sign-flip test on 5 seed-level paired deltas.",
    }


def apply_holm(rows: List[Dict[str, object]]) -> None:
    for family in sorted({str(row["family"]) for row in rows if row.get("status") == "ok"}):
        family_rows = [row for row in rows if row.get("status") == "ok" and row["family"] == family]
        adjusted = holm_adjust([float(row["raw_pvalue"]) for row in family_rows])
        for row, adj in zip(family_rows, adjusted):
            row["holm_adjusted_pvalue"] = adj


def worst_case_rows(rows: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[tuple[str, str, str, str], List[Dict[str, object]]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        key = (
            str(row["family"]),
            str(row["model"]),
            str(row["feature_group"]),
            str(row["outcome"]),
        )
        grouped.setdefault(key, []).append(row)

    out: List[Dict[str, object]] = []
    for (_, model, feature_group, outcome), group_rows in sorted(grouped.items()):
        chosen = max(group_rows, key=lambda row: float(row["raw_pvalue"]))
        out.append(
            {
                "model": model,
                "feature_group": feature_group,
                "outcome": outcome,
                "worst_case_strength": chosen["strength"],
                "paired_mean_delta": chosen["paired_mean_delta"],
                "paired_median_delta": chosen["paired_median_delta"],
                "positive_seed_count": chosen["positive_seed_count"],
                "raw_pvalue": chosen["raw_pvalue"],
                "holm_adjusted_pvalue": chosen.get("holm_adjusted_pvalue", float("nan")),
                "ci_low": chosen["ci_low"],
                "ci_high": chosen["ci_high"],
                "seed_min_delta": chosen["seed_min_delta"],
                "seed_max_delta": chosen["seed_max_delta"],
                "seed_deltas_json": chosen["seed_deltas_json"],
            }
        )
    return out


MANUSCRIPT_TABLES_DIR = Path(__file__).resolve().parents[1] / "results" / "manuscript_tables"

# Row order for the manuscript-facing Table E.1/E.2 (grouped by model, then
# feature group in each model's natural definition order from MAIN_RUNS /
# MLP_RUNS above) rather than the alphabetical order worst_case_rows() uses
# for the CSV export.
ROW_ORDER: Sequence[tuple[str, str]] = (
    ("LR", "header"),
    ("LR", "imports"),
    ("LightGBM", "header"),
    ("LightGBM", "imports"),
    ("RF", "header"),
    ("RF", "imports"),
    ("MLP", "all"),
    ("MLP", "header"),
    ("MLP", "section"),
    ("MLP", "imports"),
    ("MLP", "strings"),
)

MODEL_BLOCK_SIZES: Dict[str, int] = {"LR": 2, "LightGBM": 2, "RF": 2, "MLP": 5}
RATIO_LABELS: Dict[float, str] = {0.01: "1\\%", 0.05: "5\\%", 0.1: "10\\%"}


def worst_case_lookup(rows: Iterable[Dict[str, object]]) -> Dict[tuple[str, str, str], Dict[str, object]]:
    """Same worst-case selection as worst_case_rows() (largest raw p-value among
    the three tested ratios per model/feature-group/outcome), keyed for lookup
    in manuscript row order instead of worst_case_rows()'s alphabetical output."""
    grouped: Dict[tuple[str, str, str], List[Dict[str, object]]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        key = (str(row["model"]), str(row["feature_group"]), str(row["outcome"]))
        grouped.setdefault(key, []).append(row)
    return {key: max(group, key=lambda row: float(row["raw_pvalue"])) for key, group in grouped.items()}


def _round_half_up(value: float, digits: int) -> Decimal:
    quantum = Decimal(1).scaleb(-digits)
    return Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP)


def fmt_diff(value: float) -> str:
    quant = _round_half_up(value, 3)
    return f"{quant:+.3f}"


def fmt_p(value: float) -> str:
    quant = _round_half_up(value, 3)
    text = format(quant, "f").rstrip("0").rstrip(".")
    return text if text else "0"


def fmt_ci(low: float, high: float) -> str:
    return f"[{_round_half_up(low, 3):.3f}, {_round_half_up(high, 3):.3f}]"


def build_metric_table(
    lookup: Dict[tuple[str, str, str], Dict[str, object]],
    outcome: str,
    caption: str,
    label: str,
) -> str:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{llcccccc}",
        "\\toprule",
        "Model & Feature group & Ratio & Mean difference & 95\\% CI & Direction & Raw $p$ & Holm $p$ \\\\",
        "\\midrule",
    ]
    current_model = None
    for model, feature_group in ROW_ORDER:
        row = lookup[(model, feature_group, outcome)]
        if model != current_model:
            if current_model is not None:
                lines.append("\\midrule")
            current_model = model
            model_cell = f"\\multirow{{{MODEL_BLOCK_SIZES[model]}}}{{*}}{{{model}}}"
        else:
            model_cell = ""
        ratio_label = RATIO_LABELS[float(row["strength"])]
        mean_diff = fmt_diff(float(row["paired_mean_delta"]))
        ci = fmt_ci(float(row["ci_low"]), float(row["ci_high"]))
        direction = f"{int(row['positive_seed_count'])}/5"
        raw_p = fmt_p(float(row["raw_pvalue"]))
        holm_p = fmt_p(float(row["holm_adjusted_pvalue"]))
        feature_label = feature_group.capitalize()
        lines.append(
            f" {model_cell} & {feature_label} & {ratio_label} & {mean_diff} & {ci} & {direction} & {raw_p} & {holm_p} \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


AUC_CAPTION = (
    "Paired important-versus-random differences in raw AUC-point degradation under Wild (B), by model and "
    "feature group. Raw AUC-point degradation is "
    "$\\mathrm{AUC}_{\\mathrm{baseline}}-\\mathrm{AUC}_{\\mathrm{masked}}$, not normalized by baseline AUC or "
    "by the margin above chance; this differs from the normalized AUC degradation plotted in "
    "\\autoref{fig:figure1_rq2_2_feature_dependent_decision_instability} in the main text. "
    "For each model--feature-group cell, we report the masking ratio (1\\%, 5\\%, or 10\\%) that produced the "
    "largest -- i.e., least significant -- raw sign-flip $p$-value among the three tested ratios; its "
    "Holm-adjusted $p$-value is taken from the full ratio-level correction family (36 tests for LR/LightGBM/RF, "
    "30 tests for MLP; \\ref{app:significance}). "
    "Positive mean differences indicate greater AUC degradation under important masking than under random "
    "masking. Direction reports the number of seeds (out of 5) showing this positive pattern. "
    "The 95\\% CI is a percentile bootstrap on the same five seed-level paired differences (10{,}000 "
    "resamples); see \\ref{app:significance} for how it relates to the sign-flip $p$-value. "
    "This table provides a worst-case summary for readability; the accompanying repository retains the "
    "complete ratio-level results."
)

FLIP_CAPTION = (
    "Paired important-versus-random differences in prediction flip rate under Wild (B), by model and feature "
    "group. For each model--feature-group cell, we report the masking ratio (1\\%, 5\\%, or 10\\%) that "
    "produced the largest -- i.e., least significant -- raw sign-flip $p$-value among the three tested ratios; "
    "its Holm-adjusted $p$-value is taken from the full ratio-level correction family (36 tests for "
    "LR/LightGBM/RF, 30 tests for MLP; \\ref{app:significance}). "
    "Positive mean differences indicate more prediction flips under important masking than under random "
    "masking. Direction reports the number of seeds (out of 5) showing this positive pattern. "
    "The 95\\% CI is a percentile bootstrap on the same five seed-level paired differences (10{,}000 "
    "resamples); see \\ref{app:significance} for how it relates to the sign-flip $p$-value. "
    "This table provides a worst-case summary for readability; the accompanying repository retains the "
    "complete ratio-level results."
)


def write_report(path: Path, availability_rows: List[Dict[str, object]], detailed_rows: List[Dict[str, object]]) -> None:
    lines = [
        "# Paired Masking Reanalysis",
        "",
        "This report summarizes the paired seed-level masking analysis.",
        "",
        "## Availability",
        "",
    ]
    for row in availability_rows:
        lines.append(
            f"- {row['model']} / {row['feature_group']} / {row['outcome']}: {row['status']} ({row['path']})"
        )
        if row.get("note"):
            lines.append(f"  note: {row['note']}")
    lines.extend(["", "## Completed Rows", ""])
    for row in detailed_rows:
        if row.get("status") != "ok":
            continue
        lines.append(
            f"- {row['model']} / {row['feature_group']} / {row['outcome']} / {row['strength']}: "
            f"mean={float(row['paired_mean_delta']):+.6f}, median={float(row['paired_median_delta']):+.6f}, "
            f"95% CI=[{float(row['ci_low']):+.6f}, {float(row['ci_high']):+.6f}], "
            f"positive={int(row['positive_seed_count'])}/5, raw_p={fmt_float(float(row['raw_pvalue']), 6)}, "
            f"holm_p={fmt_float(float(row.get('holm_adjusted_pvalue', float('nan'))), 6)}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    detailed_rows: List[Dict[str, object]] = []
    availability_rows: List[Dict[str, object]] = []

    for run in MAIN_RUNS:
        run_rows, availability = summarize_main_run(run)
        detailed_rows.extend(run_rows)
        availability_rows.append(availability)

    for run in MLP_RUNS:
        run_rows, availability = summarize_mlp_run(run)
        detailed_rows.extend(run_rows)
        availability_rows.append(availability)

    apply_holm(detailed_rows)

    full_rows = [row for row in detailed_rows if row.get("status") == "ok"]
    summary_rows = worst_case_rows(full_rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "paired_masking_reanalysis_rows.csv", detailed_rows)
    write_csv(OUT_DIR / "paired_masking_reanalysis_summary.csv", summary_rows)
    write_csv(OUT_DIR / "paired_masking_reanalysis_availability.csv", availability_rows)
    write_report(OUT_DIR / "paired_masking_reanalysis_report.md", availability_rows, detailed_rows)
    print(f"Wrote paired masking reanalysis artifacts to {OUT_DIR}")

    lookup = worst_case_lookup(full_rows)
    MANUSCRIPT_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    auc_tex = build_metric_table(lookup, "auc_drop", AUC_CAPTION, "tab:significance_masking_auc")
    flip_tex = build_metric_table(lookup, "flip_rate", FLIP_CAPTION, "tab:significance_masking_flip")
    (MANUSCRIPT_TABLES_DIR / "significance_masking_auc.tex").write_text(auc_tex, encoding="utf-8")
    (MANUSCRIPT_TABLES_DIR / "significance_masking_flip.tex").write_text(flip_tex, encoding="utf-8")
    print(f"Wrote manuscript Table E.1/E.2 to {MANUSCRIPT_TABLES_DIR}")


if __name__ == "__main__":
    main()
