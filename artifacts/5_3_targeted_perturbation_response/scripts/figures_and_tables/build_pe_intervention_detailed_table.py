#!/usr/bin/env python3
"""Build the manuscript's actual Table 7 (`pe_intervention_detailed.tex`,
`\\label{tab:rq2_pe_import_validation}`), which is Wild (B)-only, LR/LightGBM/RF.

This script reads the already-bundled sources directly (LR/LightGBM from
`pe_inspired_feature_intervention/results/experiment_results/*/summary_over_weeks.csv`,
RF from this section's own `results/RF/`) and emits the plain Wild (B)-only
table (a separate wider LR+LightGBM+RF, Wild(B)+Unpacked(B) variant,
`\\label{tab:rq2_pe_import_validation_with_rf}`, is a different table not
cited by the manuscript and is not built here).
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

BASE = Path(__file__).resolve().parents[2]
PE_DIR = BASE / "pe_inspired_feature_intervention" / "results" / "experiment_results"
RUNS = [
    ("LR", PE_DIR / "pe_intervention_balanced_main" / "summary_over_weeks.csv"),
    ("LightGBM", PE_DIR / "pe_intervention_lightgbm_balanced_main" / "summary_over_weeks.csv"),
    ("RF", BASE / "results" / "RF" / "pe_intervention_balanced_main_random_forest" / "aggregate_results.csv"),
]
OUT_DIR = BASE / "results" / "manuscript_tables"


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_summary_like_rows(path: Path) -> List[Dict[str, str]]:
    rows = read_rows(path)
    if path.name == "summary_over_weeks.csv":
        return rows

    grouped: Dict[tuple[str, str, str, str], List[Dict[str, str]]] = {}
    for row in rows:
        if row["selection_type"] not in {"important", "random"}:
            continue
        selection_value = float(row["selection_value"])
        if selection_value <= 1.0:
            selection_label = f"{int(round(selection_value * 100))}%"
        else:
            selection_label = f"{int(round(selection_value))}%"
        key = (row["selection_type"], row["feature_group"], row["operator"], selection_label)
        grouped.setdefault(key, []).append(row)

    out: List[Dict[str, str]] = []
    for (selection_type, feature_group, operator, selection_label), group_rows in grouped.items():
        def mean(field: str) -> float:
            vals = [float(r[field]) for r in group_rows]
            return sum(vals) / len(vals) if vals else 0.0

        out.append({
            "feature_group": feature_group,
            "operator": operator,
            "selection_type": selection_type,
            "selection_label": selection_label,
            "malware_to_benign_rate_across_weeks_mean": str(mean("malware_to_benign_rate_mean")),
            "mean_signed_probability_shift_across_weeks_mean": str(mean("mean_signed_probability_shift_mean")),
            "delta_auc_across_weeks_mean": str(mean("delta_auc_mean")),
            "flip_rate_across_weeks_mean": str(mean("flip_rate_mean")),
        })
    return out


def fmt(v: float) -> str:
    return f"{v:.3f}"


def build_rows() -> Dict[str, Dict[str, Dict[str, str]]]:
    out: Dict[str, Dict[str, Dict[str, str]]] = {}
    for model, path in RUNS:
        rows = load_summary_like_rows(path)
        lookup = {
            (row["selection_type"], row["selection_label"]): row
            for row in rows
            if row["feature_group"] == "imports"
            and row["operator"] == "imports_benign_mass_injection"
            and row["selection_type"] in {"important", "random"}
        }
        model_rows: Dict[str, Dict[str, str]] = {}
        for selection_label in ["1%", "5%", "10%"]:
            imp = lookup[("important", selection_label)]
            rnd = lookup[("random", selection_label)]
            model_rows[selection_label] = {
                "m2b_imp": fmt(float(imp["malware_to_benign_rate_across_weeks_mean"])),
                "m2b_rand": fmt(float(rnd["malware_to_benign_rate_across_weeks_mean"])),
                "shift_imp": fmt(float(imp["mean_signed_probability_shift_across_weeks_mean"])),
                "shift_rand": fmt(float(rnd["mean_signed_probability_shift_across_weeks_mean"])),
                "auc_imp": fmt(float(imp["delta_auc_across_weeks_mean"])),
                "auc_rand": fmt(float(rnd["delta_auc_across_weeks_mean"])),
                "flip_imp": fmt(float(imp["flip_rate_across_weeks_mean"])),
                "flip_rand": fmt(float(rnd["flip_rate_across_weeks_mean"])),
            }
        out[model] = model_rows
    return out


def build_tex(data: Dict[str, Dict[str, Dict[str, str]]]) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Model responses to hashed import-feature interventions for the Wild (B) view.}",
        r"\label{tab:rq2_pe_import_validation}",
        "",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{lccccccccc}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Intervention ratio}",
        r"& \textbf{M$\rightarrow$B (Imp)}",
        r"& \textbf{M$\rightarrow$B (Rand)}",
        r"& \textbf{Mean malware-score change (Imp.)}",
        r"& \textbf{Mean malware-score change (Rand)}",
        r"& \textbf{AUC change (Imp)}",
        r"& \textbf{AUC change (Rand)}",
        r"& \textbf{Flip rate (Imp)}",
        r"& \textbf{Flip rate (Rand)} \\",
        r"\midrule",
        "",
    ]
    models = ["LR", "LightGBM", "RF"]
    padded_cols = ["shift_imp", "shift_rand", "auc_imp", "auc_rand"]
    for i, model in enumerate(models):
        lines.append(f"\\multirow{{3}}{{*}}{{{model}}}")
        rows = [data[model][label] for label in ["1%", "5%", "10%"]]
        # Each numeric column is right-justified to the widest value within
        # this model's own 3-row block (matches the manuscript source
        # exactly - it is not a fixed global width).
        widths = {col: max(len(r[col]) for r in rows) for col in padded_cols}
        for selection_label, r in zip(["1%", "5%", "10%"], rows):
            escaped_label = selection_label.replace("%", "\\%")
            lines.append(
                f"& {escaped_label:<5}& {r['m2b_imp']} & {r['m2b_rand']} "
                f"& {r['shift_imp']:>{widths['shift_imp']}} & {r['shift_rand']:>{widths['shift_rand']}} "
                f"& {r['auc_imp']:>{widths['auc_imp']}} & {r['auc_rand']:>{widths['auc_rand']}} "
                f"& {r['flip_imp']} & {r['flip_rand']} \\\\"
            )
        if i < len(models) - 1:
            lines.append(r"\midrule")
            lines.append("")
    lines.extend([
        "",
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\begin{minipage}{\linewidth}",
        r"\footnotesize",
        r"\textit{Note.} Imp.\ and Rand.\ denote interventions on importance-selected and randomly selected hashed import-feature dimensions, respectively.",
        r"M$\rightarrow$B and flip rates are computed over the intervened malware samples, whereas AUC change is computed over the full evaluation set.",
        r"We define AUC change as $\mathrm{AUC}_{\mathrm{modified}}-\mathrm{AUC}_{\mathrm{baseline}}$.",
        r"Negative AUC changes indicate AUC degradation.",
        r"\end{minipage}",
        r"\end{table*}",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    data = build_rows()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "pe_intervention_detailed.tex").write_text(build_tex(data), encoding="utf-8")
    print(f"Wrote {OUT_DIR / 'pe_intervention_detailed.tex'}")


if __name__ == "__main__":
    main()
