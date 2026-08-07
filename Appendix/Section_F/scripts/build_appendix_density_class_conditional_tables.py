#!/usr/bin/env python3
"""Build the two Appendix F.1 tables (Class-Conditional Density-Composition
Sensitivity) from the already-bundled D3/D4 class-conditional comparison CSVs.

This script builds the manuscript LaTeX directly from the comparison CSVs.

Inputs (already bundled, no new data):
- artifacts/5_4_density_conditioned_reliability/sensitivity/class_conditional/results/d3_class_conditional_comparison.csv
- artifacts/5_5_density_conditioned_fragility/sensitivity/class_conditional/results/d4_class_conditional_comparison.csv

Outputs (this directory):
- results/appendix_density_class_conditional_reliability.tex / .csv  (Table, ss:dcsa_density_reliability half)
- results/appendix_density_class_conditional_fragility.tex / .csv    (Table, ss:dcsa_density_stress-response_amplification half)
"""
from __future__ import annotations

import csv
from pathlib import Path

HERE = Path(__file__).resolve()
SECTION_F_ROOT = HERE.parents[1]  # Appendix/Section_F
PACKAGE_ROOT = HERE.parents[3]    # DCSA_Open_Science/
ARTIFACTS_ROOT = PACKAGE_ROOT / "artifacts"

D3_CSV = (
    ARTIFACTS_ROOT
    / "5_4_density_conditioned_reliability/sensitivity/class_conditional/results/d3_class_conditional_comparison.csv"
)
D4_CSV = (
    ARTIFACTS_ROOT
    / "5_5_density_conditioned_fragility/sensitivity/class_conditional/results/d4_class_conditional_comparison.csv"
)
RESULTS_DIR = SECTION_F_ROOT / "results"

MODELS = ["LR", "LightGBM", "RF"]
D3_FEATURE_GROUPS = ["all", "header", "imports", "section", "strings"]
D4_FEATURE_GROUPS = ["header", "imports"]
D4_STRENGTHS = ["0.01", "0.05", "0.1"]
D4_STRENGTH_LABELS = {"0.01": "1\\%", "0.05": "5\\%", "0.1": "10\\%"}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def find_row(rows: list[dict[str, str]], **matches: str) -> dict[str, str]:
    for row in rows:
        if all(row.get(key) == value for key, value in matches.items()):
            return row
    raise KeyError(f"No row matching {matches}")


def fmt(value: str, signed: bool = True) -> str:
    v = float(value)
    sign = "+" if (signed and v >= 0) else ""
    return f"{sign}{v:.3f}"


def build_reliability_table() -> tuple[str, list[dict[str, object]]]:
    rows = read_rows(D3_CSV)
    compact_rows: list[dict[str, object]] = []
    body_blocks = []
    for model in MODELS:
        lines = [f"\\multirow{{5}}{{*}}{{{model}}}"]
        for group in D3_FEATURE_GROUPS:
            row = find_row(rows, model=model, feature_group=group)
            pooled_auc = fmt(row["pooled_auc_gap_low_minus_high"])
            cc_auc = fmt(row["class_cond_auc_gap_low_minus_high"])
            pooled_mix = fmt(row["pooled_mix_gap_low_minus_high"])
            cb_mix = fmt(row["class_balanced_mix_gap_low_minus_high"])
            lines.append(
                f"          & {group:<7} & ${pooled_auc}$ & ${cc_auc}$ & ${pooled_mix}$ & ${cb_mix}$ \\\\"
            )
            compact_rows.append(
                {
                    "model": model,
                    "feature_group": group,
                    "pooled_delta_auc": pooled_auc,
                    "class_conditional_delta_auc": cc_auc,
                    "pooled_delta_mix_at_10": pooled_mix,
                    "class_balanced_delta_mix_at_10": cb_mix,
                }
            )
        body_blocks.append("\n".join(lines))
    body = "\n\\midrule\n".join(body_blocks)

    tex = f"""\\begin{{table}}[t]
\\centering
\\caption{{Class-conditional reliability sensitivity.}}
\\label{{tab:appendix_density_class_conditional_reliability}}
\\scriptsize
\\setlength{{\\tabcolsep}}{{2pt}}
\\resizebox{{\\columnwidth}}{{!}}{{%
\\begin{{tabular}}{{llrrrr}}
\\toprule
\\textbf{{Model}} & \\textbf{{Feature Group}} & \\textbf{{\\makecell{{Pooled AUC difference\\\\(low$-$high)}}}} & \\textbf{{\\makecell{{Class-conditional AUC\\\\difference (low$-$high)}}
}} & \\textbf{{\\makecell{{Pooled\\\\$\\Delta$Mix@10}}}} & \\textbf{{\\makecell{{Class-balanced\\\\$\\Delta$Mix@10}}}} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}%
}}
\\end{{table}}
"""
    return tex, compact_rows


def build_fragility_table() -> tuple[str, list[dict[str, object]]]:
    rows = read_rows(D4_CSV)
    compact_rows: list[dict[str, object]] = []
    body_blocks = []
    for model in MODELS:
        lines = [f"\\multirow{{6}}{{*}}{{{model}}}"]
        for group in D4_FEATURE_GROUPS:
            for strength in D4_STRENGTHS:
                row = find_row(
                    rows,
                    model=model,
                    feature_group=group,
                    perturbation_type="important",
                    strength=strength,
                )
                pooled_drop = fmt(row["pooled_auc_drop_density"])
                cc_drop = fmt(row["class_cond_auc_drop_density"])
                m2b = fmt(str(float(row["class_cond_low_m2b"]) - float(row["class_cond_high_m2b"])))
                b2m = fmt(str(float(row["class_cond_low_b2m"]) - float(row["class_cond_high_b2m"])))
                strength_label = D4_STRENGTH_LABELS[strength]
                lines.append(
                    f"          & {group:<7} & {strength_label:<4} & ${pooled_drop}$ & ${cc_drop}$ & ${m2b}$ & ${b2m}$ \\\\"
                )
                compact_rows.append(
                    {
                        "model": model,
                        "feature_group": group,
                        "masking_strength": strength,
                        "pooled_delta_auc_drop": pooled_drop,
                        "class_conditional_delta_auc_drop": cc_drop,
                        "class_conditional_delta_malware_to_benign": m2b,
                        "class_conditional_delta_benign_to_malware": b2m,
                    }
                )
        body_blocks.append("\n".join(lines))
    body = "\n\\midrule\n".join(body_blocks)

    tex = f"""\\begin{{table}}[t]
\\centering
\\caption{{Class-conditional masking sensitivity.}}
\\label{{tab:appendix_density_class_conditional_fragility}}
\\scriptsize
\\setlength{{\\tabcolsep}}{{2pt}}
\\resizebox{{\\columnwidth}}{{!}}{{%
\\begin{{tabular}}{{llrrrrr}}
\\toprule
\\textbf{{Model}} & \\textbf{{Group}} & \\textbf{{Mask}} & \\textbf{{\\makecell{{Pooled degradation\\\\difference (low$-$high)}}}} & \\textbf{{\\makecell{{Class-conditional degradation\\\\difference (low$-$high)}}}} & \\textbf{{Class-Cond.\\ $\\Delta$(M$\\rightarrow$B)}} & \\textbf{{Class-Cond.\\ $\\Delta$(B$\\rightarrow$M)}} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}%
}}
\\end{{table}}
"""
    return tex, compact_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    reliability_tex, reliability_rows = build_reliability_table()
    (RESULTS_DIR / "appendix_density_class_conditional_reliability.tex").write_text(reliability_tex, encoding="utf-8")
    write_csv(RESULTS_DIR / "appendix_density_class_conditional_reliability.csv", reliability_rows)

    fragility_tex, fragility_rows = build_fragility_table()
    (RESULTS_DIR / "appendix_density_class_conditional_fragility.tex").write_text(fragility_tex, encoding="utf-8")
    write_csv(RESULTS_DIR / "appendix_density_class_conditional_fragility.csv", fragility_rows)

    print(f"Wrote {RESULTS_DIR / 'appendix_density_class_conditional_reliability.tex'} and .csv")
    print(f"Wrote {RESULTS_DIR / 'appendix_density_class_conditional_fragility.tex'} and .csv")


if __name__ == "__main__":
    main()
