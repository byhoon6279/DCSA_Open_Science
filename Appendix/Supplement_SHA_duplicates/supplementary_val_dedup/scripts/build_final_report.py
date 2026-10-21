"""
Assemble the final A/B/C comparison:
  A = published DCSA_Open_Science results (existing results.json / aggregate_results.csv,
      produced by their original run -- different machine/library versions, same
      code+config+seeds)
  B = this workspace's own reproduction, using the ORIGINAL (uncleaned) validation
      split for permutation importance (isolates reproduction fidelity: B-A)
  C = the SAME retrained model (identical train partition, identical test weeks),
      but permutation importance computed on the CLEANED validation split
      (isolates the dedup effect: C-B)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

WS = Path(__file__).resolve().parents[4] / "Appendix/Supplement_SHA_duplicates/supplementary_val_dedup"
RESULTS = WS / "results"
DCSA_53 = Path(__file__).resolve().parents[4] / "artifacts/5_3_targeted_perturbation_response"

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)

# ---------------------------------------------------------------------------
# A: published baseline / masking (feature_perturbation, LightGBM, permutation, k_10)
# ---------------------------------------------------------------------------
pub_fp = pd.read_csv(DCSA_53 / "results/LightGBM/feature_perturbation_lightgbm_permutation_balanced_main/k_10/aggregate_results.csv")
pub_base = pub_fp[pub_fp.perturbation_type == "baseline"].groupby("feature_group")["auc_original"].mean()
pub_imp = pub_fp[pub_fp.perturbation_type == "important"].groupby(["feature_group", "strength"])["delta_auc_mean"].mean()
pub_rnd = pub_fp[pub_fp.perturbation_type == "random"].groupby(["feature_group", "strength"])["delta_auc_mean"].mean()

pub_flip = pd.read_csv(DCSA_53 / "results/LightGBM/prediction_flip_lightgbm_balanced_main/aggregate_results.csv")
pub_flip_imp = pub_flip[pub_flip.perturbation_type == "important"].groupby(["feature_group", "strength"])[
    ["flip_rate_mean", "malware_to_benign_rate_mean"]
].mean()

pub_table5_5 = pd.read_csv(
    DCSA_53 / "pe_inspired_feature_intervention/results/experiment_results/pe_intervention_lightgbm_balanced_main/summary_over_weeks.csv"
)
pub_t55 = pub_table5_5[
    (pub_table5_5.feature_group == "imports")
    & (pub_table5_5.operator == "imports_benign_mass_injection")
    & (pub_table5_5.selection_value == 0.10)
][["selection_type", "flip_rate_across_weeks_mean", "malware_to_benign_rate_across_weeks_mean"]].set_index("selection_type")

# ---------------------------------------------------------------------------
# B/C: this workspace's reproduction (step2 = masking AUC/flip, step3 = Table 5-5)
# ---------------------------------------------------------------------------
step1 = json.loads((RESULTS / "step1_summary.json").read_text())
step1_df = pd.DataFrame(step1)

step2 = json.loads((RESULTS / "step2_trial_rows.json").read_text())
step2_df = pd.DataFrame(step2)

step3 = json.loads((RESULTS / "step3_pe_intervention_rows.json").read_text())
step3_df = pd.DataFrame(step3)


def normalized_degradation(baseline_auc: float, delta_auc: float) -> float:
    # (AUC_baseline - AUC_masked) / (AUC_baseline - 0.5) ; delta_auc = masked - baseline (negative on degradation)
    return (-delta_auc) / (baseline_auc - 0.5)


print("=" * 100)
print("PART 0. Sample-level SHA-256 dedup bookkeeping (per seed)")
print("=" * 100)
cols = [
    "seed", "group", "n_val_orig", "n_val_orig_benign", "n_val_orig_malware",
    "n_train_overlap_removed", "n_internal_dup_removed", "n_val_clean",
    "n_val_clean_benign", "n_val_clean_malware",
    "train_internal_label_conflicts", "overlap_label_conflicts", "internal_dup_label_conflicts",
]
print(step1_df[cols].to_string(index=False))

print()
print("=" * 100)
print("PART 1. Baseline test-week AUC: A (published) vs B (repro, orig val) vs C (repro, clean val)")
print("=" * 100)
for group in ["header", "imports"]:
    b_mean = step2_df[(step2_df.group == group)].groupby("seed")["baseline_auc"].mean()  # baseline doesn't depend on val cleanup
    print(f"[{group}] A(published)={pub_base[group]:.4f}  B=C(repro; masking pipeline shares one baseline)={b_mean.mean():.4f}  "
          f"(per-seed: {dict(b_mean.round(4))})")

print()
print("=" * 100)
print("PART 2. Important-feature masking: normalized AUC degradation, A vs B vs C")
print("=" * 100)
rows = []
for group in ["header", "imports"]:
    base_auc_pub = pub_base[group]
    for strength in [0.01, 0.05, 0.1]:
        a_delta = pub_imp.loc[(group, strength)]
        a_norm = normalized_degradation(base_auc_pub, a_delta)

        sub = step2_df[(step2_df.group == group) & (step2_df.strength == strength)]
        per_seed_base = sub.groupby("seed")["baseline_auc"].mean()
        per_seed_b = sub.groupby("seed")["delta_auc_orig"].mean()
        per_seed_c = sub.groupby("seed")["delta_auc_clean"].mean()
        b_delta_mean = per_seed_b.mean()
        c_delta_mean = per_seed_c.mean()
        b_norm_per_seed = [normalized_degradation(per_seed_base[s], per_seed_b[s]) for s in per_seed_base.index]
        c_norm_per_seed = [normalized_degradation(per_seed_base[s], per_seed_c[s]) for s in per_seed_base.index]

        rows.append({
            "group": group, "strength": strength,
            "A_raw_delta": a_delta, "A_norm_degrad": a_norm,
            "B_raw_delta": b_delta_mean, "B_norm_degrad": np.mean(b_norm_per_seed),
            "C_raw_delta": c_delta_mean, "C_norm_degrad": np.mean(c_norm_per_seed),
            "B_minus_A_raw": b_delta_mean - a_delta,
            "C_minus_B_raw": c_delta_mean - b_delta_mean,
            "B_seed_values_raw": [round(v, 5) for v in per_seed_b.values],
            "C_seed_values_raw": [round(v, 5) for v in per_seed_c.values],
        })
auc_table = pd.DataFrame(rows)
print(auc_table[["group", "strength", "A_raw_delta", "B_raw_delta", "C_raw_delta", "B_minus_A_raw", "C_minus_B_raw"]].to_string(index=False))
print()
print("normalized degradation view:")
print(auc_table[["group", "strength", "A_norm_degrad", "B_norm_degrad", "C_norm_degrad"]].to_string(index=False))

print()
print("=" * 100)
print("PART 3. Important-feature masking: flip_rate / M->B rate, A vs B vs C")
print("=" * 100)
rows = []
for group in ["header", "imports"]:
    for strength in [0.01, 0.05, 0.1]:
        a_flip = pub_flip_imp.loc[(group, strength), "flip_rate_mean"]
        a_m2b = pub_flip_imp.loc[(group, strength), "malware_to_benign_rate_mean"]
        sub = step2_df[(step2_df.group == group) & (step2_df.strength == strength)]
        per_seed_b_flip = sub.groupby("seed")["important_flip_rate_orig"].mean()
        per_seed_c_flip = sub.groupby("seed")["important_flip_rate_clean"].mean()
        per_seed_b_m2b = sub.groupby("seed")["important_m2b_orig"].mean()
        per_seed_c_m2b = sub.groupby("seed")["important_m2b_clean"].mean()
        rows.append({
            "group": group, "strength": strength,
            "A_flip": a_flip, "B_flip": per_seed_b_flip.mean(), "C_flip": per_seed_c_flip.mean(),
            "A_m2b": a_m2b, "B_m2b": per_seed_b_m2b.mean(), "C_m2b": per_seed_c_m2b.mean(),
            "B_flip_seeds": [round(v, 4) for v in per_seed_b_flip.values],
            "C_flip_seeds": [round(v, 4) for v in per_seed_c_flip.values],
        })
flip_table = pd.DataFrame(rows)
print(flip_table[["group", "strength", "A_flip", "B_flip", "C_flip", "A_m2b", "B_m2b", "C_m2b"]].to_string(index=False))

print()
print("=" * 100)
print("PART 4. Top-1%/5%/10% feature-set stability (orig vs cleaned val importance)")
print("=" * 100)
stab = step2_df.groupby(["group", "strength"]).agg(
    n_seed_week_rows=("topk_identical", "size"),
    frac_identical=("topk_identical", "mean"),
    mean_jaccard=("topk_jaccard", "mean"),
).reset_index()
print(stab.to_string(index=False))
# per-seed identical check computed once per (seed, group) is enough (importance doesn't vary by week);
# use one row per seed/group/strength to avoid double counting across 12 weeks
stab_per_seed = step2_df.drop_duplicates(subset=["seed", "group", "strength"])[
    ["seed", "group", "strength", "topk_identical", "topk_jaccard", "n_masked_features"]
]
print()
print("per-seed top-k identity (deduplicated across weeks, since importance is week-invariant):")
print(stab_per_seed.to_string(index=False))

print()
print("=" * 100)
print("PART 5. Table 5-5: LightGBM 10% hashed-import intervention, A vs B vs C")
print("=" * 100)
a_imp_flip = pub_t55.loc["important", "flip_rate_across_weeks_mean"]
a_imp_m2b = pub_t55.loc["important", "malware_to_benign_rate_across_weeks_mean"]
a_rnd_flip = pub_t55.loc["random", "flip_rate_across_weeks_mean"]
a_rnd_m2b = pub_t55.loc["random", "malware_to_benign_rate_across_weeks_mean"]

per_seed_b_flip = step3_df.groupby("seed")["flip_rate_important_orig"].mean()
per_seed_c_flip = step3_df.groupby("seed")["flip_rate_important_clean"].mean()
per_seed_b_m2b = step3_df.groupby("seed")["m2b_rate_important_orig"].mean()
per_seed_c_m2b = step3_df.groupby("seed")["m2b_rate_important_clean"].mean()
per_seed_rnd_flip = step3_df.groupby("seed")["flip_rate_random_mean"].mean()
per_seed_rnd_m2b = step3_df.groupby("seed")["m2b_rate_random_mean"].mean()

print(f"IMPORTANT dims -- A: flip={a_imp_flip:.4f} m2b={a_imp_m2b:.4f}")
print(f"IMPORTANT dims -- B(orig val): flip={per_seed_b_flip.mean():.4f} m2b={per_seed_b_m2b.mean():.4f}  per-seed flip={dict(per_seed_b_flip.round(4))}")
print(f"IMPORTANT dims -- C(clean val): flip={per_seed_c_flip.mean():.4f} m2b={per_seed_c_m2b.mean():.4f}  per-seed flip={dict(per_seed_c_flip.round(4))}")
print(f"RANDOM dims    -- A: flip={a_rnd_flip:.4f} m2b={a_rnd_m2b:.4f}")
print(f"RANDOM dims    -- B/C (importance-independent): flip={per_seed_rnd_flip.mean():.4f} m2b={per_seed_rnd_m2b.mean():.4f}")

top_k_t55 = step3_df.drop_duplicates(subset=["seed"])[["seed", "topk_identical", "topk_jaccard", "n_selected"]]
print()
print("Top-10% hashed-import index-set stability (orig vs cleaned val):")
print(top_k_t55.to_string(index=False))

# ---------------------------------------------------------------------------
# Save everything machine-readable too
# ---------------------------------------------------------------------------
out = {
    "auc_table": auc_table.to_dict(orient="records"),
    "flip_table": flip_table.to_dict(orient="records"),
    "topk_stability_by_strength": stab.to_dict(orient="records"),
    "topk_stability_per_seed": stab_per_seed.to_dict(orient="records"),
    "table5_5": {
        "A_important": {"flip": a_imp_flip, "m2b": a_imp_m2b},
        "A_random": {"flip": a_rnd_flip, "m2b": a_rnd_m2b},
        "B_important": {"flip": per_seed_b_flip.mean(), "m2b": per_seed_b_m2b.mean(), "per_seed_flip": per_seed_b_flip.to_dict()},
        "C_important": {"flip": per_seed_c_flip.mean(), "m2b": per_seed_c_m2b.mean(), "per_seed_flip": per_seed_c_flip.to_dict()},
        "random_shared": {"flip": per_seed_rnd_flip.mean(), "m2b": per_seed_rnd_m2b.mean()},
        "topk_stability_per_seed": top_k_t55.to_dict(orient="records"),
    },
}
(RESULTS / "final_comparison.json").write_text(json.dumps(out, indent=2, default=float, ensure_ascii=False), encoding="utf-8")
print()
print("Saved -> results/final_comparison.json")
