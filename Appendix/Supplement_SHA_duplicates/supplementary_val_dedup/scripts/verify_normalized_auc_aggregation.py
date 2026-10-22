"""
Check-only script (no new model training/prediction): verifies whether this
reproduction's "normalized AUC degradation" numbers were computed with the same
aggregation order as DCSA_Open_Science's own figure-generating code
(seed_level_normalized_drops() in
scripts/figures_and_tables/plot_decision_level_masking_collapse_v2.py), and
recomputes the corrected version from already-saved data where possible.

DCSA's method (ground truth):
  1. PER ROW (per seed, per week, per trial): margin = max(baseline_auc-0.5, 1e-8);
     normalized_drop = max(0.0, -delta_auc) / margin   [floored at 0, uses that
     row's own reconstructed baseline_auc = row.auc - row.delta_auc]
  2. average normalized_drop within each (seed, week) across repeat trials
     (only relevant for "random", which has 20 trials per seed-week)
  3. average over the 12 weeks within each seed -> 5 seed-level values
  4. the reported/plotted value = mean of the 5 seed-level values

This reproduction's PART 2 table instead computed:
  normalized_degradation(mean_baseline_auc, mean_delta_auc) = -mean_delta / (mean_baseline-0.5)
  i.e. averaged delta_auc and baseline_auc FIRST, across weeks/seeds, THEN
  normalized ONCE -- a different (and generally not equal) statistic, and
  with no floor at 0.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

WS = Path(__file__).resolve().parents[4] / "Appendix/Supplement_SHA_duplicates/supplementary_val_dedup"
DCSA_53 = Path(__file__).resolve().parents[4] / "artifacts/5_3_targeted_perturbation_response"


def normalized_drop_row(baseline_auc: float, delta_auc: float) -> float:
    margin = max(baseline_auc - 0.5, 1e-8)
    return max(0.0, -delta_auc) / margin


# ---------------------------------------------------------------------------
# A (published): recompute exactly via DCSA's own procedure, from raw trial_results.csv
# ---------------------------------------------------------------------------
pub = pd.read_csv(DCSA_53 / "results/LightGBM/feature_perturbation_lightgbm_permutation_balanced_main/k_10/trial_results.csv")


def a_seed_level_values(group: str, strength: float) -> dict:
    sub = pub[(pub.feature_group == group) & (pub.perturbation_type == "important")
              & (pub.strength.round(6) == round(strength, 6))]
    sub = sub.assign(baseline_auc=sub["auc"] - sub["delta_auc"])
    sub = sub.assign(norm_drop=sub.apply(lambda r: normalized_drop_row(r["baseline_auc"], r["delta_auc"]), axis=1))
    # one row per (seed, week) already (important has exactly 1 trial), so this is
    # already the per-seed-week value; average over weeks per seed:
    per_seed = sub.groupby("seed")["norm_drop"].mean()
    return per_seed.to_dict()


# ---------------------------------------------------------------------------
# B/C (this reproduction): recompute from step2_trial_rows.json (has per-seed,
# per-week baseline_auc + delta_auc_orig/delta_auc_clean -- exact for "important",
# since there's exactly 1 important row per seed/week/strength, same as A)
# ---------------------------------------------------------------------------
step2 = pd.DataFrame(json.loads((WS / "results/step2_trial_rows.json").read_text()))


def bc_seed_level_values(group: str, strength: float, which: str) -> dict:
    col = "delta_auc_orig" if which == "B" else "delta_auc_clean"
    sub = step2[(step2.group == group) & (step2.strength == strength)]
    sub = sub.assign(norm_drop=sub.apply(lambda r: normalized_drop_row(r["baseline_auc"], r[col]), axis=1))
    per_seed = sub.groupby("seed")["norm_drop"].mean()
    return per_seed.to_dict()


print("=" * 100)
print("CORRECTED normalized AUC degradation (important arm) -- per-row floor+normalize, THEN average")
print("(matches DCSA's seed_level_normalized_drops() exactly, for the 'important' arm: no repeat-trial")
print(" ambiguity since important has exactly 1 trial per seed/week, so this is an EXACT recomputation)")
print("=" * 100)

rows = []
for group in ["header", "imports"]:
    for strength in [0.01, 0.05, 0.1]:
        a_vals = a_seed_level_values(group, strength)
        b_vals = bc_seed_level_values(group, strength, "B")
        c_vals = bc_seed_level_values(group, strength, "C")
        a_mean = sum(a_vals.values()) / len(a_vals)
        b_mean = sum(b_vals.values()) / len(b_vals)
        c_mean = sum(c_vals.values()) / len(c_vals)
        rows.append({
            "group": group, "strength": strength,
            "A_seed_vals": {k: round(v, 6) for k, v in sorted(a_vals.items())},
            "A_mean": a_mean,
            "B_seed_vals": {k: round(v, 6) for k, v in sorted(b_vals.items())},
            "B_mean": b_mean,
            "C_seed_vals": {k: round(v, 6) for k, v in sorted(c_vals.items())},
            "C_mean": c_mean,
        })
        print(f"[{group} {strength}] A={a_mean:.6f} B={b_mean:.6f} C={c_mean:.6f}  "
              f"(A_seeds={a_vals}) (B_seeds={b_vals}) (C_seeds={c_vals})")

print()
print("=" * 100)
print("COMPARISON: previously reported (WRONG aggregation order) vs CORRECTED (this script)")
print("=" * 100)
old = json.loads((WS / "results/final_comparison.json").read_text())
for old_row in old["auc_table"]:
    g, s = old_row["group"], old_row["strength"]
    match = next(r for r in rows if r["group"] == g and r["strength"] == s)
    print(f"[{g} {s}] OLD(avg-then-normalize): A={old_row['A_norm_degrad']:.4f} B={old_row['B_norm_degrad']:.4f} C={old_row['C_norm_degrad']:.4f}"
          f"   |   CORRECTED(normalize-then-avg,floor0): A={match['A_mean']:.4f} B={match['B_mean']:.4f} C={match['C_mean']:.4f}")

out = {
    "method_note": (
        "CORRECTED = per-row normalized_drop = max(0,-delta_auc)/max(baseline_auc-0.5,1e-8), "
        "averaged over 12 weeks per seed, then averaged over 5 seeds -- matches DCSA's "
        "seed_level_normalized_drops() exactly for the important arm (verified against raw "
        "trial_results.csv for A; exact recomputation, not an approximation, since 'important' "
        "has exactly 1 trial per seed-week so there is no repeat-trial-averaging ambiguity)."
    ),
    "random_arm_caveat": (
        "This script only recomputes the 'important' arm exactly. The 'random' arm's correct "
        "procedure averages the per-row normalized_drop across the 20 repeat trials WITHIN each "
        "seed-week before further averaging; this reproduction's step2 output only stored the "
        "already-trial-averaged delta_auc per seed-week (not the 20 individual trial deltas), "
        "so an exactly matching recomputation for 'random' is not possible from currently saved "
        "data without re-running step2 to persist per-trial rows."
    ),
    "results": rows,
}
(WS / "results" / "normalized_auc_aggregation_check.json").write_text(
    json.dumps(out, indent=2, default=float, ensure_ascii=False), encoding="utf-8"
)
print()
print("Saved -> results/normalized_auc_aggregation_check.json")
