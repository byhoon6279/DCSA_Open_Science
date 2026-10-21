"""Compare v1/v2 independent re-runs with (A) original DCSA trial results and (S) stored
supplementary results; aggregate per seed and overall; bootstrap CI (method of thesis 5.2.5).
No significance claims are produced; sign counts are descriptive only."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

V = Path(__file__).resolve().parents[2] / "validation/results"
S = Path(__file__).resolve().parents[2] / "Appendix/Supplement_SHA_duplicates/supplementary_val_dedup/results"
A = Path(__file__).resolve().parents[2] / "artifacts/5_3_targeted_perturbation_response"
SEEDS = [13, 42, 77, 123, 2024]
out = {}


def boot_ci(vals, n=10_000, seed=0):
    rng = np.random.default_rng(seed)
    vals = np.asarray(vals, float)
    m = rng.choice(vals, size=(n, len(vals)), replace=True).mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


# ---------------- intervention ----------------
v = pd.concat([pd.read_csv(V / f"v1_seed{s}_rows.csv") for s in SEEDS])
weeks = sorted(v.test_week.unique())
a = pd.read_csv(A / "pe_inspired_feature_intervention/results/experiment_results/pe_intervention_lightgbm_balanced_main/trial_results.csv")
a = a[(a.feature_group == "imports") & (a.operator == "imports_benign_mass_injection")]

# baseline AUC (model + 12-week eval set fingerprint), imports group
vb = v[(v.selection == "baseline") & (v.group == "imports")].set_index(["seed", "test_week"]).auc
ab = a[a.selection_type == "baseline"].set_index(["seed", "test_week"]).auc
out["baseline_auc_imports_max_abs_diff_vs_original"] = float((vb - ab.reindex(vb.index)).abs().max())

checks = {}
for s in (0.01, 0.05, 0.1):
    vi = v[(v.selection == "important_orig") & (v.strength == s)].set_index(["seed", "test_week"])
    ai = a[(a.selection_type == "important") & (a.selection_value == s)].set_index(["seed", "test_week"])
    vr = v[(v.selection == "random") & (v.strength == s)].set_index(["seed", "test_week", "trial"])
    ar = a[(a.selection_type == "random") & (a.selection_value == s)].rename(columns={"trial_id": "trial"}).set_index(["seed", "test_week", "trial"])
    checks[str(s)] = {
        "important_m2b_max_abs_diff": float((vi.m2b_rate - ai.malware_to_benign_rate.reindex(vi.index)).abs().max()),
        "important_flip_max_abs_diff": float((vi.flip_rate - ai.flip_rate.reindex(vi.index)).abs().max()),
        "random_m2b_max_abs_diff": float((vr.m2b_rate - ar.malware_to_benign_rate.reindex(vr.index)).abs().max()),
        "random_flip_max_abs_diff": float((vr.flip_rate - ar.flip_rate.reindex(vr.index)).abs().max()),
        "n_important_rows": int(len(vi)), "n_random_rows": int(len(vr)),
    }
out["intervention_vs_original_trial_rows"] = checks

# stored supplementary step3 (10% only)
st = pd.DataFrame(json.load(open(S / "step3_pe_intervention_rows.json"))).set_index(["seed", "test_week"])
v10 = v[v.strength == 0.1]
pv = lambda sel, col: v10[v10.selection == sel].set_index(["seed", "test_week"])[col]
rnd10 = v10[v10.selection == "random"].groupby(["seed", "test_week"])[["m2b_rate", "flip_rate"]].mean()
out["intervention_vs_stored_step3_max_abs_diff"] = {
    "m2b_orig": float((pv("important_orig", "m2b_rate") - st.m2b_rate_important_orig).abs().max()),
    "m2b_clean": float((pv("important_clean", "m2b_rate") - st.m2b_rate_important_clean).abs().max()),
    "flip_clean": float((pv("important_clean", "flip_rate") - st.flip_rate_important_clean).abs().max()),
    "m2b_random_mean": float((rnd10.m2b_rate - st.m2b_rate_random_mean).abs().max()),
}

# per-seed and overall aggregation (seed-week mean; random: mean over 20 trials per seed-week first)
agg_rows = []
for s in (0.01, 0.05, 0.1):
    x = v[v.strength == s]
    sw = x.groupby(["selection", "seed", "test_week"])[["m2b_rate", "flip_rate"]].mean().reset_index()
    seed_means = sw.groupby(["selection", "seed"])[["m2b_rate", "flip_rate"]].mean().reset_index()
    for sel in ("important_orig", "important_clean", "random"):
        sm = seed_means[seed_means.selection == sel].set_index("seed")
        agg_rows.append({"strength": s, "selection": sel,
                         **{f"m2b_seed{k}": float(sm.m2b_rate[k]) for k in SEEDS},
                         **{f"flip_seed{k}": float(sm.flip_rate[k]) for k in SEEDS},
                         "m2b_mean60": float(sw[sw.selection == sel].m2b_rate.mean()),
                         "m2b_mean_of_seed_means": float(sm.m2b_rate.mean()),
                         "m2b_boot95": boot_ci(sm.m2b_rate.values),
                         "flip_mean60": float(sw[sw.selection == sel].flip_rate.mean())})
    # descriptive per-seed and per-seed-week comparisons vs random
    piv = sw.pivot_table(index=["seed", "test_week"], columns="selection", values="m2b_rate")
    ps = piv.groupby("seed").mean()
    agg_rows.append({"strength": s, "selection": "comparisons",
                     "clean_gt_random_seeds": int((ps.important_clean > ps.random).sum()),
                     "orig_gt_random_seeds": int((ps.important_orig > ps.random).sum()),
                     "clean_lt_orig_seeds": int((ps.important_clean < ps.orig if False else ps.important_clean < ps.important_orig).sum()),
                     "clean_gt_random_seedweeks": int((piv.important_clean > piv.random).sum()),
                     "orig_gt_random_seedweeks": int((piv.important_orig > piv.random).sum()),
                     "min_seed_diff_clean_minus_random": float((ps.important_clean - ps.random).min()),
                     "min_seedweek_diff_clean_minus_random": float((piv.important_clean - piv.random).min()),
                     "boot95_clean_minus_random": boot_ci((ps.important_clean - ps.random).values),
                     "boot95_orig_minus_clean": boot_ci((ps.important_orig - ps.important_clean).values)})
pd.DataFrame(agg_rows).to_json(V / "v3_intervention_aggregate.json", orient="records", indent=2)

# per-seed-week random distribution: where does important_clean sit among 20 random trials?
x = v[v.strength == 0.1]
r = x[x.selection == "random"]
c = x[x.selection == "important_clean"].set_index(["seed", "test_week"]).m2b_rate
rmax = r.groupby(["seed", "test_week"]).m2b_rate.max()
out["10pct_clean_exceeds_max_of_20_random_seedweeks"] = int((c > rmax).sum())
out["10pct_random_trial_m2b_max_overall"] = float(r.m2b_rate.max())

# ---------------- top-k changes ----------------
topk = []
for s_ in SEEDS:
    sm = json.load(open(V / f"v1_seed{s_}_summary.json"))
    out.setdefault("dedup", []).append(sm["dedup"])
    out.setdefault("model_md5", {})[s_] = sm["model_md5"]
    out.setdefault("val_auc", {})[s_] = sm["val_auc"]
    topk += sm["topk"]
pd.DataFrame(topk).to_csv(V / "v3_topk_changes.csv", index=False)

# importance vectors vs stored supplementary npz
imp_cmp = {}
for s_ in SEEDS:
    for g in ("imports", "header"):
        mine = np.load(V / f"v1_importance_seed{s_}_{g}.npz")
        theirs = np.load(S / f"importance_seed{s_}_{g}.npz")
        imp_cmp[f"{s_}_{g}"] = {"orig_max_abs_diff": float(np.abs(mine["orig"] - theirs["importance_orig"]).max()),
                                "clean_max_abs_diff": float(np.abs(mine["clean"] - theirs["importance_clean"]).max())}
out["importance_vs_stored_npz"] = imp_cmp

# ---------------- masking (if v2 done) ----------------
mfiles = [V / f"v2_masking_seed{s}.csv" for s in SEEDS]
if all(f.exists() for f in mfiles):
    m = pd.concat([pd.read_csv(f) for f in mfiles])
    pf = pd.read_csv(A / "results/LightGBM/prediction_flip_lightgbm_balanced_main/trial_results.csv")
    pa = pd.read_csv(A / "results/LightGBM/feature_perturbation_lightgbm_permutation_balanced_main/k_10/trial_results.csv")
    week_list = json.load(open(A / "pe_inspired_feature_intervention/results/experiment_results/pe_intervention_lightgbm_balanced_main/results.json"))["config"]["test_weeks"]
    m["test_week"] = m.week_idx.map(dict(enumerate(week_list)))
    mc = {}
    for typ, sel in (("important", "important_orig"), ("random", "random")):
        mm = m[m.selection == sel].copy()
        mm["trial_id"] = mm.trial
        key = ["seed", "group", "test_week", "strength", "trial_id"]
        mm = mm.rename(columns={"group": "feature_group"}).set_index(["seed", "feature_group", "test_week", "strength", "trial_id"])
        f_ = pf[pf.perturbation_type == typ].set_index(["seed", "feature_group", "test_week", "strength", "trial_id"])
        a_ = pa[pa.perturbation_type == typ].set_index(["seed", "feature_group", "test_week", "strength", "trial_id"])
        mc[typ] = {"flip_max_abs_diff": float((mm.flip_rate - f_.flip_rate.reindex(mm.index)).abs().max()),
                   "delta_auc_max_abs_diff": float((mm.delta_auc - a_.delta_auc.reindex(mm.index)).abs().max()),
                   "n_rows": int(len(mm)), "n_missing_in_original": int(f_.flip_rate.reindex(mm.index).isna().sum())}
    out["masking_vs_original"] = mc
    m["norm_drop"] = np.maximum(0, -m.delta_auc) / np.maximum(m.baseline_auc - 0.5, 1e-8)
    sw = m.groupby(["group", "strength", "selection", "seed", "week_idx"])[["delta_auc", "flip_rate", "norm_drop"]].mean().reset_index()
    seed_level = sw.groupby(["group", "strength", "selection", "seed"])[["delta_auc", "flip_rate", "norm_drop"]].mean().reset_index()
    seed_level.to_csv(V / "v3_masking_seed_level.csv", index=False)
    summ = seed_level.groupby(["group", "strength", "selection"])[["delta_auc", "flip_rate", "norm_drop"]].mean().reset_index()
    # per-seed direction important vs random
    dirs = []
    for (g, s), d in seed_level.groupby(["group", "strength"]):
        p = d.pivot(index="seed", columns="selection", values=["delta_auc", "flip_rate"])
        for sel in ("important_orig", "important_clean"):
            dirs.append({"group": g, "strength": s, "selection": sel,
                         "seeds_auc_drop_gt_random": int((p["delta_auc"][sel] < p["delta_auc"]["random"]).sum()),
                         "seeds_flip_gt_random": int((p["flip_rate"][sel] > p["flip_rate"]["random"]).sum())})
    summ.to_csv(V / "v3_masking_summary.csv", index=False)
    pd.DataFrame(dirs).to_csv(V / "v3_masking_direction.csv", index=False)

(V / "v3_checks.json").write_text(json.dumps(out, indent=2, default=str))
print(json.dumps({k: out[k] for k in out if k not in ("dedup",)}, indent=1, default=str)[:6000])
