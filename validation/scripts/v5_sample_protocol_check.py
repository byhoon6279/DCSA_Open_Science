"""Execution check of sample settings and SHA duplicate structure (read-only on caches; caches
verified identical to raw EMBER JSONL in v0). Reproduces DCSA balance_samples() draws via v1.balance.
Outputs: results/v5_sample_protocol_check.json"""
import json, sys
from collections import Counter
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import v1_dedup_importance_intervention as v1

cfg = v1.load_cfg()
SEEDS = [13, 42, 77, 123, 2024]
out = {}

# ---- training pool ----
tr = v1.LazyPool(cfg["train_weeks"], "train")
wk = np.concatenate([np.full(tr.offsets[i + 1] - tr.offsets[i], i) for i in range(len(tr.files))])
out["train_pool"] = {"n_rows": int(len(tr.y)), "n_benign": int((tr.y == 0).sum()), "n_malware": int((tr.y == 1).sum()),
                     "unique_sha_benign": int(len(np.unique(tr.sha256[tr.y == 0]))),
                     "unique_sha_malware": int(len(np.unique(tr.sha256[tr.y == 1]))),
                     "unique_sha_total": int(len(np.unique(tr.sha256)))}
u, inv, c = np.unique(tr.sha256, return_inverse=True, return_counts=True)
out["train_pool"]["copies_per_sha_distribution"] = {int(k): int(v) for k, v in Counter(c.tolist()).items()}
# are duplicates within the same weekly file or across weeks?
dup_rows = c[inv] > 1
same_week = 0; cross_week = 0
order = np.argsort(inv[dup_rows], kind="stable")
inv_d, wk_d = inv[dup_rows][order], wk[dup_rows][order]
bounds = np.flatnonzero(np.diff(inv_d)) + 1
for g in np.split(wk_d, bounds):
    if len(set(g.tolist())) == 1: same_week += 1
    else: cross_week += 1
out["train_pool"]["dup_sha_all_copies_in_same_week"] = same_week
out["train_pool"]["dup_sha_copies_across_weeks"] = cross_week

# ---- test weeks ----
test_sha_all = []
weeks = []
for wi, w in enumerate(cfg["test_weeks"]):
    t = np.load(v1.CACHE / f"{w}_Win32_test.npz")
    y, s = t["y"], t["sha256"]
    test_sha_all.append(s)
    rec = {"week": w, "n_benign": int((y == 0).sum()), "n_malware": int((y == 1).sum()),
           "unique_sha_benign": int(len(np.unique(s[y == 0]))), "unique_sha_malware": int(len(np.unique(s[y == 1])))}
    per_seed = {}
    for sd in SEEDS:
        tb = v1.balance(y, sd + wi + 1, cfg["max_test_per_class"])
        ys, ss = y[tb], s[tb]
        per_seed[sd] = {"n": int(len(tb)), "n_benign": int((ys == 0).sum()), "n_malware": int((ys == 1).sum()),
                        "unique_sha": int(len(np.unique(ss))), "dup_extra_rows": int(len(tb) - len(np.unique(ss)))}
    rec["sampled_20k_per_class"] = per_seed
    weeks.append(rec)
out["test_weeks"] = weeks
ts = np.concatenate(test_sha_all)
out["test_total_unique_sha"] = int(len(np.unique(ts)))
out["train_test_sha_overlap"] = int(len(np.intersect1d(np.unique(ts), u)))

# ---- per seed train/val (DCSA LightGBM protocol: plain stratified split, no SHA grouping) ----
seeds = {}
for sd in SEEDS:
    bal = v1.balance(tr.y, sd, cfg["max_train_per_class"])
    tl, vl = v1.train_test_split(np.arange(len(bal)), test_size=cfg["validation_fraction"], random_state=sd, stratify=tr.y[bal])
    ta, va = bal[tl], bal[vl]
    ov = np.isin(tr.sha256[va], tr.sha256[ta])
    # for overlapping pairs: same weekly file or different weekly files?
    tr_week = {}
    for i in ta:
        tr_week.setdefault(tr.sha256[i], set()).add(int(wk[i]))
    same = sum(1 for i in va[ov] if int(wk[i]) in tr_week[tr.sha256[i]])
    seeds[sd] = {"n_balanced": int(len(bal)), "n_benign": int((tr.y[bal] == 0).sum()), "n_malware": int((tr.y[bal] == 1).sum()),
                 "n_train": int(len(ta)), "n_val": int(len(va)), "val_rows_sha_in_train": int(ov.sum()),
                 "overlap_pairs_same_week_file": int(same), "overlap_pairs_different_week_file": int(ov.sum() - same)}
out["seeds_lightgbm_protocol"] = seeds
(v1.OUT / "v5_sample_protocol_check.json").write_text(json.dumps(out, indent=2))
print(json.dumps({k: out[k] for k in ("train_pool", "test_total_unique_sha", "train_test_sha_overlap", "seeds_lightgbm_protocol")}, indent=1))
print(json.dumps(out["test_weeks"][0], indent=1))
