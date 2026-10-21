"""Independent verification run (does NOT import supplementary_5_3_val_dedup code).

Per seed (one process per seed):
  1. Re-derive the DCSA balanced 52-week pool -> 80/20 train/val split
     (re-implemented from DCSA_Open_Science/artifacts/shared/common.py::balance_samples
      and run_pe_feature_intervention_validation.py::split_train_validation_indices).
  2. Recompute SHA-256 train/val overlap, val-internal duplicates, cleaned val size;
     check whether same-SHA records have identical feature vectors (duplicate record vs collision).
  3. Fit LightGBM (DCSA hyperparameters, lgbm n_jobs=1 as in the original config) for imports and header.
  4. Permutation importance (roc_auc, n_repeats=3, random_state=seed) on original and cleaned val.
  5. imports_benign_mass_injection at 1/5/10% for important(orig), important(clean), random x20
     using DCSA intervention_seed() incl. selection_offset; perturbation applied column-wise
     exactly like build_perturbed_malware_scaled().
Outputs go only to verification_dcsa_final/results/.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

CACHE = Path(str(Path(__file__).resolve().parents[2] / "Appendix/Supplement_SHA_duplicates/supplementary_val_dedup/cache"))
CFG = (Path(__file__).resolve().parents[2] / "artifacts/5_3_targeted_perturbation_response/"
           "pe_inspired_feature_intervention/results/experiment_results/pe_intervention_lightgbm_balanced_main/results.json")
OUT = Path(__file__).resolve().parents[2] / "validation/results"
OUT.mkdir(parents=True, exist_ok=True)

STRENGTHS = [0.01, 0.05, 0.1]
OPERATOR = "imports_benign_mass_injection"


def log(seed, msg):
    print(f"[{time.strftime('%H:%M:%S')}] seed={seed} {msg}", flush=True)


def load_cfg():
    # results.json is ~16MB; config is the first key
    return json.load(open(CFG))["config"]


class LazyPool:
    """y/sha256 for the full pool in memory; feature rows gathered on demand (memory-safe)."""

    def __init__(self, weeks, split):
        self.files = [CACHE / f"{w}_Win32_{split}.npz" for w in weeks]
        ys, shas, self.offsets = [], [], [0]
        for f in self.files:
            d = np.load(f, allow_pickle=False)
            ys.append(d["y"]); shas.append(d["sha256"]); self.offsets.append(self.offsets[-1] + len(d["y"]))
        self.y = np.concatenate(ys); self.sha256 = np.concatenate(shas)

    def rows(self, group, idx):
        idx = np.asarray(idx)
        out = None
        for fi, f in enumerate(self.files):
            lo, hi = self.offsets[fi], self.offsets[fi + 1]
            m = (idx >= lo) & (idx < hi)
            if not m.any():
                continue
            X = np.load(f, allow_pickle=False)[group]
            if out is None:
                out = np.empty((len(idx), X.shape[1]), dtype=X.dtype)
            out[m] = X[idx[m] - lo]
        return out.astype(np.float64)


def balance(y, seed, max_per_class):
    ben = np.flatnonzero(y == 0)
    mal = np.flatnonzero(y == 1)
    target = min(len(ben), len(mal), max_per_class)
    rng = np.random.default_rng(seed)
    pick = lambda a: a.copy() if len(a) <= target else a[rng.choice(len(a), size=target, replace=False)]
    out = list(pick(ben)) + list(pick(mal))  # python list, like the original list of Samples
    rng.shuffle(out)
    return np.asarray(out)


def stabilize(X):
    X = np.clip(np.nan_to_num(np.asarray(X, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0), -1e12, 1e12)
    return np.sign(X) * np.log1p(np.abs(X))


def transform(sc, X):
    return np.clip(np.nan_to_num(sc.transform(stabilize(X)), nan=0.0, posinf=0.0, neginf=0.0), -100.0, 100.0)


def transform_cols(sc, Xsub, idx):
    s = (stabilize(Xsub) - sc.mean_[idx]) / sc.scale_[idx]
    return np.clip(np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0), -100.0, 100.0)


def proba(clf, X):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return clf.predict_proba(np.asarray(X, dtype=np.float64))[:, 1]


def fit(Xs, y, seed):
    from lightgbm import LGBMClassifier
    clf = LGBMClassifier(objective="binary", class_weight="balanced", n_estimators=300, learning_rate=0.05,
                         num_leaves=31, subsample=0.8, colsample_bytree=0.8, random_state=seed,
                         n_jobs=1, verbosity=-1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(Xs, y)
    return clf


def perm_imp(clf, X, y, seed, n_jobs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = permutation_importance(clf, X, y, scoring="roc_auc", n_repeats=3, random_state=seed, n_jobs=n_jobs)
    return np.asarray(r.importances_mean, dtype=np.float64)


def top_k(imp, k):
    return np.argsort(-imp, kind="stable")[:k]


def iseed(seed, week_idx, group, operator, value, trial_id, sel_type):
    return (seed * 100_000_000 + week_idx * 1_000_000 + sum(map(ord, group)) * 10_000
            + sum(map(ord, operator)) * 10 + int(round(value * 10_000)) + trial_id
            + (0 if sel_type == "important" else 10_000_000))


def run_seed(seed: int, perm_jobs: int = 2):
    cfg = load_cfg()
    log(seed, "loading caches")
    tr = LazyPool(cfg["train_weeks"], "train")
    tests = {w: np.load(CACHE / f"{w}_Win32_test.npz", allow_pickle=False) for w in cfg["test_weeks"]}

    bal = balance(tr.y, seed, cfg["max_train_per_class"])
    yb = tr.y[bal]
    tl, vl = train_test_split(np.arange(len(bal)), test_size=cfg["validation_fraction"], random_state=seed, stratify=yb)
    tr_abs, va_abs = bal[tl], bal[vl]
    sha_tr, sha_va = tr.sha256[tr_abs], tr.sha256[va_abs]
    y_tr, y_va = tr.y[tr_abs], tr.y[va_abs]

    # ---- SHA-256 bookkeeping (independent implementation) ----
    tr_set = set(sha_tr.tolist())
    overlap = np.isin(sha_va, list(tr_set))
    rem = np.flatnonzero(~overlap)
    _, first = np.unique(sha_va[rem], return_index=True)
    keep = np.zeros(len(sha_va), bool)
    keep[rem[np.sort(first)]] = True  # first occurrence, original order
    va_clean_abs = va_abs[keep]
    y_va_clean = tr.y[va_clean_abs]
    assert not (set(tr.sha256[va_clean_abs].tolist()) & tr_set)
    assert len(np.unique(tr.sha256[va_clean_abs])) == len(va_clean_abs)

    # same-SHA records: identical features & labels? (duplicate record vs collision)
    def same_content(sha_list):
        idx_all = np.concatenate([tr_abs, va_abs])
        sha_all = tr.sha256[idx_all]
        n_pairs = n_diff_feat = n_diff_label = 0
        for s in sha_list:
            rows = idx_all[sha_all == s]
            H, I = tr.rows("header", rows), tr.rows("imports", rows)
            for j in range(1, len(rows)):
                n_pairs += 1
                if tr.y[rows[j]] != tr.y[rows[0]]:
                    n_diff_label += 1
                if not (np.array_equal(H[j], H[0]) and np.array_equal(I[j], I[0])):
                    n_diff_feat += 1
        return n_pairs, n_diff_feat, n_diff_label

    ov_shas = np.unique(sha_va[overlap])
    va_rem_sha = sha_va[rem]
    u, c = np.unique(va_rem_sha, return_counts=True)
    internal_dup_shas = u[c > 1]
    pairs_ov = same_content(ov_shas)
    pairs_int = same_content(internal_dup_shas)
    # how many duplicated SHA in the whole 40k pool and in train-only
    u_all, c_all = np.unique(tr.sha256[bal], return_counts=True)
    u_tr, c_tr = np.unique(sha_tr, return_counts=True)
    # pool-level: same SHA across different source weeks?
    dedup = {
        "seed": seed,
        "n_balanced": int(len(bal)), "n_train": int(len(tr_abs)), "n_val_orig": int(len(va_abs)),
        "val_rows_overlapping_train": int(overlap.sum()),
        "val_unique_sha_overlapping_train": int(len(ov_shas)),
        "val_internal_dup_rows_removed": int((~overlap).sum() - keep.sum()),
        "n_val_clean": int(keep.sum()),
        "n_val_clean_benign": int((y_va_clean == 0).sum()), "n_val_clean_malware": int((y_va_clean == 1).sum()),
        "overlap_pairs_checked": pairs_ov[0], "overlap_pairs_feature_mismatch": pairs_ov[1], "overlap_pairs_label_mismatch": pairs_ov[2],
        "internal_pairs_checked": pairs_int[0], "internal_pairs_feature_mismatch": pairs_int[1], "internal_pairs_label_mismatch": pairs_int[2],
        "pool_dup_sha_count": int((c_all > 1).sum()), "pool_dup_extra_rows": int((c_all - 1).sum()),
        "train_internal_dup_extra_rows": int((c_tr - 1).sum()),
        "clean_val_train_sha_intersection": 0,
        "val_abs_md5": hashlib.md5(np.sort(va_abs).tobytes()).hexdigest(),
        "train_abs_md5": hashlib.md5(np.sort(tr_abs).tobytes()).hexdigest(),
    }
    log(seed, f"dedup {dedup['val_rows_overlapping_train']} / {dedup['val_internal_dup_rows_removed']} -> {dedup['n_val_clean']}")

    rows, imp_out, model_fp = [], {}, {}
    for group in ("imports", "header"):
        Xtr = tr.rows(group, tr_abs)
        sc = StandardScaler().fit(stabilize(Xtr))
        clf = fit(transform(sc, Xtr), y_tr, seed)
        model_fp[group] = hashlib.md5(clf.booster_.model_to_string().encode()).hexdigest()
        Xva_o = transform(sc, tr.rows(group, va_abs))
        Xva_c = transform(sc, tr.rows(group, va_clean_abs))
        t0 = time.time()
        imp_o = perm_imp(clf, Xva_o, y_va, seed, perm_jobs)
        imp_c = perm_imp(clf, Xva_c, y_va_clean, seed, perm_jobs)
        log(seed, f"{group} importance done in {time.time()-t0:.0f}s")
        imp_out[group] = {"orig": imp_o, "clean": imp_c,
                          "val_auc_orig": roc_auc_score(y_va, proba(clf, Xva_o)),
                          "val_auc_clean": roc_auc_score(y_va_clean, proba(clf, Xva_c))}
        np.savez_compressed(OUT / f"v1_importance_seed{seed}_{group}.npz", orig=imp_o, clean=imp_c)

        # baseline test AUC per week (model/eval-set fingerprint) for both groups
        benign_tr = Xtr[y_tr == 0]
        q75, q95 = np.percentile(benign_tr, 75, axis=0), np.percentile(benign_tr, 95, axis=0)
        nfeat = Xtr.shape[1]
        for wi, w in enumerate(cfg["test_weeks"]):
            t = tests[w]
            tb = balance(t["y"], seed + wi + 1, cfg["max_test_per_class"])
            yt = t["y"][tb]
            Xt_raw = t[group][tb].astype(np.float64)
            Xt = transform(sc, Xt_raw)
            p0 = proba(clf, Xt)
            base_auc = roc_auc_score(yt, p0)
            rows.append({"seed": seed, "group": group, "week_idx": wi, "test_week": w, "selection": "baseline",
                         "strength": 0.0, "trial": 0, "auc": base_auc, "n_test": int(len(tb)),
                         "n_test_malware": int((yt == 1).sum()),
                         "test_idx_md5": hashlib.md5(tb.tobytes()).hexdigest()})
            if group != "imports":
                continue
            mm = yt == 1
            Xm_raw, Xm, p0m = Xt_raw[mm], Xt[mm], p0[mm]
            b0 = (p0m >= 0.5).astype(int)

            def ev(idx):
                Xp = Xm.copy()
                newraw = np.minimum(np.maximum(Xm_raw[:, idx], q75[idx]), q95[idx])
                Xp[:, idx] = transform_cols(sc, newraw, idx)
                p1 = proba(clf, Xp)
                b1 = (p1 >= 0.5).astype(int)
                return float((b0 != b1).mean()), float(((b0 == 1) & (b1 == 0)).mean())

            for s in STRENGTHS:
                k = int(max(1, min(nfeat, math.ceil(nfeat * s))))
                for sel, imp in (("important_orig", imp_o), ("important_clean", imp_c)):
                    fr, m2b = ev(top_k(imp, k))
                    rows.append({"seed": seed, "group": group, "week_idx": wi, "test_week": w, "selection": sel,
                                 "strength": s, "trial": 0, "k": k, "flip_rate": fr, "m2b_rate": m2b})
                for trial in range(1, cfg["random_trials"] + 1):
                    rng = np.random.default_rng(iseed(seed, wi, group, OPERATOR, s, trial, "random"))
                    ridx = np.asarray(rng.choice(np.arange(nfeat), size=k, replace=False), dtype=int)
                    fr, m2b = ev(ridx)
                    rows.append({"seed": seed, "group": group, "week_idx": wi, "test_week": w, "selection": "random",
                                 "strength": s, "trial": trial, "k": k, "flip_rate": fr, "m2b_rate": m2b})
        log(seed, f"{group} interventions done")

    # top-k set comparison (imports and header) at each strength
    topk = []
    for group in ("imports", "header"):
        io, ic = imp_out[group]["orig"], imp_out[group]["clean"]
        n = len(io)
        rank_o = np.empty(n, int); rank_o[np.argsort(-io, kind="stable")] = np.arange(1, n + 1)
        rank_c = np.empty(n, int); rank_c[np.argsort(-ic, kind="stable")] = np.arange(1, n + 1)
        from scipy.stats import spearmanr
        for s in STRENGTHS:
            k = int(max(1, math.ceil(n * s)))
            a, b = set(top_k(io, k).tolist()), set(top_k(ic, k).tolist())
            inter = sorted(a & b)
            topk.append({"seed": seed, "group": group, "strength": s, "k": k, "intersection": len(inter),
                         "jaccard": len(inter) / len(a | b),
                         "orig_topk_median_rank_in_clean": float(np.median(rank_c[list(a)])),
                         "dropped_from_topk": len(a - b), "spearman_all_features": float(spearmanr(io, ic).statistic),
                         "n_features_imp_orig_gt0": int((io > 0).sum()), "n_features_imp_clean_gt0": int((ic > 0).sum()),
                         "top10_orig": top_k(io, 10).tolist(), "top10_clean": top_k(ic, 10).tolist()})

    summary = {"dedup": dedup, "model_md5": model_fp, "topk": topk,
               "val_auc": {g: {"orig": imp_out[g]["val_auc_orig"], "clean": imp_out[g]["val_auc_clean"]} for g in imp_out}}
    (OUT / f"v1_seed{seed}_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    import pandas as pd
    pd.DataFrame(rows).to_csv(OUT / f"v1_seed{seed}_rows.csv", index=False)
    log(seed, "DONE")


if __name__ == "__main__":
    run_seed(int(sys.argv[1]), perm_jobs=int(sys.argv[2]) if len(sys.argv) > 2 else 2)
