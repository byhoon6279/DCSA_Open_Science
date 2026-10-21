"""Stage 3: thesis 5.3.4 / Fig 5-5 (DCSA 5.4) density-conditioned Mix@10, Wild(B).

DCSA config replicated: density_reliability_controlled_main (balance_test=True,
max_test_per_class=20000 per week, pooled over 12 weeks; train 20,000/class;
density = -mean cosine distance to the 10 nearest training samples in the
standardized `all` space; bins at the 25th/75th percentiles; per-sample Mix@10
(euclidean) within the pooled test set, averaged per bin). Mix@10 does not depend
on the classifier, so it is identical for LR and LightGBM (verified on DCSA output).

Conditions:
  dup   -- exactly DCSA's sample (duplicate rows kept)
  dedup -- the same sample with repeated SHA-256 rows removed (first occurrence kept)
           in both the pooled test set and the training sample; scaler, density,
           bins and Mix@10 recomputed.
Usage: s3_section54.py [seed ...]
"""
from __future__ import annotations

import sys
import time

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.preprocessing import normalize  # noqa: E402

from common_dup import (CACHE, GROUPS, RESULTS, SEEDS, WEEKS, BASE_GROUPS,  # noqa: E402
                        balanced_indices, first_occurrence_mask, fit_transform,
                        group_matrix, knn_mismatch, tie_averaged_mix, unique_knn, unique_vectors)

BINS = ["high_density", "mid_density", "low_density"]


def density_scores(Xtr, Xte, k=10, chunk=4096):
    A = normalize(Xtr.astype(np.float64)).astype(np.float32)
    out = np.empty(len(Xte))
    for s in range(0, len(Xte), chunk):
        B = normalize(Xte[s:s + chunk].astype(np.float64)).astype(np.float32)
        d = 1.0 - B @ A.T
        part = np.partition(d, k - 1, axis=1)[:, :k]
        out[s:s + chunk] = -part.mean(axis=1)
    return out


def assign_bins(scores):
    q25, q75 = np.quantile(scores, 0.25), np.quantile(scores, 0.75)
    bins = np.full(len(scores), "mid_density", dtype=object)
    bins[scores <= q25] = "low_density"
    bins[scores >= q75] = "high_density"
    return bins


def main(seeds):
    train = dict(np.load(CACHE / "train_selected.npz", allow_pickle=True))
    selection = np.load(CACHE / "train_selection.npz")
    weeks = [dict(np.load(CACHE / f"test_{w}.npz", allow_pickle=True)) for w in WEEKS["test_weeks"]]
    for seed in seeds:
        t0 = time.time()
        tr_pos = np.searchsorted(train["global_idx"], selection[f"seed_{seed}"])
        parts = []
        for week_idx, wk in enumerate(weeks):
            idx = balanced_indices(wk["label"].astype(int), seed + week_idx + 1, 20000)
            parts.append({g: wk[g][idx] for g in BASE_GROUPS} | {"label": wk["label"][idx], "sha": wk["sha"][idx]})
        test = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
        del parts
        y_all = test["label"].astype(int)

        conds = {
            "dup": (tr_pos, np.arange(len(y_all))),
            "dedup": (tr_pos[first_occurrence_mask(train["sha"][tr_pos])], np.flatnonzero(first_occurrence_mask(test["sha"]))),
        }
        rows = []
        for cond, (tpos, te) in conds.items():
            y = y_all[te]
            Xtr, Xte_all = fit_transform(group_matrix(train, "all", tpos), group_matrix(test, "all", te))
            bins = assign_bins(density_scores(Xtr, Xte_all))
            del Xtr
            for group in GROUPS:
                if group == "all":
                    Xte = Xte_all
                else:
                    _, Xte = fit_transform(group_matrix(train, group, tpos), group_matrix(test, group, te))
                # DCSA implementation (drop nearest entry; ties by sort order) -- dup only, for reproduction
                mix = knn_mismatch(Xte, y, 10).mean(axis=1) if cond == "dup" else np.full(len(y), np.nan)
                U, inv = unique_vectors(Xte)
                nn_idx, nn_dist = unique_knn(U, 15)
                counts = np.zeros((len(U), 2))
                np.add.at(counts, (inv, y), 1)
                ta = tie_averaged_mix(nn_idx, nn_dist, counts, [10])[10][inv, y]
                for b in BINS:
                    m = bins == b
                    rows.append({"seed": seed, "cond": cond, "group": group, "density_bin": b,
                                 "n_samples": int(m.sum()), "n_train": len(tpos), "n_unique_vectors": len(U),
                                 "mix_at_10": float(mix[m].mean()), "mix_ta_at_10": float(ta[m].mean())})
                print(f"seed {seed} {cond} {group} n={len(y)} {time.time() - t0:.0f}s", flush=True)
            del Xte, Xte_all
        pd.DataFrame(rows).to_csv(RESULTS / f"s54_seed{seed}.csv", index=False)


if __name__ == "__main__":
    main([int(s) for s in sys.argv[1:]] or SEEDS)
