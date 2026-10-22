"""Stage 2: manuscript Section 5.1 Wild(B) LR run -- Mix@k and silhouette per
(seed, week, feature group), with all test rows (DCSA default: duplicate rows kept)
vs. one row per SHA-256 (dedup). Train sample / scaler identical to DCSA in both.

DCSA config replicated: win32_all_train_all_test_balanced_test (balance_test=True,
max_test_per_class=None -> every row of each test week), mix_k_values 5/10/15/20.
Silhouette: exact, from per-sample distance sums on the unique set; the duplicate-kept
value is derived exactly because every file appears exactly twice within a week
(asserted).
Usage: s2_section51.py [seed ...]   (default: all 5 seeds)
"""
from __future__ import annotations

import sys
import time

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common_dup import (CACHE, GROUPS, RESULTS, SEEDS, WEEKS, first_occurrence_mask,  # noqa: E402
                        fit_transform, group_matrix, knn_mismatch, silhouette_from_sums,
                        silhouette_sums, tie_averaged_mix, unique_knn, unique_vectors)

K_VALUES = [5, 10, 15, 20]


def main(seeds):
    train = dict(np.load(CACHE / "train_selected.npz", allow_pickle=True))
    selection = np.load(CACHE / "train_selection.npz")
    for seed in seeds:
        rows = []
        pos = np.searchsorted(train["global_idx"], selection[f"seed_{seed}"])
        out_path = RESULTS / f"s51_seed{seed}.csv"
        for week in WEEKS["test_weeks"]:
            t0 = time.time()
            test = dict(np.load(CACHE / f"test_{week}.npz", allow_pickle=True))
            y = test["label"].astype(int)
            keep = first_occurrence_mask(test["sha"])
            _, counts = np.unique(test["sha"], return_counts=True)
            assert (counts == 2).all(), "silhouette derivation assumes exact 2x duplication"
            for group in GROUPS:
                _, X = fit_transform(group_matrix(train, group, pos), group_matrix(test, group))
                U, inv = unique_vectors(X)
                nn_idx, nn_dist = unique_knn(U, max(K_VALUES) + 5)
                for cond, idx in [("dup", np.arange(len(y))), ("dedup", np.flatnonzero(keep))]:
                    # DCSA implementation: drop the single nearest entry, ties broken by sort order
                    mm = knn_mismatch(X[idx], y[idx], max(K_VALUES))
                    row = {"seed": seed, "week": week, "group": group, "cond": cond, "n": len(idx),
                           "n_unique_vectors": len(U)}
                    row.update({f"mix_at_{k}": float(mm[:, :k].mean()) for k in K_VALUES})
                    # tie-averaged: self excluded, expectation over random tie-breaking
                    counts = np.zeros((len(U), 2))
                    np.add.at(counts, (inv[idx], y[idx]), 1)
                    ta = tie_averaged_mix(nn_idx, nn_dist, counts, K_VALUES)
                    row.update({f"mix_ta_at_{k}": float(ta[k][inv[idx], y[idx]].mean()) for k in K_VALUES})
                    rows.append(row)
                S, T = silhouette_sums(X[keep], y[keep])
                for row, factor in zip(rows[-2:], (2, 1)):
                    row["silhouette"] = silhouette_from_sums(S, T, y[keep], factor)
            pd.DataFrame(rows).to_csv(out_path, index=False)
            print(f"seed {seed} week {week} done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main([int(s) for s in sys.argv[1:]] or SEEDS)
