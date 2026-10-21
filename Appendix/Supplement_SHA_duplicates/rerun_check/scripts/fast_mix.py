"""Vectorised tie-averaged Mix@k (same definition as ch5_dup_sensitivity common_dup.tie_averaged_mix):
self excluded, neighbours at numerically equal distance pooled into one tie group, expectation
over uniformly random tie-breaking. Also reports how often the last retrieved tie group may
extend beyond the retrieved neighbour list (possible truncation)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "Appendix/Supplement_SHA_duplicates/ch5_dup_sensitivity/scripts"))
import numpy as np  # noqa: E402

from common_dup import unique_knn, unique_vectors  # noqa: E402,F401


def tie_averaged_mix_vec(nn_idx, nn_dist, counts, k_values, rel_tol=1e-5):
    n_u, m = nn_idx.shape
    rows = np.arange(n_u)
    d = nn_dist.astype(np.float64)
    newgrp = np.ones((n_u, m), dtype=bool)
    newgrp[:, 1:] = np.abs(d[:, 1:] - d[:, :-1]) > rel_tol * np.maximum(1.0, d[:, :-1])
    gid = np.cumsum(newgrp, axis=1) - 1
    out = {}
    for lab in (0, 1):
        cs = counts[nn_idx, lab].astype(np.float64)
        co = counts[nn_idx, 1 - lab].astype(np.float64)
        cs[:, 0] -= 1.0  # column 0 is the query vector itself: exclude the query sample
        GS = np.zeros((n_u, m)); GO = np.zeros((n_u, m))
        for j in range(m):
            np.add.at(GS, (rows, gid[:, j]), cs[:, j])
            np.add.at(GO, (rows, gid[:, j]), co[:, j])
        GT = GS + GO
        before = np.cumsum(GT, axis=1) - GT
        frac = np.divide(GO, GT, out=np.zeros_like(GO), where=GT > 0)
        for k in k_values:
            take = np.clip(k - before, 0.0, GT)
            filled = take.sum(axis=1)
            val = np.divide((take * frac).sum(axis=1), filled, out=np.full(n_u, np.nan), where=filled > 0)
            val[counts[:, lab] == 0] = np.nan
            out.setdefault(k, np.full((n_u, 2), np.nan))[:, lab] = val
    last_group_open = newgrp[:, -1] == 0  # last column tied with previous: group may continue
    return out, float(last_group_open.mean())


def mix_ta(X, y, k_values, m_extra=5):
    """Per-sample tie-averaged Mix@k for each k; returns ({k: per-sample array}, open_rate)."""
    U, inv = unique_vectors(X)
    nn_idx, nn_dist = unique_knn(U, max(k_values) + m_extra)
    counts = np.zeros((len(U), 2)); np.add.at(counts, (inv, y), 1)
    res, open_rate = tie_averaged_mix_vec(nn_idx, nn_dist, counts, k_values)
    return {k: res[k][inv, y] for k in k_values}, open_rate
