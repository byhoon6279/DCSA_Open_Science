"""Shared helpers for the chapter-5 SHA-duplicate sensitivity check.

DCSA code is imported read-only (bytecode writing disabled so nothing is
written into DCSA_Open_Science). Sampling and preprocessing replicate
DCSA's balance_samples / stabilize_features / StandardScaler / clip logic.
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[4]) + "/artifacts/shared")

import numpy as np  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from common import stabilize_features  # noqa: E402,F401

WS = Path(__file__).resolve().parents[4] / "Appendix/Supplement_SHA_duplicates/ch5_dup_sensitivity"
CACHE = WS / "cache"
RESULTS = WS / "results"
DATA_ROOT = Path(__file__).resolve().parents[4] / "Data"
WEEKS = json.loads((WS / "scripts" / "weeks.json").read_text())
SEEDS = [13, 42, 77, 123, 2024]
GROUPS = ["header", "section", "imports", "strings", "all"]
BASE_GROUPS = ["header", "section", "imports", "strings"]


def balanced_indices(labels: np.ndarray, seed: int, max_per_class: int | None) -> np.ndarray:
    """Index-level replica of DCSA balance_samples / balance_records (same RNG call order)."""
    benign = np.flatnonzero(labels == 0)
    malware = np.flatnonzero(labels == 1)
    target = min(len(benign), len(malware))
    if max_per_class is not None:
        target = min(target, max_per_class)
    rng = np.random.default_rng(seed)

    def pick(rows: np.ndarray) -> list:
        if len(rows) <= target:
            return list(rows)
        idx = rng.choice(len(rows), size=target, replace=False)
        return [rows[i] for i in idx]

    balanced = pick(benign) + pick(malware)
    rng.shuffle(balanced)
    return np.asarray(balanced, dtype=np.int64)


def group_matrix(store: dict, group: str, idx: np.ndarray | None = None) -> np.ndarray:
    """Raw float32 matrix (DCSA vectors are float32; DCSA's float64 cast is exact)."""
    parts = BASE_GROUPS if group == "all" else [group]
    mats = [store[p] if idx is None else store[p][idx] for p in parts]
    return np.concatenate(mats, axis=1) if len(mats) > 1 else mats[0]


def fit_transform(X_train_raw: np.ndarray, X_test_raw: np.ndarray, chunk: int = 50000):
    """DCSA preprocessing: signed-log -> StandardScaler fit on train -> nan_to_num -> clip [-100, 100].
    Computed in float64 chunk by chunk; returned as float32."""
    scaler = StandardScaler().fit(stabilize_features(X_train_raw))

    def tr(X):
        out = np.empty(X.shape, dtype=np.float32)
        for s in range(0, len(X), chunk):
            Z = scaler.transform(stabilize_features(X[s:s + chunk]))
            out[s:s + chunk] = np.clip(np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0), -100.0, 100.0)
        return out

    return tr(X_train_raw), tr(X_test_raw)


def first_occurrence_mask(sha: np.ndarray) -> np.ndarray:
    _, first = np.unique(sha, return_index=True)
    mask = np.zeros(len(sha), dtype=bool)
    mask[first] = True
    return mask


def _chunks(n: int, size: int):
    return [(s, min(s + size, n)) for s in range(0, n, size)]


def knn_mismatch(X: np.ndarray, y: np.ndarray, k: int, chunk: int | None = None, workers: int = 10) -> np.ndarray:
    """Returns an (n, k) bool matrix: neighbour j (sorted by distance) has a different label."""
    """Per-sample mismatch among the k nearest neighbours (euclidean), query row excluded
    by index (an identical-vector twin stays a candidate).
    Exact brute force; float32 GEMM for distances."""
    Xf = np.ascontiguousarray(X, dtype=np.float32)
    sq = np.einsum("ij,ij->i", Xf, Xf)
    out = np.empty((len(Xf), k), dtype=bool)
    chunk = chunk or max(64, min(1024, int(1e8 // len(Xf))))

    def work(bounds):
        s, e = bounds
        d = sq[s:e, None] + sq[None, :] - 2.0 * (Xf[s:e] @ Xf.T)
        part = np.argpartition(d, k + 1, axis=1)[:, : k + 1]
        dsel = np.take_along_axis(d, part, axis=1)
        order = np.argsort(dsel, axis=1, kind="stable")
        nb_all = np.take_along_axis(part, order, axis=1)
        is_self = nb_all == (s + np.arange(e - s))[:, None]
        nb = np.take_along_axis(nb_all, np.argsort(is_self, axis=1, kind="stable"), axis=1)[:, :k]
        out[s:e] = y[nb] != y[s:e, None]

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(work, _chunks(len(Xf), chunk)))
    return out


def silhouette_sums(X: np.ndarray, y: np.ndarray, chunk: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """For each sample: sum of euclidean distances to same-class points (S) and
    other-class points (T). Lets silhouette be computed exactly for the unique set
    and for the set where every point appears twice."""
    Xd = np.ascontiguousarray(X, dtype=np.float64)
    sq = np.einsum("ij,ij->i", Xd, Xd)
    onehot = np.stack([y == 0, y == 1], axis=1).astype(np.float64)
    S = np.empty(len(Xd)); T = np.empty(len(Xd))

    def work(bounds):
        s, e = bounds
        d = np.sqrt(np.clip(sq[s:e, None] + sq[None, :] - 2.0 * (Xd[s:e] @ Xd.T), 0.0, None))
        rows = np.arange(e - s)
        d[rows, rows + s] = 0.0
        sums = d @ onehot
        S[s:e] = sums[rows, y[s:e]]
        T[s:e] = sums[rows, 1 - y[s:e]]

    with ThreadPoolExecutor(4) as ex:
        list(ex.map(work, _chunks(len(Xd), chunk)))
    return S, T


def silhouette_from_sums(S, T, y, dup_factor: int) -> float:
    """dup_factor=1: unique set. dup_factor=2: every point present exactly twice
    (twin at distance 0 counts in the same-class mean)."""
    n_c = np.array([(y == 0).sum(), (y == 1).sum()])
    m_same = n_c[y]; m_other = n_c[1 - y]
    a = dup_factor * S / (dup_factor * m_same - 1)
    b = T / m_other
    s = (b - a) / np.maximum(a, b)
    return float(np.nan_to_num(s).mean())


def unique_vectors(X: np.ndarray):
    """Group bit-identical rows. Returns (U, inv)."""
    U, inv = np.unique(np.ascontiguousarray(X, dtype=np.float32), axis=0, return_inverse=True)
    return U, inv.ravel()


def unique_knn(U: np.ndarray, m: int, chunk: int | None = None, workers: int = 10):
    """For each unique vector: indices/distances of its m nearest unique vectors
    (itself first, distance 0), sorted by distance."""
    Uf = np.ascontiguousarray(U, dtype=np.float32)
    sq = np.einsum("ij,ij->i", Uf, Uf)
    m = min(m, len(Uf))
    idx = np.empty((len(Uf), m), dtype=np.int64)
    dist = np.empty((len(Uf), m), dtype=np.float32)
    chunk = chunk or max(64, min(1024, int(1e8 // len(Uf))))

    def work(bounds):
        s, e = bounds
        d = sq[s:e, None] + sq[None, :] - 2.0 * (Uf[s:e] @ Uf.T)
        rows = np.arange(e - s)
        d[rows, rows + s] = -1.0  # force self first
        part = np.argpartition(d, m - 1, axis=1)[:, :m]
        dsel = np.take_along_axis(d, part, axis=1)
        order = np.argsort(dsel, axis=1, kind="stable")
        idx[s:e] = np.take_along_axis(part, order, axis=1)
        dist[s:e] = np.take_along_axis(dsel, order, axis=1)

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(work, _chunks(len(Uf), chunk)))
    dist[:, 0] = 0.0
    return idx, np.sqrt(np.clip(dist, 0.0, None))


def tie_averaged_mix(nn_idx, nn_dist, counts: np.ndarray, k_values, rel_tol: float = 1e-5):
    """Expected Mix@k under uniformly random tie-breaking, self excluded.

    counts: (n_unique, 2) per-class multiplicity of each unique vector in the evaluated set.
    Returns {k: (n_unique, 2) array} = expected mismatch rate for a query sample with
    vector u and label l. Neighbours at (numerically) equal distance form one tie group.
    """
    kmax = max(k_values)
    n_u, m = nn_idx.shape
    res = {k: np.full((n_u, 2), np.nan) for k in k_values}
    for u in range(n_u):
        for lab in (0, 1):
            if counts[u, lab] == 0:
                continue
            # build tie groups: (same-label count, other-label count)
            groups = []
            cur_d, s_cnt, o_cnt = None, 0.0, 0.0
            for j in range(m):
                v, d = nn_idx[u, j], nn_dist[u, j]
                cs, co = counts[v, lab], counts[v, 1 - lab]
                if v == u:
                    cs -= 1  # exclude the query sample itself
                if cur_d is not None and abs(d - cur_d) > rel_tol * max(1.0, cur_d):
                    groups.append((s_cnt, o_cnt)); s_cnt = o_cnt = 0.0
                cur_d = d if cur_d is None or abs(d - cur_d) > rel_tol * max(1.0, cur_d) else cur_d
                s_cnt += cs; o_cnt += co
            groups.append((s_cnt, o_cnt))
            for k in k_values:
                filled, mism = 0.0, 0.0
                for s_c, o_c in groups:
                    tot = s_c + o_c
                    if tot == 0:
                        continue
                    take = min(tot, k - filled)
                    mism += take * o_c / tot
                    filled += take
                    if filled >= k:
                        break
                res[k][u, lab] = mism / filled if filled > 0 else np.nan
    return res
