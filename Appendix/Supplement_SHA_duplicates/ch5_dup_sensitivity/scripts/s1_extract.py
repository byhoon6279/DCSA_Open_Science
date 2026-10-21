"""Stage 1: build feature caches identical to what DCSA's loaders produce.

- Light pass over the 52 train files (label + sha256 per line, regex) -> replicate
  DCSA's balanced train selection (20,000/class) for each seed.
- Full DCSA build_sample() parse for the selected train rows and for every row of
  the 12 test weeks. Raw data and DCSA code are read-only.
"""
from __future__ import annotations

import json
import re
import sys
from multiprocessing import Pool

sys.dont_write_bytecode = True
from common_dup import CACHE, DATA_ROOT, SEEDS, WEEKS, balanced_indices, BASE_GROUPS  # noqa: E402
import numpy as np  # noqa: E402

LABEL_RE = re.compile(r'"label":\s*(-?\d+|null)')
SHA_RE = re.compile(r'"sha256":\s*"([0-9a-f]+)"')


def light_pass(week: str):
    labels, shas = [], []
    with open(DATA_ROOT / f"{week}_Win32_train.jsonl", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            head = line[:4000]
            lab = LABEL_RE.search(head).group(1)
            labels.append(int(lab) if lab != "null" else -1)
            shas.append(SHA_RE.search(head).group(1))
    return week, np.array(labels, dtype=np.int8), np.array(shas)


def parse_rows(args):
    from common import build_sample  # DCSA, read-only
    path, wanted = args
    wanted = None if wanted is None else set(int(i) for i in wanted)
    out = {g: [] for g in BASE_GROUPS}
    lab, sha, line_ids = [], [], []
    with open(path, encoding="utf-8") as f:
        i = -1
        for line in f:
            line = line.strip()
            if not line:
                continue
            i += 1
            if wanted is not None and i not in wanted:
                continue
            s = build_sample(json.loads(line), path)
            if s is None:
                continue
            for g in BASE_GROUPS:
                out[g].append(getattr(s, f"{g}_vec"))
            lab.append(s.label); sha.append(s.sha256); line_ids.append(i)
    return str(path), {g: np.stack(v).astype(np.float32) for g, v in out.items()}, np.array(lab, np.int8), np.array(sha), np.array(line_ids)


def main():
    train_weeks, test_weeks = WEEKS["train_weeks"], WEEKS["test_weeks"]
    with Pool(8) as pool:
        light = pool.map(light_pass, train_weeks)
    labels = np.concatenate([l for _, l, _ in light])
    shas = np.concatenate([s for _, _, s in light])
    offsets = np.cumsum([0] + [len(l) for _, l, _ in light])
    assert set(np.unique(labels)) <= {0, 1}, "rows DCSA would drop exist; handle explicitly"
    print("train rows", len(labels), "unique sha", len(np.unique(shas)), flush=True)

    sel = {seed: balanced_indices(labels, seed, 20000) for seed in SEEDS}
    union = np.unique(np.concatenate(list(sel.values())))
    jobs = []
    for w, week in enumerate(train_weeks):
        lo, hi = offsets[w], offsets[w + 1]
        local = union[(union >= lo) & (union < hi)] - lo
        if len(local):
            jobs.append((DATA_ROOT / f"{week}_Win32_train.jsonl", local))
    test_jobs = [(DATA_ROOT / f"{week}_Win32_test.jsonl", None) for week in test_weeks]

    with Pool(10) as pool:
        train_parts = pool.map(parse_rows, jobs)
        test_parts = pool.map(parse_rows, test_jobs)

    # train cache: rows in global-index order
    week_off = {str(DATA_ROOT / f"{wk}_Win32_train.jsonl"): offsets[w] for w, wk in enumerate(train_weeks)}
    gidx = np.concatenate([week_off[p] + ids for p, _, _, _, ids in train_parts])
    order = np.argsort(gidx)
    train = {g: np.concatenate([d[g] for _, d, _, _, _ in train_parts])[order] for g in BASE_GROUPS}
    np.savez(CACHE / "train_selected.npz", global_idx=gidx[order],
             label=np.concatenate([l for _, _, l, _, _ in train_parts])[order],
             sha=np.concatenate([s for _, _, _, s, _ in train_parts])[order], **train)
    assert np.array_equal(labels[gidx[order]], np.load(CACHE / "train_selected.npz")["label"])
    np.savez(CACHE / "train_selection.npz", **{f"seed_{s}": v for s, v in sel.items()})
    for (p, d, l, s, ids), week in zip(test_parts, test_weeks):
        assert np.array_equal(ids, np.arange(len(ids)))
        np.savez(CACHE / f"test_{week}.npz", label=l, sha=s, **d)
        print("test", week, len(l), "unique", len(np.unique(s)), flush=True)
    print("dims", {g: train[g].shape[1] for g in BASE_GROUPS}, "train union rows", len(union), flush=True)


if __name__ == "__main__":
    main()
