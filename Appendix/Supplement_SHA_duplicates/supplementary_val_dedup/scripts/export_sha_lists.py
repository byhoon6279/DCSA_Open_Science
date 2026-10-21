"""Export per-seed SHA-256 audit lists (train, val-original, val-cleaned,
removed-due-to-train-overlap, removed-due-to-internal-duplicate) so the
dedup step is independently auditable."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_pool import load_pool  # noqa: E402
import model_utils as mu  # noqa: E402

CONFIG = json.loads(Path(
    str(Path(__file__).resolve().parents[4]) + "/artifacts/5_3_targeted_perturbation_response/"
    "configs/LightGBM/feature_perturbation_lightgbm_balanced_main.json"
).read_text())
SEEDS = [int(s) for s in CONFIG["seeds"]]
MAX_TRAIN_PER_CLASS = CONFIG["max_train_per_class"]
VALIDATION_FRACTION = float(CONFIG["validation_fraction"])
PLATFORM = CONFIG["platform"]
TRAIN_WEEKS = CONFIG["train_weeks"]

OUT_DIR = Path(__file__).resolve().parents[4] / "Appendix/Supplement_SHA_duplicates/supplementary_val_dedup/results/sha_lists"
OUT_DIR.mkdir(parents=True, exist_ok=True)

pool = load_pool(TRAIN_WEEKS, PLATFORM, "train")

for seed in SEEDS:
    balanced_idx = mu.balance_indices(pool.y, seed=seed, max_per_class=MAX_TRAIN_PER_CLASS)
    y_balanced = pool.y[balanced_idx]
    sha_balanced = pool.sha[balanced_idx]
    week_balanced = pool.week[balanced_idx]

    train_local, val_local = mu.split_train_val(
        n_samples=len(balanced_idx), y=y_balanced, seed=seed, validation_fraction=VALIDATION_FRACTION,
    )

    sha_train = sha_balanced[train_local]
    y_train = y_balanced[train_local]
    week_train = week_balanced[train_local]

    sha_val = sha_balanced[val_local]
    y_val = y_balanced[val_local]
    week_val = week_balanced[val_local]

    train_sha_set = set(sha_train.tolist())
    overlap_mask = np.array([s in train_sha_set for s in sha_val])

    seen = {}
    keep_mask = np.zeros(len(sha_val), dtype=bool)
    dup_of = [None] * len(sha_val)
    for i in np.flatnonzero(~overlap_mask):
        s = sha_val[i]
        if s not in seen:
            seen[s] = i
            keep_mask[i] = True
        else:
            dup_of[i] = seen[s]

    def decode(arr):
        return [x.decode() if isinstance(x, bytes) else str(x) for x in arr]

    pd.DataFrame({
        "sha256": decode(sha_train), "label": y_train, "week_id": decode(week_train),
    }).to_csv(OUT_DIR / f"seed{seed}_train.csv", index=False)

    pd.DataFrame({
        "sha256": decode(sha_val), "label": y_val, "week_id": decode(week_val),
        "removed_train_overlap": overlap_mask,
        "kept_in_cleaned_val": keep_mask & ~overlap_mask,
        "internal_duplicate_of_row": dup_of,
    }).to_csv(OUT_DIR / f"seed{seed}_val_original_with_flags.csv", index=False)

    print(f"seed={seed}: exported train_n={len(sha_train)}, val_n={len(sha_val)} "
          f"(overlap_removed={int(overlap_mask.sum())}, internal_dup_removed={int((~overlap_mask & ~keep_mask).sum())})")

print("SHA_LISTS_DONE")
