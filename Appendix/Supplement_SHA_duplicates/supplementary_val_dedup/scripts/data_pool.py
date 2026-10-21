"""Load cached weekly .npz files into concatenated pool arrays, preserving the
exact ordering the original DCSA scripts use (train_weeks / test_weeks list order,
then within-file row order) so that seeded random draws reproduce identically."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np

CACHE_ROOT = Path(__file__).resolve().parents[4] / "Appendix/Supplement_SHA_duplicates/supplementary_val_dedup/cache"


class Pool:
    def __init__(self, header: np.ndarray, imports: np.ndarray, y: np.ndarray, sha: np.ndarray, week: np.ndarray):
        self.header = header
        self.imports = imports
        self.y = y
        self.sha = sha
        self.week = week

    def __len__(self) -> int:
        return len(self.y)

    def group(self, name: str) -> np.ndarray:
        if name == "header":
            return self.header
        if name == "imports":
            return self.imports
        raise ValueError(name)


def load_week(week: str, platform: str, split: str) -> Dict[str, np.ndarray]:
    path = CACHE_ROOT / f"{week}_{platform}_{split}.npz"
    data = np.load(path, allow_pickle=False)
    return {
        "header": data["header"],
        "imports": data["imports"],
        "y": data["y"],
        "sha256": data["sha256"],
        "week_id": data["week_id"],
    }


def load_pool(weeks: List[str], platform: str, split: str) -> Pool:
    headers, imports_, ys, shas, week_ids = [], [], [], [], []
    for week in weeks:  # preserve config list order exactly
        d = load_week(week, platform, split)
        headers.append(d["header"])
        imports_.append(d["imports"])
        ys.append(d["y"])
        shas.append(d["sha256"])
        week_ids.append(d["week_id"])
    return Pool(
        header=np.concatenate(headers, axis=0),
        imports=np.concatenate(imports_, axis=0),
        y=np.concatenate(ys, axis=0),
        sha=np.concatenate(shas, axis=0),
        week=np.concatenate(week_ids, axis=0),
    )


def load_test_weeks(weeks: List[str], platform: str) -> Dict[str, Pool]:
    out = {}
    for week in weeks:
        d = load_week(week, platform, "test")
        out[week] = Pool(d["header"], d["imports"], d["y"], d["sha256"], d["week_id"])
    return out
