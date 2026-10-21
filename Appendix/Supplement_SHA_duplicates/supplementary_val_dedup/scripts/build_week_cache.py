"""
Parse one EMBER2024 weekly jsonl file into a compact .npz cache
(header_vec, imports_vec, label, sha256, week_id) for header+imports only.
Read-only against the raw dataset; writes only into this workspace's cache/ dir.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fast_vectorize import vectorize_header_batch, vectorize_imports_batch  # noqa: E402

DATA_ROOT = Path(__file__).resolve().parents[4] / "Data"
CACHE_ROOT = Path(__file__).resolve().parents[4] / "Appendix/Supplement_SHA_duplicates/supplementary_val_dedup/cache"


def extract_feature_container(row: dict) -> dict:
    raw = row.get("raw")
    if isinstance(raw, dict) and raw:
        return raw
    return row


def build_week_cache(week: str, platform: str, split: str, force: bool = False) -> Path:
    src = DATA_ROOT / f"{week}_{platform}_{split}.jsonl"
    dst = CACHE_ROOT / f"{week}_{platform}_{split}.npz"
    if dst.exists() and not force:
        return dst
    if not src.exists():
        raise FileNotFoundError(src)

    t0 = time.time()
    labels: list[int] = []
    shas: list[str] = []
    headers: list[dict] = []
    imports: list[dict] = []
    week_id = week

    with src.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            label = row.get("label")
            sha256 = row.get("sha256")
            if label not in (0, 1) or not sha256:
                continue
            raw = extract_feature_container(row)
            labels.append(int(label))
            shas.append(str(sha256))
            headers.append(raw.get("header") or {})
            imports.append(raw.get("imports") or {})

    header_mat = vectorize_header_batch(headers)
    imports_mat = vectorize_imports_batch(imports)
    y = np.asarray(labels, dtype=np.int8)
    sha = np.asarray(shas, dtype="S64")

    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dst,
        header=header_mat,
        imports=imports_mat,
        y=y,
        sha256=sha,
        week_id=np.asarray([week_id] * len(y)),
    )
    dt = time.time() - t0
    print(f"[cache] {src.name}: {len(y)} rows in {dt:.1f}s -> {dst.name}", flush=True)
    return dst


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("week")
    ap.add_argument("platform")
    ap.add_argument("split")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    build_week_cache(args.week, args.platform, args.split, force=args.force)
