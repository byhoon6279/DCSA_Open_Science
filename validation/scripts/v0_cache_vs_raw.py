"""Compare supplementary cache (.npz, float32) against raw EMBER2024 JSONL vectorized with
DCSA_Open_Science/artifacts/shared/common.py (original float64 per-row functions). Read-only."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]) + "/artifacts/shared")
from common import load_samples_from_jsonl, select_feature_group  # type: ignore
RAW = str(Path(__file__).resolve().parents[2] / "Data")
CACHE = str(Path(__file__).resolve().parents[2] / "Appendix/Supplement_SHA_duplicates/supplementary_val_dedup/cache")
out = []
for week, split in [(sys.argv[1], "train"), (sys.argv[2], "test")]:
    s = load_samples_from_jsonl(__import__("pathlib").Path(f"{RAW}/{week}_Win32_{split}.jsonl"), packer_filter="all")
    c = np.load(f"{CACHE}/{week}_Win32_{split}.npz")
    r = {"week": week, "split": split, "n_raw": len(s), "n_cache": int(len(c["y"]))}
    r["sha_order_equal"] = bool(len(s) == len(c["y"]) and all(a.sha256 == b.decode() for a, b in zip(s, c["sha256"])))
    r["label_equal"] = bool(np.array_equal(np.array([x.label for x in s]), c["y"].astype(int)))
    for g in ("header", "imports"):
        X = select_feature_group(s, g)
        C = c[g].astype(np.float64)
        d = np.abs(X - C)
        r[f"{g}_exact_equal"] = bool(np.array_equal(X, C))
        r[f"{g}_n_cells_diff"] = int((d > 0).sum())
        r[f"{g}_max_abs_diff"] = float(d.max())
        r[f"{g}_max_rel_diff"] = float((d / np.maximum(np.abs(X), 1e-12)).max())
        r[f"{g}_max_abs_raw_value"] = float(np.abs(X).max())
    out.append(r); print(json.dumps(r), flush=True)
json.dump(out, open(str(Path(__file__).resolve().parents[2] / "validation/results/v0_cache_vs_raw.json"), "w"), indent=2)
