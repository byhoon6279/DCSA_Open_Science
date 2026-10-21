"""Verify fast_vectorize.py's batched functions produce IDENTICAL output to
DCSA_Open_Science's original per-row common.py functions, on real sample rows.
Read-only against DCSA_Open_Science. Run before trusting any bulk-loaded data."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]) + "/artifacts/shared")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import vectorize_header, vectorize_imports, extract_feature_container  # type: ignore
from fast_vectorize import vectorize_header_batch, vectorize_imports_batch

SAMPLE_FILE = str(Path(__file__).resolve().parents[4] / "Data") + "/2023-09-24_2023-09-30_Win32_train.jsonl"

N_TEST = 500

rows = []
with open(SAMPLE_FILE, "r", encoding="utf-8") as fh:
    for i, line in enumerate(fh):
        if i >= N_TEST:
            break
        rows.append(json.loads(line))

raws = [extract_feature_container(r) for r in rows]
headers = [r.get("header") or {} for r in raws]
imports = [r.get("imports") or {} for r in raws]

orig_header = np.stack([vectorize_header(h) for h in headers])
orig_imports = np.stack([vectorize_imports(im) for im in imports])

batch_header = vectorize_header_batch(headers)
batch_imports = vectorize_imports_batch(imports)

h_ok = np.array_equal(orig_header, batch_header)
i_ok = np.allclose(orig_imports, batch_imports, atol=0, rtol=0)

print(f"n_rows_tested={len(rows)}")
print(f"header match: {h_ok}  shape={orig_header.shape}")
print(f"imports match: {i_ok}  shape={orig_imports.shape}")
if not h_ok:
    diff = np.abs(orig_header - batch_header)
    print("header max abs diff:", diff.max(), "at", np.unravel_index(diff.argmax(), diff.shape))
if not i_ok:
    diff = np.abs(orig_imports - batch_imports)
    print("imports max abs diff:", diff.max(), "at", np.unravel_index(diff.argmax(), diff.shape))

assert h_ok and i_ok, "Batched vectorization does NOT match original -- do not proceed."
print("OK: batched vectorization is bit-identical to common.py originals on this sample.")
