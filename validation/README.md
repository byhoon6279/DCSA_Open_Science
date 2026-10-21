# Validation: train–validation SHA overlap and sampling settings

This folder documents two checks on the LightGBM targeted-stress results in `artifacts/5_3_targeted_perturbation_response/`:

1. **Validation protocol.** The 10% hashed-import intervention result (malware-to-benign rate 0.308) comes from a label-stratified 20% validation split **without SHA-group control**. Each EMBER 2024 training file appears twice within the same weekly file, so 84–99 validation rows per seed share a SHA-256 with a training row. The MLP extension uses a SHA-grouped split instead (`artifacts/MLP/scripts/mlp_core.py`).
2. **Sampling settings.** All main models train on 20,000 samples per class per seed. The targeted-stress, intervention and density analyses evaluate on 20,000 samples per class per test week.


## Contents

| Path | What it is |
|---|---|
| `scripts/v0_cache_vs_raw.py` | Confirms that the cached weekly feature arrays are identical to raw EMBER 2024 JSONL vectorized with `artifacts/shared/common.py` |
| `scripts/v1_dedup_importance_intervention.py` | Re-derives the balanced pool and the train/validation split per seed. Counts SHA overlap and internal duplicates. Re-runs permutation importance on the original and cleaned validation sets and the 1/5/10% hashed-import intervention |
| `scripts/v3_compare_and_aggregate.py` | Compares v1 with the original `trial_results.csv` (exact match) and aggregates per seed |
| `scripts/v5_sample_protocol_check.py` | Checks sample sizes, the duplicate structure (same-week twin rows) and train–test SHA overlap |
| `results/` | Outputs of v0, v1, v3 and v5 (JSON/CSV/NPZ) |
| `data/sha_lists/` | Per-seed SHA-256 lists for the training and validation splits, with flags for rows removed in the cleaned validation set |
| `logs/` | Run logs |

## Key numbers

| Seed | Val rows sharing SHA with train | Val internal duplicates | Cleaned val | M→B important (original) | M→B important (cleaned val) | M→B random |
|---|---|---|---|---|---|---|
| 13 | 99 | 10 | 7,891 | 0.3531 | 0.1068 | 0.0382 |
| 42 | 85 | 8 | 7,907 | 0.3484 | 0.2923 | 0.0288 |
| 77 | 92 | 9 | 7,899 | 0.2603 | 0.2784 | 0.0240 |
| 123 | 98 | 12 | 7,890 | 0.2683 | 0.0633 | 0.0194 |
| 2024 | 84 | 9 | 7,907 | 0.3106 | 0.0610 | 0.0289 |
| Mean | | | | **0.3081** | **0.1604** | **0.0279** |

## Reproducing

All paths are resolved relative to the repository root (via `Path(__file__)`), so the scripts run from any clone location. Place the raw EMBER 2024 JSONL files directly under `Data/` (see the repository-level README). The weekly feature caches (about 1.1 GB) are not included; they are read from `Appendix/Supplement_SHA_duplicates/supplementary_val_dedup/cache/` and can be rebuilt with `Appendix/Supplement_SHA_duplicates/supplementary_val_dedup/scripts/build_all_caches.py`, and `v0` shows they match the raw vectorization exactly.

```bash
python3 scripts/v0_cache_vs_raw.py 2024-03-03_2024-03-09 2024-10-20_2024-10-26
for s in 13 42 77 123 2024; do python3 scripts/v1_dedup_importance_intervention.py $s 2; done
python3 scripts/v3_compare_and_aggregate.py
python3 scripts/v5_sample_protocol_check.py
```
