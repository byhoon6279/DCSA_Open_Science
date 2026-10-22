# Canonical family-aligned evaluation (Figure 2)

Re-evaluation of the family-aligned analysis behind Figure 2 on one shared sample.
Existing outputs under `../results/` are unchanged; this folder only adds files.

## What was run
- Views: `wild_b` (all family-labelled malware) and `unpacked_b` (unpacked malware only).
- Per view, seed (13, 42, 77, 123, 2024) and test week (12 weeks): train/test records are selected once and used by LR, LightGBM and RF.
- Preprocessing: `StandardScaler` fitted on the selected raw training features (no NaN replacement, clipping or signed-log), as in the original family-aligned runners.
- Same-family rate@k is computed once per cell from the shared transformed test matrix; the query is excluded by row index. Figure 2 uses k=10 only.
- Family-classification Macro-F1 is computed per model on exactly that train/test sample.
- 60 seed-week cells per view and group; 20 families per cell; 10,000 training records per cell.

## Contents
- `code/run_canonical_family_aligned.py`, `code/canonical_config.json`: the runner and config that produced `results/` (copied unmodified).
- `code/common_rq1_dependency.py`: copy of `overlap_separability/RQ1/common.py` loaded by the runner (byte-identical to `../../shared/common.py`).
- `code/summarize_and_plot.py`: summarizer shipped with the run (not used for the files below).
- `code/compare_old_new_and_plot.py`: writes the before/after tables and the k=10 figures.
- `results/{wild_b,unpacked_b}/`: `weekly_metrics.csv`, `aggregate_metrics.csv`, `selected_records.csv.gz` (ordered record identifiers), `input_file_manifest.csv` (SHA-256 of each input file), `matrix_audit.csv` (matrix and scaler hashes), `run_metadata.json`, `scalers_sha256.txt` (scaler arrays themselves are not bundled).
- `results/resolved_config.json`: resolved config and runtime versions (Python 3.13.12, numpy 2.4.6, scikit-learn 1.9.0, lightgbm 4.7.0).
- `results/before_after_comparison.csv`, `results/cell_composition_comparison.csv`, `results/figures/`.
- `SHA256SUMS`: hashes of every file in this folder.

## Reproduce
Edit `data_root`, `common_module`, `protocol_config` and `output_dir` in `code/canonical_config.json` to local paths (the bundled copy keeps the original absolute paths), then:
```bash
python code/run_canonical_family_aligned.py --config code/canonical_config.json
python code/compare_old_new_and_plot.py   # expects results/ in this layout
```
The protocol config is `../configs/LR/family_aligned_rq1_matched.json` (the original run read the same file from another checkout; only `data_root` differs). EMBER 2024 weekly files are not bundled; their SHA-256 values are in `input_file_manifest.csv`.

## Provenance
The runner and config files were last modified before the run started (14:43 vs. 14:46 local time, 2026-10-22) and their hashes were unchanged after the run. The code that was executing is not otherwise proven; this folder preserves the current files.

## Scope
Covers only the Figure 2 family-aligned evaluation. It does not address the LightGBM Unpacked binary-AUC issue or the self-exclusion rule in other Mix@k / same-family analyses (e.g. the dimensionality-matched controls).
