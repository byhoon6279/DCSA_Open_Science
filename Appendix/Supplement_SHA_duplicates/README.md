# Supplement: SHA-256 duplicate records and validation-set overlap

Self-contained supplementary analyses, run separately from the main pipeline.
No result under `artifacts/` was changed (only the Figure 3 legend text in
`artifacts/5_1.../plot_representation_mismatch.py` and its three PDFs). All script paths are relative to the repository root (via `Path(__file__)`). Raw EMBER 2024 files are
expected under `Data/` (not bundled); feature caches are written to
`supplementary_val_dedup/cache/` and `ch5_dup_sensitivity/cache/`.

| Folder | What it shows |
|---|---|
| `ch5_dup_sensitivity/` | Raw EMBER 2024 test weeks store every SHA-256 twice (e.g. 2024-09-22_2024-09-28 Win32 test: 60,000 rows = 30,000 unique files). Mix@10 / Silhouette / density-bin sensitivity with and without duplicate rows, plus tie-averaged Mix@k (`results/comparison.txt`, `s51_*.csv`, `s54_*.csv`). |
| `supplementary_val_dedup/` | LightGBM permutation-importance validation split overlapped the training split by 84-99 SHA-256 per seed. Re-derived importance on a cleaned validation set, same model / test sample (`results/step1_summary.json`, `final_comparison.json`, `step3_pe_intervention_rows.json`, `sha_lists/`). |
| `rerun_check/` | `step3_fixed_seed_rows.json`: 60 seed-week cells for the 10% hashed-import intervention. Important-arm M->B 0.3081 (original, = Table 5) vs 0.1604 (cleaned validation); random arm 0.0279 (identical to Table 5). |
| `plot_representation_mismatch_mean_mix_k.py` | Earlier Figure 3 variant whose legend states "mean Mix@k (k=5,10,15,20)". `artifacts/5_1.../plot_representation_mismatch.py` now carries the same legend text (the bubble size is the k-average `aggregate_metrics[...]["mix_at_k"]`), so this copy is redundant. |

Notes
- The cleaned-validation values are a sensitivity analysis. They were not run through the
  other LightGBM analyses (Fig. 5-7, Tables D.1/E.x, Fig. 9 / Table 7) or RF.
- A size-matched control (removing an equal number of nonduplicate validation records) and a
  permutation-seed change were also run (manuscript §5.3). Both substantially altered the selected
  import dimensions, so the 0.308 -> 0.160 difference cannot be attributed uniquely to duplicate
  removal; these checks bound, but do not isolate, the effect.
