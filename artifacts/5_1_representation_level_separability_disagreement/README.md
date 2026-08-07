# Section 5.1: Representation-Level Separability Disagreement

Audits five EMBER-derived feature representations (`all`, `header`, `section`,
`imports`, `strings`) with decision-level (AUC) and structure-level (Mix@10,
JS divergence, same-family rate@10) measurements across the Wild (U), Wild
(B), and Unpacked (B) evaluation views, for LR, LightGBM, and RF.

## Manuscript outputs

- Figure 2: feature-subset AUC across Wild (U), Wild (B), and Unpacked (B)
- Figure 3: AUC-JS-Mix mismatch scatter across feature subsets
- Figure 4: Macro-F1 vs same-family-rate@10 under Wild (B) (`classification_vs_structure_scatter_wild_b_{lr,lightgbm,rf}.pdf`,
  built by `scripts/figures_and_tables/plot_taxonomy_aligned_comparison.py`; reads
  `results/{LR,LightGBM}/family_aligned_rq1_matched{,_unpacked}/` and
  `results/RF/classification_vs_structure_{wild_b,unpacked_b}/`, all already bundled here)
- Appendix Figure C.1: UMAP feature-space geometry under Wild (B), one panel
  per feature group (generating script and output PDFs live under
  `Appendix/Section_C/` — see that section's README; the source
  `results.json`/summary CSVs the script reads stay here)
- Table 3: LR-only measurement summary across feature subsets, referenced in
  both Section 5.1 and the Appendix
- Appendix Table A.1: malware-family composition under Wild (U) and Unpacked (B)
  (generating script and output live under `Appendix/Section_A/` — see that
  section's README; the source CSV the script reads stays here)

## Directory structure

```
5_1_representation_level_separability_disagreement/
├── configs/
│   ├── LR/                  # 5 experiment configs (family-aligned + win32 baselines)
│   ├── LightGBM/             # 3 experiment configs
│   └── RF/                   # 5 experiment configs
├── results/
│   ├── LR/                   # per-run output dirs + summary CSVs (see below)
│   ├── LightGBM/              # per-run output dirs
│   ├── RF/                    # per-run output dirs
│   ├── figures/                # Figure 2-4 (9 PDFs)
│   └── manuscript_tables/       # measurement_summary.tex (Table 3)
└── scripts/
    ├── common/                 # experiment runners (`_with_rf.py` variants natively
    │                             handle RF too, via `model_type`/`--model`)
    └── figures_and_tables/      # figure/table builders, driven by --output/--input CLI args
```

Each `results/<Model>/<run_name>/` directory holds that run's raw output:
`results.json` plus aggregate CSVs (`aggregate_by_seed.csv`,
`aggregate_by_mix_k.csv`, `weekly_results.csv`, etc.). A few summary CSVs sit
directly under `results/LR/` (e.g. `table_rq1_measurement_summary.csv`,
`table_family_composition_original_vs_unpacked.csv`) — these feed the
`manuscript_tables/` exports.

## Reproducing

Run the scripts under `scripts/common/` with the matching `configs/<Model>/*.json`,
then `scripts/figures_and_tables/*.py` against the resulting `results/<Model>/`
output (each script takes `--output`/`--input`-style CLI args; run with `--help`
for the exact flags). See the top-level `artifacts/README.md` for how each
script's output maps to a specific figure/table.
