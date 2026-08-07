# Section 5.4: Density-Conditioned Reliability

Stratifies test samples into low/mid/high local-density bins (by k-NN
distance to the training set, k=10) and measures baseline (pre-perturbation)
AUC and Mix@10 within each bin, for LR, LightGBM, and RF — testing whether
sparse (low-density) regions are less reliable than dense regions before any
targeted stress is applied.

## Manuscript outputs

- Figure 8: density-conditioned AUC and Mix@10 across feature subsets under Wild (B)
- Table 5: disagreement gap between dense and sparse regions
- Appendix F.1 (reliability half): class-conditional density-composition sensitivity

## Directory structure

```
5_4_density_conditioned_reliability/
├── configs/
│   ├── LR/                     # 3 experiment configs
│   ├── LightGBM/                # 3 experiment configs
│   └── RF/                      # 2 experiment configs
├── results/
│   ├── LR/                      # per-run output + rank-disagreement CSV
│   ├── LightGBM/                 # per-run output + rank-disagreement CSV
│   ├── RF/                       # per-run output + rank-disagreement CSV
│   ├── figures/                    # Figure 8 (6 PDFs)
│   └── manuscript_tables/           # density_conditioned_rank_disagreement.tex (Table 5)
├── scripts/
│   ├── common/                     # run_density_stratified_reliability{,_with_rf}.py
│   └── figures_and_tables/          # plot_density_stratified_reliability.py,
│                                     # build_density_conditioned_rank_disagreement_table.py
└── sensitivity/                     # D3 class-conditional re-binning check backing Appendix F.1
    └── class_conditional/
        ├── configs/
        ├── results/
        └── scripts/
```

Each `results/<Model>/<run_name>/` directory holds that run's raw output
(`results.json` plus a `k_10/` subfolder with per-density-bin CSVs). Only the
Wild (B) view is reported in the manuscript, so only Wild (B) results are
retained.

## Reproducing

Run `scripts/common/run_density_stratified_reliability{,_with_rf}.py`
against the matching `configs/<Model>/*.json`, then
`scripts/figures_and_tables/plot_density_stratified_reliability.py` and
`build_density_conditioned_rank_disagreement_table.py`. For the Appendix F.1
sensitivity check, see `sensitivity/class_conditional/`.
