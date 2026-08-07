# Section 5.3: Targeted Perturbation Response

Masks 1%/5%/10% of the `header` and `imports` feature dimensions — comparing
importance-ranked ("important") masking against random masking — and
measures the resulting AUC degradation, prediction-flip rate, and change in
Mix@10, for LR, LightGBM, and RF. Also covers the PE-inspired hashed
import-feature intervention (Table 7) and its more detailed
per-week/per-model breakdown.

**On "AUC degradation" terminology:** this repository reports two distinct
quantities under that name, and the manuscript is careful to distinguish them:
- Figure 5 and Appendix Figure D.1 plot **normalized** AUC degradation —
  `(AUC_baseline - AUC_masked) / (AUC_baseline - 0.5)`, i.e. the raw AUC-point
  drop normalized by the available margin above chance (`AUC = 0.5`) —
  computed in `aggregate_setting_rows()` / `aggregate_auc_rows()` in
  `scripts/figures_and_tables/plot_decision_level_masking_collapse{,_v2}.py`
  and `plot_rf_wild_b_masking_figures.py`.
- The Appendix E significance tables (`../../Appendix/Section_E/`) and
  Appendix G's MLP table report **raw AUC-point degradation**
  (`AUC_baseline - AUC_masked`, unnormalized).

**Figure 5 error bars:** Figure 5's three panels (`plot_decision_level_masking_collapse_v2.py`
for LR/LightGBM, `plot_rf_wild_b_masking_figures.py` for RF) draw 95% CI error
bars on every mean point. For each (model, feature group, strength,
important/random arm), values are averaged over the 12 test weeks within each
seed (random repeats averaged within each seed-week first), giving 5
seed-level values; the plotted point is their mean, and the error bar is a
percentile bootstrap (10,000 resamples, fixed seed) on that same 5-value
list — point and CI always share the same unrounded seed-level data, so the
bootstrap bound can never fall on the wrong side of the plotted point. These
are **arm-level** CIs (uncertainty in the important mean and the random mean
separately) — not the important-vs-random paired-difference CI, which is a
different quantity reported in Appendix Table E.1/E.2 (`../../Appendix/Section_E/`).
The corresponding Figure 5 caption and Section 5.3 description in the
manuscript reflect this aggregation and uncertainty-quantification
procedure.

## Manuscript outputs

- Figure 5: decision-level AUC degradation under targeted masking
- Figure 6: two-axis decision/structure response under targeted masking
- Figure 7: prediction-level instability under targeted masking
- Table 7: prediction instability under PE-inspired import interventions
- Appendix Figure D.1: decision-level collapse under Wild (U) and Unpacked (B)
- Appendix Table D.1: sensitivity of ΔMix@k to neighborhood size
- Appendix Table E.1: statistical significance of the important-vs-random masking gap

## Directory structure

```
5_3_targeted_perturbation_response/
├── configs/
│   ├── LR/                       # 6 experiment configs (feature-perturbation + prediction-flip)
│   ├── LightGBM/                  # 6 experiment configs
│   └── RF/                        # 6 experiment configs
├── results/
│   ├── LR/                        # per-run output dirs + summary CSVs
│   ├── LightGBM/                   # per-run output dirs (gain/permutation importance variants)
│   ├── RF/                         # per-run output dirs
│   ├── figures/                     # Figure 5-7, Appendix Figure D.1 (17 PDFs)
│   └── manuscript_tables/            # Table 7, Appendix Table D.1 exports
├── scripts/
│   ├── common/                      # experiment runners (`_with_rf.py` variants handle RF too)
│   ├── RF/                          # run_pe_intervention_rf.sh launcher
│   └── figures_and_tables/           # figure/table builders
└── pe_inspired_feature_intervention/   # PE-inspired hashed import-feature intervention assets
    ├── configs/
    ├── notes/
    ├── results/
    │   ├── experiment_results/         # per-model/per-view raw run output
    │   └── manuscript_tables/
    └── scripts/                         # run_pe_feature_intervention_validation{,_with_rf}.py, etc.
```

Each `results/<Model>/<run_name>/` directory holds that run's raw output
(`results.json`, `aggregate_results.csv`, and for masking runs a
per-strength `k_*/` subfolder). LightGBM masking runs come in `_gain` and
`_permutation` importance-method variants (the manuscript uses permutation
for the main-text figures).

## Reproducing

Run `scripts/common/run_feature_perturbation_sensitivity{,_with_rf}.py` and
`run_prediction_flip_analysis{,_with_rf}.py` against the matching
`configs/<Model>/*.json`, then the corresponding
`scripts/figures_and_tables/*.py` builder. The PE-inspired intervention uses
its own `pe_inspired_feature_intervention/scripts/run_pe_feature_intervention_validation{,_with_rf}.py`.
See the top-level `artifacts/README.md` for the full figure/table provenance notes.
