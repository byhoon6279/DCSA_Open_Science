# Appendix Section E: Statistical Significance Tables

Backing scripts and outputs for the manuscript's Appendix Table E.1
(AUC-degradation significance) and Table E.2 (prediction-flip-rate
significance).

## Table E.1 — `significance_masking_auc.tex` and Table E.2 — `significance_masking_flip.tex`

Both are a paired seed-level reanalysis (`scripts/export_paired_masking_reanalysis.py`),
spanning LR, LightGBM, RF, and the MLP extension. For each model-feature-group cell,
the table reports the masking ratio (1%, 5%, or 10%) with the largest — i.e.,
least significant — raw sign-flip $p$-value among the three tested ratios, with a
Holm-adjusted $p$-value from the full ratio-level correction family (36 tests for
LR/LightGBM/RF, 30 for MLP). Positive mean differences indicate greater degradation
(E.1) or more flips (E.2) under important masking than random masking. The 95% CI is a percentile bootstrap (10,000 resamples, fixed seed) on the five
seed-level paired differences. Because the bootstrap CI quantifies uncertainty
in the mean paired difference whereas the sign-flip $p$-value evaluates
directional consistency across seeds, the two procedures need not yield
identical conclusions for a given cell.

Table E.1's `Mean difference` column reports **raw AUC-point degradation**
(`AUC_baseline - AUC_masked`, unnormalized) — this is a different quantity from
the normalized AUC degradation plotted in the main text's Figure 5 and Appendix
Figure D.1; see `../../artifacts/5_3_targeted_perturbation_response/README.md`
for how the two are related.

Source data: `artifacts/5_3_targeted_perturbation_response/results/` and
`artifacts/MLP/results/5_3/` (already bundled elsewhere in this package; no
new data added).

## Reproducing

```bash
python scripts/export_paired_masking_reanalysis.py
```

writes `results/paired_reanalysis/` (full per-configuration rows, worst-case
summary, input-availability audit, and a narrative report) and
`results/manuscript_tables/significance_masking_auc.tex` (Table E.1) /
`significance_masking_flip.tex` (Table E.2), matching the manuscript exactly.

## Layout

- `scripts/export_paired_masking_reanalysis.py`: generates everything below
- `results/paired_reanalysis/`: full per-configuration output
  (`paired_masking_reanalysis_rows.csv`, `_summary.csv`, `_availability.csv`,
  `paired_masking_reanalysis.tex`, `paired_mlp_masking_reanalysis.tex`,
  `paired_masking_reanalysis_report.md`)
- `results/manuscript_tables/`: `significance_masking_auc.tex` (Table E.1) and
  `significance_masking_flip.tex` (Table E.2) — the current, final content
  matching the manuscript repository's tables
