# Section 5.5: Density-Conditioned Fragility

Combines local-density stratification (Section 5.4's low/mid/high bins) with
targeted feature masking to test whether low-density regions show larger
AUC degradation and prediction-transition rates under the same stress than
high-density regions — for LR, LightGBM, and RF, plus a LightGBM-only
density-conditioned PE-inspired import-intervention probe.

## Manuscript outputs

- Figure 9: density-conditioned AUC drop under feature masking
- Table 6: density-conditioned malware-to-benign transition rates under the PE-inspired import intervention (LightGBM-only, by design)
- Appendix Figure F.1: density-conditioned flip-rate plots
- Appendix F.1 (stress-amplification half): class-conditional density-composition sensitivity

## Directory structure

```
5_5_density_conditioned_fragility/
├── configs/
│   ├── LR/                        # 3 experiment configs
│   ├── LightGBM/                   # 11 configs (gain/permutation variants + 5-seed block-3 probe)
│   └── RF/                         # 2 experiment configs
├── results/
│   ├── LR/                         # per-run output + cross-view robustness CSV
│   ├── LightGBM/                    # per-run output + block-3 probe run/merge dirs
│   ├── RF/                          # per-run output
│   ├── figures/                       # Figure 9 (10 PDFs)
│   └── manuscript_tables/              # density_import_intervention.tex (Table 6)
├── scripts/
│   ├── common/                        # run_density_conditioned_fragility{,_with_rf}.py,
│   │                                    # block-3 probe runner + merge script + parallel-run shells
│   └── figures_and_tables/             # plot_density_conditioned_fragility.py,
│                                        # build_density_import_intervention_table.py
└── sensitivity/                        # D4 class-conditional re-binning check backing Appendix F.1
    └── class_conditional/
        ├── configs/
        ├── results/
        └── scripts/
```

Each `results/<Model>/<run_name>/` directory holds that run's raw output.
Table 6 comes from a separate, LightGBM-only "block-3" density-stratified
probe (`block3_probe_lightgbm_imports_*` directories) — a 5-seed run that
pools test weeks and re-aggregates across density strata, distinct from the
main masking pipeline above it; see `results/manuscript_tables/density_import_intervention.tex`'s
own note for how it differs from Table 7's per-week-averaged protocol.

## Reproducing

Run `scripts/common/run_density_conditioned_fragility{,_with_rf}.py` against
the matching `configs/<Model>/*.json`, then
`scripts/figures_and_tables/plot_density_conditioned_fragility.py`. For
Table 6, run `run_density_conditioned_import_intervention_probe.py` (5 seeds)
then `merge_density_conditioned_import_intervention_probe_results.py`, then
`build_density_import_intervention_table.py`. For the Appendix F.1
sensitivity check, see `sensitivity/class_conditional/`.
