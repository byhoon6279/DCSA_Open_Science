# Section 5.2: Dimensionality-Matched Subset Controls

Tests whether the representation-level differences observed in Section 5.1
survive against same-dimensional random-subspace controls — i.e., whether a
semantic feature subset (e.g. `header`) behaves differently from an
arbitrary, equal-size random subset of the full representation, at matched
dimensionality `d`.

## Manuscript outputs

- Table 4: dimensionality-matched subset controls at d=32
- Appendix Table D.2: AUC departures from same-dimensional random-subspace controls
- Appendix Table D.3: structure-level departures from same-dimensional random-subspace controls

## Directory structure

```
5_2_dimensionality_matched_subset_controls/
├── configs/
│   ├── LR/                          # 3 experiment configs
│   └── LightGBM/                     # 3 experiment configs
├── results/
│   ├── LR/                           # per-run output dirs (see below)
│   ├── LightGBM/                      # per-run output dirs
│   └── manuscript_tables/              # Table 4, Appendix Table D.2/D.3 exports
├── dimension_matched_subset_audit.py    # runs the family-aligned (semantic) subset experiment
├── random_subspace_control.py           # runs the matched random-subspace null
├── postprocess_random_subspace_results.py  # builds family-vs-null paired summaries
├── summarize_dimension_controls.py      # secondary summary/reporting helper
├── common_library_adapter.py            # adapter onto artifacts/shared/common.py
├── shared.py                            # shared config/sampling/CI helpers for this section
├── run_lr_wild_b_smoke.py               # small LR smoke-test entry point
└── build_manuscript_tables.py           # builds Table 4 + Appendix Table D.2/D.3
```

Not reported for this subsection: RF (no `configs/RF/` or `results/RF/`), and
no `results/figures/` (no figures reported).

Under `results/{LR,LightGBM}/` there are two run directories per model:
- `wild_b_ra_q4_main/`: the family-aligned (semantic) subset runs
- `wild_b_ra_q4_random_null/`: the matched random-subspace null runs, whose
  `family_vs_null_paired_summary.csv` is what `build_manuscript_tables.py`
  actually reads

The existing `ra_q4` directory names are retained for compatibility with the
bundled configurations and generated outputs.

## Reproducing

```bash
python3 dimension_matched_subset_audit.py --config <configs/LR/*.json> --output-dir <results/LR/...>
python3 random_subspace_control.py --config <configs/LR/*.json> --family-seed-summary <...> --output-dir <...>
python3 postprocess_random_subspace_results.py --family-seed-summary <...> --null-seed-summary <...> --output-dir <...>
python3 build_manuscript_tables.py
```

These scripts are plain, self-contained scripts (no package-relative
imports) — run them directly from this directory with `--help` for the full
argument list.
