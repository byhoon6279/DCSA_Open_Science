# Open Science Package

This repository accompanies the manuscript *"DCSA: A Multi-View Reliability Audit for
Static Malware Detectors,"* prepared for submission to the International Journal of
Information Security. It provides the code, configurations, intermediate results, and exported
figures and tables supporting the quantitative analyses reported in the manuscript.

## Layout

`artifacts/` is organized by manuscript subsection. See `artifacts/README.md` for
the full paper-to-artifact map (which
figure/table comes from which directory) and each `artifacts/5_1`-`5_5` section's own
`README.md` for that section's entry-point scripts.

```
artifacts/
  shared/                                                # common.py: shared feature/sample-loading library used by every section below
  5_1_representation_level_separability_disagreement/   # Section 5.1 / Table 1, Table 3, Fig. 2-4, Fig. 10
  5_2_dimensionality_matched_subset_controls/            # Section 5.2, "Dimensionality-Matched Subset Controls" / Table 4, Table 12-13
  5_3_targeted_perturbation_response/                    # Section 5.3 / Fig. 5-7, Table 5, Fig. 11, Table 11
  5_4_density_conditioned_reliability/                   # Section 5.4 / Fig. 8, Table 6, Table 16
  5_5_density_conditioned_fragility/                     # Section 5.5 / Fig. 9, Table 7, Fig. 12, Table 17
  MLP/                                                   # Section 5.6, "Extension to a Neural Classifier" / Table 8, Table 18-21
```
Tables 14-15 (paired significance of masking effects) span LR, LightGBM, RF, and MLP
together, not either section above; see `Appendix/Section_E/`.

The `5_1`, `5_3`, `5_4`, `5_5` section directories follow the same internal convention:
`configs/`, `scripts/`, `data/`, `results/`, `figures/`, split by model family (`LR/`,
`LightGBM/`, `RF/`) where applicable. `5_2_dimensionality_matched_subset_controls/` and
`MLP/` are smaller, self-contained result sets without the full bucket structure. `5_4`
and `5_5` each additionally have a `sensitivity/` subfolder for a class-conditional
re-binning check backing Appendix F.1 (see below).

```
Appendix/
  Section_A/   # Appendix Table A.1 (malware-family composition)
               # — source CSV comes from artifacts/5_1.../, not here
  Section_C/   # Appendix Figure C.1 (UMAP feature-space geometry)
               # — source results.json/summary CSVs come from artifacts/5_1.../, not here
  Section_D/   # Appendix D.1-D.3 (k-value sensitivity, Wild(U)/Unpacked(B) decision collapse,
               # dimensionality-matched subset controls) — some generating scripts live here,
               # some (Table D.2/D.3, the structural-response-map figure) stay in
               # artifacts/5_2.../ and artifacts/5_3.../ since they also produce main-text
               # content; only their appendix-facing output is copied here
  Section_E/   # Appendix E tables (Paired Statistical Analysis of Targeted Stress and Density Effects)
  Section_F/   # Appendix F.1 tables (Class-Conditional Density-Composition Sensitivity)
               # — reproduction code/configs/results live in artifacts/5_4.../sensitivity/
               #   and artifacts/5_5.../sensitivity/, not here
  Section_G/   # Appendix G tables (Neural-Classifier Extension full results)
```

The `Appendix/` directory contains scripts and derived outputs supporting the
corresponding manuscript appendices where applicable.

```
Appendix/Supplement_SHA_duplicates/   # SHA-256 duplicate-record and validation-overlap sensitivity checks (see below)
validation/                           # re-execution of the LightGBM hashed-import validation protocol (see below)
```

### Numbering used in the manuscript

The manuscript numbers appendix tables and figures continuously (Table 10-21, Figure
10-12) after the main text. Some section READMEs and `.tex` labels under `artifacts/`
and `Appendix/` still use appendix-letter names (e.g. "Table D.2"). They map as follows
(the "Manuscript section" column gives where the item sits in the current manuscript):

| Manuscript | Appendix-letter name | Manuscript section | Content | Location |
|---|---|---|---|---|
| Table 10 | A.1 | A | Malware-family composition, Wild (U) / Unpacked (B) | `Appendix/Section_A/` |
| Figure 10 | C.1 | C.1 | UMAP projections of four feature representations | `Appendix/Section_C/` |
| Table 11 | D.1 | D.1 | Sensitivity of ΔMix@k to neighborhood size | `Appendix/Section_D/`, `artifacts/5_3.../` |
| Table 12 | D.2 | D.2 | Subset-minus-random AUC differences | `artifacts/5_2.../`, `Appendix/Section_D/` |
| Table 13 | D.3 | D.2 | Structure-level departures from random subspaces | `artifacts/5_2.../`, `Appendix/Section_D/` |
| Figure 11 | D.1 (figure) | D.3 | Masking collapse, Wild (U) / Unpacked (B) | `Appendix/Section_D/`, `artifacts/5_3.../` |
| Table 14 | E.1 | E | Paired AUC-degradation tests | `Appendix/Section_E/` |
| Table 15 | E.2 | E | Paired flip-rate tests | `Appendix/Section_E/` |
| Figure 12 | F.1 (figure) | F.1 | Density-conditioned flip rates | `Appendix/Section_F/`, `artifacts/5_5.../` |
| Table 16 | F.1 (reliability) | F.1 | Class-conditional reliability sensitivity | `Appendix/Section_F/` |
| Table 17 | F.1 (masking) | F.1 | Class-conditional masking sensitivity | `Appendix/Section_F/` |
| Table 18 | G.1 | G.2 | MLP representation-level results | `Appendix/Section_G/` |
| Table 19 | G.2 | G.3 | MLP important/random masking | `Appendix/Section_G/` |
| Table 20 | G.3 | G.4 | MLP density-conditioned reliability | `Appendix/Section_G/` |
| Table 21 | G.4 | G.5 | MLP density-conditioned masking amplification | `Appendix/Section_G/` |

Main-text numbers (Table 1-9, Figure 1-9) are the same everywhere.

### SHA-256 duplicates and validation overlap

EMBER 2024 stores every SHA-256 twice within its weekly file, so the manuscript's counts
(Table 1) are dataset records, not unique files. Two folders document the consequences
referred to in Sections 4.5, 5.3, and 6.4 and Appendix F of the manuscript:

- `validation/` re-executes the LightGBM hashed-import validation protocol. The 10%
  important-feature M→B rate of 0.308 (Table 5) uses a label-stratified validation split
  without SHA-group control, in which 84-99 validation rows per seed share a SHA-256 with
  a training row. Re-deriving importance on a cleaned validation set gives 0.160 (random
  control unchanged at ≈0.028). The manuscript treats this difference as not attributable
  uniquely to duplicate removal, because changing the permutation seed or the
  validation-set size also changes the selected dimensions.
- `Appendix/Supplement_SHA_duplicates/` holds the earlier sensitivity runs: Mix@k,
  silhouette and density-bin results with and without duplicate rows (main
  feature-representation ordering preserved), and the cleaned-validation intervention.

These are supplementary sensitivity analyses. They are not part of the generating pipeline
for any main-text figure or table, and the `artifacts/` results are unchanged by them. Their
scripts resolve all paths relative to the repository root and expect the raw EMBER 2024
files under `Data/` (not bundled); the ~1.1 GB feature caches are rebuilt locally.

## Quick Start

There are two different things you might want to do with this package, and they have
very different requirements:

**A. Regenerate tables and figures from the bundled intermediate results.**
Most reported outputs can be regenerated without the raw EMBER2024 dataset.
Analyses that reconstruct feature-level representations, including Appendix
Figure C.1, additionally require the EMBER2024 feature files described under
"Dataset" below. For example:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r artifacts/requirements.txt

python Appendix/Section_E/scripts/export_paired_masking_reanalysis.py
python Appendix/Section_G/scripts/build_mlp_masking_full.py
```

The relevant entry points for regenerating each reported output are documented
in the corresponding section README.

**B. Re-run an experiment from scratch against the raw EMBER2024 features.** This
requires downloading the full dataset first — see "Dataset" below — and is
computationally intensive. For example, the MLP pipeline:

```bash
python artifacts/MLP/scripts/launch_mlp_dcsa_core.py --device cpu --workers 1
```

Each section's `scripts/common/` (or top-level `scripts/`) holds its experiment
runners; each takes a `--config` pointing at one of that section's `configs/*.json`.

## Dataset

This package does not bundle the raw EMBER2024 dataset itself (~97GB). It includes
pre-computed intermediate results and exported figures/tables for most reported
analyses. Re-running experiments from scratch and regenerating analyses that require
feature-level reconstruction, including Appendix Figure C.1, requires the raw
feature files:

- Dataset/toolkit: <https://github.com/futurecomputing4ai/ember2024>
- Expected size: ~97GB
- File format: weekly `.jsonl` files, named `{week}_{platform}_{split}.jsonl` (e.g.
  `2023-09-24_2023-09-30_Win32_train.jsonl`)
- Placement: directly under `DCSA_Open_Science/Data/`, with no `dataset/features`
  subfolder (i.e. `Data/2023-09-24_2023-09-30_Win32_train.jsonl`, not
  `Data/dataset/features/2023-09-24_2023-09-30_Win32_train.jsonl`)

## Environment

`artifacts/requirements.txt` pins the Python package versions (Python 3.10) used to
produce the results in this package:

```
pip install -r artifacts/requirements.txt
```

`thrember` (the EMBER2024 feature extractor used upstream to produce the raw feature
files consumed by these scripts) is not on PyPI and is not required to run anything in
`artifacts/` — it is only needed if you want to re-extract features from raw PE files.
See the dataset/toolkit link under "Dataset" above if needed.

The experiment pipelines reuse the shared sample- and feature-loading helpers in
`artifacts/shared/common.py` (feature vectorization, `week_paths`, `balance_samples`,
etc.), either directly or through section-specific adapters. Scripts that import it
directly resolve the path themselves via `Path(__file__).resolve().parents[N]`, so no
`PYTHONPATH`/install step is required — just keep the package's directory structure
intact.

All runnable configs (`configs/**/*.json`) use paths relative to their own location, so
they work regardless of where this package is cloned.

Some frozen result records (`results.json` / `results_summary.json` /
`resolved_config.json`) retain redacted absolute `data_root` fields (e.g.
`/redacted-local-path/...`) from the original execution environment. These
fields are provenance metadata only and are not used by the table or figure
regeneration scripts.
