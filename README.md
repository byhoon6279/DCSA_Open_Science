# Open Science Package

This repository accompanies the manuscript *"DCSA: A Multi-View Reliability Audit for
Static Malware Detectors,"* prepared for submission to the Journal of Information Security
and Applications. It provides the code, configurations, intermediate results, and exported
figures and tables supporting the quantitative analyses reported in the manuscript.

## Layout

`artifacts/` is organized by manuscript subsection. See `artifacts/README.md` for
the full paper-to-artifact map (which
figure/table comes from which directory) and each `artifacts/5_1`-`5_5` section's own
`README.md` for that section's entry-point scripts.

```
artifacts/
  shared/                                                # common.py: shared feature/sample-loading library used by every section below
  5_1_representation_level_separability_disagreement/   # Section 5.1 / Fig. 2-4, App. Fig. C.1
  5_2_dimensionality_matched_subset_controls/            # Section 5.2, "Dimensionality-Matched Subset Controls" / Table 4, App. Table D.2-D.3
  5_3_targeted_perturbation_response/                    # Section 5.3 / Fig. 5-7, Table 7, App. Fig. D.1, App. Table D.1
  5_4_density_conditioned_reliability/                   # Section 5.4 / Fig. 8, Table 5
  5_5_density_conditioned_fragility/                     # Section 5.5 / Fig. 9, Table 6, App. Fig. F.1
  MLP/                                                   # Section 5.6, "Generalization to a Neural Classifier" / Table 8, App. Table G.1-G.4
```
Appendix Tables E.1-E.2 (paired significance of masking effects) span LR,
LightGBM, RF, and MLP together, not either section above; see `Appendix/Section_E/`.

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
