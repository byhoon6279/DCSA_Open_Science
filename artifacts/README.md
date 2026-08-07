# artifacts

Paper-structured artifact workspace for the journal manuscript.

The top-level layout follows the manuscript flow:
- `5_1_representation_level_separability_disagreement/`
- `5_2_dimensionality_matched_subset_controls/`
- `5_3_targeted_perturbation_response/`
- `5_4_density_conditioned_reliability/`
- `5_5_density_conditioned_fragility/`
- `MLP/`

Conventions:
- each main section keeps `configs/`, `scripts/`, and `results/` when available. `5_1`
  through `5_4` follow the reference layout: figures under `results/figures/`, and
  `scripts/` split into `common/` (multi-model runners), `<Model>/` (model-exclusive
  entry points), and `figures_and_tables/` (plotting/table builders).
  `5_2_dimensionality_matched_subset_controls` has no `figures/` (no figures reported
  for this subsection) and its `configs/`/`results/` are split into `LR/`/`LightGBM/`
  the same way; its `.py` files stay flat at the directory's own top level rather than
  moving into a `scripts/` subfolder.
- `LR/`, `LightGBM/`, and `RF/` buckets are merged under the same section so the paper
  narrative and model comparisons stay aligned
- `5_3_targeted_perturbation_response/pe_inspired_feature_intervention/` contains the
  additional PE-inspired intervention assets used by the paper and appendix
- figure/table numbers below match the current manuscript numbering (main text: Figure
  1-9, Table 1-9; appendix: lettered per appendix section, e.g. Figure C.1, Table D.2)

## Paper-to-artifact map

### `5_1_representation_level_separability_disagreement/`
Covers the main-paper representation audit in Section 5.1.

Relevant manuscript outputs:
- Figure 2: feature-subset AUC across Wild (U), Wild (B), and Unpacked (B)
- Figure 3: AUC-JS-Mix mismatch scatter across feature subsets
- Figure 4: Macro-F1 vs same-family-rate@10 under Wild (B)
- Appendix Figure C.1: UMAP feature-space geometry under Wild (B) (generating script and
  output PDFs live under `Appendix/Section_C/`, not here)
- Table 3: LR-only measurement summary across feature subsets, referenced in both Section 5.1
  and the Appendix
- Appendix Table A.1: malware-family composition under Wild (U) and Unpacked (B) (generating
  script and output live under `Appendix/Section_A/`, not here)

Where to look:
- `results/figures/`: exported manuscript figures used for Section 5.1
- `results/manuscript_tables/`: table exports, including `measurement_summary.tex`
  (Table 3, the manuscript's LR-only table, built by
  `scripts/figures_and_tables/build_measurement_summary_lr_table.py` from
  `results/LR/table_rq1_measurement_summary.csv`)
- `results/LR`, `results/LightGBM`, `results/RF`: per-model experiment run output
- `scripts/common/`: experiment runners shared across model families (the
  `_with_rf.py` files natively handle RF too, via `model_type`/`--model`)
- `scripts/figures_and_tables/`: figure/table builders, invoked per model via CLI args

### `5_2_dimensionality_matched_subset_controls/`
Covers Section 5.2, the "Dimensionality-Matched Subset Controls" subsection, which
follows directly after Section 5.1 in the main text. Tests whether the representation-level
signal survives against same-dimensional random-subspace controls. Appendix grouping
places its detailed appendix tables under Appendix Section D, alongside the
neighborhood-size sensitivity check.

Relevant manuscript outputs:
- Table 4: dimensionality-matched subset controls at d=32
- Appendix Table D.2: AUC departures from same-dimensional random-subspace controls
- Appendix Table D.3: structure-level departures from same-dimensional random-subspace controls

Where to look:
- `results/manuscript_tables/`: exported Table 4 and Appendix Table D.2/D.3 sources,
  built by `build_manuscript_tables.py` (top-level of this section) from
  `results/{LR,LightGBM}/wild_b_ra_q4_random_null/family_vs_null_paired_summary.csv`
  (per model/subset/dimension/metric mean paired difference between the
  family-aligned subspace and the matched random-subspace null, with its 95% CI).
  AUC is model-dependent (LR vs. LightGBM columns in D.2); `mix_at_10`,
  `js_divergence`, and `same_family_rate_at_10` are structure-level metrics
  computed from the same feature subsets independent of which classifier scores
  them, so LR's and LightGBM's rows for those three are identical — D.3 reads
  them from the LR file only. Table 4 is the d=32 slice of D.2+D.3, dropping the
  JS column and the CIs on mix_at_10/same-family.
- `results/LR`, `results/LightGBM`: per-model experiment run output (no RF; not
  reported for this subsection)
- `configs/LR`, `configs/LightGBM`: experiment configs

### `5_3_targeted_perturbation_response/`
Covers the targeted masking results in Section 5.3.

Relevant manuscript outputs:
- Figure 5: decision-level AUC degradation under targeted masking
- Figure 6: two-axis decision/structure response under targeted masking
- Figure 7: prediction-level instability under targeted masking
- Table 7: prediction instability under PE-inspired import interventions
- Appendix Figure D.1: decision-level collapse under Wild (U) and Unpacked (B) (generating
  script and Wild (U)/Unpacked (B) decision-collapse panels live under
  `Appendix/Section_D/`, not here; the structural-response-map
  companion panels are built here, see below, and copied there too)
- Appendix Table D.1: sensitivity of Delta Mix@k to neighborhood size (generating script and
  output live under `Appendix/Section_D/`, not here)
- Appendix Tables E.1-E.2: paired significance of the important-vs-random masking gap
  for AUC degradation (E.1) and prediction flip rate (E.2), spanning LR, LightGBM, RF,
  and MLP together (generating script and output live under `Appendix/Section_E/`, not
  here)

Where to look:
- `results/figures/`: final LR/LightGBM/RF PDFs used in the paper body for Figures 5-7,
  plus the Appendix Figure D.1 Wild (U)/Unpacked (B) structural-response-map panels
  (`app_structural_response_map_{lr,lightgbm}_{wild_u,unpacked_b}.pdf`), built by
  `scripts/figures_and_tables/plot_structural_response_map.py`. This script also builds
  main-text Figure 6 in the same run, so it stays here rather than moving to
  `Appendix/Section_D/`; its four appendix-facing panels are copied there.
- `results/manuscript_tables/`: exported masking summary tables. `k_sensitivity_wild_b_table.csv`
  (the k-sensitivity data underlying Appendix Table D.1) regenerates byte-for-byte from the
  raw per-k data bundled under `results/LR/feature_perturbation_balanced_main/k_*/` and
  `results/LightGBM/feature_perturbation_lightgbm_{gain,permutation}_balanced_main/k_*/`
  via `scripts/figures_and_tables/export_k_sensitivity_table.py --results-dir
  5_3_targeted_perturbation_response/results`, and matches Appendix Table D.1 in the
  manuscript exactly.
- `results/LR`, `results/LightGBM`, `results/RF`: per-model experiment run output
- `scripts/common/`: experiment runners shared across model families (the
  `_with_rf.py` files natively handle RF too, via `model_type`/`--model`)
- `scripts/figures_and_tables/`: figure/table builders
- `results/manuscript_tables/pe_intervention_detailed.tex`: Table 7 (Wild (B)-only,
  LR/LightGBM/RF), built by `scripts/figures_and_tables/build_pe_intervention_detailed_table.py`
- `pe_inspired_feature_intervention/`: assets backing the Table 7 PE-inspired intervention
  table in the main text. `scripts/run_pe_feature_intervention_validation.py` handles
  LR/LightGBM; `scripts/run_pe_feature_intervention_validation_with_rf.py` is a self-contained
  fork that natively adds RF (`--model random_forest`), invoked via
  `scripts/RF/run_pe_intervention_rf.sh`

### `5_4_density_conditioned_reliability/`
Covers Section 5.4 on density-conditioned reliability before perturbation.

Relevant manuscript outputs:
- Figure 8: density-conditioned AUC and Mix@10 across feature subsets under Wild (B)
- Table 5: disagreement gap between dense and sparse regions
- Appendix subsection F.1 "Class-Conditional Density-Composition Sensitivity" (reliability half)

Where to look:
- `results/figures/`: density-conditioned AUC and Mix@10 manuscript plots
  (`density_conditioned_{auc,mix_at_10}_wild_b_{lr,lightgbm,rf}.pdf`), built by
  `scripts/figures_and_tables/plot_density_stratified_reliability.py --input-dir
  results/<Model>/<controlled_main_run>/k_10 --single-panels-only --skip-conflict-table`,
  renaming its `figure_density_reliability_{auc,mix_at_10}.pdf` output to
  `density_conditioned_{auc,mix_at_10}_wild_b_<model>.pdf`.
- `results/manuscript_tables/`: rank-disagreement and significance tables used in Section 5.4.
  `density_conditioned_rank_disagreement.tex` (Table 5)'s RF column is generated
  via `plot_density_stratified_reliability.py --input-dir results/RF/rf_full_wild_b/k_10`,
  producing `results/RF/density_reliability_rf_controlled_table_rq3_density_reliability_conflicts.csv`;
  `scripts/figures_and_tables/build_density_conditioned_rank_disagreement_table.py`
  combines all three models into the table.
- `results/LR`, `results/LightGBM`, `results/RF`: per-model experiment run output
- `scripts/common/`: density-stratified experiment runners shared across model families
- `scripts/figures_and_tables/`: figure builder
- `sensitivity/`: D3 class-conditional re-binning sensitivity check backing Appendix F.1
  (patched runners, derived configs, rerun outputs, comparison CSV); the resulting
  table lives under `Appendix/Section_F/`

### `5_5_density_conditioned_fragility/`
Covers Section 5.5 on density-conditioned fragility under masking and intervention.

Relevant manuscript outputs:
- Figure 9: density-conditioned AUC drop under feature masking
- Table 6: density-conditioned malware-to-benign transition rates under the PE-inspired import intervention
- Appendix Figure F.1: density-conditioned flip-rate plots
- Appendix subsection F.1 "Class-Conditional Density-Composition Sensitivity" (stress-amplification
  half; not to be confused with Appendix Figure F.1 above, a different counter)

Where to look:
- `results/figures/`: manuscript plots for AUC drop and flip-rate comparisons
- `results/manuscript_tables/`: density fragility and significance table exports,
  including `density_import_intervention.tex` (Table 6, LightGBM-only, built by
  `scripts/figures_and_tables/build_density_import_intervention_table.py` from
  `density_import_intervention.csv`). The manuscript's own text frames this check
  as "an additional LightGBM-only check" by design, not an RF result pending
  completion; no RF variant of this table is generated here.
- `results/LR`, `results/LightGBM`, `results/RF`: per-model experiment run output
- `scripts/common/`: experiment runners shared across model families (the
  `_with_rf.py` files natively handle RF too, via `model_type`/`--model`), plus
  `run_density_conditioned_import_intervention_probe.py` (the LightGBM-only
  block-3 import-intervention probe backing Table 6 — configs under
  `configs/LightGBM/block3_probe_lightgbm_imports_seed*.json`, raw/merged
  outputs under `results/LightGBM/block3_probe_lightgbm_imports_*`) and its merge
  script (`merge_density_conditioned_import_intervention_probe_results.py`) and parallel-run shell scripts
- `scripts/figures_and_tables/`: figure/table builders
- `sensitivity/`: D4 class-conditional re-binning sensitivity check backing Appendix F.1
  (patched runners, derived configs, rerun outputs, comparison CSV); the resulting
  table lives under `Appendix/Section_F/`

### `MLP/`
Covers Section 5.6, the "Generalization to a Neural Classifier" subsection.

Relevant manuscript outputs:
- Table 8: cross-model comparison of the main DCSA findings in Wild (B)
- Appendix Tables E.1-E.2: MLP rows in the paired masking significance analysis
  (see `Appendix/Section_E/`, not here — shared with LR/LightGBM/RF)
- Appendix Tables G.1-G.4: full MLP representation, masking, density-reliability, and density-masking results

## Notes

- Table 8 ("Cross-model comparison of the main DCSA findings in Wild (B)"), like Table 9, is a
  narrative synthesis table (bullet-point comparison of LR/LightGBM vs. MLP findings) authored
  directly in the manuscript. It has no generating script, and none is expected — there is no
  numeric table to derive it from, unlike the other MLP appendix tables G.1-G.4, which are
  data-derived (see `MLP/`).
- Table 9 ("DCSA signals, supported interpretations, and candidate follow-up analyses") is a
  narrative synthesis table authored directly in the manuscript's Discussion section; it has
  no separate generating script or artifact in this workspace.
- Table 2 (evaluation-metric interpretation criteria) belongs to the Experimental Protocol
  section and is narrative/methodological, not derived from any script in this workspace.
- Table 1 (dataset composition) is data-derived:
  `5_1_representation_level_separability_disagreement/scripts/figures_and_tables/build_dataset_composition_table.py`
  computes exactly the row/column values in `tables/dataset.tex`, writing
  `results/LR/table_rq1_class_packing_composition.csv`.
