# Appendix Section C: Representation-Level Disagreement Details

Backing script and output for the manuscript's Appendix Figure C.1
(`\label{fig:umap}`, qualitative UMAP projections of four feature
representations under Wild (B)), part of Appendix C.1's "Hierarchy Stability
and Structural Mismatch Across Settings" subsection.

The plotting script uses the bundled metric summaries together with the
EMBER2024 test feature files to reconstruct the balanced sample pool and
generate the UMAP projections.

## Figure mapping

| Manuscript label | Files here | Built from |
|---|---|---|
| `fig:umap` | `results/umap_feature_subset_{all,header,section,imports}_mixed_balanced.pdf` | `artifacts/5_1_representation_level_separability_disagreement/results/{LR,LightGBM}/win32_all_train_all_test*balanced_test/` |

## How to run

```bash
python3 scripts/plot_umap_feature_subset_comparison.py --setting-preset mixed_balanced
```

Requires `umap-learn`, `seaborn`, the bundled `results.json` and summary
CSVs above, and the EMBER2024 feature files placed under `Data/` as
described in the repository-level README.
