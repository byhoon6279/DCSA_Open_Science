# Appendix Section A: Taxonomy-Adjacent Alignment and Label Limitations

Backing script and output for the manuscript's Appendix Table A.1
(`\label{tab:appendix_family_distribution}`, malware-family composition in
the Wild (U) and Unpacked (B) views).

The table builder uses
`artifacts/5_1_representation_level_separability_disagreement/results/LR/table_family_composition_original_vs_unpacked.csv`,
which is included in this package.

## Table mapping

| Manuscript label | File here | Built from |
|---|---|---|
| `tab:appendix_family_distribution` | `results/appendix_family_distribution.tex` | `artifacts/5_1_representation_level_separability_disagreement/results/LR/table_family_composition_original_vs_unpacked.csv` |

## How to run

```bash
python3 scripts/build_appendix_family_distribution_table.py
```

Requires the source CSV above (already bundled under
`5_1_representation_level_separability_disagreement/`).
