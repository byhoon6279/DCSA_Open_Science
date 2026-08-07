# Appendix Section G: Full MLP Result Tables

Backing scripts and derived outputs for the manuscript's Appendix Section G
tables covering the full MLP representation, masking, density-reliability,
and density-masking results. The tables are generated from the bundled
per-seed data under `artifacts/MLP/results/`.

Unlike Appendix Section E, these tables report descriptive aggregations
(means and counts) of the bundled per-seed/per-week results.

## Table G.1 — `appendix_mlp_representation_full.tex` / `.csv`

- Script: `scripts/build_mlp_representation_full.py`
- Source data: `artifacts/MLP/results/5_1/mlp_rq1_wild_b_main/metric_rows.csv`
- Unweighted mean over 5 seeds x 12 test weeks (n=60 per feature group).
- Status: **verified exact match**, all 20 reported cells (5 feature groups x
  4 metrics).

## Table G.2 — `appendix_mlp_masking_full.tex` / `.csv`

- Script: `scripts/build_mlp_masking_full.py`
- Source data: `artifacts/MLP/results/5_3/mlp_rq2_targeted_masking_wild_b_main/masking_rows.csv`
- Important-masking columns are averaged over the 5 seeds directly; random-masking
  columns are averaged over the 3 control repeats within each seed first, then
  over the 5 seeds, matching the manuscript caption.
- Status: **verified exact match**, all 90 reported cells (5 feature groups x
  3 masking strengths x 6 columns). Three cell means land exactly on a rounding
  midpoint (e.g. a raw mean of `0.33945`); `fmt()` rounds these with an explicit
  decimal round-half-up rule rather than Python's default binary-float
  formatting, which is what the manuscript's own values use.

## Table G.3 — `appendix_mlp_density_reliability_full.tex` / `.csv`

- Script: `scripts/build_mlp_density_reliability_full.py`
- Source data: `artifacts/MLP/results/5_4/mlp_rq3_density_reliability_wild_b_main/metric_rows.csv`
- Mean over the 5 seeds per feature group x density bin.
- Status: **verified exact match**, all 45 reported cells (5 feature groups x
  3 density bins x 3 metrics).

## Table G.4 — `appendix_mlp_density_masking_full.tex` / `.csv`

- Script: `scripts/build_mlp_density_masking_full.py`
- Source data: `artifacts/MLP/results/5_5/mlp_rq3_density_fragility_wild_b_main/amplification_rows.csv`
- Mean and positive-case count over the 5 seeds x 3 masking strengths (15
  conditions per feature group), for important and random masking separately.
- Status: **verified exact match**, all 20 reported cells (5 feature groups x
  2 masking types x (mean + positive-case count)).
