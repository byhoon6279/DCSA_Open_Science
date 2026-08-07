# Paired Masking Reanalysis

This report summarizes the paired seed-level masking analysis.

## Availability

- LR / header / auc_drop: ok (artifacts/5_3_targeted_perturbation_response/results/LR/feature_perturbation_balanced_main/k_10/trial_results.csv)
  note: Used exact one-sided sign-flip test on 5 seed-level paired deltas.
- LR / imports / auc_drop: ok (artifacts/5_3_targeted_perturbation_response/results/LR/feature_perturbation_balanced_main/k_10/trial_results.csv)
  note: Used exact one-sided sign-flip test on 5 seed-level paired deltas.
- LR / header / flip_rate: ok (artifacts/5_3_targeted_perturbation_response/results/LR/prediction_flip_balanced_main/trial_results.csv)
  note: Used exact one-sided sign-flip test on 5 seed-level paired deltas.
- LR / imports / flip_rate: ok (artifacts/5_3_targeted_perturbation_response/results/LR/prediction_flip_balanced_main/trial_results.csv)
  note: Used exact one-sided sign-flip test on 5 seed-level paired deltas.
- LightGBM / header / auc_drop: ok (artifacts/5_3_targeted_perturbation_response/results/LightGBM/feature_perturbation_lightgbm_permutation_balanced_main/k_10/trial_results.csv)
  note: Used exact one-sided sign-flip test on 5 seed-level paired deltas.
- LightGBM / imports / auc_drop: ok (artifacts/5_3_targeted_perturbation_response/results/LightGBM/feature_perturbation_lightgbm_permutation_balanced_main/k_10/trial_results.csv)
  note: Used exact one-sided sign-flip test on 5 seed-level paired deltas.
- LightGBM / header / flip_rate: ok (artifacts/5_3_targeted_perturbation_response/results/LightGBM/prediction_flip_lightgbm_balanced_main/trial_results.csv)
  note: Used exact one-sided sign-flip test on 5 seed-level paired deltas.
- LightGBM / imports / flip_rate: ok (artifacts/5_3_targeted_perturbation_response/results/LightGBM/prediction_flip_lightgbm_balanced_main/trial_results.csv)
  note: Used exact one-sided sign-flip test on 5 seed-level paired deltas.
- RF / header / auc_drop: ok (artifacts/5_3_targeted_perturbation_response/results/RF/rq2_2_rf_full_wild_b/trial_results.csv)
  note: Used exact one-sided sign-flip test on 5 seed-level paired deltas.
- RF / imports / auc_drop: ok (artifacts/5_3_targeted_perturbation_response/results/RF/rq2_2_rf_full_wild_b/trial_results.csv)
  note: Used exact one-sided sign-flip test on 5 seed-level paired deltas.
- RF / header / flip_rate: ok (artifacts/5_3_targeted_perturbation_response/results/RF/rq2_3_rf_full_wild_b/trial_results.csv)
  note: Used exact one-sided sign-flip test on 5 seed-level paired deltas.
- RF / imports / flip_rate: ok (artifacts/5_3_targeted_perturbation_response/results/RF/rq2_3_rf_full_wild_b/trial_results.csv)
  note: Used exact one-sided sign-flip test on 5 seed-level paired deltas.
- MLP / all / auc_drop: ok (artifacts/MLP/results/5_3/mlp_rq2_targeted_masking_wild_b_main/masking_rows.csv)
  note: Used exact one-sided sign-flip test on 5 seed-level paired deltas.
- MLP / header / auc_drop: ok (artifacts/MLP/results/5_3/mlp_rq2_targeted_masking_wild_b_main/masking_rows.csv)
  note: Used exact one-sided sign-flip test on 5 seed-level paired deltas.
- MLP / section / auc_drop: ok (artifacts/MLP/results/5_3/mlp_rq2_targeted_masking_wild_b_main/masking_rows.csv)
  note: Used exact one-sided sign-flip test on 5 seed-level paired deltas.
- MLP / imports / auc_drop: ok (artifacts/MLP/results/5_3/mlp_rq2_targeted_masking_wild_b_main/masking_rows.csv)
  note: Used exact one-sided sign-flip test on 5 seed-level paired deltas.
- MLP / strings / auc_drop: ok (artifacts/MLP/results/5_3/mlp_rq2_targeted_masking_wild_b_main/masking_rows.csv)
  note: Used exact one-sided sign-flip test on 5 seed-level paired deltas.
- MLP / all / flip_rate: ok (artifacts/MLP/results/5_3/mlp_rq2_targeted_masking_wild_b_main/masking_rows.csv)
  note: Used exact one-sided sign-flip test on 5 seed-level paired deltas.
- MLP / header / flip_rate: ok (artifacts/MLP/results/5_3/mlp_rq2_targeted_masking_wild_b_main/masking_rows.csv)
  note: Used exact one-sided sign-flip test on 5 seed-level paired deltas.
- MLP / section / flip_rate: ok (artifacts/MLP/results/5_3/mlp_rq2_targeted_masking_wild_b_main/masking_rows.csv)
  note: Used exact one-sided sign-flip test on 5 seed-level paired deltas.
- MLP / imports / flip_rate: ok (artifacts/MLP/results/5_3/mlp_rq2_targeted_masking_wild_b_main/masking_rows.csv)
  note: Used exact one-sided sign-flip test on 5 seed-level paired deltas.
- MLP / strings / flip_rate: ok (artifacts/MLP/results/5_3/mlp_rq2_targeted_masking_wild_b_main/masking_rows.csv)
  note: Used exact one-sided sign-flip test on 5 seed-level paired deltas.

## Completed Rows

- LR / header / auc_drop / 0.01: mean=+0.065927, median=+0.078124, 95% CI=[+0.032765, +0.086357], positive=5/5, raw_p=0.03125, holm_p=1
- LR / header / auc_drop / 0.05: mean=+0.110459, median=+0.098178, 95% CI=[+0.085679, +0.146039], positive=5/5, raw_p=0.03125, holm_p=1
- LR / header / auc_drop / 0.1: mean=+0.113959, median=+0.116120, 95% CI=[+0.094746, +0.133515], positive=5/5, raw_p=0.03125, holm_p=1
- LR / imports / auc_drop / 0.01: mean=+0.038283, median=+0.033125, 95% CI=[+0.031471, +0.050078], positive=5/5, raw_p=0.03125, holm_p=1
- LR / imports / auc_drop / 0.05: mean=+0.088009, median=+0.081474, 95% CI=[+0.069779, +0.114640], positive=5/5, raw_p=0.03125, holm_p=1
- LR / imports / auc_drop / 0.1: mean=+0.117938, median=+0.115721, 95% CI=[+0.093285, +0.143965], positive=5/5, raw_p=0.03125, holm_p=1
- LR / header / flip_rate / 0.01: mean=+0.124365, median=+0.095467, 95% CI=[+0.090647, +0.181707], positive=5/5, raw_p=0.03125, holm_p=1
- LR / header / flip_rate / 0.05: mean=+0.055655, median=+0.040910, 95% CI=[-0.001093, +0.125602], positive=3/5, raw_p=0.125, holm_p=1
- LR / header / flip_rate / 0.1: mean=+0.104314, median=+0.131097, 95% CI=[+0.041630, +0.141239], positive=4/5, raw_p=0.0625, holm_p=1
- LR / imports / flip_rate / 0.01: mean=+0.079416, median=+0.068456, 95% CI=[+0.065860, +0.102659], positive=5/5, raw_p=0.03125, holm_p=1
- LR / imports / flip_rate / 0.05: mean=+0.122653, median=+0.107440, 95% CI=[+0.102527, +0.156311], positive=5/5, raw_p=0.03125, holm_p=1
- LR / imports / flip_rate / 0.1: mean=+0.139547, median=+0.128457, 95% CI=[+0.113559, +0.171517], positive=5/5, raw_p=0.03125, holm_p=1
- LightGBM / header / auc_drop / 0.01: mean=+0.003291, median=-0.000670, 95% CI=[-0.000962, +0.007611], positive=2/5, raw_p=0.25, holm_p=1
- LightGBM / header / auc_drop / 0.05: mean=+0.039180, median=+0.041651, 95% CI=[+0.031320, +0.044681], positive=5/5, raw_p=0.03125, holm_p=1
- LightGBM / header / auc_drop / 0.1: mean=+0.070248, median=+0.070852, 95% CI=[+0.056348, +0.084148], positive=5/5, raw_p=0.03125, holm_p=1
- LightGBM / imports / auc_drop / 0.01: mean=+0.025743, median=+0.022769, 95% CI=[+0.021104, +0.030381], positive=5/5, raw_p=0.03125, holm_p=1
- LightGBM / imports / auc_drop / 0.05: mean=+0.057225, median=+0.054326, 95% CI=[+0.049788, +0.067433], positive=5/5, raw_p=0.03125, holm_p=1
- LightGBM / imports / auc_drop / 0.1: mean=+0.087385, median=+0.083646, 95% CI=[+0.069870, +0.110704], positive=5/5, raw_p=0.03125, holm_p=1
- LightGBM / header / flip_rate / 0.01: mean=+0.003766, median=-0.010705, 95% CI=[-0.012767, +0.020829], positive=2/5, raw_p=0.34375, holm_p=1
- LightGBM / header / flip_rate / 0.05: mean=+0.090014, median=+0.083421, 95% CI=[+0.075528, +0.107171], positive=5/5, raw_p=0.03125, holm_p=1
- LightGBM / header / flip_rate / 0.1: mean=+0.156045, median=+0.142313, 95% CI=[+0.105943, +0.206146], positive=5/5, raw_p=0.03125, holm_p=1
- LightGBM / imports / flip_rate / 0.01: mean=+0.094305, median=+0.088247, 95% CI=[+0.072001, +0.116608], positive=5/5, raw_p=0.03125, holm_p=1
- LightGBM / imports / flip_rate / 0.05: mean=+0.161853, median=+0.151661, 95% CI=[+0.143782, +0.186370], positive=5/5, raw_p=0.03125, holm_p=1
- LightGBM / imports / flip_rate / 0.1: mean=+0.211073, median=+0.193090, 95% CI=[+0.168587, +0.272418], positive=5/5, raw_p=0.03125, holm_p=1
- RF / header / auc_drop / 0.01: mean=+0.007050, median=+0.010921, 95% CI=[+0.001711, +0.012389], positive=3/5, raw_p=0.125, holm_p=1
- RF / header / auc_drop / 0.05: mean=+0.018982, median=+0.019659, 95% CI=[+0.017623, +0.020312], positive=5/5, raw_p=0.03125, holm_p=1
- RF / header / auc_drop / 0.1: mean=+0.062548, median=+0.057708, 95% CI=[+0.048289, +0.077581], positive=5/5, raw_p=0.03125, holm_p=1
- RF / imports / auc_drop / 0.01: mean=+0.004593, median=+0.003861, 95% CI=[+0.001324, +0.008000], positive=4/5, raw_p=0.0625, holm_p=1
- RF / imports / auc_drop / 0.05: mean=+0.020832, median=+0.012854, 95% CI=[+0.010903, +0.032083], positive=5/5, raw_p=0.03125, holm_p=1
- RF / imports / auc_drop / 0.1: mean=+0.038080, median=+0.039204, 95% CI=[+0.026323, +0.049838], positive=5/5, raw_p=0.03125, holm_p=1
- RF / header / flip_rate / 0.01: mean=+0.018133, median=+0.019168, 95% CI=[+0.004335, +0.033734], positive=5/5, raw_p=0.03125, holm_p=1
- RF / header / flip_rate / 0.05: mean=+0.110272, median=+0.113337, 95% CI=[+0.102265, +0.117510], positive=5/5, raw_p=0.03125, holm_p=1
- RF / header / flip_rate / 0.1: mean=+0.148703, median=+0.137295, 95% CI=[+0.130409, +0.166997], positive=5/5, raw_p=0.03125, holm_p=1
- RF / imports / flip_rate / 0.01: mean=+0.039619, median=+0.029472, 95% CI=[+0.024822, +0.056579], positive=5/5, raw_p=0.03125, holm_p=1
- RF / imports / flip_rate / 0.05: mean=+0.084167, median=+0.052518, 95% CI=[+0.041448, +0.139266], positive=5/5, raw_p=0.03125, holm_p=1
- RF / imports / flip_rate / 0.1: mean=+0.106194, median=+0.098308, 95% CI=[+0.069137, +0.146927], positive=5/5, raw_p=0.03125, holm_p=1
- MLP / all / auc_drop / 0.01: mean=+0.113100, median=+0.080401, 95% CI=[+0.051226, +0.194205], positive=5/5, raw_p=0.03125, holm_p=0.9375
- MLP / all / auc_drop / 0.05: mean=+0.122895, median=+0.130990, 95% CI=[+0.093297, +0.152494], positive=5/5, raw_p=0.03125, holm_p=0.9375
- MLP / all / auc_drop / 0.1: mean=+0.133728, median=+0.125409, 95% CI=[+0.111479, +0.162177], positive=5/5, raw_p=0.03125, holm_p=0.9375
- MLP / header / auc_drop / 0.01: mean=+0.008110, median=-0.011304, 95% CI=[-0.011639, +0.029515], positive=2/5, raw_p=0.28125, holm_p=0.9375
- MLP / header / auc_drop / 0.05: mean=+0.022794, median=+0.029925, 95% CI=[-0.010539, +0.051642], positive=4/5, raw_p=0.125, holm_p=0.9375
- MLP / header / auc_drop / 0.1: mean=+0.086084, median=+0.037164, 95% CI=[-0.012002, +0.237832], positive=4/5, raw_p=0.125, holm_p=0.9375
- MLP / section / auc_drop / 0.01: mean=+0.131936, median=+0.113542, 95% CI=[+0.092667, +0.173848], positive=5/5, raw_p=0.03125, holm_p=0.9375
- MLP / section / auc_drop / 0.05: mean=+0.182741, median=+0.172533, 95% CI=[+0.140552, +0.224931], positive=5/5, raw_p=0.03125, holm_p=0.9375
- MLP / section / auc_drop / 0.1: mean=+0.206073, median=+0.217521, 95% CI=[+0.169182, +0.242963], positive=5/5, raw_p=0.03125, holm_p=0.9375
- MLP / imports / auc_drop / 0.01: mean=+0.018671, median=+0.014958, 95% CI=[+0.013399, +0.026470], positive=5/5, raw_p=0.03125, holm_p=0.9375
- MLP / imports / auc_drop / 0.05: mean=+0.037008, median=+0.042124, 95% CI=[+0.030018, +0.042673], positive=5/5, raw_p=0.03125, holm_p=0.9375
- MLP / imports / auc_drop / 0.1: mean=+0.055174, median=+0.050079, 95% CI=[+0.045868, +0.066429], positive=5/5, raw_p=0.03125, holm_p=0.9375
- MLP / strings / auc_drop / 0.01: mean=+0.088214, median=+0.096087, 95% CI=[+0.022804, +0.153625], positive=5/5, raw_p=0.03125, holm_p=0.9375
- MLP / strings / auc_drop / 0.05: mean=+0.171901, median=+0.114933, 95% CI=[+0.092248, +0.279275], positive=5/5, raw_p=0.03125, holm_p=0.9375
- MLP / strings / auc_drop / 0.1: mean=+0.061812, median=+0.064156, 95% CI=[+0.018774, +0.099259], positive=4/5, raw_p=0.0625, holm_p=0.9375
- MLP / all / flip_rate / 0.01: mean=+0.347827, median=+0.346950, 95% CI=[+0.236422, +0.439570], positive=5/5, raw_p=0.03125, holm_p=0.9375
- MLP / all / flip_rate / 0.05: mean=+0.292930, median=+0.363408, 95% CI=[+0.175248, +0.369942], positive=5/5, raw_p=0.03125, holm_p=0.9375
- MLP / all / flip_rate / 0.1: mean=+0.229972, median=+0.233733, 95% CI=[+0.153427, +0.291390], positive=5/5, raw_p=0.03125, holm_p=0.9375
- MLP / header / flip_rate / 0.01: mean=+0.113522, median=+0.080067, 95% CI=[+0.035890, +0.191635], positive=5/5, raw_p=0.03125, holm_p=0.9375
- MLP / header / flip_rate / 0.05: mean=+0.054702, median=+0.061192, 95% CI=[-0.059282, +0.181678], positive=4/5, raw_p=0.28125, holm_p=0.9375
- MLP / header / flip_rate / 0.1: mean=+0.011562, median=-0.023700, 95% CI=[-0.111522, +0.190008], positive=1/5, raw_p=0.5, holm_p=0.9375
- MLP / section / flip_rate / 0.01: mean=+0.175462, median=+0.171275, 95% CI=[+0.129508, +0.224043], positive=5/5, raw_p=0.03125, holm_p=0.9375
- MLP / section / flip_rate / 0.05: mean=+0.184442, median=+0.173217, 95% CI=[+0.129643, +0.243130], positive=5/5, raw_p=0.03125, holm_p=0.9375
- MLP / section / flip_rate / 0.1: mean=+0.230365, median=+0.229825, 95% CI=[+0.157205, +0.299487], positive=5/5, raw_p=0.03125, holm_p=0.9375
- MLP / imports / flip_rate / 0.01: mean=+0.059693, median=+0.056392, 95% CI=[+0.048655, +0.073852], positive=5/5, raw_p=0.03125, holm_p=0.9375
- MLP / imports / flip_rate / 0.05: mean=+0.098133, median=+0.094908, 95% CI=[+0.088355, +0.110103], positive=5/5, raw_p=0.03125, holm_p=0.9375
- MLP / imports / flip_rate / 0.1: mean=+0.124478, median=+0.120792, 95% CI=[+0.112592, +0.139007], positive=5/5, raw_p=0.03125, holm_p=0.9375
- MLP / strings / flip_rate / 0.01: mean=+0.308753, median=+0.476875, 95% CI=[+0.123048, +0.494458], positive=5/5, raw_p=0.03125, holm_p=0.9375
- MLP / strings / flip_rate / 0.05: mean=+0.367810, median=+0.386783, 95% CI=[+0.283215, +0.436320], positive=5/5, raw_p=0.03125, holm_p=0.9375
- MLP / strings / flip_rate / 0.1: mean=+0.308280, median=+0.361908, 95% CI=[+0.154502, +0.442512], positive=5/5, raw_p=0.03125, holm_p=0.9375
