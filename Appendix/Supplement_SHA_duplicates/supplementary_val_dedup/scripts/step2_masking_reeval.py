"""
Step 2: reproduce the Figure 5 / Figure 7 masking response (AUC degradation +
prediction flip rate) on the same fixed 12-week evaluation set, comparing
masking derived from the ORIGINAL (contaminated) validation importance vs.
the CLEANED (train-overlap + internal-duplicate removed) validation importance.
Train samples, test samples, thresholds, strengths, random-trial scheme and
aggregation are all unchanged from DCSA_Open_Science's
run_feature_perturbation_sensitivity.py / run_prediction_flip_analysis.py.
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_pool import load_pool, load_test_weeks  # noqa: E402
import model_utils as mu  # noqa: E402

CONFIG_PATH = Path(
    str(Path(__file__).resolve().parents[4]) + "/artifacts/5_3_targeted_perturbation_response/"
    "configs/LightGBM/feature_perturbation_lightgbm_balanced_main.json"
)
RESULTS_DIR = Path(__file__).resolve().parents[4] / "Appendix/Supplement_SHA_duplicates/supplementary_val_dedup/results"

CONFIG = json.loads(CONFIG_PATH.read_text())
SEEDS = [int(s) for s in CONFIG["seeds"]]
GROUPS = list(CONFIG["feature_groups"])
STRENGTHS = [float(s) for s in CONFIG["perturbation_strengths"]]
RANDOM_TRIALS = int(CONFIG["random_trials"])
MAX_TRAIN_PER_CLASS = CONFIG["max_train_per_class"]
MAX_TEST_PER_CLASS = CONFIG["max_test_per_class"]
VALIDATION_FRACTION = float(CONFIG["validation_fraction"])
PLATFORM = CONFIG["platform"]
TRAIN_WEEKS = CONFIG["train_weeks"]
TEST_WEEKS = CONFIG["test_weeks"]
THRESHOLD = 0.5


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    log("loading train pool + test weeks ...")
    train_pool = load_pool(TRAIN_WEEKS, PLATFORM, "train")
    test_pools = load_test_weeks(TEST_WEEKS, PLATFORM)
    log(f"train_pool n={len(train_pool)}, {len(test_pools)} test weeks loaded")

    rows = []

    for seed in SEEDS:
        balanced_idx = mu.balance_indices(train_pool.y, seed=seed, max_per_class=MAX_TRAIN_PER_CLASS)
        y_balanced = train_pool.y[balanced_idx]
        train_local, _val_local = mu.split_train_val(
            n_samples=len(balanced_idx), y=y_balanced, seed=seed, validation_fraction=VALIDATION_FRACTION,
        )
        train_abs = balanced_idx[train_local]
        y_train_fit = y_balanced[train_local]

        for group in GROUPS:
            imp_path = RESULTS_DIR / f"importance_seed{seed}_{group}.npz"
            imp_data = np.load(imp_path)
            importance_orig = imp_data["importance_orig"]
            importance_clean = imp_data["importance_clean"]
            n_features = len(importance_orig)

            X_pool = train_pool.group(group)
            X_train_fit_raw = X_pool[train_abs]
            scaler = mu.fit_scaler(X_train_fit_raw)
            X_train_scaled = mu.transform_with_scaler(scaler, X_train_fit_raw)
            clf = mu.fit_lightgbm(X_train_scaled, y_train_fit, seed=seed, n_jobs=2)
            zero_scaled_values = mu.zero_scaled_template(scaler, n_features)

            for week_idx, week in enumerate(TEST_WEEKS):
                test_pool = test_pools[week]
                test_balanced_idx = mu.balance_indices(test_pool.y, seed=seed + week_idx + 1, max_per_class=MAX_TEST_PER_CLASS)
                y_test = test_pool.y[test_balanced_idx]
                X_test_raw = test_pool.group(group)[test_balanced_idx]
                X_test_scaled = mu.transform_with_scaler(scaler, X_test_raw)

                baseline_proba = mu.positive_proba(clf, X_test_scaled)
                baseline_auc = mu.auc_score(y_test, baseline_proba)

                for strength in STRENGTHS:
                    idx_orig = mu.select_top_indices(importance_orig, strength)
                    idx_clean = mu.select_top_indices(importance_clean, strength)
                    n_mask = len(idx_orig)
                    same_set = set(idx_orig.tolist()) == set(idx_clean.tolist())
                    jaccard = len(set(idx_orig.tolist()) & set(idx_clean.tolist())) / len(set(idx_orig.tolist()) | set(idx_clean.tolist()))

                    def eval_masked(indices):
                        masked_scaled = mu.apply_zero_mask_scaled(X_test_scaled, indices, zero_scaled_values)
                        proba = mu.positive_proba(clf, masked_scaled)
                        auc = mu.auc_score(y_test, proba)
                        flip = mu.evaluate_flip_metrics(baseline_proba, proba, y_test, THRESHOLD)
                        return auc, flip

                    auc_orig, flip_orig = eval_masked(idx_orig)
                    auc_clean, flip_clean = eval_masked(idx_clean)

                    group_offset = sum(ord(ch) for ch in group)
                    random_index_sets = []
                    for trial_id in range(1, RANDOM_TRIALS + 1):
                        rng = np.random.default_rng(seed * 10_000 + week_idx * 1_000 + group_offset + trial_id)
                        random_index_sets.append(rng.choice(n_features, size=n_mask, replace=False))

                    random_aucs, random_flips, random_m2b = [], [], []
                    CHUNK = 5
                    for start in range(0, len(random_index_sets), CHUNK):
                        chunk = random_index_sets[start:start + CHUNK]
                        stacked = np.concatenate([
                            mu.apply_zero_mask_scaled(X_test_scaled, idx, zero_scaled_values) for idx in chunk
                        ], axis=0)
                        probs = mu.positive_proba(clf, stacked)
                        n = len(y_test)
                        for j in range(len(chunk)):
                            proba_j = probs[j * n:(j + 1) * n]
                            random_aucs.append(mu.auc_score(y_test, proba_j))
                            flip_j = mu.evaluate_flip_metrics(baseline_proba, proba_j, y_test, THRESHOLD)
                            random_flips.append(flip_j["flip_rate"])
                            random_m2b.append(flip_j["malware_to_benign_rate"])

                    rows.append({
                        "seed": seed,
                        "group": group,
                        "test_week": week,
                        "strength": strength,
                        "n_masked_features": n_mask,
                        "topk_identical": same_set,
                        "topk_jaccard": jaccard,
                        "baseline_auc": baseline_auc,
                        "important_auc_orig": auc_orig,
                        "important_auc_clean": auc_clean,
                        "delta_auc_orig": auc_orig - baseline_auc,
                        "delta_auc_clean": auc_clean - baseline_auc,
                        "random_auc_mean": float(np.mean(random_aucs)),
                        "delta_auc_random_mean": float(np.mean(random_aucs) - baseline_auc),
                        "important_flip_rate_orig": flip_orig["flip_rate"],
                        "important_flip_rate_clean": flip_clean["flip_rate"],
                        "important_m2b_orig": flip_orig["malware_to_benign_rate"],
                        "important_m2b_clean": flip_clean["malware_to_benign_rate"],
                        "important_b2m_orig": flip_orig["benign_to_malware_rate"],
                        "important_b2m_clean": flip_clean["benign_to_malware_rate"],
                        "random_flip_rate_mean": float(np.mean(random_flips)),
                        "random_m2b_mean": float(np.mean(random_m2b)),
                    })
                log(f"seed={seed} group={group} week={week_idx+1}/12 done")

            (RESULTS_DIR / "step2_trial_rows.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
            if rows:
                with (RESULTS_DIR / "step2_trial_rows.csv").open("w", newline="", encoding="utf-8") as fh:
                    writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                    writer.writeheader()
                    writer.writerows(rows)

    log("STEP2_DONE")


if __name__ == "__main__":
    main()
