"""
Step 3: reproduce Table 5-5 (LightGBM, 10% hashed-import PE-inspired
intervention, Wild(B)) with the same clean-vs-original-importance comparison
as steps 1-2. This mirrors
pe_inspired_feature_intervention/scripts/run_pe_feature_intervention_validation.py's
`imports_benign_mass_injection` operator exactly: malware-only rows have their
top-k importance-ranked (or random) hashed-import dimensions clamped into the
[q75, q95] range computed from BENIGN TRAIN rows (train partition unchanged),
then flip/M->B rate is measured against the unperturbed baseline prediction.
"""
from __future__ import annotations

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
    "pe_inspired_feature_intervention/configs/pe_intervention_balanced_main.json"
)
RESULTS_DIR = Path(__file__).resolve().parents[4] / "Appendix/Supplement_SHA_duplicates/supplementary_val_dedup/results"

CONFIG = json.loads(CONFIG_PATH.read_text())
SEEDS = [int(s) for s in CONFIG["seeds"]]
MAX_TRAIN_PER_CLASS = CONFIG["max_train_per_class"]
MAX_TEST_PER_CLASS = CONFIG["max_test_per_class"]
VALIDATION_FRACTION = float(CONFIG["validation_fraction"])
PLATFORM = CONFIG["platform"]
TRAIN_WEEKS = CONFIG["train_weeks"]
TEST_WEEKS = CONFIG["test_weeks"]
STRENGTH_10PCT = 0.10
GROUP = "imports"
THRESHOLD = float(CONFIG.get("decision_threshold", 0.5))
RANDOM_TRIALS = int(CONFIG["random_trials"])


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def apply_clamp(X_raw_selected: np.ndarray, q75_sel: np.ndarray, q95_sel: np.ndarray) -> np.ndarray:
    lower = np.broadcast_to(q75_sel, X_raw_selected.shape)
    upper = np.broadcast_to(q95_sel, X_raw_selected.shape)
    return np.minimum(np.maximum(X_raw_selected, lower), upper)


def main() -> None:
    log("loading train pool + test weeks ...")
    train_pool = load_pool(TRAIN_WEEKS, PLATFORM, "train")
    test_pools = load_test_weeks(TEST_WEEKS, PLATFORM)

    rows = []

    for seed in SEEDS:
        balanced_idx = mu.balance_indices(train_pool.y, seed=seed, max_per_class=MAX_TRAIN_PER_CLASS)
        y_balanced = train_pool.y[balanced_idx]
        train_local, _val_local = mu.split_train_val(
            n_samples=len(balanced_idx), y=y_balanced, seed=seed, validation_fraction=VALIDATION_FRACTION,
        )
        train_abs = balanced_idx[train_local]
        y_train_fit = y_balanced[train_local]

        imp_path = RESULTS_DIR / f"importance_seed{seed}_{GROUP}.npz"
        imp_data = np.load(imp_path)
        importance_orig = imp_data["importance_orig"]
        importance_clean = imp_data["importance_clean"]
        n_features = len(importance_orig)
        n_select = max(1, int(np.ceil(n_features * STRENGTH_10PCT)))

        X_pool = train_pool.group(GROUP)
        X_train_fit_raw = X_pool[train_abs]
        scaler = mu.fit_scaler(X_train_fit_raw)
        X_train_scaled = mu.transform_with_scaler(scaler, X_train_fit_raw)
        clf = mu.fit_lightgbm(X_train_scaled, y_train_fit, seed=seed, n_jobs=2)

        benign_raw = X_train_fit_raw[y_train_fit == 0]
        q75 = np.percentile(benign_raw, 75, axis=0)
        q95 = np.percentile(benign_raw, 95, axis=0)

        idx_orig = mu.select_top_indices(importance_orig, STRENGTH_10PCT)
        idx_clean = mu.select_top_indices(importance_clean, STRENGTH_10PCT)
        same_set = set(idx_orig.tolist()) == set(idx_clean.tolist())
        jaccard = len(set(idx_orig.tolist()) & set(idx_clean.tolist())) / len(set(idx_orig.tolist()) | set(idx_clean.tolist()))

        for week_idx, week in enumerate(TEST_WEEKS):
            test_pool = test_pools[week]
            test_balanced_idx = mu.balance_indices(test_pool.y, seed=seed + week_idx + 1, max_per_class=MAX_TEST_PER_CLASS)
            y_test = test_pool.y[test_balanced_idx]
            X_test_raw = test_pool.group(GROUP)[test_balanced_idx]
            X_test_scaled = mu.transform_with_scaler(scaler, X_test_raw)
            baseline_proba_full = mu.positive_proba(clf, X_test_scaled)

            malware_mask = y_test == 1
            X_mal_raw = X_test_raw[malware_mask]
            X_mal_scaled = X_test_scaled[malware_mask]
            baseline_proba_mal = baseline_proba_full[malware_mask]
            y_mal = y_test[malware_mask]

            def eval_intervention(indices):
                q75_sel = q75[indices]
                q95_sel = q95[indices]
                new_selected = apply_clamp(X_mal_raw[:, indices], q75_sel, q95_sel)
                perturbed_raw = np.array(X_mal_raw, copy=True)
                perturbed_raw[:, indices] = new_selected
                perturbed_scaled = mu.transform_with_scaler(scaler, perturbed_raw)
                proba = mu.positive_proba(clf, perturbed_scaled)
                flip = mu.evaluate_flip_metrics(baseline_proba_mal, proba, y_mal, THRESHOLD)
                return flip

            flip_orig = eval_intervention(idx_orig)
            flip_clean = eval_intervention(idx_clean)

            group_offset = sum(ord(ch) for ch in GROUP)
            op_offset = sum(ord(ch) for ch in "imports_benign_mass_injection")
            SELECTION_OFFSET_RANDOM = 10_000_000  # matches DCSA's intervention_seed(): 0 for "important", 1e7 for "random"
            random_flips, random_m2b = [], []
            for trial_id in range(1, RANDOM_TRIALS + 1):
                seed_val = (
                    seed * 100_000_000 + week_idx * 1_000_000 + group_offset * 10_000
                    + op_offset * 10 + int(round(STRENGTH_10PCT * 10_000)) + trial_id
                    + SELECTION_OFFSET_RANDOM
                )
                rng = np.random.default_rng(seed_val)
                random_indices = rng.choice(n_features, size=n_select, replace=False)
                flip_r = eval_intervention(random_indices)
                random_flips.append(flip_r["flip_rate"])
                random_m2b.append(flip_r["malware_to_benign_rate"])
            # (kept as a simple loop: malware-only subset here is small compared to
            # step2's full test-week batches, so per-call overhead is not the bottleneck)

            rows.append({
                "seed": seed,
                "test_week": week,
                "topk_identical": same_set,
                "topk_jaccard": jaccard,
                "n_selected": n_select,
                "flip_rate_important_orig": flip_orig["flip_rate"],
                "flip_rate_important_clean": flip_clean["flip_rate"],
                "m2b_rate_important_orig": flip_orig["malware_to_benign_rate"],
                "m2b_rate_important_clean": flip_clean["malware_to_benign_rate"],
                "flip_rate_random_mean": float(np.mean(random_flips)),
                "m2b_rate_random_mean": float(np.mean(random_m2b)),
            })
            log(f"seed={seed} week={week_idx+1}/12 flip_orig={flip_orig['flip_rate']:.4f} flip_clean={flip_clean['flip_rate']:.4f}")

        (RESULTS_DIR / "step3_pe_intervention_rows.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    log("STEP3_DONE")


if __name__ == "__main__":
    main()
