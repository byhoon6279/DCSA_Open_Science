"""
Step 1 of the supplementary analysis.

For each of the 5 published seeds and both feature groups (header, imports):
  1. Reproduce the exact 52-week balanced pool -> 80/20 train/val split used by
     DCSA_Open_Science's feature_perturbation_lightgbm_*_balanced_main pipeline.
  2. Refit LightGBM on the 80% train partition (train samples/eval samples unchanged).
  3. Compute permutation importance on the ORIGINAL validation split (reproduction check).
  4. Build a CLEANED validation split: drop any val row whose SHA-256 also appears in the
     80% train partition, then drop internal duplicate SHA-256 within what remains
     (keep first occurrence). Flag -- do not silently resolve -- any SHA-256 that maps to
     more than one label or feature vector.
  5. Compute permutation importance on the CLEANED validation split (same model, same seed).
  6. Save both importance vectors + split/overlap bookkeeping for step 2.

Everything here only reads DCSA_Open_Science (shared/common.py constants only, no
DCSA code executed) and the raw EMBER2024 dataset; all outputs go under this
folder's results/ directory.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_pool import load_pool  # noqa: E402
import model_utils as mu  # noqa: E402

CONFIG_PATH = Path(
    str(Path(__file__).resolve().parents[4]) + "/artifacts/5_3_targeted_perturbation_response/"
    "configs/LightGBM/feature_perturbation_lightgbm_balanced_main.json"
)
RESULTS_DIR = Path(__file__).resolve().parents[4] / "Appendix/Supplement_SHA_duplicates/supplementary_val_dedup/results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CONFIG = json.loads(CONFIG_PATH.read_text())
SEEDS = [int(s) for s in CONFIG["seeds"]]
GROUPS = list(CONFIG["feature_groups"])  # ["header", "imports"]
MAX_TRAIN_PER_CLASS = CONFIG["max_train_per_class"]
VALIDATION_FRACTION = float(CONFIG["validation_fraction"])
PERMUTATION_REPEATS = int(CONFIG["permutation_repeats"])
PLATFORM = CONFIG["platform"]
TRAIN_WEEKS = CONFIG["train_weeks"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    log("loading 52-week train pool (header + imports) ...")
    pool = load_pool(TRAIN_WEEKS, PLATFORM, "train")
    log(f"pool loaded: n={len(pool)} benign={int((pool.y==0).sum())} malware={int((pool.y==1).sum())}")

    all_records = []

    for seed in SEEDS:
        balanced_idx = mu.balance_indices(pool.y, seed=seed, max_per_class=MAX_TRAIN_PER_CLASS)
        y_balanced = pool.y[balanced_idx]
        sha_balanced = pool.sha[balanced_idx]

        train_local, val_local = mu.split_train_val(
            n_samples=len(balanced_idx), y=y_balanced, seed=seed, validation_fraction=VALIDATION_FRACTION,
        )
        train_abs = balanced_idx[train_local]
        val_abs = balanced_idx[val_local]

        sha_train = sha_balanced[train_local]
        sha_val_orig = sha_balanced[val_local]
        y_train_fit = y_balanced[train_local]
        y_val_orig = y_balanced[val_local]

        train_sha_to_label = {}
        train_conflicts = 0
        for sha, lab in zip(sha_train, y_train_fit):
            if sha in train_sha_to_label and train_sha_to_label[sha] != lab:
                train_conflicts += 1
            else:
                train_sha_to_label[sha] = lab
        train_sha_set = set(train_sha_to_label.keys())

        overlap_mask = np.array([sha in train_sha_set for sha in sha_val_orig])
        overlap_label_conflicts = int(sum(
            1 for sha, lab in zip(sha_val_orig[overlap_mask], y_val_orig[overlap_mask])
            if train_sha_to_label[sha] != lab
        ))

        kept_after_overlap_removal = ~overlap_mask
        seen: dict[bytes, int] = {}
        internal_dup_conflict_count = 0
        internal_dup_removed_count = 0
        keep_mask = np.zeros(len(sha_val_orig), dtype=bool)
        for i in np.flatnonzero(kept_after_overlap_removal):
            sha = sha_val_orig[i]
            lab = y_val_orig[i]
            if sha not in seen:
                seen[sha] = i
                keep_mask[i] = True
            else:
                first_i = seen[sha]
                internal_dup_removed_count += 1
                if y_val_orig[first_i] != lab:
                    internal_dup_conflict_count += 1

        val_local_clean = val_local[keep_mask]
        val_abs_clean = balanced_idx[val_local_clean]
        sha_val_clean = pool.sha[val_abs_clean]
        y_val_clean = pool.y[val_abs_clean]

        assert not (set(sha_val_clean.tolist()) & train_sha_set), "cleaned val still overlaps train"
        assert len(set(sha_val_clean.tolist())) == len(sha_val_clean), "cleaned val still has internal duplicates"

        log(
            f"seed={seed} balanced_pool_n={len(balanced_idx)} train_n={len(train_local)} "
            f"val_orig_n={len(val_local)} (benign={int((y_val_orig==0).sum())}, malware={int((y_val_orig==1).sum())}) "
            f"train_overlap_removed={int(overlap_mask.sum())} internal_dup_removed={internal_dup_removed_count} "
            f"val_clean_n={len(val_local_clean)} (benign={int((y_val_clean==0).sum())}, malware={int((y_val_clean==1).sum())}) "
            f"train_internal_label_conflicts={train_conflicts} overlap_label_conflicts={overlap_label_conflicts} "
            f"internal_dup_label_conflicts={internal_dup_conflict_count}"
        )

        for group in GROUPS:
            X_pool = pool.group(group)
            X_train_fit_raw = X_pool[train_abs]
            X_val_orig_raw = X_pool[val_abs]
            X_val_clean_raw = X_pool[val_abs_clean]

            scaler = mu.fit_scaler(X_train_fit_raw)
            X_train_scaled = mu.transform_with_scaler(scaler, X_train_fit_raw)
            clf = mu.fit_lightgbm(X_train_scaled, y_train_fit, seed=seed, n_jobs=2)

            X_val_orig_scaled = mu.transform_with_scaler(scaler, X_val_orig_raw)
            X_val_clean_scaled = mu.transform_with_scaler(scaler, X_val_clean_raw)

            imp_orig = mu.permutation_importance_scores(
                clf, X_val_orig_scaled, y_val_orig, seed=seed, n_repeats=PERMUTATION_REPEATS, n_jobs=8,
            )
            imp_clean = mu.permutation_importance_scores(
                clf, X_val_clean_scaled, y_val_clean, seed=seed, n_repeats=PERMUTATION_REPEATS, n_jobs=8,
            )

            val_auc_orig = mu.auc_score(y_val_orig, mu.positive_proba(clf, X_val_orig_scaled))
            val_auc_clean = mu.auc_score(y_val_clean, mu.positive_proba(clf, X_val_clean_scaled))

            out_path = RESULTS_DIR / f"importance_seed{seed}_{group}.npz"
            np.savez_compressed(
                out_path,
                importance_orig=imp_orig,
                importance_clean=imp_clean,
                train_abs=train_abs,
                val_abs=val_abs,
                val_abs_clean=val_abs_clean,
            )

            log(
                f"  group={group} val_auc_orig={val_auc_orig:.4f} val_auc_clean={val_auc_clean:.4f} "
                f"-> saved {out_path.name}"
            )

            all_records.append({
                "seed": seed,
                "group": group,
                "n_balanced_pool": int(len(balanced_idx)),
                "n_train_fit": int(len(train_local)),
                "n_val_orig": int(len(val_local)),
                "n_val_orig_benign": int((y_val_orig == 0).sum()),
                "n_val_orig_malware": int((y_val_orig == 1).sum()),
                "n_train_overlap_removed": int(overlap_mask.sum()),
                "n_internal_dup_removed": int(internal_dup_removed_count),
                "n_val_clean": int(len(val_local_clean)),
                "n_val_clean_benign": int((y_val_clean == 0).sum()),
                "n_val_clean_malware": int((y_val_clean == 1).sum()),
                "train_internal_label_conflicts": int(train_conflicts),
                "overlap_label_conflicts": int(overlap_label_conflicts),
                "internal_dup_label_conflicts": int(internal_dup_conflict_count),
                "val_auc_orig": val_auc_orig,
                "val_auc_clean": val_auc_clean,
            })

            (RESULTS_DIR / "step1_summary.json").write_text(json.dumps(all_records, indent=2), encoding="utf-8")

    log("STEP1_DONE")


if __name__ == "__main__":
    main()
