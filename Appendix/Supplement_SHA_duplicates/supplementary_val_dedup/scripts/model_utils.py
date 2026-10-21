"""Index-based re-implementation of the numeric core of DCSA's
run_feature_perturbation_sensitivity.py / run_prediction_flip_analysis.py,
operating on plain numpy arrays instead of Sample objects so we can track
SHA-256 identity through every split. The math (balance_samples, the 80/20
split, scaling, LightGBM hyperparameters, permutation importance, masking,
flip metrics) is copied verbatim from DCSA_Open_Science/artifacts/shared/common.py
and artifacts/5_3_targeted_perturbation_response/scripts/common/*.py (read-only
reference, not imported by mutation) to guarantee the same numeric behavior."""
from __future__ import annotations

import math
import warnings
from typing import Tuple

import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def stabilize_features(X: np.ndarray) -> np.ndarray:
    X = np.nan_to_num(np.asarray(X, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    X = np.clip(X, -1e12, 1e12)
    return np.sign(X) * np.log1p(np.abs(X))


def fit_scaler(X_train_raw: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(stabilize_features(X_train_raw))
    return scaler


def transform_with_scaler(scaler: StandardScaler, X_raw: np.ndarray) -> np.ndarray:
    X_scaled = scaler.transform(stabilize_features(X_raw))
    return np.clip(np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0), -100.0, 100.0)


def zero_scaled_template(scaler: StandardScaler, n_features: int) -> np.ndarray:
    zero_row = np.zeros((1, n_features), dtype=np.float64)
    return transform_with_scaler(scaler, zero_row)[0]


def balance_indices(y_pool: np.ndarray, seed: int, max_per_class: int | None) -> np.ndarray:
    """Reproduces common.balance_samples() exactly, but returns absolute
    pool indices instead of Sample objects. RNG call order matches the
    original: pick(benign) then pick(malware) then shuffle(balanced)."""
    benign_idx = np.flatnonzero(y_pool == 0)
    malware_idx = np.flatnonzero(y_pool == 1)
    target = min(len(benign_idx), len(malware_idx))
    if max_per_class is not None:
        target = min(target, max_per_class)
    rng = np.random.default_rng(seed)

    def pick(idx_arr: np.ndarray) -> np.ndarray:
        if len(idx_arr) <= target:
            return idx_arr.copy()
        chosen_local = rng.choice(len(idx_arr), size=target, replace=False)
        return idx_arr[chosen_local]

    picked_benign = pick(benign_idx)
    picked_malware = pick(malware_idx)
    balanced = np.concatenate([picked_benign, picked_malware])
    rng.shuffle(balanced)
    return balanced


def split_train_val(n_samples: int, y: np.ndarray, seed: int, validation_fraction: float) -> Tuple[np.ndarray, np.ndarray]:
    indices = np.arange(n_samples)
    train_idx, val_idx = train_test_split(
        indices, test_size=validation_fraction, random_state=seed, stratify=y,
    )
    return np.asarray(train_idx), np.asarray(val_idx)


def fit_lightgbm(X_train_scaled: np.ndarray, y_train: np.ndarray, seed: int, n_jobs: int = 2):
    from lightgbm import LGBMClassifier

    clf = LGBMClassifier(
        objective="binary",
        class_weight="balanced",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        n_jobs=n_jobs,
        verbosity=-1,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        clf.fit(np.asarray(X_train_scaled, dtype=np.float64), np.asarray(y_train))
    return clf


def positive_proba(clf, X: np.ndarray) -> np.ndarray:
    proba = clf.predict_proba(np.asarray(X, dtype=np.float64))
    return proba[:, 1]


def permutation_importance_scores(clf, X_val_scaled: np.ndarray, y_val: np.ndarray, seed: int, n_repeats: int, n_jobs: int = 1) -> np.ndarray:
    # NOTE: sklearn's permutation_importance seeds each (feature, repeat) draw
    # deterministically from `random_state`, independent of joblib scheduling;
    # verified empirically (n_jobs=1 vs n_jobs=4 -> bit-identical importances_mean)
    # before relying on this. n_jobs only changes wall-clock time, never the
    # numeric result, so raising it here does not affect reproduction fidelity.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
            category=UserWarning,
        )
        result = permutation_importance(
            clf,
            np.asarray(X_val_scaled, dtype=np.float64),
            np.asarray(y_val),
            scoring="roc_auc",
            n_repeats=n_repeats,
            random_state=seed,
            n_jobs=n_jobs,
        )
    return np.asarray(result.importances_mean, dtype=np.float64)


def select_top_indices(importance: np.ndarray, strength: float) -> np.ndarray:
    n_features = len(importance)
    n_mask = max(1, math.ceil(n_features * strength))
    ranked = np.argsort(-importance, kind="stable")
    return ranked[:n_mask]


def apply_zero_mask_scaled(X_scaled: np.ndarray, indices: np.ndarray, zero_scaled_values: np.ndarray) -> np.ndarray:
    masked = np.array(X_scaled, copy=True)
    masked[:, indices] = zero_scaled_values[indices]
    return masked


def apply_zero_mask_raw(X: np.ndarray, indices: np.ndarray) -> np.ndarray:
    masked = np.array(X, copy=True)
    masked[:, indices] = 0.0
    return masked


def evaluate_flip_metrics(baseline_pred_proba: np.ndarray, perturbed_pred_proba: np.ndarray, y: np.ndarray, threshold: float) -> dict:
    baseline_pred = (baseline_pred_proba >= threshold).astype(int)
    perturbed_pred = (perturbed_pred_proba >= threshold).astype(int)
    flips = baseline_pred != perturbed_pred
    benign_mask = y == 0
    malware_mask = y == 1
    benign_to_malware = flips & benign_mask & (perturbed_pred == 1)
    malware_to_benign = flips & malware_mask & (perturbed_pred == 0)
    return {
        "flip_rate": float(flips.mean()),
        "benign_to_malware_rate": float(benign_to_malware.mean()),
        "malware_to_benign_rate": float(malware_to_benign.mean()),
    }


def auc_score(y: np.ndarray, proba: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, proba))
