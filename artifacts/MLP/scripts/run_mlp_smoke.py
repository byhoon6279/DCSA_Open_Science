#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import random
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ModuleNotFoundError as exc:  # pragma: no cover - dependency check
    raise ModuleNotFoundError(
        "torch is required for run_mlp_smoke.py. Use the overlap environment or install torch first."
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[3]
_SHARED_LIB_DIR = REPO_ROOT / "artifacts" / "shared"
if str(_SHARED_LIB_DIR) not in sys.path:
    sys.path.insert(0, str(_SHARED_LIB_DIR))

from common import (  # type: ignore
    balance_samples,
    labels_array,
    load_samples_from_jsonl,
    load_samples_from_paths,
    select_feature_group,
    stabilize_features,
    week_paths,
)


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "smoke" / "mlp_wild_b_smoke.template.json"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "results" / "smoke" / "mlp_wild_b_5seed"
DEFAULT_MASKING_STRENGTHS = [0.01]
DEFAULT_RANDOM_CONTROL_REPEATS = 1


@dataclass
class SeedRunArtifacts:
    seed: int
    rows: List[Dict[str, object]]
    warnings: List[Dict[str, object]]
    checks: Dict[str, object]


class TabularMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int], dropout: float) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, int(hidden_dim)))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(float(dropout)))
            prev_dim = int(hidden_dim)
        self.backbone = nn.Sequential(*layers)
        self.output = nn.Linear(prev_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(x)
        return self.output(hidden).squeeze(-1)


def write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def resolve_config(config_path: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    data_root = Path(config["data_root"])
    if not data_root.is_absolute():
        config["data_root"] = str((config_path.parent / data_root).resolve())
    return config


def ensure_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def configure_torch_determinism(enabled: bool) -> None:
    if not enabled:
        return
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def split_train_validation_indices(
    *,
    samples: Sequence[object],
    y_train: np.ndarray,
    seed: int,
    validation_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    grouped_indices: Dict[str, List[int]] = {}
    grouped_labels: Dict[str, int] = {}
    for idx, sample in enumerate(samples):
        sha = str(getattr(sample, "sha256", f"row_{idx}"))
        grouped_indices.setdefault(sha, []).append(idx)
        grouped_labels.setdefault(sha, int(y_train[idx]))

    target_val_size = max(1, int(round(len(samples) * float(validation_fraction))))
    rng = np.random.default_rng(seed)
    val_group_keys: set[str] = set()

    for label in sorted(set(grouped_labels.values())):
        label_groups = [key for key, group_label in grouped_labels.items() if group_label == label]
        rng.shuffle(label_groups)
        label_target = int(round(target_val_size * float(np.mean(y_train == label))))
        taken = 0
        for key in label_groups:
            if taken >= label_target and len(val_group_keys) > 0:
                break
            val_group_keys.add(key)
            taken += len(grouped_indices[key])

    val_indices = np.asarray(
        sorted(idx for key in val_group_keys for idx in grouped_indices[key]),
        dtype=int,
    )
    if val_indices.size == 0 or val_indices.size >= len(samples):
        indices = np.arange(len(samples))
        train_idx, val_idx = train_test_split(
            indices,
            test_size=validation_fraction,
            random_state=seed,
            stratify=y_train,
        )
        return np.asarray(train_idx), np.asarray(val_idx)

    train_indices = np.asarray(
        sorted(idx for idx in range(len(samples)) if idx not in set(val_indices.tolist())),
        dtype=int,
    )
    return train_indices, val_indices


def fit_scaler(X_train_raw: np.ndarray, safe_std_eps: float) -> dict[str, np.ndarray]:
    stabilized = stabilize_features(X_train_raw)
    mean = np.asarray(stabilized.mean(axis=0), dtype=np.float64)
    std = np.asarray(stabilized.std(axis=0), dtype=np.float64)
    safe_std = np.maximum(std, float(safe_std_eps))
    near_zero_mask = std < float(safe_std_eps)
    return {
        "mean": mean,
        "std": std,
        "safe_std": safe_std,
        "near_zero_mask": near_zero_mask,
        "safe_std_eps": float(safe_std_eps),
    }


def transform_with_scaler(scaler: dict[str, np.ndarray], X_raw: np.ndarray) -> np.ndarray:
    stabilized = stabilize_features(X_raw)
    centered = stabilized - scaler["mean"]
    scaled = centered / scaler["safe_std"]
    scaled[:, scaler["near_zero_mask"]] = centered[:, scaler["near_zero_mask"]]
    scaled = np.clip(np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0), -100.0, 100.0)
    return np.asarray(scaled, dtype=np.float32)


def apply_raw_zero_mask_then_scale(
    X_raw: np.ndarray,
    scaler: dict[str, np.ndarray],
    indices: np.ndarray,
) -> np.ndarray:
    masked = np.asarray(X_raw, dtype=np.float64).copy()
    if indices.size > 0:
        masked[:, indices] = 0.0
    return transform_with_scaler(scaler, masked)


def positive_proba_from_logits(logits: np.ndarray) -> np.ndarray:
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))
    return np.asarray(probs, dtype=np.float64)


def auc_or_nan(y_true: np.ndarray, scores: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    *,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(np.asarray(X, dtype=np.float32)),
        torch.from_numpy(np.asarray(y, dtype=np.float32)),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=shuffle,
        generator=generator,
        drop_last=False,
    )


def make_eval_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(np.asarray(X, dtype=np.float32)),
        torch.from_numpy(np.asarray(y, dtype=np.float32)),
    )
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        drop_last=False,
    )


def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    y_true: np.ndarray,
    device: torch.device,
    loss_fn: nn.Module,
) -> dict[str, object]:
    model.eval()
    logits_parts: List[np.ndarray] = []
    total_loss = 0.0
    n_rows = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            logits_parts.append(logits.detach().cpu().numpy())
            total_loss += float(loss.item()) * int(yb.numel())
            n_rows += int(yb.numel())
    logits_np = np.concatenate(logits_parts) if logits_parts else np.zeros(len(y_true), dtype=np.float32)
    proba_np = positive_proba_from_logits(logits_np)
    pred_np = (proba_np >= 0.5).astype(int)
    return {
        "loss": float(total_loss / max(1, n_rows)),
        "logits": logits_np,
        "proba": proba_np,
        "pred": pred_np,
        "auc": auc_or_nan(np.asarray(y_true), proba_np),
    }


def evaluate_model(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    device: torch.device,
    loss_fn: nn.Module,
) -> dict[str, object]:
    loader = make_eval_loader(X, y, batch_size)
    return evaluate_loader(model, loader, np.asarray(y), device, loss_fn)


def train_one_model(
    *,
    input_dim: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: dict,
    seed: int,
    device: torch.device,
) -> tuple[nn.Module, dict[str, object]]:
    ensure_seed(seed)
    model = TabularMLP(input_dim=input_dim, hidden_dims=config["hidden_dims"], dropout=float(config["dropout"]))
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    loss_fn = nn.BCEWithLogitsLoss()
    train_loader = make_loader(X_train, y_train, int(config["batch_size"]), shuffle=True, seed=seed)
    val_loader = make_eval_loader(X_val, y_val, int(config["batch_size"]))

    best_state = deepcopy(model.state_dict())
    best_val_auc = float("-inf")
    best_epoch = -1
    patience = int(config["early_stopping_patience"])
    max_epochs = int(config["max_epochs"])
    patience_counter = 0
    epoch_rows: List[Dict[str, object]] = []

    for epoch in range(max_epochs):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.item()) * int(yb.numel())
            train_count += int(yb.numel())

        epoch_train_loss = float(train_loss_sum / max(1, train_count))
        val_metrics = evaluate_loader(model, val_loader, y_val, device, loss_fn)
        current_val_auc = float(val_metrics["auc"])
        epoch_rows.append(
            {
                "epoch": epoch + 1,
                "train_loss": epoch_train_loss,
                "val_loss": val_metrics["loss"],
                "val_auc": current_val_auc,
            }
        )

        if np.isfinite(current_val_auc) and current_val_auc > best_val_auc:
            best_val_auc = current_val_auc
            best_epoch = epoch + 1
            best_state = deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_state)
    history = {
        "epochs_ran": len(epoch_rows),
        "best_epoch": best_epoch,
        "best_val_auc": best_val_auc,
        "early_stopped": len(epoch_rows) < max_epochs,
        "epoch_rows": epoch_rows,
        "last_train_loss": epoch_rows[-1]["train_loss"] if epoch_rows else float("nan"),
        "last_val_loss": epoch_rows[-1]["val_loss"] if epoch_rows else float("nan"),
    }
    return model, history


def compute_logit_gradient_importance(
    model: nn.Module,
    X_val_scaled: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    abs_grad_sum = np.zeros(X_val_scaled.shape[1], dtype=np.float64)
    n_rows = 0
    loader = DataLoader(torch.from_numpy(np.asarray(X_val_scaled, dtype=np.float32)), batch_size=int(batch_size), shuffle=False)
    for xb in loader:
        xb = xb.to(device)
        xb.requires_grad_(True)
        logits = model(xb)
        logits.sum().backward()
        grads = xb.grad.detach().cpu().numpy()
        abs_grad_sum += np.abs(grads).sum(axis=0)
        n_rows += grads.shape[0]
        model.zero_grad(set_to_none=True)
    if n_rows == 0:
        return np.zeros(X_val_scaled.shape[1], dtype=np.float32)
    return np.asarray(abs_grad_sum / float(n_rows), dtype=np.float32)


def top_mask_indices(scores: np.ndarray, strength: float) -> np.ndarray:
    n_features = len(scores)
    n_mask = max(1, int(np.ceil(n_features * float(strength))))
    ranked = np.argsort(-scores, kind="stable")
    return np.asarray(ranked[:n_mask], dtype=int)


def random_mask_indices(n_features: int, n_mask: int, seed: int, group: str, strength: float, repeat: int) -> np.ndarray:
    seed_material = f"{seed}:{group}:{strength:.5f}:{repeat}".encode("utf-8")
    digest = hashlib.sha256(seed_material).hexdigest()
    rng_seed = int(digest[:16], 16) % (2**32)
    rng = np.random.default_rng(rng_seed)
    return np.asarray(rng.choice(n_features, size=n_mask, replace=False), dtype=int)


def summarize_masking_check(
    *,
    model: nn.Module,
    scaler: dict[str, np.ndarray],
    X_test_raw: np.ndarray,
    baseline_pred: np.ndarray,
    importance_scores: np.ndarray,
    config: dict,
    group: str,
    seed: int,
    batch_size: int,
    device: torch.device,
    loss_fn: nn.Module,
    baseline_proba: np.ndarray,
    baseline_auc: float,
    y_test: np.ndarray,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    strengths = [float(v) for v in config.get("smoke_masking_strengths", DEFAULT_MASKING_STRENGTHS)]
    random_repeats = int(config.get("smoke_random_control_repeats", DEFAULT_RANDOM_CONTROL_REPEATS))
    dummy_y = np.zeros(len(X_test_raw), dtype=np.float32)

    for strength in strengths:
        important_idx = top_mask_indices(importance_scores, strength)
        important_scaled = apply_raw_zero_mask_then_scale(X_test_raw, scaler, important_idx)
        important_metrics = evaluate_model(model, important_scaled, dummy_y, batch_size, device, loss_fn)
        important_flip_rate = float(np.mean(important_metrics["pred"] != baseline_pred))
        important_proba = np.asarray(important_metrics["proba"])
        important_pred = np.asarray(important_metrics["pred"])
        rows.append(
            {
                "seed": seed,
                "feature_group": group,
                "mask_type": "important",
                "strength": strength,
                "repeat": 0,
                "n_masked_features": int(important_idx.size),
                "flip_rate": important_flip_rate,
                "baseline_auc": baseline_auc,
                "masked_auc": auc_or_nan(y_test, important_proba),
                "mean_probability_shift": float(np.mean(np.abs(important_proba - baseline_proba))),
                "malware_to_benign_flip_rate": float(np.mean((baseline_pred == 1) & (important_pred == 0))),
                "benign_to_malware_flip_rate": float(np.mean((baseline_pred == 0) & (important_pred == 1))),
                "mean_probability": float(np.mean(important_metrics["proba"])),
            }
        )
        for repeat in range(1, random_repeats + 1):
            random_idx = random_mask_indices(X_test_raw.shape[1], int(important_idx.size), seed, group, strength, repeat)
            random_scaled = apply_raw_zero_mask_then_scale(X_test_raw, scaler, random_idx)
            random_metrics = evaluate_model(model, random_scaled, dummy_y, batch_size, device, loss_fn)
            random_proba = np.asarray(random_metrics["proba"])
            random_pred = np.asarray(random_metrics["pred"])
            rows.append(
                {
                    "seed": seed,
                    "feature_group": group,
                    "mask_type": "random",
                    "strength": strength,
                    "repeat": repeat,
                    "n_masked_features": int(random_idx.size),
                    "flip_rate": float(np.mean(random_metrics["pred"] != baseline_pred)),
                    "baseline_auc": baseline_auc,
                    "masked_auc": auc_or_nan(y_test, random_proba),
                    "mean_probability_shift": float(np.mean(np.abs(random_proba - baseline_proba))),
                    "malware_to_benign_flip_rate": float(np.mean((baseline_pred == 1) & (random_pred == 0))),
                    "benign_to_malware_flip_rate": float(np.mean((baseline_pred == 0) & (random_pred == 1))),
                    "mean_probability": float(np.mean(random_metrics["proba"])),
                }
            )
    return rows


def finite_scalar(value: object) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def run_for_seed(
    *,
    seed: int,
    config: dict,
    train_samples: Sequence[object],
    test_samples: Sequence[object],
    device: torch.device,
) -> SeedRunArtifacts:
    ensure_seed(seed)
    y_train_full = labels_array(train_samples)
    y_test = labels_array(test_samples)
    train_idx, val_idx = split_train_validation_indices(
        samples=train_samples,
        y_train=y_train_full,
        seed=seed,
        validation_fraction=float(config["validation_fraction"]),
    )
    train_subset = [train_samples[i] for i in train_idx]
    val_subset = [train_samples[i] for i in val_idx]
    y_train = labels_array(train_subset)
    y_val = labels_array(val_subset)

    warning_rows: List[Dict[str, object]] = []
    metric_rows: List[Dict[str, object]] = []
    masking_rows: List[Dict[str, object]] = []
    group_checks: Dict[str, object] = {}
    loss_fn = nn.BCEWithLogitsLoss()

    for group in list(config["feature_groups"]):
        X_train_raw = select_feature_group(train_subset, group)
        X_val_raw = select_feature_group(val_subset, group)
        X_test_raw = select_feature_group(test_samples, group)

        scaler = fit_scaler(X_train_raw, float(config.get("safe_std_eps", 1e-8)))
        X_train_scaled = transform_with_scaler(scaler, X_train_raw)
        X_val_scaled = transform_with_scaler(scaler, X_val_raw)
        X_test_scaled = transform_with_scaler(scaler, X_test_raw)

        model, history = train_one_model(
            input_dim=X_train_scaled.shape[1],
            X_train=X_train_scaled,
            y_train=y_train,
            X_val=X_val_scaled,
            y_val=y_val,
            config=config,
            seed=seed,
            device=device,
        )
        val_metrics = evaluate_model(model, X_val_scaled, y_val, int(config["batch_size"]), device, loss_fn)
        test_metrics = evaluate_model(model, X_test_scaled, y_test, int(config["batch_size"]), device, loss_fn)
        importance_scores = compute_logit_gradient_importance(model, X_val_scaled, int(config["batch_size"]), device)
        masking_rows.extend(
            summarize_masking_check(
                model=model,
                scaler=scaler,
                X_test_raw=X_test_raw,
                baseline_pred=np.asarray(test_metrics["pred"]),
                importance_scores=importance_scores,
                config=config,
                group=group,
                seed=seed,
                batch_size=int(config["batch_size"]),
                device=device,
                loss_fn=loss_fn,
                baseline_proba=np.asarray(test_metrics["proba"]),
                baseline_auc=float(test_metrics["auc"]),
                y_test=y_test,
            )
        )

        probability_std = float(np.std(np.asarray(test_metrics["proba"])))
        unique_pred = int(np.unique(np.asarray(test_metrics["pred"])).size)
        lr_reference_auc = config.get("lr_reference_auc")
        if unique_pred < 2:
            warning_rows.append(
                {
                    "seed": seed,
                    "feature_group": group,
                    "warning_type": "single_class_hard_prediction",
                    "detail": "threshold-0.5 hard predictions collapsed to one class",
                }
            )
        if lr_reference_auc is not None and finite_scalar(lr_reference_auc):
            if finite_scalar(test_metrics["auc"]) and float(test_metrics["auc"]) < float(lr_reference_auc) - 0.05:
                warning_rows.append(
                    {
                        "seed": seed,
                        "feature_group": group,
                        "warning_type": "auc_below_lr_reference",
                        "detail": f"test_auc={float(test_metrics['auc']):.6f} lr_reference_auc={float(lr_reference_auc):.6f}",
                    }
                )

        metric_rows.append(
            {
                "seed": seed,
                "feature_group": group,
                "input_dim": int(X_train_scaled.shape[1]),
                "train_size": int(len(train_subset)),
                "val_size": int(len(val_subset)),
                "test_size": int(len(test_samples)),
                "train_loss_last": float(history["last_train_loss"]),
                "val_loss_last": float(history["last_val_loss"]),
                "best_val_auc": float(history["best_val_auc"]),
                "best_epoch": int(history["best_epoch"]),
                "epochs_ran": int(history["epochs_ran"]),
                "early_stopped": bool(history["early_stopped"]),
                "val_auc": float(val_metrics["auc"]),
                "test_auc": float(test_metrics["auc"]),
                "test_probability_std": probability_std,
                "test_positive_rate": float(np.mean(np.asarray(test_metrics["pred"]))),
                "test_unique_hard_predictions": unique_pred,
                "importance_mean_abs_gradient_mean": float(np.mean(importance_scores)),
                "importance_mean_abs_gradient_max": float(np.max(importance_scores)),
                "safe_std_eps": float(config.get("safe_std_eps", 1e-8)),
            }
        )
        group_checks[group] = {
            "history": history,
            "train_val_disjoint": bool(set(train_idx.tolist()).isdisjoint(set(val_idx.tolist()))),
            "test_probability_std": probability_std,
            "single_class_hard_prediction": unique_pred < 2,
            "masking_rows_added": len([row for row in masking_rows if row["feature_group"] == group]),
        }

    train_sha = {getattr(sample, "sha256", None) for sample in train_subset}
    val_sha = {getattr(sample, "sha256", None) for sample in val_subset}
    test_sha = {getattr(sample, "sha256", None) for sample in test_samples}
    checks = {
        "seed": seed,
        "train_val_disjoint_by_index": bool(set(train_idx.tolist()).isdisjoint(set(val_idx.tolist()))),
        "train_val_sha_overlap_count": int(len((train_sha & val_sha) - {None})),
        "train_test_sha_overlap_count": int(len((train_sha & test_sha) - {None})),
        "val_test_sha_overlap_count": int(len((val_sha & test_sha) - {None})),
        "group_checks": group_checks,
    }
    return SeedRunArtifacts(
        seed=seed,
        rows=metric_rows + masking_rows,
        warnings=warning_rows,
        checks=checks,
    )


def validate_smoke(all_metric_rows: List[Dict[str, object]], all_warning_rows: List[Dict[str, object]], checks_by_seed: List[Dict[str, object]]) -> dict:
    summary_rows = [row for row in all_metric_rows if "test_auc" in row]
    hard_failures = {
        "has_metric_rows": len(summary_rows) > 0,
        "all_train_loss_finite": all(finite_scalar(row["train_loss_last"]) for row in summary_rows),
        "all_val_loss_finite": all(finite_scalar(row["val_loss_last"]) for row in summary_rows),
        "all_val_auc_finite": all(finite_scalar(row["val_auc"]) for row in summary_rows),
        "all_test_auc_finite": all(finite_scalar(row["test_auc"]) for row in summary_rows),
        "all_probability_std_nonconstant": all(float(row["test_probability_std"]) > 0.0 for row in summary_rows),
        "all_best_epoch_valid": all(int(row["best_epoch"]) >= 1 for row in summary_rows),
        "all_train_val_disjoint": all(bool(seed_check["train_val_disjoint_by_index"]) for seed_check in checks_by_seed),
        "no_train_test_sha_overlap": all(int(seed_check["train_test_sha_overlap_count"]) == 0 for seed_check in checks_by_seed),
        "masking_rows_present": any("flip_rate" in row for row in all_metric_rows),
    }
    warnings_summary = {
        "single_class_hard_prediction_count": sum(
            row["warning_type"] == "single_class_hard_prediction" for row in all_warning_rows
        ),
        "auc_below_lr_reference_count": sum(
            row["warning_type"] == "auc_below_lr_reference" for row in all_warning_rows
        ),
    }
    return {
        "passed": all(hard_failures.values()),
        "hard_failures": hard_failures,
        "warnings": warnings_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the journal MLP smoke test.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    config = resolve_config(args.config)
    configure_torch_determinism(bool(config.get("torch_deterministic", True)))
    data_root = Path(config["data_root"])
    if not data_root.exists():
        raise FileNotFoundError(f"Configured data_root does not exist: {data_root}")

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    train_paths = week_paths(data_root, str(config.get("platform", "Win32")), config["train_weeks"], "train")
    all_train_samples = load_samples_from_paths(train_paths, packer_filter=str(config.get("packer_filter", "all")))

    pooled_test_samples: List[object] = []
    for week in config["test_weeks"]:
        test_path = week_paths(data_root, str(config.get("platform", "Win32")), [week], "test")[0]
        pooled_test_samples.extend(
            load_samples_from_jsonl(test_path, packer_filter=str(config.get("packer_filter", "all")))
        )

    if bool(config.get("balance_train", True)):
        balanced_train_by_seed = {
            int(seed): balance_samples(
                all_train_samples,
                seed=int(seed),
                max_per_class=config.get("max_train_per_class"),
            )
            for seed in config["seeds"]
        }
    else:
        balanced_train_by_seed = {int(seed): list(all_train_samples) for seed in config["seeds"]}

    if bool(config.get("balance_test", False)):
        balanced_test_by_seed = {
            int(seed): balance_samples(
                pooled_test_samples,
                seed=int(seed),
                max_per_class=config.get("max_test_per_class"),
            )
            for seed in config["seeds"]
        }
    else:
        balanced_test_by_seed = {int(seed): list(pooled_test_samples) for seed in config["seeds"]}

    # Free the raw full-period pools now that every seed's balanced subset is built;
    # this is the largest chunk of resident memory and nothing below needs it anymore.
    del all_train_samples
    del pooled_test_samples
    gc.collect()

    all_rows: List[Dict[str, object]] = []
    all_warning_rows: List[Dict[str, object]] = []
    checks_by_seed: List[Dict[str, object]] = []
    for seed in [int(seed) for seed in config["seeds"]]:
        artifacts = run_for_seed(
            seed=seed,
            config=config,
            train_samples=balanced_train_by_seed[seed],
            test_samples=balanced_test_by_seed[seed],
            device=device,
        )
        all_rows.extend(artifacts.rows)
        all_warning_rows.extend(artifacts.warnings)
        checks_by_seed.append(artifacts.checks)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows = [row for row in all_rows if "test_auc" in row]
    masking_rows = [row for row in all_rows if "flip_rate" in row]

    if metric_rows:
        write_csv(args.output_dir / "seed_metrics.csv", list(metric_rows[0].keys()), metric_rows)
    if masking_rows:
        write_csv(args.output_dir / "masking_smoke_rows.csv", list(masking_rows[0].keys()), masking_rows)
    if all_warning_rows:
        write_csv(args.output_dir / "diagnostic_warnings.csv", list(all_warning_rows[0].keys()), all_warning_rows)

    smoke_check = validate_smoke(metric_rows + masking_rows, all_warning_rows, checks_by_seed)
    (args.output_dir / "smoke_check.json").write_text(json.dumps(smoke_check, indent=2), encoding="utf-8")
    (args.output_dir / "checks_by_seed.json").write_text(json.dumps(checks_by_seed, indent=2), encoding="utf-8")
    (args.output_dir / "resolved_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    if not smoke_check["passed"]:
        raise SystemExit("MLP smoke test failed. See smoke_check.json for hard-failure details.")

    warning_count = sum(int(v) for v in smoke_check["warnings"].values())
    if warning_count:
        print(f"MLP smoke test passed with {warning_count} diagnostic warning(s). Results saved to {args.output_dir}")
    else:
        print(f"MLP smoke test passed. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
