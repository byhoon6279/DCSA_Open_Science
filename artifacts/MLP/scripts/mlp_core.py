#!/usr/bin/env python3
from __future__ import annotations

import csv
import gc
import hashlib
import json
import random
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "torch is required for the MLP journal runners. Use the overlap environment or install torch first."
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[3]
_SHARED_LIB_DIR = REPO_ROOT / "artifacts" / "shared"
if str(_SHARED_LIB_DIR) not in sys.path:
    sys.path.insert(0, str(_SHARED_LIB_DIR))

from common import (  # type: ignore
    balance_samples,
    feature_rankings,
    labels_array,
    load_samples_from_jsonl,
    load_samples_from_paths,
    select_feature_group,
    stabilize_features,
    summarize,
    week_paths,
)


FEATURE_GROUPS = ["all", "header", "section", "imports", "strings"]
DENSITY_BINS = ["high_density", "mid_density", "low_density"]


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


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


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

    val_set = set(val_indices.tolist())
    train_indices = np.asarray(
        sorted(idx for idx in range(len(samples)) if idx not in val_set),
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


def make_eval_loader(X: np.ndarray, y: np.ndarray, batch_size: int) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(np.asarray(X, dtype=np.float32)),
        torch.from_numpy(np.asarray(y, dtype=np.float32)),
    )
    return DataLoader(dataset, batch_size=int(batch_size), shuffle=False, drop_last=False)


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
        "acc": float(accuracy_score(y_true, pred_np)),
        "f1": float(f1_score(y_true, pred_np, zero_division=0)),
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


def js_divergence_feature_mass(X_raw: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    a = np.asarray(X_raw, dtype=float)
    a = np.clip(a, 0.0, None)
    c0 = a[y == 0].sum(axis=0)
    c1 = a[y == 1].sum(axis=0)
    p = c0 + eps
    q = c1 + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = float(np.sum(p * (np.log(p) - np.log(m))))
    kl_qm = float(np.sum(q * (np.log(q) - np.log(m))))
    return 0.5 * (kl_pm + kl_qm)


def safe_silhouette(X: np.ndarray, y: np.ndarray) -> float:
    if len(X) < 3 or np.unique(y).size < 2:
        return float("nan")
    try:
        return float(silhouette_score(X, y, metric="euclidean"))
    except ValueError:
        return float("nan")


def compute_mix_at_k_values(X: np.ndarray, y: np.ndarray, k_values: Sequence[int]) -> Dict[int, float]:
    if not k_values:
        return {}
    if len(X) <= 1:
        return {int(k): 0.0 for k in k_values}
    max_k = max(int(k) for k in k_values)
    effective_max_k = max(1, min(max_k, len(X) - 1))
    nn = NearestNeighbors(n_neighbors=effective_max_k + 1, metric="euclidean")
    nn.fit(X)
    _, indices = nn.kneighbors(X)
    neighbor_labels = y[indices[:, 1:]]
    return {
        int(k): float((neighbor_labels[:, : max(1, min(int(k), len(X) - 1))] != y[:, None]).mean())
        for k in k_values
    }


def compute_density_scores(X_train_scaled: np.ndarray, X_test_scaled: np.ndarray, k_density: int) -> np.ndarray:
    effective_k = max(1, min(k_density, len(X_train_scaled)))
    nn = NearestNeighbors(n_neighbors=effective_k, metric="cosine")
    nn.fit(X_train_scaled)
    distances, _ = nn.kneighbors(X_test_scaled)
    return -distances.mean(axis=1)


def assign_density_bins(scores: np.ndarray) -> tuple[np.ndarray, Dict[str, float]]:
    q25 = float(np.quantile(scores, 0.25))
    q75 = float(np.quantile(scores, 0.75))
    bins = np.full(len(scores), "mid_density", dtype=object)
    bins[scores <= q25] = "low_density"
    bins[scores >= q75] = "high_density"
    return bins, {"q25": q25, "q75": q75}


def evaluate_flip_metrics(
    baseline_proba: np.ndarray,
    perturbed_proba: np.ndarray,
    y_true: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    baseline_pred = (baseline_proba >= threshold).astype(int)
    perturbed_pred = (perturbed_proba >= threshold).astype(int)
    flips = baseline_pred != perturbed_pred
    benign_mask = y_true == 0
    malware_mask = y_true == 1
    benign_to_malware = flips & benign_mask & (perturbed_pred == 1)
    malware_to_benign = flips & malware_mask & (perturbed_pred == 0)
    return {
        "flip_rate": float(flips.mean()) if len(y_true) else float("nan"),
        "benign_to_malware_rate_overall": float(benign_to_malware.mean()) if len(y_true) else float("nan"),
        "malware_to_benign_rate_overall": float(malware_to_benign.mean()) if len(y_true) else float("nan"),
        "mean_abs_probability_shift": float(np.mean(np.abs(perturbed_proba - baseline_proba))) if len(y_true) else float("nan"),
        "mean_signed_probability_shift": float(np.mean(perturbed_proba - baseline_proba)) if len(y_true) else float("nan"),
    }


def pooled_train_test_samples(config: dict) -> tuple[Dict[int, List[object]], Dict[int, List[object]]]:
    data_root = Path(config["data_root"])
    train_paths = week_paths(data_root, str(config.get("platform", "Win32")), config["train_weeks"], "train")
    all_train_samples = load_samples_from_paths(train_paths, packer_filter=str(config.get("packer_filter", "all")))

    pooled_test_samples: List[object] = []
    for week in config["test_weeks"]:
        test_path = week_paths(data_root, str(config.get("platform", "Win32")), [week], "test")[0]
        pooled_test_samples.extend(
            load_samples_from_jsonl(test_path, packer_filter=str(config.get("packer_filter", "all")))
        )

    if bool(config.get("balance_train", True)):
        train_by_seed = {
            int(seed): balance_samples(
                all_train_samples,
                seed=int(seed),
                max_per_class=config.get("max_train_per_class"),
            )
            for seed in config["seeds"]
        }
    else:
        train_by_seed = {int(seed): list(all_train_samples) for seed in config["seeds"]}

    if bool(config.get("balance_test", False)):
        test_by_seed = {
            int(seed): balance_samples(
                pooled_test_samples,
                seed=int(seed),
                max_per_class=config.get("max_test_per_class"),
            )
            for seed in config["seeds"]
        }
    else:
        test_by_seed = {int(seed): list(pooled_test_samples) for seed in config["seeds"]}

    del all_train_samples
    del pooled_test_samples
    gc.collect()
    return train_by_seed, test_by_seed


def seed_split_subsets(
    train_samples: Sequence[object],
    seed: int,
    validation_fraction: float,
) -> dict[str, object]:
    y_train_full = labels_array(train_samples)
    train_idx, val_idx = split_train_validation_indices(
        samples=train_samples,
        y_train=y_train_full,
        seed=seed,
        validation_fraction=validation_fraction,
    )
    train_subset = [train_samples[i] for i in train_idx]
    val_subset = [train_samples[i] for i in val_idx]
    train_sha = {getattr(sample, "sha256", None) for sample in train_subset}
    val_sha = {getattr(sample, "sha256", None) for sample in val_subset}
    return {
        "train_idx": train_idx,
        "val_idx": val_idx,
        "train_subset": train_subset,
        "val_subset": val_subset,
        "y_train": labels_array(train_subset),
        "y_val": labels_array(val_subset),
        "sha_overlap_count": int(len((train_sha & val_sha) - {None})),
    }


def train_group_model(
    *,
    train_subset: Sequence[object],
    val_subset: Sequence[object],
    group: str,
    config: dict,
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    y_train = labels_array(train_subset)
    y_val = labels_array(val_subset)
    X_train_raw = select_feature_group(train_subset, group)
    X_val_raw = select_feature_group(val_subset, group)
    scaler = fit_scaler(X_train_raw, float(config.get("safe_std_eps", 1e-8)))
    X_train_scaled = transform_with_scaler(scaler, X_train_raw)
    X_val_scaled = transform_with_scaler(scaler, X_val_raw)
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
    return {
        "model": model,
        "history": history,
        "scaler": scaler,
        "X_train_raw": X_train_raw,
        "X_train_scaled": X_train_scaled,
        "X_val_raw": X_val_raw,
        "X_val_scaled": X_val_scaled,
        "y_train": y_train,
        "y_val": y_val,
    }


def aggregate_rows(
    rows: List[Dict[str, object]],
    group_keys: Sequence[str],
    metric_keys: Sequence[str],
) -> List[Dict[str, object]]:
    grouped: Dict[tuple, List[Dict[str, object]]] = {}
    for row in rows:
        key = tuple(row[key_name] for key_name in group_keys)
        grouped.setdefault(key, []).append(row)

    out: List[Dict[str, object]] = []
    for key, bucket in sorted(grouped.items()):
        row: Dict[str, object] = {name: value for name, value in zip(group_keys, key)}
        row["n_rows"] = len(bucket)
        for metric in metric_keys:
            values = [float(item[metric]) for item in bucket if np.isfinite(float(item[metric]))]
            if not values:
                row[f"{metric}_mean"] = float("nan")
                row[f"{metric}_std"] = float("nan")
                row[f"{metric}_min"] = float("nan")
                row[f"{metric}_max"] = float("nan")
                continue
            stats = summarize(values)
            row[f"{metric}_mean"] = stats["mean"]
            row[f"{metric}_std"] = stats["std"]
            row[f"{metric}_min"] = stats["min"]
            row[f"{metric}_max"] = stats["max"]
        out.append(row)
    return out


def device_from_arg(device_name: str) -> torch.device:
    return torch.device("cuda" if device_name == "cuda" and torch.cuda.is_available() else "cpu")

