from __future__ import annotations

import importlib.util
import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings(
    "ignore",
    message="divide by zero encountered in matmul",
    category=RuntimeWarning,
    module=r"sklearn\.linear_model\._linear_loss",
)
warnings.filterwarnings(
    "ignore",
    message="overflow encountered in matmul",
    category=RuntimeWarning,
    module=r"sklearn\.linear_model\._linear_loss",
)
warnings.filterwarnings(
    "ignore",
    message="invalid value encountered in matmul",
    category=RuntimeWarning,
    module=r"sklearn\.linear_model\._linear_loss",
)
warnings.filterwarnings(
    "ignore",
    message=r"X does not have valid feature names, but LGBMClassifier was fitted with feature names",
    category=UserWarning,
    module=r"sklearn\.utils\.validation",
)


REPO_ROOT = Path(__file__).resolve().parents[2]

# `common.py` is bundled alongside this package at
# `DCSA_Open_Science/artifacts/shared/common.py`. Pass `common_py=`
# explicitly to override.
KNOWN_COMMON_CANDIDATES = (
    REPO_ROOT / "artifacts" / "shared" / "common.py",
)

# Resolves to the EMBER2024 features bundled alongside this package
# (`DCSA_Open_Science/Data/...`).
KNOWN_DATA_ROOT_CANDIDATES = (
    REPO_ROOT / "Data",
)


@dataclass(frozen=True)
class SampleWithFamily:
    base_sample: Any
    family: str | None


def load_common_module(common_py: Path | None = None) -> Any:
    candidate = common_py
    if candidate is None:
        for path in KNOWN_COMMON_CANDIDATES:
            if path.exists():
                candidate = path
                break
    if candidate is None or not candidate.exists():
        raise FileNotFoundError(
            "Could not locate the shared common.py library. "
            "Pass an explicit path or edit KNOWN_COMMON_CANDIDATES."
        )
    spec = importlib.util.spec_from_file_location("ra_q4_beyond_auc_common", candidate)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to create import spec for {candidate}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def resolve_data_root(config_path: Path, configured_data_root: str) -> Path:
    data_root = Path(configured_data_root)
    if data_root.is_absolute() and data_root.exists():
        return data_root
    candidate = (config_path.parent / data_root).resolve()
    if candidate.exists():
        return candidate
    for known in KNOWN_DATA_ROOT_CANDIDATES:
        if known.exists():
            return known
    raise FileNotFoundError(
        "Could not resolve dataset feature root. Checked config-relative path "
        f"{candidate} and known candidates: {', '.join(str(path) for path in KNOWN_DATA_ROOT_CANDIDATES)}"
    )


def load_samples_with_family(common: Any, path: Path, packer_filter: str) -> list[SampleWithFamily]:
    rows: list[SampleWithFamily] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not common.matches_packer_filter(row, packer_filter):
                continue
            sample = common.build_sample(row, path)
            if sample is None:
                continue
            family = row.get("family")
            rows.append(
                SampleWithFamily(
                    base_sample=sample,
                    family=None if family in (None, "") else str(family),
                )
            )
    return rows


def fit_train_scaler(common: Any, X_train_raw: np.ndarray) -> tuple[np.ndarray, StandardScaler]:
    X_train = common.stabilize_features(X_train_raw)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_train_scaled = np.clip(np.nan_to_num(X_train_scaled, nan=0.0, posinf=0.0, neginf=0.0), -100.0, 100.0)
    return X_train_scaled, scaler


def transform_with_scaler(common: Any, X_raw: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    X = common.stabilize_features(X_raw)
    X_scaled = scaler.transform(X)
    return np.clip(np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0), -100.0, 100.0)


def fit_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    model_type: str,
    lightgbm_n_jobs: int | None = None,
) -> Any:
    if model_type == "logistic_regression":
        clf = LogisticRegression(
            solver="lbfgs",
            class_weight="balanced",
            max_iter=5000,
            random_state=seed,
        )
    elif model_type == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "lightgbm is not installed in the current Python environment. "
                "Install it (`pip install lightgbm`) or switch to a Python "
                "environment that already has it available."
            ) from exc
        clf = LGBMClassifier(
            objective="binary",
            class_weight="balanced",
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=-1 if lightgbm_n_jobs is None else int(lightgbm_n_jobs),
            verbosity=-1,
        )
    else:
        raise NotImplementedError(
            f"RA-Q4 adapter currently supports logistic_regression and lightgbm, got {model_type}."
        )
    clf.fit(np.asarray(X_train, dtype=np.float64), np.asarray(y_train))
    return clf


def positive_proba(clf: Any, X: np.ndarray) -> np.ndarray:
    proba = clf.predict_proba(np.asarray(X, dtype=np.float64))
    return proba[:, 1]


def compute_mix_at_k(X: np.ndarray, y: np.ndarray, k: int) -> float:
    if len(X) <= 1:
        return 0.0
    effective_k = max(1, min(k, len(X) - 1))
    nn = NearestNeighbors(n_neighbors=effective_k + 1, metric="euclidean")
    nn.fit(X)
    _, indices = nn.kneighbors(X)
    return float((y[indices[:, 1:]] != y[:, None]).mean())


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


def compute_same_family_rate_at_k(X: np.ndarray, families: Sequence[str], k: int) -> float:
    if len(X) <= 1:
        return 0.0
    fam = np.asarray(list(families), dtype=object)
    effective_k = max(1, min(k, len(X) - 1))
    nn = NearestNeighbors(n_neighbors=effective_k + 1, metric="euclidean")
    nn.fit(X)
    _, indices = nn.kneighbors(X)
    return float((fam[indices[:, 1:]] == fam[:, None]).mean())


class CommonLibraryAdapter:
    def __init__(
        self,
        *,
        base_rq1_config: Path,
        common_py: Path | None = None,
        model_type: str = "logistic_regression",
        mix_k: int = 10,
        show_progress: bool = False,
    ) -> None:
        self.common = load_common_module(common_py)
        self.base_rq1_config_path = base_rq1_config.resolve()
        self.base_config = json.loads(self.base_rq1_config_path.read_text(encoding="utf-8"))
        data_root = resolve_data_root(self.base_rq1_config_path, str(self.base_config["data_root"]))
        self.base_config["data_root"] = str(data_root)
        self.model_type = model_type
        self.mix_k = int(mix_k)
        self.show_progress = show_progress

        cfg = self.base_config
        self.platform = cfg["platform"]
        self.train_weeks = list(cfg["train_weeks"])
        self.test_weeks = list(cfg["test_weeks"])
        self.packer_filter = str(cfg.get("packer_filter", "all"))
        self.balance_train = bool(cfg.get("balance_train", True))
        self.balance_test = bool(cfg.get("balance_test", False))
        self.max_train_per_class = cfg.get("max_train_per_class")
        self.max_test_per_class = cfg.get("max_test_per_class")
        self.lightgbm_n_jobs = cfg.get("lightgbm_n_jobs")

        train_paths = self.common.week_paths(data_root, self.platform, self.train_weeks, "train")
        self.all_train_samples = self.common.load_samples_from_paths(train_paths, packer_filter=self.packer_filter)
        self.test_samples_by_week = {
            week: load_samples_with_family(
                self.common,
                self.common.week_paths(data_root, self.platform, [week], "test")[0],
                self.packer_filter,
            )
            for week in self.test_weeks
        }
        self.family_to_size = self._family_sizes()
        self.total_feature_count = int(sum(self.family_to_size.values()))

    def _family_sizes(self) -> dict[str, int]:
        sample = self.all_train_samples[0]
        return {
            "header": int(sample.header_vec.shape[0]),
            "section": int(sample.section_vec.shape[0]),
            "imports": int(sample.imports_vec.shape[0]),
            "strings": int(sample.strings_vec.shape[0]),
        }

    def resolve_family_indices(self) -> dict[str, list[int]]:
        return {family: list(range(size)) for family, size in self.family_to_size.items()}

    def resolve_all_feature_indices(self) -> list[int]:
        return list(range(self.total_feature_count))

    def _select_group(self, samples: Sequence[Any], family: str) -> np.ndarray:
        return self.common.select_feature_group(samples, family)

    def _run_group_subset_metrics(self, *, group: str, selected_indices: Sequence[int], seed: int) -> dict[str, float]:
        train_samples = list(self.all_train_samples)
        if self.balance_train:
            train_samples = self.common.balance_samples(
                self.all_train_samples,
                seed=seed,
                max_per_class=self.max_train_per_class,
            )
        y_train = self.common.labels_array(train_samples)
        X_train_full = self._select_group(train_samples, group)
        X_train_raw = X_train_full[:, np.asarray(selected_indices, dtype=int)]
        X_train_scaled, scaler = fit_train_scaler(self.common, X_train_raw)
        clf = fit_classifier(
            X_train_scaled,
            y_train,
            seed,
            self.model_type,
            lightgbm_n_jobs=self.lightgbm_n_jobs,
        )

        weekly_metrics: list[dict[str, float]] = []
        for week_index, week in enumerate(self.test_weeks):
            week_samples_with_family = self.test_samples_by_week[week]
            week_samples = [row.base_sample for row in week_samples_with_family]
            if self.balance_test:
                balanced = self.common.balance_samples(
                    week_samples,
                    seed=seed + week_index + 1,
                    max_per_class=self.max_test_per_class,
                )
                balanced_sha = {sample.sha256 for sample in balanced}
                week_samples_with_family = [row for row in week_samples_with_family if row.base_sample.sha256 in balanced_sha]
                week_samples = [row.base_sample for row in week_samples_with_family]

            y_test = self.common.labels_array(week_samples)
            X_test_full = self._select_group(week_samples, group)
            X_test_raw = X_test_full[:, np.asarray(selected_indices, dtype=int)]
            X_test_scaled = transform_with_scaler(self.common, X_test_raw, scaler)
            proba = positive_proba(clf, X_test_scaled)
            auc = float(roc_auc_score(y_test, proba))
            mix_at_10 = compute_mix_at_k(X_test_scaled, y_test, self.mix_k)
            js = js_divergence_feature_mass(X_test_raw, y_test)

            malware_rows = [
                (idx, row.family)
                for idx, row in enumerate(week_samples_with_family)
                if row.base_sample.label == 1 and row.family is not None
            ]
            if malware_rows:
                malware_indices = np.asarray([idx for idx, _ in malware_rows], dtype=int)
                malware_families = [family_name for _, family_name in malware_rows]
                same_family = compute_same_family_rate_at_k(X_test_scaled[malware_indices], malware_families, self.mix_k)
            else:
                same_family = float("nan")

            weekly_metrics.append(
                {
                    "auc": auc,
                    "mix_at_10": mix_at_10,
                    "js_divergence": js,
                    "same_family_rate_at_10": same_family,
                }
            )

        return {
            metric: float(np.nanmean([row[metric] for row in weekly_metrics]))
            for metric in ("auc", "mix_at_10", "js_divergence", "same_family_rate_at_10")
        }

    def run_subset_metrics(self, *, family: str, selected_indices: Sequence[int], seed: int) -> dict[str, float]:
        return self._run_group_subset_metrics(
            group=family,
            selected_indices=selected_indices,
            seed=seed,
        )

    def run_global_subset_metrics(self, *, selected_indices: Sequence[int], seed: int) -> dict[str, float]:
        return self._run_group_subset_metrics(
            group="all",
            selected_indices=selected_indices,
            seed=seed,
        )
