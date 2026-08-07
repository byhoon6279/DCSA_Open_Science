#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
from torch import nn

from mlp_core import (
    DENSITY_BINS,
    FEATURE_GROUPS,
    aggregate_rows,
    assign_density_bins,
    auc_or_nan,
    compute_density_scores,
    configure_torch_determinism,
    device_from_arg,
    evaluate_model,
    js_divergence_feature_mass,
    labels_array,
    pooled_train_test_samples,
    resolve_config,
    safe_silhouette,
    seed_split_subsets,
    select_feature_group,
    train_group_model,
    transform_with_scaler,
    write_csv,
    write_json,
)


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "5_4" / "mlp_rq3_density_reliability_wild_b_main.json"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "results" / "5_4" / "mlp_rq3_density_reliability_wild_b_main"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MLP DCSA section 5.4.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    config = resolve_config(args.config)
    configure_torch_determinism(bool(config.get("torch_deterministic", True)))
    device = device_from_arg(args.device)
    train_by_seed, test_by_seed = pooled_train_test_samples(config)
    loss_fn = nn.BCEWithLogitsLoss()
    k_density = int(config.get("k_density", 10))

    rows: List[Dict[str, object]] = []
    density_rows: List[Dict[str, object]] = []
    for seed in [int(seed) for seed in config["seeds"]]:
        split = seed_split_subsets(train_by_seed[seed], seed=seed, validation_fraction=float(config["validation_fraction"]))
        train_subset = split["train_subset"]
        val_subset = split["val_subset"]
        test_samples = test_by_seed[seed]
        y_test = labels_array(test_samples)
        for group in list(config.get("feature_groups", FEATURE_GROUPS)):
            artifact = train_group_model(
                train_subset=train_subset,
                val_subset=val_subset,
                group=group,
                config=config,
                seed=seed,
                device=device,
            )
            X_test_raw = select_feature_group(test_samples, group)
            X_test_scaled = transform_with_scaler(artifact["scaler"], X_test_raw)
            baseline = evaluate_model(
                artifact["model"],
                X_test_scaled,
                y_test,
                int(config["batch_size"]),
                device,
                loss_fn,
            )
            density_scores = compute_density_scores(artifact["X_train_scaled"], X_test_scaled, k_density)
            density_bins, thresholds = assign_density_bins(density_scores)
            for density_bin in DENSITY_BINS:
                mask = density_bins == density_bin
                y_bin = y_test[mask]
                proba_bin = np.asarray(baseline["proba"])[mask]
                pred_bin = np.asarray(baseline["pred"])[mask]
                X_bin = X_test_scaled[mask]
                X_raw_bin = X_test_raw[mask]
                rows.append(
                    {
                        "seed": seed,
                        "feature_group": group,
                        "density_bin": density_bin,
                        "n_samples": int(mask.sum()),
                        "auc": auc_or_nan(y_bin, proba_bin),
                        "acc": float(np.mean(pred_bin == y_bin)) if mask.sum() else float("nan"),
                        "positive_rate": float(np.mean(pred_bin)) if mask.sum() else float("nan"),
                        "mean_probability": float(np.mean(proba_bin)) if mask.sum() else float("nan"),
                        "silhouette": safe_silhouette(X_bin, y_bin) if mask.sum() else float("nan"),
                        "js_divergence": js_divergence_feature_mass(X_raw_bin, y_bin) if mask.sum() else float("nan"),
                        "density_q25": thresholds["q25"],
                        "density_q75": thresholds["q75"],
                    }
                )
            for idx, sample in enumerate(test_samples):
                density_rows.append(
                    {
                        "seed": seed,
                        "feature_group": group,
                        "sample_index": idx,
                        "sha256": getattr(sample, "sha256", None),
                        "label": getattr(sample, "label", None),
                        "density_score": float(density_scores[idx]),
                        "density_bin": str(density_bins[idx]),
                    }
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        write_csv(args.output_dir / "metric_rows.csv", list(rows[0].keys()), rows)
        aggregate = aggregate_rows(
            rows,
            group_keys=["feature_group", "density_bin"],
            metric_keys=["auc", "acc", "positive_rate", "mean_probability", "silhouette", "js_divergence"],
        )
        write_csv(args.output_dir / "aggregate_metric_rows.csv", list(aggregate[0].keys()), aggregate)
    if density_rows:
        write_csv(args.output_dir / "density_rows.csv", list(density_rows[0].keys()), density_rows)
    write_json(args.output_dir / "resolved_config.json", config)
    print(f"RQ5.3 MLP run complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
