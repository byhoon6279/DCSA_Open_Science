#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
from torch import nn

from mlp_core import (
    FEATURE_GROUPS,
    aggregate_rows,
    apply_raw_zero_mask_then_scale,
    auc_or_nan,
    compute_logit_gradient_importance,
    configure_torch_determinism,
    device_from_arg,
    evaluate_model,
    labels_array,
    pooled_train_test_samples,
    random_mask_indices,
    resolve_config,
    seed_split_subsets,
    select_feature_group,
    top_mask_indices,
    train_group_model,
    transform_with_scaler,
    write_csv,
    write_json,
)


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "5_3" / "mlp_rq2_targeted_masking_wild_b_main.json"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "results" / "5_3" / "mlp_rq2_targeted_masking_wild_b_main"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MLP DCSA section 5.3.")
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
    strengths = [float(v) for v in config["masking_strengths"]]
    random_repeats = int(config["random_control_repeats"])
    threshold = float(config.get("threshold", 0.5))

    rows: List[Dict[str, object]] = []
    checks_by_seed: List[Dict[str, object]] = []
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
            importance_scores = compute_logit_gradient_importance(
                artifact["model"],
                artifact["X_val_scaled"],
                int(config["batch_size"]),
                device,
            )
            baseline_proba = np.asarray(baseline["proba"])
            baseline_pred = np.asarray(baseline["pred"])
            for strength in strengths:
                important_idx = top_mask_indices(importance_scores, strength)
                masked_scaled = apply_raw_zero_mask_then_scale(X_test_raw, artifact["scaler"], important_idx)
                masked = evaluate_model(
                    artifact["model"],
                    masked_scaled,
                    y_test,
                    int(config["batch_size"]),
                    device,
                    loss_fn,
                )
                masked_proba = np.asarray(masked["proba"])
                masked_pred = np.asarray(masked["pred"])
                rows.append(
                    {
                        "seed": seed,
                        "feature_group": group,
                        "mask_type": "important",
                        "strength": strength,
                        "repeat": 0,
                        "n_masked_features": int(important_idx.size),
                        "baseline_auc": float(baseline["auc"]),
                        "masked_auc": auc_or_nan(y_test, masked_proba),
                        "delta_auc": float(baseline["auc"]) - auc_or_nan(y_test, masked_proba),
                        "flip_rate": float(np.mean(masked_pred != baseline_pred)),
                        "malware_to_benign_flip_rate": float(np.mean((baseline_pred == 1) & (masked_pred == 0))),
                        "benign_to_malware_flip_rate": float(np.mean((baseline_pred == 0) & (masked_pred == 1))),
                        "mean_probability_shift": float(np.mean(np.abs(masked_proba - baseline_proba))),
                        "mean_signed_probability_shift": float(np.mean(masked_proba - baseline_proba)),
                        "threshold": threshold,
                    }
                )
                for repeat in range(1, random_repeats + 1):
                    random_idx = random_mask_indices(X_test_raw.shape[1], int(important_idx.size), seed, group, strength, repeat)
                    random_scaled = apply_raw_zero_mask_then_scale(X_test_raw, artifact["scaler"], random_idx)
                    random_eval = evaluate_model(
                        artifact["model"],
                        random_scaled,
                        y_test,
                        int(config["batch_size"]),
                        device,
                        loss_fn,
                    )
                    random_proba = np.asarray(random_eval["proba"])
                    random_pred = np.asarray(random_eval["pred"])
                    rows.append(
                        {
                            "seed": seed,
                            "feature_group": group,
                            "mask_type": "random",
                            "strength": strength,
                            "repeat": repeat,
                            "n_masked_features": int(random_idx.size),
                            "baseline_auc": float(baseline["auc"]),
                            "masked_auc": auc_or_nan(y_test, random_proba),
                            "delta_auc": float(baseline["auc"]) - auc_or_nan(y_test, random_proba),
                            "flip_rate": float(np.mean(random_pred != baseline_pred)),
                            "malware_to_benign_flip_rate": float(np.mean((baseline_pred == 1) & (random_pred == 0))),
                            "benign_to_malware_flip_rate": float(np.mean((baseline_pred == 0) & (random_pred == 1))),
                            "mean_probability_shift": float(np.mean(np.abs(random_proba - baseline_proba))),
                            "mean_signed_probability_shift": float(np.mean(random_proba - baseline_proba)),
                            "threshold": threshold,
                        }
                    )
        checks_by_seed.append({"seed": seed, "train_val_sha_overlap_count": int(split["sha_overlap_count"])})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        write_csv(args.output_dir / "masking_rows.csv", list(rows[0].keys()), rows)
        aggregate = aggregate_rows(
            rows,
            group_keys=["feature_group", "mask_type", "strength"],
            metric_keys=["baseline_auc", "masked_auc", "delta_auc", "flip_rate", "malware_to_benign_flip_rate", "benign_to_malware_flip_rate", "mean_probability_shift", "mean_signed_probability_shift"],
        )
        write_csv(args.output_dir / "aggregate_masking_rows.csv", list(aggregate[0].keys()), aggregate)
    write_json(args.output_dir / "checks_by_seed.json", checks_by_seed)
    write_json(args.output_dir / "resolved_config.json", config)
    print(f"RQ5.2 MLP run complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
