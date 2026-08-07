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
    apply_raw_zero_mask_then_scale,
    assign_density_bins,
    auc_or_nan,
    compute_density_scores,
    compute_logit_gradient_importance,
    configure_torch_determinism,
    device_from_arg,
    evaluate_flip_metrics,
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


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "5_5" / "mlp_rq3_density_fragility_wild_b_main.json"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "results" / "5_5" / "mlp_rq3_density_fragility_wild_b_main"


def build_amplification_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[tuple[int, str, str, float, str], List[Dict[str, object]]] = {}
    for row in rows:
        key = (
            int(row["seed"]),
            str(row["feature_group"]),
            str(row["mask_type"]),
            float(row["strength"]),
            str(row["density_bin"]),
        )
        grouped.setdefault(key, []).append(row)

    lookup: Dict[tuple[int, str, str, float, str], Dict[str, object]] = {}
    for key, bucket in grouped.items():
        lookup[key] = {
            "flip_rate": float(np.mean([float(item["flip_rate"]) for item in bucket])),
        }
    keys = sorted({(int(r["seed"]), str(r["feature_group"]), str(r["mask_type"]), float(r["strength"])) for r in rows})
    out: List[Dict[str, object]] = []
    for seed, feature_group, mask_type, strength in keys:
        high = lookup.get((seed, feature_group, mask_type, strength, "high_density"))
        low = lookup.get((seed, feature_group, mask_type, strength, "low_density"))
        if high is None or low is None:
            continue
        high_flip = float(high["flip_rate"])
        low_flip = float(low["flip_rate"])
        out.append(
            {
                "seed": seed,
                "feature_group": feature_group,
                "mask_type": mask_type,
                "strength": strength,
                "high_density_flip_rate": high_flip,
                "low_density_flip_rate": low_flip,
                "delta_flip_density": low_flip - high_flip,
                "flip_density_ratio": float(low_flip / high_flip) if high_flip > 0 else float("nan"),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MLP DCSA section 5.5.")
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
    k_density = int(config.get("k_density", 10))

    rows: List[Dict[str, object]] = []
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
            baseline_proba = np.asarray(baseline["proba"])
            density_scores = compute_density_scores(artifact["X_train_scaled"], X_test_scaled, k_density)
            density_bins, _ = assign_density_bins(density_scores)
            importance_scores = compute_logit_gradient_importance(
                artifact["model"],
                artifact["X_val_scaled"],
                int(config["batch_size"]),
                device,
            )
            for strength in strengths:
                important_idx = top_mask_indices(importance_scores, strength)
                perturbations = [("important", 0, important_idx)]
                for repeat in range(1, random_repeats + 1):
                    perturbations.append(
                        ("random", repeat, random_mask_indices(X_test_raw.shape[1], int(important_idx.size), seed, group, strength, repeat))
                    )
                for mask_type, repeat, mask_idx in perturbations:
                    masked_scaled = apply_raw_zero_mask_then_scale(X_test_raw, artifact["scaler"], mask_idx)
                    masked = evaluate_model(
                        artifact["model"],
                        masked_scaled,
                        y_test,
                        int(config["batch_size"]),
                        device,
                        loss_fn,
                    )
                    masked_proba = np.asarray(masked["proba"])
                    for density_bin in DENSITY_BINS:
                        mask = density_bins == density_bin
                        metrics = evaluate_flip_metrics(
                            baseline_proba[mask],
                            masked_proba[mask],
                            y_test[mask],
                            threshold,
                        )
                        rows.append(
                            {
                                "seed": seed,
                                "feature_group": group,
                                "mask_type": mask_type,
                                "repeat": repeat,
                                "strength": strength,
                                "density_bin": density_bin,
                                "n_samples": int(mask.sum()),
                                "baseline_auc": auc_or_nan(y_test[mask], baseline_proba[mask]),
                                "masked_auc": auc_or_nan(y_test[mask], masked_proba[mask]),
                                "delta_auc": auc_or_nan(y_test[mask], baseline_proba[mask]) - auc_or_nan(y_test[mask], masked_proba[mask]),
                                **metrics,
                            }
                        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        write_csv(args.output_dir / "density_masking_rows.csv", list(rows[0].keys()), rows)
        aggregate = aggregate_rows(
            rows,
            group_keys=["feature_group", "mask_type", "strength", "density_bin"],
            metric_keys=["baseline_auc", "masked_auc", "delta_auc", "flip_rate", "benign_to_malware_rate_overall", "malware_to_benign_rate_overall", "mean_abs_probability_shift", "mean_signed_probability_shift"],
        )
        write_csv(args.output_dir / "aggregate_density_masking_rows.csv", list(aggregate[0].keys()), aggregate)
        amp_rows = build_amplification_rows(rows)
        write_csv(args.output_dir / "amplification_rows.csv", list(amp_rows[0].keys()), amp_rows)
    write_json(args.output_dir / "resolved_config.json", config)
    print(f"RQ5.4 MLP run complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
