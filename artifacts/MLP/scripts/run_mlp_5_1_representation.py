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
    compute_mix_at_k_values,
    configure_torch_determinism,
    device_from_arg,
    feature_rankings,
    js_divergence_feature_mass,
    labels_array,
    resolve_config,
    safe_silhouette,
    seed_split_subsets,
    select_feature_group,
    train_group_model,
    transform_with_scaler,
    week_paths,
    load_samples_from_jsonl,
    balance_samples,
    write_csv,
    write_json,
)


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "5_1" / "mlp_rq1_wild_b_main.json"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "results" / "5_1" / "mlp_rq1_wild_b_main"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MLP DCSA section 5.1.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    config = resolve_config(args.config)
    configure_torch_determinism(bool(config.get("torch_deterministic", True)))
    device = device_from_arg(args.device)
    data_root = Path(config["data_root"])
    train_paths = week_paths(data_root, str(config.get("platform", "Win32")), config["train_weeks"], "train")
    from mlp_core import load_samples_from_paths  # local import to keep callsite explicit

    all_train_samples = load_samples_from_paths(train_paths, packer_filter=str(config.get("packer_filter", "all")))
    loss_fn = nn.BCEWithLogitsLoss()
    mix_k_values = [int(v) for v in config.get("mix_k_values", [10])]

    metric_rows: List[Dict[str, object]] = []
    ranking_rows: List[Dict[str, object]] = []
    checks_by_seed: List[Dict[str, object]] = []

    for seed in [int(seed) for seed in config["seeds"]]:
        if bool(config.get("balance_train", True)):
            train_samples = balance_samples(
                all_train_samples,
                seed=seed,
                max_per_class=config.get("max_train_per_class"),
            )
        else:
            train_samples = list(all_train_samples)
        split = seed_split_subsets(
            train_samples,
            seed=seed,
            validation_fraction=float(config["validation_fraction"]),
        )
        train_subset = split["train_subset"]
        val_subset = split["val_subset"]
        group_artifacts: Dict[str, Dict[str, object]] = {}
        for group in list(config.get("feature_groups", FEATURE_GROUPS)):
            group_artifacts[group] = train_group_model(
                train_subset=train_subset,
                val_subset=val_subset,
                group=group,
                config=config,
                seed=seed,
                device=device,
            )

        for week_idx, week in enumerate(config["test_weeks"]):
            test_path = week_paths(data_root, str(config.get("platform", "Win32")), [week], "test")[0]
            week_samples = load_samples_from_jsonl(test_path, packer_filter=str(config.get("packer_filter", "all")))
            if bool(config.get("balance_test", False)):
                week_samples = balance_samples(
                    week_samples,
                    seed=seed + week_idx + 1,
                    max_per_class=config.get("max_test_per_class"),
                )
            y_test = labels_array(week_samples)

            per_group_metrics: Dict[str, Dict[str, float]] = {}
            for group in list(config.get("feature_groups", FEATURE_GROUPS)):
                artifact = group_artifacts[group]
                X_test_raw = select_feature_group(week_samples, group)
                X_test_scaled = transform_with_scaler(artifact["scaler"], X_test_raw)
                from mlp_core import evaluate_model  # local import keeps script surface small

                eval_metrics = evaluate_model(
                    artifact["model"],
                    X_test_scaled,
                    y_test,
                    int(config["batch_size"]),
                    device,
                    loss_fn,
                )
                mix_values = compute_mix_at_k_values(X_test_scaled, y_test, mix_k_values)
                silhouette = safe_silhouette(X_test_scaled, y_test)
                js_divergence = js_divergence_feature_mass(X_test_raw, y_test)
                per_group_metrics[group] = {
                    "acc": float(eval_metrics["acc"]),
                    "f1": float(eval_metrics["f1"]),
                    "auc": float(eval_metrics["auc"]),
                    "positive_rate": float(np.mean(np.asarray(eval_metrics["pred"]))),
                    "silhouette": silhouette,
                    "js_divergence": js_divergence,
                }
                for mix_k, mix_value in mix_values.items():
                    metric_rows.append(
                        {
                            "seed": seed,
                            "test_week": week,
                            "feature_group": group,
                            "mix_k": mix_k,
                            "input_dim": int(X_test_scaled.shape[1]),
                            "train_size": int(len(train_subset)),
                            "val_size": int(len(val_subset)),
                            "test_size": int(len(week_samples)),
                            "best_epoch": int(artifact["history"]["best_epoch"]),
                            "epochs_ran": int(artifact["history"]["epochs_ran"]),
                            "best_val_auc": float(artifact["history"]["best_val_auc"]),
                            "test_auc": float(eval_metrics["auc"]),
                            "test_acc": float(eval_metrics["acc"]),
                            "test_f1": float(eval_metrics["f1"]),
                            "test_positive_rate": float(np.mean(np.asarray(eval_metrics["pred"]))),
                            "mix_at_k": float(mix_value),
                            "silhouette": silhouette,
                            "js_divergence": js_divergence,
                        }
                    )

            auc_rank = feature_rankings({group: metrics["auc"] for group, metrics in per_group_metrics.items()})
            mix_rank_source = {}
            for group in per_group_metrics:
                group_week_rows = [
                    row for row in metric_rows
                    if int(row["seed"]) == seed and str(row["test_week"]) == week and str(row["feature_group"]) == group
                ]
                mix_rank_source[group] = float(group_week_rows[0]["mix_at_k"]) if group_week_rows else float("nan")
            mix_rank = feature_rankings(mix_rank_source, higher_is_better=False)
            sil_rank = feature_rankings({group: metrics["silhouette"] for group, metrics in per_group_metrics.items()})
            js_rank = feature_rankings({group: metrics["js_divergence"] for group, metrics in per_group_metrics.items()})
            for group in per_group_metrics:
                ranking_rows.append(
                    {
                        "seed": seed,
                        "test_week": week,
                        "feature_group": group,
                        "auc_rank": auc_rank[group],
                        "mix_at_k_rank": mix_rank[group],
                        "silhouette_rank": sil_rank[group],
                        "js_divergence_rank": js_rank[group],
                    }
                )
        checks_by_seed.append(
            {
                "seed": seed,
                "train_val_sha_overlap_count": int(split["sha_overlap_count"]),
                "train_size": int(len(train_subset)),
                "val_size": int(len(val_subset)),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if metric_rows:
        write_csv(args.output_dir / "metric_rows.csv", list(metric_rows[0].keys()), metric_rows)
        aggregate_metric_rows = aggregate_rows(
            metric_rows,
            group_keys=["feature_group", "mix_k"],
            metric_keys=["test_auc", "test_acc", "test_f1", "test_positive_rate", "mix_at_k", "silhouette", "js_divergence"],
        )
        write_csv(args.output_dir / "aggregate_metric_rows.csv", list(aggregate_metric_rows[0].keys()), aggregate_metric_rows)
    if ranking_rows:
        write_csv(args.output_dir / "ranking_rows.csv", list(ranking_rows[0].keys()), ranking_rows)
        aggregate_ranking_rows = aggregate_rows(
            ranking_rows,
            group_keys=["feature_group"],
            metric_keys=["auc_rank", "mix_at_k_rank", "silhouette_rank", "js_divergence_rank"],
        )
        write_csv(args.output_dir / "aggregate_ranking_rows.csv", list(aggregate_ranking_rows[0].keys()), aggregate_ranking_rows)
    write_json(args.output_dir / "checks_by_seed.json", checks_by_seed)
    write_json(args.output_dir / "resolved_config.json", config)
    print(f"RQ5.1 MLP run complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
