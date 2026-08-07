#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent


def run_step(script_name: str, config_rel: str, output_rel: str, device: str, workers: int) -> None:
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / script_name),
        "--config",
        str(BASE / config_rel),
        "--output-dir",
        str(BASE / output_rel),
        "--device",
        device,
        "--workers",
        str(workers),
    ]
    print(f"\n[launch] running {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch MLP DCSA core sections 5.1, 5.3-5.5 in sequence.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    steps = [
        ("run_mlp_5_1_representation.py", "configs/5_1/mlp_rq1_wild_b_main.json", "results/5_1/mlp_rq1_wild_b_main"),
        ("run_mlp_5_3_masking.py", "configs/5_3/mlp_rq2_targeted_masking_wild_b_main.json", "results/5_3/mlp_rq2_targeted_masking_wild_b_main"),
        ("run_mlp_5_4_reliability.py", "configs/5_4/mlp_rq3_density_reliability_wild_b_main.json", "results/5_4/mlp_rq3_density_reliability_wild_b_main"),
        ("run_mlp_5_5_fragility.py", "configs/5_5/mlp_rq3_density_fragility_wild_b_main.json", "results/5_5/mlp_rq3_density_fragility_wild_b_main"),
    ]
    for script_name, config_rel, output_rel in steps:
        run_step(script_name, config_rel, output_rel, args.device, args.workers)
    print(f"\nMLP DCSA core launch complete. Workspace root: {BASE}")


if __name__ == "__main__":
    main()
