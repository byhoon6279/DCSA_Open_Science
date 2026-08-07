#!/usr/bin/env python3
"""Generate derived configs for the D4 (density-conditioned fragility, Section 5.5)
class-conditional re-binning sensitivity check backing Appendix D.1.

Split out of a combined `prepare_class_conditional_configs.py`
(which generated both the D3 and D4 jobs in one file) so that D4 assets live under
5_5_density_conditioned_fragility/ per this package's section-first layout. See the
sibling D3 script under 5_4_density_conditioned_reliability/sensitivity/class_conditional/
for the reliability half.

Usage (from anywhere, no PYTHONPATH needed):
    python3 prepare_5_5_class_conditional_configs.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

HERE = Path(__file__).resolve()
CLASS_CONDITIONAL_ROOT = HERE.parents[1]          # .../5_5_density_conditioned_fragility/sensitivity/class_conditional
SECTION_ROOT = HERE.parents[3]                    # .../5_5_density_conditioned_fragility
ARTIFACTS_ROOT = HERE.parents[4]                  # .../artifacts
PACKAGE_ROOT = HERE.parents[5]                    # DCSA_Open_Science/
EMBER_ROOT = PACKAGE_ROOT / "Data"

CONFIG_ROOT = CLASS_CONDITIONAL_ROOT / "configs"
RUNNER_DIR = CLASS_CONDITIONAL_ROOT / "scripts"

JOBS = [
    {
        "name": "d4_lr_controlled_main",
        "base_config": SECTION_ROOT / "configs/LR/density_fragility_controlled_main.json",
        "runner": RUNNER_DIR / "run_class_conditional_density_conditioned_fragility.py",
        "output_dir": CLASS_CONDITIONAL_ROOT / "results/LR/density_fragility_controlled_main",
        "config_updates": {
            "save_class_conditional_rows": True,
            "class_conditional_low_quantile": 0.25,
            "class_conditional_high_quantile": 0.75,
            "num_workers": 1,
        },
    },
    {
        "name": "d4_lightgbm_controlled_main_permutation",
        "base_config": SECTION_ROOT / "configs/LightGBM/density_fragility_controlled_main_lightgbm.json",
        "runner": RUNNER_DIR / "run_class_conditional_density_conditioned_fragility.py",
        "output_dir": CLASS_CONDITIONAL_ROOT / "results/LightGBM/density_fragility_controlled_main_lightgbm_permutation",
        "config_updates": {
            "save_class_conditional_rows": True,
            "class_conditional_low_quantile": 0.25,
            "class_conditional_high_quantile": 0.75,
            "num_workers": 1,
            "lgbm_n_jobs": 6,
        },
    },
    {
        "name": "d4_rf_controlled_main",
        "base_config": SECTION_ROOT / "configs/RF/density_fragility_controlled_main_random_forest.json",
        "runner": RUNNER_DIR / "run_class_conditional_density_conditioned_fragility_with_rf.py",
        "output_dir": CLASS_CONDITIONAL_ROOT / "results/RF/rf_full_wild_b",
        "config_updates": {
            "save_class_conditional_rows": True,
            "class_conditional_low_quantile": 0.25,
            "class_conditional_high_quantile": 0.75,
            "num_workers": 1,
            "rf_n_jobs": 6,
        },
    },
]


def main() -> None:
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    run_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        # Matches this repo's other run_*.sh scripts (e.g.
        # scripts/RF/run_pe_intervention_rf.sh): resolve everything relative to
        # this script's own directory so the derived paths stay valid however
        # deep the package is cloned or copied, on any machine.
        'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
        'PYTHON_BIN="${PYTHON_BIN:-python3}"',
        "",
    ]

    manifest = []
    for job in JOBS:
        config = json.loads(job["base_config"].read_text(encoding="utf-8"))
        if "data_root" in config:
            # Relative to this derived config's own directory (CONFIG_ROOT),
            # matching the relative-path convention the base configs already
            # use (e.g. "../../../../EMBER2024/") instead of an absolute path
            # that only resolves on the machine that generated it.
            config["data_root"] = os.path.relpath(EMBER_ROOT, CONFIG_ROOT)
        config.update(job["config_updates"])
        out_path = CONFIG_ROOT / f"{job['name']}.json"
        out_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

        runner_rel = os.path.relpath(job["runner"], CONFIG_ROOT)
        config_rel = os.path.relpath(out_path, CONFIG_ROOT)
        output_dir_rel = os.path.relpath(job["output_dir"], CONFIG_ROOT)
        run_lines.append(
            f'"${{PYTHON_BIN}}" "${{SCRIPT_DIR}}/{runner_rel}" '
            f'--config "${{SCRIPT_DIR}}/{config_rel}" '
            f'--output-dir "${{SCRIPT_DIR}}/{output_dir_rel}"'
        )

        manifest.append(
            {
                "name": job["name"],
                "base_config": str(job["base_config"].relative_to(PACKAGE_ROOT)),
                "runner": str(job["runner"].relative_to(PACKAGE_ROOT)),
                "python_bin": "python3 (or $PYTHON_BIN, see run_all.sh)",
                "derived_config": str(out_path.relative_to(PACKAGE_ROOT)),
                "output_dir": str(job["output_dir"].relative_to(PACKAGE_ROOT)),
            }
        )

    run_script = CONFIG_ROOT / "run_all.sh"
    run_script.write_text("\n".join(run_lines) + "\n", encoding="utf-8")
    run_script.chmod(0o755)

    (CONFIG_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
