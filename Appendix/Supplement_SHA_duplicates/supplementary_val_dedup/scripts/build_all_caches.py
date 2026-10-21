import json
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_week_cache import build_week_cache

CONFIG = json.loads(Path(
    str(Path(__file__).resolve().parents[4]) + "/artifacts/5_3_targeted_perturbation_response/"
    "configs/LightGBM/feature_perturbation_lightgbm_balanced_main.json"
).read_text())

PLATFORM = CONFIG["platform"]
TRAIN_WEEKS = CONFIG["train_weeks"]
TEST_WEEKS = CONFIG["test_weeks"]

jobs = [(w, PLATFORM, "train") for w in TRAIN_WEEKS] + [(w, PLATFORM, "test") for w in TEST_WEEKS]
print(f"Total weeks to cache: {len(jobs)}")

if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(build_week_cache, w, p, s): (w, s) for w, p, s in jobs}
        done = 0
        for fut in as_completed(futs):
            w, s = futs[fut]
            try:
                fut.result()
            except Exception as e:
                print(f"[ERROR] {w} {s}: {e}", flush=True)
            done += 1
            print(f"progress: {done}/{len(jobs)}", flush=True)
    print("ALL_CACHES_DONE")
