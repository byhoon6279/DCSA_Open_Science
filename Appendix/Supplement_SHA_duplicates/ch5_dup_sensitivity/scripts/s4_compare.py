"""Stage 4: reproduction check against DCSA outputs + dup vs dedup comparison tables."""
from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True
import glob  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common_dup import RESULTS  # noqa: E402

DCSA = str(Path(__file__).resolve().parents[4]) + "/artifacts"
SIL_RUN = (DCSA + "/5_1_representation_level_separability_disagreement/results/RF/feature_subset_wild_b/weekly_results.csv")  # silhouette is classifier-independent, so the bundled RF run is used
ORDER = ["all", "header", "section", "strings", "imports"]
K = [5, 10, 15, 20]
pd.set_option("display.width", 200)


def rank_str(s: pd.Series, ascending=True) -> str:
    return " < ".join(s.sort_values(ascending=ascending).index) if ascending else " > ".join(s.sort_values(ascending=False).index)


def section51():
    ours = pd.concat(pd.read_csv(f) for f in sorted(glob.glob(str(RESULTS / "s51_seed*.csv"))))
    out = []
    # --- reproduction: DCSA weekly Mix@k (dup) and silhouette
    ref = pd.read_csv(f"{DCSA}/5_1_representation_level_separability_disagreement/results/LR/"
                      "win32_all_train_all_test_balanced_test/weekly_results.csv")
    ref = ref.rename(columns={"test_week": "week"})
    dup = ours[ours.cond == "dup"]
    diffs = []
    for g in ORDER:
        for k in K:
            r = ref[ref.mix_k == k][["seed", "week", f"{g}_mix_at_k"]]
            m = dup[dup.group == g].merge(r, on=["seed", "week"])
            diffs.append({"group": g, "k": k, "n_cells": len(m),
                          "max_abs_diff": (m[f"mix_at_{k}"] - m[f"{g}_mix_at_k"]).abs().max(),
                          "mean_ours": m[f"mix_at_{k}"].mean(), "mean_dcsa": m[f"{g}_mix_at_k"].mean()})
    rep = pd.DataFrame(diffs)
    sil = pd.read_csv(SIL_RUN).rename(columns={"test_week": "week"})
    sil = sil[sil.mix_k == sil.mix_k.min()]
    sil_rep = []
    for g in ORDER:
        m = dup[dup.group == g].merge(sil[["seed", "week", f"{g}_silhouette"]], on=["seed", "week"])
        sil_rep.append({"group": g, "n_cells": len(m), "max_abs_diff": (m.silhouette - m[f"{g}_silhouette"]).abs().max(),
                        "mean_ours": m.silhouette.mean(), "mean_dcsa": m[f"{g}_silhouette"].mean()})
    out.append(("5.1 reproduction (dup vs DCSA), Mix@k", rep))
    out.append(("5.1 reproduction (dup vs DCSA silhouette run)", pd.DataFrame(sil_rep)))

    # --- dup vs dedup summary (mean over 12 weeks x 5 seeds)
    metrics = [f"mix_at_{k}" for k in K] + [f"mix_ta_at_{k}" for k in K] + ["silhouette"]
    agg = ours.groupby(["group", "cond"])[metrics].mean().unstack("cond")
    rows = []
    for g in ORDER:
        r = {"group": g}
        for mt in ["mix_at_10", "mix_ta_at_10", "silhouette"]:
            d, u = agg.loc[g, (mt, "dup")], agg.loc[g, (mt, "dedup")]
            r.update({f"{mt}_dup": d, f"{mt}_dedup": u, f"{mt}_abs": u - d, f"{mt}_rel%": 100 * (u - d) / d})
        kavg_d = agg.loc[g, [(f"mix_at_{k}", "dup") for k in K]].mean()
        kavg_u = agg.loc[g, [(f"mix_at_{k}", "dedup") for k in K]].mean()
        r.update({"mix_kavg_dup(Table3 def)": kavg_d, "mix_kavg_dedup": kavg_u})
        rows.append(r)
    summ = pd.DataFrame(rows).set_index("group")
    out.append(("5.1 dup vs dedup (12 weeks x 5 seeds)", summ))

    ranks = pd.DataFrame({
        "Mix@10 dup (low->high)": [rank_str(summ["mix_at_10_dup"])],
        "Mix@10 dedup": [rank_str(summ["mix_at_10_dedup"])],
        "Mix@10 tie-avg dup": [rank_str(summ["mix_ta_at_10_dup"])],
        "Mix@10 tie-avg dedup": [rank_str(summ["mix_ta_at_10_dedup"])],
        "Mix k-avg dup": [rank_str(summ["mix_kavg_dup(Table3 def)"])],
        "Mix k-avg dedup": [rank_str(summ["mix_kavg_dedup"])],
        "Silhouette dup (high->low)": [rank_str(summ["silhouette_dup"], ascending=False)],
        "Silhouette dedup": [rank_str(summ["silhouette_dedup"], ascending=False)],
    }).T
    out.append(("5.1 group ordering", ranks))

    # per (seed, week) rank stability
    for metric in ["mix_at_10", "mix_ta_at_10"]:
        cell = ours.pivot_table(index=["seed", "week", "cond"], columns="group", values=metric)
        rk = cell.rank(axis=1)
        stab = []
        for g in ORDER:
            a = rk.xs("dup", level="cond")[g]; b = rk.xs("dedup", level="cond")[g]
            stab.append({"group": g, "mean_rank_dup": a.mean(), "mean_rank_dedup": b.mean(),
                         "cells_rank_changed": int((a != b).sum()), "n_cells": len(a)})
        out.append((f"5.1 {metric} rank per (seed, week) cell (1 = least mixed)", pd.DataFrame(stab)))
    return out


def section54():
    ours = pd.concat(pd.read_csv(f) for f in sorted(glob.glob(str(RESULTS / "s54_seed*.csv"))))
    ref = pd.read_csv(f"{DCSA}/5_4_density_conditioned_reliability/results/LR/density_reliability_controlled_main/k_10/metric_rows.csv")
    m = ours[ours.cond == "dup"].merge(ref, left_on=["seed", "group", "density_bin"], right_on=["seed", "feature_group", "density_bin"], suffixes=("", "_dcsa"))
    rep = m.groupby("group").apply(lambda d: pd.Series({
        "max_abs_diff_mix": (d.mix_at_10 - d.mix_at_10_dcsa).abs().max(),
        "n_samples_match": bool((d.n_samples == d.n_samples_dcsa).all())}), include_groups=False)
    out = [("5.4 reproduction (dup vs DCSA metric_rows, per seed x bin)", rep)]
    for metric in ["mix_at_10", "mix_ta_at_10"]:
        agg = ours.groupby(["group", "cond", "density_bin"])[metric].mean().unstack("density_bin")[["high_density", "mid_density", "low_density"]]
        agg["low-high"] = agg.low_density - agg.high_density
        out.append((f"5.4 {metric} by density bin (mean of 5 seeds)", agg))
        piv = ours.pivot_table(index=["group", "cond", "seed"], columns="density_bin", values=metric)
        piv["low>high"] = piv.low_density > piv.high_density
        piv["monotone low>mid>high"] = (piv.low_density > piv.mid_density) & (piv.mid_density > piv.high_density)
        out.append((f"5.4 {metric} direction counts over 5 seeds", piv.groupby(["group", "cond"])[["low>high", "monotone low>mid>high"]].sum()))
    n = ours.groupby(["cond", "seed"]).agg(n_test=("n_samples", "sum"), n_train=("n_train", "first")).groupby("cond").mean()
    out.append(("5.4 sample sizes", n))
    return out


if __name__ == "__main__":
    parts = []
    which = sys.argv[1:] or ["51", "54"]
    if "51" in which:
        parts += section51()
    if "54" in which:
        parts += section54()
    with open(RESULTS / "comparison.txt", "w") as f:
        for title, df in parts:
            block = f"\n## {title}\n{df.round(4).to_string()}\n"
            print(block); f.write(block)
