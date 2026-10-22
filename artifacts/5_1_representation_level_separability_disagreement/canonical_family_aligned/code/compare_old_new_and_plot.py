#!/usr/bin/env python3
"""Compare the canonical Figure 2 results with the earlier model-specific outputs and plot k=10 panels.

Reads  ../results/{wild_b,unpacked_b}/{aggregate_metrics,weekly_metrics}.csv  (canonical run)
and    ../../results/{LR,LightGBM,RF}/...                                       (earlier outputs)
Writes ../results/before_after_comparison.csv, ../results/cell_composition_comparison.csv,
       ../results/figures/classification_vs_structure_scatter_{view}_{model}.{pdf,png}
Nothing under ../../results is modified.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
NEW = HERE.parent / "results"
OLD = HERE.parents[1] / "results"
GROUPS = ["all", "header", "section", "strings", "imports"]
LABEL = {"all": "All", "header": "Header", "section": "Section", "strings": "Strings", "imports": "Imports"}
COLOR = {"all": "#264653", "header": "#2A9D8F", "section": "#E9C46A", "strings": "#F4A261", "imports": "#E76F51"}
OLD_DIR = {
    ("lr", "wild_b"): (OLD / "LR/family_aligned_rq1_matched", "matched_aggregate_summary.csv"),
    ("lr", "unpacked_b"): (OLD / "LR/family_aligned_rq1_matched_unpacked", "matched_unpacked_aggregate_summary.csv"),
    ("lightgbm", "wild_b"): (OLD / "LightGBM/family_aligned_rq1_matched_lightgbm", "aggregate_summary.csv"),
    ("lightgbm", "unpacked_b"): (OLD / "LightGBM/family_aligned_rq1_matched_unpacked_lightgbm", "aggregate_summary.csv"),
    ("rf", "wild_b"): (OLD / "RF/classification_vs_structure_wild_b", "aggregate_summary.csv"),
    ("rf", "unpacked_b"): (OLD / "RF/classification_vs_structure_unpacked_b", "aggregate_summary.csv"),
}


def read(path):
    with open(path, newline="") as h:
        return list(csv.DictReader(h))


def write(path, rows):
    with open(path, "w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def rank(values):
    order = sorted(values, key=lambda g: (-values[g], g))
    return {g: i + 1 for i, g in enumerate(order)}


def load_new(view):
    out = {}
    for r in read(NEW / view / "aggregate_metrics.csv"):
        if int(r["k"]) == 10:
            out[(r["model"], r["feature_group"])] = (float(r["same_family_rate_mean"]), float(r["family_macro_f1_mean"]))
    return out


def load_old(model, view):
    d, summary = OLD_DIR[(model, view)]
    f1 = {r["feature_group"]: float(r["mean"]) for r in read(d / summary)
          if r["scope"] == "metric" and r["metric"] == "family_macro_f1"}
    geo = {r["feature_group"]: float(r["mean"]) for r in read(d / "aggregate_by_k.csv")
           if r["k"] == "10" and r["scope"] == "metric" and r["metric"] == "same_family_rate_at_k"}
    return f1, geo


def main():
    comp, cells = [], []
    for view in ["wild_b", "unpacked_b"]:
        new = load_new(view)
        weekly = read(NEW / view / "weekly_metrics.csv")
        new_cells = {(r["seed"], r["week"]): (r["n_train"], r["n_test"], r["n_families"])
                     for r in weekly if r["k"] == "10" and r["model"] == "lr" and r["feature_group"] == "all"}
        for model in ["lr", "lightgbm", "rf"]:
            f1o, geoo = load_old(model, view)
            gn = {g: new[(model, g)][0] for g in GROUPS}
            fn = {g: new[(model, g)][1] for g in GROUPS}
            rgn, rfn, rgo, rfo = rank(gn), rank(fn), rank(geoo), rank(f1o)
            for g in GROUPS:
                comp.append({
                    "view": view, "model": model, "feature_group": g,
                    "old_same_family_rate_k10": round(geoo[g], 6), "new_same_family_rate_k10": round(gn[g], 6),
                    "geometry_delta": round(gn[g] - geoo[g], 6), "old_geometry_rank": rgo[g], "new_geometry_rank": rgn[g],
                    "old_macro_f1": round(f1o[g], 6), "new_macro_f1": round(fn[g], 6),
                    "macro_f1_delta": round(fn[g] - f1o[g], 6), "old_f1_rank": rfo[g], "new_f1_rank": rfn[g],
                })
            wk = "weekly_results.csv"
            d, _ = OLD_DIR[(model, view)]
            old_cells = {(r["seed"], r["test_week"]): (r["n_train_samples"], r["n_test_samples"], r["n_test_families"])
                         for r in read(d / wk) if r["k"] == "10"}
            both = set(old_cells) & set(new_cells)
            cells.append({
                "view": view, "model": model, "new_cells": len(new_cells), "old_cells": len(old_cells),
                "cells_in_both": len(both),
                "same_n_train": sum(old_cells[c][0] == new_cells[c][0] for c in both),
                "same_n_test": sum(old_cells[c][1] == new_cells[c][1] for c in both),
                "old_cells_with_test_families_lt_2": sum(int(v[2]) < 2 for v in old_cells.values()),
                "new_cells_with_train_families_lt_2": sum(int(v[2]) < 2 for v in new_cells.values()),
                "old_test_family_count_dist": dict(sorted({k: sum(v[2] == k for v in old_cells.values())
                                                          for k in {v[2] for v in old_cells.values()}}.items())),
            })
    write(NEW / "before_after_comparison.csv", comp)
    write(NEW / "cell_composition_comparison.csv", cells)

    pts = [load_new(v) for v in ("wild_b", "unpacked_b")]
    xs = [p[k][1] for p in pts for k in p]
    ys = [p[k][0] for p in pts for k in p]
    xp, yp = max(0.01, np.ptp(xs) * 0.08), max(0.01, np.ptp(ys) * 0.08)
    lim = (min(xs) - xp, max(xs) + xp, min(ys) - yp, max(ys) + yp)
    figs = NEW / "figures"
    figs.mkdir(exist_ok=True)
    plt.rcParams.update({"axes.labelsize": 19, "xtick.labelsize": 16, "ytick.labelsize": 16})
    for view, data in zip(("wild_b", "unpacked_b"), pts):
        for model in ("lr", "lightgbm", "rf"):
            fig, ax = plt.subplots(figsize=(6.2, 5.2))
            for g in GROUPS:
                y, x = data[(model, g)]
                ax.scatter(x, y, s=70, color=COLOR[g], zorder=3)
                ax.annotate(LABEL[g], (x, y), textcoords="offset points", xytext=(5, 5), fontsize=14)
            ax.set_xlim(lim[0], lim[1])
            ax.set_ylim(lim[2], lim[3])
            lo, hi = max(lim[0], lim[2]), min(lim[1], lim[3])
            if hi > lo:
                ax.plot([lo, hi], [lo, hi], "--", lw=1.1, color="#888888", alpha=0.8, zorder=1)
            ax.set_xlabel("Classification Performance (Macro-F1)")
            ax.set_ylabel("Same-Family Rate@10")
            ax.grid(True, lw=0.9, alpha=0.4)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            fig.tight_layout()
            stem = f"classification_vs_structure_scatter_{view}_{model}"
            fig.savefig(figs / f"{stem}.pdf", bbox_inches="tight")
            fig.savefig(figs / f"{stem}.png", dpi=200, bbox_inches="tight")
            plt.close(fig)
    (NEW / "figures" / "axis_limits.txt").write_text(
        f"common limits over both views and all three models: x=({lim[0]:.4f},{lim[1]:.4f}) y=({lim[2]:.4f},{lim[3]:.4f})\n")
    print("limits", lim)


if __name__ == "__main__":
    main()
