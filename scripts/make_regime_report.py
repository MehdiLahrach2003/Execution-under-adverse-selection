# scripts/make_regime_report.py
"""
Make regime report assets from reports/regime_grid.csv

Outputs:
- reports/figures/regime_delta_mean_heatmap.png
- reports/figures/regime_delta_p90_heatmap.png
- reports/figures/regime_fillrate_delta_heatmap.png
- reports/REGIME_REPORT.md
"""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


CSV_PATH = "reports/regime_grid.csv"
FIG_DIR = "reports/figures"

FIG_DELTA_MEAN = os.path.join(FIG_DIR, "regime_delta_mean_heatmap.png")
FIG_DELTA_P90 = os.path.join(FIG_DIR, "regime_delta_p90_heatmap.png")
FIG_FILLRATE = os.path.join(FIG_DIR, "regime_fillrate_delta_heatmap.png")

REPORT_MD = "reports/REGIME_REPORT.md"


def ensure_dirs() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)


def pivot_heatmap(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    # rows = tox_persist, cols = as_kick_scale (nice to read)
    return df.pivot(index="tox_persist", columns="as_kick_scale", values=value_col).sort_index()


def plot_heatmap(mat: pd.DataFrame, title: str, out_path: str) -> None:
    arr = mat.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(arr, aspect="auto")

    ax.set_title(title)
    ax.set_xlabel("as_kick_scale")
    ax.set_ylabel("tox_persist")

    ax.set_xticks(np.arange(mat.shape[1]))
    ax.set_xticklabels([f"{c:.2f}" for c in mat.columns])

    ax.set_yticks(np.arange(mat.shape[0]))
    ax.set_yticklabels([f"{r:.2f}" for r in mat.index])

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("value")

    # Annotate cells (short)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = arr[i, j]
            txt = "nan" if np.isnan(v) else f"{v:.3f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def compute_summary(df: pd.DataFrame) -> str:
    # We interpret delta_* as ToxicityAware - AlwaysMarket
    best_mean = df.loc[df["delta_mean"].idxmin()]  # lower delta_mean = improvement if costs are costs
    worst_mean = df.loc[df["delta_mean"].idxmax()]

    best_p90 = df.loc[df["delta_p90"].idxmin()]
    worst_p90 = df.loc[df["delta_p90"].idxmax()]

    # Fill-rate delta
    df["delta_fill_rate"] = df["ToxicityAware_fill_rate"] - df["AlwaysMarket_fill_rate"]
    best_fill = df.loc[df["delta_fill_rate"].idxmax()]
    worst_fill = df.loc[df["delta_fill_rate"].idxmin()]

    def fmt_row(r: pd.Series) -> str:
        return (
            f"(as_kick_scale={r['as_kick_scale']:.2f}, tox_persist={r['tox_persist']:.2f})"
        )

    lines = []
    lines.append("## Key takeaways (grid)")
    lines.append("")
    lines.append("We sweep `as_kick_scale` (adverse-selection kick intensity) and `tox_persist` (toxicity persistence).")
    lines.append("All deltas are `ToxicityAware - AlwaysMarket` (negative = improvement if IS is a cost).")
    lines.append("")
    lines.append(f"- Best **mean IS delta**: {best_mean['delta_mean']:.6f} at {fmt_row(best_mean)}")
    lines.append(f"- Worst **mean IS delta**: {worst_mean['delta_mean']:.6f} at {fmt_row(worst_mean)}")
    lines.append("")
    lines.append(f"- Best **p90 IS delta**: {best_p90['delta_p90']:.6f} at {fmt_row(best_p90)}")
    lines.append(f"- Worst **p90 IS delta**: {worst_p90['delta_p90']:.6f} at {fmt_row(worst_p90)}")
    lines.append("")
    lines.append(f"- Best **fill-rate delta**: {best_fill['delta_fill_rate']:.3f} at {fmt_row(best_fill)}")
    lines.append(f"- Worst **fill-rate delta**: {worst_fill['delta_fill_rate']:.3f} at {fmt_row(worst_fill)}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ensure_dirs()

    df = pd.read_csv(CSV_PATH)

    # Safety: compute fill delta here too
    df["delta_fill_rate"] = df["ToxicityAware_fill_rate"] - df["AlwaysMarket_fill_rate"]

    # Heatmaps
    hm_mean = pivot_heatmap(df, "delta_mean")
    hm_p90 = pivot_heatmap(df, "delta_p90")
    hm_fill = pivot_heatmap(df, "delta_fill_rate")

    plot_heatmap(hm_mean, "Regime grid: delta_mean (ToxicityAware - AlwaysMarket)", FIG_DELTA_MEAN)
    plot_heatmap(hm_p90, "Regime grid: delta_p90 (ToxicityAware - AlwaysMarket)", FIG_DELTA_P90)
    plot_heatmap(hm_fill, "Regime grid: delta_fill_rate (ToxicityAware - AlwaysMarket)", FIG_FILLRATE)

    # Report
    summary = compute_summary(df)

    md = []
    md.append("# Regime Report — Execution under Adverse Selection\n")
    md.append("Artifacts generated from `reports/regime_grid.csv`.\n")
    md.append(summary)
    md.append("\n## Figures\n")
    md.append(f"- ![delta_mean]({FIG_DELTA_MEAN})")
    md.append(f"- ![delta_p90]({FIG_DELTA_P90})")
    md.append(f"- ![delta_fill_rate]({FIG_FILLRATE})")
    md.append("")

    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print(f"Saved figures in: {FIG_DIR}")
    print(f"Saved report: {REPORT_MD}")


if __name__ == "__main__":
    main()
