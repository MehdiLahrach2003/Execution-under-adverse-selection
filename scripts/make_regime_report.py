"""
make_regime_report.py

Build a markdown report + heatmaps from reports/regime_grid.csv.

This version matches YOUR current regime_grid.csv columns:
Available:
- as_kick_scale, tox_persist
- AlwaysMarket_mean, AlwaysMarket_p90, AlwaysMarket_fill_rate, AlwaysMarket_avg_first_fill_t
- ToxicityAware_mean, ToxicityAware_p90, ToxicityAware_fill_rate, ToxicityAware_avg_first_fill_t
- delta_mean, delta_p90, delta_avg_first_fill_t

We will:
- compute delta_fill_rate = ToxicityAware_fill_rate - AlwaysMarket_fill_rate (if missing)
- add risk-adjusted deltas:
    delta_mean_rel = delta_mean / abs(AlwaysMarket_mean)
    delta_p90_rel  = delta_p90  / abs(AlwaysMarket_p90)
- plot heatmaps for:
    delta_mean, delta_p90, delta_fill_rate, delta_avg_first_fill_t, delta_mean_rel, delta_p90_rel
- write reports/REGIME_REPORT.md
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


REPORTS_DIR = Path("reports")
FIG_DIR = REPORTS_DIR / "figures"
CSV_PATH = REPORTS_DIR / "regime_grid.csv"
OUT_MD = REPORTS_DIR / "REGIME_REPORT.md"


# Minimal columns we truly need
COLS_REQUIRED = [
    "as_kick_scale",
    "tox_persist",
    "AlwaysMarket_mean",
    "ToxicityAware_mean",
    "delta_mean",
    "AlwaysMarket_p90",
    "ToxicityAware_p90",
    "delta_p90",
    "AlwaysMarket_fill_rate",
    "ToxicityAware_fill_rate",
    # delta_fill_rate may be missing -> we compute it
    # delta_avg_first_fill_t exists and is useful
    "delta_avg_first_fill_t",
]


@dataclass(frozen=True)
class HeatmapSpec:
    value_col: str
    title: str
    filename: str


def _assert_cols(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "regime_grid.csv is missing expected columns:\n"
            f"- Missing: {missing}\n"
            f"- Available: {list(df.columns)}\n"
        )


def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    den_abs = np.abs(den)
    out = np.full_like(num, fill_value=np.nan, dtype=float)
    mask = den_abs > 0.0
    out[mask] = num[mask] / den_abs[mask]
    return out


def _load() -> pd.DataFrame:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Missing {CSV_PATH}. Run the regime grid script first.")

    df = pd.read_csv(CSV_PATH)
    _assert_cols(df, COLS_REQUIRED)

    # numeric conversion
    for c in df.columns:
        if c in [
            "as_kick_scale",
            "tox_persist",
            "AlwaysMarket_mean",
            "ToxicityAware_mean",
            "delta_mean",
            "AlwaysMarket_p90",
            "ToxicityAware_p90",
            "delta_p90",
            "AlwaysMarket_fill_rate",
            "ToxicityAware_fill_rate",
            "AlwaysMarket_avg_first_fill_t",
            "ToxicityAware_avg_first_fill_t",
            "delta_avg_first_fill_t",
        ]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # If delta_fill_rate missing, compute it
    if "delta_fill_rate" not in df.columns:
        df["delta_fill_rate"] = df["ToxicityAware_fill_rate"] - df["AlwaysMarket_fill_rate"]

    # Add risk-adjusted deltas
    df["delta_mean_rel"] = _safe_div(df["delta_mean"].to_numpy(), df["AlwaysMarket_mean"].to_numpy())
    df["delta_p90_rel"] = _safe_div(df["delta_p90"].to_numpy(), df["AlwaysMarket_p90"].to_numpy())

    return df


def _pivot(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    piv = df.pivot(index="tox_persist", columns="as_kick_scale", values=value_col)
    piv = piv.sort_index(axis=0).sort_index(axis=1)
    return piv


def _plot_heatmap(piv: pd.DataFrame, title: str, out_path: Path) -> None:
    arr = piv.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(arr, aspect="auto")

    ax.set_title(title)
    ax.set_xlabel("as_kick_scale")
    ax.set_ylabel("tox_persist")

    ax.set_xticks(np.arange(piv.shape[1]))
    ax.set_yticks(np.arange(piv.shape[0]))
    ax.set_xticklabels([f"{x:.2f}" for x in piv.columns.to_numpy(dtype=float)])
    ax.set_yticklabels([f"{y:.2f}" for y in piv.index.to_numpy(dtype=float)])

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=9)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("value")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _best_worst(df: pd.DataFrame, col: str) -> tuple[pd.Series, pd.Series]:
    d = df[np.isfinite(df[col].to_numpy())].copy()
    if d.empty:
        return df.iloc[0], df.iloc[0]
    best = d.loc[d[col].idxmin()]
    worst = d.loc[d[col].idxmax()]
    return best, worst


def _write_report(df: pd.DataFrame, specs: list[HeatmapSpec]) -> None:
    best_mean, worst_mean = _best_worst(df, "delta_mean")
    best_p90, worst_p90 = _best_worst(df, "delta_p90")
    best_mean_rel, worst_mean_rel = _best_worst(df, "delta_mean_rel")
    best_p90_rel, worst_p90_rel = _best_worst(df, "delta_p90_rel")
    best_fill, worst_fill = _best_worst(df, "delta_fill_rate")
    best_firstfill, worst_firstfill = _best_worst(df, "delta_avg_first_fill_t")

    def at(r: pd.Series) -> str:
        return f"(as_kick_scale={r['as_kick_scale']:.2f}, tox_persist={r['tox_persist']:.2f})"

    lines: list[str] = []
    lines.append("# Regime Report — Execution under Adverse Selection\n")
    lines.append(f"Artifacts generated from `{CSV_PATH.as_posix()}`.\n")
    lines.append("## What this evaluates\n")
    lines.append(
        "We sweep the *true* market regime parameters:\n"
        "- `as_kick_scale` (adverse-selection kick intensity)\n"
        "- `tox_persist` (toxicity persistence)\n\n"
        "We compare `ToxicityAware` vs `AlwaysMarket`.\n"
        "All deltas are `ToxicityAware - AlwaysMarket` (negative = improvement if IS is a cost).\n"
    )

    lines.append("## Key takeaways (raw deltas)\n")
    lines.append(f"- Best **mean IS delta**: {best_mean['delta_mean']:.6f} at {at(best_mean)}")
    lines.append(f"- Worst **mean IS delta**: {worst_mean['delta_mean']:.6f} at {at(worst_mean)}\n")
    lines.append(f"- Best **p90 IS delta**: {best_p90['delta_p90']:.6f} at {at(best_p90)}")
    lines.append(f"- Worst **p90 IS delta**: {worst_p90['delta_p90']:.6f} at {at(worst_p90)}\n")

    lines.append("## Key takeaways (risk-adjusted deltas)\n")
    lines.append("We report relative deltas to normalize by baseline magnitude:\n")
    lines.append("- `delta_mean_rel = delta_mean / abs(AlwaysMarket_mean)`\n")
    lines.append("- `delta_p90_rel  = delta_p90 / abs(AlwaysMarket_p90)`\n")
    lines.append(f"- Best **mean IS delta (rel)**: {best_mean_rel['delta_mean_rel']:.6f} at {at(best_mean_rel)}")
    lines.append(f"- Worst **mean IS delta (rel)**: {worst_mean_rel['delta_mean_rel']:.6f} at {at(worst_mean_rel)}\n")
    lines.append(f"- Best **p90 IS delta (rel)**: {best_p90_rel['delta_p90_rel']:.6f} at {at(best_p90_rel)}")
    lines.append(f"- Worst **p90 IS delta (rel)**: {worst_p90_rel['delta_p90_rel']:.6f} at {at(worst_p90_rel)}\n")

    lines.append("## Execution quality metrics\n")
    lines.append(f"- Best **delta_fill_rate**: {best_fill['delta_fill_rate']:.6f} at {at(best_fill)}")
    lines.append(f"- Worst **delta_fill_rate**: {worst_fill['delta_fill_rate']:.6f} at {at(worst_fill)}\n")
    lines.append(
        "We also track `avg_first_fill_t` (average time-to-first-fill). "
        "A negative `delta_avg_first_fill_t` indicates ToxicityAware fills earlier.\n"
    )
    lines.append(f"- Best **delta_avg_first_fill_t**: {best_firstfill['delta_avg_first_fill_t']:.6f} at {at(best_firstfill)}")
    lines.append(f"- Worst **delta_avg_first_fill_t**: {worst_firstfill['delta_avg_first_fill_t']:.6f} at {at(worst_firstfill)}\n")

    lines.append("## Figures\n")
    for spec in specs:
        lines.append(f"- ![{spec.value_col}](reports/figures/{spec.filename})")

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = _load()

    specs = [
        HeatmapSpec("delta_mean", "Regime grid: delta_mean (ToxicityAware - AlwaysMarket)", "regime_delta_mean_heatmap.png"),
        HeatmapSpec("delta_p90", "Regime grid: delta_p90 (ToxicityAware - AlwaysMarket)", "regime_delta_p90_heatmap.png"),
        HeatmapSpec("delta_fill_rate", "Regime grid: delta_fill_rate (ToxicityAware - AlwaysMarket)", "regime_delta_fillrate_delta_heatmap.png"),
        HeatmapSpec("delta_avg_first_fill_t", "Regime grid: delta_avg_first_fill_t (ToxicityAware - AlwaysMarket)", "regime_delta_avg_first_fill_t_heatmap.png"),
        HeatmapSpec("delta_mean_rel", "Regime grid: delta_mean_rel (delta_mean / |AlwaysMarket_mean|)", "regime_delta_mean_rel_heatmap.png"),
        HeatmapSpec("delta_p90_rel", "Regime grid: delta_p90_rel (delta_p90 / |AlwaysMarket_p90|)", "regime_delta_p90_rel_heatmap.png"),
    ]

    for spec in specs:
        piv = _pivot(df, spec.value_col)
        out_path = FIG_DIR / spec.filename
        _plot_heatmap(piv, spec.title, out_path)

    _write_report(df, specs)

    print(f"Saved figures in: {FIG_DIR.as_posix()}")
    print(f"Saved report: {OUT_MD.as_posix()}")


if __name__ == "__main__":
    main()
