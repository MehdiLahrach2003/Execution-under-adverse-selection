"""
make_misspec_report.py

Build a markdown report + heatmaps from reports/misspec_grid.csv.

We support the column names produced by your generator:
- tox_persist_true
- tox_persist_model
- as_kick_scale
and deltas + baseline stats.

We also add risk-adjusted deltas:
- delta_mean_rel = delta_mean / abs(AlwaysMarket_mean)
- delta_p90_rel  = delta_p90  / abs(AlwaysMarket_p90)
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
CSV_PATH = REPORTS_DIR / "misspec_grid.csv"
OUT_MD = REPORTS_DIR / "MISSPEC_REPORT.md"


# Your actual generator columns (based on your error log / available list)
COLS_REQUIRED = [
    "tox_persist_true",
    "tox_persist_model",
    "as_kick_scale",
    "AlwaysMarket_mean",
    "ToxicityAware_mean",
    "delta_mean",
    "AlwaysMarket_p90",
    "ToxicityAware_p90",
    "delta_p90",
    "AlwaysMarket_fill_rate",
    "ToxicityAware_fill_rate",
    "delta_fill_rate",
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
            "misspec_grid.csv is missing expected columns:\n"
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
        raise FileNotFoundError(f"Missing {CSV_PATH}. Run the misspec grid script first.")

    df = pd.read_csv(CSV_PATH)
    _assert_cols(df, COLS_REQUIRED)

    # Ensure numeric
    for c in COLS_REQUIRED:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Risk-adjusted deltas
    df["delta_mean_rel"] = _safe_div(df["delta_mean"].to_numpy(), df["AlwaysMarket_mean"].to_numpy())
    df["delta_p90_rel"] = _safe_div(df["delta_p90"].to_numpy(), df["AlwaysMarket_p90"].to_numpy())

    return df


def _pivot(df: pd.DataFrame, value_col: str, tox_model_value: float) -> pd.DataFrame:
    # slice for fixed model belief, then pivot true regime vs as_kick_scale
    sub = df[np.isclose(df["tox_persist_model"].to_numpy(), tox_model_value)].copy()
    piv = sub.pivot(index="tox_persist_true", columns="as_kick_scale", values=value_col)
    piv = piv.sort_index(axis=0).sort_index(axis=1)
    return piv


def _plot_heatmap(piv: pd.DataFrame, title: str, out_path: Path) -> None:
    arr = piv.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(arr, aspect="auto")

    ax.set_title(title)
    ax.set_xlabel("as_kick_scale (true)")
    ax.set_ylabel("tox_persist_true")

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


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = _load()

    # For misspec plots we condition on tox_persist_model (belief).
    model_vals = sorted(df["tox_persist_model"].dropna().unique().tolist())

    # Global best/worst across all beliefs + truths
    best_mean, worst_mean = _best_worst(df, "delta_mean")
    best_p90, worst_p90 = _best_worst(df, "delta_p90")
    best_mean_rel, worst_mean_rel = _best_worst(df, "delta_mean_rel")
    best_p90_rel, worst_p90_rel = _best_worst(df, "delta_p90_rel")

    # Build figures: for each belief, heatmap vs true regime
    specs = [
        HeatmapSpec("delta_mean", "Misspec grid: delta_mean (ToxicityAware - AlwaysMarket)", "misspec_delta_mean"),
        HeatmapSpec("delta_p90", "Misspec grid: delta_p90 (ToxicityAware - AlwaysMarket)", "misspec_delta_p90"),
        HeatmapSpec("delta_fill_rate", "Misspec grid: delta_fill_rate (ToxicityAware - AlwaysMarket)", "misspec_delta_fillrate"),
        HeatmapSpec("delta_mean_rel", "Misspec grid: delta_mean_rel (delta_mean / |AlwaysMarket_mean|)", "misspec_delta_mean_rel"),
        HeatmapSpec("delta_p90_rel", "Misspec grid: delta_p90_rel (delta_p90 / |AlwaysMarket_p90|)", "misspec_delta_p90_rel"),
    ]

    fig_links: list[str] = []

    for tox_model in model_vals:
        for spec in specs:
            piv = _pivot(df, spec.value_col, tox_model_value=float(tox_model))
            out_name = f"{spec.filename}_modeltox_{float(tox_model):.2f}.png"
            out_path = FIG_DIR / out_name
            title = f"{spec.title} — belief tox_persist_model={float(tox_model):.2f}"
            _plot_heatmap(piv, title, out_path)
            fig_links.append(f"- ![{spec.value_col}](reports/figures/{out_name})")

    # Write report
    lines: list[str] = []
    lines.append("# Misspecification Report — Execution under Adverse Selection\n")
    lines.append(f"Artifacts generated from `{CSV_PATH.as_posix()}`.\n")
    lines.append("## What this evaluates\n")
    lines.append("We compare a policy that uses a *belief* about market toxicity persistence (`tox_persist_model`) ")
    lines.append("while the *true* market regime is `tox_persist_true` and `as_kick_scale`.\n")
    lines.append("All deltas are `ToxicityAware - AlwaysMarket` (negative = improvement if IS is a cost).\n")

    lines.append("## Best / worst (raw deltas)\n")
    lines.append(
        f"- Best **mean IS delta**: {best_mean['delta_mean']:.6f} at "
        f"(as_kick_scale={best_mean['as_kick_scale']:.2f}, tox_true={best_mean['tox_persist_true']:.2f}, tox_model={best_mean['tox_persist_model']:.2f})"
    )
    lines.append(
        f"- Worst **mean IS delta**: {worst_mean['delta_mean']:.6f} at "
        f"(as_kick_scale={worst_mean['as_kick_scale']:.2f}, tox_true={worst_mean['tox_persist_true']:.2f}, tox_model={worst_mean['tox_persist_model']:.2f})\n"
    )

    lines.append(
        f"- Best **p90 IS delta**: {best_p90['delta_p90']:.6f} at "
        f"(as_kick_scale={best_p90['as_kick_scale']:.2f}, tox_true={best_p90['tox_persist_true']:.2f}, tox_model={best_p90['tox_persist_model']:.2f})"
    )
    lines.append(
        f"- Worst **p90 IS delta**: {worst_p90['delta_p90']:.6f} at "
        f"(as_kick_scale={worst_p90['as_kick_scale']:.2f}, tox_true={worst_p90['tox_persist_true']:.2f}, tox_model={worst_p90['tox_persist_model']:.2f})\n"
    )

    lines.append("## Best / worst (risk-adjusted deltas)\n")
    lines.append("We use `delta_*_rel = delta_* / abs(AlwaysMarket_*)` to interpret improvements in relative terms.\n")
    lines.append(
        f"- Best **mean IS delta (rel)**: {best_mean_rel['delta_mean_rel']:.6f} at "
        f"(as_kick_scale={best_mean_rel['as_kick_scale']:.2f}, tox_true={best_mean_rel['tox_persist_true']:.2f}, tox_model={best_mean_rel['tox_persist_model']:.2f})"
    )
    lines.append(
        f"- Worst **mean IS delta (rel)**: {worst_mean_rel['delta_mean_rel']:.6f} at "
        f"(as_kick_scale={worst_mean_rel['as_kick_scale']:.2f}, tox_true={worst_mean_rel['tox_persist_true']:.2f}, tox_model={worst_mean_rel['tox_persist_model']:.2f})\n"
    )
    lines.append(
        f"- Best **p90 IS delta (rel)**: {best_p90_rel['delta_p90_rel']:.6f} at "
        f"(as_kick_scale={best_p90_rel['as_kick_scale']:.2f}, tox_true={best_p90_rel['tox_persist_true']:.2f}, tox_model={best_p90_rel['tox_persist_model']:.2f})"
    )
    lines.append(
        f"- Worst **p90 IS delta (rel)**: {worst_p90_rel['delta_p90_rel']:.6f} at "
        f"(as_kick_scale={worst_p90_rel['as_kick_scale']:.2f}, tox_true={worst_p90_rel['tox_persist_true']:.2f}, tox_model={worst_p90_rel['tox_persist_model']:.2f})\n"
    )

    lines.append("## Figures (grouped by model belief)\n")
    lines.extend(fig_links)

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Saved figures in: {FIG_DIR.as_posix()}")
    print(f"Saved report: {OUT_MD.as_posix()}")


if __name__ == "__main__":
    main()
