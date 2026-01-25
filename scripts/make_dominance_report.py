"""
make_dominance_report.py

Create dominance / robustness diagnostics from:
- reports/regime_grid.csv
- reports/misspec_grid.csv

Outputs:
- reports/DOMINANCE_REPORT.md
- figures in reports/figures/

We define:
- Dominance (regime): ToxicityAware dominates AlwaysMarket if:
    delta_mean < 0 AND delta_p90 < 0
  dominated if:
    delta_mean > 0 AND delta_p90 > 0
  tradeoff otherwise.

- Robustness (misspec): use regret metrics if available:
    regret_mean, regret_p90
  Robust if regret_p90 <= threshold.

If regret columns do not exist in misspec grid yet, this script will compute
proxy regrets from deltas (valid because regret = max(delta, 0) when comparing against AlwaysMarket).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


REPORTS_DIR = Path("reports")
FIG_DIR = REPORTS_DIR / "figures"
REGIME_CSV = REPORTS_DIR / "regime_grid.csv"
MISSPEC_CSV = REPORTS_DIR / "misspec_grid.csv"
OUT_MD = REPORTS_DIR / "DOMINANCE_REPORT.md"


# ----------------------------
# Helpers
# ----------------------------

def _ensure_dirs() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def _assert_cols(df: pd.DataFrame, required: Iterable[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{name} is missing expected columns:\n"
            f"- Missing: {missing}\n"
            f"- Available: {list(df.columns)}\n"
        )


def _to_num(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="ignore")
    return df


def _plot_heatmap_numeric(piv: pd.DataFrame, title: str, out_path: Path) -> None:
    arr = piv.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(arr, aspect="auto")

    ax.set_title(title)
    ax.set_xlabel(str(piv.columns.name) if piv.columns.name else "x")
    ax.set_ylabel(str(piv.index.name) if piv.index.name else "y")

    ax.set_xticks(np.arange(piv.shape[1]))
    ax.set_yticks(np.arange(piv.shape[0]))
    ax.set_xticklabels([f"{x:.2f}" for x in piv.columns.to_numpy(dtype=float)])
    ax.set_yticklabels([f"{y:.2f}" for y in piv.index.to_numpy(dtype=float)])

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("value")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_heatmap_categorical(piv: pd.DataFrame, title: str, out_path: Path) -> None:
    """
    piv contains values in {-1, 0, +1}:
      +1 = dominates, -1 = dominated, 0 = tradeoff
    We'll plot as numeric heatmap with annotated labels.
    """
    arr = piv.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(arr, aspect="auto")

    ax.set_title(title)
    ax.set_xlabel(str(piv.columns.name) if piv.columns.name else "x")
    ax.set_ylabel(str(piv.index.name) if piv.index.name else "y")

    ax.set_xticks(np.arange(piv.shape[1]))
    ax.set_yticks(np.arange(piv.shape[0]))
    ax.set_xticklabels([f"{x:.2f}" for x in piv.columns.to_numpy(dtype=float)])
    ax.set_yticklabels([f"{y:.2f}" for y in piv.index.to_numpy(dtype=float)])

    def lab(v: float) -> str:
        if v > 0.5:
            return "DOM"
        if v < -0.5:
            return "BAD"
        return "TRD"

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            if np.isfinite(v):
                ax.text(j, i, lab(v), ha="center", va="center", fontsize=9)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("DOM=+1, TRD=0, BAD=-1")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


# ----------------------------
# Regime dominance
# ----------------------------

def _load_regime() -> pd.DataFrame:
    if not REGIME_CSV.exists():
        raise FileNotFoundError(f"Missing {REGIME_CSV}")

    df = pd.read_csv(REGIME_CSV)
    df = _to_num(df)

    req = ["as_kick_scale", "tox_persist", "delta_mean", "delta_p90"]
    _assert_cols(df, req, "regime_grid.csv")

    # Dominance coding: +1 dominates, -1 dominated, 0 tradeoff
    dom = np.zeros(len(df), dtype=int)
    dom[(df["delta_mean"] < 0) & (df["delta_p90"] < 0)] = 1
    dom[(df["delta_mean"] > 0) & (df["delta_p90"] > 0)] = -1
    df["dominance"] = dom

    return df


def _pivot_regime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    piv = df.pivot(index="tox_persist", columns="as_kick_scale", values=col)
    piv = piv.sort_index(axis=0).sort_index(axis=1)
    piv.index.name = "tox_persist"
    piv.columns.name = "as_kick_scale"
    return piv


# ----------------------------
# Misspec robustness
# ----------------------------

@dataclass(frozen=True)
class RobustThresholds:
    regret_p90: float = 0.0  # can be adjusted: e.g. 0.0005, depending on scale


def _load_misspec() -> pd.DataFrame:
    if not MISSPEC_CSV.exists():
        raise FileNotFoundError(f"Missing {MISSPEC_CSV}")

    df = pd.read_csv(MISSPEC_CSV)
    df = _to_num(df)

    req = ["tox_persist_true", "tox_persist_model", "as_kick_scale", "delta_mean", "delta_p90"]
    _assert_cols(df, req, "misspec_grid.csv")

    # If regret columns are missing, build proxy regrets from deltas.
    # regret = max(delta, 0) when the alternative is AlwaysMarket baseline.
    if "regret_mean" not in df.columns:
        df["regret_mean"] = np.maximum(df["delta_mean"].astype(float), 0.0)
    if "regret_p90" not in df.columns:
        df["regret_p90"] = np.maximum(df["delta_p90"].astype(float), 0.0)

    return df


def _robust_fraction_by_belief(df: pd.DataFrame, thr: RobustThresholds) -> pd.DataFrame:
    """
    For each belief tox_persist_model, compute fraction of (true tox, as) pairs that are robust.
    robust := regret_p90 <= thr.regret_p90
    """
    rows = []
    for tox_model, g in df.groupby("tox_persist_model"):
        g = g.copy()
        robust = (g["regret_p90"].astype(float) <= float(thr.regret_p90)).mean()
        rows.append({"tox_persist_model": float(tox_model), "robust_frac": float(robust), "n": int(len(g))})
    out = pd.DataFrame(rows).sort_values("tox_persist_model").reset_index(drop=True)
    return out


def _pivot_misspec(df: pd.DataFrame, value_col: str, tox_model_value: float) -> pd.DataFrame:
    sub = df[np.isclose(df["tox_persist_model"].astype(float), float(tox_model_value))].copy()
    piv = sub.pivot(index="tox_persist_true", columns="as_kick_scale", values=value_col)
    piv = piv.sort_index(axis=0).sort_index(axis=1)
    piv.index.name = "tox_persist_true"
    piv.columns.name = "as_kick_scale"
    return piv


# ----------------------------
# Report
# ----------------------------

def main() -> None:
    _ensure_dirs()

    # --- Regime dominance
    reg = _load_regime()
    piv_dom = _pivot_regime(reg, "dominance")
    p_dom = FIG_DIR / "regime_dominance_heatmap.png"
    _plot_heatmap_categorical(
        piv_dom,
        "Regime dominance map: DOM (both mean & p90 improve), BAD (both worsen), TRD (trade-off)",
        p_dom,
    )

    # Add also a heatmap of how 'often' TA dominates in the regime grid
    share_dom = float((reg["dominance"] == 1).mean())
    share_bad = float((reg["dominance"] == -1).mean())
    share_trd = float((reg["dominance"] == 0).mean())

    # --- Misspec robustness
    mis = _load_misspec()
    thr = RobustThresholds(regret_p90=0.0)  # strict: zero-regret
    robust_table = _robust_fraction_by_belief(mis, thr)

    # Plot robustness fraction vs belief (simple bar plot)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(robust_table["tox_persist_model"].astype(float), robust_table["robust_frac"].astype(float))
    ax.set_title("Robustness by belief: fraction of scenarios with regret_p90 <= threshold")
    ax.set_xlabel("tox_persist_model (belief)")
    ax.set_ylabel("robust_frac")
    fig.tight_layout()
    p_rob = FIG_DIR / "misspec_robust_fraction_by_belief.png"
    fig.savefig(p_rob, dpi=180)
    plt.close(fig)

    # Also output a robustness heatmap per belief for regret_p90
    model_vals = sorted(mis["tox_persist_model"].dropna().unique().tolist())
    fig_links: list[str] = []
    for tm in model_vals:
        piv_r = _pivot_misspec(mis, "regret_p90", tox_model_value=float(tm))
        out_name = f"misspec_regret_p90_modeltox_{float(tm):.2f}.png"
        out_path = FIG_DIR / out_name
        _plot_heatmap_numeric(piv_r, f"Misspec regret_p90 heatmap — belief tox_persist_model={float(tm):.2f}", out_path)
        fig_links.append(f"- ![regret_p90](reports/figures/{out_name})")

    # --- Write report
    lines: list[str] = []
    lines.append("# Dominance & Robustness Report — Execution under Adverse Selection\n")
    lines.append("This report summarizes **where** ToxicityAware dominates AlwaysMarket and how robust it is under model misspecification.\n")

    lines.append("## Regime dominance\n")
    lines.append(
        "We label each (tox_persist, as_kick_scale) regime as:\n"
        "- **DOM**: delta_mean < 0 AND delta_p90 < 0 (dominates)\n"
        "- **BAD**: delta_mean > 0 AND delta_p90 > 0 (dominated)\n"
        "- **TRD**: trade-off (one improves, the other worsens)\n"
    )
    lines.append(f"- Share DOM: {share_dom:.3f}\n- Share BAD: {share_bad:.3f}\n- Share TRD: {share_trd:.3f}\n")
    lines.append(f"- ![regime_dominance](reports/figures/{p_dom.name})\n")

    lines.append("## Misspecification robustness\n")
    lines.append(
        "We quantify robustness using **regret_p90**. If regret columns are not present in the CSV, "
        "we use a proxy: regret_p90 = max(delta_p90, 0).\n"
    )
    lines.append(f"- Robustness threshold: regret_p90 <= {thr.regret_p90:.6f}\n")
    lines.append(f"- ![robust_by_belief](reports/figures/{p_rob.name})\n")

    lines.append("### Robustness heatmaps (regret_p90)\n")
    lines.extend(fig_links)

    # Table snapshot
    lines.append("\n## Robustness table\n")
    lines.append("Columns: tox_persist_model, robust_frac, n\n")
    lines.append("```\n")
    lines.append(robust_table.to_string(index=False))
    lines.append("\n```\n")

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Saved figures in: {FIG_DIR.as_posix()}")
    print(f"Saved report: {OUT_MD.as_posix()}")


if __name__ == "__main__":
    main()
