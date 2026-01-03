# scripts/run_regime_grid.py
"""
Regime analysis for execution under adverse selection.

We sweep two market knobs:
- as_kick_scale
- tox_persist

BacktestParams (from your traceback) has only:
- horizon
- parent_qty
- side

So MarketParams must be passed separately to run_backtest.
"""

from __future__ import annotations

from dataclasses import is_dataclass, fields
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from execlab.sim.market import MarketParams


# -----------------------------
# Experiment configuration
# -----------------------------
SEEDS = list(range(50))
AS_KICK_SCALES = [0.00, 0.01, 0.02, 0.03, 0.05]
TOX_PERSIST = [0.60, 0.75, 0.85, 0.95]

PARENT_QTY = 1.0
HORIZON = 200          # discrete steps (keep simple)
SIDE = "buy"           # or "sell" depending on your enum; we'll keep as string

OUT_CSV = "reports/regime_grid.csv"


# -----------------------------
# Helpers
# -----------------------------
def ensure_reports_dir() -> None:
    import os
    os.makedirs("reports", exist_ok=True)


def mean(x: List[float]) -> float:
    return float(np.mean(np.asarray(x, dtype=float))) if len(x) else float("nan")


def q90(x: List[float]) -> float:
    return float(np.quantile(np.asarray(x, dtype=float), 0.9)) if len(x) else float("nan")


def _get_field(out: Any, name: str, default: Any = None) -> Any:
    if isinstance(out, dict):
        return out.get(name, default)
    return getattr(out, name, default)


# -----------------------------
# Dynamic resolution
# -----------------------------
def resolve_runner() -> Callable[..., Any]:
    import execlab.backtest.mvp as mvp
    fn = getattr(mvp, "run_backtest", None)
    if callable(fn):
        return fn
    public = [n for n in dir(mvp) if not n.startswith("_")]
    raise ImportError(
        "Expected execlab.backtest.mvp.run_backtest to exist, but it was not found. "
        f"Available symbols: {public}"
    )


def resolve_backtest_params_class():
    import execlab.backtest.mvp as mvp
    cls = getattr(mvp, "BacktestParams", None)
    if cls is None:
        public = [n for n in dir(mvp) if not n.startswith("_")]
        raise ImportError(
            "Expected execlab.backtest.mvp.BacktestParams to exist, but it was not found. "
            f"Available symbols: {public}"
        )
    return cls


def resolve_policy_instance(kind: str) -> Any:
    import execlab.strategy.execution as ex

    if kind == "always_market":
        candidates = ["AlwaysMarketExecution", "AlwaysMarket", "AlwaysMarketPolicy", "AlwaysMarketExec"]
        for name in candidates:
            cls = getattr(ex, name, None)
            if isinstance(cls, type):
                return cls()
        for n in dir(ex):
            if n.startswith("_"):
                continue
            obj = getattr(ex, n)
            if isinstance(obj, type):
                low = n.lower()
                if "market" in low and ("always" in low or "simple" in low or "immediate" in low):
                    return obj()

    if kind == "toxicity_aware":
        candidates = ["ToxicityAwareExecution", "ToxicityAware", "ToxicityFilterExecution", "ToxicityBasedExecution"]
        for name in candidates:
            cls = getattr(ex, name, None)
            if isinstance(cls, type):
                return cls()
        for n in dir(ex):
            if n.startswith("_"):
                continue
            obj = getattr(ex, n)
            if isinstance(obj, type):
                low = n.lower()
                if "tox" in low or "toxic" in low:
                    return obj()

    available = [
        n for n in dir(ex)
        if not n.startswith("_") and isinstance(getattr(ex, n), type)
    ]
    raise ImportError(f"Could not resolve policy kind='{kind}'. Available classes: {available}")


# -----------------------------
# Params builders
# -----------------------------
def build_market_params(as_kick_scale: float, tox_persist: float) -> MarketParams:
    return MarketParams(
        as_kick_scale=float(as_kick_scale),
        tox_persist=float(tox_persist),
    )


def build_backtest_params() -> Any:
    BacktestParams = resolve_backtest_params_class()
    bp = BacktestParams()  # rely on defaults if any

    # We KNOW fields are: horizon, parent_qty, side
    # We'll set them directly if present.
    if hasattr(bp, "horizon"):
        setattr(bp, "horizon", int(HORIZON))
    if hasattr(bp, "parent_qty"):
        setattr(bp, "parent_qty", float(PARENT_QTY))
    if hasattr(bp, "side"):
        setattr(bp, "side", SIDE)

    return bp


# -----------------------------
# Runner call logic
# -----------------------------
def call_runner(runner: Callable[..., Any], policy: Any, bp: Any, mp: MarketParams, seed: int) -> Any:
    """
    Try several signatures until one works.
    We already know runner expects bp.parent_qty, so bp must be BacktestParams.
    MarketParams is passed separately.
    """
    candidates = [
        # Most likely
        {"policy": policy, "bp": bp, "market_params": mp, "seed": seed},
        {"policy": policy, "bp": bp, "mp": mp, "seed": seed},
        {"policy": policy, "bp": bp, "market": mp, "seed": seed},

        # Some codebases use params naming
        {"policy": policy, "params": bp, "market_params": mp, "seed": seed},
        {"policy": policy, "backtest_params": bp, "market_params": mp, "seed": seed},

        # Maybe no seed kwarg
        {"policy": policy, "bp": bp, "market_params": mp},
        {"policy": policy, "bp": bp, "mp": mp},
    ]

    for kw in candidates:
        try:
            return runner(**kw)
        except TypeError:
            pass

    # Positional fallback variants
    for args in (
        (policy, bp, mp, seed),
        (policy, bp, mp),
        (policy, bp, seed),
        (policy, bp),
    ):
        try:
            return runner(*args)
        except TypeError:
            pass

    raise TypeError(
        "Could not call run_backtest with any known signature. "
        "Please open src/execlab/backtest/mvp.py and tell me the def run_backtest(...) signature."
    )


def run_policy_on_regime(
    runner: Callable[..., Any],
    policy_name: str,
    policy: Any,
    bp: Any,
    mp: MarketParams,
    seeds: List[int],
) -> Dict[str, float]:
    costs: List[float] = []
    fill_times: List[float] = []
    fill_rates: List[float] = []

    for seed in seeds:
        out = call_runner(runner, policy, bp, mp, seed)

        is_cost = float(_get_field(out, "implementation_shortfall"))
        remaining = float(_get_field(out, "remaining"))

        first_fill_t = _get_field(out, "first_fill_t", None)
        if first_fill_t is not None:
            fill_times.append(float(first_fill_t))

        costs.append(is_cost)
        fill_rates.append(1.0 if remaining <= 1e-12 else 0.0)

    return {
        f"{policy_name}_mean": mean(costs),
        f"{policy_name}_p90": q90(costs),
        f"{policy_name}_fill_rate": mean(fill_rates),
        f"{policy_name}_avg_first_fill_t": mean(fill_times),
    }


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ensure_reports_dir()

    runner = resolve_runner()
    print(f"[regime_grid] Using runner: execlab.backtest.mvp.{runner.__name__}")

    always_market = resolve_policy_instance("always_market")
    toxicity_aware = resolve_policy_instance("toxicity_aware")

    print(f"[regime_grid] Using policy AlwaysMarket: {always_market.__class__.__name__}")
    print(f"[regime_grid] Using policy ToxicityAware: {toxicity_aware.__class__.__name__}")

    # BacktestParams is fixed across regimes in this sweep
    bp = build_backtest_params()

    rows: List[Dict[str, float]] = []

    for as_kick_scale in AS_KICK_SCALES:
        for tox_persist in TOX_PERSIST:
            mp = build_market_params(as_kick_scale=float(as_kick_scale), tox_persist=float(tox_persist))

            row: Dict[str, float] = {
                "as_kick_scale": float(as_kick_scale),
                "tox_persist": float(tox_persist),
            }

            row.update(run_policy_on_regime(runner, "AlwaysMarket", always_market, bp, mp, SEEDS))
            row.update(run_policy_on_regime(runner, "ToxicityAware", toxicity_aware, bp, mp, SEEDS))

            row["delta_mean"] = row["ToxicityAware_mean"] - row["AlwaysMarket_mean"]
            row["delta_p90"] = row["ToxicityAware_p90"] - row["AlwaysMarket_p90"]
            row["delta_avg_first_fill_t"] = (
                row["ToxicityAware_avg_first_fill_t"] - row["AlwaysMarket_avg_first_fill_t"]
            )

            rows.append(row)

    df = pd.DataFrame(rows).sort_values(["as_kick_scale", "tox_persist"]).reset_index(drop=True)
    df.to_csv(OUT_CSV, index=False)

    key_cols = [
        "as_kick_scale",
        "tox_persist",
        "AlwaysMarket_mean",
        "ToxicityAware_mean",
        "delta_mean",
        "AlwaysMarket_p90",
        "ToxicityAware_p90",
        "delta_p90",
        "AlwaysMarket_avg_first_fill_t",
        "ToxicityAware_avg_first_fill_t",
        "delta_avg_first_fill_t",
        "AlwaysMarket_fill_rate",
        "ToxicityAware_fill_rate",
    ]

    print("\n=== Regime grid summary (key columns) ===")
    print(df[key_cols].to_string(index=False))

    print(f"\nSaved: {OUT_CSV}")


if __name__ == "__main__":
    main()
