# scripts/run_regime_grid.py
"""
Regime analysis for execution under adverse selection.

We sweep:
- as_kick: strength of trade-triggered adverse selection kick
- tox_persist: persistence of latent toxicity

We compare:
- AlwaysMarket (baseline, immediate)
- ToxicityAware (adaptive, waits/filters on toxicity)

Output:
- reports/regime_grid.csv
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Type

import numpy as np
import pandas as pd

from execlab.sim.market import MarketParams


# -----------------------------
# Experiment configuration
# -----------------------------
SEEDS = list(range(50))

AS_KICKS = [0.00, 0.01, 0.02, 0.03, 0.05]
TOX_PERSIST = [0.60, 0.75, 0.85, 0.95]

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
    """
    Support both dict outputs and dataclass-like outputs (attributes).
    """
    if isinstance(out, dict):
        return out.get(name, default)
    return getattr(out, name, default)


# -----------------------------
# Dynamic resolution (robust)
# -----------------------------
def resolve_mvp_runner() -> Callable[..., Any]:
    """
    Find an episode runner in execlab.backtest.mvp without assuming the exact name.
    """
    import execlab.backtest.mvp as mvp

    candidates = [
        "run_single_episode",
        "run_episode",
        "run_one_episode",
        "simulate_episode",
        "run_mvp_episode",
        "run",
        "simulate",
    ]

    for name in candidates:
        fn = getattr(mvp, name, None)
        if callable(fn):
            return fn

    public = [n for n in dir(mvp) if not n.startswith("_")]
    raise ImportError(
        "Could not find an episode runner in execlab.backtest.mvp. "
        f"Tried: {candidates}. Available: {public}"
    )


def call_runner(runner: Callable[..., Any], policy: Any, market_params: MarketParams, seed: int) -> Any:
    """
    Call the runner with common signatures.
    """
    for kwargs in (
        {"policy": policy, "market_params": market_params, "seed": seed},
        {"policy": policy, "bp": market_params, "seed": seed},
        {"policy": policy, "params": market_params, "seed": seed},
    ):
        try:
            return runner(**kwargs)
        except TypeError:
            pass

    # positional fallback
    return runner(policy, market_params, seed)


def resolve_policy_instance(kind: str) -> Any:
    """
    Resolve and instantiate a policy class from execlab.strategy.execution.

    kind in {"always_market", "toxicity_aware"}
    """
    import execlab.strategy.execution as ex

    # Heuristic candidate names (covers most naming conventions)
    if kind == "always_market":
        candidates = [
            "AlwaysMarketExecution",
            "AlwaysMarket",
            "AlwaysMarketExec",
            "AlwaysMarketPolicy",
            "MarketExecution",
            "ImmediateExecution",
        ]
    elif kind == "toxicity_aware":
        candidates = [
            "ToxicityAwareExecution",
            "ToxicityAware",
            "ToxicityAwareExec",
            "ToxicityAwarePolicy",
            "ToxicityFilterExecution",
            "ToxicityBasedExecution",
        ]
    else:
        raise ValueError(f"Unknown policy kind: {kind}")

    # 1) direct name match
    for name in candidates:
        cls = getattr(ex, name, None)
        if isinstance(cls, type):
            return cls()

    # 2) fallback: search by substring in available classes
    # (useful if you named it e.g. "ExecutionAlwaysMarket" or "ExecutionToxicityAware")
    wanted_tokens = ["market"] if kind == "always_market" else ["tox", "toxic"]
    available_classes = [
        (n, getattr(ex, n)) for n in dir(ex) if not n.startswith("_") and isinstance(getattr(ex, n), type)
    ]

    for n, cls in available_classes:
        low = n.lower()
        if kind == "always_market":
            if "market" in low and ("always" in low or "immediate" in low):
                return cls()
        else:
            if any(tok in low for tok in wanted_tokens):
                return cls()

    # 3) give a great error message with what exists
    names = [n for n, _ in available_classes]
    raise ImportError(
        f"Could not resolve policy '{kind}' in execlab.strategy.execution. "
        f"Looked for candidates={candidates}. Available classes={names}"
    )


# -----------------------------
# Core evaluation
# -----------------------------
def run_policy_on_regime(
    runner: Callable[..., Any],
    policy_name: str,
    policy: Any,
    bp: MarketParams,
    seeds: List[int],
) -> Dict[str, float]:
    costs: List[float] = []
    fill_times: List[float] = []
    fill_rates: List[float] = []

    for seed in seeds:
        out = call_runner(runner=runner, policy=policy, market_params=bp, seed=seed)

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

    runner = resolve_mvp_runner()
    print(f"[regime_grid] Using runner: execlab.backtest.mvp.{runner.__name__}")

    always_market = resolve_policy_instance("always_market")
    toxicity_aware = resolve_policy_instance("toxicity_aware")

    print(f"[regime_grid] Using policy AlwaysMarket: {always_market.__class__.__name__}")
    print(f"[regime_grid] Using policy ToxicityAware: {toxicity_aware.__class__.__name__}")

    policies = {
        "AlwaysMarket": always_market,
        "ToxicityAware": toxicity_aware,
    }

    rows: List[Dict[str, float]] = []

    for as_kick in AS_KICKS:
        for tox_persist in TOX_PERSIST:
            bp = MarketParams(as_kick=float(as_kick), tox_persist=float(tox_persist))

            row: Dict[str, float] = {
                "as_kick": float(as_kick),
                "tox_persist": float(tox_persist),
            }

            for name, policy in policies.items():
                row.update(run_policy_on_regime(runner, name, policy, bp, SEEDS))

            row["delta_mean"] = row["ToxicityAware_mean"] - row["AlwaysMarket_mean"]
            row["delta_p90"] = row["ToxicityAware_p90"] - row["AlwaysMarket_p90"]
            row["delta_avg_first_fill_t"] = (
                row["ToxicityAware_avg_first_fill_t"] - row["AlwaysMarket_avg_first_fill_t"]
            )

            rows.append(row)

    df = pd.DataFrame(rows).sort_values(["as_kick", "tox_persist"]).reset_index(drop=True)
    df.to_csv(OUT_CSV, index=False)

    key_cols = [
        "as_kick",
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
