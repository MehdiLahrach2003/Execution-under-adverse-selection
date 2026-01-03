# scripts/run_misspec_grid.py
"""
Misspecification grid experiment.

Idea:
- The "true" market has some (as_kick_scale_true, tox_persist_true).
- The policy is calibrated / tuned using (as_kick_scale_belief, tox_persist_belief).
- We evaluate AlwaysMarket vs ToxicityAware across a grid of mis-specified beliefs.

Outputs:
- reports/misspec_grid.csv
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from execlab.backtest.mvp import BacktestParams, run_backtest
from execlab.sim.market import MarketParams
from execlab.strategy.execution import AlwaysMarket, ToxicityAwareExecution


# -----------------------------
# Utilities
# -----------------------------
def _get_value(obj: Any, keys: Sequence[str]) -> Optional[float]:
    """
    Robustly extract a numeric field from either:
      - a dict (preferred in our current version)
      - an object with attributes (fallback)

    Returns None if not found or not convertible.
    """
    if obj is None:
        return None

    # dict-like
    if isinstance(obj, dict):
        for k in keys:
            if k in obj:
                try:
                    v = obj[k]
                    if v is None:
                        return None
                    return float(v)
                except Exception:
                    return None
        return None

    # attribute-like
    for k in keys:
        if hasattr(obj, k):
            try:
                v = getattr(obj, k)
                if v is None:
                    return None
                return float(v)
            except Exception:
                return None

    return None


def summarize(results: List[Any]) -> Dict[str, float]:
    """
    Summary stats for IS/cost and fill rate.

    Handles multiple naming conventions across versions.
    """
    # IS / cost candidates
    is_keys = [
        "is_cost",
        "implementation_shortfall",
        "implementation_shortfall_cost",
        "impl_shortfall",
        "is",
        "cost",
        "mean_is",  # just in case
    ]

    # fill rate candidates
    fill_keys = [
        "fill_rate",
        "fillrate",
        "fill",
        "fill_ratio",
        "fill_fraction",
        "mean_fill_rate",  # just in case
    ]

    is_list: List[float] = []
    fill_list: List[float] = []

    for r in results:
        is_v = _get_value(r, is_keys)
        fill_v = _get_value(r, fill_keys)

        is_list.append(np.nan if is_v is None else is_v)
        fill_list.append(np.nan if fill_v is None else fill_v)

    is_arr = np.array(is_list, dtype=float)
    fill_arr = np.array(fill_list, dtype=float)

    # nan-safe aggregations (no warnings)
    mean = float(np.nanmean(is_arr)) if np.isfinite(is_arr).any() else float("nan")
    p90 = float(np.nanpercentile(is_arr, 90)) if np.isfinite(is_arr).any() else float("nan")

    fill_rate = float(np.nanmean(fill_arr)) if np.isfinite(fill_arr).any() else float("nan")

    return {"mean": mean, "p90": p90, "fill_rate": fill_rate}


def run_policy(
    policy: Any,
    market_params: MarketParams,
    backtest: BacktestParams,
    seeds: Sequence[int],
) -> List[Dict[str, float]]:
    """
    Run many seeds and return raw result dicts.
    """
    outs: List[Dict[str, float]] = []
    for s in seeds:
        out = run_backtest(
            policy=policy,
            market_params=market_params,
            backtest=backtest,
            seed=int(s),
        )

        # Ensure dict output (we standardize)
        if isinstance(out, dict):
            outs.append(out)
        else:
            # If some version returns an object, convert best-effort
            outs.append(asdict(out) if hasattr(out, "__dict__") else {"_raw": float("nan")})

    return outs


# -----------------------------
# Experiment
# -----------------------------
def main() -> None:
    # ---- fixed backtest params ----
    backtest = BacktestParams(
        horizon=80,
        parent_qty=1.0,
        side="buy",
    )

    # ---- "true market" regime ----
    # We keep these fixed, and we misspecify beliefs.
    as_kick_scale_true = 0.03
    tox_persist_true = 0.85

    # You can keep other market params at defaults (depends on MarketParams dataclass),
    # but here we explicitly set the two we care about + keep the rest default.
    market_true = MarketParams(
        as_kick_scale=float(as_kick_scale_true),
        tox_persist=float(tox_persist_true),
    )

    # ---- misspec grid (beliefs) ----
    as_kick_scale_grid = [0.00, 0.01, 0.02, 0.03, 0.05]
    tox_persist_grid = [0.60, 0.75, 0.85, 0.95]

    # seeds
    seeds = list(range(50))  # increase later (e.g., 200) when you want smoother surfaces

    rows: List[Dict[str, float]] = []

    for as_belief in as_kick_scale_grid:
        for tox_belief in tox_persist_grid:
            # --- policies: belief enters HERE for ToxicityAwareExecution ---
            always = AlwaysMarket()

            tox_policy = ToxicityAwareExecution(
                # belief about adverse selection "kick intensity"
                as_kick_scale=float(as_belief),
                # belief about toxicity persistence
                tox_persist=float(tox_belief),
            )

            # Run both policies on the SAME true market params
            am_results = run_policy(always, market_true, backtest, seeds)
            tx_results = run_policy(tox_policy, market_true, backtest, seeds)

            am_stats = summarize(am_results)
            tx_stats = summarize(tx_results)

            # deltas = ToxicityAware - AlwaysMarket
            delta_mean = tx_stats["mean"] - am_stats["mean"]
            delta_p90 = tx_stats["p90"] - am_stats["p90"]
            delta_fill = tx_stats["fill_rate"] - am_stats["fill_rate"]

            rows.append(
                {
                    "as_kick_scale_true": float(as_kick_scale_true),
                    "tox_persist_true": float(tox_persist_true),
                    "as_kick_scale_belief": float(as_belief),
                    "tox_persist_belief": float(tox_belief),
                    "AlwaysMarket_mean": am_stats["mean"],
                    "AlwaysMarket_p90": am_stats["p90"],
                    "AlwaysMarket_fill_rate": am_stats["fill_rate"],
                    "ToxicityAware_mean": tx_stats["mean"],
                    "ToxicityAware_p90": tx_stats["p90"],
                    "ToxicityAware_fill_rate": tx_stats["fill_rate"],
                    "delta_mean": float(delta_mean),
                    "delta_p90": float(delta_p90),
                    "delta_fill_rate": float(delta_fill),
                    "n_seeds": float(len(seeds)),
                }
            )

    # Save CSV
    import os
    import csv

    os.makedirs("reports", exist_ok=True)
    out_path = os.path.join("reports", "misspec_grid.csv")

    # write
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
