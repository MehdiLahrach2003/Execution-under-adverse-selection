# scripts/run_misspec_grid.py
"""
Misspecification grid experiment (aligned with current ToxicityAwareExecution).

Key idea (this codebase):
- ToxicityAwareExecution is NOT parameterized by (as_kick_scale, tox_persist).
- It reacts to observed state.toxicity via a threshold tox_trigger.

So we define misspecification as:
- The true market regime is (as_kick_scale_true, tox_persist_true).
- The policy is calibrated with a belief/choice of tox_trigger (threshold).
- We evaluate AlwaysMarket vs ToxicityAwareExecution across:
    true regimes x tox_trigger_belief grid.

Outputs:
- reports/misspec_grid.csv
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from execlab.backtest.mvp import BacktestParams, run_backtest
from execlab.sim.market import MarketParams
from execlab.strategy.execution import AlwaysMarket, ToxicityAwareExecution


# -----------------------------
# Utilities
# -----------------------------
def _get_value(obj: Any, keys: Sequence[str]) -> Optional[float]:
    """Robustly extract a numeric field from dict or attribute-like object."""
    if obj is None:
        return None

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
    """Summary stats for IS/cost and fill rate (handles multiple naming conventions)."""
    is_keys = [
        "is_cost",
        "implementation_shortfall",
        "implementation_shortfall_cost",
        "impl_shortfall",
        "is",
        "cost",
        "mean_is",
    ]
    fill_keys = [
        "fill_rate",
        "fillrate",
        "fill",
        "fill_ratio",
        "fill_fraction",
        "mean_fill_rate",
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

    mean = float(np.nanmean(is_arr)) if np.isfinite(is_arr).any() else float("nan")
    p90 = float(np.nanpercentile(is_arr, 90)) if np.isfinite(is_arr).any() else float("nan")
    fill_rate = float(np.nanmean(fill_arr)) if np.isfinite(fill_arr).any() else float("nan")

    return {"mean": mean, "p90": p90, "fill_rate": fill_rate}


def _coerce_to_dict(out: Any) -> Dict[str, Any]:
    """Convert output to a dict robustly."""
    if isinstance(out, dict):
        return out
    try:
        return dict(out)
    except Exception:
        pass
    try:
        return vars(out)
    except Exception:
        return {"_raw": float("nan")}


def run_policy(
    policy: Any,
    market_params: MarketParams,
    backtest: BacktestParams,
    seeds: Sequence[int],
) -> List[Dict[str, Any]]:
    """
    Run many seeds and return raw result dicts.

    In your codebase version, run_backtest signature is:
      run_backtest(policy, bp, seed=0, market_params=None) -> dict
    """
    outs: List[Dict[str, Any]] = []
    for s in seeds:
        out = run_backtest(
            policy=policy,
            bp=backtest,
            seed=int(s),
            market_params=market_params,
        )
        outs.append(_coerce_to_dict(out))
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

    # ---- true regime grid ----
    # We vary the true market regime (these affect the dynamics that generate state.toxicity).
    as_kick_scale_true_grid = [0.00, 0.01, 0.02, 0.03, 0.05]
    tox_persist_true_grid = [0.60, 0.75, 0.85, 0.95]

    # ---- "belief" grid = policy threshold tox_trigger ----
    # This is the only meaningful policy calibration knob in the current class.
    tox_trigger_grid = [0.50, 0.60, 0.70, 0.80]

    # Optional: also vary max_wait and slice_qty later (D.4 if you want)
    max_wait = 30
    slice_qty = 0.25

    seeds = list(range(50))

    rows: List[Dict[str, float]] = []

    for as_true in as_kick_scale_true_grid:
        for tox_true in tox_persist_true_grid:
            market_true = MarketParams(
                as_kick_scale=float(as_true),
                tox_persist=float(tox_true),
            )

            # Baseline run once per true regime
            always = AlwaysMarket()
            am_results = run_policy(always, market_true, backtest, seeds)
            am_stats = summarize(am_results)

            for tox_trigger_belief in tox_trigger_grid:
                tox_policy = ToxicityAwareExecution(
                    tox_trigger=float(tox_trigger_belief),
                    max_wait=int(max_wait),
                    slice_qty=float(slice_qty),
                )

                tx_results = run_policy(tox_policy, market_true, backtest, seeds)
                tx_stats = summarize(tx_results)

                delta_mean = tx_stats["mean"] - am_stats["mean"]
                delta_p90 = tx_stats["p90"] - am_stats["p90"]
                delta_fill = tx_stats["fill_rate"] - am_stats["fill_rate"]

                rows.append(
                    {
                        # Truth (market regime)
                        "as_kick_scale_true": float(as_true),
                        "tox_persist_true": float(tox_true),

                        # Belief (policy calibration)
                        "tox_trigger_belief": float(tox_trigger_belief),

                        # Aliases for reports if needed
                        "as_kick_scale": float(as_true),
                        "tox_persist_model": float(tox_trigger_belief),  # used as a conditioning key in some reports

                        # Stats
                        "AlwaysMarket_mean": float(am_stats["mean"]),
                        "AlwaysMarket_p90": float(am_stats["p90"]),
                        "AlwaysMarket_fill_rate": float(am_stats["fill_rate"]),
                        "ToxicityAware_mean": float(tx_stats["mean"]),
                        "ToxicityAware_p90": float(tx_stats["p90"]),
                        "ToxicityAware_fill_rate": float(tx_stats["fill_rate"]),

                        # Deltas
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

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
