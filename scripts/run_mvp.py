from __future__ import annotations

import numpy as np

from execlab.backtest.mvp import BacktestParams, run_backtest
from execlab.strategy.execution import AlwaysLimit, AlwaysMarket, ToxicityAwareExecution


def summarize(x: list[float]) -> dict[str, float]:
    a = np.array(x, dtype=float)
    a = a[~np.isnan(a)]
    if a.size == 0:
        return {"mean": float("nan"), "p10": float("nan"), "p50": float("nan"), "p90": float("nan")}
    return {
        "mean": float(a.mean()),
        "p10": float(np.quantile(a, 0.10)),
        "p50": float(np.quantile(a, 0.50)),
        "p90": float(np.quantile(a, 0.90)),
    }


def main() -> None:
    bp = BacktestParams(horizon=200, parent_qty=1.0, side="buy")

    policies = {
        "AlwaysMarket": AlwaysMarket(),
        "AlwaysLimit@0": AlwaysLimit(limit_offset=0.0),
        "ToxicityAware": ToxicityAwareExecution(),
    }

    seeds = list(range(50))
    results = {}

    for name, policy in policies.items():
        costs = []
        fill_rates = []
        fill_times = []

        for seed in seeds:
            out = run_backtest(policy=policy, bp=bp, seed=seed)
            costs.append(out["implementation_shortfall"])
            filled = 1.0 if out["remaining"] <= 1e-12 else 0.0
            fill_rates.append(filled)

            t = out["first_fill_t"]
            if t is not None:
                fill_times.append(float(t))


        results[name] = {
            "IS": summarize(costs),
            "fill_rate": float(np.mean(fill_rates)),
            "avg_first_fill_t": float(np.mean(fill_times)) if len(fill_times) > 0 else float("nan"),

        }

    print("\n=== Execution baseline comparison ===")
    for name, res in results.items():
        print("-" * 60)
        print(name)
        print(f"Fill rate: {res['fill_rate']:.2f}")
        print(f"IS stats : {res['IS']}")
        print(f"Avg first fill t: {res['avg_first_fill_t']:.2f}")



if __name__ == "__main__":
    main()
