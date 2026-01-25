# Execution under Adverse Selection

Research-grade (but lightweight) project studying execution cost under adverse selection.
The project implements a simplified market simulator with toxicity and adverse selection,
and evaluates execution strategies using implementation shortfall and tail risk metrics.

The repository includes:
- A baseline market simulator with adverse selection and toxicity regimes
- Execution policies (AlwaysMarket, ToxicityAware)
- Regime and misspecification grid experiments
- Automated reports and figures (mean IS, p90 IS, dominance analysis)

Key finding: in this market model, a simple AlwaysMarket strategy consistently outperforms
a toxicity-aware execution policy, highlighting the limits of short-term toxicity signals
when they do not strongly predict future adverse selection.

All results are fully reproducible via a single command (`scripts/run_all.py`).
