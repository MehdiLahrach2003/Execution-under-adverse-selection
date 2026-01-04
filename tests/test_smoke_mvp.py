from execlab.backtest.mvp import BacktestParams, run_backtest
from execlab.strategy.execution import AlwaysMarket


def test_backtest_runs() -> None:
    bp = BacktestParams(horizon=50, parent_qty=1.0, side="buy")
    out = run_backtest(policy=AlwaysMarket(), bp=bp, seed=0)

    assert "implementation_shortfall" in out
    assert isinstance(out["fills"], list)
