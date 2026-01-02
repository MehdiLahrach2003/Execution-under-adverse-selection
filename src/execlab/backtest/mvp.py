from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from execlab.metrics.execution import fill_price_stats, implementation_shortfall
from execlab.sim.market import MarketParams, SimpleMarket
from execlab.strategy.execution import ExecutionPolicy, Action, AlwaysMarket
from execlab.types import Fill, Side


@dataclass
class BacktestParams:
    horizon: int = 200
    parent_qty: float = 1.0
    side: Side = "buy"


def _try_execute_action(mkt: SimpleMarket, action: Action, state, side: Side, remaining: float) -> tuple[list[Fill], float]:
    """
    Apply one Action to the market state and return (fills, new_remaining).

    Note: this is a simplified execution model for clarity (MVP).
    """
    qty = min(action.qty, remaining)
    if qty <= 0:
        return [], remaining

    fills: list[Fill] = []

    if action.order_type == "market":
        px = state.ask if side == "buy" else state.bid
        mkt.apply_trade(side=side, aggressiveness=action.aggressiveness)
        fills.append(Fill(t=state.t, side=side, qty=qty, price=px, order_type="market"))
        return fills, remaining - qty

    # limit order
    if side == "buy":
        limit_px = state.bid - action.limit_offset
        if state.mid <= limit_px:
            mkt.apply_trade(side=side, aggressiveness=action.aggressiveness)
            fills.append(Fill(t=state.t, side=side, qty=qty, price=limit_px, order_type="limit"))
            return fills, remaining - qty
        return [], remaining
    else:
        limit_px = state.ask + action.limit_offset
        if state.mid >= limit_px:
            mkt.apply_trade(side=side, aggressiveness=action.aggressiveness)
            fills.append(Fill(t=state.t, side=side, qty=qty, price=limit_px, order_type="limit"))
            return fills, remaining - qty
        return [], remaining


def run_backtest(
    policy: ExecutionPolicy,
    bp: BacktestParams,
    seed: int = 0,
    market_params: Optional[MarketParams] = None,
) -> dict:
    """
    Run a single execution episode with a given policy.

    Returns metrics + fills.
    """
    mp = market_params or MarketParams()
    mkt = SimpleMarket(mp, seed=seed)

    # Arrival benchmark
    s0 = mkt.step()
    arrival_mid = s0.mid

    fills: list[Fill] = []
    remaining = bp.parent_qty

    for _ in range(bp.horizon):
        state = mkt.step()

        if remaining <= 0:
            break

        action = policy.decide(state=state, remaining=remaining, side=bp.side)
        if action is None:
            continue

        new_fills, remaining = _try_execute_action(
            mkt=mkt,
            action=action,
            state=state,
            side=bp.side,
            remaining=remaining,
        )
        fills.extend(new_fills)

    is_cost = implementation_shortfall(fills, arrival_mid=arrival_mid, side=bp.side)

    first_fill_t = fills[0].t if len(fills) > 0 else None

    return {
        "arrival_mid": arrival_mid,
        "fills": fills,
        "remaining": remaining,
        "implementation_shortfall": is_cost,
        "price_stats": fill_price_stats(fills),
        "first_fill_t": first_fill_t,
    }


def run_mvp_example(seed: int = 42) -> None:
    bp = BacktestParams()

    for policy in [AlwaysMarket(), ]:
        out = run_backtest(policy=policy, bp=bp, seed=seed)
        print("=" * 60)
        print(f"Policy: {policy.__class__.__name__}")
        print(f"Arrival mid: {out['arrival_mid']:.4f}")
        print(f"IS (cost):   {out['implementation_shortfall']:.6f}")
        print(f"Remaining:   {out['remaining']}")
