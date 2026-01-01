from __future__ import annotations

from dataclasses import dataclass

from execlab.sim.market import MarketParams, SimpleMarket
from execlab.types import Fill, Side, OrderType
from execlab.metrics.execution import implementation_shortfall, fill_price_stats


@dataclass
class ExecParams:
    horizon: int = 200
    parent_qty: float = 1.0
    side: Side = "buy"
    order_type: OrderType = "market"
    limit_offset: float = 0.0


def simulate_execution(ep: ExecParams, seed: int = 0) -> dict:
    """
    Minimal execution simulator for MVP.

    The goal is not realism but clarity:
    - compare market vs limit execution
    - measure implementation shortfall
    - expose adverse selection effects later
    """
    mkt = SimpleMarket(MarketParams(), seed=seed)

    # Arrival benchmark
    s0 = mkt.step()
    arrival_mid = s0.mid

    fills: list[Fill] = []
    remaining = ep.parent_qty

    for _ in range(ep.horizon):
        s = mkt.step()

        if remaining <= 0:
            break

        if ep.order_type == "market":
            px = s.ask if ep.side == "buy" else s.bid
            fills.append(
                Fill(
                    t=s.t,
                    side=ep.side,
                    qty=remaining,
                    price=px,
                    order_type="market",
                )
            )
            remaining = 0.0

        else:  # limit order
            if ep.side == "buy":
                limit_px = s.bid - ep.limit_offset
                if s.mid <= limit_px:
                    fills.append(
                        Fill(
                            t=s.t,
                            side=ep.side,
                            qty=remaining,
                            price=limit_px,
                            order_type="limit",
                        )
                    )
                    remaining = 0.0
            else:
                limit_px = s.ask + ep.limit_offset
                if s.mid >= limit_px:
                    fills.append(
                        Fill(
                            t=s.t,
                            side=ep.side,
                            qty=remaining,
                            price=limit_px,
                            order_type="limit",
                        )
                    )
                    remaining = 0.0

    is_cost = implementation_shortfall(
        fills, arrival_mid=arrival_mid, side=ep.side
    )

    return {
        "arrival_mid": arrival_mid,
        "fills": fills,
        "remaining": remaining,
        "implementation_shortfall": is_cost,
        "price_stats": fill_price_stats(fills),
    }
