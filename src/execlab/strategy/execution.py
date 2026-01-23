from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Optional

from execlab.types import Side, OrderType, MarketState


@dataclass(frozen=True)
class Action:
    """
    One-step decision for the execution engine.

    qty: quantity to attempt this step (engine will cap by remaining)
    order_type: 'market' or 'limit'
    limit_offset: for limit orders only (distance from best bid/ask)
    aggressiveness: in [0, 1], used by the market model to trigger adverse selection
    """
    qty: float
    order_type: OrderType
    limit_offset: float = 0.0
    aggressiveness: float = 1.0


class ExecutionPolicy(Protocol):
    """Policy interface: decides what to do given current market state and remaining qty."""
    def decide(self, state: MarketState, remaining: float, side: Side) -> Optional[Action]:
        ...


@dataclass
class AlwaysMarket:
    """Execute everything immediately with a market order."""
    def decide(self, state: MarketState, remaining: float, side: Side) -> Optional[Action]:
        if remaining <= 0:
            return None
        return Action(qty=remaining, order_type="market", aggressiveness=1.0)


@dataclass
class AlwaysLimit:
    """
    Always place a passive limit order.

    limit_offset=0 means join best bid/ask (very passive in our simplified model).
    Increasing limit_offset makes it even less likely to fill.
    """
    limit_offset: float = 0.0
    aggressiveness: float = 0.2  # less aggressive than market

    def decide(self, state: MarketState, remaining: float, side: Side) -> Optional[Action]:
        if remaining <= 0:
            return None
        return Action(
            qty=remaining,
            order_type="limit",
            limit_offset=self.limit_offset,
            aggressiveness=self.aggressiveness,
        )

@dataclass
class ToxicityAwareExecution:
    tox_trigger: float = 0.60   # if above, we prefer to wait
    max_wait: int = 30          # max steps to wait
    slice_qty: float = 0.25     # execute in slices (fraction of remaining)

    def __post_init__(self):
        self._wait_counter = 0

    def decide(self, state: MarketState, remaining: float, side: Side) -> Optional[Action]:
        if remaining <= 0:
            return None

        tox = state.toxicity

        # If toxicity is high, wait (unless deadline)
        if tox >= self.tox_trigger and self._wait_counter < self.max_wait:
            self._wait_counter += 1
            return None

        # Either toxicity is acceptable, or we reached deadline: execute a slice
        self._wait_counter = 0
        qty = min(remaining, remaining * self.slice_qty)

        return Action(
            qty=qty,
            order_type="market",
            aggressiveness=1.0,
        )