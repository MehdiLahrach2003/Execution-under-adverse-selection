from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

Side = Literal["buy", "sell"]
OrderType = Literal["market", "limit"]


@dataclass(frozen=True)
class MarketState:
    """One time-step snapshot of the simulated market."""
    t: int
    mid: float
    bid: float
    ask: float
    toxicity: float  # latent adverse selection indicator in [0, 1]


@dataclass(frozen=True)
class Fill:
    """Execution fill event."""
    t: int
    side: Side
    qty: float
    price: float
    order_type: OrderType
    note: Optional[str] = None
