from __future__ import annotations

import numpy as np
from execlab.types import Fill, Side


def implementation_shortfall(
    fills: list[Fill],
    arrival_mid: float,
    side: Side,
) -> float:
    """
    Implementation shortfall (IS), positive = worse for trader.

    Buy:  IS = (avg_fill - arrival_mid)
    Sell: IS = (arrival_mid - avg_fill)
    """
    if len(fills) == 0:
        return float("nan")

    qty = np.array([f.qty for f in fills], dtype=float)
    px = np.array([f.price for f in fills], dtype=float)
    avg_fill = float(np.sum(qty * px) / np.sum(qty))

    if side == "buy":
        return avg_fill - arrival_mid
    return arrival_mid - avg_fill


def fill_price_stats(fills: list[Fill]) -> dict[str, float]:
    """Basic summary stats for executed prices."""
    if len(fills) == 0:
        return {"avg": float("nan"), "min": float("nan"), "max": float("nan")}

    px = np.array([f.price for f in fills], dtype=float)
    return {"avg": float(px.mean()), "min": float(px.min()), "max": float(px.max())}
