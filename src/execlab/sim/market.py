from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from execlab.types import MarketState


@dataclass
class MarketParams:
    """
    Simple discrete-time market simulator.

    - mid follows: mid_{t+1} = mid_t + drift_t + sigma * eps_t
    - drift_t depends on latent 'toxicity' (adverse selection proxy)
    - spread is fixed for MVP
    """
    mid0: float = 100.0
    spread: float = 0.02
    sigma: float = 0.05
    tox_persist: float = 0.95  # persistence of toxicity (0..1)
    tox_vol: float = 0.05      # innovation scale for toxicity
    tox_drift_scale: float = 0.03  # how much toxicity moves drift


class SimpleMarket:
    """
    A minimal market model designed for execution experiments.

    Toxicity is a latent state in [0, 1]. Higher toxicity means higher probability
    that short-term price moves against the trader right after they trade.
    """

    def __init__(self, params: MarketParams, seed: int = 0) -> None:
        self.p = params
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self) -> None:
        self.t = 0
        self.mid = float(self.p.mid0)
        self.tox = 0.5  # start neutral

    def step(self) -> MarketState:
        # Update latent toxicity (bounded AR(1)-like)
        z = self.p.tox_persist * self.tox + (1.0 - self.p.tox_persist) * 0.5
        z += self.p.tox_vol * self.rng.normal()
        self.tox = float(np.clip(z, 0.0, 1.0))

        # Drift: toxicity pushes drift in a random direction (informative flow)
        # This keeps it simple but creates "adverse selection regimes".
        sign = 1.0 if self.rng.random() < 0.5 else -1.0
        drift = sign * self.p.tox_drift_scale * (self.tox - 0.5)

        # Mid-price evolution
        self.mid = float(self.mid + drift + self.p.sigma * self.rng.normal())

        bid = self.mid - 0.5 * self.p.spread
        ask = self.mid + 0.5 * self.p.spread

        state = MarketState(t=self.t, mid=self.mid, bid=bid, ask=ask, toxicity=self.tox)
        self.t += 1
        return state
