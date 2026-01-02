from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from execlab.types import MarketState, Side


@dataclass
class MarketParams:
    """
    Simple discrete-time market simulator (MVP -> v1).

    Mid evolution:
        mid_{t+1} = mid_t + drift_t + sigma * eps_t + AS_kick_t

    where AS_kick_t is an adverse selection shock triggered by the trader's
    own aggressive trades.

    Toxicity is a latent state in [0, 1]. Higher toxicity means stronger
    adverse selection against the trader right after trading.
    """
    mid0: float = 100.0
    spread: float = 0.02
    sigma: float = 0.05

    # Latent toxicity process
    tox_persist: float = 0.95
    tox_vol: float = 0.05
    tox_drift_scale: float = 0.03

    # Adverse selection (endogenous) parameters
    as_prob_base: float = 0.05        # baseline probability of AS kick after a trade
    as_prob_slope: float = 0.40       # how much toxicity increases probability
    as_kick_scale: float = 0.08       # magnitude scale of the adverse move


class SimpleMarket:
    """
    Minimal market model for execution experiments.

    Key feature (v1):
    - Adverse selection is *endogenous*: it depends on the trader's aggressive trades.
    """

    def __init__(self, params: MarketParams, seed: int = 0) -> None:
        self.p = params
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self) -> None:
        self.t = 0
        self.mid = float(self.p.mid0)
        self.tox = 0.5
        self._recent_aggr = 0.0

        # Stores a one-step shock to apply at next step (signed)
        self._pending_as_kick = 0.0

    def apply_trade(self, side: Side, aggressiveness: float = 1.0) -> None:
        """
        Register an aggressive trade by the trader.

        aggressiveness in [0, 1] controls how "market-like" the action is.
        For MVP:
        - market order => aggressiveness = 1
        - passive/limit => aggressiveness ~ 0
        """
        a = float(np.clip(aggressiveness, 0.0, 1.0))

        # Probability of adverse selection increases with toxicity and aggressiveness.
        p_as = self.p.as_prob_base + self.p.as_prob_slope * self.tox
        p_as = float(np.clip(p_as, 0.0, 1.0)) * a

        if self.rng.random() < p_as:
            # Adverse move direction is against the trader:
            # - after a buy, price tends to go up (you bought before an up-move)
            # - after a sell, price tends to go down (you sold before a down-move)
            direction = +1.0 if side == "buy" else -1.0

            # Magnitude scales with toxicity and aggressiveness
            mag = self.p.as_kick_scale * (0.5 + self.tox) * (0.5 + 0.5 * a)

            self._pending_as_kick += direction * mag
            
        # Track recent aggressiveness (decays over time in step()).
        self._recent_aggr = float(np.clip(self._recent_aggr + a, 0.0, 5.0))


    def step(self) -> MarketState:
        # Exponential decay of recent aggressiveness
        self._recent_aggr *= 0.85
        # Update latent toxicity (bounded AR(1)-like)
        # Mean reversion stronger when no recent aggressive trade
        # Stronger cooling when we have not traded aggressively recently
        cooling = 0.20 * float(np.exp(-self._recent_aggr))
        z = self.p.tox_persist * self.tox + (1.0 - self.p.tox_persist) * 0.5
        z -= cooling * (self.tox - 0.5)
        z += self.p.tox_vol * self.rng.normal()
        self.tox = float(np.clip(z, 0.0, 1.0))

        # Exogenous drift component (regime-like)
        sign = 1.0 if self.rng.random() < 0.5 else -1.0
        drift = sign * self.p.tox_drift_scale * (self.tox - 0.5)

        # Apply pending adverse selection shock (one-step)
        as_kick = float(self._pending_as_kick)
        self._pending_as_kick = 0.0

        # Mid-price evolution
        self.mid = float(self.mid + drift + as_kick + self.p.sigma * self.rng.normal())

        bid = self.mid - 0.5 * self.p.spread
        ask = self.mid + 0.5 * self.p.spread

        state = MarketState(t=self.t, mid=self.mid, bid=bid, ask=ask, toxicity=self.tox)
        self.t += 1
        return state
