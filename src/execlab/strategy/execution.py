# Ce fichier répond à la question : Que décide de faire la stratégie à chaque instant ?
# ce fichier ne simule pas le marché, il ne calcule pas encore les métriques : il définit la décision du trader.


from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Optional

from execlab.types import Side, OrderType, MarketState



"""La classe suivante représente ce que la stratégie veut faire à un pas de temps donné"""
@dataclass(frozen=True)
class Action:
    """
    One-step decision for the execution engine.

    qty: quantity to attempt this step (engine will cap by remaining)
    order_type: 'market' or 'limit'
    limit_offset: for limit orders only (distance from best bid/ask)
    aggressiveness: in [0, 1], used by the market model to trigger adverse selection
    """
    qty: float  # C'est la quantité que la stratégie aimerait exécuter à cet instant 
    order_type: OrderType  # limit ou marché 
    
    """Ce champ est utile seulement pour les ordres limites
    Il représente une distance par rapport au meilleur bid/ask."""
    limit_offset: float = 0.0
    
    aggressiveness: float = 1.0  # Ce champ mesure à quel point l’action est agressive
    
    

"""Ce bloc définit une interface logique pour les stratégies.
Il dit : toute stratégie d’exécution doit savoir répondre à la question :
“que fais-tu maintenant, étant donné l’état du marché et la quantité restante ?”"""
class ExecutionPolicy(Protocol):
    """Policy interface: decides what to do given current market state and remaining qty."""
    def decide(self, state: MarketState, remaining: float, side: Side) -> Optional[Action]:
        ...


"""Cette classe définit la stratégie la plus naïve : s'il reste quelque chose à exécuter, alors
j'exécute tout immédiatement au marché"""

@dataclass
class AlwaysMarket:
    """Execute everything immediately with a market order."""
    def decide(self, state: MarketState, remaining: float, side: Side) -> Optional[Action]:
        if remaining <= 0:
            return None
        return Action(qty=remaining, order_type="market", aggressiveness=1.0)



"""Cette classe définit une stratégie qui fait l'inverse : je place toujours un ordre limite passif"""
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
        
   
        
"""Cette classe définit la stratégie la plus importante du projet. 
C'est une stratégie qui dit : si le marché paraît toxique, j'attends, 
sinon, j'exécute progressivement"""
@dataclass
class ToxicityAwareExecution:
    tox_trigger: float = 0.60   # if above, we prefer to wait (seuil de toxicité)
    max_wait: int = 30          # max steps to wait (la stratégie n'attend pas éternellement)
    slice_qty: float = 0.25     # execute in slices (fraction of remaining) - (Quand elle décide d’agir, elle n’exécute pas forcément tout)

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