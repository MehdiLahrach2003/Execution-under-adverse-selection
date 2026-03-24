"""the simulated market - this is where we build the world in which strategies compete"""

from __future__ import annotations   #makes types more flexible and avoids errors
from dataclasses import dataclass
import numpy as np
from execlab.types import MarketState, Side

"""The model's idea is: after an aggressive trade, a price move in the wrong direction
for the trader may occur, with a probability that increases with toxicity"""

"""Now we create an object that holds all the important parameters/numbers of the market model
For example:
* initial price (prix initial)
* spread (différence entre le prix vendeur (ask) et le prix acheteur (bid))
* volatility (volatilié)
* toxicity dynamics (dynamique de toxicité)
* adverse selection strength (force de l’adverse selection)
So `MarketParams` is kind of the simulator's dashboard"""

@dataclass
class MarketParams:
    
    
    """
    Simple discrete-time market simulator (MVP -> v1).
    (on travaille avec un petit modèle de marché, simple, qui évolue par pas de temps)

    Mid evolution: (The mid is the middle price between bid and ask)
        mid_{t+1} = mid_t + drift_t + sigma * eps_t + AS_kick_t

    where AS_kick_t is an adverse selection shock triggered by the trader's
    own aggressive trades.
    
    (le prix central au temps suivant = prix actuel + plusieurs effets)

    Toxicity is a latent state in [0, 1]. Higher toxicity means stronger
    adverse selection against the trader right after trading.
    latent means here:
    not a directly observed price like bid or ask, but a hidden market state
    """
    
    
    mid0: float = 100.0  #initial price 
    spread: float = 0.02  #bid/ask spread
    sigma: float = 0.05  #random noise in price evolution

    # Latent toxicity process
    
    """The tox_persist parameter describes how persistent toxicity is over time. 
    A high value means that if the market is toxic at a given point, it tends to remain so at the next time step. 
    This allows modeling the existence of market regimes, either calm or more dangerous."""
    tox_persist: float = 0.95
    
    """The tox_vol parameter represents the random variability of toxicity. 
    Even if toxicity is persistent, it is not completely fixed; it can fluctuate stochastically."""
    tox_vol: float = 0.05
    
    """The tox_drift_scale parameter links the toxicity level to a price drift component. 
    It should not be interpreted as a direct indication of the direction of the move, 
    but rather as a way to make the intensity of certain price moves depend on the market's toxicity state."""
    tox_drift_scale: float = 0.03

    # Adverse selection (endogenous) parameters
    as_prob_base: float = 0.05        # baseline probability of AS kick after a trade
    as_prob_slope: float = 0.40       # how much toxicity increases probability
    as_kick_scale: float = 0.08       # magnitude scale of the adverse move
    




"""While MarketParams only contained the settings, SimpleMarket uses these settings to:

- initialize the market,
- record the trader's trades,
- evolve toxicity,
- evolve the price,
- produce a MarketState"""



"""Alors que MarketParams contenait seulement les réglages, SimpleMarket utilise ces réglages pour :

- initialiser le marché,
- enregistrer les trades du trader,
- faire évoluer la toxicité,
- faire évoluer le prix,
- produire un MarketState"""



class SimpleMarket:   #Simple pour modèle "simplifié"
    """
    Minimal market model for execution experiments.

    Key feature (v1):
    - Adverse selection is endogenous: it depends on the trader's aggressive trades.
    """

    # The constructor (function that runs when a SimpleMarket object is created)
    # Le constructeur (fonction qui s'exécute quand on crée un objet SimpleMarket)

    def __init__(self, params: MarketParams, seed: int = 0) -> None:
        self.p = params  #here, we store the parameters in the object
        self.rng = np.random.default_rng(seed)   #here, we create a random number generator
        self.reset()   #when we create a market, we immediately set it to its initial state

    "The `reset()` method resets the market to its initial state."

    def reset(self) -> None:
        self.t = 0  #here, we initialize discrete time to 0.
        self.mid = float(self.p.mid0)  #we initialize the central market price.
        self.tox = 0.5  #we initialize toxicity to 0.5 (0.5 correspond à un "niveau neutre")
        self._recent_aggr = 0.0   #this variable stores a memory of the trader's recent aggressiveness.

        # Stores a one-step shock to apply at next step (signed, i.e. it can be positive or negative)
        self._pending_as_kick = 0.0  #here, we initialize this future shock to zero.
        
    """The following method is fundamental: it does not advance time yet. 
    It does not directly update the price. It is used to tell the market:
    "the trader just made a trade on this side, with this level of aggressiveness"
    And the market will potentially prepare a future punishment."""
    
    
    """Cette méthode prépare une éventuelle réaction du marché juste après le trade"""

    def apply_trade(self, side: Side, aggressiveness: float = 1.0) -> None:
        """
        Register an aggressive trade by the trader.

        aggressiveness in [0, 1] controls how "market-like" the action is.
        For MVP:
        - market order => aggressiveness = 1
        - passive/limit => aggressiveness ~ 0
        """
        a = float(np.clip(aggressiveness, 0.0, 1.0))  #this line forces aggressiveness to stay within the [0, 1] interval

        # Probability of adverse selection increases with toxicity and aggressiveness.
        p_as = self.p.as_prob_base + self.p.as_prob_slope * self.tox  #this line computes a "raw" adverse selection probability.
        p_as = float(np.clip(p_as, 0.0, 1.0)) * a

        if self.rng.random() < p_as:
            # Adverse move direction is against the trader:
            # - after a buy, price tends to go up (you bought before an up-move)
            # - after a sell, price tends to go down (you sold before a down-move)
            # +1 = le choc sera orienté vers le haut
            # -1 = le choc sera orienté vers le bas
            direction = +1.0 if side == "buy" else -1.0
            # Magnitude scales with toxicity and aggressiveness (taille du choc)
            mag = self.p.as_kick_scale * (0.5 + self.tox) * (0.5 + 0.5 * a)

            self._pending_as_kick += direction * mag
            
        # Track recent aggressiveness (decays over time in step()).
        self._recent_aggr = float(np.clip(self._recent_aggr + a, 0.0, 5.0))


    """If `apply_trade()` was used to prepare a potential market reaction, 
    then `step()` is used to : advance the market by one time step
    This is where:
    * the aggressiveness memory decays,
    * toxicity is updated,
    * a drift is generated,
    * the pending adverse selection shock is applied,
    * the mid-price evolves,
    * then `bid`, `ask` and a new `MarketState` object are reconstructed"""

    def step(self) -> MarketState:
        
        """The return type `-> MarketState` is important: 
        it indicates that this method will return:
        * the time,
        * the mid,
        * the bid,
        * the ask,
        * the toxicity."""
        
        
        # Exponential decay of recent aggressiveness
        self._recent_aggr *= 0.85  #plus le temps passe sans nouveau trade agressif, plus la mémoire s’efface
        
        
        """ Maintenant on va calculer la nouvelle toxicité """
        
        
        # Update latent toxicity (bounded AR(1)-like ; Le mot AR(1)-like signifie: la nouvelle valeur dépend beaucoup de l’ancienne)
        # Mean reversion stronger when no recent aggressive trade
        # Stronger cooling when we have not traded aggressively recently
        
        
        # this is a variable that measures: how fast toxicity reverts to a neutral state
        cooling = 0.20 * float(np.exp(-self._recent_aggr))   
        
        
        # Cette ligne commence à construire la nouvelle toxicité, dans une variable temporaire z
        # Cette ligne dit : la toxicité future ressemble beaucoup à la toxicité passée, mais avec une légère attraction vers le niveau neutre
        z = self.p.tox_persist * self.tox + (1.0 - self.p.tox_persist) * 0.5
        
        
        # Cette ligne ajoute explicitement un effet de retour vers le niveau neutre (i.e. on rapproche la toxicité de 0.5)
        z -= cooling * (self.tox - 0.5)
        
        
        # Ici, on ajoute un bruit aléatoire puis on borne la toxicité 
        z += self.p.tox_vol * self.rng.normal()
        self.tox = float(np.clip(z, 0.0, 1.0))


        """ On vient de détérminer la nouvelle toxicité """
        
        
        
        """ Maintenant, on calcule un petit drift du prix """
        
        # Exogenous drift component (regime-like)
        sign = 1.0 if self.rng.random() < 0.5 else -1.0  #Donne la direction (hausse ou baisse)
        drift = sign * self.p.tox_drift_scale * (self.tox - 0.5)


        """ Maintenant on applique le choc d’adverse selection en attente"""
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
