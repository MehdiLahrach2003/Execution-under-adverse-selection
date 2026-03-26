from __future__ import annotations 
from dataclasses import dataclass
from typing import Literal, Optional  #it can be a string… or nothing


# French version

"""Dans ce code, on définit le vocabulaire du projet en introduisant des types comme
Side et OrderType, qui restreignent les valeurs possibles (buy/sell, market/limit). 
Ensuite, on définit deux classes principales : 
MarketState, qui représente l’état du marché à un instant donné, 
et Fill, qui représente une exécution réelle d’un ordre."""

# English version 

"""In this code, we define the project's vocabulary by introducing types such as 
`Side` and `OrderType`, which restrict the possible values (buy/sell, market/limit). 
Then we define two main classes: `MarketState`, representing the market state at a given point in time, 
and `Fill`, representing an actual order execution."""


"'Literal' allows defining two text types"

Side = Literal["buy", "sell"]   #this means a variable of type `Side` can ONLY be 'buy' or 'sell'

"Side for : buy(achat) or sell(vente)"

OrderType = Literal["market", "limit"]

"OrderType for : market order(ordre au marché) or limit order(ordre limite)"
#market order(ordre au marché) means: execute immediately, regardless of the price
#limit order(ordre limite) means: I want this price or better, otherwise I don't trade



"Now we create two immutable/frozen classes: a `MarketState` class representing" 
"the market state, and a `Fill` class representing an actually executed trade"

"we simulate the market of a single instrument"

@dataclass(frozen=True)
class MarketState:
    """One time-step snapshot of the simulated market."""
    t: int  #discrete time 
    mid: float  #mid-price, the central market price
    bid: float  #best bid price
    ask: float  #best ask price
    toxicity: float  
    #latent toxicity indicator between 0 and 1
    #toxicity is an abstract variable representing how much trading now risks triggering or revealing an adverse price move right after execution
    #toxicity est une variable abstraite qui représente à quel point trader maintenant risque de déclencher ou révéler une évolution de prix défavorable juste après l'exécution
    
"A Fill = an actually executed trade" 
"un ordre = intention ; un fill = réalité"
"an order = intention ; a fill = reality"

@dataclass(frozen=True)
class Fill:
    """Execution fill event (événement d'exécution)."""
    t: int  #time at which execution occurs
    side: Side  #buy or sell
    qty : float  #executed quantity
    price: float   #price at which you were executed
    order_type: OrderType   #how you were executed : market : aggressive / limit : passive
    note: Optional[str] = None  #additional info (debug, analysis)
