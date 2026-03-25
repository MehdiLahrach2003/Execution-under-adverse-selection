


"""
Ce fichier sert à construire une expérience minimale complète.

Autrement dit, il prend : une stratégie, un marché simulé, et une métrique

et il les relie pour répondre à la question : 

        Si j’utilise cette stratégie dans ce marché, qu’est-ce qui se passe concrètement ?

Donc ce fichier :

- lance une simulation,
- exécute les ordres,
- collecte les fills,
- calcule l’Implementation Shortfall,
- renvoie les résultats.

        C'est le premier vrai laboratoire d'expériences du projet
"""





from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from execlab.metrics.execution import fill_price_stats, implementation_shortfall
from execlab.sim.market import MarketParams, SimpleMarket
from execlab.strategy.execution import ExecutionPolicy, Action, AlwaysMarket
from execlab.types import Fill, Side






"""
On crée une classe contenant les paramètres du backtest 
Cette classe permet de répondre à la question : dans quelles conditions je lance mon expérience ?
"""

@dataclass
class BacktestParams:
    
    # Nombre maximum de pas de temps dans la simulation
    horizon: int = 200  
    
    # C'est la quantité totale à exécuter (ici 1 unité, peut être 1M d'euros, 1000 actions ...)
    parent_qty: float = 1.0  
    
     # Indique dans quelle direction on trade
    side: Side = "buy" 






"""
La fonction définie ci-dessous permet de transformer une décision de la stratégie en exécution réelle
La stratégie dit :   “je veux acheter 0.3 au marché”
Cette fonction répond :

1-/ Est-ce que c’est possible ?
2-/ A quel prix ?
3-/ Combien est exécuté ?
4-/ Combien il reste ? 
"""

def _try_execute_action(mkt: SimpleMarket, action: Action, state, side: Side, remaining: float) -> tuple[list[Fill], float]:
    
    """
    Apply one Action to the market state and return (fills, new_remaining).

    Note: this is a simplified execution model for clarity (MVP).
    """
    
    # La stratégie peut demander une certaine quantité X, mais s’il ne reste à exécuter que Y<X, alors on ne peut exécuter que Y 
    qty = min(action.qty, remaining)
    
    if qty <= 0:
        return [], remaining

    # on prépare une liste vide qui contiendra les exécutions éventuelles
    fills: list[Fill] = []  

    # Cas 1 : ordre au marché
    if action.order_type == "market":
        
        # Si on veut acheter au marché, on exécute au ask
        # Si on veut vendre au marché, on exécute au bid
        px = state.ask if side == "buy" else state.bid
        
        """
        La ligne suivante n’exécute pas le trade, mais ça dit au marché : le trader vient d'agir agressivement (car ordre au marché donc agressif)
        Conséquence : Le marché peut soit déclencher adverse selection, soit augmenter la toxicité, soit préparer un choc futur
        """
        
        mkt.apply_trade(side=side, aggressiveness=action.aggressiveness)
        
        # On crée le fill réel (temps, côté, quantité, prix, type d'ordre)
        fills.append(Fill(t=state.t, side=side, qty=qty, price=px, order_type="market"))
        
        return fills, remaining - qty

    # Cas 2 : ordre limite d'achat
    if side == "buy":
        
        # dans ce cas, le bid est le meilleur prix auquel quelqu'un peut acheter
        limit_px = state.bid - action.limit_offset   
        
        # Règle simplifiée de remplissage : si le mid descend suffisamment bas, alors on considère que l’ordre limite est exécuté.
        if state.mid <= limit_px:
            
            # Idem que dans le premier cas (l'agressivité du trade sera ici plus faible)
            
            mkt.apply_trade(side=side, aggressiveness=action.aggressiveness)
            fills.append(Fill(t=state.t, side=side, qty=qty, price=limit_px, order_type="limit"))
            return fills, remaining - qty
        return [], remaining
    
    # Cas 3 : Ordre limite de vente
    else:
        
        # dans ce cas, le ask est le meilleur prix auquel quelqu'un peut vendre
        limit_px = state.ask + action.limit_offset
        
        # Règle simplifiée de remplissage : si le mid monte suffisamment bas, alors on considère que l’ordre limite est exécuté.
        if state.mid <= limit_px:
            if state.mid >= limit_px:
                mkt.apply_trade(side=side, aggressiveness=action.aggressiveness)
                fills.append(Fill(t=state.t, side=side, qty=qty, price=limit_px, order_type="limit"))
                return fills, remaining - qty
        return [], remaining






"""La fonction définie ci-dessous est la fonction centrale du fichier
Elle lance une expérience complète avec : 
- une stratégie
- des paramètres de backtest
- une seed
Elle renvoie un dictionnaire de résultats
"""

def run_backtest(
    policy: ExecutionPolicy,
    bp: BacktestParams,
    seed: int = 0,
    market_params: Optional[MarketParams] = None,
) -> dict:
    
    
    # Si on a fourni des paramètres de marché on les utilise, sinon on prend ceux par défaut
    mp = market_params or MarketParams()
    
    # On crée le marché simulé avec ces paramètres et cette seed
    mkt = SimpleMarket(mp, seed=seed)
    # A ce moment-là, le marché existe, il est initialisé et il est prêt à évoluer 

    # On fait évoluer le marché une première fois pour obtenir un premier état observable.
    s0 = mkt.step()
    
    # Puis on récupère le mid de cet état
    arrival_mid = s0.mid
    # Ce arrival_mid sera le prix de référence utilisé pour l'Implementation Shortfall

    # On prépare une liste vide pour accumuler toutes les exécutions réelles
    fills: list[Fill] = []
    
    # Au début, toute la quantité reste à exécuter.
    remaining = bp.parent_qty
    
    """
    Quand on avance d’un pas de temps, on obtient un nouvel état du marché. 
    À partir de cet état, la stratégie décide soit d’agir, soit d’attendre. 
    Si elle décide d’agir, le moteur d’exécution tente alors de transformer cette décision en fills réels.
    """

    # La simulation va avancer au maximum sur horizon étapes
    for _ in range(bp.horizon):
        
        """
        À chaque étape, on va :
            - faire évoluer le marché
            - demander à la stratégie quoi faire
            - essayer d’exécuter
        """
        
        # Le marché évolue : nouveau mid, nouveau bid, nouveau ask, nouvelle toxicité, et on obtient un nouvel état state
        state = mkt.step()

        # Si la stratégie a tout exécuté, alors on arrête la simulation
        if remaining <= 0:
            break

        """
        La stratégie observe :
            - l’état du marché,
            - la quantité restante,
            - le sens buy/sell
        Puis elle décide :
            - soit une Action
            - soit None
        """
        
        """
        Cette ligne veut dire : on donne à la stratégie l’état actuel du marché, la quantité restante,
        et le côté buy/sell, puis on lui demande : que veux-tu faire ?
        """
        
        """
        Cette ligne n’exécute rien sur le marché
        """
        
        action = policy.decide(state=state, remaining=remaining, side=bp.side)
        
        """Si la stratégie ne veut rien faire à ce step :
            - on ne crée aucun fill
            - on passe au step suivant
        C’est très important pour ToxicityAwareExecution, qui peut choisir d’attendre
        """
        
        if action is None:
            continue

        # Cette fonction renvoie les nouveaux fills obtenus ainsi que la nouvelle quantité restante
        new_fills, remaining = _try_execute_action(
            mkt=mkt,
            action=action,
            state=state,
            side=bp.side,
            remaining=remaining,
        )
        
        # On ajoute les nouveaux fills à la liste globale
        fills.extend(new_fills)
        # À la fin du backtest, fills contiendra toute l’histoire réelle des exécutions
    
    
    
    
    """
    Une fois la simulation terminée, on calcule l’Implementation Shortfall à partir :
        - des fills,
        - du benchmark initial,
        - du côté buy/sell.
    Donc la ligne suivante résume le coût final de la stratégie.
    """

    is_cost = implementation_shortfall(fills, arrival_mid=arrival_mid, side=bp.side)
    
    
    
    
    """
    Si au moins un fill existe, on récupère le temps du premier fill.
    Ça sert à savoir :
        - si la stratégie a exécuté rapidement
        - ou si elle a attendu longtemps avant de commencer
    Très utile pour analyser les stratégies adaptatives.
    """

    first_fill_t = fills[0].t if len(fills) > 0 else None




    """
    La fonction renvoie un dictionnaire avec toutes les informations importantes :
        - arrial-mid : le benchmark initial
        - fills : la liste des exécutions réelles
        - remaining : ce qu'il reste non exécuté
        - implementation_shortfall : la métrique principale
        - price_stats : statistiques descriptives sur les prix des fills
        - first_fill_t : le temps du premier fill
    """
    
    return {
        "arrival_mid": arrival_mid,
        "fills": fills,
        "remaining": remaining,
        "implementation_shortfall": is_cost,
        "price_stats": fill_price_stats(fills),
        "first_fill_t": first_fill_t,
    }
    
    

"""
La fonction ci-dessous est juste un petit exemple de lancement
Elle :
    - crée des paramètres de backtest par défaut
    - teste la stratégie AlwaysMarket
    - lance run_backtest
    - affiche quelques résultats
Donc cette fonction sert surtout à :
    - vérifier que le pipeline marche
    - avoir une démonstration minimale
"""

def run_mvp_example(seed: int = 42) -> None:
    bp = BacktestParams()

    for policy in [AlwaysMarket(), ]:
        out = run_backtest(policy=policy, bp=bp, seed=seed)
        print("=" * 60)
        print(f"Policy: {policy.__class__.__name__}")
        print(f"Arrival mid: {out['arrival_mid']:.4f}")
        print(f"IS (cost):   {out['implementation_shortfall']:.6f}")
        print(f"Remaining:   {out['remaining']}")
