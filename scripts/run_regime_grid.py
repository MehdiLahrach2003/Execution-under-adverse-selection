from __future__ import annotations

from dataclasses import is_dataclass, fields
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from execlab.sim.market import MarketParams


"""
Le script run_mvp.py nous disait seulement :
Dans une configuration donnée, quelle stratégie gagne ?

Alors que le script run_regime_grid.py cherche à répondre à une question beaucoup plus interessante : 
Dans quels régimes de marché chaque stratégie est-elle meilleure ? 

Donc ce script sert à construire une carte expérimentale de la performance des stratégies selon le régime de marché.
"""



"""
Ce script fait une grille d’expériences.

Il va faire varier deux paramètres du marché :

    - as_kick_scale
    --> C'est l'intensité du choc d'adverse selection, plus il est grand, plus un trade agressif sera puni fortement
    
    - tox_persist
    --> C'est la persistance de la toxicité donc :
        - plus il est grand, plus la toxicité reste élevée longtemps
        - plus le marché garde une mémoire de son état toxique
        Donc il mesure à quel point la toxicité est durable 

Puis, pour chaque combinaison de ces deux paramètres, il va comparer :
    - AlwaysMarket
    - ToxicityAware
sur plusieurs seeds.

Enfin, il va enregistrer les résultats dans un CSV
"""



"""
Configuration expérimentale (On fixe les paramètres globaux de l'expérience)
"""



# On utilisera 50 simulations par configuration
SEEDS = list(range(50))

# Liste de valeurs testées pour l'intensité de l'adverse selection
AS_KICK_SCALES = [0.00, 0.01, 0.02, 0.03, 0.05]

# Liste des valeurs testées pour la persistance de la toxicité
TOX_PERSIST = [0.60, 0.75, 0.85, 0.95]


PARENT_QTY = 1.0
HORIZON = 200        
SIDE = "buy"           # ou "vente"

OUT_CSV = "reports/regime_grid.csv"



"""
On définit quelques "helpers"
"""



# Cette fonction crée le dossier reports/ s'il n'existe pas
def ensure_reports_dir() -> None:
    import os
    os.makedirs("reports", exist_ok=True)

# Cette fonction calcule la moyenne d'une liste 
def mean(x: List[float]) -> float:
    return float(np.mean(np.asarray(x, dtype=float))) if len(x) else float("nan")

# Cette fonction calcule le 90e percentile d'une liste 
def q90(x: List[float]) -> float:
    return float(np.quantile(np.asarray(x, dtype=float), 0.9)) if len(x) else float("nan")

# Cette fonction sert à récupérer une valeur dans out
def _get_field(out: Any, name: str, default: Any = None) -> Any:
    if isinstance(out, dict):
        return out.get(name, default)
    return getattr(out, name, default)



""" 
On définit des fonctions de "dynamic resolution", facultatives ...
"""


# Cette fonction va chercher execlab.backtest.mvp.run_backtest, donc elle récupère dynamiquement la fonction principale de backtest
def resolve_runner() -> Callable[..., Any]:
    import execlab.backtest.mvp as mvp
    fn = getattr(mvp, "run_backtest", None)
    if callable(fn):
        return fn
    public = [n for n in dir(mvp) if not n.startswith("_")]
    raise ImportError(
        "Expected execlab.backtest.mvp.run_backtest to exist, but it was not found. "
        f"Available symbols: {public}"
    )

# Même logique : va dans execlab.backtest.mvp et récupère BacktestParams
def resolve_backtest_params_class():
    import execlab.backtest.mvp as mvp
    cls = getattr(mvp, "BacktestParams", None)
    if cls is None:
        public = [n for n in dir(mvp) if not n.startswith("_")]
        raise ImportError(
            "Expected execlab.backtest.mvp.BacktestParams to exist, but it was not found. "
            f"Available symbols: {public}"
        )
    return cls

# Cette fonction cherche une stratégie selon un mot-clé : "always_market", "toxicity_aware", ...
def resolve_policy_instance(kind: str) -> Any:
    import execlab.strategy.execution as ex

    if kind == "always_market":
        candidates = ["AlwaysMarketExecution", "AlwaysMarket", "AlwaysMarketPolicy", "AlwaysMarketExec"]
        for name in candidates:
            cls = getattr(ex, name, None)
            if isinstance(cls, type):
                return cls()
        for n in dir(ex):
            if n.startswith("_"):
                continue
            obj = getattr(ex, n)
            if isinstance(obj, type):
                low = n.lower()
                if "market" in low and ("always" in low or "simple" in low or "immediate" in low):
                    return obj()

    if kind == "toxicity_aware":
        candidates = ["ToxicityAwareExecution", "ToxicityAware", "ToxicityFilterExecution", "ToxicityBasedExecution"]
        for name in candidates:
            cls = getattr(ex, name, None)
            if isinstance(cls, type):
                return cls()
        for n in dir(ex):
            if n.startswith("_"):
                continue
            obj = getattr(ex, n)
            if isinstance(obj, type):
                low = n.lower()
                if "tox" in low or "toxic" in low:
                    return obj()

    available = [
        n for n in dir(ex)
        if not n.startswith("_") and isinstance(getattr(ex, n), type)
    ]
    raise ImportError(f"Could not resolve policy kind='{kind}'. Available classes: {available}")



""" 
On construit les paramètres 
"""



# Cette fonction crée des paramètres de marché en modifiant seulement as_kick_scale et tox_persist
# Le reste des paramètres de marché reste à sa valeur par défaut
# Donc chaque expérience change uniquement ces deux dimensions 
def build_market_params(as_kick_scale: float, tox_persist: float) -> MarketParams:
    return MarketParams(
        as_kick_scale=float(as_kick_scale),
        tox_persist=float(tox_persist),
    )

# Cette fonction crée un objet BacktestParams
def build_backtest_params() -> Any:
    BacktestParams = resolve_backtest_params_class()
    bp = BacktestParams()  

    if hasattr(bp, "horizon"):
        setattr(bp, "horizon", int(HORIZON))
    if hasattr(bp, "parent_qty"):
        setattr(bp, "parent_qty", float(PARENT_QTY))
    if hasattr(bp, "side"):
        setattr(bp, "side", SIDE)

    return bp



"""
On construit le Runner call logic
"""


# Cette fonction est technique parce qu'elle essaie plusieurs signatures possibles pour appeler run_backtest
# le point important est qu'elle permet de lancer le backtest avec une stratégie, un backtest paramétré, et une seed
def call_runner(runner: Callable[..., Any], policy: Any, bp: Any, mp: MarketParams, seed: int) -> Any:
    """
    Try several signatures until one works.
    We already know runner expects bp.parent_qty, so bp must be BacktestParams.
    MarketParams is passed separately.
    """
    candidates = [
        # Most likely
        {"policy": policy, "bp": bp, "market_params": mp, "seed": seed},
        {"policy": policy, "bp": bp, "mp": mp, "seed": seed},
        {"policy": policy, "bp": bp, "market": mp, "seed": seed},

        # Some codebases use params naming
        {"policy": policy, "params": bp, "market_params": mp, "seed": seed},
        {"policy": policy, "backtest_params": bp, "market_params": mp, "seed": seed},

        # Maybe no seed kwarg
        {"policy": policy, "bp": bp, "market_params": mp},
        {"policy": policy, "bp": bp, "mp": mp},
    ]

    for kw in candidates:
        try:
            return runner(**kw)
        except TypeError:
            pass

    # Positional fallback variants
    for args in (
        (policy, bp, mp, seed),
        (policy, bp, mp),
        (policy, bp, seed),
        (policy, bp),
    ):
        try:
            return runner(*args)
        except TypeError:
            pass

    raise TypeError(
        "Could not call run_backtest with any known signature. "
        "Please open src/execlab/backtest/mvp.py and tell me the def run_backtest(...) signature."
    )

# Cette fonction permet d'exécuter une stratégie donnée sur un régime de marché donné, sur plusieurs seeds, puis résumer les résultats
def run_policy_on_regime(
    runner: Callable[..., Any],
    policy_name: str,
    policy: Any,
    bp: Any,
    mp: MarketParams,
    seeds: List[int],
) -> Dict[str, float]:
    
    # On crée trois listes
    costs: List[float] = []
    fill_times: List[float] = []
    fill_rates: List[float] = []

    """
    Et pour chaque seed, la fonction
            - lance le backtest
            - récupère l'IS
            - récupère la quantité restante
            - récupère le temps du premier fill
    """
    for seed in seeds:
        out = call_runner(runner, policy, bp, mp, seed)

        # Coût d'exécution
        is_cost = float(_get_field(out, "implementation_shortfall"))
        
        # Permet de savoir si tout a été exécuté ou non
        remaining = float(_get_field(out, "remaining"))

        # Si disponible, on l’ajoute à la liste des temps de premier fill
        first_fill_t = _get_field(out, "first_fill_t", None)
        if first_fill_t is not None:
            fill_times.append(float(first_fill_t))

        costs.append(is_cost)
        fill_rates.append(1.0 if remaining <= 1e-12 else 0.0)
        
    """
    Donc pour chaque stratégie dans un régime donné, on obtient :
        - coût moyen
        - coût en queue haute
        - taux de remplissage
        - temps moyen du premier fill
    """    

    return {
        f"{policy_name}_mean": mean(costs),
        f"{policy_name}_p90": q90(costs),
        f"{policy_name}_fill_rate": mean(fill_rates),
        f"{policy_name}_avg_first_fill_t": mean(fill_times),
    }



""" 
On construit la fonction main() dans laquelle tout se met ensemble
"""



def main() -> None:
    
    # On crée le dossier de sortie
    ensure_reports_dir()

    # On récupère le run_backtest
    runner = resolve_runner()
    print(f"[regime_grid] Using runner: execlab.backtest.mvp.{runner.__name__}")

    # On crée les deux stratégies qu'on veut comparer (On ne compare pas AlwaysLimit)
    always_market = resolve_policy_instance("always_market")
    toxicity_aware = resolve_policy_instance("toxicity_aware")

    print(f"[regime_grid] Using policy AlwaysMarket: {always_market.__class__.__name__}")
    print(f"[regime_grid] Using policy ToxicityAware: {toxicity_aware.__class__.__name__}")

    # On construit un b cktest fixe
    bp = build_backtest_params()

    rows: List[Dict[str, float]] = []

    """
    C’est la vraie grille expérimentale.
    On a 5 valeurs de as_kick_scale et 4 valeurs de tox_persist
    Donc : 5x4 = 20 régimes de marché différents.
    Pour chacun : on compare 2 stratégies, chacune sur 50 seeds
    Donc au total : 20x2x50 = 2000 backtests.
    """
    
    for as_kick_scale in AS_KICK_SCALES:
        for tox_persist in TOX_PERSIST:
            
            # On crée le marché correspondant à ce régime
            mp = build_market_params(as_kick_scale=float(as_kick_scale), tox_persist=float(tox_persist))

            # On construit une ligne de résultats. Cette ligne contiendra tous les résultats pour ce régime
            row: Dict[str, float] = {
                "as_kick_scale": float(as_kick_scale),
                "tox_persist": float(tox_persist),
            }

            # On ajoute les résultats des stratégies. Cette ligne contient les performances des deux stratégies
            row.update(run_policy_on_regime(runner, "AlwaysMarket", always_market, bp, mp, SEEDS))
            row.update(run_policy_on_regime(runner, "ToxicityAware", toxicity_aware, bp, mp, SEEDS))

            # On calcule les deltats. Ces deltats donnent la différence entre ToxicityAware et AlwaysMarket
            
            """
            1-/ delta_mean
            Si :
                - négatif → ToxicityAware a un coût moyen plus faible → meilleure
                - positif → ToxicityAware a un coût moyen plus élevé → pire
            2-/ delta_p90
            Même logique, mais pour le risque en queue haute.
            3_/ delta_avg_first_fill_t
            Mesure combien ToxicityAware attend plus longtemps en moyenne avant le premier fill
            """
            
            row["delta_mean"] = row["ToxicityAware_mean"] - row["AlwaysMarket_mean"]
            row["delta_p90"] = row["ToxicityAware_p90"] - row["AlwaysMarket_p90"]
            row["delta_avg_first_fill_t"] = (
                row["ToxicityAware_avg_first_fill_t"] - row["AlwaysMarket_avg_first_fill_t"]
            )

            # On ajoute la ligne 
            rows.append(row)

    # On construit un tableau pandas avec toutes les lignes
    df = pd.DataFrame(rows).sort_values(["as_kick_scale", "tox_persist"]).reset_index(drop=True)
    # Puis on le sauvegarde dans :
    df.to_csv(OUT_CSV, index=False)

    """
    Le script affiche un sous-ensemble important des colonnes :
        - les paramètres du régime
        - les performances de chaque stratégie
        - les deltas
        - les fill rates
    """
    
    key_cols = [
        "as_kick_scale",
        "tox_persist",
        "AlwaysMarket_mean",
        "ToxicityAware_mean",
        "delta_mean",
        "AlwaysMarket_p90",
        "ToxicityAware_p90",
        "delta_p90",
        "AlwaysMarket_avg_first_fill_t",
        "ToxicityAware_avg_first_fill_t",
        "delta_avg_first_fill_t",
        "AlwaysMarket_fill_rate",
        "ToxicityAware_fill_rate",
    ]

    print("\n=== Regime grid summary (key columns) ===")
    print(df[key_cols].to_string(index=False))

    print(f"\nSaved: {OUT_CSV}")


if __name__ == "__main__":
    main()
