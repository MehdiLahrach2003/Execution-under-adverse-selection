


"""
Ce script sert à :
    - définir un cadre simple de backtest,
    - choisir plusieurs stratégies,
    - lancer plusieurs simulations avec des seeds différentes,
    - récupérer les résultats,
    - comparer les stratégies avec quelques statistiques simples.

Donc ce script répond à la question :
sur plusieurs simulations, quelle stratégie semble meilleure ?

Il répond à la question : “dans une configuration donnée, quelle stratégie gagne ?”
"""



from __future__ import annotations
import numpy as np



"""
On importe depuis le backtest :
    - la classe des paramètres de l’expérience,
    - la fonction qui lance une simulation complète.
"""
from execlab.backtest.mvp import BacktestParams, run_backtest



"""
On importe les trois stratégies qu’on veut comparer
"""
from execlab.strategy.execution import AlwaysLimit, AlwaysMarket, ToxicityAwareExecution



"""
La fonction suivante sert à résumer une liste de nombres
Elle est très utile parce que le projet ne veut pas seulement regarder la moyenne, mais aussi :
à quoi ressemblent les mauvais cas.
Cette fonction est déjà plus informative qu’une simple moyenne.
"""
def summarize(x: list[float]) -> dict[str, float]:
    
    # On transforme la liste Python en tableau NumPy
    # En effet, Numpy facilite les calculs statistiques
    a = np.array(x, dtype=float)
    
    # On enlève les NaN
    # Par exemple pour les cas ou il n'y a aucun fill, donc pas d'Implementation Shortfall définie
    a = a[~np.isnan(a)]
    
    if a.size == 0:
        
        # Dans ce cas, on renvoie un dictionnaire rempli de nan
        return {"mean": float("nan"), "p10": float("nan"), "p50": float("nan"), "p90": float("nan")}
    
    return {
        # La moyenne
        "mean": float(a.mean()),
        # Le 10e percentile (valeur basse)
        "p10": float(np.quantile(a, 0.10)),
        # La médiane (milieu)
        "p50": float(np.quantile(a, 0.50)),
        # Le 90e percentile (valeur haute, donc plutôt cas défavorable)
        "p90": float(np.quantile(a, 0.90)),
    }



"""
La fonction suivante est la fonction principale du script. C'est elle qui lance toute l'expérience
"""
def main() -> None:
    
    # On définit un cadre d'expérience ici
    bp = BacktestParams(horizon=200, parent_qty=1.0, side="buy")
    # Donc on compare les tratégies ou il faut acheter une unité au maximum sur 200 steps

    # On crée un dictionnaire de stratégies
    # Chaque entrée associe un nom lisible à un objet stratégie
    policies = {
        # Stratégie agressive (tout de suite, au marché)
        "AlwaysMarket": AlwaysMarket(),
        # Stratégie passive
        "AlwaysLimit@0": AlwaysLimit(limit_offset=0.0),
        # Stratégie adaptative (attende si toxicité élevée, sinon exécuter par tranches, au marché)
        "ToxicityAware": ToxicityAwareExecution(),
    }

    """
    Une seule simulation ne suffit pas :
       - il y a du bruit,
       - de l’aléa,
       - des trajectoires différentes.
       Donc on veut comparer les stratégies sur plusieurs mondes aléatoires différents.
    """

    # On va lancer 50 simulations sur chaque stratégie 
    seeds = list(range(50))
    
    # Dictionnaire vide qui contiendra les résultats agrégés de chaque stratégie
    results = {}

    for name, policy in policies.items():
        
        # Cette liste contiendra les valeurs d'Implementation Shortfall obtenues sur les 50 simulations
        costs = []
        
        # Cette liste contiendra des 0 ou 1 selon que l’ordre a été complètement exécuté ou non
        fill_rates = []
        
        # Cette liste contiendra le temps du premier fill, quand il existe
        fill_times = []

        for seed in seeds:
            
            # Dans la ligne de code suivante on lance le backtest 50 fois pour la stratégie courante, avec des seeds différentes
            # out est le dictionnaire renvoyé par run_backtest(...)
            out = run_backtest(policy=policy, bp=bp, seed=seed)
            
            # On prend le coût d’exécution renvoyé par le backtest et on l’ajoute à la liste des coûts
            # Donc à la fin des 50 runs : costs contient 50 IS
            costs.append(out["implementation_shortfall"])
            
            # On regarde si la stratégie a tout exécuté
            filled = 1.0 if out["remaining"] <= 1e-12 else 0.0
            fill_rates.append(filled)
            """
            En calcul numérique, on évite souvent les comparaisons exactes à zéro avec des floats
            Donc 1e-12 est un seuil minuscule qui joue le rôle de "pratiquement zéro"
            """

            # On prend le temps du premier fill
            t = out["first_fill_t"]
            if t is not None:
                fill_times.append(float(t))

        # À la fin des 50 runs pour une stratégie, on construit un résumé
        results[name] = {
            
            # On applique la fonction summarize aux coûts
            "IS": summarize(costs),
            
            # Comme fill_rates contient des 0 et des 1 alors la moyenne correspond à la proportion de succès de remplissage
            "fill_rate": float(np.mean(fill_rates)),
            
            # On calcule le temps moyen du premier fill
            "avg_first_fill_t": float(np.mean(fill_times)) if len(fill_times) > 0 else float("nan"),

        }

    print("\n=== Execution baseline comparison ===")
    for name, res in results.items():
        print("-" * 60)
        print(name)
        print(f"Fill rate: {res['fill_rate']:.2f}")
        print(f"IS stats : {res['IS']}")
        print(f"Avg first fill t: {res['avg_first_fill_t']:.2f}")


# Cette ligne signifie : si on lance ce fichier directement, alors exécuter main()
if __name__ == "__main__":
    main()
