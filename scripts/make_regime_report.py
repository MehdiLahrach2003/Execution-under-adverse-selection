


"""
Ce script construit un rapport markdown + heatmaps à partir de reports/regime_grid.csv.

Cette version correspond aux colonnes du regime_grid.csv actuel :
Disponibles :
- as_kick_scale, tox_persist
- AlwaysMarket_mean, AlwaysMarket_p90, AlwaysMarket_fill_rate, AlwaysMarket_avg_first_fill_t
- ToxicityAware_mean, ToxicityAware_p90, ToxicityAware_fill_rate, ToxicityAware_avg_first_fill_t
- delta_mean, delta_p90, delta_avg_first_fill_t

On va :
- calculer delta_fill_rate = ToxicityAware_fill_rate - AlwaysMarket_fill_rate (si absent)
- ajouter des deltas ajustés au risque :
    delta_mean_rel = delta_mean / abs(AlwaysMarket_mean)
    delta_p90_rel  = delta_p90  / abs(AlwaysMarket_p90)
- tracer des heatmaps pour :
    delta_mean, delta_p90, delta_fill_rate, delta_avg_first_fill_t, delta_mean_rel, delta_p90_rel
- écrire reports/REGIME_REPORT.md
"""




from __future__ import annotations  # Ça permet à Python de gérer plus souplement les annotations de type.

from dataclasses import dataclass  # pour définir une petite structure de configuration des figures
from pathlib import Path  # pour gérer les chemins des fichiers proprement 
from typing import Iterable

import numpy as np   # pour les calculs numériques 
import pandas as pd  # pour lire le CSV et manipuler les tableaux 
import matplotlib.pyplot as plt  # pour dessiner les heatmaps



"""
Le bloc suivant fixe les entrées et les sorties
"""



# le dossier général des rapports
REPORTS_DIR = Path("reports")

# Sous-dossier où on va mettre les figures
FIG_DIR = REPORTS_DIR / "figures"

# Le fichier csv qu'on lit
CSV_PATH = REPORTS_DIR / "regime_grid.csv"

# Le fichier Markdown qu'on va générer
OUT_MD = REPORTS_DIR / "REGIME_REPORT.md"




"""
Le script fixe les les colonnes qu'il juge indispensables
"""



COLS_REQUIRED = [
    "as_kick_scale",
    "tox_persist",
    "AlwaysMarket_mean",
    "ToxicityAware_mean",
    "delta_mean",
    "AlwaysMarket_p90",
    "ToxicityAware_p90",
    "delta_p90",
    "AlwaysMarket_fill_rate",
    "ToxicityAware_fill_rate",
    # delta_fill_rate peut manquer -> on la calcule
    "delta_avg_first_fill_t",
]




"""
Cette classe sert à décrire une heatmap
"""



@dataclass(frozen=True)
class HeatmapSpec:
    # Colonne à visualiser
    value_col: str
    # Le titre
    title: str
    # Le nom du fichier image
    filename: str



"""
La fonction suivante vérifie que le DataFrame contient bien toutes les colonnes requises
Donc cette fonction protège le script contre un CSV mal formé
"""



def _assert_cols(df: pd.DataFrame, required: Iterable[str]) -> None:
    
    # On construit la liste des colonnes demandées mais absente
    missing = [c for c in required if c not in df.columns]
    
    # Si cette liste n’est pas vide, on lève une erreur
    if missing:
        raise ValueError(
            "regime_grid.csv is missing expected columns:\n"
            f"- Missing: {missing}\n"
            f"- Available: {list(df.columns)}\n"
        )



"""
Cette fonction fait une division en évitant les divisions par zéro
Cette fonction sera utilisée pour calculer les deltas relatifs
"""



def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    
    # On prend la valeur absolue du dénominateur
    # En effet, on veut diviser par la taille du benchmark, peu importe son signe
    den_abs = np.abs(den)
    
    # On crée un tableau de sortie rempli de NaN
    out = np.full_like(num, fill_value=np.nan, dtype=float)
    
    # On repère les positions où le dénominateur est strictement non nul
    mask = den_abs > 0.0
    
    # On ne fait la division que là où c’est autorisé.
    out[mask] = num[mask] / den_abs[mask]
    
    # On renvoie le tableau final
    return out



"""
Cette fonction lit le CSV et le prépare
"""

def _load() -> pd.DataFrame:
    
    # On vérifie si e CSV existe. Si il n'existe pas, le message dit : lance d’abord le script de grille de régimes
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Missing {CSV_PATH}. Run the regime grid script first.")
    
    # On charge le csv dans un DataFrame pandas
    df = pd.read_csv(CSV_PATH)
    
    # On appelle la fonction précédente pour vérifier que tout est là
    _assert_cols(df, COLS_REQUIRED)

    # On parcourt les colonnes, et pour celles qui nous intéressent, on force leur conversion en numérique
    # Si une valeur n’est pas convertible, elle devient NaN. C'est plus sûr que de laisser planter
    for c in df.columns:
        if c in [
            "as_kick_scale",
            "tox_persist",
            "AlwaysMarket_mean",
            "ToxicityAware_mean",
            "delta_mean",
            "AlwaysMarket_p90",
            "ToxicityAware_p90",
            "delta_p90",
            "AlwaysMarket_fill_rate",
            "ToxicityAware_fill_rate",
            "AlwaysMarket_avg_first_fill_t",
            "ToxicityAware_avg_first_fill_t",
            "delta_avg_first_fill_t",
        ]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # On calcule delta_fill_rate si absent. Si la colonne n’est pas déjà dans le CSV, on la crée
    if "delta_fill_rate" not in df.columns:
        df["delta_fill_rate"] = df["ToxicityAware_fill_rate"] - df["AlwaysMarket_fill_rate"]

    """
    On crée deux nouvelles colonnes :
    1-/ delta_mean_rel : Écart relatif du coût moyen.
    2-/ delta_p90_rel : Écart relatif du risque p90.
    """
    
    df["delta_mean_rel"] = _safe_div(df["delta_mean"].to_numpy(), df["AlwaysMarket_mean"].to_numpy())
    df["delta_p90_rel"] = _safe_div(df["delta_p90"].to_numpy(), df["AlwaysMarket_p90"].to_numpy())

    # On retourne le DataFrame
    return df



"""
La fonction suivante transforme le DataFrame en matrice 2D.
"""



def _pivot(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    
    """
    On veut :
    - lignes = tox_persist
    - colonnes = as_kick_scale
    - cellules = valeur de value_col
    Donc on prépare les données pour un affichage en heatmap
    """
    piv = df.pivot(index="tox_persist", columns="as_kick_scale", values=value_col)
    
    # On trie les lignes et les colonnes pour que les axes soient dans l’ordre croissant
    piv = piv.sort_index(axis=0).sort_index(axis=1)
    
    return piv



"""
La fonction suivante dessine et sauvegarde une heatmap
"""

def _plot_heatmap(piv: pd.DataFrame, title: str, out_path: Path) -> None:
    
    # On convertit le pivot pandas en tableau NumPy
    arr = piv.to_numpy(dtype=float)

    # On ouvre une figure de taille 10x6
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # imshow affiche le tableau comme une image colorée. Chaque case devient une cellule colorée.
    im = ax.imshow(arr, aspect="auto")

    # On précise ce que représente la figure
    ax.set_title(title)
    ax.set_xlabel("as_kick_scale")
    ax.set_ylabel("tox_persist")

    # On place les repères sur les axes, puis on affiche les vraies valeurs des paramètres
    # L’utilisateur pourra lire : as_kick_scale = 0.03, tox_persist = 0.85 directement sur le graphique
    ax.set_xticks(np.arange(piv.shape[1]))
    ax.set_yticks(np.arange(piv.shape[0]))
    ax.set_xticklabels([f"{x:.2f}" for x in piv.columns.to_numpy(dtype=float)])
    ax.set_yticklabels([f"{y:.2f}" for y in piv.index.to_numpy(dtype=float)])
    
    """
    On parcourt toutes les cases de la heatmap.
    Si la valeur est finie, on l’écrit au centre de la case.
    C’est très utile  car la couleur donne une intuition visuelle et le texte donne la valeur exacte
    """

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=9)

    # On ajoute une échelle des couleurs.
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("value")

    """
    Le bloc suivant permet d'ajuster et de sauvegarder
    """
    
    # La ligne ci-dessous évite que les éléments se chevauchent
    fig.tight_layout()
    
    # La ligne ci-dessous enregistre l’image
    fig.savefig(out_path, dpi=180)
    
    # La ligne ci-dessous ferme la figure en mémoire
    plt.close(fig)



"""
La fonction suivante cherche :
- la meilleure ligne
- la pire ligne
pour une colonne donnée
"""



def _best_worst(df: pd.DataFrame, col: str) -> tuple[pd.Series, pd.Series]:
    
    # On enlève les lignes où la colonne est NaN
    d = df[np.isfinite(df[col].to_numpy())].copy()
    
    # Si tout est vide, on renvoie une ligne par défaut.
    if d.empty:
        return df.iloc[0], df.iloc[0]
    
    # Trouve l’indice de la valeur minimale
    best = d.loc[d[col].idxmin()]
    
    # Trouve l’indice de la valeur maximale
    worst = d.loc[d[col].idxmax()]
    
    return best, worst



"""
La fonction suivante fabrique le fichier REGIME_REPORT.md
"""



def _write_report(df: pd.DataFrame, specs: list[HeatmapSpec]) -> None:
    
    # On cherche les extrêmes pour chaque métrique importante
    best_mean, worst_mean = _best_worst(df, "delta_mean")
    best_p90, worst_p90 = _best_worst(df, "delta_p90")
    best_mean_rel, worst_mean_rel = _best_worst(df, "delta_mean_rel")
    best_p90_rel, worst_p90_rel = _best_worst(df, "delta_p90_rel")
    best_fill, worst_fill = _best_worst(df, "delta_fill_rate")
    best_firstfill, worst_firstfill = _best_worst(df, "delta_avg_first_fill_t")



    """
    La fonction suivante prend une ligne du DataFrame et renvoie une jolie chaîne qui dit à quel régime on se trouve
    """
    
    
    
    def at(r: pd.Series) -> str:
        return f"(as_kick_scale={r['as_kick_scale']:.2f}, tox_persist={r['tox_persist']:.2f})"

    # Le rapport sera construit ligne par ligne dans cette liste
    lines: list[str] = []
    
    # Titre principal
    lines.append("# Regime Report — Execution under Adverse Selection\n")
    
    # Le rapport précise d’où viennent les données
    lines.append(f"Artifacts generated from `{CSV_PATH.as_posix()}`.\n")
    
    
    
    """
    La partie suivante explique le sens économique du rapport.
    Très important : tous les deltas sont calculés comme ToxicityAware - AlwaysMarket
    Donc :
    - delta négatif = ToxicityAware meilleure
    - delta positif = ToxicityAware pire
    """
    lines.append("## What this evaluates\n")
    lines.append(
        "We sweep the *true* market regime parameters:\n"
        "- `as_kick_scale` (adverse-selection kick intensity)\n"
        "- `tox_persist` (toxicity persistence)\n\n"
        "We compare `ToxicityAware` vs `AlwaysMarket`.\n"
        "All deltas are `ToxicityAware - AlwaysMarket` (negative = improvement if IS is a cost).\n"
    )



    """
    La partie suivante permet de résumer les meilleurs et pires cas pour :
    - coût moyen
    - coût p90
    C’est un résumé automatique des extrêmes
    """
    lines.append("## Key takeaways (raw deltas)\n")
    lines.append(f"- Best **mean IS delta**: {best_mean['delta_mean']:.6f} at {at(best_mean)}")
    lines.append(f"- Worst **mean IS delta**: {worst_mean['delta_mean']:.6f} at {at(worst_mean)}\n")
    lines.append(f"- Best **p90 IS delta**: {best_p90['delta_p90']:.6f} at {at(best_p90)}")
    lines.append(f"- Worst **p90 IS delta**: {worst_p90['delta_p90']:.6f} at {at(worst_p90)}\n")
    


    """
    Dans la partie suivante le rapport va expliquer les deltas relatifs. Puis il affiche :
    - meilleur et pire delta_mean_rel
    - meilleur et pire delta_p90_rel
    Donc on a une double lecture :
    - absolue
    - relative
    """
    lines.append("## Key takeaways (risk-adjusted deltas)\n")
    lines.append("We report relative deltas to normalize by baseline magnitude:\n")
    lines.append("- `delta_mean_rel = delta_mean / abs(AlwaysMarket_mean)`\n")
    lines.append("- `delta_p90_rel  = delta_p90 / abs(AlwaysMarket_p90)`\n")
    lines.append(f"- Best **mean IS delta (rel)**: {best_mean_rel['delta_mean_rel']:.6f} at {at(best_mean_rel)}")
    lines.append(f"- Worst **mean IS delta (rel)**: {worst_mean_rel['delta_mean_rel']:.6f} at {at(worst_mean_rel)}\n")
    lines.append(f"- Best **p90 IS delta (rel)**: {best_p90_rel['delta_p90_rel']:.6f} at {at(best_p90_rel)}")
    lines.append(f"- Worst **p90 IS delta (rel)**: {worst_p90_rel['delta_p90_rel']:.6f} at {at(worst_p90_rel)}\n")
    
    
    
    """
    Dans la partie suivante le rapport regarde :
    - delta_fill_rate
    - delta_avg_first_fill_t
    Et il explique que delta_avg_first_fill_t < 0 signifie que ToxicityAware remplit plus tôt
    """
    lines.append("## Execution quality metrics\n")
    lines.append(f"- Best **delta_fill_rate**: {best_fill['delta_fill_rate']:.6f} at {at(best_fill)}")
    lines.append(f"- Worst **delta_fill_rate**: {worst_fill['delta_fill_rate']:.6f} at {at(worst_fill)}\n")
    lines.append(
        "We also track `avg_first_fill_t` (average time-to-first-fill). "
        "A negative `delta_avg_first_fill_t` indicates ToxicityAware fills earlier.\n"
    )
    lines.append(f"- Best **delta_avg_first_fill_t**: {best_firstfill['delta_avg_first_fill_t']:.6f} at {at(best_firstfill)}")
    lines.append(f"- Worst **delta_avg_first_fill_t**: {worst_firstfill['delta_avg_first_fill_t']:.6f} at {at(worst_firstfill)}\n")

    # Pour chaque heatmap, on ajoute une ligne Markdown qui insère l’image. Donc le rapport final contiendra directement les figures.
    lines.append("## Figures\n")
    for spec in specs:
        lines.append(f"- ![{spec.value_col}](reports/figures/{spec.filename})")

    # On concatène toutes les lignes et on les écrit dans reports/REGIME_REPORT.md
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")



"""
La fonction suivante est la fonction principale du script
"""



def main() -> None:
    
    # On s’assure que les dossiers existent et on les crée si nécessaire
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # On lit et prépare le CSV
    df = _load()



    """
    On prépare 6 figures. Chaque ligne décrit :
    - la colonne à utiliser
    - le titre de la figure
    - le nom du fichier image
    """
    specs = [
        HeatmapSpec("delta_mean", "Regime grid: delta_mean (ToxicityAware - AlwaysMarket)", "regime_delta_mean_heatmap.png"),
        HeatmapSpec("delta_p90", "Regime grid: delta_p90 (ToxicityAware - AlwaysMarket)", "regime_delta_p90_heatmap.png"),
        HeatmapSpec("delta_fill_rate", "Regime grid: delta_fill_rate (ToxicityAware - AlwaysMarket)", "regime_delta_fillrate_delta_heatmap.png"),
        HeatmapSpec("delta_avg_first_fill_t", "Regime grid: delta_avg_first_fill_t (ToxicityAware - AlwaysMarket)", "regime_delta_avg_first_fill_t_heatmap.png"),
        HeatmapSpec("delta_mean_rel", "Regime grid: delta_mean_rel (delta_mean / |AlwaysMarket_mean|)", "regime_delta_mean_rel_heatmap.png"),
        HeatmapSpec("delta_p90_rel", "Regime grid: delta_p90_rel (delta_p90 / |AlwaysMarket_p90|)", "regime_delta_p90_rel_heatmap.png"),
    ]



    """
    Pour chaque figure :
    - on construit le pivot
    - on construit le chemin de sortie
    - on trace et on sauvegarde
    """
    for spec in specs:
        piv = _pivot(df, spec.value_col)
        out_path = FIG_DIR / spec.filename
        _plot_heatmap(piv, spec.title, out_path)

    # Après les figures, on produit le Markdown
    _write_report(df, specs)

    # Le script nous dit où trouver les images et le rapport 
    print(f"Saved figures in: {FIG_DIR.as_posix()}")
    print(f"Saved report: {OUT_MD.as_posix()}")



"""
Les lignes suivantes signifient que si on lance directement ce fichier, alors Python appelle la fonction main()
"""



if __name__ == "__main__":
    main()
