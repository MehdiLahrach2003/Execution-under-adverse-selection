"""
Construit un rapport markdown + des heatmaps à partir de reports/misspec_grid.csv.

Colonnes attendues dans le CSV :
- tox_persist_true
- tox_trigger_belief
- as_kick_scale
- AlwaysMarket_mean, AlwaysMarket_p90, AlwaysMarket_fill_rate
- ToxicityAware_mean, ToxicityAware_p90, ToxicityAware_fill_rate
- delta_mean, delta_p90, delta_fill_rate

On ajoute aussi des deltas relatifs :
- delta_mean_rel = delta_mean / abs(AlwaysMarket_mean)
- delta_p90_rel  = delta_p90  / abs(AlwaysMarket_p90)

Important :
Dans ce projet, la "belief" ne correspond pas à une croyance sur tox_persist.
La vraie variable de calibration de la stratégie est :
- tox_trigger_belief

Autrement dit :
- le marché réel est décrit par tox_persist_true et as_kick_scale
- la stratégie ToxicityAware est calibrée par tox_trigger_belief
"""

from __future__ import annotations

# On importe dataclass pour définir une petite classe de configuration (HeatmapSpec)
from dataclasses import dataclass

# Path permet de manipuler les chemins de fichiers proprement
from pathlib import Path

# Iterable sert à typer une fonction qui reçoit une collection de colonnes
from typing import Iterable

# NumPy pour les calculs numériques
import numpy as np

# Pandas pour charger et manipuler le CSV
import pandas as pd

# Matplotlib pour tracer les heatmaps
import matplotlib.pyplot as plt


"""
On définit les chemins importants.

Ce script :
- lit reports/misspec_grid.csv
- écrit des images dans reports/figures
- écrit un rapport markdown dans reports/MISSPEC_REPORT.md
"""

# Dossier général des rapports
REPORTS_DIR = Path("reports")

# Sous-dossier des figures
FIG_DIR = REPORTS_DIR / "figures"

# Fichier CSV d'entrée
CSV_PATH = REPORTS_DIR / "misspec_grid.csv"

# Fichier markdown de sortie
OUT_MD = REPORTS_DIR / "MISSPEC_REPORT.md"


"""
La liste suivante définit les colonnes minimales que le script exige.

Pourquoi ?
Parce que si une colonne importante manque, on préfère arrêter le script
avec une erreur claire plutôt que produire des figures fausses ou vides.
"""
COLS_REQUIRED = [
    "tox_persist_true",
    "tox_trigger_belief",
    "as_kick_scale",
    "AlwaysMarket_mean",
    "ToxicityAware_mean",
    "delta_mean",
    "AlwaysMarket_p90",
    "ToxicityAware_p90",
    "delta_p90",
    "AlwaysMarket_fill_rate",
    "ToxicityAware_fill_rate",
    "delta_fill_rate",
]


"""
Cette classe sert à décrire une heatmap.

Chaque heatmap a besoin de :
- la colonne à afficher
- le titre
- le nom du fichier image

On la rend immuable (frozen=True) parce qu’il s’agit d’une configuration fixe :
une fois définie, on ne veut pas la modifier.
"""
@dataclass(frozen=True)
class HeatmapSpec:
    value_col: str
    title: str
    filename: str


"""
La fonction suivante vérifie que le DataFrame contient bien toutes les colonnes attendues.
"""
def _assert_cols(df: pd.DataFrame, required: Iterable[str]) -> None:

    # On construit la liste des colonnes manquantes
    missing = [c for c in required if c not in df.columns]

    # S’il manque des colonnes, on arrête le script avec un message clair
    if missing:
        raise ValueError(
            "misspec_grid.csv is missing expected columns:\n"
            f"- Missing: {missing}\n"
            f"- Available: {list(df.columns)}\n"
        )


"""
Cette fonction permet d'effectuer une division sécurisée.

Pourquoi ?
Parce que pour calculer des deltas relatifs, on divise par une baseline
(AlwaysMarket_mean ou AlwaysMarket_p90), et il faut éviter les divisions par zéro.
"""
def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:

    # On prend la valeur absolue du dénominateur
    den_abs = np.abs(den)

    # On prépare un tableau de sortie rempli de NaN
    out = np.full_like(num, fill_value=np.nan, dtype=float)

    # On ne divisera que là où le dénominateur est strictement non nul
    mask = den_abs > 0.0

    # Division uniquement aux endroits valides
    out[mask] = num[mask] / den_abs[mask]

    return out


"""
Cette fonction lit le CSV et prépare les données.

Elle :
- vérifie que le fichier existe
- vérifie les colonnes
- convertit les colonnes utiles en numérique
- ajoute les deltas relatifs
"""
def _load() -> pd.DataFrame:

    # Si le fichier CSV n’existe pas, on demande d’abord de lancer run_misspec_grid.py
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Missing {CSV_PATH}. Run the misspec grid script first.")

    # On lit le CSV avec pandas
    df = pd.read_csv(CSV_PATH)

    # On vérifie que les colonnes attendues existent
    _assert_cols(df, COLS_REQUIRED)

    # On force toutes les colonnes requises à être numériques
    # Si une valeur est invalide, elle devient NaN
    for c in COLS_REQUIRED:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    """
    On calcule maintenant deux colonnes supplémentaires :

    1) delta_mean_rel
       différence relative sur le coût moyen

    2) delta_p90_rel
       différence relative sur le risque extrême

    L’idée est la suivante :
    un delta absolu de 0.01 n’a pas la même importance
    selon que la baseline vaut 0.10 ou 0.005.
    """
    df["delta_mean_rel"] = _safe_div(
        df["delta_mean"].to_numpy(),
        df["AlwaysMarket_mean"].to_numpy(),
    )

    df["delta_p90_rel"] = _safe_div(
        df["delta_p90"].to_numpy(),
        df["AlwaysMarket_p90"].to_numpy(),
    )

    return df


"""
Cette fonction prépare les données nécessaires pour tracer une heatmap.

Elle prend :
- le DataFrame complet
- la métrique à afficher
- une valeur fixée de tox_trigger_belief

Puis elle construit un tableau 2D où :
- les lignes = tox_persist_true
- les colonnes = as_kick_scale
- les cellules = la métrique choisie
"""
def _pivot(df: pd.DataFrame, value_col: str, tox_trigger_value: float) -> pd.DataFrame:

    # On garde uniquement les lignes correspondant à la valeur fixée du seuil de toxicité
    sub = df[np.isclose(df["tox_trigger_belief"].to_numpy(), tox_trigger_value)].copy()

    # On construit un pivot :
    # lignes = vraie persistance de toxicité
    # colonnes = vraie intensité de l'adverse selection
    # valeurs = métrique choisie
    piv = sub.pivot(index="tox_persist_true", columns="as_kick_scale", values=value_col)

    # On trie les axes pour avoir un ordre croissant
    piv = piv.sort_index(axis=0).sort_index(axis=1)

    return piv


"""
Cette fonction trace une heatmap à partir d'un pivot pandas.
"""
def _plot_heatmap(piv: pd.DataFrame, title: str, out_path: Path) -> None:

    # On convertit le pivot en matrice NumPy
    arr = piv.to_numpy(dtype=float)

    # On crée une figure matplotlib
    fig, ax = plt.subplots(figsize=(10, 6))

    # On affiche la matrice sous forme d'image colorée
    im = ax.imshow(arr, aspect="auto")

    # Titre et noms des axes
    ax.set_title(title)
    ax.set_xlabel("as_kick_scale (true)")
    ax.set_ylabel("tox_persist_true")

    # On place les repères sur les axes
    ax.set_xticks(np.arange(piv.shape[1]))
    ax.set_yticks(np.arange(piv.shape[0]))

    # On affiche les vraies valeurs des paramètres en format décimal
    ax.set_xticklabels([f"{x:.2f}" for x in piv.columns.to_numpy(dtype=float)])
    ax.set_yticklabels([f"{y:.2f}" for y in piv.index.to_numpy(dtype=float)])

    """
    On inscrit la valeur numérique dans chaque case de la heatmap.

    Cela permet une lecture précise :
    - la couleur donne l’intuition générale
    - le nombre donne la valeur exacte
    """
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=9)

    # On ajoute une barre de couleur
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("value")

    # On ajuste la mise en page, on sauvegarde et on ferme
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


"""
Cette fonction cherche :
- la ligne où une métrique est minimale
- la ligne où elle est maximale

Pour les deltas de coût :
- minimum = meilleur cas
- maximum = pire cas
"""
def _best_worst(df: pd.DataFrame, col: str) -> tuple[pd.Series, pd.Series]:

    # On filtre les lignes où la colonne est bien finie (pas NaN, pas inf)
    d = df[np.isfinite(df[col].to_numpy())].copy()

    # Si tout est vide, on renvoie une ligne arbitraire pour éviter un plantage
    if d.empty:
        return df.iloc[0], df.iloc[0]

    # Meilleur cas = valeur minimale
    best = d.loc[d[col].idxmin()]

    # Pire cas = valeur maximale
    worst = d.loc[d[col].idxmax()]

    return best, worst


"""
Fonction principale du script.

C’est ici que tout est orchestré :
- lecture du CSV
- construction des heatmaps
- recherche des meilleurs/pire cas
- écriture du rapport markdown
"""
def main() -> None:

    # On crée les dossiers de sortie si nécessaire
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # On charge les données
    df = _load()

    """
    On extrait les différentes valeurs distinctes de tox_trigger_belief.

    Chaque valeur correspond à une calibration différente de la stratégie ToxicityAware.
    Le script produira donc une famille de heatmaps :
    une par seuil de toxicité testé.
    """
    trigger_vals = sorted(df["tox_trigger_belief"].dropna().unique().tolist())

    """
    On calcule maintenant les meilleurs et pires cas globaux,
    tous seuils confondus, pour plusieurs métriques.
    """
    best_mean, worst_mean = _best_worst(df, "delta_mean")
    best_p90, worst_p90 = _best_worst(df, "delta_p90")
    best_mean_rel, worst_mean_rel = _best_worst(df, "delta_mean_rel")
    best_p90_rel, worst_p90_rel = _best_worst(df, "delta_p90_rel")

    """
    Le script produira cinq familles de heatmaps :

    - delta_mean       : écart absolu sur le coût moyen
    - delta_p90        : écart absolu sur le risque extrême
    - delta_fill_rate  : écart sur le taux de remplissage
    - delta_mean_rel   : écart relatif sur le coût moyen
    - delta_p90_rel    : écart relatif sur le risque extrême
    """
    specs = [
        HeatmapSpec(
            "delta_mean",
            "Misspec grid: delta_mean (ToxicityAware - AlwaysMarket)",
            "misspec_delta_mean",
        ),
        HeatmapSpec(
            "delta_p90",
            "Misspec grid: delta_p90 (ToxicityAware - AlwaysMarket)",
            "misspec_delta_p90",
        ),
        HeatmapSpec(
            "delta_fill_rate",
            "Misspec grid: delta_fill_rate (ToxicityAware - AlwaysMarket)",
            "misspec_delta_fillrate",
        ),
        HeatmapSpec(
            "delta_mean_rel",
            "Misspec grid: delta_mean_rel (delta_mean / |AlwaysMarket_mean|)",
            "misspec_delta_mean_rel",
        ),
        HeatmapSpec(
            "delta_p90_rel",
            "Misspec grid: delta_p90_rel (delta_p90 / |AlwaysMarket_p90|)",
            "misspec_delta_p90_rel",
        ),
    ]

    # Cette liste contiendra les liens Markdown vers les images générées
    fig_links: list[str] = []

    """
    Pour chaque valeur de tox_trigger_belief, et pour chaque métrique,
    on :
    - construit le pivot
    - trace la heatmap
    - sauvegarde l'image
    - mémorise le lien Markdown vers la figure
    """
    for tox_trigger in trigger_vals:
        for spec in specs:
            piv = _pivot(df, spec.value_col, tox_trigger_value=float(tox_trigger))
            out_name = f"{spec.filename}_trigger_{float(tox_trigger):.2f}.png"
            out_path = FIG_DIR / out_name
            title = f"{spec.title} — belief tox_trigger={float(tox_trigger):.2f}"
            _plot_heatmap(piv, title, out_path)
            fig_links.append(f"- ![{spec.value_col}](reports/figures/{out_name})")

    """
    On prépare maintenant le contenu du rapport Markdown ligne par ligne.
    """
    lines: list[str] = []

    # Titre principal
    lines.append("# Misspecification Report — Execution under Adverse Selection\n")

    # Référence au CSV source
    lines.append(f"Artifacts generated from `{CSV_PATH.as_posix()}`.\n")

    # Introduction
    lines.append("## What this evaluates\n")
    lines.append(
        "We compare a policy calibrated with a toxicity threshold belief (`tox_trigger_belief`) "
        "while the true market regime is defined by `tox_persist_true` and `as_kick_scale`.\n"
    )
    lines.append(
        "All deltas are `ToxicityAware - AlwaysMarket` "
        "(negative = improvement if Implementation Shortfall is a cost).\n"
    )

    # Résumé des meilleurs / pires cas en deltas bruts
    lines.append("## Best / worst (raw deltas)\n")
    lines.append(
        f"- Best **mean IS delta**: {best_mean['delta_mean']:.6f} at "
        f"(as_kick_scale={best_mean['as_kick_scale']:.2f}, "
        f"tox_true={best_mean['tox_persist_true']:.2f}, "
        f"tox_trigger={best_mean['tox_trigger_belief']:.2f})"
    )
    lines.append(
        f"- Worst **mean IS delta**: {worst_mean['delta_mean']:.6f} at "
        f"(as_kick_scale={worst_mean['as_kick_scale']:.2f}, "
        f"tox_true={worst_mean['tox_persist_true']:.2f}, "
        f"tox_trigger={worst_mean['tox_trigger_belief']:.2f})\n"
    )

    lines.append(
        f"- Best **p90 IS delta**: {best_p90['delta_p90']:.6f} at "
        f"(as_kick_scale={best_p90['as_kick_scale']:.2f}, "
        f"tox_true={best_p90['tox_persist_true']:.2f}, "
        f"tox_trigger={best_p90['tox_trigger_belief']:.2f})"
    )
    lines.append(
        f"- Worst **p90 IS delta**: {worst_p90['delta_p90']:.6f} at "
        f"(as_kick_scale={worst_p90['as_kick_scale']:.2f}, "
        f"tox_true={worst_p90['tox_persist_true']:.2f}, "
        f"tox_trigger={worst_p90['tox_trigger_belief']:.2f})\n"
    )

    # Résumé des meilleurs / pires cas en deltas relatifs
    lines.append("## Best / worst (risk-adjusted deltas)\n")
    lines.append(
        "We use `delta_*_rel = delta_* / abs(AlwaysMarket_*)` to interpret differences in relative terms.\n"
    )
    lines.append(
        f"- Best **mean IS delta (rel)**: {best_mean_rel['delta_mean_rel']:.6f} at "
        f"(as_kick_scale={best_mean_rel['as_kick_scale']:.2f}, "
        f"tox_true={best_mean_rel['tox_persist_true']:.2f}, "
        f"tox_trigger={best_mean_rel['tox_trigger_belief']:.2f})"
    )
    lines.append(
        f"- Worst **mean IS delta (rel)**: {worst_mean_rel['delta_mean_rel']:.6f} at "
        f"(as_kick_scale={worst_mean_rel['as_kick_scale']:.2f}, "
        f"tox_true={worst_mean_rel['tox_persist_true']:.2f}, "
        f"tox_trigger={worst_mean_rel['tox_trigger_belief']:.2f})\n"
    )
    lines.append(
        f"- Best **p90 IS delta (rel)**: {best_p90_rel['delta_p90_rel']:.6f} at "
        f"(as_kick_scale={best_p90_rel['as_kick_scale']:.2f}, "
        f"tox_true={best_p90_rel['tox_persist_true']:.2f}, "
        f"tox_trigger={best_p90_rel['tox_trigger_belief']:.2f})"
    )
    lines.append(
        f"- Worst **p90 IS delta (rel)**: {worst_p90_rel['delta_p90_rel']:.6f} at "
        f"(as_kick_scale={worst_p90_rel['as_kick_scale']:.2f}, "
        f"tox_true={worst_p90_rel['tox_persist_true']:.2f}, "
        f"tox_trigger={worst_p90_rel['tox_trigger_belief']:.2f})\n"
    )

    # Section figures
    lines.append("## Figures (grouped by belief threshold)\n")
    lines.extend(fig_links)

    # Écriture finale du fichier markdown
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Messages de sortie
    print(f"Saved figures in: {FIG_DIR.as_posix()}")
    print(f"Saved report: {OUT_MD.as_posix()}")


# Bloc de lancement
if __name__ == "__main__":
    main()