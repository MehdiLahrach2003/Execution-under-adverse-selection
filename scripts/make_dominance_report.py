"""
make_dominance_report.py

Crée un rapport de dominance / robustesse à partir de :
- reports/regime_grid.csv
- reports/misspec_grid.csv

Sorties :
- reports/DOMINANCE_REPORT.md
- des figures dans reports/figures/

Définitions :

1) Dominance (partie regime)
On dit que ToxicityAware domine AlwaysMarket si :
    delta_mean < 0 ET delta_p90 < 0

On dit qu'elle est dominée si :
    delta_mean > 0 ET delta_p90 > 0

Sinon :
    il y a un compromis (trade-off)

2) Robustesse (partie misspec)
On utilise des métriques de regret si elles existent :
    regret_mean, regret_p90

Une stratégie est dite robuste si :
    regret_p90 <= seuil

Si les colonnes de regret n’existent pas dans le misspec grid,
ce script construit des regrets proxy à partir des deltas :
    regret = max(delta, 0)

Cela est cohérent ici car on compare toujours à la baseline AlwaysMarket.
"""

from __future__ import annotations

# On importe dataclass pour définir une petite structure de seuils
from dataclasses import dataclass

# Path permet de manipuler proprement les chemins de fichiers
from pathlib import Path

# Ces types rendent le code plus lisible
from typing import Iterable

# NumPy pour les calculs numériques
import numpy as np

# Pandas pour charger et manipuler les CSV
import pandas as pd

# Matplotlib pour tracer les figures
import matplotlib.pyplot as plt


"""
On définit ici les chemins importants.

Le script lit :
- regime_grid.csv
- misspec_grid.csv

Le script écrit :
- des figures dans reports/figures/
- un rapport markdown dans reports/DOMINANCE_REPORT.md
"""

REPORTS_DIR = Path("reports")
FIG_DIR = REPORTS_DIR / "figures"
REGIME_CSV = REPORTS_DIR / "regime_grid.csv"
MISSPEC_CSV = REPORTS_DIR / "misspec_grid.csv"
OUT_MD = REPORTS_DIR / "DOMINANCE_REPORT.md"


# ----------------------------
# Helpers
# ----------------------------

"""
Cette fonction crée les dossiers de sortie si nécessaire.
"""
def _ensure_dirs() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


"""
Cette fonction vérifie qu’un DataFrame contient bien toutes les colonnes attendues.
"""
def _assert_cols(df: pd.DataFrame, required: Iterable[str], name: str) -> None:

    # On construit la liste des colonnes manquantes
    missing = [c for c in required if c not in df.columns]

    # Si des colonnes manquent, on lève une erreur claire
    if missing:
        raise ValueError(
            f"{name} is missing expected columns:\n"
            f"- Missing: {missing}\n"
            f"- Available: {list(df.columns)}\n"
        )


"""
Cette fonction essaie de convertir les colonnes du DataFrame en numériques
quand cela est possible.

Si une colonne n’est pas convertible, pandas la laisse telle quelle
grâce à errors="ignore".
"""
def _to_num(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="ignore")
    return df


"""
Cette fonction trace une heatmap numérique classique.

Elle est utilisée pour des valeurs quantitatives comme :
- regret_p90
- robust_frac
- toute autre métrique réelle
"""
def _plot_heatmap_numeric(piv: pd.DataFrame, title: str, out_path: Path) -> None:

    # On convertit le pivot en matrice NumPy
    arr = piv.to_numpy(dtype=float)

    # On crée la figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # On affiche la matrice comme une image colorée
    im = ax.imshow(arr, aspect="auto")

    # Titre et noms des axes
    ax.set_title(title)
    ax.set_xlabel(str(piv.columns.name) if piv.columns.name else "x")
    ax.set_ylabel(str(piv.index.name) if piv.index.name else "y")

    # On place les graduations sur les axes
    ax.set_xticks(np.arange(piv.shape[1]))
    ax.set_yticks(np.arange(piv.shape[0]))

    # On affiche les vraies valeurs des paramètres
    ax.set_xticklabels([f"{x:.2f}" for x in piv.columns.to_numpy(dtype=float)])
    ax.set_yticklabels([f"{y:.2f}" for y in piv.index.to_numpy(dtype=float)])

    # On écrit la valeur dans chaque case
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)

    # Barre de couleur
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("value")

    # Ajustement, sauvegarde, fermeture
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


"""
Cette fonction trace une heatmap catégorielle.

Ici, les valeurs de la matrice sont dans {-1, 0, +1} :
- +1 = ToxicityAware domine AlwaysMarket
- -1 = ToxicityAware est dominée
-  0 = compromis (trade-off)

On les affiche avec des labels :
- DOM = dominance
- BAD = dominée
- TRD = trade-off
"""
def _plot_heatmap_categorical(piv: pd.DataFrame, title: str, out_path: Path) -> None:
    arr = piv.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(arr, aspect="auto")

    ax.set_title(title)
    ax.set_xlabel(str(piv.columns.name) if piv.columns.name else "x")
    ax.set_ylabel(str(piv.index.name) if piv.index.name else "y")

    ax.set_xticks(np.arange(piv.shape[1]))
    ax.set_yticks(np.arange(piv.shape[0]))
    ax.set_xticklabels([f"{x:.2f}" for x in piv.columns.to_numpy(dtype=float)])
    ax.set_yticklabels([f"{y:.2f}" for y in piv.index.to_numpy(dtype=float)])

    """
    Petite fonction locale qui transforme :
    +1 en DOM
    -1 en BAD
     0 en TRD
    """
    def lab(v: float) -> str:
        if v > 0.5:
            return "DOM"
        if v < -0.5:
            return "BAD"
        return "TRD"

    # On inscrit l’étiquette dans chaque case
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            if np.isfinite(v):
                ax.text(j, i, lab(v), ha="center", va="center", fontsize=9)

    # Barre de couleur
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("DOM=+1, TRD=0, BAD=-1")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


# ----------------------------
# Regime dominance
# ----------------------------

"""
Cette fonction lit regime_grid.csv et construit une colonne de dominance.

Rappel :
- dominance = +1 si ToxicityAware domine
- dominance = -1 si ToxicityAware est dominée
- dominance = 0 sinon
"""
def _load_regime() -> pd.DataFrame:

    # Vérifie que le fichier existe
    if not REGIME_CSV.exists():
        raise FileNotFoundError(f"Missing {REGIME_CSV}")

    # Lecture du CSV
    df = pd.read_csv(REGIME_CSV)

    # Conversion numérique quand possible
    df = _to_num(df)

    # Colonnes minimales nécessaires
    req = ["as_kick_scale", "tox_persist", "delta_mean", "delta_p90"]
    _assert_cols(df, req, "regime_grid.csv")

    """
    On code la dominance ainsi :
    +1 si delta_mean < 0 et delta_p90 < 0
       => ToxicityAware est meilleure sur la moyenne ET le risque extrême
    -1 si delta_mean > 0 et delta_p90 > 0
       => ToxicityAware est pire sur les deux dimensions
     0 sinon
       => compromis
    """
    dom = np.zeros(len(df), dtype=int)
    dom[(df["delta_mean"] < 0) & (df["delta_p90"] < 0)] = 1
    dom[(df["delta_mean"] > 0) & (df["delta_p90"] > 0)] = -1
    df["dominance"] = dom

    return df


"""
Cette fonction construit un pivot pour la partie regime.

On veut :
- lignes = tox_persist
- colonnes = as_kick_scale
- valeurs = la colonne choisie
"""
def _pivot_regime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    piv = df.pivot(index="tox_persist", columns="as_kick_scale", values=col)
    piv = piv.sort_index(axis=0).sort_index(axis=1)
    piv.index.name = "tox_persist"
    piv.columns.name = "as_kick_scale"
    return piv


# ----------------------------
# Misspec robustness
# ----------------------------

"""
Cette petite classe stocke les seuils utilisés pour définir la robustesse.

Ici, par défaut :
- regret_p90 <= 0.0

Cela signifie une définition stricte :
la stratégie n’est robuste que si elle n’est pas pire que la baseline
sur le p90.
"""
@dataclass(frozen=True)
class RobustThresholds:
    regret_p90: float = 0.0


"""
Cette fonction lit misspec_grid.csv et ajoute des colonnes de regret si besoin.

Important :
dans ton projet actuel, la variable de calibration est :
- tox_trigger_belief

et non tox_persist_model.
"""
def _load_misspec() -> pd.DataFrame:

    # Vérifie que le fichier existe
    if not MISSPEC_CSV.exists():
        raise FileNotFoundError(f"Missing {MISSPEC_CSV}")

    # Lecture du CSV
    df = pd.read_csv(MISSPEC_CSV)

    # Conversion numérique quand possible
    df = _to_num(df)

    # Colonnes minimales nécessaires
    req = ["tox_persist_true", "tox_trigger_belief", "as_kick_scale", "delta_mean", "delta_p90"]
    _assert_cols(df, req, "misspec_grid.csv")

    """
    Si les colonnes de regret n’existent pas déjà, on les construit.

    Idée :
    - si delta < 0, ToxicityAware est meilleure => regret = 0
    - si delta > 0, ToxicityAware est pire => regret = delta

    Donc :
    regret = max(delta, 0)
    """
    if "regret_mean" not in df.columns:
        df["regret_mean"] = np.maximum(df["delta_mean"].astype(float), 0.0)

    if "regret_p90" not in df.columns:
        df["regret_p90"] = np.maximum(df["delta_p90"].astype(float), 0.0)

    return df


"""
Cette fonction calcule, pour chaque valeur de tox_trigger_belief,
la fraction de scénarios où la stratégie est robuste.

Définition :
robuste si regret_p90 <= seuil
"""
def _robust_fraction_by_belief(df: pd.DataFrame, thr: RobustThresholds) -> pd.DataFrame:

    rows = []

    # On groupe les lignes par valeur de tox_trigger_belief
    for tox_trigger, g in df.groupby("tox_trigger_belief"):
        g = g.copy()

        # On calcule la fraction de cas où regret_p90 est sous le seuil
        robust = (g["regret_p90"].astype(float) <= float(thr.regret_p90)).mean()

        rows.append(
            {
                "tox_trigger_belief": float(tox_trigger),
                "robust_frac": float(robust),
                "n": int(len(g)),
            }
        )

    out = pd.DataFrame(rows).sort_values("tox_trigger_belief").reset_index(drop=True)
    return out


"""
Cette fonction construit un pivot de misspecification pour une valeur donnée de tox_trigger_belief.

On veut :
- lignes = tox_persist_true
- colonnes = as_kick_scale
- valeurs = métrique choisie (ex: regret_p90)
"""
def _pivot_misspec(df: pd.DataFrame, value_col: str, tox_trigger_value: float) -> pd.DataFrame:
    sub = df[np.isclose(df["tox_trigger_belief"].astype(float), float(tox_trigger_value))].copy()
    piv = sub.pivot(index="tox_persist_true", columns="as_kick_scale", values=value_col)
    piv = piv.sort_index(axis=0).sort_index(axis=1)
    piv.index.name = "tox_persist_true"
    piv.columns.name = "as_kick_scale"
    return piv


# ----------------------------
# Report
# ----------------------------

"""
Fonction principale du script.

Elle orchestre :
- la dominance sur regime_grid
- la robustesse sur misspec_grid
- l’écriture du rapport final
"""
def main() -> None:
    _ensure_dirs()

    # --- Partie 1 : dominance sur la grille de régimes
    reg = _load_regime()

    # On construit la heatmap catégorielle de dominance
    piv_dom = _pivot_regime(reg, "dominance")
    p_dom = FIG_DIR / "regime_dominance_heatmap.png"

    _plot_heatmap_categorical(
        piv_dom,
        "Regime dominance map: DOM (both mean & p90 improve), BAD (both worsen), TRD (trade-off)",
        p_dom,
    )

    """
    On calcule aussi la part de régimes dans chaque catégorie :
    - DOM : ToxicityAware domine
    - BAD : ToxicityAware est dominée
    - TRD : compromis
    """
    share_dom = float((reg["dominance"] == 1).mean())
    share_bad = float((reg["dominance"] == -1).mean())
    share_trd = float((reg["dominance"] == 0).mean())

    # --- Partie 2 : robustesse sous misspecification
    mis = _load_misspec()

    # On adopte un seuil strict de robustesse
    thr = RobustThresholds(regret_p90=0.0)

    # Tableau de robustesse par valeur de belief threshold
    robust_table = _robust_fraction_by_belief(mis, thr)

    """
    On trace maintenant un bar plot :
    - axe x = tox_trigger_belief
    - axe y = fraction de scénarios robustes
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(
        robust_table["tox_trigger_belief"].astype(float),
        robust_table["robust_frac"].astype(float),
    )
    ax.set_title("Robustness by belief: fraction of scenarios with regret_p90 <= threshold")
    ax.set_xlabel("tox_trigger_belief")
    ax.set_ylabel("robust_frac")
    fig.tight_layout()

    p_rob = FIG_DIR / "misspec_robust_fraction_by_belief.png"
    fig.savefig(p_rob, dpi=180)
    plt.close(fig)

    """
    On génère aussi une heatmap de regret_p90 pour chaque valeur de tox_trigger_belief.
    Cela permet de voir, pour chaque calibration de stratégie,
    comment le regret varie selon les vrais régimes de marché.
    """
    trigger_vals = sorted(mis["tox_trigger_belief"].dropna().unique().tolist())
    fig_links: list[str] = []

    for tt in trigger_vals:
        piv_r = _pivot_misspec(mis, "regret_p90", tox_trigger_value=float(tt))
        out_name = f"misspec_regret_p90_trigger_{float(tt):.2f}.png"
        out_path = FIG_DIR / out_name

        _plot_heatmap_numeric(
            piv_r,
            f"Misspec regret_p90 heatmap — belief tox_trigger={float(tt):.2f}",
            out_path,
        )

        fig_links.append(f"- ![regret_p90](reports/figures/{out_name})")

    # --- Écriture du rapport markdown
    lines: list[str] = []
    lines.append("# Dominance & Robustness Report — Execution under Adverse Selection\n")
    lines.append(
        "This report summarizes **where** ToxicityAware dominates AlwaysMarket "
        "and how robust it is under model misspecification.\n"
    )

    # Section dominance
    lines.append("## Regime dominance\n")
    lines.append(
        "We label each (tox_persist, as_kick_scale) regime as:\n"
        "- **DOM**: delta_mean < 0 AND delta_p90 < 0 (dominates)\n"
        "- **BAD**: delta_mean > 0 AND delta_p90 > 0 (dominated)\n"
        "- **TRD**: trade-off (one improves, the other worsens)\n"
    )
    lines.append(
        f"- Share DOM: {share_dom:.3f}\n"
        f"- Share BAD: {share_bad:.3f}\n"
        f"- Share TRD: {share_trd:.3f}\n"
    )
    lines.append(f"- ![regime_dominance](reports/figures/{p_dom.name})\n")

    # Section robustesse
    lines.append("## Misspecification robustness\n")
    lines.append(
        "We quantify robustness using **regret_p90**. "
        "If regret columns are not present in the CSV, "
        "we use a proxy: regret_p90 = max(delta_p90, 0).\n"
    )
    lines.append(f"- Robustness threshold: regret_p90 <= {thr.regret_p90:.6f}\n")
    lines.append(f"- ![robust_by_belief](reports/figures/{p_rob.name})\n")

    # Heatmaps de regret
    lines.append("### Robustness heatmaps (regret_p90)\n")
    lines.extend(fig_links)

    # Tableau récapitulatif
    lines.append("\n## Robustness table\n")
    lines.append("Columns: tox_trigger_belief, robust_frac, n\n")
    lines.append("```\n")
    lines.append(robust_table.to_string(index=False))
    lines.append("\n```\n")

    # Écriture finale
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Messages finaux
    print(f"Saved figures in: {FIG_DIR.as_posix()}")
    print(f"Saved report: {OUT_MD.as_posix()}")


# Bloc de lancement
if __name__ == "__main__":
    main()