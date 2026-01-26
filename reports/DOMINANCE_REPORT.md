# Dominance & Robustness Report — Execution under Adverse Selection

This report summarizes **where** ToxicityAware dominates AlwaysMarket and how robust it is under model misspecification.

## Regime dominance

We label each (tox_persist, as_kick_scale) regime as:
- **DOM**: delta_mean < 0 AND delta_p90 < 0 (dominates)
- **BAD**: delta_mean > 0 AND delta_p90 > 0 (dominated)
- **TRD**: trade-off (one improves, the other worsens)

- Share DOM: 0.000
- Share BAD: 1.000
- Share TRD: 0.000

- ![regime_dominance](reports/figures/regime_dominance_heatmap.png)

## Misspecification robustness

We quantify robustness using **regret_p90**. If regret columns are not present in the CSV, we use a proxy: regret_p90 = max(delta_p90, 0).

- Robustness threshold: regret_p90 <= 0.000000

- ![robust_by_belief](reports/figures/misspec_robust_fraction_by_belief.png)

### Robustness heatmaps (regret_p90)

- ![regret_p90](reports/figures/misspec_regret_p90_modeltox_0.50.png)
- ![regret_p90](reports/figures/misspec_regret_p90_modeltox_0.60.png)
- ![regret_p90](reports/figures/misspec_regret_p90_modeltox_0.70.png)
- ![regret_p90](reports/figures/misspec_regret_p90_modeltox_0.80.png)

## Robustness table

Columns: tox_persist_model, robust_frac, n

```

 tox_persist_model  robust_frac  n
               0.5          0.0 20
               0.6          0.0 20
               0.7          0.0 20
               0.8          0.0 20

```

