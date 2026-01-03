# Regime Report — Execution under Adverse Selection

Artifacts generated from `reports/regime_grid.csv`.

## Key takeaways (grid)

We sweep `as_kick_scale` (adverse-selection kick intensity) and `tox_persist` (toxicity persistence).
All deltas are `ToxicityAware - AlwaysMarket` (negative = improvement if IS is a cost).

- Best **mean IS delta**: 0.004490 at (as_kick_scale=0.00, tox_persist=0.95)
- Worst **mean IS delta**: 0.040032 at (as_kick_scale=0.05, tox_persist=0.75)

- Best **p90 IS delta**: 0.063739 at (as_kick_scale=0.00, tox_persist=0.85)
- Worst **p90 IS delta**: 0.097220 at (as_kick_scale=0.05, tox_persist=0.60)

- Best **fill-rate delta**: 0.000 at (as_kick_scale=0.00, tox_persist=0.60)
- Worst **fill-rate delta**: 0.000 at (as_kick_scale=0.00, tox_persist=0.60)


## Figures

- ![delta_mean](reports/figures/regime_delta_mean_heatmap.png)
- ![delta_p90](reports/figures/regime_delta_p90_heatmap.png)
- ![delta_fill_rate](reports/figures/regime_fillrate_delta_heatmap.png)
