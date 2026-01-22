# Regime Report — Execution under Adverse Selection

Artifacts generated from `reports/regime_grid.csv`.

## What this evaluates

We sweep the *true* market regime parameters:
- `as_kick_scale` (adverse-selection kick intensity)
- `tox_persist` (toxicity persistence)

We compare `ToxicityAware` vs `AlwaysMarket`.
All deltas are `ToxicityAware - AlwaysMarket` (negative = improvement if IS is a cost).

## Key takeaways (raw deltas)

- Best **mean IS delta**: 0.004490 at (as_kick_scale=0.00, tox_persist=0.95)
- Worst **mean IS delta**: 0.040032 at (as_kick_scale=0.05, tox_persist=0.75)

- Best **p90 IS delta**: 0.063739 at (as_kick_scale=0.00, tox_persist=0.85)
- Worst **p90 IS delta**: 0.097220 at (as_kick_scale=0.05, tox_persist=0.60)

## Key takeaways (risk-adjusted deltas)

We report relative deltas to normalize by baseline magnitude:

- `delta_mean_rel = delta_mean / abs(AlwaysMarket_mean)`

- `delta_p90_rel  = delta_p90 / abs(AlwaysMarket_p90)`

- Best **mean IS delta (rel)**: 0.531379 at (as_kick_scale=0.00, tox_persist=0.95)
- Worst **mean IS delta (rel)**: 4.759788 at (as_kick_scale=0.05, tox_persist=0.75)

- Best **p90 IS delta (rel)**: 1.056701 at (as_kick_scale=0.00, tox_persist=0.85)
- Worst **p90 IS delta (rel)**: 1.610787 at (as_kick_scale=0.05, tox_persist=0.60)

## Execution quality metrics

- Best **delta_fill_rate**: 0.000000 at (as_kick_scale=0.00, tox_persist=0.60)
- Worst **delta_fill_rate**: 0.000000 at (as_kick_scale=0.00, tox_persist=0.60)

We also track `avg_first_fill_t` (average time-to-first-fill). A negative `delta_avg_first_fill_t` indicates ToxicityAware fills earlier.

- Best **delta_avg_first_fill_t**: 0.000000 at (as_kick_scale=0.00, tox_persist=0.60)
- Worst **delta_avg_first_fill_t**: 0.020000 at (as_kick_scale=0.00, tox_persist=0.85)

## Figures

- ![delta_mean](reports/figures/regime_delta_mean_heatmap.png)
- ![delta_p90](reports/figures/regime_delta_p90_heatmap.png)
- ![delta_fill_rate](reports/figures/regime_delta_fillrate_delta_heatmap.png)
- ![delta_avg_first_fill_t](reports/figures/regime_delta_avg_first_fill_t_heatmap.png)
- ![delta_mean_rel](reports/figures/regime_delta_mean_rel_heatmap.png)
- ![delta_p90_rel](reports/figures/regime_delta_p90_rel_heatmap.png)
