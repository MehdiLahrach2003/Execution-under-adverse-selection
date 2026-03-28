# Misspecification Report — Execution under Adverse Selection

Artifacts generated from `reports/misspec_grid.csv`.

## What this evaluates

We compare a policy calibrated with a toxicity threshold belief (`tox_trigger_belief`) while the true market regime is defined by `tox_persist_true` and `as_kick_scale`.

All deltas are `ToxicityAware - AlwaysMarket` (negative = improvement if Implementation Shortfall is a cost).

## Best / worst (raw deltas)

- Best **mean IS delta**: -0.023009 at (as_kick_scale=0.00, tox_true=0.60, tox_trigger=0.50)
- Worst **mean IS delta**: 0.040032 at (as_kick_scale=0.05, tox_true=0.75, tox_trigger=0.60)

- Best **p90 IS delta**: 0.059079 at (as_kick_scale=0.00, tox_true=0.60, tox_trigger=0.50)
- Worst **p90 IS delta**: 0.122052 at (as_kick_scale=0.05, tox_true=0.95, tox_trigger=0.50)

## Best / worst (risk-adjusted deltas)

We use `delta_*_rel = delta_* / abs(AlwaysMarket_*)` to interpret differences in relative terms.

- Best **mean IS delta (rel)**: -2.745145 at (as_kick_scale=0.00, tox_true=0.60, tox_trigger=0.50)
- Worst **mean IS delta (rel)**: 4.759788 at (as_kick_scale=0.05, tox_true=0.75, tox_trigger=0.60)

- Best **p90 IS delta (rel)**: 0.978846 at (as_kick_scale=0.00, tox_true=0.60, tox_trigger=0.50)
- Worst **p90 IS delta (rel)**: 2.023930 at (as_kick_scale=0.05, tox_true=0.95, tox_trigger=0.50)

## Figures (grouped by belief threshold)

- ![delta_mean](reports/figures/misspec_delta_mean_trigger_0.50.png)
- ![delta_p90](reports/figures/misspec_delta_p90_trigger_0.50.png)
- ![delta_fill_rate](reports/figures/misspec_delta_fillrate_trigger_0.50.png)
- ![delta_mean_rel](reports/figures/misspec_delta_mean_rel_trigger_0.50.png)
- ![delta_p90_rel](reports/figures/misspec_delta_p90_rel_trigger_0.50.png)
- ![delta_mean](reports/figures/misspec_delta_mean_trigger_0.60.png)
- ![delta_p90](reports/figures/misspec_delta_p90_trigger_0.60.png)
- ![delta_fill_rate](reports/figures/misspec_delta_fillrate_trigger_0.60.png)
- ![delta_mean_rel](reports/figures/misspec_delta_mean_rel_trigger_0.60.png)
- ![delta_p90_rel](reports/figures/misspec_delta_p90_rel_trigger_0.60.png)
- ![delta_mean](reports/figures/misspec_delta_mean_trigger_0.70.png)
- ![delta_p90](reports/figures/misspec_delta_p90_trigger_0.70.png)
- ![delta_fill_rate](reports/figures/misspec_delta_fillrate_trigger_0.70.png)
- ![delta_mean_rel](reports/figures/misspec_delta_mean_rel_trigger_0.70.png)
- ![delta_p90_rel](reports/figures/misspec_delta_p90_rel_trigger_0.70.png)
- ![delta_mean](reports/figures/misspec_delta_mean_trigger_0.80.png)
- ![delta_p90](reports/figures/misspec_delta_p90_trigger_0.80.png)
- ![delta_fill_rate](reports/figures/misspec_delta_fillrate_trigger_0.80.png)
- ![delta_mean_rel](reports/figures/misspec_delta_mean_rel_trigger_0.80.png)
- ![delta_p90_rel](reports/figures/misspec_delta_p90_rel_trigger_0.80.png)
