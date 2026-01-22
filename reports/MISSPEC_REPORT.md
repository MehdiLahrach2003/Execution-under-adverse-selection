# Misspecification Report — Execution under Adverse Selection

Artifacts generated from `reports/misspec_grid.csv`.

## What this evaluates

We compare a policy that uses a *belief* about market toxicity persistence (`tox_persist_model`) 
while the *true* market regime is `tox_persist_true` and `as_kick_scale`.

All deltas are `ToxicityAware - AlwaysMarket` (negative = improvement if IS is a cost).

## Best / worst (raw deltas)

- Best **mean IS delta**: 0.016058 at (as_kick_scale=0.00, tox_true=0.95, tox_model=0.60)
- Worst **mean IS delta**: 0.050303 at (as_kick_scale=0.05, tox_true=0.60, tox_model=0.60)

- Best **p90 IS delta**: 0.062991 at (as_kick_scale=0.00, tox_true=0.95, tox_model=0.60)
- Worst **p90 IS delta**: 0.102520 at (as_kick_scale=0.05, tox_true=0.75, tox_model=0.60)

## Best / worst (risk-adjusted deltas)

We use `delta_*_rel = delta_* / abs(AlwaysMarket_*)` to interpret improvements in relative terms.

- Best **mean IS delta (rel)**: 1.121729 at (as_kick_scale=0.00, tox_true=0.95, tox_model=0.60)
- Worst **mean IS delta (rel)**: 3.572014 at (as_kick_scale=0.05, tox_true=0.60, tox_model=0.60)

- Best **p90 IS delta (rel)**: 0.884829 at (as_kick_scale=0.00, tox_true=0.95, tox_model=0.60)
- Worst **p90 IS delta (rel)**: 1.438621 at (as_kick_scale=0.05, tox_true=0.75, tox_model=0.60)

## Figures (grouped by model belief)

- ![delta_mean](reports/figures/misspec_delta_mean_modeltox_0.60.png)
- ![delta_p90](reports/figures/misspec_delta_p90_modeltox_0.60.png)
- ![delta_fill_rate](reports/figures/misspec_delta_fillrate_modeltox_0.60.png)
- ![delta_mean_rel](reports/figures/misspec_delta_mean_rel_modeltox_0.60.png)
- ![delta_p90_rel](reports/figures/misspec_delta_p90_rel_modeltox_0.60.png)
- ![delta_mean](reports/figures/misspec_delta_mean_modeltox_0.75.png)
- ![delta_p90](reports/figures/misspec_delta_p90_modeltox_0.75.png)
- ![delta_fill_rate](reports/figures/misspec_delta_fillrate_modeltox_0.75.png)
- ![delta_mean_rel](reports/figures/misspec_delta_mean_rel_modeltox_0.75.png)
- ![delta_p90_rel](reports/figures/misspec_delta_p90_rel_modeltox_0.75.png)
- ![delta_mean](reports/figures/misspec_delta_mean_modeltox_0.85.png)
- ![delta_p90](reports/figures/misspec_delta_p90_modeltox_0.85.png)
- ![delta_fill_rate](reports/figures/misspec_delta_fillrate_modeltox_0.85.png)
- ![delta_mean_rel](reports/figures/misspec_delta_mean_rel_modeltox_0.85.png)
- ![delta_p90_rel](reports/figures/misspec_delta_p90_rel_modeltox_0.85.png)
- ![delta_mean](reports/figures/misspec_delta_mean_modeltox_0.95.png)
- ![delta_p90](reports/figures/misspec_delta_p90_modeltox_0.95.png)
- ![delta_fill_rate](reports/figures/misspec_delta_fillrate_modeltox_0.95.png)
- ![delta_mean_rel](reports/figures/misspec_delta_mean_rel_modeltox_0.95.png)
- ![delta_p90_rel](reports/figures/misspec_delta_p90_rel_modeltox_0.95.png)
