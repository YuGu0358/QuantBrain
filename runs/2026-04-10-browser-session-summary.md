# Browser Session Summary

## Batch 1

### operating_income_scaled_by_price

- Simulation: `3z4NqDcE75aobDOPoilNXL4`
- Alpha: `wpJqVEpl`
- Expression: `ts_rank(operating_income / close, 252)`
- Settings: `USA / TOP3000 / d1 / decay 0 / truncation 0.08 / neutralization SUBINDUSTRY`
- IS: Sharpe `1.51`, Fitness `1.17`, Turnover `0.1305`, Returns `0.0782`, Drawdown `0.0686`
- Train: Sharpe `1.83`, Fitness `1.59`
- Test: Sharpe `-0.26`, Fitness `-0.07`
- Checks: passes Sharpe, Fitness, Turnover, Concentration, Sub-universe; Self-correlation pending
- Read: strongest first-round candidate, but OOS instability is the main issue

### balance_sheet_quality

- Simulation: `1pR50r72K5agalllOMSUOyc`
- Alpha: `A1O9dP5X`
- Expression: `-rank(liabilities / assets)`
- Settings: `USA / TOP3000 / d1 / decay 0 / truncation 0.08 / neutralization SUBINDUSTRY`
- IS: Sharpe `-1.51`, Fitness `-1.26`, Turnover `0.0166`, Returns `-0.0870`, Drawdown `0.4605`
- Test: Sharpe `0.27`, Fitness `0.08`
- Checks: fails Sharpe, Fitness, Sub-universe
- Read: direction is likely wrong or too naive

### industry_profitability_efficiency

- Simulation: `3LaSv8jf56BbrrSWeUUJUq`
- Alpha: `RRkMLmWo`
- Expression: `group_rank(ts_rank(operating_income / assets, 252), industry)`
- Settings: `USA / TOP3000 / d1 / decay 0 / truncation 0.08 / neutralization SUBINDUSTRY`
- IS: Sharpe `0.99`, Fitness `0.54`, Turnover `0.0602`, Returns `0.0373`, Drawdown `0.0465`
- Test: Sharpe `0.19`, Fitness `0.04`
- Checks: fails Sharpe and Fitness, passes Turnover, Concentration, Sub-universe
- Read: more robust than the balance-sheet candidate, but not strong enough

## Batch 2 Submitted

- `peer_relative_income_price`
  - Simulation: `4zQU1iYB4ETbwXCe5NF1MA`
  - Expression: `group_rank(ts_rank(operating_income / close, 252), industry)`
  - Settings: `neutralization INDUSTRY`
  - Alpha: `P0XNe0Yw`
  - IS: Sharpe `1.58`, Fitness `1.16`, Turnover `0.1585`, Returns `0.0859`, Drawdown `0.0666`
  - Test: Sharpe `-0.14`, Fitness `-0.02`
  - Read: best IS so far, OOS still negative but less bad than the base `operating_income / close`
- `income_price_decay4`
  - Simulation: `2XU10Xf1T4PR9mTYK1oSahR`
  - Expression: `ts_rank(operating_income / close, 252)`
  - Settings: `decay 4`
  - Alpha: `mLxpk1vp`
  - IS: Sharpe `1.36`, Fitness `1.03`, Turnover `0.0795`, Returns `0.0716`, Drawdown `0.0724`
  - Test: Sharpe `-0.29`, Fitness `-0.09`
  - Read: smoothing reduced turnover but did not improve robustness
- `income_asset_trend`
  - Simulation: `3mZwEF4H54hHb8AkmDO0yPz`
  - Expression: `ts_rank(operating_income / assets, 252)`
  - Settings: baseline
  - Alpha: `78KWVrb1`
  - IS: Sharpe `0.92`, Fitness `0.50`, Turnover `0.0488`, Returns `0.0365`, Drawdown `0.0545`
  - Test: Sharpe `0.29`, Fitness `0.09`
  - Read: OOS sign is positive, but the signal is too weak in-sample

## Batch 3 Submitted

- `income_dual_scale_hybrid`
  - Simulation: `52eYUaIq4AJc83wLX4gyAE`
  - Expression: `rank(ts_rank(operating_income / close, 252) + ts_rank(operating_income / assets, 252))`
  - Status: pending at `35%` when last checked
