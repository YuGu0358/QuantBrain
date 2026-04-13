# Phase 0 BRAIN API Probe Report

- Generated at: 2026-04-12T21:42:28Z
- Decision: Case C
- Detail: BRAIN appears to ignore submitted date splits and no daily PnL series was found for the regular-tier probe.

## Evidence Summary

- The IS probe requested `startDate=2015-01-01` and `endDate=2021-12-31`.
- The OOS probe requested `startDate=2023-01-01` and `endDate=2025-12-31`.
- The returned alpha settings normalized both runs to the platform fixed window `startDate=2019-01-01` and `endDate=2023-12-31`.
- The simulations completed, but no daily numeric PnL series was found in `/simulations/{id}`, `/alphas/{alpha_id}`, `/simulations/{id}/details`, or `/simulations/{id}/pnl`.
- `/simulations/{id}/details` and `/simulations/{id}/pnl` returned unavailable responses during the live probe.
- Aggregate fields such as `alpha.is.pnl`, `alpha.is.returns`, `alpha.is.sharpe`, `alpha.is.fitness`, and `alpha.is.turnover` are not daily series and must not be treated as PnL vectors.

## Probe IDs

- IS simulation: `3cdbEFc1W4wTaNY11neYzRGo`
- OOS simulation: `3CvHsN3sh5cAcrqB2LKXIng`
- Returned alpha: `om92mNm2`

## Follow-up Policy

- Case C is treated as regular-tier degraded mode: do not fabricate daily PnL and do not run DSR/MVO from aggregate values.
- Use BRAIN `/alphas/{id}/check` as the platform-native self-correlation proxy when an `alpha_id` exists.
- Use expression and aggregate-metric similarity only as weak pre-filters, not as replacements for PnL Pearson.
- Optional follow-up probes are now implemented behind `PHASE0_VISUALIZATION_PROBE=1` and `PHASE0_SPECULATIVE_ENDPOINT_PROBE=1`.

