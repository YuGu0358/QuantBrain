"""FactorFactory — deterministic skeleton enumeration for alpha generation.

Provides a library of parameterized WorldQuant BRAIN expression templates.
Each skeleton captures an economic hypothesis; enumerate_skeleton() expands
a template into concrete candidate expressions by filling in field / window
parameter slots.

Usage:
    from alpha_miner.modules.m_factor_factory import enumerate_skeletons

    candidates = enumerate_skeletons(category="MOMENTUM", n=12)
    for c in candidates:
        print(c["expression"], c["hypothesis"])
"""
from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# Field lists — organised by category for cross-category blending
# ---------------------------------------------------------------------------

_PRICE_FIELDS = ["close", "open", "vwap", "adv20"]
_RETURN_FIELDS = ["returns"]
_VOLUME_FIELDS = ["volume", "adv20"]
_FUNDAMENTAL_FIELDS = [
    "assets", "cashflow_op", "operating_income", "sales", "net_income",
    "book_to_price", "earnings_yield", "est_eps",
]
_SENTIMENT_FIELDS = ["analyst_rating", "short_interest"]
_MICRO_FIELDS = ["bid_ask_spread", "amihud", "vwap"]

# Long lookback windows preferred for IS/OOS robustness
_SHORT_WINDOWS = [5, 10]
_MED_WINDOWS = [21, 42]
_LONG_WINDOWS = [63, 126, 252]
_ALL_WINDOWS = _SHORT_WINDOWS + _MED_WINDOWS + _LONG_WINDOWS

_NEUTRALIZATIONS = ["industry", "subindustry"]


# ---------------------------------------------------------------------------
# Skeleton registry
# ---------------------------------------------------------------------------

@dataclass
class Skeleton:
    """A parameterized BRAIN expression template.

    Parameters:
        name:       unique identifier
        category:   alpha factor category (MOMENTUM, VALUE, QUALITY, …)
        template:   expression string with {field}, {field2}, {window}, etc.
        hypothesis: economic rationale (1-2 sentences)
        params:     dict mapping placeholder → list of allowed values
        robustness_note: short note on IS/OOS stability
    """
    name: str
    category: str
    template: str
    hypothesis: str
    params: dict[str, list]
    robustness_note: str = ""


_SKELETONS: list[Skeleton] = [
    # ── MOMENTUM ─────────────────────────────────────────────────────────
    Skeleton(
        name="momentum_ts_rank_neutralized",
        category="MOMENTUM",
        template="group_rank(ts_rank({field}, {window}), {neut})",
        hypothesis=(
            "Stocks sustaining top relative performance over a {window}-day window "
            "tend to continue outperforming within their industry group. "
            "group_rank sector-neutralizes macro trends, isolating pure momentum."
        ),
        params={
            "field": _RETURN_FIELDS + ["close"],
            "window": [42, 63, 126],
            "neut": _NEUTRALIZATIONS,
        },
        robustness_note="Long window reduces IS overfitting; group_rank neutralizes regime changes.",
    ),
    Skeleton(
        name="momentum_smooth_delta",
        category="MOMENTUM",
        template="group_rank(ts_mean(ts_delta({field}, {d_short}), {d_smooth}), {neut})",
        hypothesis=(
            "Short-term price change smoothed over a medium window captures "
            "persistent momentum without same-day noise. group_rank removes "
            "cross-sector drift."
        ),
        params={
            "field": ["close", "vwap"],
            "d_short": [3, 5],
            "d_smooth": [10, 21],
            "neut": _NEUTRALIZATIONS,
        },
        robustness_note="Smoothing wrapper (ts_mean) is essential for OOS stability.",
    ),
    Skeleton(
        name="momentum_52w_high_proximity",
        category="MOMENTUM",
        template=(
            "group_rank(divide(ts_mean({field}, {d_short}), "
            "ts_max({field}, {d_long})), {neut})"
        ),
        hypothesis=(
            "Proximity to the {d_long}-day high is a well-documented momentum "
            "predictor. Normalizing by current price smoothed over {d_short} days "
            "reduces microstructure noise. sector-neutral version strips macro moves."
        ),
        params={
            "field": ["close", "vwap"],
            "d_short": [5, 10],
            "d_long": [126, 252],
            "neut": _NEUTRALIZATIONS,
        },
        robustness_note="52-week-high effect is persistent across IS/OOS studies.",
    ),

    # ── VALUE ─────────────────────────────────────────────────────────────
    Skeleton(
        name="value_fundamental_rank_neutralized",
        category="VALUE",
        template="group_rank(ts_rank({fund_field}, {window}), {neut})",
        hypothesis=(
            "Ranking stocks by {fund_field} over {window} days within sector "
            "identifies attractively priced names relative to industry peers. "
            "Quarterly-updating fundamental data is naturally slow-moving and robust."
        ),
        params={
            "fund_field": ["cashflow_op", "book_to_price", "earnings_yield", "est_eps"],
            "window": [63, 126],
            "neut": _NEUTRALIZATIONS,
        },
        robustness_note="Quarterly fundamentals have high IS/OOS consistency.",
    ),
    Skeleton(
        name="value_price_to_fundamental_ratio",
        category="VALUE",
        template=(
            "group_rank(divide(ts_mean({price_field}, {d_price}), "
            "ts_mean({fund_field}, {d_fund})), {neut})"
        ),
        hypothesis=(
            "The ratio of smoothed price to smoothed {fund_field} captures "
            "relative valuation within sector. Smoothing both numerator and "
            "denominator reduces noise while preserving the cross-sectional signal."
        ),
        params={
            "price_field": ["close", "vwap"],
            "fund_field": ["cashflow_op", "assets", "operating_income"],
            "d_price": [21],
            "d_fund": [63, 126],
            "neut": _NEUTRALIZATIONS,
        },
        robustness_note="Dual smoothing stabilizes the ratio; long fundamental window prevents quarterly noise.",
    ),

    # ── QUALITY ───────────────────────────────────────────────────────────
    Skeleton(
        name="quality_earnings_stability",
        category="QUALITY",
        template=(
            "group_rank(divide(ts_mean({fund_field}, {d_long}), "
            "ts_std_dev({fund_field}, {d_long})), {neut})"
        ),
        hypothesis=(
            "The coefficient of variation (mean/std) of {fund_field} measures "
            "earnings consistency. High-quality firms with stable fundamentals "
            "carry less risk and often generate persistent alpha."
        ),
        params={
            "fund_field": ["cashflow_op", "operating_income", "net_income"],
            "d_long": [126, 252],
            "neut": _NEUTRALIZATIONS,
        },
        robustness_note="Stability metric is mean-reverting; long window reduces quarterly seasonality.",
    ),
    Skeleton(
        name="quality_accrual_signal",
        category="QUALITY",
        template=(
            "group_rank(ts_delta({fund_accrual}, {d_delta}), {neut})"
        ),
        hypothesis=(
            "Rising accruals relative to peers predict earnings quality deterioration "
            "and subsequent underperformance. Sector-neutral version removes "
            "industry-wide accounting changes."
        ),
        params={
            "fund_accrual": ["assets", "net_income", "sales"],
            "d_delta": [63, 126],
            "neut": _NEUTRALIZATIONS,
        },
        robustness_note="Accrual anomaly is among the most replicated accounting-based effects.",
    ),

    # ── REVERSAL ──────────────────────────────────────────────────────────
    Skeleton(
        name="reversal_short_term_smooth",
        category="REVERSAL",
        template=(
            "group_rank(ts_zscore({field}, {d_zscore}), {neut})"
        ),
        hypothesis=(
            "Z-score standardizes recent returns, flagging over-extended moves. "
            "Sector neutralization ensures the reversion signal is idiosyncratic, "
            "not driven by sector-wide selloffs. {d_zscore}-day window balances "
            "speed and noise."
        ),
        params={
            "field": ["returns", "close"],
            "d_zscore": [21, 42],
            "neut": _NEUTRALIZATIONS,
        },
        robustness_note="Z-score reversal works best at 21–42 day windows; shorter windows overfit to noise.",
    ),
    Skeleton(
        name="reversal_volume_adjusted",
        category="REVERSAL",
        template=(
            "group_rank(divide(ts_delta({price}, {d_short}), "
            "ts_mean({vol}, {d_vol})), {neut})"
        ),
        hypothesis=(
            "Price change divided by smoothed volume isolates price moves "
            "not supported by trading activity — these tend to revert. "
            "sector-neutral version removes macro-driven reversals."
        ),
        params={
            "price": ["close", "vwap"],
            "vol": ["volume", "adv20"],
            "d_short": [5, 10],
            "d_vol": [21, 42],
            "neut": _NEUTRALIZATIONS,
        },
        robustness_note="Volume-adjusted reversal has lower turnover than pure price reversal.",
    ),

    # ── GROWTH ────────────────────────────────────────────────────────────
    Skeleton(
        name="growth_fundamental_acceleration",
        category="GROWTH",
        template=(
            "group_rank(ts_delta(ts_mean({fund_field}, {d_short}), {d_long}), {neut})"
        ),
        hypothesis=(
            "Acceleration in {fund_field} — the change in recent trend compared to "
            "the long-run average — predicts earnings surprise and analyst revision. "
            "Sector-neutral version isolates stock-specific growth acceleration."
        ),
        params={
            "fund_field": ["est_eps", "operating_income", "cashflow_op", "sales"],
            "d_short": [21, 42],
            "d_long": [63, 126],
            "neut": _NEUTRALIZATIONS,
        },
        robustness_note="Fundamental acceleration is a leading indicator; long outer window stabilizes OOS.",
    ),

    # ── VOLATILITY (low-vol anomaly) ──────────────────────────────────────
    Skeleton(
        name="low_vol_anomaly",
        category="QUALITY",
        template=(
            "group_rank(multiply(-1, ts_std_dev({field}, {window})), {neut})"
        ),
        hypothesis=(
            "The low-volatility anomaly: stocks with lower realized volatility "
            "over {window} days outperform within sector, contradicting CAPM. "
            "Inverting std_dev ranks least-volatile stocks highest."
        ),
        params={
            "field": ["returns", "close"],
            "window": [63, 126],
            "neut": _NEUTRALIZATIONS,
        },
        robustness_note="Low-vol anomaly persists cross-sectionally even in degraded data environments.",
    ),

    # ── MULTI-SIGNAL BLENDS ───────────────────────────────────────────────
    Skeleton(
        name="blend_momentum_value",
        category="MOMENTUM",
        template=(
            "group_rank(ts_mean({price_field}, {d_mom}) + "
            "ts_rank({fund_field}, {d_val}), {neut})"
        ),
        hypothesis=(
            "Blending momentum (smoothed {price_field}) with value ({fund_field} rank) "
            "reduces correlation to pure momentum, improving OOS stability. "
            "The two signals are empirically uncorrelated within sectors."
        ),
        params={
            "price_field": ["returns", "close"],
            "fund_field": ["book_to_price", "earnings_yield", "cashflow_op"],
            "d_mom": [42, 63],
            "d_val": [126],
            "neut": _NEUTRALIZATIONS,
        },
        robustness_note="Uncorrelated signal blend is the key IS/OOS diversifier.",
    ),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_skeletons(category: str | None = None) -> list[Skeleton]:
    """Return all skeletons, optionally filtered by category."""
    if category is None:
        return list(_SKELETONS)
    return [s for s in _SKELETONS if s.category.upper() == category.upper()]


def enumerate_skeleton(skeleton: Skeleton, max_per_skeleton: int = 4) -> list[dict[str, Any]]:
    """Expand a skeleton into concrete candidate dicts by filling parameter slots.

    Returns up to `max_per_skeleton` candidates.  When the parameter space has
    more combinations than the limit, samples without replacement for diversity.
    """
    keys = list(skeleton.params.keys())
    value_lists = [skeleton.params[k] for k in keys]
    all_combos = list(itertools.product(*value_lists))

    if len(all_combos) > max_per_skeleton:
        sample = random.sample(all_combos, max_per_skeleton)
    else:
        sample = all_combos

    candidates = []
    for combo in sample:
        param_map = dict(zip(keys, combo))
        try:
            expression = skeleton.template.format(**param_map)
        except KeyError:
            continue
        hypothesis = skeleton.hypothesis.format(**{
            k: v for k, v in param_map.items()
            if f"{{{k}}}" in skeleton.hypothesis
        })
        candidates.append({
            "id": f"factory_{skeleton.name}_{'_'.join(str(v) for v in combo)}",
            "category": skeleton.category,
            "expression": expression,
            "hypothesis": hypothesis,
            "origin_refs": ["factor_factory", skeleton.name],
            "opt_rounds": 0,
        })
    return candidates


def enumerate_skeletons(
    category: str | None = None,
    n: int = 10,
    max_per_skeleton: int = 4,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Generate up to `n` candidate expressions from skeletons.

    Args:
        category:         Filter skeletons to this category (None = all).
        n:                Maximum total candidates to return.
        max_per_skeleton: Maximum candidates per skeleton before sampling.
        seed:             Optional random seed for reproducibility.

    Returns:
        List of candidate dicts with expression, hypothesis, category, origin_refs.
    """
    if seed is not None:
        random.seed(seed)

    skeletons = get_skeletons(category)
    if not skeletons:
        return []

    all_candidates: list[dict[str, Any]] = []
    for sk in skeletons:
        all_candidates.extend(enumerate_skeleton(sk, max_per_skeleton=max_per_skeleton))

    if len(all_candidates) > n:
        all_candidates = random.sample(all_candidates, n)

    return all_candidates


__all__ = [
    "Skeleton",
    "get_skeletons",
    "enumerate_skeleton",
    "enumerate_skeletons",
]
