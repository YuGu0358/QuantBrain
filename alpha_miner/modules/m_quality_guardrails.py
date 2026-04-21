from __future__ import annotations

import re
from dataclasses import dataclass

from .m4_brain_backtester import BacktestResult


_FUNCTION_STYLE_DIVIDE = re.compile(r"\bdivide\s*\(", re.IGNORECASE)
_SMOOTH_RATIO = re.compile(
    r"ts_mean\(\s*(?P<left>[a-z_][a-z0-9_]*)\s*,\s*\d+\s*\)\s*/\s*ts_mean\(\s*(?P<right>[a-z_][a-z0-9_]*)\s*,\s*\d+\s*\)",
    re.IGNORECASE,
)
_FUND_STABILITY_RATIO = re.compile(
    r"ts_mean\(\s*(?P<field>cashflow_op|operating_income|est_eps)\s*,\s*\d+\s*\)\s*/\s*ts_std_dev\(\s*(?P=field)\s*,\s*\d+\s*\)",
    re.IGNORECASE,
)

_PRICE_FIELDS = {"close", "open", "high", "low", "vwap", "returns"}
_VOLUME_FIELDS = {"volume", "adv20"}
_FUNDAMENTAL_FIELDS = {"cashflow_op", "operating_income", "est_eps", "assets"}


@dataclass(frozen=True)
class EconomicPrescreenResult:
    is_valid: bool
    reasons: list[str]
    warnings: list[str]


def economic_logic_prescreen(expression: str) -> EconomicPrescreenResult:
    expression = str(expression or "").strip()
    reasons: list[str] = []
    warnings: list[str] = []

    if _FUNCTION_STYLE_DIVIDE.search(expression):
        reasons.append("function_style_divide_unverified")

    for match in _SMOOTH_RATIO.finditer(expression):
        left = match.group("left").lower()
        right = match.group("right").lower()
        if _is_price_volume_pair(left, right):
            reasons.append("price_volume_ratio_unit_mismatch")
        elif _is_price_fundamental_pair(left, right):
            reasons.append("price_fundamental_ratio_unit_mismatch")

    if _FUND_STABILITY_RATIO.search(expression):
        warnings.append("fundamental_mean_std_ratio_is_stepwise_and_fragile")

    return EconomicPrescreenResult(
        is_valid=not reasons,
        reasons=_unique(reasons),
        warnings=_unique(warnings),
    )


def should_try_sign_flip(result: BacktestResult, sharpe_threshold: float = -0.20) -> bool:
    return bool(
        result.status == "completed"
        and result.has_daily_pnl
        and result.pnl_path
        and result.sharpe is not None
        and result.sharpe <= sharpe_threshold
    )


def sign_flip_expression(expression: str) -> str:
    return f"multiply(-1, {str(expression or '').strip()})"


def _is_price_volume_pair(left: str, right: str) -> bool:
    return (left in _PRICE_FIELDS and right in _VOLUME_FIELDS) or (left in _VOLUME_FIELDS and right in _PRICE_FIELDS)


def _is_price_fundamental_pair(left: str, right: str) -> bool:
    return (left in _PRICE_FIELDS and right in _FUNDAMENTAL_FIELDS) or (left in _FUNDAMENTAL_FIELDS and right in _PRICE_FIELDS)


def _unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered
