from __future__ import annotations

import re
from typing import Any


_IDENTIFIER = re.compile(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b")
_FUNCTION = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(")
_RESERVED = {"if", "else", "true", "false", "nan"}
_GROUP_FIELDS = {"industry", "subindustry", "sector", "market", "country"}
_WINDOW_OPERATORS = {
    "correlation",
    "covariance",
    "delay",
    "ts_arg_max",
    "ts_arg_min",
    "ts_backfill",
    "ts_corr",
    "ts_covariance",
    "ts_decay_linear",
    "ts_delta",
    "ts_mean",
    "ts_rank",
    "ts_std_dev",
    "ts_sum",
    "ts_zscore",
}
_SMOOTHING_OPERATORS = {"hump", "ts_decay_linear", "ts_mean", "winsorize"}
_TS_OPERATORS = {operator for operator in _WINDOW_OPERATORS if operator.startswith("ts_")}
_QUALITY_FIELDS = {"assets", "cashflow_op", "operating_income", "est_eps"}
_LIQUIDITY_FIELDS = {"adv20", "volume"}
_PRICE_FIELDS = {"close", "high", "low", "open", "returns", "vwap"}
_SENTIMENT_FIELDS = {"news_sentiment"}
_RISK_FIELDS = {"beta", "volatility", "sigma"}
_THEMES = {"momentum", "reversal", "liquidity", "quality", "fundamental", "sentiment", "risk"}


def analyze_math_profile(expression: str) -> dict[str, Any]:
    expression = str(expression or "").strip()
    operators = _unique_sorted(_FUNCTION.findall(expression))
    group_fields = _extract_group_fields(expression)
    fields = _extract_fields(expression, operators, group_fields)
    windows = _extract_windows(expression, operators)
    complexity = len(operators) + expression.count("+") + expression.count("-") + expression.count("*") + expression.count("/")
    has_group = bool(group_fields)
    has_ts = any(operator in _TS_OPERATORS or operator in _WINDOW_OPERATORS for operator in operators)
    has_cross = any(operator in {"rank", "zscore", "group_rank", "group_neutralize"} for operator in operators)
    if has_group and has_ts:
        dominant = "hybrid_group_time_series"
    elif has_group:
        dominant = "cross_sectional_group"
    elif has_ts:
        dominant = "time_series"
    elif has_cross:
        dominant = "cross_sectional"
    elif operators:
        dominant = "operator_transform"
    elif expression:
        dominant = "arithmetic"
    else:
        dominant = "empty"

    return {
        "operators": operators,
        "fields": fields,
        "windows": windows,
        "group_fields": group_fields,
        "has_group_neutralization": has_group,
        "complexity": complexity,
        "dominant_structure": dominant,
        "family_signature": _family_signature(operators, fields, windows, group_fields, dominant),
    }


def infer_economic_profile(
    expression: str,
    category: str | None = None,
    fields: list[str] | None = None,
) -> dict[str, Any]:
    math_profile = analyze_math_profile(expression)
    field_set = set(fields or math_profile.get("fields") or [])
    category_theme = _normalize_category(category)
    if category_theme:
        theme = category_theme
    elif field_set & _SENTIMENT_FIELDS:
        theme = "sentiment"
    elif field_set & _LIQUIDITY_FIELDS:
        theme = "liquidity"
    elif field_set & _QUALITY_FIELDS:
        theme = "quality"
    elif field_set & _RISK_FIELDS:
        theme = "risk"
    elif "returns" in field_set and re.search(r"(^|[^a-zA-Z_])-+\s*returns\b", str(expression or "")):
        theme = "reversal"
    elif field_set & _PRICE_FIELDS:
        theme = "momentum"
    else:
        theme = "unknown"

    return {
        "theme": theme,
        "fields": sorted(field_set),
        "thesis": _thesis_for_theme(theme, math_profile),
    }


def compare_repair(parent_expression: str, candidate_expression: str) -> dict[str, Any]:
    parent_math = analyze_math_profile(parent_expression)
    candidate_math = analyze_math_profile(candidate_expression)
    parent_econ = infer_economic_profile(parent_expression, fields=parent_math.get("fields", []))
    candidate_econ = infer_economic_profile(candidate_expression, fields=candidate_math.get("fields", []))

    parent_fields = set(parent_math.get("fields") or [])
    candidate_fields = set(candidate_math.get("fields") or [])
    parent_windows = set(parent_math.get("windows") or [])
    candidate_windows = set(candidate_math.get("windows") or [])
    parent_groups = set(parent_math.get("group_fields") or [])
    candidate_groups = set(candidate_math.get("group_fields") or [])
    parent_ops = set(parent_math.get("operators") or [])
    candidate_ops = set(candidate_math.get("operators") or [])

    field_shift = bool(parent_fields != candidate_fields)
    group_change = bool(parent_groups != candidate_groups)
    smoothing = bool((candidate_ops & _SMOOTHING_OPERATORS) - (parent_ops & _SMOOTHING_OPERATORS))
    if not smoothing and candidate_windows and parent_windows:
        smoothing = max(candidate_windows) > max(parent_windows)
    horizon_change = bool(parent_windows != candidate_windows)
    quality_anchor = bool(candidate_fields & _QUALITY_FIELDS)
    cross_family_escape = (
        parent_econ["theme"] != candidate_econ["theme"]
        or (
            field_shift
            and parent_math.get("dominant_structure") != candidate_math.get("dominant_structure")
        )
    )
    thesis_preserved = parent_econ["theme"] == candidate_econ["theme"] and not cross_family_escape

    actions = [
        key
        for key, enabled in {
            "smoothing": smoothing,
            "horizon_change": horizon_change,
            "field_shift": field_shift,
            "group_change": group_change,
            "quality_anchor": quality_anchor,
            "cross_family_escape": cross_family_escape,
            "thesis_preserved": thesis_preserved,
        }.items()
        if enabled
    ]
    return {
        "smoothing": smoothing,
        "horizon_change": horizon_change,
        "field_shift": field_shift,
        "group_change": group_change,
        "quality_anchor": quality_anchor,
        "cross_family_escape": cross_family_escape,
        "thesis_preserved": thesis_preserved,
        "actions": actions,
        "parent_theme": parent_econ["theme"],
        "candidate_theme": candidate_econ["theme"],
        "parent_signature": parent_math["family_signature"],
        "candidate_signature": candidate_math["family_signature"],
    }


def score_repair_outcome(
    metrics: dict[str, Any] | None = None,
    gate: dict[str, Any] | None = None,
    accepted: bool = False,
    platform_outcome: dict[str, Any] | None = None,
) -> float:
    metrics = metrics or {}
    platform_outcome = platform_outcome or {}
    score = 1.0 if accepted else -0.2
    sharpe = _float_or_none(metrics.get("sharpe") or metrics.get("isSharpe") or metrics.get("testSharpe"))
    fitness = _float_or_none(metrics.get("fitness") or metrics.get("isFitness") or metrics.get("testFitness"))
    turnover = _float_or_none(metrics.get("turnover"))
    if sharpe is not None:
        score += max(min(sharpe, 3.0), -3.0) * 0.35
    if fitness is not None:
        score += max(min(fitness, 2.0), -2.0) * 0.25
    if turnover is not None:
        if 0.01 <= turnover <= 0.7:
            score += 0.2
        elif turnover > 0.7:
            score -= min((turnover - 0.7) * 0.8, 0.5)
        else:
            score -= 0.1
    failed_checks = [
        check for check in (gate or {}).get("checks", [])
        if isinstance(check, dict) and check.get("result") != "PASS"
    ]
    score -= min(len(failed_checks) * 0.05, 0.3)
    outcome = str(platform_outcome.get("outcome") or platform_outcome.get("status") or "").lower()
    if outcome in {"submitted", "target-met", "accepted"}:
        score += 0.5
    elif outcome in {"no_daily_pnl", "degraded_no_daily_pnl"}:
        score -= 0.2
    elif outcome in {"abandoned", "no-candidate", "rejected"}:
        score -= 0.3
    if platform_outcome.get("degradedQualified"):
        score += 0.15
    return round(score, 6)


def derive_symptom_tags(
    diagnosis: dict[str, Any] | None = None,
    gate: dict[str, Any] | None = None,
    platform_outcome: dict[str, Any] | None = None,
) -> list[str]:
    diagnosis = diagnosis or {}
    gate = gate or {}
    platform_outcome = platform_outcome or {}
    tags: list[str] = []

    for symptom in [
        diagnosis.get("primary_symptom"),
        *list(diagnosis.get("secondary_symptoms") or []),
    ]:
        normalized = _normalize_tag(symptom)
        if normalized:
            tags.append(normalized)

    for check in gate.get("checks", []):
        if isinstance(check, dict) and check.get("result") != "PASS":
            normalized = _normalize_tag(check.get("name"))
            if normalized:
                tags.append(normalized)

    for reason in gate.get("reasons", []):
        reason_text = str(reason or "").lower()
        if "daily pnl" in reason_text:
            tags.extend(["no_daily_pnl", "degraded_no_daily_pnl"])
        if "testsharpe" in reason_text or "test sharpe" in reason_text:
            tags.append("negative_test_sharpe")

    outcome = _normalize_tag(platform_outcome.get("outcome") or platform_outcome.get("status"))
    if outcome:
        tags.append(outcome)
    if outcome == "no_daily_pnl":
        tags.append("degraded_no_daily_pnl")
    if platform_outcome.get("degradedQualified"):
        tags.append("degraded_qualified")

    repair_depth = _int_or_default(platform_outcome.get("repairDepth"), 0)
    if repair_depth > 0:
        tags.append("recursive_repair")
        tags.append(f"repair_depth_{repair_depth}")

    return _unique_sorted(tags)


def build_recursive_repair_guidance(repair_context: dict[str, Any] | None = None) -> list[str]:
    if not repair_context:
        return []
    guidance: list[str] = []
    repair_depth = _int_or_default(repair_context.get("repairDepth"), 0)
    if repair_depth > 0:
        guidance.append(f"Repair depth: {repair_depth}")
        guidance.append(
            "Recursive repair: avoid pure window tuning; make a material change in field family, "
            "dominant structure, or peer grouping granularity."
        )

    next_action = str(repair_context.get("nextAction") or "").strip()
    if next_action:
        guidance.append(f"Next action hint: {next_action}")

    gate_reasons = repair_context.get("gate", {}).get("reasons") or []
    if any("daily pnl" in str(reason or "").lower() for reason in gate_reasons):
        guidance.append(
            "Platform symptom: no_daily_pnl / degraded_no_daily_pnl. Prefer broader-coverage, "
            "rank-based, economically transparent blends that survive degraded evaluation."
        )
    return guidance


def _extract_fields(expression: str, operators: list[str], group_fields: list[str]) -> list[str]:
    identifiers = set(_IDENTIFIER.findall(expression))
    return sorted(
        identifier
        for identifier in identifiers - set(operators) - set(group_fields) - _RESERVED
        if not identifier.isupper()
    )


def _extract_group_fields(expression: str) -> list[str]:
    fields = set()
    for match in re.finditer(r",\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\)", expression):
        field = match.group(1)
        if field in _GROUP_FIELDS:
            fields.add(field)
    return sorted(fields)


def _normalize_tag(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    text = re.sub(r"[^a-z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def _int_or_default(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _extract_windows(expression: str, operators: list[str]) -> list[int]:
    if not expression or not set(operators).intersection(_WINDOW_OPERATORS):
        return []
    values = set()
    for match in re.finditer(r",\s*(\d+)\s*\)", expression):
        try:
            values.add(int(match.group(1)))
        except ValueError:
            continue
    return sorted(values)


def _family_signature(
    operators: list[str],
    fields: list[str],
    windows: list[int],
    group_fields: list[str],
    dominant: str,
) -> str:
    return "|".join(
        [
            dominant,
            "ops=" + ",".join(operators),
            "fields=" + ",".join(fields),
            "windows=" + ",".join(str(window) for window in windows),
            "groups=" + ",".join(group_fields),
        ]
    )


def _normalize_category(category: str | None) -> str:
    value = str(category or "").strip().lower()
    aliases = {
        "mom": "momentum",
        "momentum": "momentum",
        "reversal": "reversal",
        "liquidity": "liquidity",
        "volume": "liquidity",
        "quality": "quality",
        "fundamental": "fundamental",
        "fundamentals": "fundamental",
        "sentiment": "sentiment",
        "risk": "risk",
    }
    return aliases.get(value, value if value in _THEMES else "")


def _thesis_for_theme(theme: str, math_profile: dict[str, Any]) -> str:
    structure = str(math_profile.get("dominant_structure") or "structure")
    return {
        "momentum": f"Momentum signal using {structure} price behavior.",
        "reversal": f"Reversal signal using {structure} mean-reversion behavior.",
        "liquidity": f"Liquidity signal using {structure} volume or trading-activity pressure.",
        "quality": f"Quality signal using {structure} profitability or balance-sheet strength.",
        "fundamental": f"Fundamental signal using {structure} accounting anchors.",
        "sentiment": f"Sentiment signal using {structure} news or expectation changes.",
        "risk": f"Risk signal using {structure} volatility or exposure control.",
    }.get(theme, f"Unknown economic thesis using {structure}.")


def _unique_sorted(values: list[str]) -> list[str]:
    return sorted({str(value) for value in values if value})


def _float_or_none(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "analyze_math_profile",
    "infer_economic_profile",
    "compare_repair",
    "score_repair_outcome",
]
