from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .m_repair_intelligence import analyze_math_profile, infer_economic_profile


_PRICE_FIELDS = {"close", "open", "high", "low", "returns", "vwap"}
_LIQUIDITY_FIELDS = {"volume", "adv20"}
_QUALITY_FIELDS = {"assets", "cashflow_op", "operating_income", "est_eps"}
_SENTIMENT_FIELDS = {"news_sentiment"}
_RISK_FIELDS = {"beta", "volatility", "sigma"}


@dataclass(frozen=True)
class GenerationQualityAssessment:
    candidate_id: str
    expression: str
    score: float
    passed: bool
    theme: str
    dominant_structure: str
    field_families: list[str]
    signature: str
    reasons: list[str]
    warnings: list[str]


def assess_generation_candidate_quality(
    expression: str,
    *,
    category: str = "",
    candidate_id: str = "",
    seen_signatures: set[str] | None = None,
    min_score: float = 0.45,
) -> GenerationQualityAssessment:
    math_profile = analyze_math_profile(expression)
    economic_profile = infer_economic_profile(
        expression,
        category=None,
        fields=math_profile.get("fields", []),
    )

    fields = set(math_profile.get("fields") or [])
    families = _field_families(fields)
    reasons: list[str] = []
    warnings: list[str] = []
    score = 0.0

    if len(families) >= 2:
        score += 1.2
    else:
        reasons.append("single_family_exposure")
        score -= 1.0

    if len(fields) >= 3:
        score += 0.4
    elif len(fields) >= 2:
        score += 0.2
    else:
        warnings.append("thin_field_set")
        score -= 0.4

    if math_profile.get("has_group_neutralization"):
        score += 0.35
    else:
        warnings.append("missing_group_neutralization")
        score -= 0.1

    dominant_structure = str(math_profile.get("dominant_structure") or "")
    if dominant_structure == "hybrid_group_time_series":
        score += 0.55
    elif dominant_structure in {"time_series", "cross_sectional_group"}:
        score += 0.2
    elif dominant_structure == "cross_sectional":
        score -= 0.05

    windows = list(math_profile.get("windows") or [])
    if windows:
        if max(windows) >= 20:
            score += 0.25
        elif max(windows) < 10:
            warnings.append("short_horizon_fragility")
            score -= 0.35

    if _is_generic_single_field_expression(math_profile):
        reasons.append("insufficient_structure")
        score -= 0.6

    if category and _category_matches_quality_theme(category, economic_profile.get("theme"), families):
        score += 0.2

    signature = str(math_profile.get("family_signature") or "")
    if seen_signatures and signature and signature in seen_signatures:
        reasons.append("duplicate_signature")
        score -= 1.0

    critical_reasons = {"single_family_exposure", "duplicate_signature", "insufficient_structure"}
    passed = score >= min_score and not critical_reasons.intersection(reasons)

    return GenerationQualityAssessment(
        candidate_id=str(candidate_id or ""),
        expression=str(expression or ""),
        score=round(score, 6),
        passed=passed,
        theme=str(economic_profile.get("theme") or "unknown"),
        dominant_structure=dominant_structure,
        field_families=families,
        signature=signature,
        reasons=reasons,
        warnings=warnings,
    )


def summarize_generation_quality(
    assessments: list[GenerationQualityAssessment],
    *,
    selected_count: int,
    fallback_count: int,
    judge_applied: bool,
) -> dict[str, Any]:
    passed = [item for item in assessments if item.passed]
    scores = [item.score for item in assessments]
    return {
        "judge_applied": bool(judge_applied),
        "assessed_count": len(assessments),
        "passed_count": len(passed),
        "rejected_count": max(len(assessments) - len(passed), 0),
        "selected_count": int(selected_count),
        "fallback_count": int(fallback_count),
        "median_score": round(_median(scores), 6) if scores else 0.0,
        "sample_confidence": round(_median([item.score for item in passed]), 6) if passed else 0.0,
    }


def _field_families(fields: set[str]) -> list[str]:
    families: list[str] = []
    if fields & _PRICE_FIELDS:
        families.append("price")
    if fields & _LIQUIDITY_FIELDS:
        families.append("liquidity")
    if fields & _QUALITY_FIELDS:
        families.append("quality")
    if fields & _SENTIMENT_FIELDS:
        families.append("sentiment")
    if fields & _RISK_FIELDS:
        families.append("risk")
    return families


def _is_generic_single_field_expression(math_profile: dict[str, Any]) -> bool:
    fields = set(math_profile.get("fields") or [])
    operators = set(math_profile.get("operators") or [])
    complexity = int(math_profile.get("complexity") or 0)
    return len(fields) <= 1 and len(operators) <= 1 and complexity <= 2


def _category_matches_quality_theme(category: str, theme: Any, families: list[str]) -> bool:
    normalized = str(category or "").strip().lower()
    current_theme = str(theme or "").strip().lower()
    if normalized in {"quality", "fundamental"} and ("quality" in families or current_theme in {"quality", "fundamental"}):
        return True
    if normalized == "liquidity" and ("liquidity" in families or current_theme == "liquidity"):
        return True
    if normalized in {"momentum", "reversal", "volatility", "microstructure"} and current_theme not in {"unknown", ""}:
        return True
    return False


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0
