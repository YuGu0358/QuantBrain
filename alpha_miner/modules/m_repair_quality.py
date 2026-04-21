from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .m_repair_intelligence import analyze_math_profile, compare_repair


@dataclass(frozen=True)
class RepairRouteDecision:
    route: str
    reasons: list[str]
    use_rule_seeds: bool


@dataclass(frozen=True)
class RepairQualityAssessment:
    candidate_id: str
    expression: str
    score: float
    passed: bool
    reasons: list[str]
    warnings: list[str]


def decide_repair_route(
    repair_context: dict[str, Any] | None,
    *,
    diagnosis: Any = None,
    has_chain: bool,
    rule_candidate_count: int,
) -> RepairRouteDecision:
    failed_checks = {
        str(item or "").upper()
        for item in ((repair_context or {}).get("failedChecks") or [])
    }
    gate_reasons = [str(item or "").lower() for item in ((repair_context or {}).get("gate", {}) or {}).get("reasons", [])]
    repair_depth = int((repair_context or {}).get("repairDepth") or 0)
    primary_symptom = str(
        diagnosis.primary_symptom if hasattr(diagnosis, "primary_symptom") else (diagnosis or {}).get("primary_symptom") or ""
    ).lower()

    complex_failure = (
        repair_depth > 0
        or len(failed_checks) >= 2
        or bool(failed_checks.intersection({"SELF_CORRELATION", "ORTHOGONALITY", "LOW_SUB_UNIVERSE_SHARPE"}))
        or primary_symptom in {"low_sharpe", "low_fitness"}
        or any("daily pnl" in reason for reason in gate_reasons)
    )

    if not has_chain:
        if rule_candidate_count > 0:
            return RepairRouteDecision(route="rule_only", reasons=["langchain_unavailable"], use_rule_seeds=False)
        return RepairRouteDecision(route="legacy", reasons=["langchain_unavailable", "no_rule_candidates"], use_rule_seeds=False)

    if rule_candidate_count > 0:
        reasons = ["rule_candidates_available"]
        if complex_failure:
            reasons.append("complex_failure_requires_llm_reasoning")
        return RepairRouteDecision(route="hybrid", reasons=reasons, use_rule_seeds=True)

    reasons = ["langchain_required"]
    if complex_failure:
        reasons.append("complex_failure_requires_llm_reasoning")
    return RepairRouteDecision(route="langchain", reasons=reasons, use_rule_seeds=False)


def assess_repair_candidate_quality(
    *,
    parent_expression: str,
    candidate: Any,
    diagnosis: dict[str, Any] | Any | None,
    repair_context: dict[str, Any] | None,
) -> RepairQualityAssessment:
    expression = str(getattr(candidate, "expression", "") or "")
    candidate_id = str(getattr(candidate, "id", "") or "")
    reasons: list[str] = []
    warnings: list[str] = []
    score = 0.0

    if not expression.strip():
        reasons.append("empty_expression")
        return RepairQualityAssessment(candidate_id=candidate_id, expression=expression, score=-1.0, passed=False, reasons=reasons, warnings=warnings)

    if expression.strip() == str(parent_expression or "").strip():
        reasons.append("no_expression_change")

    parent_math = analyze_math_profile(parent_expression)
    candidate_math = analyze_math_profile(expression)
    delta = compare_repair(parent_expression, expression)
    do_not_change = set(_do_not_change(diagnosis))

    if "industry neutralization" in do_not_change and "industry" in (parent_math.get("group_fields") or []) and "industry" not in (candidate_math.get("group_fields") or []):
        reasons.append("lost_required_industry_neutralization")
    if "subindustry neutralization" in do_not_change and "subindustry" in (parent_math.get("group_fields") or []) and "subindustry" not in (candidate_math.get("group_fields") or []):
        reasons.append("lost_required_subindustry_neutralization")

    failed_checks = {
        str(item or "").upper()
        for item in ((repair_context or {}).get("failedChecks") or [])
    }
    gate_reasons = [str(item or "").lower() for item in ((repair_context or {}).get("gate", {}) or {}).get("reasons", [])]
    repair_depth = int((repair_context or {}).get("repairDepth") or 0)
    primary_symptom = str(
        diagnosis.primary_symptom if hasattr(diagnosis, "primary_symptom") else (diagnosis or {}).get("primary_symptom") or ""
    ).lower()

    material_change = bool(
        delta.get("field_shift")
        or delta.get("group_change")
        or delta.get("quality_anchor")
        or delta.get("cross_family_escape")
    )
    if repair_depth > 0 and not material_change:
        reasons.append("recursive_repair_requires_material_change")

    if any("daily pnl" in reason for reason in gate_reasons):
        if not (candidate_math.get("has_group_neutralization") or material_change):
            reasons.append("platform_fragility_not_addressed")
        else:
            score += 0.25

    if failed_checks.intersection({"TURNOVER", "HIGH_TURNOVER"}):
        if delta.get("smoothing"):
            score += 0.4
        else:
            warnings.append("turnover_failure_without_smoothing")

    if primary_symptom in {"low_sharpe", "low_fitness"}:
        if material_change or delta.get("horizon_change"):
            score += 0.6
        else:
            reasons.append("weak_fix_for_strength_failure")

    if delta.get("thesis_preserved"):
        score += 0.2
    if delta.get("cross_family_escape"):
        score += 0.35
    if candidate_math.get("has_group_neutralization"):
        score += 0.2

    passed = not reasons and score > 0
    return RepairQualityAssessment(
        candidate_id=candidate_id,
        expression=expression,
        score=round(score, 6),
        passed=passed,
        reasons=reasons,
        warnings=warnings,
    )


def summarize_repair_quality(
    assessments: list[RepairQualityAssessment],
    *,
    route: str,
    seed_candidate_count: int,
) -> dict[str, Any]:
    passed = [item for item in assessments if item.passed]
    scores = [item.score for item in assessments]
    return {
        "route": route,
        "seed_candidate_count": int(seed_candidate_count),
        "assessed_count": len(assessments),
        "passed_count": len(passed),
        "rejected_count": max(len(assessments) - len(passed), 0),
        "sample_confidence": round(max((item.score for item in passed), default=0.0), 6),
        "median_score": round(_median(scores), 6) if scores else 0.0,
    }


def _do_not_change(diagnosis: dict[str, Any] | Any | None) -> list[str]:
    if diagnosis is None:
        return []
    if hasattr(diagnosis, "do_not_change"):
        return list(getattr(diagnosis, "do_not_change") or [])
    if isinstance(diagnosis, dict):
        return list(diagnosis.get("do_not_change") or [])
    return []


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0
