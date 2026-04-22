from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any

from alpha_miner.modules.llm_router import LLMRouter


@dataclass
class DiagnosisReport:
    primary_symptom: str
    secondary_symptoms: list[str]
    root_causes: list[str]
    repair_priorities: list[dict]
    do_not_change: list[str]
    raw: dict


class Diagnoser:
    def __init__(self, router: LLMRouter | None):
        self.router = router

    def diagnose(
        self,
        expression: str,
        metrics: dict,
        failed_checks: list[str],
        gate_reasons: list[str],
    ) -> DiagnosisReport:
        enriched_metrics = _metrics_with_complexity(expression, metrics)
        user_payload = {
            "expression": expression,
            "metrics": enriched_metrics,
            "failed_checks": failed_checks,
            "gate_reasons": gate_reasons,
        }
        request_payload = {
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
            "max_tokens": 600,
            "max_completion_tokens": 600,
        }

        provider = None
        try:
            primary_role = os.environ.get("DIAGNOSE_LLM_ROLE", "diagnose")
            fallback_role = os.environ.get("DIAGNOSE_FALLBACK_ROLE", "repair")
            provider = _pick_provider(self.router, primary_role, fallback_role)
            if provider is None:
                raise RuntimeError("No LLM provider available")
            raw, tokens_in, tokens_out, latency_ms = _invoke_provider(self, self.router, provider, request_payload)
            parsed = _extract_json(raw)
            report = _report_from_mapping(parsed)
            _record_result(self.router, provider, True, latency_ms, tokens_in, tokens_out)
            return report
        except Exception as exc:
            if provider is not None:
                _record_result(self.router, provider, False, 0.0, 0, 0)
            # One-shot high-quality fallback retry when lightweight diagnose fails.
            if provider is not None and fallback_role and provider.role != fallback_role:
                hq_provider = _pick_provider(self.router, fallback_role)
                if hq_provider is not None:
                    try:
                        raw, tokens_in, tokens_out, latency_ms = _invoke_provider(self, self.router, hq_provider, request_payload)
                        parsed = _extract_json(raw)
                        report = _report_from_mapping(parsed)
                        report.raw.update(
                            {
                                "llm_fallback": True,
                                "primary_error": str(exc),
                                "primary_provider": provider.name,
                                "fallback_provider": hq_provider.name,
                                "fallback": False,
                                "error": None,
                            }
                        )
                        _record_result(self.router, hq_provider, True, latency_ms, tokens_in, tokens_out)
                        return report
                    except Exception:
                        _record_result(self.router, hq_provider, False, 0.0, 0, 0)
            return _fallback_report(
                expression=expression,
                metrics=enriched_metrics,
                failed_checks=failed_checks,
                gate_reasons=gate_reasons,
                error=str(exc),
            )

    def _call_provider(self, provider: Any, request_payload: dict[str, Any]) -> tuple[str, int, int, float]:
        t0 = time.time()
        if provider.client_type == "openai_compat":
            from openai import OpenAI

            client = OpenAI(api_key=os.environ.get(provider.api_key_env, ""), base_url=provider.api_base or None)
            kwargs: dict[str, Any] = {
                "model": provider.model_id,
                "messages": request_payload["messages"],
            }
            if _is_reasoning_model(provider.model_id):
                kwargs["max_completion_tokens"] = request_payload.get(
                    "max_completion_tokens",
                    request_payload.get("max_tokens", 2000),
                )
            else:
                kwargs["max_tokens"] = request_payload.get("max_tokens", 800)
                if "temperature" in request_payload:
                    kwargs["temperature"] = request_payload["temperature"]
            resp = client.chat.completions.create(**kwargs)
            raw = resp.choices[0].message.content or ""
            tokens_in = resp.usage.prompt_tokens if resp.usage else 0
            tokens_out = resp.usage.completion_tokens if resp.usage else 0
        elif provider.client_type == "anthropic":
            from anthropic import Anthropic

            client = Anthropic(api_key=os.environ.get(provider.api_key_env, ""))
            messages = [message for message in request_payload["messages"] if message["role"] != "system"]
            system = next(
                (message["content"] for message in request_payload["messages"] if message["role"] == "system"),
                "",
            )
            resp = client.messages.create(
                model=provider.model_id,
                max_tokens=request_payload.get("max_tokens", 800),
                system=system,
                messages=messages,
            )
            raw = resp.content[0].text if resp.content else ""
            tokens_in = resp.usage.input_tokens if resp.usage else 0
            tokens_out = resp.usage.output_tokens if resp.usage else 0
        else:
            raise ValueError(f"Unsupported provider client_type: {provider.client_type}")
        latency_ms = (time.time() - t0) * 1000
        return raw, tokens_in, tokens_out, latency_ms


_SYSTEM_PROMPT = (
    "You are a quantitative factor repair Diagnoser. Convert the failed factor metrics into a structured diagnosis.\n"
    "Available symptom types: low_sharpe (sharpe<1.0), low_fitness (fitness<0.5), high_turnover (turnover>0.7),\n"
    "low_turnover (turnover<0.01), high_corrlib (max_abs_correlation>0.7), high_complexity (complexity>15 operators),\n"
    "compile_fail (expression failed validation).\n"
    "Return ONLY valid JSON matching this schema exactly: "
    '{"primary_symptom":"one of the listed symptom types","secondary_symptoms":["symptom_type"],'
    '"root_causes":["short cause"],"repair_priorities":[{"rank":1,"target_metric":"metric","suggested_action_type":"action"}],'
    '"do_not_change":["constraint"]}. Use snake_case symptom labels exactly as listed.'
)

_VALID_SYMPTOMS = {
    "low_sharpe",
    "low_fitness",
    "high_turnover",
    "low_turnover",
    "high_corrlib",
    "high_complexity",
    "compile_fail",
}

_OPERATORS = [
    "abs",
    "correlation",
    "covariance",
    "delay",
    "delta",
    "divide",
    "group_neutralize",
    "group_rank",
    "hump",
    "log",
    "max",
    "min",
    "multiply",
    "power",
    "rank",
    "regression_neut",
    "scale",
    "sign",
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
    "winsorize",
    "zscore",
]


def _metrics_with_complexity(expression: str, metrics: dict) -> dict[str, Any]:
    return {
        "sharpe": metrics.get("sharpe"),
        "fitness": metrics.get("fitness"),
        "turnover": metrics.get("turnover"),
        "max_abs_correlation": metrics.get("max_abs_correlation"),
        "complexity": _expression_complexity(expression),
    }


def _expression_complexity(expression: str) -> int:
    pattern = r"\b(?:" + "|".join(re.escape(operator) for operator in sorted(_OPERATORS, key=len, reverse=True)) + r")\s*\("
    return len(re.findall(pattern, expression))


def _report_from_mapping(value: dict[str, Any]) -> DiagnosisReport:
    normalized = _normalize_diagnosis_mapping(value)
    primary = str(normalized.get("primary_symptom", ""))
    if primary not in _VALID_SYMPTOMS:
        raise ValueError(f"Invalid primary_symptom: {primary!r}")
    return DiagnosisReport(
        primary_symptom=primary,
        secondary_symptoms=_string_list(normalized.get("secondary_symptoms", [])),
        root_causes=_string_list(normalized.get("root_causes", [])),
        repair_priorities=[dict(item) for item in normalized.get("repair_priorities", []) if isinstance(item, dict)],
        do_not_change=_string_list(normalized.get("do_not_change", [])),
        raw=normalized,
    )


def _fallback_report(
    expression: str,
    metrics: dict[str, Any],
    failed_checks: list[str],
    gate_reasons: list[str],
    error: str,
) -> DiagnosisReport:
    primary = _fallback_primary_symptom(metrics)
    symptoms = _all_rule_symptoms(metrics, failed_checks, gate_reasons)
    secondary = [symptom for symptom in symptoms if symptom != primary]
    return DiagnosisReport(
        primary_symptom=primary,
        secondary_symptoms=secondary,
        root_causes=[f"rule_based_fallback:{primary}"],
        repair_priorities=[_priority_for_symptom(primary)],
        do_not_change=_preservation_hints(expression),
        raw={
            "fallback": True,
            "error": error,
            "expression": expression,
            "metrics": metrics,
            "failed_checks": failed_checks,
            "gate_reasons": gate_reasons,
        },
    )


def _fallback_primary_symptom(metrics: dict[str, Any]) -> str:
    sharpe = _float_or_none(metrics.get("sharpe"))
    turnover = _float_or_none(metrics.get("turnover"))
    fitness = _float_or_none(metrics.get("fitness"))
    max_abs_correlation = _float_or_none(metrics.get("max_abs_correlation"))
    if sharpe is not None and sharpe < 1.0:
        return "low_sharpe"
    if turnover is not None and turnover > 0.7:
        return "high_turnover"
    if turnover is not None and turnover < 0.01:
        return "low_turnover"
    if fitness is not None and fitness < 0.5:
        return "low_fitness"
    if max_abs_correlation is not None and max_abs_correlation > 0.7:
        return "high_corrlib"
    return "low_sharpe"


def _all_rule_symptoms(metrics: dict[str, Any], failed_checks: list[str], gate_reasons: list[str]) -> list[str]:
    symptoms: list[str] = []
    sharpe = _float_or_none(metrics.get("sharpe"))
    fitness = _float_or_none(metrics.get("fitness"))
    turnover = _float_or_none(metrics.get("turnover"))
    max_abs_correlation = _float_or_none(metrics.get("max_abs_correlation"))
    complexity = _float_or_none(metrics.get("complexity"))
    if sharpe is not None and sharpe < 1.0:
        symptoms.append("low_sharpe")
    if fitness is not None and fitness < 0.5:
        symptoms.append("low_fitness")
    if turnover is not None and turnover > 0.7:
        symptoms.append("high_turnover")
    if turnover is not None and turnover < 0.01:
        symptoms.append("low_turnover")
    if max_abs_correlation is not None and max_abs_correlation > 0.7:
        symptoms.append("high_corrlib")
    if complexity is not None and complexity > 15:
        symptoms.append("high_complexity")
    combined_failures = " ".join([*failed_checks, *gate_reasons]).lower()
    if any(token in combined_failures for token in ("compile", "syntax", "validation")):
        symptoms.append("compile_fail")
    return list(dict.fromkeys(symptoms))


def _priority_for_symptom(symptom: str) -> dict[str, Any]:
    by_symptom = {
        "low_sharpe": ("sharpe", "add_neutralization_or_complementary_signal"),
        "low_fitness": ("fitness", "normalize_and_blend_signals"),
        "high_turnover": ("turnover", "smooth_fast_signal"),
        "low_turnover": ("turnover", "shorten_windows_or_add_fast_component"),
        "high_corrlib": ("max_abs_correlation", "change_primary_data_or_horizon"),
        "high_complexity": ("complexity", "remove_redundant_operator_layers"),
        "compile_fail": ("validation", "repair_expression_syntax"),
    }
    target_metric, action = by_symptom.get(symptom, by_symptom["low_sharpe"])
    return {
        "rank": 1,
        "target_metric": target_metric,
        "suggested_action_type": action,
        "keep_constraints": ["preserve core economic thesis"],
    }


def _preservation_hints(expression: str) -> list[str]:
    hints = []
    for token in ("industry", "subindustry"):
        if re.search(rf"\b{token}\b", expression):
            hints.append(token)
    return hints


def _pick_provider(router: Any, *roles: str) -> Any | None:
    if router is None:
        return None
    for role in roles:
        if not role:
            continue
        if not _router_has_role(router, role):
            continue
        try:
            return router.pick(role)
        except Exception:
            continue
    providers_by_role = getattr(router, "_providers_by_role", None)
    if isinstance(providers_by_role, dict):
        for providers in providers_by_role.values():
            if providers:
                return providers[0]
    providers = getattr(router, "_providers", None)
    if isinstance(providers, dict) and providers:
        return next(iter(providers.values()))
    providers = getattr(router, "providers", None)
    if isinstance(providers, list) and providers:
        return providers[0]
    return None


def _invoke_provider(agent: Any, router: Any, provider: Any, request_payload: dict[str, Any]) -> tuple[str, int, int, float]:
    if router is not None and hasattr(router, "_call_provider"):
        return router._call_provider(provider, request_payload)
    return agent._call_provider(provider, request_payload)


def _record_result(router: Any, provider: Any, passed: bool, latency_ms: float, tokens_in: int, tokens_out: int) -> None:
    if router is None or not hasattr(router, "record_result"):
        return
    try:
        router.record_result(provider.name, provider.role, passed, latency_ms, tokens_in, tokens_out)
    except Exception:
        return


def _router_has_role(router: Any, role: str) -> bool:
    providers_by_role = getattr(router, "_providers_by_role", None)
    if isinstance(providers_by_role, dict):
        candidates = providers_by_role.get(role)
        if isinstance(candidates, list):
            return len(candidates) > 0
    providers = getattr(router, "_providers", None)
    if isinstance(providers, dict):
        return any(r == role for _, r in providers.keys())
    return True


def _extract_json(raw: str) -> dict[str, Any]:
    text = raw.strip()
    match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    if match:
        text = match.group(1)
    elif not text.startswith("{"):
        brace = text.find("{")
        if brace != -1:
            text = text[brace:]
    decoder = json.JSONDecoder()
    parsed, _end = decoder.raw_decode(text)
    if not isinstance(parsed, dict):
        raise ValueError("Diagnosis payload is not a JSON object")
    return parsed


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        cleaned = value.strip()
        return [cleaned] if cleaned else []
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _normalize_diagnosis_mapping(value: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Diagnosis payload must be a mapping")
    normalized = dict(value)
    normalized["primary_symptom"] = _normalize_symptom(value.get("primary_symptom"))
    normalized["secondary_symptoms"] = _normalize_symptom_list(value.get("secondary_symptoms"))
    normalized["root_causes"] = _string_list(value.get("root_causes", []))
    normalized["do_not_change"] = _string_list(value.get("do_not_change", []))
    repair_priorities = value.get("repair_priorities", [])
    normalized["repair_priorities"] = repair_priorities if isinstance(repair_priorities, list) else []
    return normalized


def _normalize_symptom_list(value: Any) -> list[str]:
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, list):
        items = value
    else:
        return []
    normalized: list[str] = []
    for item in items:
        symptom = _normalize_symptom(item)
        if symptom:
            normalized.append(symptom)
    return list(dict.fromkeys(normalized))


def _normalize_symptom(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    normalized = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    if normalized in _VALID_SYMPTOMS:
        return normalized
    aliases = {
        "sharpe_low": "low_sharpe",
        "low_sharpe_signal": "low_sharpe",
        "fitness_low": "low_fitness",
        "turnover_high": "high_turnover",
        "turnover_low": "low_turnover",
        "high_correlation": "high_corrlib",
        "high_corr": "high_corrlib",
        "high_corr_lib": "high_corrlib",
        "high_correlation_library": "high_corrlib",
        "complexity_high": "high_complexity",
        "validation_fail": "compile_fail",
        "validation_failed": "compile_fail",
        "syntax_error": "compile_fail",
        "compile_error": "compile_fail",
    }
    if normalized in aliases:
        return aliases[normalized]
    tokens = set(part for part in normalized.split("_") if part)
    if "turnover" in tokens and "high" in tokens:
        return "high_turnover"
    if "turnover" in tokens and "low" in tokens:
        return "low_turnover"
    if "fitness" in tokens and "low" in tokens:
        return "low_fitness"
    if "sharpe" in tokens and "low" in tokens:
        return "low_sharpe"
    if ({"corr", "correlation", "corrlib"} & tokens) and "high" in tokens:
        return "high_corrlib"
    if "complexity" in tokens and "high" in tokens:
        return "high_complexity"
    if ({"compile", "validation", "syntax"} & tokens) and ({"fail", "failed", "error", "invalid"} & tokens):
        return "compile_fail"
    return normalized


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_reasoning_model(model_id: str) -> bool:
    return model_id.startswith(("o1", "o3", "o4", "gpt-5"))


__all__ = ["DiagnosisReport", "Diagnoser"]
