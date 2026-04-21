from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict
from dataclasses import is_dataclass
from typing import Any

from alpha_miner.modules.llm_router import LLMRouter
from alpha_miner.modules.m_diagnoser import DiagnosisReport
from alpha_miner.modules.m_repair_memory import RepairMemory


class Distiller:
    def __init__(self, router: LLMRouter | None, memory: RepairMemory):
        self.router = router
        self.memory = memory

    def distill(
        self,
        original_expression: str,
        diagnosis: DiagnosisReport,
        tried_candidates: list[dict],
        accepted_expression: str | None,
    ) -> dict:
        distilled = _empty_distillation()
        provider = None
        primary_role = os.environ.get("DISTILL_LLM_ROLE", "distill")
        fallback_role = os.environ.get("DISTILL_FALLBACK_ROLE", "repair")
        request_payload = None
        try:
            provider = _pick_provider(self.router, primary_role, fallback_role)
            if provider is None:
                raise RuntimeError("No LLM provider available")
            request_payload = {
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "original_expression": original_expression,
                                "diagnosis": _diagnosis_to_dict(diagnosis),
                                "tried_candidates": _candidate_summary(tried_candidates),
                                "accepted_expression": accepted_expression,
                            },
                            ensure_ascii=False,
                        ),
                    },
                ],
                "max_tokens": 800,
                "max_completion_tokens": 800,
            }
            _max_tries = 4
            _backoff_529 = [20, 45, 90]
            _backoff_429 = [5, 15, 30]
            for _attempt in range(_max_tries):
                try:
                    raw, tokens_in, tokens_out, latency_ms = _invoke_provider(self, self.router, provider, request_payload)
                    break
                except Exception as _invoke_e:
                    _emsg = str(_invoke_e)
                    if _attempt < _max_tries - 1:
                        if "529" in _emsg:
                            _wait = _backoff_529[_attempt]
                            print(f"[distiller] distill overloaded (529), retry {_attempt+1}/{_max_tries-1} in {_wait}s", flush=True)
                            time.sleep(_wait)
                            continue
                        if "429" in _emsg:
                            _wait = _backoff_429[_attempt]
                            print(f"[distiller] distill rate-limited (429), retry {_attempt+1}/{_max_tries-1} in {_wait}s", flush=True)
                            time.sleep(_wait)
                            continue
                    raise
            distilled = _normalize_distillation(_extract_json(raw))
            _record_result(self.router, provider, True, latency_ms, tokens_in, tokens_out)
        except Exception as _e:
            print(f"[distiller] distill failed: {type(_e).__name__}: {_e}", flush=True)
            if provider is not None:
                _record_result(self.router, provider, False, 0.0, 0, 0)
            _fallback_succeeded = False
            if request_payload is not None and provider is not None and fallback_role and provider.role != fallback_role:
                hq_provider = _pick_provider(self.router, fallback_role)
                if hq_provider is not None:
                    try:
                        raw, tokens_in, tokens_out, latency_ms = _invoke_provider(self, self.router, hq_provider, request_payload)
                        distilled = _normalize_distillation(_extract_json(raw))
                        _record_result(self.router, hq_provider, True, latency_ms, tokens_in, tokens_out)
                        _fallback_succeeded = True
                    except Exception as _e2:
                        print(f"[distiller] fallback distill failed: {type(_e2).__name__}: {_e2}", flush=True)
                        _record_result(self.router, hq_provider, False, 0.0, 0, 0)
            if not _fallback_succeeded:
                distilled = _empty_distillation()

        self.memory.add_record(
            {
                "expression": original_expression,
                "symptom_tags": _diagnosis_symptom_tags(diagnosis),
                "repair_actions": [],
                "accept_decision": "accepted" if accepted_expression else "rejected",
                "rejection_reason": "" if accepted_expression else "no accepted repair candidate",
                "recommended_directions": distilled["recommended_directions"],
                "forbidden_directions": distilled["forbidden_directions"],
                "metrics": _best_candidate_metrics(tried_candidates, accepted_expression),
                "notes": json.dumps(
                    {
                        "accepted_expression": accepted_expression,
                        "reusable_patterns": distilled["reusable_patterns"],
                        "regime_lessons": distilled["regime_lessons"],
                    },
                    ensure_ascii=False,
                ),
            }
        )
        return distilled

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
    "You are a quantitative factor repair Distiller. Extract reusable lessons from this repair round.\n"
    "Return ONLY valid JSON with keys: recommended_directions (list[str]), forbidden_directions (list[str]),\n"
    "reusable_patterns (list[str]), regime_lessons (list[str]), symptom_tags (list[str]).\n"
    "Directions should mention reusable mathematical actions and economic themes when visible, "
    "for example smoothing liquidity signals or avoiding group-only wrappers for self-correlation."
)


def _diagnosis_to_dict(diagnosis: DiagnosisReport) -> dict[str, Any]:
    if is_dataclass(diagnosis):
        return asdict(diagnosis)
    if isinstance(diagnosis, dict):
        return diagnosis
    return {}


def _diagnosis_symptom_tags(diagnosis: DiagnosisReport) -> list[str]:
    data = _diagnosis_to_dict(diagnosis)
    primary = data.get("primary_symptom")
    secondary = data.get("secondary_symptoms", [])
    tags = [primary] if primary else []
    if isinstance(secondary, list):
        tags.extend(str(item) for item in secondary)
    return list(dict.fromkeys(str(tag) for tag in tags if tag))


def _candidate_summary(tried_candidates: list[dict]) -> list[dict[str, Any]]:
    keys = (
        "expression",
        "sharpe",
        "fitness",
        "turnover",
        "max_abs_correlation",
        "accepted",
        "repair_delta",
        "economic_profile",
        "math_profile",
    )
    return [{key: candidate.get(key) for key in keys if key in candidate} for candidate in tried_candidates]


def _best_candidate_metrics(tried_candidates: list[dict], accepted_expression: str | None) -> dict[str, Any]:
    if not tried_candidates:
        return {}
    best = _best_candidate(tried_candidates, accepted_expression)
    metrics = {}
    for key in ("sharpe", "fitness", "turnover", "max_abs_correlation"):
        if key in best and best[key] is not None:
            metrics[key] = best[key]
    return metrics


def _best_candidate(tried_candidates: list[dict], accepted_expression: str | None) -> dict:
    if accepted_expression:
        for candidate in tried_candidates:
            if candidate.get("expression") == accepted_expression:
                return candidate
    accepted = [candidate for candidate in tried_candidates if candidate.get("accepted")]
    pool = accepted or tried_candidates
    return max(
        pool,
        key=lambda candidate: (
            _float_or_default(candidate.get("sharpe")),
            _float_or_default(candidate.get("fitness")),
        ),
    )


def _normalize_distillation(value: dict[str, Any]) -> dict[str, list[str]]:
    return {
        "recommended_directions": _string_list(value.get("recommended_directions", [])),
        "forbidden_directions": _string_list(value.get("forbidden_directions", [])),
        "reusable_patterns": _string_list(value.get("reusable_patterns", [])),
        "regime_lessons": _string_list(value.get("regime_lessons", [])),
        "symptom_tags": _string_list(value.get("symptom_tags", [])),
    }


def _empty_distillation() -> dict[str, list[str]]:
    return {
        "recommended_directions": [],
        "forbidden_directions": [],
        "reusable_patterns": [],
        "regime_lessons": [],
        "symptom_tags": [],
    }


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
        if any(r == role for _, r in providers.keys()):
            return True
        # Test doubles may switch provider roles dynamically via router.pick.
        if providers_by_role is None and callable(getattr(router, "pick", None)):
            return True
        return False
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
    return json.loads(text)


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _float_or_default(value: Any, default: float = float("-inf")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _is_reasoning_model(model_id: str) -> bool:
    return model_id.startswith(("o1", "o3", "o4", "gpt-5"))


__all__ = ["Distiller"]
