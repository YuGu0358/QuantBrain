from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from dataclasses import is_dataclass
from typing import Any

from alpha_miner.modules.m_diagnoser import DiagnosisReport
from alpha_miner.modules.m_repair_memory import RepairMemory


class Retriever:
    def __init__(self, memory: RepairMemory, router=None):
        self.memory = memory
        self.router = router

    def retrieve(
        self,
        diagnosis: DiagnosisReport,
        expression: str,
        family_tag: str | None = None,
    ) -> dict:
        symptom_tags = _diagnosis_symptom_tags(diagnosis)
        retrieval = self.memory.retrieve(symptom_tags, expression, family_tag, topk=5)
        family_saturated = self.memory.family_saturation(family_tag) if family_tag else False
        retrieval_summary = ""

        if self.router is not None:
            provider = None
            try:
                provider = _pick_provider(self.router, "repair")
                if provider is None:
                    raise RuntimeError("No LLM provider available")
                request_payload = {
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are a repair Retriever. Given memory search results, write a 2-sentence "
                                "summary of the most actionable insights for this repair. Focus on: what worked, "
                                "what to avoid, and whether the family is saturated. Be concise and specific."
                            ),
                        },
                        {
                            "role": "user",
                            "content": json.dumps(
                                {
                                    "primary_symptom": diagnosis.primary_symptom,
                                    "top_positive_expressions": [
                                        record.get("expression", "")
                                        for record in retrieval.get("positive", [])[:3]
                                    ],
                                    "top_forbidden_directions": retrieval.get("forbidden_directions", [])[:3],
                                    "family_saturated": family_saturated,
                                },
                                ensure_ascii=False,
                            ),
                        },
                    ],
                    "max_tokens": 300,
                    "max_completion_tokens": 300,
                }
                raw, tokens_in, tokens_out, latency_ms = _invoke_provider(self, self.router, provider, request_payload)
                retrieval_summary = raw.strip()
                _record_result(self.router, provider, bool(retrieval_summary), latency_ms, tokens_in, tokens_out)
            except Exception:
                if provider is not None:
                    _record_result(self.router, provider, False, 0.0, 0, 0)
                retrieval_summary = ""

        return {
            **retrieval,
            "family_saturated": family_saturated,
            "retrieval_summary": retrieval_summary,
        }

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
                    request_payload.get("max_tokens", 300),
                )
            else:
                kwargs["max_tokens"] = request_payload.get("max_tokens", 300)
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
                max_tokens=request_payload.get("max_tokens", 300),
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


def _diagnosis_symptom_tags(diagnosis: DiagnosisReport) -> list[str]:
    data = asdict(diagnosis) if is_dataclass(diagnosis) else {}
    primary = data.get("primary_symptom")
    secondary = data.get("secondary_symptoms", [])
    tags = [primary] if primary else []
    if isinstance(secondary, list):
        tags.extend(str(item) for item in secondary)
    return list(dict.fromkeys(str(tag) for tag in tags if tag))


def _pick_provider(router: Any, role: str) -> Any | None:
    if router is None:
        return None
    try:
        return router.pick(role)
    except Exception:
        pass
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


def _is_reasoning_model(model_id: str) -> bool:
    return model_id.startswith(("o1", "o3", "o4", "gpt-5"))


__all__ = ["Retriever"]
