from __future__ import annotations

class KnowledgeDistiller:
    def __init__(self, kb, router=None):
        self.kb = kb
        self.router = router

    def distill(self, batch_result: dict) -> None:
        records = batch_result.get("records", [])
        for record in records:
            candidate = record.get("candidate", {})
            scorecard = record.get("scorecard", {})
            category = candidate.get("category", "")
            gate = scorecard.get("gate", "UNKNOWN")
            expression = candidate.get("expression", "")
            skeleton = _extract_operator_skeleton(expression)

            self.kb.record_strategy_stat(
                category=category,
                gate_result=gate,
                operator_skeleton=skeleton,
            )
            for operator in _extract_operators(expression):
                self.kb.record_operator_stat(
                    operator=operator,
                    category=category,
                    passed=gate == "PASS",
                )

        if self.router is None:
            return

        failed = [
            record
            for record in records
            if record.get("scorecard", {}).get("gate") == "FAIL"
        ]
        if not failed:
            return

        provider = self.router.pick("distill")
        prompt_data = {
            "task": "extract_failure_patterns",
            "failed_candidates": [
                {
                    "expression": record.get("candidate", {}).get("expression", ""),
                    "reason": record.get("scorecard", {}).get("reason", "UNKNOWN"),
                }
                for record in failed[:10]
            ],
        }
        system = (
            'Extract failure patterns. Return JSON: {"failure_patterns":'
            '[{"reason":str,"expression":str,"suggested_fix":str}],'
            '"operator_stats":[],"market_regime":null or '
            '{"summary":str,"top_categories":[str]}}'
        )
        result = _call_distill_llm(provider, system, prompt_data)

        for pattern in result.get("failure_patterns", []):
            self.kb.record_failure_pattern(
                reason=pattern.get("reason", ""),
                expression=pattern.get("expression", ""),
                suggested_fix=pattern.get("suggested_fix", ""),
            )

        regime = result.get("market_regime")
        if regime and isinstance(regime, dict):
            self.kb.upsert_market_regime(
                regime_key="current",
                summary=regime.get("summary", ""),
                top_categories=regime.get("top_categories", []),
            )


def _call_distill_llm(provider, system, data):
    import json
    import os

    try:
        if provider.client_type == "openai_compat":
            from openai import OpenAI

            client = OpenAI(
                api_key=os.environ.get(provider.api_key_env, ""),
                base_url=provider.api_base or None,
            )
            resp = client.chat.completions.create(
                model=provider.model_id,
                max_tokens=800,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(data)},
                ],
            )
            raw = resp.choices[0].message.content or "{}"
        else:
            from anthropic import Anthropic

            client = Anthropic(api_key=os.environ.get(provider.api_key_env, ""))
            resp = client.messages.create(
                model=provider.model_id,
                max_tokens=800,
                system=system,
                messages=[{"role": "user", "content": json.dumps(data)}],
            )
            raw = resp.content[0].text if resp.content else "{}"
        return json.loads(raw)
    except Exception:
        return {}


def _extract_operator_skeleton(expr: str) -> str:
    import re

    without_numbers = re.sub(r"\b\d+\b", "N", expr)
    return re.sub(r"\b[a-z][a-z0-9_]*\b(?!\s*\()", "X", without_numbers)


def _extract_operators(expr: str) -> list[str]:
    import re

    return re.findall(r"\b([a-z][a-z0-9_]*)\s*\(", expr)
