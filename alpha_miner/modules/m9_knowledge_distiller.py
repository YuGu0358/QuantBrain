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
            expression = candidate.get("expression", "")
            hypothesis = candidate.get("hypothesis", "")
            skeleton = _extract_operator_skeleton(expression)

            # Determine gate result from actual pool record fields
            bt = record.get("backtest", {}) or {}
            sharpe = bt.get("sharpe")
            dsr = record.get("dsr")
            degraded_qualified = record.get("degradedQualified", False)
            passed = bool(
                degraded_qualified
                or (dsr is not None and dsr >= 0.95)
            )
            gate = "PASS" if passed else scorecard.get("gate", "FAIL")

            self.kb.record_strategy_stat(
                category=category,
                gate_result=gate,
                operator_skeleton=skeleton,
            )
            for operator in _extract_operators(expression):
                self.kb.record_operator_stat(
                    operator=operator,
                    category=category,
                    passed=passed,
                )

            # Record passing alphas as positive examples so future runs benefit
            alpha_id = bt.get("alpha_id") or candidate.get("id", "")
            if passed and expression and alpha_id and sharpe is not None and sharpe >= 0.5:
                self.kb.upsert_example(
                    item_id=f"distilled_{alpha_id}",
                    expression=expression,
                    category=category,
                    hypothesis=hypothesis or f"Passed quality gate with IS Sharpe {sharpe:.3f}.",
                    is_negative_example=False,
                    metadata={"sharpe": sharpe, "source": "distilled", "dsr": dsr},
                )

            # Record turnover failures directly (no LLM needed)
            if not passed and expression:
                turnover = bt.get("turnover")
                if turnover is not None and (turnover < 0.01 or turnover > 0.70):
                    direction = "too low" if turnover < 0.01 else "too high"
                    fix = (
                        "Add ts_mean(signal, 21) smoothing or use ts_rank(x, 60) to reduce turnover."
                        if turnover > 0.70
                        else "Shorten smoothing windows or use ts_delta with smaller lookback to increase turnover."
                    )
                    self.kb.record_failure_pattern(
                        reason=f"TURNOVER_{direction.upper().replace(' ', '_')}:{turnover:.3f}",
                        expression=expression,
                        suggested_fix=fix,
                    )

        if self.router is None:
            return

        failed = [
            record
            for record in records
            if not (record.get("degradedQualified") or (record.get("dsr") or 0) >= 0.95)
        ]
        if not failed:
            return

        provider = self.router.pick("distill")

        def _turnover_tag(record):
            bt = record.get("backtest") or {}
            t = bt.get("turnover")
            if t is None:
                return None
            if t < 0.01:
                return f"LOW_TURNOVER({t:.3f})"
            if t > 0.70:
                return f"HIGH_TURNOVER({t:.3f})"
            return None

        prompt_data = {
            "task": "extract_failure_patterns",
            "failed_candidates": [
                {
                    "expression": record.get("candidate", {}).get("expression", ""),
                    "reason": record.get("scorecard", {}).get("reason", "UNKNOWN"),
                    "turnover_issue": _turnover_tag(record),
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
