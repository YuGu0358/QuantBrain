from __future__ import annotations


CATEGORIES = [
    "QUALITY",
    "MOMENTUM",
    "REVERSAL",
    "LIQUIDITY",
    "VOLATILITY",
    "MICROSTRUCTURE",
    "SENTIMENT",
]


class IdeaOptimizer:
    def __init__(self, router=None, kb=None):
        self.router = router
        self.kb = kb

    def optimize(self, raw_idea: str) -> dict:
        failure_patterns = []
        strategy_stats = {}
        if self.kb:
            failure_patterns = self.kb.get_failure_patterns(limit=5)
            strategy_stats = {cat: self.kb.get_strategy_stats(cat) for cat in CATEGORIES}

        system = (
            "You are a WorldQuant BRAIN alpha research specialist. Convert the user idea into a "
            "structured research direction. Available categories: QUALITY,MOMENTUM,REVERSAL,"
            'LIQUIDITY,VOLATILITY,MICROSTRUCTURE,SENTIMENT. Return strict JSON: {"objective":str,'
            '"category":str,"hypothesis":str,"constraints":[str],"suggested_data_fields":[str]}'
        )
        user_content = {
            "idea": raw_idea,
            "known_failure_patterns": [
                p["reason"] + ": " + p.get("suggested_fix", "") for p in failure_patterns
            ],
            "category_win_rates": {
                cat: stats.get("win_rate", 0.5)
                for cat, stats in strategy_stats.items()
                if stats
            },
        }

        if self.router is None:
            return {
                "objective": raw_idea,
                "category": "QUALITY",
                "hypothesis": raw_idea,
                "constraints": [],
                "suggested_data_fields": [],
            }

        provider = self.router.pick("idea")
        import json
        import os
        import time

        t0 = time.time()
        try:
            if provider.client_type == "anthropic":
                from anthropic import Anthropic

                client = Anthropic(api_key=os.environ.get(provider.api_key_env, ""))
                resp = client.messages.create(
                    model=provider.model_id,
                    max_tokens=600,
                    system=system,
                    messages=[
                        {
                            "role": "user",
                            "content": json.dumps(user_content, ensure_ascii=False),
                        }
                    ],
                )
                raw = resp.content[0].text if resp.content else "{}"
                ti = resp.usage.input_tokens if resp.usage else 0
                to = resp.usage.output_tokens if resp.usage else 0
            else:
                from openai import OpenAI

                client = OpenAI(
                    api_key=os.environ.get(provider.api_key_env, ""),
                    base_url=provider.api_base or None,
                )
                resp = client.chat.completions.create(
                    model=provider.model_id,
                    max_tokens=600,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": json.dumps(user_content)},
                    ],
                )
                raw = resp.choices[0].message.content or "{}"
                ti = resp.usage.prompt_tokens if resp.usage else 0
                to = resp.usage.completion_tokens if resp.usage else 0
            result = json.loads(raw)
            self.router.record_result(
                provider.name,
                "idea",
                True,
                (time.time() - t0) * 1000,
                ti,
                to,
            )
            return result
        except Exception:
            try:
                self.router.record_result(
                    provider.name,
                    "idea",
                    False,
                    (time.time() - t0) * 1000,
                    0,
                    0,
                )
            except Exception:
                pass
            return {
                "objective": raw_idea,
                "category": "QUALITY",
                "hypothesis": raw_idea,
                "constraints": [],
                "suggested_data_fields": [],
            }
