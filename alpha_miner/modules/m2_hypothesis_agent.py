from __future__ import annotations

import os, json, re, time
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .common import PACKAGE_ROOT, read_json
from .llm_cache import LLMCache
from .m1_knowledge_base import KnowledgeBase


@dataclass(frozen=True)
class Candidate:
    id: str
    category: str
    hypothesis: str
    expression: str
    origin_refs: list[str]
    opt_rounds: int = 0


class HypothesisAgent:
    def __init__(
        self,
        kb: KnowledgeBase,
        cache: LLMCache,
        taxonomy: dict[str, Any],
        model: str = "gpt-5.4-mini",
        temperature: float = 0.4,
        top_p: float = 0.9,
        max_tokens: int = 800,
        seed: int = 42,
        router=None,
        use_llm: bool = False,
    ):
        self.kb = kb
        self.cache = cache
        self.taxonomy = taxonomy.get("categories", taxonomy)
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.random = random.Random(seed)
        self.router = router
        self.use_llm = use_llm

    def sample_underweight_category(self, existing_counts: dict[str, int] | None = None) -> str:
        counts = existing_counts or {}
        categories = sorted(self.taxonomy)
        if not categories:
            return "QUALITY"
        return min(categories, key=lambda item: (counts.get(item, 0), item))

    def generate_batch(self, objective: str, category: str, n: int = 10, use_llm: bool = False) -> list[Candidate]:
        request_payload = self._request_payload(objective, category, n)
        cached = self.cache.get(request_payload)
        if cached.hit and cached.payload:
            response = cached.payload["response"]
        elif (self.use_llm or use_llm) and self.router:
            payload1 = request_payload
            payload2 = {**payload1, "seed": 43}
            raw_candidates: list[Any] = []
            with ThreadPoolExecutor(max_workers=2) as executor:
                for result in executor.map(self._call_llm, [payload1, payload2]):
                    raw_candidates.extend(result.get("candidates", []))

            candidate_pool: list[Candidate] = []
            unique_candidates: list[Candidate] = []
            seen_expressions: set[str] = set()
            for item in raw_candidates:
                candidate = item if isinstance(item, Candidate) else Candidate(**item)
                candidate_pool.append(candidate)
                if candidate.expression in seen_expressions:
                    continue
                seen_expressions.add(candidate.expression)
                unique_candidates.append(candidate)

            judge_pool = unique_candidates if len(unique_candidates) > n else candidate_pool
            judged = self._judge_candidates(judge_pool, n, self.router)
            selected: list[Candidate] = []
            selected_expressions: set[str] = set()
            for candidate in [*judged, *unique_candidates]:
                if candidate.expression in selected_expressions:
                    continue
                selected_expressions.add(candidate.expression)
                selected.append(candidate)
                if len(selected) >= n:
                    break

            if not selected:
                selected = self._deterministic_candidates(objective, category, n)
            response = {"candidates": [asdict(item) for item in selected]}
            self.cache.put(request_payload, response)
        else:
            response = {"candidates": [asdict(item) for item in self._deterministic_candidates(objective, category, n)]}
            self.cache.put(request_payload, response)

        return [Candidate(**item) for item in response.get("candidates", [])][:n]

    def _get_operators(self) -> list[str]:
        return [
            "abs", "correlation", "covariance", "delay", "delta", "divide", "group_neutralize",
            "group_rank", "hump", "log", "max", "min", "multiply", "power", "rank", "regression_neut",
            "scale", "sign", "ts_arg_max", "ts_arg_min", "ts_backfill", "ts_corr", "ts_covariance",
            "ts_decay_linear", "ts_delta", "ts_mean", "ts_rank", "ts_std_dev", "ts_sum", "ts_zscore",
            "winsorize", "zscore",
        ]

    def _get_fields(self) -> list[str]:
        return [
            "adv20", "assets", "cashflow_op", "close", "est_eps", "high", "industry", "low",
            "news_sentiment", "open", "operating_income", "returns", "subindustry", "volume", "vwap",
        ]

    def _request_payload(self, objective: str, category: str, n: int) -> dict[str, Any]:
        context = self.kb.rag_context(category)
        user_content = json.dumps({
            "objective": objective,
            "category": category,
            "n": n,
            "allowed_operators": sorted(self._get_operators()),
            "allowed_fields": sorted(self._get_fields()),
            "positive_context": context.positive,
            "negative_wq101_context": context.negative,
            "failure_patterns_to_avoid": [p["reason"] for p in getattr(context, "failure_patterns", [])],
        }, ensure_ascii=False)
        return {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (PACKAGE_ROOT / "prompts" / "system_generation.txt").read_text(encoding="utf-8"),
                },
                {
                    "role": "user",
                    "content": user_content,
                },
            ],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "response_format": "strict_json_candidates_v1",
            "seed": 42,
        }

    @staticmethod
    def _extract_json(raw: str) -> dict[str, Any]:
        """Parse JSON from LLM output, stripping markdown code fences if present."""
        text = raw.strip()
        # Strip ```json ... ``` or ``` ... ``` wrappers
        match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
        if match:
            text = match.group(1)
        # Fallback: find first { ... } block in case of leading explanation text
        elif not text.startswith("{"):
            brace = text.find("{")
            if brace != -1:
                text = text[brace:]
        return json.loads(text)

    def _call_llm(self, request_payload: dict[str, Any]) -> dict[str, Any]:
        if self.router is None:
            raise NotImplementedError("No router")
        role = request_payload.get("_llm_role", "generate")
        try:
            provider = request_payload.get("_provider") or self.router.pick(role)
            raw, ti, to, latency = self._call_provider(provider, request_payload)
            if request_payload.get("response_format") == "strict_json_selected_indices_v1":
                selected_indices = self._extract_json(raw).get("selected_indices", []) if raw else []
                self.router.record_result(provider.name, role, len(selected_indices) > 0, latency, ti, to)
                return {"selected_indices": selected_indices}
            parsed = self._extract_json(raw) if raw else {}
            candidates = parsed.get("candidates", [])
            self.router.record_result(provider.name, role, len(candidates) > 0, latency, ti, to)
            if not candidates:
                print(f"[llm] {provider.name}/{role} returned 0 candidates. raw[:200]={raw[:200]!r}", flush=True)
            return {"candidates": candidates}
        except Exception as exc:
            print(f"[llm] {role} call failed: {exc}. raw[:200]={locals().get('raw','')[:200]!r}", flush=True)
            if request_payload.get("response_format") == "strict_json_selected_indices_v1":
                return {"selected_indices": []}
            return {"candidates": []}

    def _judge_candidates(self, candidates: list[Candidate], n: int, router) -> list[Candidate]:
        if not router or len(candidates) <= n:
            return candidates[:n]
        try:
            provider = router.pick("judge")
            sys = 'Select best n alpha candidates. Return JSON {"selected_indices":[list of ints]}'
            user = json.dumps({"n": n, "candidates": [{"index": i, "expression": c.expression} for i, c in enumerate(candidates)]})
            request_payload = {
                "_llm_role": "judge",
                "_provider": provider,
                "messages": [
                    {"role": "system", "content": sys},
                    {"role": "user", "content": user},
                ],
                "temperature": 0.0,
                "max_tokens": self.max_tokens,
                "response_format": "strict_json_selected_indices_v1",
            }
            indices = self._call_llm(request_payload).get("selected_indices", [])
            if not indices:
                return candidates[:n]
            return [candidates[int(i)] for i in indices if 0 <= int(i) < len(candidates)][:n]
        except Exception:
            return candidates[:n]

    def _call_provider(self, provider, request_payload: dict[str, Any]) -> tuple[str, int, int, float]:
        t0 = time.time()
        if provider.client_type == "openai_compat":
            from openai import OpenAI

            client = OpenAI(api_key=os.environ.get(provider.api_key_env, ""), base_url=provider.api_base or None)
            resp = client.chat.completions.create(
                model=provider.model_id,
                messages=request_payload["messages"],
                temperature=request_payload.get("temperature", 0.4),
                max_tokens=request_payload.get("max_tokens", 800),
            )
            raw = resp.choices[0].message.content or ""
            ti = resp.usage.prompt_tokens if resp.usage else 0
            to = resp.usage.completion_tokens if resp.usage else 0
        elif provider.client_type == "anthropic":
            from anthropic import Anthropic

            client = Anthropic(api_key=os.environ.get(provider.api_key_env, ""))
            msgs = [m for m in request_payload["messages"] if m["role"] != "system"]
            sys = next((m["content"] for m in request_payload["messages"] if m["role"] == "system"), "")
            resp = client.messages.create(
                model=provider.model_id,
                max_tokens=request_payload.get("max_tokens", 800),
                system=sys,
                messages=msgs,
            )
            raw = resp.content[0].text if resp.content else ""
            ti = resp.usage.input_tokens if resp.usage else 0
            to = resp.usage.output_tokens if resp.usage else 0
        else:
            raise ValueError(f"Unsupported provider client_type: {provider.client_type}")
        latency = (time.time() - t0) * 1000
        return raw, ti, to, latency

    def _deterministic_candidates(self, objective: str, category: str, n: int) -> list[Candidate]:
        # Collect templates from target category first, then others to ensure uniqueness
        primary = _templates_for_category(category)
        all_templates: list[tuple[str, str, str]] = [(expr, hyp, category) for expr, hyp in primary]
        for cat, pairs in _ALL_TEMPLATES.items():
            if cat != category:
                for expr, hyp in pairs:
                    all_templates.append((expr, hyp, cat))
        seen: set[str] = set()
        candidates = []
        for expr, hyp, cat in all_templates:
            if expr in seen:
                continue
            seen.add(expr)
            candidates.append(
                Candidate(
                    id=f"{cat.lower()}_{len(candidates):03d}",
                    category=cat,
                    hypothesis=f"{hyp} Objective: {objective}",
                    expression=expr,
                    origin_refs=["taxonomy", "wq101_negative_examples"],
                )
            )
            if len(candidates) >= n:
                break
        return candidates


_ALL_TEMPLATES: dict[str, list[tuple[str, str]]] = {
    "QUALITY": [
        ("group_rank(ts_rank(operating_income / assets, 252), industry)", "Peer-relative profitability rank within industry."),
        ("rank(ts_rank(cashflow_op / assets, 252) + ts_rank(operating_income / assets, 252))", "Dual quality: cash-flow-backed operating income rank."),
        ("group_rank(rank(cashflow_op / assets) - rank(ts_mean(cashflow_op / assets, 60)), industry)", "Improving cash generation vs trailing average, industry-neutral."),
        ("rank(ts_delta(operating_income / assets, 63))", "Quarterly improvement in asset profitability as momentum signal."),
    ],
    "MOMENTUM": [
        ("rank(ts_mean(returns, 21) - ts_mean(returns, 252))", "Short-term momentum minus long-term drift."),
        ("group_rank(ts_mean(returns, 63), industry) - group_rank(ts_mean(returns, 252), industry)", "Industry-relative medium vs long momentum spread."),
        ("rank(ts_corr(rank(returns), rank(volume), 60))", "Correlation of return rank and volume rank over 60 days."),
    ],
    "REVERSAL": [
        ("rank(-ts_mean(returns, 5))", "Short-term weekly return reversal."),
        ("rank(ts_delta(volume, 20)) * rank(-returns)", "Volume spike with price reversal signal."),
        ("rank(-ts_zscore(returns, 20))", "Z-score reversal: extreme moves revert over 20 days."),
    ],
    "LIQUIDITY": [
        ("rank(ts_mean(adv20, 60) / adv20)", "Relative liquidity trend: improving availability."),
        ("group_rank(-ts_std_dev(volume / adv20, 20), industry)", "Stable volume relative to peers signals lower friction."),
    ],
    "VOLATILITY": [
        ("rank(-ts_std_dev(returns, 60))", "Low-volatility premium over 60-day window."),
        ("rank(-ts_std_dev(returns, 20) + ts_mean(returns, 20))", "Low vol combined with positive short return."),
        ("group_rank(-ts_std_dev(returns, 60), industry)", "Industry-relative low volatility."),
    ],
    "MICROSTRUCTURE": [
        ("rank(close / vwap - 1)", "Close-to-VWAP ratio: buying pressure indicator."),
        ("rank(ts_mean(close / vwap - 1, 10))", "10-day persistent VWAP premium signal."),
        ("rank(ts_corr(returns, ts_delta(volume, 1), 20))", "Price-volume correlation over 20 days."),
    ],
    "SENTIMENT": [
        ("rank(ts_zscore(news_sentiment, 20))", "Abnormal sentiment Z-score over 20 days."),
        ("group_rank(ts_mean(news_sentiment, 5), industry)", "Relative near-term sentiment within industry."),
        ("rank(ts_delta(est_eps, 63))", "Quarterly change in EPS estimates as analyst sentiment."),
    ],
}


def _templates_for_category(category: str) -> list[tuple[str, str]]:
    return _ALL_TEMPLATES.get(category, _ALL_TEMPLATES["QUALITY"])
