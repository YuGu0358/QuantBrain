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
from alpha_miner.modules.m_diagnoser import DiagnosisReport
from alpha_miner.modules.m_repair_memory import RepairMemory
from alpha_miner.modules.m_retriever import Retriever
from alpha_miner.modules.m_planner import Planner, RepairPlan
from alpha_miner.modules.m_scheduler import BanditScheduler


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
        model: str = "gpt-4o-mini",
        temperature: float = 0.4,
        top_p: float = 0.9,
        max_tokens: int = 800,
        seed: int = 42,
        router=None,
        use_llm: bool = False,
        repair_memory: RepairMemory | None = None,
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
        self.repair_memory = repair_memory
        self.retriever: Retriever | None = None
        self.planner: Planner | None = None
        self.scheduler: BanditScheduler | None = None

    def sample_underweight_category(self, existing_counts: dict[str, int] | None = None) -> str:
        counts = existing_counts or {}
        categories = sorted(self.taxonomy)
        if not categories:
            return "QUALITY"
        return min(categories, key=lambda item: (counts.get(item, 0), item))

    def generate_batch(
        self,
        objective: str,
        category: str,
        n: int = 10,
        use_llm: bool = False,
        repair_context: dict | None = None,
        diagnosis: DiagnosisReport | None = None,
    ) -> list[Candidate]:
        request_payload = self._request_payload(objective, category, n, repair_context=repair_context, diagnosis=diagnosis)
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

    def _get_field_semantics(self) -> dict[str, str]:
        return {
            "close": "daily closing price — trend anchor, use for price ratios",
            "open": "daily opening price — overnight gap = open/delay(close,1)-1",
            "high": "daily high — resistance proxy, range = high-low",
            "low": "daily low — support proxy, combine with high for volatility",
            "vwap": "volume-weighted avg price — institutional anchor; close/vwap>1 = net buying pressure",
            "returns": "daily log returns — basis for momentum, reversal, volatility signals",
            "volume": "daily shares traded — attention proxy; volume/adv20 = relative activity vs norm",
            "adv20": "20-day avg daily volume — liquidity level; ts_rank(adv20,126) = improving liquidity",
            "assets": "total assets — scale denominator; use as divisor for ROA ratios",
            "cashflow_op": "operating cash flow — cash earnings quality; cashflow_op/assets = cash ROA",
            "operating_income": "EBIT — economic profit; operating_income/assets = ROA profitability",
            "est_eps": "consensus EPS estimate — ts_delta(est_eps,63) = analyst revision momentum",
            "news_sentiment": "daily NLP sentiment score — ts_zscore(news_sentiment,20) = abnormal attention",
            "industry": "GICS sector — use ONLY inside group_rank(expr, industry) for sector neutralization",
            "subindustry": "GICS sub-industry — use ONLY inside group_rank(expr, subindustry) for finer peers",
        }

    def _request_payload(
        self,
        objective: str,
        category: str,
        n: int,
        repair_context: dict | None = None,
        diagnosis: DiagnosisReport | None = None,
    ) -> dict[str, Any]:
        context = self.kb.rag_context(category)
        # Templates for the target category shown as positive examples
        positive_templates = [
            {"expression": expr, "hypothesis": hyp}
            for expr, hyp in _templates_for_category(category)[:3]
        ]

        is_repair = repair_context is not None and bool(repair_context.get("expression"))
        if is_repair:
            parent_expr = repair_context.get("expression", "")
            failed_checks = repair_context.get("failedChecks") or []
            gate_reasons = repair_context.get("gate", {}).get("reasons") or []
            if diagnosis is not None:
                repair_priorities = diagnosis.repair_priorities[:2]
                requirement = (
                    f"REPAIR MODE: Do NOT generate generic alphas. You must produce {n} targeted variants of "
                    "the parent expression below using this structured diagnosis. "
                    f"Primary symptom: {diagnosis.primary_symptom}. "
                    f"Repair priorities: {json.dumps(repair_priorities, ensure_ascii=False)}. "
                    f"Do not change: {', '.join(diagnosis.do_not_change) if diagnosis.do_not_change else 'none'}. "
                    "Each variant must preserve the core economic thesis but structurally address the diagnosed failures. "
                    f"Parent expression: {parent_expr}. "
                    f"Failed checks: {', '.join(failed_checks)}. "
                    f"Gate reasons: {'; '.join(gate_reasons[:3])}. "
                    + _repair_instructions(failed_checks)
                )
            else:
                requirement = (
                    f"REPAIR MODE: Do NOT generate generic alphas. You must produce {n} targeted variants of "
                    f"the parent expression below that fix the specific failures listed. "
                    f"Each variant must preserve the core economic thesis but structurally address the failures. "
                    f"Parent expression: {parent_expr}. "
                    f"Failed checks: {', '.join(failed_checks)}. "
                    f"Gate reasons: {'; '.join(gate_reasons[:3])}. "
                    + _repair_instructions(failed_checks)
                )
            user_dict: dict[str, Any] = {
                "mode": "repair",
                "parent_expression": parent_expr,
                "failed_checks": failed_checks,
                "gate_reasons": gate_reasons,
                "requirement": requirement,
            }
            if diagnosis is not None:
                symptoms = [diagnosis.primary_symptom, *diagnosis.secondary_symptoms]
                user_dict["diagnosis"] = {
                    "primary_symptom": diagnosis.primary_symptom,
                    "repair_priorities": repair_priorities,
                    "do_not_change": diagnosis.do_not_change,
                }
                if self.retriever is not None:
                    retrieval = self.retriever.retrieve(
                        diagnosis,
                        parent_expr,
                        family_tag=repair_context.get("_category"),
                    )
                    forbidden_directions = _dedupe_strings(_string_list(retrieval.get("forbidden_directions", [])))
                    recommended_directions = _dedupe_strings(_string_list(retrieval.get("recommended_directions", [])))
                    if retrieval.get("family_saturated"):
                        requirement = (
                            requirement
                            + " IMPORTANT: This alpha family is saturated — generate candidates from a DIFFERENT signal category."
                        )
                        user_dict["requirement"] = requirement
                    if forbidden_directions:
                        user_dict["forbidden_directions"] = forbidden_directions
                    if recommended_directions:
                        user_dict["recommended_directions"] = recommended_directions
                    if retrieval.get("retrieval_summary"):
                        user_dict["retrieval_summary"] = str(retrieval["retrieval_summary"])
                    # Planner: decide candidate mix and update requirement
                    if self.planner is not None:
                        sched_weights = self.scheduler.get_weights() if self.scheduler else None
                        plan = self.planner.plan(diagnosis, retrieval, total_budget=n, scheduler_weights=sched_weights)
                        mix = plan.candidate_mix
                        requirement = (
                            f"Generate exactly {n} repair candidates with this mix: "
                            f"{mix.get('param_tune',0)} param_tune (only adjust window/lag/threshold params), "
                            f"{mix.get('struct_mutation',0)} struct_mutation (replace subtrees/operators), "
                            f"{mix.get('template_retrieval',0)} template_retrieval (from positive memory), "
                            f"{mix.get('llm_mutation',0)} llm_mutation (creative LLM rewrite). "
                            "For each candidate add the action_type string to origin_refs list."
                        )
                        user_dict["requirement"] = requirement
                        user_dict["repair_plan"] = {
                            "candidate_mix": mix,
                            "prioritized_actions": plan.prioritized_actions,
                            "hard_constraints": plan.hard_constraints,
                        }
                else:
                    raw_forbidden = _string_list(diagnosis.raw.get("forbidden_directions", [])) if isinstance(diagnosis.raw, dict) else []
                    memory_forbidden = self.repair_memory.get_forbidden_for_symptoms(symptoms) if self.repair_memory else []
                    positive_examples = self.repair_memory.get_positive_for_symptoms([diagnosis.primary_symptom]) if self.repair_memory else []
                    forbidden_directions = _dedupe_strings([*raw_forbidden, *memory_forbidden])
                    if forbidden_directions:
                        user_dict["forbidden_directions"] = forbidden_directions
                    if positive_examples:
                        user_dict["positive_memory_examples"] = positive_examples
        else:
            requirement = (
                f"Generate {n} diverse alphas for the {category} category. "
                "Each MUST combine data from at least 2 distinct field types "
                "(fundamentals/price/volume/sentiment). Prefer industry-neutralized expressions."
            )
            user_dict = {}

        user_dict.update({
            "objective": objective,
            "category": category,
            "n": n,
            "requirement": requirement,
            "allowed_operators": sorted(self._get_operators()),
            "allowed_fields": sorted(self._get_fields()),
            "field_semantics": self._get_field_semantics(),
            "positive_multi_variable_examples": positive_templates,
            "positive_context": context.positive,
            "negative_wq101_context": context.negative,
            "failure_patterns_to_avoid": [p["reason"] for p in getattr(context, "failure_patterns", [])],
        })
        user_content = json.dumps(user_dict, ensure_ascii=False)
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

    @staticmethod
    def _is_reasoning_model(model_id: str) -> bool:
        return model_id.startswith(("o1", "o3", "o4", "gpt-5"))

    def _call_provider(self, provider, request_payload: dict[str, Any]) -> tuple[str, int, int, float]:
        t0 = time.time()
        if provider.client_type == "openai_compat":
            from openai import OpenAI

            client = OpenAI(api_key=os.environ.get(provider.api_key_env, ""), base_url=provider.api_base or None)
            kwargs: dict[str, Any] = {
                "model": provider.model_id,
                "messages": request_payload["messages"],
            }
            if self._is_reasoning_model(provider.model_id):
                kwargs["max_completion_tokens"] = request_payload.get("max_tokens", 2000)
            else:
                kwargs["temperature"] = request_payload.get("temperature", 0.4)
                kwargs["max_tokens"] = request_payload.get("max_tokens", 800)
            resp = client.chat.completions.create(**kwargs)
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
        # quality × momentum
        ("group_rank(ts_rank(operating_income / assets, 252) + ts_rank(ts_mean(returns, 63), 63), industry)",
         "Persistent profitability (year-long ROA rank) combined with industry-relative price momentum. High-quality companies with confirming medium-term price action."),
        # dual cash quality + improvement
        ("rank(ts_rank(cashflow_op / assets, 252) + ts_rank(operating_income / assets, 252) + ts_delta(operating_income / assets, 63))",
         "Triple quality signal: persistent cash-flow quality, persistent earnings quality, and recent improvement in ROA. All three pointing same direction minimizes noise."),
        # quality × low-vol
        ("group_rank(ts_rank(operating_income / assets, 252) - ts_rank(ts_std_dev(returns, 60), 252), industry)",
         "High and persistent profitability combined with low volatility premium within sector. Quality-low-vol combination has historically been less crowded than each alone."),
        # improving quality + analyst confirmation
        ("rank(ts_delta(cashflow_op / assets, 126) + ts_delta(est_eps, 63))",
         "Improving cash ROA over two quarters AND concurrent analyst EPS estimate upgrades. Fundamental improvement confirmed by analyst consensus shift."),
        # quality × sentiment × neutralized
        ("group_rank(ts_rank(operating_income / assets, 252) + ts_zscore(news_sentiment, 20), industry)",
         "Long-term profitability rank within industry, boosted by abnormal near-term news sentiment. Fundamental anchor prevents chasing sentiment noise."),
    ],
    "MOMENTUM": [
        # volume-confirmed momentum
        ("group_rank(ts_mean(returns, 63) * ts_rank(volume / adv20, 60), industry)",
         "Industry-relative 3-month return momentum multiplied by relative volume rank. Volume expansion confirming a trend reduces false breakout noise."),
        # triple-confirmation momentum
        ("rank(ts_mean(returns, 63) + ts_zscore(news_sentiment, 20) + ts_delta(est_eps, 21))",
         "Price momentum, abnormal news sentiment, and analyst EPS revision all aligned. Three independent signals pointing the same direction dramatically increases conviction."),
        # momentum × fundamental quality filter
        ("rank(ts_mean(returns, 63) * sign(ts_delta(operating_income / assets, 63)))",
         "Medium-term momentum only in stocks where fundamentals are actually improving. Eliminates momentum in deteriorating businesses that tend to crash hard."),
        # medium vs long momentum + volume
        ("group_rank(ts_mean(returns, 63) - ts_mean(returns, 252) + ts_rank(volume / adv20, 60), industry)",
         "Medium-minus-long momentum spread (captures acceleration) plus volume participation relative to sector. Sector-neutralized to avoid industry beta contamination."),
        # residual momentum post-reversal hedge
        ("rank(ts_mean(returns, 63) - ts_mean(returns, 5) + ts_delta(est_eps, 63))",
         "Medium-term momentum hedged against short-term reversal, with analyst upgrade support. Reduces whipsaw from overextended short-term moves while keeping earnings-backed trend."),
    ],
    "REVERSAL": [
        # capitulation reversal with volume
        ("rank(-ts_mean(returns, 5) * ts_rank(volume / adv20, 20))",
         "Weekly reversal signal amplified by high relative volume. High volume down-moves suggest capitulation and mean-reversion entry; low-volume moves are less reliable."),
        # Z-score reversal with quality filter
        ("group_rank(-ts_zscore(returns, 20) + ts_rank(operating_income / assets, 126), industry)",
         "Statistical return overreaction (Z-score extremes revert) combined with underlying quality rank. Buying oversold high-quality companies within sector is historically robust."),
        # reversal × sentiment normalization
        ("rank(-ts_mean(returns, 5) - ts_zscore(news_sentiment, 5))",
         "Short-term price reversal combined with reversal of abnormally negative news sentiment. Both price and news overreactions tend to revert to mean together."),
        # overnight gap reversal
        ("rank(-ts_mean(open / delay(close, 1) - 1, 10) + ts_mean(returns, 5))",
         "Mean-reversion of persistent negative overnight gaps, combined with intraday momentum. Overnight weakness that doesn't propagate intraday is typically overpriced."),
    ],
    "LIQUIDITY": [
        # liquidity trend × fundamental quality
        ("group_rank(ts_rank(adv20, 126) + ts_rank(operating_income / assets, 252), industry)",
         "Rising liquidity trend over half year combined with profitability rank within sector. Improving institutional interest (volume) in fundamentally strong companies."),
        # volume stability × momentum direction
        ("rank(-ts_std_dev(volume / adv20, 20) * sign(ts_mean(returns, 21)))",
         "Stable volume participation (low noise) in the direction of momentum. Consistent buying interest without volatility spikes signals cleaner institutional accumulation."),
        # Amihud-proxy × reversal
        ("group_rank(-ts_mean(abs(returns) / volume, 20) + ts_rank(adv20, 63), industry)",
         "Low price-impact (Amihud illiquidity proxy inverted) combined with improving liquidity trend. Stocks becoming more liquid while showing low friction command liquidity premium."),
    ],
    "VOLATILITY": [
        # low-vol × quality
        ("group_rank(-ts_std_dev(returns, 60) + ts_rank(cashflow_op / assets, 252), industry)",
         "Low idiosyncratic volatility within sector combined with cash flow quality rank. Stable cash earnings businesses with low return noise: two independent risk-reduction signals."),
        # vol-of-vol × returns
        ("rank(-ts_std_dev(ts_std_dev(returns, 20), 20) + ts_mean(returns, 63))",
         "Low second-order volatility (vol-of-vol) combined with positive medium return. Stocks with stable, predictable risk profile and positive drift are less crowded than pure low-vol."),
        # declining volatility trend with sentiment
        ("group_rank(ts_delta(-ts_std_dev(returns, 20), 20) + ts_zscore(news_sentiment, 20), industry)",
         "Improving (declining) volatility trend combined with positive news sentiment within sector. Calm-down + positive news: risk-off positioning unwind signal."),
    ],
    "MICROSTRUCTURE": [
        # VWAP × volume × fundamentals
        ("rank(ts_mean(close / vwap - 1, 10) * ts_rank(volume / adv20, 20))",
         "Sustained price strength above institutional VWAP anchor, multiplied by elevated relative volume. Institutional buying pushes both close/VWAP ratio and volume above normal simultaneously."),
        # price-volume correlation × EPS
        ("group_rank(ts_corr(close / vwap - 1, volume / adv20, 20) + ts_delta(est_eps, 63), industry)",
         "Stocks where VWAP premium correlates with volume surge (demand-supply alignment) AND analysts are raising EPS estimates. Microstructure confirming fundamental upgrade."),
        # open-close intraday momentum × news
        ("rank(ts_mean(close / open - 1, 10) + ts_zscore(news_sentiment, 20))",
         "Persistent intraday upward drift (open-to-close return) combined with abnormal positive sentiment. Intraday buying pressure sustained over weeks with news catalyst."),
        # high-low range contraction + volume surge
        ("rank(-ts_mean((high - low) / vwap, 10) * ts_rank(volume / adv20, 20))",
         "Contracting daily price range (consolidation) with expanding relative volume: classic pre-breakout microstructure. Tightening range + volume pickup signals potential directional move."),
    ],
    "SENTIMENT": [
        # EPS revision × price momentum × fundamental anchor
        ("group_rank(ts_delta(est_eps, 63) + ts_mean(returns, 63) + ts_rank(operating_income / assets, 126), industry)",
         "EPS estimate upgrades, confirming price momentum, on companies with solid existing profitability. All three pillars aligned sector-neutral reduces false positives from speculative upward revisions."),
        # abnormal sentiment × volume × reversal hedge
        ("rank(ts_zscore(news_sentiment, 20) + ts_rank(volume / adv20, 20) - ts_mean(returns, 5))",
         "Abnormal positive news sentiment combined with high relative volume, hedged by short-term reversal. News + volume confirms signal; subtracting recent return avoids chasing spike."),
        # analyst divergence × price
        ("rank(ts_delta(est_eps, 21) * sign(ts_mean(returns, 21)))",
         "EPS estimate revision in the same direction as recent price: analyst and market confirming each other. Cross-confirmation reduces noise from analyst upgrades that market already discounts."),
        # sentiment persistence × liquidity × quality
        ("group_rank(ts_mean(news_sentiment, 21) + ts_rank(adv20, 63) + ts_rank(cashflow_op / assets, 252), industry)",
         "Sustained positive news coverage over a month, rising liquidity, and fundamental cash quality within sector. Three-pillar signal requiring all dimensions positive simultaneously."),
    ],
}


def _templates_for_category(category: str) -> list[tuple[str, str]]:
    return _ALL_TEMPLATES.get(category, _ALL_TEMPLATES["QUALITY"])


def _repair_instructions(failed_checks: list[str]) -> str:
    checks = {c.upper() for c in failed_checks}
    parts = []
    if checks & {"SHARPE", "LOW_SHARPE", "FITNESS", "LOW_FITNESS", "DSR"}:
        parts.append(
            "LOW_SHARPE/FITNESS fix: add industry neutralization via group_rank(expr, industry); "
            "blend a complementary signal (e.g., add ts_rank(cashflow_op/assets,252) or ts_delta(est_eps,63)); "
            "try a different category (QUALITY→MOMENTUM or vice versa)."
        )
    if checks & {"HIGH_TURNOVER", "TURNOVER"}:
        parts.append(
            "HIGH_TURNOVER fix: replace ts_delta(x,d<10) with ts_mean(x,21) or ts_rank(x,60); "
            "wrap volatile sub-expressions in ts_mean(x,10); use delay(x,1) on fast signals."
        )
    if checks & {"LOW_TURNOVER"}:
        parts.append(
            "LOW_TURNOVER fix: use ts_delta(x,5) instead of ts_mean; "
            "add a faster signal component; shorten the rank window to ≤20 days."
        )
    if checks & {"SELF_CORRELATION", "ORTHOGONALITY"}:
        parts.append(
            "SELF_CORRELATION fix: you MUST materially change the signal structure — "
            "switch the primary data source (e.g., from price to fundamentals); "
            "change the time horizon (short→long or vice versa); "
            "use regression_neut(expr, existing_signal) to orthogonalize."
        )
    if checks & {"LOW_SUB_UNIVERSE_SHARPE"}:
        parts.append(
            "LOW_SUB_UNIVERSE_SHARPE fix: use group_rank(expr, subindustry) for finer peer comparison; "
            "combine with a fundamental ratio to filter low-quality sub-universe stocks."
        )
    if checks & {"CONCENTRATED_WEIGHT"}:
        parts.append(
            "CONCENTRATED_WEIGHT fix: apply rank() or group_rank() at the outermost level; "
            "add winsorize() to cap extreme values; combine 2-3 signals additively."
        )
    return " ".join(parts)


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _dedupe_strings(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))
