"""LangChain Tool-calling Agent for alpha factor repair.

The agent autonomously decides which tools to call and in what order:
  diagnose_alpha → analyze_factor_logic → retrieve_logic_patterns →
  retrieve_repair_memory → generate_repair_variants → validate_expression
"""
from __future__ import annotations

import json
import re
import time
from typing import Any

from alpha_miner.modules.m_repair_memory import RepairMemory
from alpha_miner.modules.m_repair_intelligence import (
    analyze_math_profile,
    build_recursive_repair_guidance,
    compare_repair,
    derive_symptom_tags,
    infer_economic_profile,
    score_repair_outcome,
)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a WorldQuant BRAIN alpha factor repair agent.

Your goal: produce THE SINGLE BEST repair for a failing alpha expression.
Do not generate multiple options — think carefully and commit to the one repair
you are most confident will pass all quality checks.

PRIORITY ORDER: (1) Maximize IS Sharpe  (2) Maximize Fitness  (3) Keep turnover 3–70%
Do NOT blindly reduce turnover — a strong signal with 40% turnover beats a weak signal with 8%.

BRAIN OPERATORS:
ts_mean(x,d), ts_rank(x,d), ts_std_dev(x,d), ts_delta(x,d), ts_zscore(x,d),
ts_decay_linear(x,d), ts_corr(x,y,d), delay(x,d), rank(x), zscore(x),
group_rank(x, industry), group_rank(x, subindustry),
divide(x,y), multiply(x,y), abs(x), log(x), power(x,n), sign(x), winsorize(x,p)

REPAIR PRINCIPLES:
- low_sharpe: add group_rank sector neutralization, extend lookback (63→126d),
  blend an independent second signal (quality + momentum hybrid), switch category if needed
- low_fitness: Fitness = PnL/TVR. Fix depends on the root cause:
  * If turnover > 65%: THEN reduce it — wrap fastest sub-expression with ts_decay_linear(x, 15)
  * If turnover < 50%: turnover is NOT the problem — signal is too weak or concentrated.
    Add a second independent factor, switch to a fundamentals anchor, or diversify fields.
  * Never over-smooth a signal just to reduce turnover below 20%.
- high_turnover (>70%): ts_decay_linear(x, 10-15) or ts_mean(x, 21) on the fastest sub-expression
- high_corrlib: change primary data field or time horizon entirely; use regression_neut()
- no_daily_pnl / degraded_no_daily_pnl: treat this as a platform-level fragility signal.
  Prefer broader-coverage rank-based blends, transparent economic anchors, and simpler
  structures that should survive degraded evaluation. Avoid fragile micro-tuning.
- recursive repair (depth >= 1): pure window tuning is not enough. Change at least one of:
  primary field family, dominant structure, or peer grouping granularity.

WORKFLOW (follow this order):
1. Call diagnose_alpha to understand the failure
2. Call analyze_factor_logic to extract mathematical structure and economic thesis
3. Call retrieve_logic_patterns to learn which math/economic repair deltas worked or failed
4. Call retrieve_repair_memory for legacy successful/failed repair directions
5. Call generate_repair_variants to create candidates
6. Call validate_expression for the best candidate
7. Return your final answer as JSON ONLY — no other text:
{
  "diagnosis": {"primary_symptom": "<symptom>", "root_cause": "<cause>"},
  "candidates": [
    {
      "expression": "<valid BRAIN expr>",
      "hypothesis": "<economic logic>",
      "fix_applied": "<what changed>",
      "math_logic": "<mathematical repair rationale>",
      "economic_logic": "<economic thesis preserved or changed>",
      "expected_failure_reduction": "<which failed check should improve and why>"
    }
  ]
}
The "candidates" array must contain EXACTLY ONE entry — your single most confident repair."""


DEFAULT_REPAIR_CHAIN_MODEL = "gpt-5.4-2026-03-05"
_REPAIR_MEMORY_SUCCESS_LIMIT = 2
_REPAIR_MEMORY_FAILURE_LIMIT = 1
_LOGIC_PATTERN_SUCCESS_LIMIT = 2
_LOGIC_PATTERN_FAILURE_LIMIT = 1
_PROMPT_CONTEXT_MAX_LINES = 3
_PROMPT_CONTEXT_MAX_TOTAL_CHARS = 420
_PROMPT_LINE_MAX_CHARS = 140


def _clip_text(value: Any, limit: int = 80) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(limit - 3, 0)].rstrip() + "..."


def _compact_prompt_lines(text: str, max_lines: int = _PROMPT_CONTEXT_MAX_LINES) -> str:
    if not text:
        return "none"
    lines = []
    total_chars = 0
    for raw_line in str(text).splitlines():
        line = _clip_text(raw_line.strip(), _PROMPT_LINE_MAX_CHARS)
        if not line:
            continue
        projected = total_chars + len(line)
        if lines and projected > _PROMPT_CONTEXT_MAX_TOTAL_CHARS:
            break
        lines.append(line)
        total_chars = projected
        if len(lines) >= max_lines:
            break
    return "\n".join(lines) if lines else "none"


def _summarize_math_profile(profile: dict[str, Any] | None) -> str:
    profile = profile or {}
    parts = []
    if profile.get("operators"):
        parts.append(f"ops={','.join(str(item) for item in profile['operators'][:4])}")
    if profile.get("fields"):
        parts.append(f"fields={','.join(str(item) for item in profile['fields'][:3])}")
    if profile.get("windows"):
        parts.append(f"windows={','.join(str(item) for item in profile['windows'][:5])}")
    if profile.get("group_fields"):
        parts.append(f"groups={','.join(str(item) for item in profile['group_fields'][:2])}")
    if profile.get("dominant_structure"):
        parts.append(f"struct={profile['dominant_structure']}")
    complexity = profile.get("complexity")
    if complexity not in (None, ""):
        parts.append(f"complexity={complexity}")
    return "; ".join(parts) if parts else "none"


def _summarize_economic_profile(profile: dict[str, Any] | None) -> str:
    profile = profile or {}
    theme = _clip_text(profile.get("theme"), 32)
    thesis = _clip_text(profile.get("thesis"), 110)
    parts = []
    if theme:
        parts.append(f"theme={theme}")
    if thesis:
        parts.append(f"thesis={thesis}")
    return "; ".join(parts) if parts else "none"


# ---------------------------------------------------------------------------
# Tool factory — returns tools bound to shared state
# ---------------------------------------------------------------------------

def build_tools(memory: RepairMemory, embeddings: Any, validator: Any) -> list:
    """Build LangChain tools with injected dependencies."""
    from langchain_core.tools import tool
    memory.set_embedder(embeddings)
    blocked_operators = sorted(str(item) for item in getattr(validator, "blocked_operators", []) if str(item))

    @tool
    def analyze_factor_logic(expression: str, category: str = "") -> str:
        """Analyze mathematical structure and economic logic of a BRAIN alpha expression.
        Args:
            expression: The BRAIN alpha expression
            category: Optional factor category hint
        Returns JSON with math_profile and economic_profile.
        """
        try:
            math_profile = analyze_math_profile(expression)
            economic_profile = infer_economic_profile(
                expression,
                category=category,
                fields=math_profile.get("fields", []),
            )
            return json.dumps(
                {
                    "math_profile": math_profile,
                    "economic_profile": economic_profile,
                },
                ensure_ascii=False,
            )
        except Exception as exc:
            return json.dumps(
                {
                    "error": str(exc),
                    "math_profile": analyze_math_profile(""),
                    "economic_profile": infer_economic_profile("", category=category),
                },
                ensure_ascii=False,
            )

    @tool
    def diagnose_alpha(expression: str, sharpe: str, fitness: str, turnover: str, failed_checks: str) -> str:
        """Diagnose why a BRAIN alpha expression is failing.
        Args:
            expression: The failing BRAIN alpha expression
            sharpe: IS Sharpe ratio as string (e.g. '0.32')
            fitness: IS Fitness score as string
            turnover: Daily turnover as string (e.g. '0.45')
            failed_checks: Comma-separated list of failed check names
        Returns JSON with primary_symptom, root_cause, do_not_change, repair_strategy.
        """
        try:
            s = float(sharpe) if sharpe not in ("", "null", "None") else None
            f = float(fitness) if fitness not in ("", "null", "None") else None
            t = float(turnover) if turnover not in ("", "null", "None") else None
            checks = [c.strip() for c in failed_checks.split(",") if c.strip()]

            # Rule-based diagnosis — priority: Sharpe first, Fitness second, Turnover as constraint
            if s is not None and s < 0.5:
                primary = "low_sharpe"
                strategy = "Add group_rank sector neutralization; blend an independent second signal (e.g. quality+momentum); extend lookback to 63-126 days; try switching or blending category"
            elif f is not None and f < 0.5:
                primary = "low_fitness"
                if t is not None and t > 0.65:
                    strategy = "Turnover >65% is hurting Fitness — wrap the fastest sub-expression with ts_decay_linear(x, 15) or ts_mean(x, 21)"
                else:
                    strategy = "Turnover is acceptable; signal itself is too weak or concentrated — add a second independent factor, use a fundamentals anchor (cashflow_op/assets, operating_income/assets), or diversify fields"
            elif t is not None and t > 0.7:
                primary = "high_turnover"
                strategy = "Turnover >70% hits BRAIN hard limit — wrap fastest sub-expression with ts_decay_linear(x, 10) or ts_mean(x, 21)"
            elif "CORRLIB" in failed_checks.upper() or "CORRELATION" in failed_checks.upper():
                primary = "high_corrlib"
                strategy = "Replace primary data field with uncorrelated alternative; change time horizon; use regression_neut(new_expr, old_expr)"
            else:
                primary = "low_sharpe"
                strategy = "Diversify signal construction; add group_rank neutralization; blend independent signals"

            # Detect elements to preserve
            do_not_change = []
            if "group_rank" in expression and "industry" in expression:
                do_not_change.append("industry neutralization")
            if "group_rank" in expression and "subindustry" in expression:
                do_not_change.append("subindustry neutralization")

            result = {
                "primary_symptom": primary,
                "root_cause": f"Metrics: sharpe={sharpe}, fitness={fitness}, turnover={turnover}. Failed: {failed_checks}",
                "do_not_change": do_not_change,
                "repair_strategy": strategy,
            }
            return json.dumps(result)
        except Exception as exc:
            return json.dumps({"error": str(exc), "primary_symptom": "low_sharpe", "repair_strategy": "extend lookback and add neutralization"})

    @tool
    def retrieve_repair_memory(expression: str, symptoms: str) -> str:
        """Search historical repair cases semantically similar to the current failing expression.
        Args:
            expression: The failing expression to search similar cases for
            symptoms: Comma-separated symptom names (e.g. 'low_sharpe,low_fitness')
        Returns formatted list of past successful and failed repairs.
        """
        try:
            symptom_tags = [c.strip() for c in symptoms.split(",") if c.strip()]
            if not memory.get_recent(limit=1):
                return "No historical repair cases found yet."
            retrieval = memory.retrieve(symptom_tags, expression, topk=5)
            lines = []
            selected = [
                *list(retrieval.get("positive", []))[:_REPAIR_MEMORY_SUCCESS_LIMIT],
                *list(retrieval.get("negative", []))[:_REPAIR_MEMORY_FAILURE_LIMIT],
            ]
            for i, r in enumerate(selected, 1):
                decision = r.get("accept_decision", "?")
                expr = _clip_text(r.get("expression"), 70)
                tags = ",".join(str(v) for v in (r.get("symptom_tags") or [])[:3] if v)
                semantic = r.get("semantic_score", 0.0)
                if decision == "accepted":
                    dirs = _clip_text(", ".join(v for v in (r.get("recommended_directions") or []) if v), 72)
                    lines.append(f"{i}. [SUCCESS] {expr} | tags={tags or '?'} | fix={dirs or '?'} | sem={semantic:.3f}")
                else:
                    dirs = _clip_text(", ".join(v for v in (r.get("forbidden_directions") or []) if v), 72)
                    lines.append(f"{i}. [FAILED] {expr} | tags={tags or '?'} | avoid={dirs or '?'} | sem={semantic:.3f}")
            return "\n".join(lines) if lines else "No relevant past repairs found."
        except Exception as exc:
            return f"Retrieval error: {exc}"

    @tool
    def retrieve_logic_patterns(expression: str, symptoms: str, category: str = "", k: Any = "5") -> str:
        """Search repair memory for math/economic repair patterns relevant to this expression.
        Args:
            expression: The failing expression
            symptoms: Comma-separated symptom names
            category: Optional factor category hint
            k: Number of patterns to retrieve
        Returns concise successful and failed structured repair patterns.
        """
        try:
            symptom_tags = [c.strip() for c in symptoms.split(",") if c.strip()]
            math_profile = analyze_math_profile(expression)
            economic_profile = infer_economic_profile(
                expression,
                category=category,
                fields=math_profile.get("fields", []),
            )
            k_limit = max(1, int(k or "5"))
            retrieval = memory.retrieve(
                symptom_tags,
                expression,
                family_tag=category or None,
                topk=k_limit,
                math_profile=math_profile,
                economic_profile=economic_profile,
            )
            lines: list[str] = []
            grouped = (
                ("SUCCESS", list(retrieval.get("positive", []))[: min(k_limit, _LOGIC_PATTERN_SUCCESS_LIMIT)]),
                ("FAILED", list(retrieval.get("negative", []))[: min(k_limit, _LOGIC_PATTERN_FAILURE_LIMIT)]),
            )
            line_index = 1
            for label, records in grouped:
                for record in records:
                    delta = record.get("repair_delta") if isinstance(record.get("repair_delta"), dict) else {}
                    actions = delta.get("actions") or [
                        key for key, enabled in delta.items() if isinstance(enabled, bool) and enabled
                    ]
                    theme = (record.get("economic_profile") or {}).get("theme") if isinstance(record.get("economic_profile"), dict) else ""
                    score = record.get("outcome_score", 0.0)
                    platform = ""
                    if isinstance(record.get("platform_outcome"), dict):
                        platform = str(record.get("platform_outcome", {}).get("outcome") or record.get("platform_outcome", {}).get("status") or "")
                    expr = _clip_text(record.get("expression"), 68)
                    lines.append(
                        f"{line_index}. [{label}] {expr} | theme={_clip_text(theme or '?', 20)} | "
                        f"actions={_clip_text(','.join(str(a) for a in actions) or '?', 44)} | "
                        f"platform={platform or '?'} | score={score}"
                    )
                    line_index += 1
            return "\n".join(lines) if lines else "No structured logic patterns found."
        except Exception as exc:
            return f"Structured retrieval error: {exc}"

    @tool
    def validate_expression(expression: str) -> str:
        """Validate a BRAIN alpha expression for syntax correctness.
        Args:
            expression: The BRAIN alpha expression to validate
        Returns 'VALID' if correct, or an error description if invalid.
        """
        if validator is None:
            return "VALID (validator not available)"
        try:
            result = validator.validate(expression)
            valid_attr = getattr(result, "valid", None)
            if isinstance(valid_attr, bool):
                is_valid = valid_attr
            else:
                is_valid_attr = getattr(result, "is_valid", False)
                is_valid = bool(is_valid_attr) if isinstance(is_valid_attr, bool) else False
            if is_valid:
                return "VALID"
            return f"INVALID: {'; '.join(result.errors)}"
        except Exception as exc:
            return f"INVALID: {exc}"

    @tool
    def generate_repair_variants(
        expression: str,
        diagnosis_json: str,
        memory_context: str,
        logic_json: str = "",
        logic_patterns: str = "",
    ) -> str:
        """Generate the single best repair for a failing expression.
        Args:
            expression: The original failing BRAIN expression
            diagnosis_json: JSON string from diagnose_alpha tool
            memory_context: Context string from retrieve_repair_memory tool
            logic_json: JSON string from analyze_factor_logic
            logic_patterns: Structured repair patterns from retrieve_logic_patterns
        Returns JSON array with exactly one candidate: expression, hypothesis, fix_applied.
        """
        # This tool returns a structured prompt for the LLM to fill in.
        # The agent calling this tool will see its result and produce candidates.
        try:
            diag = json.loads(diagnosis_json) if isinstance(diagnosis_json, str) else diagnosis_json
        except Exception:
            diag = {"primary_symptom": "low_sharpe", "repair_strategy": "improve signal"}

        symptom = diag.get("primary_symptom", "low_sharpe")
        strategy = diag.get("repair_strategy", "improve signal construction")
        do_not_change = diag.get("do_not_change", [])
        try:
            logic = json.loads(logic_json) if logic_json else {}
        except Exception:
            logic = {}

        hint = (
            f"Parent expression: {expression}\n"
            f"Symptom: {symptom}\n"
            f"Strategy: {strategy}\n"
            f"Preserve: {', '.join(do_not_change) or 'nothing specific'}\n"
            f"Blocked operators for this account: {', '.join(blocked_operators) if blocked_operators else 'none'}\n"
            f"Math focus: {_summarize_math_profile(logic.get('math_profile', {}))}\n"
            f"Economic thesis: {_summarize_economic_profile(logic.get('economic_profile', {}))}\n"
            f"Pattern cues:\n{_compact_prompt_lines(logic_patterns)}\n"
            f"Memory cues:\n{_compact_prompt_lines(memory_context)}\n\n"
            f"Generate EXACTLY 1 repair — your single most confident fix — as a JSON array with one item:\n"
            '[\n  {"expression": "...", "hypothesis": "...", "fix_applied": "...", '
            '"math_logic": "...", "economic_logic": "...", "expected_failure_reduction": "..."}\n]'
        )
        return hint

    return [
        analyze_factor_logic,
        diagnose_alpha,
        retrieve_repair_memory,
        retrieve_logic_patterns,
        validate_expression,
        generate_repair_variants,
    ]


# ---------------------------------------------------------------------------
# RepairChain — Tool-calling Agent
# ---------------------------------------------------------------------------

class RepairChain:
    """LangChain Tool-calling Agent for alpha repair.

    Supports both Claude (claude-*) and OpenAI (gpt-*) models.
    Pass the appropriate API key via ``api_key``; embeddings always use
    OpenAI text-embedding-3-small (pass ``openai_api_key`` separately when
    the primary LLM is Claude).
    """

    def __init__(
        self,
        memory: RepairMemory,
        api_key: str = "",
        model_id: str = DEFAULT_REPAIR_CHAIN_MODEL,
        temperature: float = 0.7,
        openai_api_key: str = "",
        **kwargs: Any,
    ):
        self.memory = memory
        self._model_id = model_id
        self._api_key = api_key
        self._openai_key = openai_api_key
        self._temperature = temperature
        self._embeddings = None
        self.last_retrieval_mode = "unused"
        self.last_retrieval_error: str | None = None
        self.last_embedding_status = "uninitialized" if openai_api_key else "missing_openai_key"
        self.last_usage: dict[str, float | int] = {"input_tokens": 0, "output_tokens": 0, "latency_ms": 0.0, "calls": 0}

    @property
    def _is_claude(self) -> bool:
        return self._model_id.startswith("claude")

    def _ensure_llm(self, validator: Any = None) -> tuple:
        """Initialize LLM, tools, and embeddings. Returns (llm_with_tools, tools, tool_map)."""
        import os
        tracing = os.environ.get("LANGCHAIN_TRACING_V2", "false")
        project = os.environ.get("LANGCHAIN_PROJECT", "(not set)")
        print(f"[repair_agent] initializing model={self._model_id} langsmith_tracing={tracing} project={project}", flush=True)

        require_semantic_memory = os.environ.get("REPAIR_REQUIRE_SEMANTIC_MEMORY", "true").lower() != "false"
        if self._embeddings is None:
            if self._openai_key:
                try:
                    from langchain_openai import OpenAIEmbeddings
                    self._embeddings = OpenAIEmbeddings(api_key=self._openai_key, model="text-embedding-3-small")
                    self.last_embedding_status = "ready"
                except Exception as exc:
                    self.last_embedding_status = f"init_failed:{type(exc).__name__}"
                    self.last_retrieval_mode = "disabled_embedder_init_failed"
                    self.last_retrieval_error = str(exc)
                    if require_semantic_memory:
                        raise RuntimeError(f"semantic repair memory unavailable: {exc}") from exc
            else:
                self.last_embedding_status = "missing_openai_key"
                self.last_retrieval_mode = "disabled_no_embedder"
                self.last_retrieval_error = "missing_openai_key"
                if require_semantic_memory:
                    raise RuntimeError("semantic repair memory unavailable: missing_openai_key")

        tools = build_tools(self.memory, self._embeddings, validator)
        tool_map = {t.name: t for t in tools}

        if self._is_claude:
            from langchain_anthropic import ChatAnthropic
            llm_kwargs: dict[str, Any] = {
                "api_key": self._api_key,
                "model": self._model_id,
                "max_tokens": 4096,  # required when using bind_tools with Anthropic API
            }
            if not self._model_id.startswith(("claude-opus-4", "claude-sonnet-4")):
                llm_kwargs["temperature"] = self._temperature
            llm = ChatAnthropic(**llm_kwargs)
        else:
            from langchain_openai import ChatOpenAI
            is_reasoning = self._model_id.startswith(("o1", "o3", "o4", "gpt-5"))
            llm_kwargs = {"api_key": self._api_key, "model": self._model_id}
            if not is_reasoning:
                llm_kwargs["temperature"] = self._temperature
            llm = ChatOpenAI(**llm_kwargs)

        llm_with_tools = llm.bind_tools(tools)
        return llm_with_tools, tools, tool_map

    def _run_tool_loop(
        self,
        llm_with_tools: Any,
        tool_map: dict,
        user_input: str,
        max_iterations: int = 8,
    ) -> tuple[str, dict[str, float | int]]:
        """Manual tool-calling loop compatible with LangChain 1.x."""
        from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage

        messages: list[Any] = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=user_input),
        ]
        usage = {"input_tokens": 0, "output_tokens": 0, "latency_ms": 0.0, "calls": 0}
        json_retry_budget = 1

        for iteration in range(max_iterations):
            t0 = time.time()
            response = llm_with_tools.invoke(messages)
            usage["latency_ms"] += (time.time() - t0) * 1000
            usage["calls"] += 1
            in_tokens, out_tokens = _extract_usage_metadata(response)
            usage["input_tokens"] += in_tokens
            usage["output_tokens"] += out_tokens
            messages.append(response)

            tool_calls = getattr(response, "tool_calls", None) or []
            if not tool_calls:
                output_text = _coerce_agent_output_text(response)
                probe_candidates, _probe_diagnosis = _parse_agent_output(output_text, "probe")
                if probe_candidates or json_retry_budget <= 0 or iteration >= max_iterations - 1:
                    return output_text, usage
                messages.append(
                    HumanMessage(
                        content=(
                            "Return ONLY valid JSON matching the required schema with EXACTLY ONE candidate. "
                            "No prose, no markdown fences, no explanation."
                        )
                    )
                )
                json_retry_budget -= 1
                continue

            # Execute each requested tool
            for tc in tool_calls:
                tool_name = tc.get("name") or tc.get("function", {}).get("name", "")
                tool_args = tc.get("args") or tc.get("function", {}).get("arguments", {})
                tool_id = tc.get("id", f"call_{iteration}")
                tool = tool_map.get(tool_name)
                if tool is None:
                    tool_result = f"Error: unknown tool '{tool_name}'"
                else:
                    try:
                        tool_result = str(tool.invoke(tool_args))
                    except Exception as exc:
                        tool_result = f"Error running {tool_name}: {exc}"
                messages.append(ToolMessage(content=tool_result, tool_call_id=tool_id))

        # Exceeded max_iterations — return last AI message content if any
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                return msg.content, usage
        return "", usage

    def run(
        self,
        expression: str,
        metrics: dict,
        failed_checks: list[str],
        gate_reasons: list[str],
        n: int = 3,
        category: str = "UNKNOWN",
        validator: Any = None,
        repair_context: dict | None = None,
        seed_candidates: list[dict[str, Any]] | None = None,
        repair_policy: dict[str, Any] | None = None,
    ) -> tuple[list[dict], dict]:
        """Run the repair agent. Returns (candidates, diagnosis)."""
        sharpe = metrics.get("sharpe") or metrics.get("isSharpe")
        fitness = metrics.get("fitness") or metrics.get("isFitness")
        turnover = metrics.get("turnover")
        recursive_guidance = build_recursive_repair_guidance(repair_context)

        user_input = (
            f"Repair this failing alpha expression:\n"
            f"Expression: {expression}\n"
            f"Category: {category}\n"
            f"Sharpe: {sharpe}, Fitness: {fitness}, Turnover: {turnover}\n"
            f"Failed checks: {', '.join(failed_checks) or 'none'}\n"
            f"Gate reasons: {'; '.join(gate_reasons[:3]) or 'none'}\n"
            f"Generate {n} repair variants."
        )
        if recursive_guidance:
            user_input += "\n" + "\n".join(recursive_guidance)
        if repair_policy:
            user_input += (
                "\nRepair routing policy:\n"
                + json.dumps(repair_policy, ensure_ascii=False)
            )
        if seed_candidates:
            user_input += (
                "\nRule-based seed candidates (use as mathematical/economic starting points, not as final answers):\n"
                + json.dumps(seed_candidates, ensure_ascii=False)
            )

        try:
            import os

            self.last_usage = {"input_tokens": 0, "output_tokens": 0, "latency_ms": 0.0, "calls": 0}
            self.last_retrieval_mode = "tool_loop_no_retrieval"
            self.last_retrieval_error = None
            llm_with_tools, tools, tool_map = self._ensure_llm(validator)
            max_iterations = max(8, int(os.environ.get("REPAIR_AGENT_MAX_ITERATIONS", "8")))
            loop_result = self._run_tool_loop(llm_with_tools, tool_map, user_input, max_iterations=max_iterations)
            if isinstance(loop_result, tuple):
                output, usage = loop_result
            else:
                output = str(loop_result or "")
                usage = {"input_tokens": 0, "output_tokens": 0, "latency_ms": 0.0, "calls": 0}
            self.last_usage = usage
            self.last_retrieval_mode = getattr(self.memory, "last_retrieval_mode", self.last_retrieval_mode)
            self.last_retrieval_error = getattr(self.memory, "last_retrieval_error", self.last_retrieval_error)
            output_text = _coerce_agent_output_text(output)
            candidates, diagnosis = _parse_agent_output(output_text, category)
            if candidates:
                print(f"[repair_agent] generated {len(candidates)} candidates", flush=True)
                return candidates, diagnosis
            print(f"[repair_agent] no candidates parsed, output[:200]: {output_text[:200]}", flush=True)
        except Exception as exc:
            import traceback
            print(f"[repair_agent] ERROR {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()

        return [], {}

    def record_outcome(
        self,
        expression: str,
        diagnosis: dict,
        candidates: list[dict],
        accepted: bool,
        candidate_metrics: dict | None = None,
        gate: dict | None = None,
        platform_outcome: dict | None = None,
        category: str | None = None,
    ) -> None:
        """Persist repair outcome to memory and reset FAISS cache."""
        symptom_tags = derive_symptom_tags(diagnosis, gate=gate, platform_outcome=platform_outcome)
        fix_list = [
            c["origin_refs"][1] for c in candidates
            if len(c.get("origin_refs", [])) > 1 and c["origin_refs"][1]
        ]
        candidate_expression = next((c.get("expression") for c in candidates if c.get("expression")), expression)
        math_profile = analyze_math_profile(candidate_expression)
        economic_profile = infer_economic_profile(
            candidate_expression,
            category=category,
            fields=math_profile.get("fields", []),
        )
        repair_delta = compare_repair(expression, candidate_expression)
        outcome_score = score_repair_outcome(
            candidate_metrics,
            gate=gate,
            accepted=accepted,
            platform_outcome=platform_outcome,
        )
        self.memory.add_record({
            "expression": expression,
            "symptom_tags": symptom_tags,
            "repair_actions": fix_list,
            "accept_decision": "accepted" if accepted else "rejected",
            "recommended_directions": fix_list if accepted else [],
            "forbidden_directions": fix_list if not accepted else [],
            "metrics": candidate_metrics or {},
            "family_tag": category or "",
            "notes": "; ".join(build_recursive_repair_guidance(platform_outcome)),
            "math_profile": math_profile,
            "economic_profile": economic_profile,
            "repair_delta": repair_delta,
            "outcome_score": outcome_score,
            "platform_outcome": platform_outcome or {},
        })
        # Reset embeddings so FAISS index rebuilds on next call
        self._embeddings = None

    def semantic_memory_enabled(self) -> bool:
        return bool(self._openai_key)


# ---------------------------------------------------------------------------
# Output parser
# ---------------------------------------------------------------------------

def _coerce_agent_output_text(output: Any) -> str:
    """Normalize LangChain message content into parseable text.

    ChatAnthropic can return content blocks such as
    ``[{"type": "text", "text": "..."}]`` instead of a plain string. The
    repair parser only needs the text blocks; tool-use blocks are ignored.
    """
    if output is None:
        return ""
    if isinstance(output, str):
        return output
    if isinstance(output, bytes):
        return output.decode("utf-8", errors="replace")
    if isinstance(output, dict):
        for key in ("text", "content"):
            value = output.get(key)
            if isinstance(value, (str, bytes, list, dict)):
                return _coerce_agent_output_text(value)
        if "candidates" in output or "diagnosis" in output:
            return json.dumps(output)
        return ""
    if isinstance(output, list):
        parts = []
        for item in output:
            text = _coerce_agent_output_text(item)
            if text:
                parts.append(text)
        return "\n".join(parts)
    content = getattr(output, "content", None)
    if content is not None:
        return _coerce_agent_output_text(content)
    text = getattr(output, "text", None)
    if text is not None:
        return _coerce_agent_output_text(text)
    return str(output)


def _parse_agent_output(output: Any, category: str) -> tuple[list[dict], dict]:
    """Extract candidates and diagnosis from agent final answer."""
    output = _coerce_agent_output_text(output)
    # Try to find JSON block
    json_match = re.search(r'\{[\s\S]*"candidates"[\s\S]*\}', output)
    if json_match:
        try:
            data = json.loads(json_match.group(0))
            diagnosis = data.get("diagnosis", {})
            raw_candidates = data.get("candidates", [])
            candidates = [
                {
                    "id": f"repair_{category.lower()}_{i:03d}",
                    "category": category,
                    "hypothesis": item.get("hypothesis", ""),
                    "expression": item.get("expression", ""),
                    "origin_refs": ["langchain_agent", item.get("fix_applied", "")],
                    "metadata": _candidate_metadata(item),
                    "opt_rounds": 0,
                }
                for i, item in enumerate(raw_candidates)
                if isinstance(item, dict) and item.get("expression")
            ]
            return candidates, diagnosis
        except json.JSONDecodeError:
            pass

    # Fallback: scan for JSON array of candidates
    array_match = re.search(r'\[[\s\S]*"expression"[\s\S]*\]', output)
    if array_match:
        try:
            raw_candidates = json.loads(array_match.group(0))
            candidates = [
                {
                    "id": f"repair_{category.lower()}_{i:03d}",
                    "category": category,
                    "hypothesis": item.get("hypothesis", ""),
                    "expression": item.get("expression", ""),
                    "origin_refs": ["langchain_agent", item.get("fix_applied", "")],
                    "metadata": _candidate_metadata(item),
                    "opt_rounds": 0,
                }
                for i, item in enumerate(raw_candidates)
                if isinstance(item, dict) and item.get("expression")
            ]
            return candidates, {}
        except json.JSONDecodeError:
            pass

    return [], {}


def _candidate_metadata(item: dict[str, Any]) -> dict[str, str]:
    metadata = {}
    for key in ("math_logic", "economic_logic", "expected_failure_reduction"):
        value = item.get(key)
        if value not in (None, ""):
            metadata[key] = str(value)
    return metadata


__all__ = ["RepairChain"]


def _extract_usage_metadata(response: Any) -> tuple[int, int]:
    usage = getattr(response, "usage_metadata", None)
    if isinstance(usage, dict):
        in_tokens = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
        out_tokens = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
        return in_tokens, out_tokens

    metadata = getattr(response, "response_metadata", None)
    if isinstance(metadata, dict):
        usage_meta = metadata.get("usage")
        if isinstance(usage_meta, dict):
            in_tokens = int(usage_meta.get("input_tokens") or usage_meta.get("prompt_tokens") or 0)
            out_tokens = int(usage_meta.get("output_tokens") or usage_meta.get("completion_tokens") or 0)
            return in_tokens, out_tokens
        token_usage = metadata.get("token_usage")
        if isinstance(token_usage, dict):
            in_tokens = int(token_usage.get("prompt_tokens") or token_usage.get("input_tokens") or 0)
            out_tokens = int(token_usage.get("completion_tokens") or token_usage.get("output_tokens") or 0)
            return in_tokens, out_tokens
    return 0, 0
