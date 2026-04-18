"""LangChain Tool-calling Agent for alpha factor repair.

The agent autonomously decides which tools to call and in what order:
  diagnose_alpha → retrieve_repair_memory → generate_repair_variants → validate_expression
"""
from __future__ import annotations

import json
import re
from typing import Any

import numpy as np

from alpha_miner.modules.m_repair_memory import RepairMemory

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a WorldQuant BRAIN alpha factor repair agent.

Your goal: repair a failing alpha expression so it passes all quality checks.

BRAIN OPERATORS:
ts_mean(x,d), ts_rank(x,d), ts_std_dev(x,d), ts_delta(x,d), ts_zscore(x,d),
ts_decay_linear(x,d), ts_corr(x,y,d), delay(x,d), rank(x), zscore(x),
group_rank(x, industry), group_rank(x, subindustry),
divide(x,y), multiply(x,y), abs(x), log(x), power(x,n), sign(x), winsorize(x,p)

REPAIR PRINCIPLES:
- low_sharpe: longer lookback, add group_rank sector neutralization, blend complementary signals
- low_fitness: wrap with rank() or zscore(), add ts_mean for cross-sectional stability
- high_turnover: wrap with ts_decay_linear(x, 5-20) or extend window to 21+ days
- high_corrlib: change primary data field or time horizon entirely

WORKFLOW (follow this order):
1. Call diagnose_alpha to understand the failure
2. Call retrieve_repair_memory to learn from past repairs
3. Call generate_repair_variants to create candidates
4. Call validate_expression for each candidate
5. Return your final answer as JSON ONLY — no other text:
{
  "diagnosis": {"primary_symptom": "<symptom>", "root_cause": "<cause>"},
  "candidates": [
    {"expression": "<valid BRAIN expr>", "hypothesis": "<economic logic>", "fix_applied": "<what changed>"}
  ]
}"""


# ---------------------------------------------------------------------------
# Tool factory — returns tools bound to shared state
# ---------------------------------------------------------------------------

def build_tools(memory: RepairMemory, embeddings: Any, validator: Any) -> list:
    """Build LangChain tools with injected dependencies."""
    from langchain_core.tools import tool

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

            # Rule-based diagnosis
            if s is not None and s < 0.5:
                primary = "low_sharpe"
                strategy = "Add sector neutralization via group_rank, extend lookback window to 21-63 days, blend with complementary signal"
            elif f is not None and f < 0.5:
                primary = "low_fitness"
                strategy = "Apply rank() or zscore() cross-sectionally, add ts_mean smoothing for stability"
            elif t is not None and t > 0.7:
                primary = "high_turnover"
                strategy = "Wrap signal with ts_decay_linear(x, 10) or ts_mean(x, 5), use longer windows"
            elif "CORRLIB" in failed_checks.upper() or "CORRELATION" in failed_checks.upper():
                primary = "high_corrlib"
                strategy = "Replace primary data field with uncorrelated alternative, change time horizon"
            else:
                primary = "low_sharpe"
                strategy = "Diversify signal construction, add neutralization"

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
    def retrieve_repair_memory(expression: str, symptoms: str, k: int = 5) -> str:
        """Search historical repair cases semantically similar to the current failing expression.
        Args:
            expression: The failing expression to search similar cases for
            symptoms: Comma-separated symptom names (e.g. 'low_sharpe,low_fitness')
            k: Number of similar cases to retrieve
        Returns formatted list of past successful and failed repairs.
        """
        try:
            records = memory.get_recent(limit=500)
            if not records:
                return "No historical repair cases found yet."

            if embeddings is not None:
                query = f"{expression} {symptoms}"
                import faiss
                def _to_str(v: Any) -> str:
                    return " ".join(v) if isinstance(v, list) else str(v or "")
                texts = [
                    f"{r.get('expression', '')} {_to_str(r.get('symptom_tags', []))} {_to_str(r.get('recommended_directions', []))}"
                    for r in records
                ]
                vecs = np.array(embeddings.embed_documents(texts), dtype=np.float32)
                faiss.normalize_L2(vecs)
                index = faiss.IndexFlatIP(vecs.shape[1])
                index.add(vecs)
                q = np.array([embeddings.embed_query(query)], dtype=np.float32)
                faiss.normalize_L2(q)
                _, idxs = index.search(q, min(k, len(records)))
                top = [records[i] for i in idxs[0] if 0 <= i < len(records)]
            else:
                top = records[:k]

            lines = []
            for i, r in enumerate(top, 1):
                decision = r.get("accept_decision", "?")
                expr = (r.get("expression") or "")[:80]
                tags = ", ".join(v for v in (r.get("symptom_tags") or []) if v)
                if decision == "accepted":
                    dirs = ", ".join(v for v in (r.get("recommended_directions") or []) if v)
                    lines.append(f"{i}. [SUCCESS] {expr} | {tags} | fix: {dirs}")
                else:
                    dirs = ", ".join(v for v in (r.get("forbidden_directions") or []) if v)
                    lines.append(f"{i}. [FAILED]  {expr} | {tags} | avoid: {dirs}")
            return "\n".join(lines) if lines else "No relevant past repairs found."
        except Exception as exc:
            return f"Retrieval error: {exc}"

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
            if result.valid:
                return "VALID"
            return f"INVALID: {'; '.join(result.errors)}"
        except Exception as exc:
            return f"INVALID: {exc}"

    @tool
    def generate_repair_variants(expression: str, diagnosis_json: str, memory_context: str, n: int = 3) -> str:
        """Generate N repair variants of a failing expression.
        Args:
            expression: The original failing BRAIN expression
            diagnosis_json: JSON string from diagnose_alpha tool
            memory_context: Context string from retrieve_repair_memory tool
            n: Number of variants to generate (default 3)
        Returns JSON array of candidates with expression, hypothesis, fix_applied.
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

        hint = (
            f"Parent expression: {expression}\n"
            f"Symptom: {symptom}\n"
            f"Strategy: {strategy}\n"
            f"Preserve: {', '.join(do_not_change) or 'nothing specific'}\n"
            f"Memory context:\n{memory_context}\n\n"
            f"Generate {n} repair variants as JSON array:\n"
            '[\n  {"expression": "...", "hypothesis": "...", "fix_applied": "..."},\n  ...\n]'
        )
        return hint

    return [diagnose_alpha, retrieve_repair_memory, validate_expression, generate_repair_variants]


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
        model_id: str = "claude-opus-4-6",
        temperature: float = 0.7,
        openai_api_key: str = "",
        # Legacy positional alias kept for backward compat
        **kwargs: Any,
    ):
        self.memory = memory
        self._model_id = model_id
        self._api_key = api_key or kwargs.get("openai_api_key", "")
        self._openai_key = openai_api_key or self._api_key
        self._temperature = temperature
        self._embeddings = None

    @property
    def _is_claude(self) -> bool:
        return self._model_id.startswith("claude")

    def _ensure_llm(self, validator: Any = None) -> tuple:
        """Initialize LLM, tools, and embeddings. Returns (llm_with_tools, tools, tool_map)."""
        import os
        tracing = os.environ.get("LANGCHAIN_TRACING_V2", "false")
        project = os.environ.get("LANGCHAIN_PROJECT", "(not set)")
        print(f"[repair_agent] initializing model={self._model_id} langsmith_tracing={tracing} project={project}", flush=True)

        from langchain_openai import OpenAIEmbeddings

        if self._embeddings is None and self._openai_key:
            self._embeddings = OpenAIEmbeddings(api_key=self._openai_key, model="text-embedding-3-small")

        tools = build_tools(self.memory, self._embeddings, validator)
        tool_map = {t.name: t for t in tools}

        if self._is_claude:
            from langchain_anthropic import ChatAnthropic
            llm_kwargs: dict[str, Any] = {
                "api_key": self._api_key,
                "model": self._model_id,
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

    def _run_tool_loop(self, llm_with_tools: Any, tool_map: dict, user_input: str, max_iterations: int = 8) -> str:
        """Manual tool-calling loop compatible with LangChain 1.x. Returns final text output."""
        from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage

        messages: list[Any] = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=user_input),
        ]

        for iteration in range(max_iterations):
            response = llm_with_tools.invoke(messages)
            messages.append(response)

            tool_calls = getattr(response, "tool_calls", None) or []
            if not tool_calls:
                # No more tool calls — this is the final answer
                return response.content or ""

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
                return msg.content
        return ""

    def run(
        self,
        expression: str,
        metrics: dict,
        failed_checks: list[str],
        gate_reasons: list[str],
        n: int = 3,
        category: str = "UNKNOWN",
        validator: Any = None,
    ) -> tuple[list[dict], dict]:
        """Run the repair agent. Returns (candidates, diagnosis)."""
        sharpe = metrics.get("sharpe") or metrics.get("isSharpe")
        fitness = metrics.get("fitness") or metrics.get("isFitness")
        turnover = metrics.get("turnover")

        user_input = (
            f"Repair this failing alpha expression:\n"
            f"Expression: {expression}\n"
            f"Sharpe: {sharpe}, Fitness: {fitness}, Turnover: {turnover}\n"
            f"Failed checks: {', '.join(failed_checks) or 'none'}\n"
            f"Gate reasons: {'; '.join(gate_reasons[:3]) or 'none'}\n"
            f"Generate {n} repair variants."
        )

        try:
            llm_with_tools, tools, tool_map = self._ensure_llm(validator)
            output = self._run_tool_loop(llm_with_tools, tool_map, user_input)
            candidates, diagnosis = _parse_agent_output(output, category)
            if candidates:
                print(f"[repair_agent] generated {len(candidates)} candidates", flush=True)
                return candidates, diagnosis
            print(f"[repair_agent] no candidates parsed, output[:200]: {output[:200]}", flush=True)
        except Exception as exc:
            import traceback
            print(f"[repair_agent] ERROR {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()

        return [], {}

    def record_outcome(self, expression: str, diagnosis: dict, candidates: list[dict], accepted: bool) -> None:
        """Persist repair outcome to memory and reset FAISS cache."""
        symptom = diagnosis.get("primary_symptom", "")
        fix_list = [
            c["origin_refs"][1] for c in candidates
            if len(c.get("origin_refs", [])) > 1 and c["origin_refs"][1]
        ]
        self.memory.add_record({
            "expression": expression,
            "symptom_tags": [symptom] if symptom else [],
            "repair_actions": fix_list,
            "accept_decision": "accepted" if accepted else "rejected",
            "recommended_directions": fix_list if accepted else [],
            "forbidden_directions": fix_list if not accepted else [],
            "family_tag": "",
        })
        # Reset embeddings so FAISS index rebuilds on next call
        self._embeddings = None


# ---------------------------------------------------------------------------
# Output parser
# ---------------------------------------------------------------------------

def _parse_agent_output(output: str, category: str) -> tuple[list[dict], dict]:
    """Extract candidates and diagnosis from agent final answer."""
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
                    "opt_rounds": 0,
                }
                for i, item in enumerate(raw_candidates)
                if isinstance(item, dict) and item.get("expression")
            ]
            return candidates, {}
        except json.JSONDecodeError:
            pass

    return [], {}


__all__ = ["RepairChain"]
