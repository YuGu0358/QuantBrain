from __future__ import annotations

import json
from typing import Any

import numpy as np

from alpha_miner.modules.m_repair_memory import RepairMemory

_SYSTEM_PROMPT = """You are an expert WorldQuant BRAIN alpha factor repair specialist.

BRAIN OPERATOR REFERENCE:
- Time series: ts_mean(x,d), ts_rank(x,d), ts_std_dev(x,d), ts_delta(x,d), ts_zscore(x,d), ts_corr(x,y,d), delay(x,d)
- Cross-section: rank(x), zscore(x), group_rank(x, industry), group_rank(x, subindustry)
- Math: divide(x,y), multiply(x,y), abs(x), log(x), power(x,n), sign(x)
- Smoothing: ts_decay_linear(x,d) reduces turnover, ts_mean(x,d) smooths signals
- Normalization: winsorize(x,p) clips outliers

REPAIR PRINCIPLES:
- low_sharpe -> add complementary signals, try group_rank for sector neutralization, longer lookback windows
- low_fitness -> normalize cross-sectionally with rank() or zscore(), blend with ts_mean for stability
- high_turnover -> wrap with ts_decay_linear(x, 5-20) or ts_mean(x, 5-10), use longer windows (21+ days)
- high_corrlib -> change primary data field or time horizon entirely

GOLDEN RULES:
1. Preserve the economic thesis — fix the symptom, not the concept
2. Each candidate must be meaningfully different from the parent and from each other
3. Use industry/subindustry ONLY inside group_rank(expr, industry)
4. Keep operator depth <=12
Return ONLY valid JSON — no markdown fences, no explanation outside JSON."""

_HUMAN_PROMPT = """Repair this failing alpha expression.

Parent expression: {expression}
Metrics: {metrics_json}
Failed checks: {failed_checks}
Gate reasons: {gate_reasons}

Historical repair context:
{retrieved_context}

Generate exactly {n} repair variants. Return JSON:
{{
  "diagnosis": {{
    "primary_symptom": "low_sharpe|low_fitness|high_turnover|high_corrlib|high_complexity",
    "root_cause": "one sentence",
    "do_not_change": ["element1"]
  }},
  "candidates": [
    {{
      "expression": "valid BRAIN expression",
      "hypothesis": "1-2 sentences: economic logic and what was fixed",
      "fix_applied": "structural change description"
    }}
  ]
}}"""


class RepairChain:
    """LangChain LCEL repair chain with FAISS semantic retrieval over RepairMemory."""

    def __init__(
        self,
        memory: RepairMemory,
        openai_api_key: str,
        model_id: str = "gpt-4o",
        temperature: float = 0.7,
    ):
        self.memory = memory
        self._model_id = model_id
        self._api_key = openai_api_key
        self._temperature = temperature
        self._index_cache: tuple | None = None
        self._chain = None
        self._embeddings = None

    def _ensure_chain(self) -> None:
        if self._chain is not None:
            return
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import JsonOutputParser

        self._embeddings = OpenAIEmbeddings(api_key=self._api_key, model="text-embedding-3-small")
        is_reasoning = self._model_id.startswith(("o1", "o3", "o4", "gpt-5"))
        llm_kwargs: dict[str, Any] = {"api_key": self._api_key, "model": self._model_id}
        if not is_reasoning:
            llm_kwargs["temperature"] = self._temperature
        llm = ChatOpenAI(**llm_kwargs)
        prompt = ChatPromptTemplate.from_messages([
            ("system", _SYSTEM_PROMPT),
            ("human", _HUMAN_PROMPT),
        ])
        self._chain = prompt | llm | JsonOutputParser()

    def run(
        self,
        expression: str,
        metrics: dict,
        failed_checks: list[str],
        gate_reasons: list[str],
        n: int = 3,
        category: str = "UNKNOWN",
    ) -> tuple[list[dict], dict]:
        """Run repair. Returns (candidates_list, diagnosis_dict)."""
        self._ensure_chain()
        retrieved = self._semantic_retrieve(expression, failed_checks, k=5)
        result = self._chain.invoke({
            "expression": expression,
            "metrics_json": json.dumps(
                {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items() if v is not None},
                ensure_ascii=False,
            ),
            "failed_checks": ", ".join(failed_checks) or "none",
            "gate_reasons": "; ".join(gate_reasons[:3]) or "none",
            "retrieved_context": self._format_retrieved(retrieved),
            "n": n,
        })
        candidates_raw = result.get("candidates", [])
        diagnosis = result.get("diagnosis", {})
        candidates = [
            {
                "id": f"repair_{category.lower()}_{i:03d}",
                "category": category,
                "hypothesis": item.get("hypothesis", ""),
                "expression": item.get("expression", ""),
                "origin_refs": ["langchain_repair", item.get("fix_applied", "")],
                "opt_rounds": 0,
            }
            for i, item in enumerate(candidates_raw)
            if isinstance(item, dict) and item.get("expression")
        ]
        return candidates, diagnosis

    def record_outcome(self, expression: str, diagnosis: dict, candidates: list[dict], accepted: bool) -> None:
        """Persist repair outcome and invalidate FAISS cache."""
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
        self._index_cache = None

    def _semantic_retrieve(self, expression: str, failed_checks: list[str], k: int = 5) -> list[dict]:
        records = self.memory.get_recent(limit=500)
        if not records or self._embeddings is None:
            return []
        query = f"{expression} {' '.join(failed_checks)}"
        try:
            import faiss
            if self._index_cache is None:
                def _to_str(v: Any) -> str:
                    return " ".join(v) if isinstance(v, list) else str(v or "")
                texts = [
                    f"{r.get('expression', '')} {_to_str(r.get('symptom_tags', []))} {_to_str(r.get('recommended_directions', []))}"
                    for r in records
                ]
                vecs = np.array(self._embeddings.embed_documents(texts), dtype=np.float32)
                faiss.normalize_L2(vecs)
                index = faiss.IndexFlatIP(vecs.shape[1])
                index.add(vecs)
                self._index_cache = (index, records)
            index, stored = self._index_cache
            q = np.array([self._embeddings.embed_query(query)], dtype=np.float32)
            faiss.normalize_L2(q)
            _, idxs = index.search(q, min(k, len(stored)))
            return [stored[i] for i in idxs[0] if 0 <= i < len(stored)]
        except Exception:
            return records[:k]

    def _format_retrieved(self, records: list[dict]) -> str:
        if not records:
            return "No historical repairs found."
        lines = []
        for i, r in enumerate(records, 1):
            decision = r.get("accept_decision", "?")
            expr = (r.get("expression") or "")[:80]
            tags = ", ".join(v for v in (r.get("symptom_tags") or []) if v)
            if decision == "accepted":
                dirs = ", ".join(v for v in (r.get("recommended_directions") or []) if v)
                lines.append(f"{i}. [SUCCESS] {expr} | {tags} | fix: {dirs}")
            else:
                dirs = ", ".join(v for v in (r.get("forbidden_directions") or []) if v)
                lines.append(f"{i}. [FAILED]  {expr} | {tags} | avoid: {dirs}")
        return "\n".join(lines)


__all__ = ["RepairChain"]
