from __future__ import annotations
import json, re
from dataclasses import dataclass, field

from alpha_miner.modules.m_diagnoser import DiagnosisReport

__all__ = ["RepairPlan", "Planner"]

ACTION_TYPES = ["param_tune", "struct_mutation", "template_retrieval", "llm_mutation"]
_DEFAULT_W = {"param_tune": 0.2, "struct_mutation": 0.3, "template_retrieval": 0.2, "llm_mutation": 0.3}


@dataclass
class RepairPlan:
    candidate_mix: dict = field(default_factory=dict)
    total_budget: int = 10
    prioritized_actions: list = field(default_factory=list)
    hard_constraints: list = field(default_factory=list)
    acceptance_rules: list = field(default_factory=list)
    fallback_to_param_tune: bool = False


class Planner:
    def __init__(self, router=None):
        self.router = router

    def plan(
        self,
        diagnosis: DiagnosisReport,
        retrieval: dict,
        total_budget: int = 10,
        scheduler_weights: dict | None = None,
    ) -> RepairPlan:
        weights = dict(scheduler_weights) if scheduler_weights else dict(_DEFAULT_W)
        # symptom-based overrides
        sym = (diagnosis.primary_symptom or "").lower()
        if sym == "high_turnover":
            weights["param_tune"] = 0.4
            weights["llm_mutation"] = 0.1
        elif sym == "high_corrlib":
            weights["struct_mutation"] = 0.5
        if retrieval.get("family_saturated"):
            weights["template_retrieval"] = 0.0
        # normalise
        total_w = sum(weights.values()) or 1.0
        weights = {k: v / total_w for k, v in weights.items()}
        # integer counts
        mix = {k: max(0, round(v * total_budget)) for k, v in weights.items()}
        diff = total_budget - sum(mix.values())
        if diff != 0:
            key = max(mix, key=mix.get)
            mix[key] = max(0, mix[key] + diff)
        plan = RepairPlan(
            candidate_mix=mix,
            total_budget=total_budget,
            prioritized_actions=sorted(mix, key=mix.get, reverse=True),
            fallback_to_param_tune=mix.get("param_tune", 0) >= total_budget * 0.5,
        )
        if self.router is not None:
            try:
                plan = self._llm_refine(plan, diagnosis, retrieval)
            except Exception:
                pass
        return plan

    def _llm_refine(self, plan: RepairPlan, diagnosis: DiagnosisReport, retrieval: dict) -> RepairPlan:
        provider = self._pick_provider()
        if provider is None:
            return plan
        system = (
            "You are a repair Planner. Review this candidate allocation and adjust if needed. "
            'Return ONLY JSON: {"candidate_mix":{"param_tune":int,"struct_mutation":int,'
            '"template_retrieval":int,"llm_mutation":int},"prioritized_actions":[str],'
            '"hard_constraints":[str],"acceptance_rules":[str],"fallback_to_param_tune":bool}'
        )
        user_obj = {
            "primary_symptom": diagnosis.primary_symptom,
            "retrieval_summary": retrieval.get("retrieval_summary", ""),
            "family_saturated": retrieval.get("family_saturated", False),
            "proposed_mix": plan.candidate_mix,
            "total_budget": plan.total_budget,
        }
        payload: dict = {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user_obj)},
            ]
        }
        if provider.model_id.startswith(("o1", "o3", "o4", "gpt-5")):
            payload["max_completion_tokens"] = 400
        else:
            payload["temperature"] = 0.0
            payload["max_tokens"] = 400
        raw, *_ = self.router._call_provider(provider, payload)
        data = _extract_json(raw)
        mix = {k: int(data["candidate_mix"].get(k, v)) for k, v in plan.candidate_mix.items()}
        return RepairPlan(
            candidate_mix=mix,
            total_budget=plan.total_budget,
            prioritized_actions=data.get("prioritized_actions", plan.prioritized_actions),
            hard_constraints=data.get("hard_constraints", []),
            acceptance_rules=data.get("acceptance_rules", []),
            fallback_to_param_tune=bool(data.get("fallback_to_param_tune", False)),
        )

    def _pick_provider(self):
        if self.router is None:
            return None
        try:
            for role in ("repair", "generate", "judge"):
                return self.router.pick(role)
        except Exception:
            return None


def _extract_json(raw: str) -> dict:
    text = raw.strip()
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        text = m.group(0)
    return json.loads(text)
