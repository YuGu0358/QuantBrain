import json
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from pathlib import Path

from alpha_miner.modules.llm_cache import LLMCache
from alpha_miner.modules.m1_knowledge_base import KnowledgeBase
from alpha_miner.modules.m2_hypothesis_agent import HypothesisAgent
from alpha_miner.modules.m3_validator import ExpressionValidator
from alpha_miner.modules.m_diagnoser import DiagnosisReport
from alpha_miner.modules.m_repair_intelligence import analyze_math_profile
from alpha_miner.modules.m_repair_intelligence import infer_economic_profile
from alpha_miner.modules.m_repair_memory import RepairMemory
from alpha_miner.modules.m_retriever import Retriever


def _timestamp(days_ago: int = 0) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()


def _agent(tmp_path: Path) -> HypothesisAgent:
    return HypothesisAgent(
        kb=KnowledgeBase(tmp_path / "kb.db"),
        cache=LLMCache(tmp_path / "cache"),
        taxonomy={"QUALITY": {}, "MOMENTUM": {}},
    )


def test_repair_memory_retrieve_scores_by_symptoms_family_and_age(tmp_path: Path):
    memory = RepairMemory(tmp_path / "repair_memory.sqlite")
    memory.add_record(
        {
            "record_id": "accepted-old-perfect",
            "timestamp": _timestamp(days_ago=45),
            "expression": "rank(ts_delta(volume, 5))",
            "symptom_tags": ["low_sharpe", "high_turnover"],
            "accept_decision": "accepted",
            "family_tag": "MOMENTUM",
            "recommended_directions": ["use volume confirmation"],
        }
    )
    memory.add_record(
        {
            "record_id": "accepted-family-partial",
            "timestamp": _timestamp(),
            "expression": "group_rank(ts_mean(volume, 21), industry)",
            "symptom_tags": ["high_turnover"],
            "accept_decision": "accepted",
            "family_tag": "QUALITY",
            "recommended_directions": ["smooth fast volume", "use volume confirmation"],
        }
    )
    memory.add_record(
        {
            "record_id": "rejected-family-match",
            "timestamp": _timestamp(),
            "expression": "rank(ts_delta(close, 3))",
            "symptom_tags": ["high_turnover"],
            "accept_decision": "rejected",
            "family_tag": "QUALITY",
            "forbidden_directions": ["avoid short deltas", "avoid pure price"],
        }
    )
    memory.add_record(
        {
            "record_id": "rejected-duplicate-direction",
            "timestamp": _timestamp(days_ago=2),
            "expression": "rank(ts_delta(volume, 3))",
            "symptom_tags": ["high_turnover", "low_fitness"],
            "accept_decision": "rejected",
            "family_tag": "QUALITY",
            "forbidden_directions": ["avoid short deltas", "avoid volume-only repairs"],
        }
    )

    result = memory.retrieve(
        symptom_tags=["low_sharpe", "high_turnover"],
        expression="rank(ts_delta(volume, 5))",
        family_tag="QUALITY",
        topk=2,
    )

    assert [record["record_id"] for record in result["positive"]] == [
        "accepted-family-partial",
        "accepted-old-perfect",
    ]
    assert [record["record_id"] for record in result["negative"]] == [
        "rejected-family-match",
        "rejected-duplicate-direction",
    ]
    assert result["recommended_directions"] == [
        "smooth fast volume",
        "use volume confirmation",
    ]
    assert result["forbidden_directions"] == [
        "avoid short deltas",
        "avoid pure price",
        "avoid volume-only repairs",
    ]


def test_repair_memory_family_saturation_counts_only_accepted_records(tmp_path: Path):
    memory = RepairMemory(tmp_path / "repair_memory.sqlite")
    for index in range(2):
        memory.add_record(
            {
                "record_id": f"accepted-quality-{index}",
                "expression": f"expr_{index}",
                "symptom_tags": ["low_sharpe"],
                "accept_decision": "accepted",
                "family_tag": "QUALITY",
            }
        )
    memory.add_record(
        {
            "record_id": "rejected-quality",
            "expression": "bad_expr",
            "symptom_tags": ["low_sharpe"],
            "accept_decision": "rejected",
            "family_tag": "QUALITY",
        }
    )

    assert memory.family_saturation("QUALITY", threshold=2) is True
    assert memory.family_saturation("QUALITY", threshold=3) is False
    assert memory.family_saturation("MOMENTUM", threshold=1) is False


def test_retriever_gracefully_degrades_without_router(tmp_path: Path):
    memory = RepairMemory(tmp_path / "repair_memory.sqlite")
    memory.add_record(
        {
            "record_id": "accepted-quality",
            "expression": "group_rank(ts_mean(volume, 21), industry)",
            "symptom_tags": ["high_turnover"],
            "accept_decision": "accepted",
            "family_tag": "QUALITY",
            "recommended_directions": ["smooth fast volume"],
        }
    )
    diagnosis = DiagnosisReport(
        primary_symptom="high_turnover",
        secondary_symptoms=[],
        root_causes=[],
        repair_priorities=[],
        do_not_change=[],
        raw={},
    )

    result = Retriever(memory=memory, router=None).retrieve(
        diagnosis=diagnosis,
        expression="rank(ts_delta(volume, 5))",
        family_tag="QUALITY",
    )

    assert result["recommended_directions"] == ["smooth fast volume"]
    assert result["family_saturated"] is False
    assert result["retrieval_summary"] == ""


def test_retriever_surfaces_theme_and_math_saturation(tmp_path: Path):
    memory = RepairMemory(tmp_path / "repair_memory.sqlite")
    expression = "rank(ts_delta(volume, 5))"
    math_profile = analyze_math_profile(expression)
    economic_profile = infer_economic_profile(expression, fields=math_profile["fields"])

    for index in range(3):
        memory.add_record(
            {
                "record_id": f"accepted-liquidity-{index}",
                "timestamp": _timestamp(days_ago=index),
                "expression": expression,
                "symptom_tags": ["high_turnover"],
                "accept_decision": "accepted",
                "family_tag": "QUALITY",
                "math_profile": math_profile,
                "economic_profile": economic_profile,
            }
        )

    diagnosis = DiagnosisReport(
        primary_symptom="high_turnover",
        secondary_symptoms=[],
        root_causes=[],
        repair_priorities=[],
        do_not_change=[],
        raw={},
    )

    result = Retriever(memory=memory, router=None).retrieve(
        diagnosis=diagnosis,
        expression=expression,
        family_tag="QUALITY",
    )

    assert result["theme_saturated"] is True
    assert result["math_saturated"] is True
    assert result["saturated_themes"] == ["liquidity"]
    assert result["saturated_math_signatures"] == [math_profile["family_signature"]]


def test_hypothesis_agent_repair_payload_uses_attached_retriever(tmp_path: Path):
    class FakeRetriever:
        def retrieve(self, diagnosis, expression, family_tag=None):
            return {
                "positive": [],
                "negative": [],
                "forbidden_directions": ["avoid short deltas"],
                "recommended_directions": ["smooth fast volume"],
                "family_saturated": True,
                "retrieval_summary": "Smoothing worked before. Avoid short deltas.",
            }

    diagnosis = DiagnosisReport(
        primary_symptom="high_turnover",
        secondary_symptoms=["low_fitness"],
        root_causes=[],
        repair_priorities=[
            {"rank": 1, "target_metric": "turnover", "suggested_action_type": "smooth_fast_signal"},
            {"rank": 2, "target_metric": "fitness", "suggested_action_type": "add_quality_anchor"},
        ],
        do_not_change=["volume thesis"],
        raw={"forbidden_directions": ["legacy raw direction should not be used when retriever is attached"]},
    )
    agent = HypothesisAgent(
        kb=KnowledgeBase(tmp_path / "kb.db"),
        cache=LLMCache(tmp_path / "cache"),
        taxonomy={"QUALITY": {}},
    )
    agent.retriever = FakeRetriever()

    payload = agent._request_payload(
        "repair objective",
        "QUALITY",
        2,
        repair_context={
            "expression": "rank(ts_delta(volume, 5))",
            "_category": "QUALITY",
            "failedChecks": ["HIGH_TURNOVER"],
            "gate": {"reasons": ["turnover too high"]},
        },
        diagnosis=diagnosis,
    )

    user_dict = json.loads(payload["messages"][1]["content"])
    assert user_dict["forbidden_directions"] == ["avoid short deltas"]
    assert user_dict["recommended_directions"] == ["smooth fast volume"]
    assert user_dict["retrieval_summary"] == "Smoothing worked before. Avoid short deltas."
    assert "This alpha family is saturated" in user_dict["requirement"]


def test_rule_repair_prioritizes_turnover_smoothing(tmp_path: Path):
    agent = _agent(tmp_path)

    candidates = agent._rule_based_repair_candidates(
        {
            "expression": "rank(ts_delta(volume, 5))",
            "failedChecks": ["HIGH_TURNOVER"],
        },
        category="QUALITY",
        n=3,
    )

    assert candidates
    assert candidates[0].origin_refs == ["rule_repair", "rule_turnover_smooth"]


def test_rule_repair_avoids_duplicate_group_rank_wrapper(tmp_path: Path):
    agent = _agent(tmp_path)
    parent = "group_rank(ts_rank(returns, 63), industry)"

    candidates = agent._rule_based_repair_candidates(
        {
            "expression": parent,
            "failedChecks": ["LOW_SHARPE"],
        },
        category="MOMENTUM",
        n=3,
    )

    expressions = [candidate.expression for candidate in candidates]
    assert f"group_rank({parent}, industry)" not in expressions


def test_rule_repair_filters_invalid_parent_wrappers_and_keeps_valid_escape(tmp_path: Path):
    agent = _agent(tmp_path)
    validator = ExpressionValidator()

    candidates = agent._rule_based_repair_candidates(
        {
            "expression": "magic_alpha(close)",
            "failedChecks": ["SELF_CORRELATION"],
        },
        category="MOMENTUM",
        n=3,
    )

    assert candidates
    assert candidates[0].origin_refs == ["rule_repair", "rule_cross_family_escape"]
    assert all(validator.validate(candidate.expression).is_valid for candidate in candidates)


def test_rule_repair_respects_account_blocked_operators(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("BRAIN_OPERATOR_DENYLIST", "delay")
    agent = _agent(tmp_path)

    candidates = agent._rule_based_repair_candidates(
        {
            "expression": "rank(ts_delta(volume, 5))",
            "failedChecks": ["HIGH_TURNOVER"],
        },
        category="QUALITY",
        n=4,
    )

    assert candidates
    assert all("delay(" not in candidate.expression for candidate in candidates)


def test_complex_repair_without_agent_blocks_without_semantic_memory(tmp_path: Path):
    agent = _agent(tmp_path)

    candidates = agent.generate_batch(
        "repair objective",
        category="MOMENTUM",
        n=2,
        repair_context={
            "expression": "magic_alpha(close)",
            "failedChecks": ["SELF_CORRELATION"],
        },
    )

    assert candidates == []
    assert agent.last_repair_quality["route"] == "blocked"
    assert "semantic_memory_required" in agent.last_repair_quality["reasons"]
