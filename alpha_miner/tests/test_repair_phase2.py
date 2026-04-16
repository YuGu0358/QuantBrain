import json
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from pathlib import Path

from alpha_miner.modules.llm_cache import LLMCache
from alpha_miner.modules.m1_knowledge_base import KnowledgeBase
from alpha_miner.modules.m2_hypothesis_agent import HypothesisAgent
from alpha_miner.modules.m_diagnoser import DiagnosisReport
from alpha_miner.modules.m_repair_memory import RepairMemory
from alpha_miner.modules.m_retriever import Retriever


def _timestamp(days_ago: int = 0) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()


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
