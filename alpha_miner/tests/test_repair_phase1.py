import json
from argparse import Namespace
from pathlib import Path

from alpha_miner.modules.llm_router import LLMProvider
from alpha_miner.modules.llm_cache import LLMCache
from alpha_miner.modules.m1_knowledge_base import KnowledgeBase
from alpha_miner.modules.m_diagnoser import DiagnosisReport
from alpha_miner.modules.m_diagnoser import Diagnoser
from alpha_miner.modules.m_distiller import Distiller
from alpha_miner.modules.m2_hypothesis_agent import HypothesisAgent
from alpha_miner.modules.m_repair_memory import RepairMemory


class FakeRouter:
    def __init__(self, raw_response: dict, role: str = "repair", model_id: str = "gpt-5.4-2026-03-05"):
        self.raw_response = raw_response
        self.provider = LLMProvider(
            name=f"fake_{role}",
            role=role,
            model_id=model_id,
            client_type="openai_compat",
        )
        self._providers = {(self.provider.name, self.provider.role): self.provider}
        self.spent_usd = 0.0
        self.picked_roles = []
        self.payloads = []

    def pick(self, role: str):
        self.picked_roles.append(role)
        if role != self.provider.role:
            raise ValueError(f"No provider for {role}")
        return self.provider

    def _call_provider(self, provider, request_payload):
        self.payloads.append(request_payload)
        return json.dumps(self.raw_response), 10, 20, 30.0

    def record_result(self, *args, **kwargs):
        pass

    def save_state(self, path):
        path.write_text(json.dumps({"providers": {}, "spent_usd": self.spent_usd}), encoding="utf-8")


class MultiRoleFakeRouter:
    def __init__(self, raw_responses: dict[str, dict | str], model_id: str = "gpt-5.4-2026-03-05"):
        self.raw_responses = dict(raw_responses)
        self.providers = {}
        for role in raw_responses:
            provider = LLMProvider(
                name=f"fake_{role}",
                role=role,
                model_id=model_id,
                client_type="openai_compat",
            )
            self.providers[role] = provider
        self._providers = {(provider.name, provider.role): provider for provider in self.providers.values()}
        self.spent_usd = 0.0
        self.picked_roles = []
        self.payloads = []

    def pick(self, role: str):
        self.picked_roles.append(role)
        provider = self.providers.get(role)
        if provider is None:
            raise ValueError(f"No provider for {role}")
        return provider

    def _call_provider(self, provider, request_payload):
        self.payloads.append((provider.role, request_payload))
        raw_response = self.raw_responses[provider.role]
        if isinstance(raw_response, str):
            return raw_response, 10, 20, 30.0
        return json.dumps(raw_response), 10, 20, 30.0

    def record_result(self, *args, **kwargs):
        pass

    def save_state(self, path):
        path.write_text(json.dumps({"providers": {}, "spent_usd": self.spent_usd}), encoding="utf-8")


def test_diagnoser_falls_back_without_router_and_counts_complexity():
    report = Diagnoser(router=None).diagnose(
        expression="group_rank(ts_mean(rank(close), 20) + ts_delta(volume, 5), industry)",
        metrics={
            "sharpe": 0.8,
            "fitness": 0.8,
            "turnover": 0.2,
            "max_abs_correlation": 0.2,
        },
        failed_checks=[],
        gate_reasons=[],
    )

    assert report.primary_symptom == "low_sharpe"
    assert report.raw["metrics"]["complexity"] == 4
    assert report.repair_priorities[0]["target_metric"] == "sharpe"


def test_diagnoser_uses_repair_provider_and_parses_json_response():
    raw_response = {
        "primary_symptom": "high_turnover",
        "secondary_symptoms": ["low_fitness"],
        "root_causes": ["fast ts_delta window"],
        "repair_priorities": [
            {
                "rank": 1,
                "target_metric": "turnover",
                "suggested_action_type": "smooth_fast_signal",
                "keep_constraints": ["preserve volume thesis"],
            }
        ],
        "do_not_change": ["volume"],
    }
    router = FakeRouter(raw_response=raw_response)

    report = Diagnoser(router).diagnose(
        expression="rank(ts_delta(volume, 5))",
        metrics={
            "sharpe": 1.2,
            "fitness": 0.4,
            "turnover": 0.9,
            "max_abs_correlation": 0.2,
        },
        failed_checks=["HIGH_TURNOVER"],
        gate_reasons=["turnover too high"],
    )

    assert report.primary_symptom == "high_turnover"
    assert report.secondary_symptoms == ["low_fitness"]
    assert router.picked_roles == ["repair"]
    payload = router.payloads[0]
    assert payload["max_completion_tokens"] == 600
    assert "temperature" not in payload
    assert payload["messages"][0]["role"] == "system"
    assert payload["messages"][1]["role"] == "user"
    prompt_input = json.loads(payload["messages"][1]["content"])
    assert prompt_input["metrics"]["complexity"] == 2


def test_diagnoser_normalizes_primary_symptom_aliases_from_diagnose_role():
    router = FakeRouter(
        raw_response={
            "primary_symptom": "High Turnover",
            "secondary_symptoms": "low fitness",
            "root_causes": ["fast ts_delta window"],
            "repair_priorities": [{"rank": 1, "target_metric": "turnover"}],
            "do_not_change": ["volume"],
        },
        role="diagnose",
    )

    report = Diagnoser(router).diagnose(
        expression="rank(ts_delta(volume, 5))",
        metrics={
            "sharpe": 1.2,
            "fitness": 0.4,
            "turnover": 0.9,
            "max_abs_correlation": 0.2,
        },
        failed_checks=["HIGH_TURNOVER"],
        gate_reasons=["turnover too high"],
    )

    assert report.primary_symptom == "high_turnover"
    assert report.secondary_symptoms == ["low_fitness"]
    assert router.picked_roles == ["diagnose"]
    assert report.raw.get("fallback") is not True


def test_diagnoser_preserves_llm_fallback_metadata_when_repair_rescues_primary_failure():
    router = MultiRoleFakeRouter(
        raw_responses={
            "diagnose": {
                "primary_symptom": "bad symptom",
                "secondary_symptoms": [],
                "root_causes": ["unstructured output"],
                "repair_priorities": [],
                "do_not_change": [],
            },
            "repair": {
                "primary_symptom": "low_sharpe",
                "secondary_symptoms": ["high_turnover"],
                "root_causes": ["weak residual signal"],
                "repair_priorities": [{"rank": 1, "target_metric": "sharpe"}],
                "do_not_change": ["industry"],
            },
        }
    )

    report = Diagnoser(router).diagnose(
        expression="group_rank(ts_mean(rank(close), 20), industry)",
        metrics={
            "sharpe": 0.8,
            "fitness": 0.6,
            "turnover": 0.3,
            "max_abs_correlation": 0.2,
        },
        failed_checks=[],
        gate_reasons=[],
    )

    assert report.primary_symptom == "low_sharpe"
    assert router.picked_roles == ["diagnose", "repair"]
    assert report.raw["llm_fallback"] is True
    assert report.raw["primary_provider"] == "fake_diagnose"
    assert report.raw["fallback_provider"] == "fake_repair"
    assert "Invalid primary_symptom" in report.raw["primary_error"]


def test_repair_memory_persists_records_and_filters_by_symptom_overlap(tmp_path: Path):
    memory = RepairMemory(tmp_path / "repair_memory.sqlite")
    memory.add_record(
        {
            "record_id": "rejected-1",
            "expression": "rank(close)",
            "symptom_tags": ["low_sharpe", "high_turnover"],
            "accept_decision": "rejected",
            "forbidden_directions": ["do not shorten windows", "avoid pure price rank"],
            "recommended_directions": [],
            "metrics": {"sharpe": 0.4},
        }
    )
    memory.add_record(
        {
            "record_id": "accepted-1",
            "expression": "group_rank(ts_mean(close, 21), industry)",
            "symptom_tags": ["low_sharpe"],
            "accept_decision": "accepted",
            "forbidden_directions": [],
            "recommended_directions": ["add industry neutralization"],
            "metrics": {"sharpe": 1.3},
        }
    )

    assert memory.get_forbidden_for_symptoms(["low_sharpe"]) == [
        "do not shorten windows",
        "avoid pure price rank",
    ]
    assert memory.get_forbidden_for_symptoms(["high_corrlib"]) == []
    assert memory.get_positive_for_symptoms(["low_sharpe"]) == [
        {
            "expression": "group_rank(ts_mean(close, 21), industry)",
            "recommended_directions": ["add industry neutralization"],
        }
    ]
    assert [record["record_id"] for record in memory.get_recent(limit=2)] == [
        "accepted-1",
        "rejected-1",
    ]


def test_distiller_writes_memory_record_from_llm_lessons(tmp_path: Path):
    memory = RepairMemory(tmp_path / "repair_memory.sqlite")
    router = FakeRouter(
        raw_response={
            "recommended_directions": ["smooth fast components"],
            "forbidden_directions": ["do not add another short delta"],
            "reusable_patterns": ["replace ts_delta(x, 5) with ts_mean(x, 21)"],
            "regime_lessons": ["high turnover repairs favored smoothing"],
            "symptom_tags": ["high_turnover"],
        },
        role="distill",
        model_id="gpt-4o-mini",
    )
    diagnosis = DiagnosisReport(
        primary_symptom="high_turnover",
        secondary_symptoms=["low_fitness"],
        root_causes=["fast component"],
        repair_priorities=[],
        do_not_change=["volume"],
        raw={},
    )

    result = Distiller(router=router, memory=memory).distill(
        original_expression="rank(ts_delta(volume, 5))",
        diagnosis=diagnosis,
        tried_candidates=[
            {
                "expression": "rank(ts_delta(volume, 5))",
                "sharpe": 0.6,
                "fitness": 0.3,
                "turnover": 0.9,
                "accepted": False,
            },
            {
                "expression": "rank(ts_mean(volume, 21))",
                "sharpe": 1.4,
                "fitness": 0.8,
                "turnover": 0.4,
                "accepted": True,
            },
        ],
        accepted_expression="rank(ts_mean(volume, 21))",
    )

    assert result["recommended_directions"] == ["smooth fast components"]
    payload = router.payloads[0]
    assert payload["max_completion_tokens"] == 800
    prompt_input = json.loads(payload["messages"][1]["content"])
    assert prompt_input["diagnosis"]["primary_symptom"] == "high_turnover"

    recent = memory.get_recent(limit=1)[0]
    assert recent["expression"] == "rank(ts_delta(volume, 5))"
    assert recent["accept_decision"] == "accepted"
    assert recent["symptom_tags"] == ["high_turnover", "low_fitness"]
    assert recent["metrics"] == {"sharpe": 1.4, "fitness": 0.8, "turnover": 0.4}


def test_distiller_gracefully_degrades_without_router(tmp_path: Path):
    memory = RepairMemory(tmp_path / "repair_memory.sqlite")
    diagnosis = DiagnosisReport(
        primary_symptom="low_sharpe",
        secondary_symptoms=[],
        root_causes=[],
        repair_priorities=[],
        do_not_change=[],
        raw={},
    )

    result = Distiller(router=None, memory=memory).distill(
        original_expression="rank(close)",
        diagnosis=diagnosis,
        tried_candidates=[],
        accepted_expression=None,
    )

    assert result == {
        "recommended_directions": [],
        "forbidden_directions": [],
        "reusable_patterns": [],
        "regime_lessons": [],
        "symptom_tags": [],
    }
    assert memory.get_recent(limit=1)[0]["accept_decision"] == "rejected"


def test_hypothesis_agent_repair_payload_uses_diagnosis_and_memory(tmp_path: Path):
    memory = RepairMemory(tmp_path / "repair_memory.sqlite")
    memory.add_record(
        {
            "record_id": "rejected-1",
            "expression": "rank(ts_delta(volume, 5))",
            "symptom_tags": ["high_turnover"],
            "accept_decision": "rejected",
            "forbidden_directions": ["avoid another short delta"],
        }
    )
    memory.add_record(
        {
            "record_id": "accepted-1",
            "expression": "group_rank(ts_mean(volume, 21), industry)",
            "symptom_tags": ["high_turnover"],
            "accept_decision": "accepted",
            "recommended_directions": ["smooth fast volume components"],
        }
    )
    diagnosis = DiagnosisReport(
        primary_symptom="high_turnover",
        secondary_symptoms=["low_fitness"],
        root_causes=[],
        repair_priorities=[
            {"rank": 1, "target_metric": "turnover", "suggested_action_type": "smooth_fast_signal"},
            {"rank": 2, "target_metric": "fitness", "suggested_action_type": "add_quality_anchor"},
            {"rank": 3, "target_metric": "correlation", "suggested_action_type": "change_family"},
        ],
        do_not_change=["volume thesis"],
        raw={"forbidden_directions": ["do not remove volume"]},
    )
    agent = HypothesisAgent(
        kb=KnowledgeBase(tmp_path / "kb.db"),
        cache=LLMCache(tmp_path / "cache"),
        taxonomy={"QUALITY": {}},
        repair_memory=memory,
    )

    payload = agent._request_payload(
        "repair objective",
        "QUALITY",
        2,
        repair_context={
            "expression": "rank(ts_delta(volume, 5))",
            "failedChecks": ["HIGH_TURNOVER"],
            "gate": {"reasons": ["turnover too high"]},
        },
        diagnosis=diagnosis,
    )

    user_dict = json.loads(payload["messages"][1]["content"])
    assert user_dict["diagnosis"] == {
        "primary_symptom": "high_turnover",
        "repair_priorities": diagnosis.repair_priorities[:2],
        "do_not_change": ["volume thesis"],
    }
    assert "Primary symptom: high_turnover" in user_dict["requirement"]
    assert user_dict["forbidden_directions"] == [
        "do not remove volume",
        "avoid another short delta",
    ]
    assert user_dict["positive_memory_examples"] == [
        {
            "expression": "group_rank(ts_mean(volume, 21), industry)",
            "recommended_directions": ["smooth fast volume components"],
        }
    ]


def test_main_repair_pipeline_passes_diagnosis_and_distills_memory(tmp_path: Path, monkeypatch):
    import alpha_miner.main as main_module

    repair_context_path = tmp_path / "repair_context.json"
    repair_context_path.write_text(
        json.dumps(
            {
                "expression": "rank(ts_delta(volume, 5))",
                "_category": "QUALITY",
                "metrics": {
                    "isSharpe": 0.7,
                    "isFitness": 0.3,
                    "turnover": 0.9,
                    "max_abs_correlation": 0.2,
                },
                "failedChecks": ["HIGH_TURNOVER"],
                "gate": {"reasons": ["turnover too high"]},
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "runs" / "round-1"
    router = FakeRouter(
        raw_response={
            "primary_symptom": "high_turnover",
            "secondary_symptoms": ["low_fitness"],
            "root_causes": ["fast delta"],
            "repair_priorities": [{"rank": 1, "target_metric": "turnover"}],
            "do_not_change": ["volume"],
        }
    )
    distill_response = {
        "recommended_directions": ["smooth fast signal"],
        "forbidden_directions": ["avoid short deltas"],
        "reusable_patterns": [],
        "regime_lessons": [],
        "symptom_tags": ["high_turnover"],
    }

    def fake_pick(role: str):
        router.picked_roles.append(role)
        if role == "repair":
            router.raw_response = {
                "primary_symptom": "high_turnover",
                "secondary_symptoms": ["low_fitness"],
                "root_causes": ["fast delta"],
                "repair_priorities": [{"rank": 1, "target_metric": "turnover"}],
                "do_not_change": ["volume"],
            }
            router.provider.role = role
            return router.provider
        if role == "distill":
            router.raw_response = distill_response
            router.provider.role = role
            return router.provider
        raise ValueError(role)

    seen = {}

    def fake_generate_batch(self, objective, category, n=10, use_llm=False, repair_context=None, diagnosis=None):
        seen["repair_memory_attached"] = self.repair_memory is not None
        seen["retriever_attached"] = self.retriever is not None
        seen["diagnosis"] = diagnosis
        return []

    router.pick = fake_pick
    monkeypatch.setattr(main_module.LLMRouter, "from_yaml", classmethod(lambda cls: router))
    monkeypatch.setattr(main_module.HypothesisAgent, "generate_batch", fake_generate_batch)
    monkeypatch.setattr(
        main_module,
        "parse_args",
        lambda: Namespace(
            mode="generate",
            objective="repair objective",
            output_dir=str(output_dir),
            batch_size=1,
            rounds=1,
            category=None,
            config=None,
            concurrency=1,
            resume_from=None,
            repair_context=str(repair_context_path),
            use_llm=False,
            sim_settings=None,
            verbose="true",
        ),
    )

    main_module.main()

    assert seen["repair_memory_attached"] is True
    assert seen["retriever_attached"] is True
    assert seen["diagnosis"].primary_symptom == "high_turnover"
    assert "repair" in router.picked_roles
    assert "distill" in router.picked_roles
    progress_lines = [
        json.loads(line)
        for line in (output_dir / "progress.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(line["stage"] == "diagnosis" and line["primary_symptom"] == "high_turnover" for line in progress_lines)
    assert any(line["stage"] == "distilled" and line["accepted"] is False for line in progress_lines)
    memory = RepairMemory(output_dir.parent / "repair_memory.db")
    assert memory.get_recent(limit=1)[0]["forbidden_directions"] == ["avoid short deltas"]
