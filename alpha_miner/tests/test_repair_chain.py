"""Tests for LangChain Tool-calling Agent RepairChain."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from alpha_miner.modules.m_repair_memory import RepairMemory
from alpha_miner.modules.m_repair_chain import RepairChain, build_tools, _parse_agent_output


@pytest.fixture
def tmp_memory(tmp_path):
    return RepairMemory(tmp_path / "repair_memory.db")


@pytest.fixture
def chain(tmp_memory):
    return RepairChain(memory=tmp_memory, openai_api_key="test-key", model_id="gpt-4o")


# ---------------------------------------------------------------------------
# build_tools — tool creation and naming
# ---------------------------------------------------------------------------

def test_build_tools_returns_four_tools(tmp_memory):
    tools = build_tools(tmp_memory, None, None)
    names = [t.name for t in tools]
    assert "diagnose_alpha" in names
    assert "retrieve_repair_memory" in names
    assert "validate_expression" in names
    assert "generate_repair_variants" in names


# ---------------------------------------------------------------------------
# diagnose_alpha tool
# ---------------------------------------------------------------------------

def test_diagnose_alpha_low_sharpe(tmp_memory):
    tools = build_tools(tmp_memory, None, None)
    diagnose = next(t for t in tools if t.name == "diagnose_alpha")
    result = json.loads(diagnose.invoke({
        "expression": "rank(returns)",
        "sharpe": "0.2",
        "fitness": "0.6",
        "turnover": "0.3",
        "failed_checks": "SHARPE",
    }))
    assert result["primary_symptom"] == "low_sharpe"
    assert "repair_strategy" in result


def test_diagnose_alpha_high_turnover(tmp_memory):
    tools = build_tools(tmp_memory, None, None)
    diagnose = next(t for t in tools if t.name == "diagnose_alpha")
    result = json.loads(diagnose.invoke({
        "expression": "rank(delta(close,1))",
        "sharpe": "0.8",
        "fitness": "0.7",
        "turnover": "0.85",
        "failed_checks": "TURNOVER",
    }))
    assert result["primary_symptom"] == "high_turnover"
    assert "ts_decay_linear" in result["repair_strategy"] or "window" in result["repair_strategy"].lower()


def test_diagnose_alpha_preserves_group_rank(tmp_memory):
    tools = build_tools(tmp_memory, None, None)
    diagnose = next(t for t in tools if t.name == "diagnose_alpha")
    result = json.loads(diagnose.invoke({
        "expression": "group_rank(rank(returns), industry)",
        "sharpe": "0.3",
        "fitness": "0.4",
        "turnover": "0.2",
        "failed_checks": "SHARPE,FITNESS",
    }))
    assert "industry neutralization" in result["do_not_change"]


# ---------------------------------------------------------------------------
# retrieve_repair_memory tool — empty and non-empty
# ---------------------------------------------------------------------------

def test_retrieve_memory_empty(tmp_memory):
    tools = build_tools(tmp_memory, None, None)
    retrieve = next(t for t in tools if t.name == "retrieve_repair_memory")
    result = retrieve.invoke({"expression": "rank(returns)", "symptoms": "low_sharpe"})
    assert "No historical" in result


def test_retrieve_memory_returns_records(tmp_memory):
    tmp_memory.add_record({
        "expression": "group_rank(ts_mean(returns, 21), industry)",
        "symptom_tags": ["low_sharpe"],
        "repair_actions": ["added ts_mean"],
        "accept_decision": "accepted",
        "recommended_directions": ["add ts_mean smoothing"],
        "forbidden_directions": [],
        "family_tag": "",
    })
    tools = build_tools(tmp_memory, None, None)
    retrieve = next(t for t in tools if t.name == "retrieve_repair_memory")
    result = retrieve.invoke({"expression": "group_rank(returns, industry)", "symptoms": "low_sharpe", "k": 3})
    assert "[SUCCESS]" in result
    assert "ts_mean" in result


# ---------------------------------------------------------------------------
# validate_expression tool
# ---------------------------------------------------------------------------

def test_validate_expression_no_validator(tmp_memory):
    tools = build_tools(tmp_memory, None, None)
    validate = next(t for t in tools if t.name == "validate_expression")
    result = validate.invoke({"expression": "rank(returns)"})
    assert "VALID" in result


def test_validate_expression_with_mock_validator(tmp_memory):
    mock_validator = MagicMock()
    mock_result = MagicMock()
    mock_result.valid = False
    mock_result.errors = ["unknown operator 'fake_op'"]
    mock_validator.validate.return_value = mock_result

    tools = build_tools(tmp_memory, None, mock_validator)
    validate = next(t for t in tools if t.name == "validate_expression")
    result = validate.invoke({"expression": "fake_op(returns)"})
    assert "INVALID" in result
    assert "fake_op" in result


def test_validate_expression_passes_with_valid(tmp_memory):
    mock_validator = MagicMock()
    mock_result = MagicMock()
    mock_result.valid = True
    mock_result.errors = []
    mock_validator.validate.return_value = mock_result

    tools = build_tools(tmp_memory, None, mock_validator)
    validate = next(t for t in tools if t.name == "validate_expression")
    result = validate.invoke({"expression": "rank(returns)"})
    assert result == "VALID"


# ---------------------------------------------------------------------------
# generate_repair_variants tool
# ---------------------------------------------------------------------------

def test_generate_repair_variants_returns_hint(tmp_memory):
    tools = build_tools(tmp_memory, None, None)
    generate = next(t for t in tools if t.name == "generate_repair_variants")
    diagnosis = json.dumps({"primary_symptom": "low_sharpe", "repair_strategy": "add neutralization", "do_not_change": []})
    result = generate.invoke({
        "expression": "rank(returns)",
        "diagnosis_json": diagnosis,
        "memory_context": "No past repairs",
        "n": 3,
    })
    assert "rank(returns)" in result
    assert "low_sharpe" in result
    assert "3" in result


# ---------------------------------------------------------------------------
# _parse_agent_output
# ---------------------------------------------------------------------------

def test_parse_agent_output_full_json():
    output = json.dumps({
        "diagnosis": {"primary_symptom": "low_sharpe", "root_cause": "noisy signal"},
        "candidates": [
            {"expression": "group_rank(ts_mean(returns,21),industry)", "hypothesis": "smooth", "fix_applied": "added ts_mean"},
        ],
    })
    candidates, diagnosis = _parse_agent_output(output, "MOMENTUM")
    assert len(candidates) == 1
    assert candidates[0]["expression"] == "group_rank(ts_mean(returns,21),industry)"
    assert candidates[0]["category"] == "MOMENTUM"
    assert "langchain_agent" in candidates[0]["origin_refs"]
    assert diagnosis["primary_symptom"] == "low_sharpe"


def test_parse_agent_output_array_fallback():
    output = 'Here are the candidates:\n' + json.dumps([
        {"expression": "rank(ts_mean(returns,21))", "hypothesis": "smoothed", "fix_applied": "ts_mean"},
    ])
    candidates, diagnosis = _parse_agent_output(output, "QUALITY")
    assert len(candidates) == 1
    assert candidates[0]["category"] == "QUALITY"


def test_parse_agent_output_no_json():
    candidates, diagnosis = _parse_agent_output("Sorry, I could not repair this.", "MOMENTUM")
    assert candidates == []
    assert diagnosis == {}


# ---------------------------------------------------------------------------
# record_outcome — persists to memory and resets agent
# ---------------------------------------------------------------------------

def test_record_outcome_resets_agent(chain):
    chain._agent_executor = MagicMock()  # simulate initialized agent
    chain.record_outcome(
        expression="rank(returns)",
        diagnosis={"primary_symptom": "low_sharpe"},
        candidates=[{"origin_refs": ["langchain_agent", "added ts_mean"]}],
        accepted=True,
    )
    assert chain._agent_executor is None  # must be reset
    records = chain.memory.get_recent(limit=10)
    assert records[0]["accept_decision"] == "accepted"


# ---------------------------------------------------------------------------
# run() — mocked AgentExecutor
# ---------------------------------------------------------------------------

def test_run_returns_candidates_from_agent(chain):
    agent_output = json.dumps({
        "diagnosis": {"primary_symptom": "low_sharpe", "root_cause": "noisy"},
        "candidates": [
            {"expression": "group_rank(ts_mean(returns,21),industry)", "hypothesis": "smoothed", "fix_applied": "ts_mean"},
            {"expression": "group_rank(ts_rank(returns,63),industry)", "hypothesis": "longer", "fix_applied": "longer window"},
        ],
    })
    mock_executor = MagicMock()
    mock_executor.invoke.return_value = {"output": agent_output, "intermediate_steps": []}
    chain._agent_executor = mock_executor

    candidates, diagnosis = chain.run(
        expression="group_rank(returns, industry)",
        metrics={"sharpe": 0.3, "fitness": 0.4, "turnover": 0.25},
        failed_checks=["SHARPE"],
        gate_reasons=["sharpe below 0.5"],
        n=2,
        category="MOMENTUM",
    )
    assert len(candidates) == 2
    assert candidates[0]["category"] == "MOMENTUM"
    assert diagnosis["primary_symptom"] == "low_sharpe"


def test_run_returns_empty_on_agent_failure(chain):
    mock_executor = MagicMock()
    mock_executor.invoke.side_effect = RuntimeError("LLM error")
    chain._agent_executor = mock_executor

    candidates, diagnosis = chain.run(
        expression="rank(returns)",
        metrics={"sharpe": 0.2},
        failed_checks=["SHARPE"],
        gate_reasons=[],
    )
    assert candidates == []
    assert diagnosis == {}
