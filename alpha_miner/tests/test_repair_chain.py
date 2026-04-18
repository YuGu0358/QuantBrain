"""Tests for LangChain-based RepairChain (m_repair_chain.py)."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from alpha_miner.modules.m_repair_memory import RepairMemory
from alpha_miner.modules.m_repair_chain import RepairChain


@pytest.fixture
def tmp_memory(tmp_path):
    return RepairMemory(tmp_path / "repair_memory.db")


@pytest.fixture
def chain(tmp_memory):
    return RepairChain(memory=tmp_memory, openai_api_key="test-key", model_id="gpt-4o")


# ---------------------------------------------------------------------------
# _format_retrieved
# ---------------------------------------------------------------------------

def test_format_retrieved_empty(chain):
    result = chain._format_retrieved([])
    assert "No historical" in result


def test_format_retrieved_accepted(chain):
    records = [{
        "accept_decision": "accepted",
        "expression": "rank(returns)",
        "symptom_tags": ["low_sharpe"],
        "recommended_directions": ["add ts_decay_linear"],
        "forbidden_directions": [],
    }]
    result = chain._format_retrieved(records)
    assert "[SUCCESS]" in result
    assert "rank(returns)" in result
    assert "add ts_decay_linear" in result


def test_format_retrieved_rejected(chain):
    records = [{
        "accept_decision": "rejected",
        "expression": "rank(volume)",
        "symptom_tags": ["high_turnover"],
        "recommended_directions": [],
        "forbidden_directions": ["shorten windows"],
    }]
    result = chain._format_retrieved(records)
    assert "[FAILED]" in result
    assert "shorten windows" in result


# ---------------------------------------------------------------------------
# record_outcome — persists to RepairMemory and invalidates FAISS cache
# ---------------------------------------------------------------------------

def test_record_outcome_accepted(chain):
    chain._index_cache = ("fake_index", [])  # simulate warm cache
    candidates = [{"origin_refs": ["langchain_repair", "added ts_decay_linear"], "expression": "rank(returns)"}]
    chain.record_outcome(
        expression="rank(volume)",
        diagnosis={"primary_symptom": "high_turnover"},
        candidates=candidates,
        accepted=True,
    )
    records = chain.memory.get_recent(limit=10)
    assert len(records) == 1
    assert records[0]["accept_decision"] == "accepted"
    assert records[0]["symptom_tags"] == ["high_turnover"]
    assert "added ts_decay_linear" in records[0]["recommended_directions"]
    # Cache must be invalidated
    assert chain._index_cache is None


def test_record_outcome_rejected(chain):
    candidates = [{"origin_refs": ["langchain_repair", "shortened window"], "expression": "rank(volume)"}]
    chain.record_outcome(
        expression="rank(volume)",
        diagnosis={"primary_symptom": "low_sharpe"},
        candidates=candidates,
        accepted=False,
    )
    records = chain.memory.get_recent(limit=10)
    assert records[0]["accept_decision"] == "rejected"
    assert "shortened window" in records[0]["forbidden_directions"]


# ---------------------------------------------------------------------------
# _semantic_retrieve — returns empty when no records
# ---------------------------------------------------------------------------

def test_semantic_retrieve_empty_memory(chain):
    result = chain._semantic_retrieve("rank(returns)", ["low_sharpe"])
    assert result == []


def test_semantic_retrieve_falls_back_without_embeddings(chain):
    """When embeddings is None, should return first k records from DB."""
    chain.memory.add_record({
        "expression": "rank(returns)",
        "symptom_tags": ["low_sharpe"],
        "repair_actions": [],
        "accept_decision": "accepted",
        "recommended_directions": ["neutralize"],
        "forbidden_directions": [],
        "family_tag": "",
    })
    # embeddings not initialized (chain._ensure_chain not called)
    result = chain._semantic_retrieve("ts_rank(returns, 20)", ["low_sharpe"])
    # Should fall back to recent records (k=5) since embeddings is None
    assert len(result) <= 5


# ---------------------------------------------------------------------------
# run() — mocked LangChain chain
# ---------------------------------------------------------------------------

def test_run_returns_candidates(chain):
    mock_result = {
        "diagnosis": {
            "primary_symptom": "low_sharpe",
            "root_cause": "signal too noisy",
            "do_not_change": ["industry neutralization"],
        },
        "candidates": [
            {
                "expression": "group_rank(ts_mean(returns, 21), industry)",
                "hypothesis": "Smooth returns over 21 days to reduce noise.",
                "fix_applied": "added ts_mean smoothing",
            },
            {
                "expression": "group_rank(ts_rank(returns, 63), industry)",
                "hypothesis": "Use longer lookback to improve IS Sharpe.",
                "fix_applied": "extended lookback to 63 days",
            },
        ],
    }
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = mock_result
    chain._chain = mock_chain
    chain._embeddings = MagicMock()
    chain._embeddings.embed_documents.return_value = []
    chain._embeddings.embed_query.return_value = []

    candidates, diagnosis = chain.run(
        expression="group_rank(returns, industry)",
        metrics={"sharpe": 0.3, "fitness": 0.4, "turnover": 0.25},
        failed_checks=["SHARPE"],
        gate_reasons=["sharpe below 0.5"],
        n=2,
        category="MOMENTUM",
    )

    assert len(candidates) == 2
    assert candidates[0]["expression"] == "group_rank(ts_mean(returns, 21), industry)"
    assert candidates[0]["category"] == "MOMENTUM"
    assert "langchain_repair" in candidates[0]["origin_refs"]
    assert diagnosis["primary_symptom"] == "low_sharpe"


def test_run_filters_empty_expressions(chain):
    mock_result = {
        "diagnosis": {"primary_symptom": "low_sharpe", "root_cause": "x", "do_not_change": []},
        "candidates": [
            {"expression": "", "hypothesis": "bad", "fix_applied": "nothing"},
            {"expression": "rank(returns)", "hypothesis": "good", "fix_applied": "fixed"},
        ],
    }
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = mock_result
    chain._chain = mock_chain
    chain._embeddings = MagicMock()
    chain._embeddings.embed_documents.return_value = []

    candidates, _ = chain.run(
        expression="rank(volume)",
        metrics={"sharpe": 0.2},
        failed_checks=["SHARPE"],
        gate_reasons=[],
        n=2,
    )
    # Empty expression filtered out
    assert len(candidates) == 1
    assert candidates[0]["expression"] == "rank(returns)"


def test_run_falls_back_gracefully_on_chain_error(chain):
    """If chain.invoke raises, run() should propagate (caller handles fallback)."""
    mock_chain = MagicMock()
    mock_chain.invoke.side_effect = RuntimeError("LLM unavailable")
    chain._chain = mock_chain
    chain._embeddings = MagicMock()
    chain._embeddings.embed_documents.return_value = []

    with pytest.raises(RuntimeError, match="LLM unavailable"):
        chain.run(
            expression="rank(returns)",
            metrics={"sharpe": 0.1},
            failed_checks=["SHARPE"],
            gate_reasons=[],
        )
