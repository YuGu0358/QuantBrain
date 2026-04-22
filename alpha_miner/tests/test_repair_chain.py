"""Tests for LangChain Tool-calling Agent RepairChain."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from alpha_miner.modules.m_repair_memory import RepairMemory
from alpha_miner.modules.m_repair_chain import RepairChain, build_tools, _parse_agent_output
from alpha_miner.modules.m_repair_intelligence import (
    analyze_math_profile,
    compare_repair,
    infer_economic_profile,
)


@pytest.fixture
def tmp_memory(tmp_path):
    return RepairMemory(tmp_path / "repair_memory.db")


@pytest.fixture
def chain(tmp_memory):
    return RepairChain(memory=tmp_memory, api_key="test-key", model_id="gpt-4o", openai_api_key="test-key")


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


def test_build_tools_returns_logic_tools(tmp_memory):
    tools = build_tools(tmp_memory, None, None)
    names = [t.name for t in tools]
    assert "analyze_factor_logic" in names
    assert "retrieve_logic_patterns" in names


# ---------------------------------------------------------------------------
# repair intelligence — math/economic extraction
# ---------------------------------------------------------------------------

def test_repair_intelligence_extracts_math_and_economic_logic():
    expression = "group_rank(ts_mean(volume, 21) / ts_mean(adv20, 63), industry)"

    math_profile = analyze_math_profile(expression)
    economic_profile = infer_economic_profile(expression, category="LIQUIDITY")

    assert math_profile["operators"] == ["group_rank", "ts_mean"]
    assert math_profile["windows"] == [21, 63]
    assert math_profile["group_fields"] == ["industry"]
    assert math_profile["has_group_neutralization"] is True
    assert math_profile["dominant_structure"] == "hybrid_group_time_series"
    assert economic_profile["theme"] == "liquidity"
    assert "liquidity" in economic_profile["thesis"].lower()


def test_repair_intelligence_classifies_repair_delta():
    parent = "rank(ts_delta(close, 5))"
    candidate = "group_rank(ts_mean(volume, 21) + ts_rank(cashflow_op / assets, 252), subindustry)"

    delta = compare_repair(parent, candidate)

    assert delta["smoothing"] is True
    assert delta["horizon_change"] is True
    assert delta["field_shift"] is True
    assert delta["group_change"] is True
    assert delta["quality_anchor"] is True
    assert delta["cross_family_escape"] is True
    assert delta["thesis_preserved"] is False


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


def test_retrieve_memory_compacts_to_top_success_and_failure(tmp_memory):
    for index in range(4):
        tmp_memory.add_record({
            "expression": f"group_rank(ts_mean(volume, {21 + index}), industry)",
            "symptom_tags": ["low_sharpe"],
            "accept_decision": "accepted",
            "recommended_directions": [f"accepted direction {index}"],
            "forbidden_directions": [],
            "family_tag": "",
        })
    for index in range(4):
        tmp_memory.add_record({
            "expression": f"group_rank(ts_delta(returns, {5 + index}), industry)",
            "symptom_tags": ["low_sharpe"],
            "accept_decision": "rejected",
            "recommended_directions": [],
            "forbidden_directions": [f"failed direction {index}"],
            "family_tag": "",
        })

    tools = build_tools(tmp_memory, None, None)
    retrieve = next(t for t in tools if t.name == "retrieve_repair_memory")
    result = retrieve.invoke({"expression": "group_rank(returns, industry)", "symptoms": "low_sharpe"})

    lines = [line for line in result.splitlines() if line.strip()]
    assert len(lines) <= 3
    assert sum("[SUCCESS]" in line for line in lines) <= 2
    assert sum("[FAILED]" in line for line in lines) <= 1


def test_repair_memory_persists_structured_logic_fields(tmp_path):
    memory = RepairMemory(tmp_path / "repair_memory.db")
    record = {
        "expression": "group_rank(ts_mean(volume, 21), industry)",
        "symptom_tags": ["high_turnover"],
        "accept_decision": "accepted",
        "math_profile": {"operators": ["group_rank", "ts_mean"], "windows": [21]},
        "economic_profile": {"theme": "liquidity", "thesis": "peer-relative liquidity pressure"},
        "repair_delta": {"smoothing": True, "field_shift": False},
        "outcome_score": 1.42,
        "platform_outcome": {"outcome": "target-met"},
    }

    memory.add_record(record)

    stored = memory.get_recent(limit=1)[0]
    assert stored["math_profile"]["windows"] == [21]
    assert stored["economic_profile"]["theme"] == "liquidity"
    assert stored["repair_delta"]["smoothing"] is True
    assert stored["outcome_score"] == 1.42
    assert stored["platform_outcome"]["outcome"] == "target-met"


def test_repair_memory_migrates_legacy_schema(tmp_path):
    db_path = tmp_path / "legacy_repair_memory.db"
    import sqlite3

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE repair_records (
                record_id TEXT PRIMARY KEY,
                timestamp TEXT,
                expression TEXT,
                symptom_tags TEXT,
                repair_actions TEXT,
                accept_decision TEXT,
                rejection_reason TEXT,
                recommended_directions TEXT,
                forbidden_directions TEXT,
                metrics TEXT,
                family_tag TEXT,
                notes TEXT
            )
            """
        )

    memory = RepairMemory(db_path)
    memory.add_record({
        "expression": "rank(returns)",
        "symptom_tags": ["low_sharpe"],
        "accept_decision": "rejected",
        "economic_profile": {"theme": "momentum"},
    })

    stored = memory.get_recent(limit=1)[0]
    assert stored["economic_profile"]["theme"] == "momentum"
    assert stored["math_profile"] == {}


def test_repair_memory_retrieval_prefers_matching_logic_profiles(tmp_memory):
    query_math = analyze_math_profile("group_rank(ts_mean(volume, 21), industry)")
    query_econ = infer_economic_profile("group_rank(ts_mean(volume, 21), industry)", category="LIQUIDITY")
    tmp_memory.add_record({
        "expression": "rank(ts_delta(close, 5))",
        "symptom_tags": ["high_turnover"],
        "accept_decision": "accepted",
        "recommended_directions": ["price momentum escape"],
        "math_profile": analyze_math_profile("rank(ts_delta(close, 5))"),
        "economic_profile": infer_economic_profile("rank(ts_delta(close, 5))", category="MOMENTUM"),
    })
    tmp_memory.add_record({
        "expression": "group_rank(ts_mean(volume, 63), industry)",
        "symptom_tags": ["high_turnover"],
        "accept_decision": "accepted",
        "recommended_directions": ["liquidity smoothing"],
        "math_profile": analyze_math_profile("group_rank(ts_mean(volume, 63), industry)"),
        "economic_profile": infer_economic_profile("group_rank(ts_mean(volume, 63), industry)", category="LIQUIDITY"),
    })

    result = tmp_memory.retrieve(
        ["high_turnover"],
        "group_rank(ts_mean(volume, 21), industry)",
        topk=2,
        math_profile=query_math,
        economic_profile=query_econ,
    )

    assert result["positive"][0]["expression"] == "group_rank(ts_mean(volume, 63), industry)"


class _KeywordEmbedder:
    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        lowered = str(text or "").lower()
        if "volume" in lowered or "adv20" in lowered:
            return [1.0, 0.0]
        if "cashflow" in lowered or "operating_income" in lowered:
            return [0.0, 1.0]
        return [0.5, 0.5]


def test_repair_memory_persists_embeddings_for_semantic_retrieval(tmp_path):
    db_path = tmp_path / "repair_memory.db"
    memory = RepairMemory(db_path, embedder=_KeywordEmbedder())
    memory.add_record({
        "expression": "group_rank(ts_mean(volume, 63), industry)",
        "symptom_tags": ["low_sharpe"],
        "accept_decision": "accepted",
        "recommended_directions": ["use liquidity confirmation"],
        "math_profile": analyze_math_profile("group_rank(ts_mean(volume, 63), industry)"),
        "economic_profile": infer_economic_profile("group_rank(ts_mean(volume, 63), industry)", category="LIQUIDITY"),
    })
    memory.add_record({
        "expression": "rank(ts_mean(cashflow_op / assets, 126))",
        "symptom_tags": ["low_sharpe"],
        "accept_decision": "accepted",
        "recommended_directions": ["use profitability anchor"],
        "math_profile": analyze_math_profile("rank(ts_mean(cashflow_op / assets, 126))"),
        "economic_profile": infer_economic_profile("rank(ts_mean(cashflow_op / assets, 126))", category="QUALITY"),
    })

    reloaded = RepairMemory(db_path, embedder=_KeywordEmbedder())
    result = reloaded.retrieve(
        ["low_sharpe"],
        "group_rank(ts_rank(volume / adv20, 21), industry)",
        topk=2,
        math_profile=analyze_math_profile("group_rank(ts_rank(volume / adv20, 21), industry)"),
        economic_profile=infer_economic_profile("group_rank(ts_rank(volume / adv20, 21), industry)", category="LIQUIDITY"),
    )

    assert result["positive"][0]["expression"] == "group_rank(ts_mean(volume, 63), industry)"
    assert reloaded.last_retrieval_mode == "semantic"
    assert reloaded.last_retrieval_error is None


def test_repair_memory_retrieve_records_no_embedder_mode(tmp_path):
    memory = RepairMemory(tmp_path / "repair_memory.db")
    memory.add_record({
        "expression": "group_rank(ts_mean(returns, 21), industry)",
        "symptom_tags": ["low_sharpe"],
        "accept_decision": "accepted",
        "recommended_directions": ["add industry neutralization"],
        "forbidden_directions": [],
    })

    result = memory.retrieve(
        ["low_sharpe"],
        "group_rank(returns, industry)",
        topk=1,
    )

    assert result["positive"]
    assert memory.last_retrieval_mode == "scored_no_embedder"
    assert memory.last_retrieval_error is None


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
    assert "1" in result  # always generates exactly 1 candidate


def test_generate_repair_variants_uses_compact_profiles_and_context(tmp_memory):
    tools = build_tools(tmp_memory, None, None)
    generate = next(t for t in tools if t.name == "generate_repair_variants")
    diagnosis = json.dumps({"primary_symptom": "low_sharpe", "repair_strategy": "add neutralization", "do_not_change": []})
    logic = json.dumps({
        "math_profile": {
            "operators": ["group_rank", "ts_rank", "ts_mean", "ts_delta", "ts_std_dev"],
            "fields": ["est_eps", "returns"],
            "windows": [5, 21, 42, 63, 126],
            "group_fields": ["subindustry"],
            "dominant_structure": "hybrid_group_time_series",
            "complexity": 7,
        },
        "economic_profile": {
            "theme": "revision_reversal",
            "thesis": "Blend analyst revision drift with medium-horizon reversal after volatility shocks.",
        },
    })
    result = generate.invoke({
        "expression": "group_rank(rank(returns), subindustry)",
        "diagnosis_json": diagnosis,
        "memory_context": "\n".join(f"{i}. [SUCCESS] very long memory item {i} | avoid repeating this line" for i in range(1, 8)),
        "logic_json": logic,
        "logic_patterns": "\n".join(f"{i}. [SUCCESS] very long pattern item {i} | actions=smoothing,field_shift,quality_anchor" for i in range(1, 8)),
    })

    assert "Math focus:" in result
    assert "Economic thesis:" in result
    assert "Math profile:" not in result
    assert "Economic profile:" not in result
    assert len(result) < 2200


def test_analyze_factor_logic_tool_returns_profiles(tmp_memory):
    tools = build_tools(tmp_memory, None, None)
    analyze = next(t for t in tools if t.name == "analyze_factor_logic")

    result = json.loads(analyze.invoke({
        "expression": "group_rank(ts_mean(volume, 21), industry)",
        "category": "LIQUIDITY",
    }))

    assert result["math_profile"]["windows"] == [21]
    assert result["economic_profile"]["theme"] == "liquidity"


def test_retrieve_logic_patterns_tool_returns_structured_patterns(tmp_memory):
    tmp_memory.add_record({
        "expression": "group_rank(ts_mean(volume, 63), industry)",
        "symptom_tags": ["high_turnover"],
        "accept_decision": "accepted",
        "recommended_directions": ["liquidity smoothing"],
        "math_profile": analyze_math_profile("group_rank(ts_mean(volume, 63), industry)"),
        "economic_profile": infer_economic_profile("group_rank(ts_mean(volume, 63), industry)", category="LIQUIDITY"),
        "repair_delta": {"smoothing": True, "horizon_change": True},
        "outcome_score": 1.1,
    })
    tools = build_tools(tmp_memory, None, None)
    retrieve = next(t for t in tools if t.name == "retrieve_logic_patterns")

    result = retrieve.invoke({
        "expression": "group_rank(ts_mean(volume, 21), industry)",
        "symptoms": "high_turnover",
        "category": "LIQUIDITY",
        "k": 3,
    })

    assert "[SUCCESS]" in result
    assert "smoothing" in result
    assert "score=1.1" in result


def test_retrieve_logic_patterns_surfaces_platform_outcomes(tmp_memory):
    tmp_memory.add_record({
        "expression": "group_rank(ts_mean(volume, 63), industry)",
        "symptom_tags": ["high_turnover", "degraded_no_daily_pnl"],
        "accept_decision": "rejected",
        "forbidden_directions": ["pure window tuning"],
        "math_profile": analyze_math_profile("group_rank(ts_mean(volume, 63), industry)"),
        "economic_profile": infer_economic_profile("group_rank(ts_mean(volume, 63), industry)", category="LIQUIDITY"),
        "repair_delta": {"smoothing": True, "horizon_change": True},
        "outcome_score": -0.9,
        "platform_outcome": {"status": "no_daily_pnl", "repairDepth": 1},
    })
    tools = build_tools(tmp_memory, None, None)
    retrieve = next(t for t in tools if t.name == "retrieve_logic_patterns")

    result = retrieve.invoke({
        "expression": "group_rank(ts_mean(volume, 21), industry)",
        "symptoms": "high_turnover",
        "category": "LIQUIDITY",
        "k": 3,
    })

    assert "platform=no_daily_pnl" in result


def test_retrieve_logic_patterns_compacts_to_top_success_and_failure(tmp_memory):
    for index in range(4):
        tmp_memory.add_record({
            "expression": f"group_rank(ts_mean(volume, {21 + index}), industry)",
            "symptom_tags": ["high_turnover"],
            "accept_decision": "accepted",
            "recommended_directions": [f"liquidity smoothing {index}"],
            "math_profile": analyze_math_profile(f"group_rank(ts_mean(volume, {21 + index}), industry)"),
            "economic_profile": infer_economic_profile("group_rank(ts_mean(volume, 21), industry)", category="LIQUIDITY"),
            "repair_delta": {"smoothing": True, "horizon_change": True},
            "outcome_score": 1.5 - index * 0.1,
        })
    for index in range(4):
        tmp_memory.add_record({
            "expression": f"group_rank(ts_delta(returns, {5 + index}), industry)",
            "symptom_tags": ["high_turnover"],
            "accept_decision": "rejected",
            "forbidden_directions": [f"avoid micro tuning {index}"],
            "math_profile": analyze_math_profile(f"group_rank(ts_delta(returns, {5 + index}), industry)"),
            "economic_profile": infer_economic_profile("group_rank(ts_delta(returns, 5), industry)", category="REVERSAL"),
            "repair_delta": {"smoothing": False, "field_shift": False},
            "outcome_score": -0.8 - index * 0.1,
            "platform_outcome": {"status": "no_daily_pnl"},
        })

    tools = build_tools(tmp_memory, None, None)
    retrieve = next(t for t in tools if t.name == "retrieve_logic_patterns")
    result = retrieve.invoke({
        "expression": "group_rank(ts_mean(volume, 21), industry)",
        "symptoms": "high_turnover",
        "category": "LIQUIDITY",
        "k": 5,
    })

    lines = [line for line in result.splitlines() if line.strip()]
    assert len(lines) <= 3
    assert sum("[SUCCESS]" in line for line in lines) <= 2
    assert sum("[FAILED]" in line for line in lines) <= 1


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


def test_parse_agent_output_preserves_logic_metadata():
    output = json.dumps({
        "diagnosis": {"primary_symptom": "high_turnover", "root_cause": "fast signal"},
        "candidates": [
            {
                "expression": "rank(ts_mean(volume,21))",
                "hypothesis": "smooth liquidity signal",
                "fix_applied": "smoothing",
                "math_logic": "replace fast delta with 21d mean",
                "economic_logic": "liquidity pressure changes slowly",
                "expected_failure_reduction": "lower turnover",
            },
        ],
    })

    candidates, _diagnosis = _parse_agent_output(output, "LIQUIDITY")

    assert candidates[0]["metadata"] == {
        "math_logic": "replace fast delta with 21d mean",
        "economic_logic": "liquidity pressure changes slowly",
        "expected_failure_reduction": "lower turnover",
    }


def test_parse_agent_output_array_fallback():
    output = 'Here are the candidates:\n' + json.dumps([
        {"expression": "rank(ts_mean(returns,21))", "hypothesis": "smoothed", "fix_applied": "ts_mean"},
    ])
    candidates, diagnosis = _parse_agent_output(output, "QUALITY")
    assert len(candidates) == 1
    assert candidates[0]["category"] == "QUALITY"


def test_parse_agent_output_accepts_langchain_content_blocks():
    output = [
        {
            "type": "text",
            "text": json.dumps({
                "diagnosis": {"primary_symptom": "low_fitness"},
                "candidates": [
                    {
                        "expression": "group_rank(ts_mean(operating_income / assets, 63), industry)",
                        "hypothesis": "smooth quality anchor",
                        "fix_applied": "smoothed operating quality",
                    }
                ],
            }),
        }
    ]

    candidates, diagnosis = _parse_agent_output(output, "QUALITY")

    assert len(candidates) == 1
    assert candidates[0]["expression"] == "group_rank(ts_mean(operating_income / assets, 63), industry)"
    assert candidates[0]["origin_refs"] == ["langchain_agent", "smoothed operating quality"]
    assert diagnosis["primary_symptom"] == "low_fitness"


def test_parse_agent_output_no_json():
    candidates, diagnosis = _parse_agent_output("Sorry, I could not repair this.", "MOMENTUM")
    assert candidates == []
    assert diagnosis == {}


# ---------------------------------------------------------------------------
# record_outcome — persists to memory and resets agent
# ---------------------------------------------------------------------------

def test_record_outcome_resets_embeddings(chain):
    chain._embeddings = MagicMock()  # simulate initialized embeddings
    chain.record_outcome(
        expression="rank(returns)",
        diagnosis={"primary_symptom": "low_sharpe"},
        candidates=[{"origin_refs": ["langchain_agent", "added ts_mean"]}],
        accepted=True,
    )
    assert chain._embeddings is None  # must be reset so FAISS rebuilds
    records = chain.memory.get_recent(limit=10)
    assert records[0]["accept_decision"] == "accepted"


def test_record_outcome_writes_structured_logic_fields(chain):
    chain.record_outcome(
        expression="rank(ts_delta(close, 5))",
        diagnosis={"primary_symptom": "high_turnover"},
        candidates=[
            {
                "expression": "group_rank(ts_mean(volume, 21), industry)",
                "origin_refs": ["langchain_agent", "rule_turnover_smooth"],
            }
        ],
        accepted=True,
        candidate_metrics={"sharpe": 1.4, "fitness": 0.8, "turnover": 0.32},
        platform_outcome={"outcome": "target-met"},
    )

    record = chain.memory.get_recent(limit=1)[0]
    assert record["math_profile"]["windows"] == [21]
    assert record["economic_profile"]["theme"] == "liquidity"
    assert record["repair_delta"]["smoothing"] is True
    assert record["repair_delta"]["field_shift"] is True
    assert record["outcome_score"] > 0
    assert record["platform_outcome"]["outcome"] == "target-met"


def test_record_outcome_persists_platform_symptom_tags(chain):
    chain.record_outcome(
        expression="rank(ts_delta(close, 5))",
        diagnosis={
            "primary_symptom": "low_fitness",
            "secondary_symptoms": ["weak_signal"],
        },
        candidates=[
            {
                "expression": "group_rank(ts_mean(volume, 21), industry)",
                "origin_refs": ["langchain_agent", "rule_turnover_smooth"],
            }
        ],
        accepted=False,
        gate={"checks": [{"name": "FITNESS", "result": "FAIL"}]},
        platform_outcome={"status": "no_daily_pnl", "repairDepth": 1, "degradedQualified": False},
        category="LIQUIDITY",
    )

    record = chain.memory.get_recent(limit=1)[0]
    assert "low_fitness" in record["symptom_tags"]
    assert "weak_signal" in record["symptom_tags"]
    assert "fitness" in record["symptom_tags"]
    assert "no_daily_pnl" in record["symptom_tags"]
    assert "degraded_no_daily_pnl" in record["symptom_tags"]


# ---------------------------------------------------------------------------
# run() — mock _ensure_llm + _run_tool_loop
# ---------------------------------------------------------------------------

def test_run_returns_candidates_from_agent(chain):
    agent_output = json.dumps({
        "diagnosis": {"primary_symptom": "low_sharpe", "root_cause": "noisy"},
        "candidates": [
            {"expression": "group_rank(ts_mean(returns,21),industry)", "hypothesis": "smoothed", "fix_applied": "ts_mean"},
            {"expression": "group_rank(ts_rank(returns,63),industry)", "hypothesis": "longer", "fix_applied": "longer window"},
        ],
    })
    with patch.object(chain, "_ensure_llm", return_value=(MagicMock(), [], {})), \
         patch.object(chain, "_run_tool_loop", return_value=agent_output):
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
    with patch.object(chain, "_ensure_llm", side_effect=RuntimeError("LLM error")):
        candidates, diagnosis = chain.run(
            expression="rank(returns)",
            metrics={"sharpe": 0.2},
            failed_checks=["SHARPE"],
            gate_reasons=[],
        )
    assert candidates == []
    assert diagnosis == {}


def test_run_uses_full_tool_budget(chain):
    with patch.object(chain, "_ensure_llm", return_value=(MagicMock(), [], {})), \
         patch.object(chain, "_run_tool_loop", return_value="[]") as run_loop:
        chain.run(
            expression="rank(returns)",
            metrics={"sharpe": 0.2, "fitness": 0.1, "turnover": 0.4},
            failed_checks=["FITNESS"],
            gate_reasons=["fitness below threshold"],
            category="QUALITY",
        )

    assert run_loop.call_args.kwargs["max_iterations"] >= 8


def test_run_includes_recursive_repair_constraints(chain):
    with patch.object(chain, "_ensure_llm", return_value=(MagicMock(), [], {})), \
         patch.object(chain, "_run_tool_loop", return_value="[]") as run_loop:
        chain.run(
            expression="rank(returns)",
            metrics={"sharpe": 0.2, "fitness": 0.1, "turnover": 0.4},
            failed_checks=["FITNESS", "SHARPE"],
            gate_reasons=["testSharpe must be positive", "no daily pnl available"],
            category="QUALITY",
            repair_context={
                "repairDepth": 2,
                "nextAction": "crowding repair via material concept/horizon/group change",
            },
        )

    user_input = run_loop.call_args.args[2]
    assert "Repair depth: 2" in user_input
    assert "Recursive repair" in user_input
    assert "avoid pure window tuning" in user_input
    assert "Next action hint: crowding repair via material concept/horizon/group change" in user_input


def test_run_tool_loop_reprompts_for_json_only(chain):
    prose_response = AIMessage(
        content="Let me also validate an alternative that's even more aggressive in restructuring:"
    )
    json_response = AIMessage(
        content=json.dumps(
            {
                "diagnosis": {"primary_symptom": "low_fitness"},
                "candidates": [
                    {
                        "expression": "group_rank(ts_mean(adv20, 21) + ts_rank(cashflow_op / assets, 252), industry)",
                        "hypothesis": "blend smoothed liquidity with quality anchor",
                        "fix_applied": "smoothed liquidity plus quality blend",
                    }
                ],
            }
        )
    )

    llm = MagicMock()
    llm.invoke.side_effect = [prose_response, json_response]

    output, usage = chain._run_tool_loop(llm, {}, "repair this expression", max_iterations=4)

    candidates, diagnosis = _parse_agent_output(output, "LIQUIDITY")
    assert len(candidates) == 1
    assert diagnosis["primary_symptom"] == "low_fitness"
    assert usage["calls"] == 2
