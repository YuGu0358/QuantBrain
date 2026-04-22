import os
import json
from dataclasses import asdict
from unittest.mock import MagicMock
from unittest.mock import patch

from alpha_miner.modules.llm_cache import LLMCache
from alpha_miner.modules.llm_router import LLMProvider
from alpha_miner.modules.m1_knowledge_base import KnowledgeBase
from alpha_miner.modules.m2_hypothesis_agent import Candidate
from alpha_miner.modules.m2_hypothesis_agent import HypothesisAgent


def _kb_and_cache(tmp_path):
    return KnowledgeBase(tmp_path / "kb.db"), LLMCache(tmp_path / "cache")


def _provider(role: str) -> LLMProvider:
    return LLMProvider(
        name="mock",
        role=role,
        model_id="mock",
        client_type="openai_compat",
        api_key_env="OPENAI_API_KEY",
        api_base=None,
    )


def test_generate_batch_calls_router_when_use_llm_true(tmp_path):
    kb, cache = _kb_and_cache(tmp_path)
    mock_router = MagicMock()
    mock_router.pick.return_value = _provider("generate")

    with patch.object(
        HypothesisAgent,
        "_call_llm",
        return_value={
            "candidates": [
                {
                    "id": "c1",
                    "category": "QUALITY",
                    "hypothesis": "h",
                    "expression": "rank(close)",
                    "origin_refs": [],
                }
            ]
        },
    ) as llm_call:
        agent = HypothesisAgent(
            kb=kb,
            cache=cache,
            taxonomy={"QUALITY": {}},
            router=mock_router,
            use_llm=True,
        )
        result = agent.generate_batch("test objective", category="QUALITY", n=1)

    assert len(result) == 1
    assert llm_call.call_count >= 1


def test_prompt_includes_rag_failure_patterns(tmp_path):
    kb, cache = _kb_and_cache(tmp_path)
    kb.record_failure_pattern(
        reason="LOW_SHARPE",
        expression="rank(close)",
        suggested_fix="add ts_rank",
    )
    agent = HypothesisAgent(kb=kb, cache=cache, taxonomy={"QUALITY": {}})

    payload = agent._request_payload("obj", "QUALITY", 1)

    assert "LOW_SHARPE" in str(payload)


def test_judge_selects_best_candidates(tmp_path):
    kb, cache = _kb_and_cache(tmp_path)
    agent = HypothesisAgent(kb=kb, cache=cache, taxonomy={"QUALITY": {}})
    candidates = [
        Candidate(
            id="c1",
            category="QUALITY",
            hypothesis="h1",
            expression="rank(close)",
            origin_refs=[],
        ),
        Candidate(
            id="c2",
            category="QUALITY",
            hypothesis="h2",
            expression="ts_rank(close,5)",
            origin_refs=[],
        ),
        Candidate(
            id="c3",
            category="QUALITY",
            hypothesis="h3",
            expression="rank(volume)",
            origin_refs=[],
        ),
        Candidate(
            id="c4",
            category="QUALITY",
            hypothesis="h4",
            expression="rank(returns)",
            origin_refs=[],
        ),
    ]
    mock_router = MagicMock()
    mock_router.pick.return_value = _provider("judge")

    with patch.object(HypothesisAgent, "_call_llm", return_value={"selected_indices": [0, 2]}):
        result = agent._judge_candidates(candidates, n=2, router=mock_router)

    assert len(result) == 2
    assert result[0].expression == "rank(close)"


def test_generate_batch_runs_judge_for_small_pool_when_multiple_candidates_available(tmp_path):
    kb, cache = _kb_and_cache(tmp_path)
    mock_router = MagicMock()
    mock_router.pick.side_effect = [_provider("generate"), _provider("judge")]
    agent = HypothesisAgent(
        kb=kb,
        cache=cache,
        taxonomy={"QUALITY": {}},
        router=mock_router,
        use_llm=True,
    )
    with patch.object(
        HypothesisAgent,
        "_call_llm",
        side_effect=[
            {
                "candidates": [
                    {
                        "id": "c1",
                        "category": "QUALITY",
                        "hypothesis": "h1",
                        "expression": "group_rank(ts_rank(operating_income / assets, 252) + ts_rank(volume / adv20, 63), industry)",
                        "origin_refs": [],
                    },
                    {
                        "id": "c2",
                        "category": "QUALITY",
                        "hypothesis": "h2",
                        "expression": "rank(volume)",
                        "origin_refs": [],
                    },
                ]
            },
            {"selected_indices": [0, 1]},
        ],
    ), patch.object(agent, "_judge_candidates", wraps=agent._judge_candidates) as judge_spy:
        result = agent.generate_batch("test objective", category="QUALITY", n=2)

    assert len(result) == 2
    assert judge_spy.call_count == 1


def test_generate_batch_forces_judge_when_optimized_disabled(tmp_path):
    kb, cache = _kb_and_cache(tmp_path)
    mock_router = MagicMock()
    mock_router.pick.return_value = _provider("generate")
    with patch.dict(os.environ, {"LLM_OPTIMIZED_GENERATION_ENABLED": "false"}), patch.object(
        HypothesisAgent,
        "_call_llm",
        side_effect=[
            {
                "candidates": [
                    {"id": "c1", "category": "QUALITY", "hypothesis": "h1", "expression": "rank(close)", "origin_refs": []},
                    {"id": "c2", "category": "QUALITY", "hypothesis": "h2", "expression": "rank(volume)", "origin_refs": []},
                ]
            },
            {
                "candidates": [
                    {"id": "c3", "category": "QUALITY", "hypothesis": "h3", "expression": "rank(returns)", "origin_refs": []},
                ]
            },
        ],
    ):
        agent = HypothesisAgent(
            kb=kb,
            cache=cache,
            taxonomy={"QUALITY": {}},
            router=mock_router,
            use_llm=True,
        )
        agent._judge_candidates = MagicMock(return_value=[
            Candidate(id="c1", category="QUALITY", hypothesis="h1", expression="rank(close)", origin_refs=[]),
            Candidate(id="c2", category="QUALITY", hypothesis="h2", expression="rank(volume)", origin_refs=[]),
        ])
        result = agent.generate_batch("test objective", category="QUALITY", n=2)

    assert len(result) == 2
    assert agent._judge_candidates.call_count == 1


def test_generate_batch_filters_low_quality_single_family_candidates(tmp_path):
    kb, cache = _kb_and_cache(tmp_path)
    mock_router = MagicMock()
    mock_router.pick.side_effect = [_provider("generate"), _provider("judge")]

    with patch.object(
        HypothesisAgent,
        "_call_llm",
        side_effect=[
            {
                "candidates": [
                    {
                        "id": "weak_1",
                        "category": "QUALITY",
                        "hypothesis": "single price level",
                        "expression": "rank(close)",
                        "origin_refs": [],
                    },
                    {
                        "id": "weak_2",
                        "category": "QUALITY",
                        "hypothesis": "single liquidity level",
                        "expression": "rank(volume)",
                        "origin_refs": [],
                    },
                    {
                        "id": "strong_1",
                        "category": "QUALITY",
                        "hypothesis": "profitability with liquidity confirmation",
                        "expression": "group_rank(ts_rank(operating_income / assets, 252) + ts_rank(volume / adv20, 63), industry)",
                        "origin_refs": [],
                    },
                ]
            },
            {"selected_indices": [0, 1, 2]},
        ],
    ):
        agent = HypothesisAgent(
            kb=kb,
            cache=cache,
            taxonomy={"QUALITY": {}},
            router=mock_router,
            use_llm=True,
        )
        result = agent.generate_batch("test objective", category="QUALITY", n=2)

    expressions = [candidate.expression for candidate in result]
    assert "group_rank(ts_rank(operating_income / assets, 252) + ts_rank(volume / adv20, 63), industry)" in expressions
    assert "rank(close)" not in expressions


def test_generate_batch_repair_chain_accepts_metadata_payload(tmp_path):
    kb, cache = _kb_and_cache(tmp_path)
    agent = HypothesisAgent(
        kb=kb,
        cache=cache,
        taxonomy={"QUALITY": {}},
    )
    agent.repair_chain = MagicMock()
    agent.repair_chain.run.return_value = (
        [
            {
                "id": "repair_quality_000",
                "category": "QUALITY",
                "hypothesis": "repair hypothesis",
                "expression": "rank(close)",
                "origin_refs": ["langchain_agent", "fix_applied"],
                "metadata": {
                    "math_logic": "extended lookback",
                    "economic_logic": "preserve quality anchor",
                },
                "unexpected_field": "ignored",
            }
        ],
        {},
    )

    with patch.object(HypothesisAgent, "_rule_based_repair_candidates", return_value=[]):
        result = agent.generate_batch(
            "repair objective",
            category="QUALITY",
            n=1,
            repair_context={
                "expression": "rank(returns)",
                "failedChecks": ["SHARPE"],
                "gate": {"reasons": ["sharpe too low"]},
                "metrics": {"isSharpe": 0.2},
            },
        )

    assert len(result) == 1
    assert result[0].expression == "rank(close)"
    assert result[0].metadata["math_logic"] == "extended lookback"


def test_generate_batch_passes_validator_into_repair_chain(tmp_path):
    kb, cache = _kb_and_cache(tmp_path)
    validator = MagicMock()
    agent = HypothesisAgent(
        kb=kb,
        cache=cache,
        taxonomy={"QUALITY": {}},
        validator=validator,
    )
    agent.repair_chain = MagicMock()
    agent.repair_chain.run.return_value = (
        [
            {
                "id": "repair_quality_000",
                "category": "QUALITY",
                "hypothesis": "repair hypothesis",
                "expression": "rank(close)",
                "origin_refs": ["langchain_agent"],
            }
        ],
        {},
    )

    with patch.object(HypothesisAgent, "_rule_based_repair_candidates", return_value=[]):
        result = agent.generate_batch(
            "repair objective",
            category="QUALITY",
            n=1,
            repair_context={
                "expression": "rank(returns)",
                "failedChecks": ["SHARPE"],
                "gate": {"reasons": ["sharpe too low"]},
                "metrics": {"isSharpe": 0.2},
            },
        )

    assert len(result) == 1
    _, kwargs = agent.repair_chain.run.call_args
    assert kwargs["validator"] is validator


def test_generate_batch_repair_chain_receives_rule_seed_candidates(tmp_path):
    kb, cache = _kb_and_cache(tmp_path)
    validator = MagicMock()
    agent = HypothesisAgent(
        kb=kb,
        cache=cache,
        taxonomy={"QUALITY": {}},
        validator=validator,
    )
    agent.repair_chain = MagicMock()
    agent.repair_chain.run.return_value = (
        [
            {
                "id": "repair_quality_001",
                "category": "QUALITY",
                "hypothesis": "repair hypothesis",
                "expression": "group_rank(ts_rank(cashflow_op / assets, 252) + ts_rank(volume / adv20, 63), industry)",
                "origin_refs": ["langchain_agent", "llm_mutation"],
                "metadata": {
                    "math_logic": "introduce second signal family",
                    "economic_logic": "quality plus liquidity confirmation",
                },
            }
        ],
        {},
    )

    seed_candidates = [
        Candidate(
            id="repair_rule_000",
            category="QUALITY",
            hypothesis="smooth the parent expression",
            expression="rank(ts_mean(rank(returns), 20))",
            origin_refs=["rule_repair", "rule_turnover_smooth"],
        )
    ]

    with patch.object(HypothesisAgent, "_rule_based_repair_candidates", return_value=seed_candidates):
        result = agent.generate_batch(
            "repair objective",
            category="QUALITY",
            n=1,
            repair_context={
                "expression": "rank(returns)",
                "failedChecks": ["TURNOVER"],
                "gate": {"reasons": ["turnover too high"]},
                "metrics": {"turnover": 0.9},
            },
        )

    assert len(result) == 1
    _, kwargs = agent.repair_chain.run.call_args
    assert kwargs["seed_candidates"] == [asdict(seed_candidates[0])]


def test_generate_batch_blocks_complex_repair_without_semantic_memory(tmp_path):
    kb, cache = _kb_and_cache(tmp_path)
    agent = HypothesisAgent(
        kb=kb,
        cache=cache,
        taxonomy={"MOMENTUM": {}},
    )

    with patch.object(
        HypothesisAgent,
        "_rule_based_repair_candidates",
        return_value=[
            Candidate(
                id="repair_rule_000",
                category="MOMENTUM",
                hypothesis="cross family escape",
                expression="rank(ts_delta(volume, 20)) * rank(-returns)",
                origin_refs=["rule_repair", "rule_cross_family_escape"],
            )
        ],
    ):
        result = agent.generate_batch(
            "repair objective",
            category="MOMENTUM",
            n=1,
            repair_context={
                "expression": "magic_alpha(close)",
                "failedChecks": ["SELF_CORRELATION"],
                "gate": {"reasons": ["daily pnl degraded"]},
            },
        )

    assert result == []
    assert agent.last_repair_quality["route"] == "blocked"
    assert "semantic_memory_required" in agent.last_repair_quality["reasons"]


def test_generation_payload_includes_cross_category_examples(tmp_path):
    kb, cache = _kb_and_cache(tmp_path)
    agent = HypothesisAgent(kb=kb, cache=cache, taxonomy={"QUALITY": {}, "MOMENTUM": {}, "REVERSAL": {}})

    payload = agent._request_payload("discover robust diversified alphas", "QUALITY", 3)
    user_dict = json.loads(payload["messages"][1]["content"])

    example_categories = {item["category"] for item in user_dict["positive_multi_variable_examples"]}
    assert "QUALITY" in example_categories
    assert len(example_categories) >= 2


def test_deterministic_candidates_interleave_categories_before_repeating(tmp_path):
    kb, cache = _kb_and_cache(tmp_path)
    agent = HypothesisAgent(
        kb=kb,
        cache=cache,
        taxonomy={"QUALITY": {}, "MOMENTUM": {}, "REVERSAL": {}, "LIQUIDITY": {}},
    )

    candidates = agent._deterministic_candidates("discover robust diversified alphas", "QUALITY", 4)

    assert [candidate.category for candidate in candidates] == [
        "QUALITY",
        "MOMENTUM",
        "REVERSAL",
        "LIQUIDITY",
    ]
