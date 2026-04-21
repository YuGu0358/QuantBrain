import os
import json
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


def test_generate_batch_skips_judge_for_small_pool_when_optimized(tmp_path):
    kb, cache = _kb_and_cache(tmp_path)
    mock_router = MagicMock()
    mock_router.pick.return_value = _provider("generate")
    with patch.object(
        HypothesisAgent,
        "_call_llm",
        return_value={
            "candidates": [
                {"id": "c1", "category": "QUALITY", "hypothesis": "h1", "expression": "rank(close)", "origin_refs": []},
                {"id": "c2", "category": "QUALITY", "hypothesis": "h2", "expression": "rank(volume)", "origin_refs": []},
            ]
        },
    ), patch.object(HypothesisAgent, "_judge_candidates", wraps=HypothesisAgent._judge_candidates) as judge_spy:
        agent = HypothesisAgent(
            kb=kb,
            cache=cache,
            taxonomy={"QUALITY": {}},
            router=mock_router,
            use_llm=True,
        )
        result = agent.generate_batch("test objective", category="QUALITY", n=2)

    assert len(result) == 2
    assert judge_spy.call_count == 0


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
