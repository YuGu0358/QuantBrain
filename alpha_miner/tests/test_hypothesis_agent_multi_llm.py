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
    ):
        agent = HypothesisAgent(
            kb=kb,
            cache=cache,
            taxonomy={"QUALITY": {}},
            router=mock_router,
            use_llm=True,
        )
        result = agent.generate_batch("test objective", category="QUALITY", n=1)

    assert len(result) == 1
    mock_router.pick.assert_called()


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
