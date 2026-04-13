import sys
from types import SimpleNamespace
from unittest.mock import MagicMock
from unittest.mock import patch

from alpha_miner.modules.idea_optimizer import IdeaOptimizer
from alpha_miner.modules.llm_router import LLMProvider
from alpha_miner.modules.m1_knowledge_base import KnowledgeBase


def _anthropic_provider() -> LLMProvider:
    return LLMProvider(
        name="mock-anthropic",
        role="idea",
        model_id="claude-mock",
        client_type="anthropic",
        api_key_env="ANTHROPIC_API_KEY",
    )


def _anthropic_response(text: str):
    return SimpleNamespace(
        content=[SimpleNamespace(text=text)],
        usage=SimpleNamespace(input_tokens=10, output_tokens=20),
    )


def _patch_anthropic(anthropic_cls):
    return patch.dict(
        sys.modules,
        {"anthropic": SimpleNamespace(Anthropic=anthropic_cls)},
    )


def test_optimize_returns_structured_direction():
    mock_router = MagicMock()
    mock_router.pick.return_value = _anthropic_provider()
    response_text = (
        '{"objective":"supply chain alpha","category":"QUALITY",'
        '"hypothesis":"stable suppliers reduce volatility",'
        '"constraints":[],"suggested_data_fields":["accounts_payable"]}'
    )

    anthropic_cls = MagicMock()
    with _patch_anthropic(anthropic_cls):
        anthropic_cls.return_value.messages.create.return_value = _anthropic_response(response_text)
        optimizer = IdeaOptimizer(router=mock_router)
        result = optimizer.optimize("供应链因子")

    assert result["category"] == "QUALITY"
    assert "hypothesis" in result


def test_optimize_fallback_on_llm_error():
    mock_router = MagicMock()
    mock_router.pick.return_value = _anthropic_provider()

    anthropic_cls = MagicMock(side_effect=Exception("LLM unavailable"))
    with _patch_anthropic(anthropic_cls):
        optimizer = IdeaOptimizer(router=mock_router)
        result = optimizer.optimize("any idea")

    assert result["objective"] == "any idea"
    assert "category" in result


def test_optimize_with_kb_context(tmp_path):
    kb = KnowledgeBase(tmp_path / "kb.db")
    kb.record_failure_pattern(
        reason="LOW_SHARPE",
        expression="rank(close)",
        suggested_fix="add peer-relative normalisation",
    )
    mock_router = MagicMock()
    mock_router.pick.return_value = _anthropic_provider()
    response_text = (
        '{"objective":"supply chain alpha","category":"QUALITY",'
        '"hypothesis":"stable suppliers reduce volatility",'
        '"constraints":[],"suggested_data_fields":["accounts_payable"]}'
    )

    anthropic_cls = MagicMock()
    with _patch_anthropic(anthropic_cls):
        anthropic_cls.return_value.messages.create.return_value = _anthropic_response(response_text)
        optimizer = IdeaOptimizer(router=mock_router, kb=kb)
        optimizer.optimize("供应链因子")

    call_args = anthropic_cls.return_value.messages.create.call_args
    assert "LOW_SHARPE" in str(call_args)
