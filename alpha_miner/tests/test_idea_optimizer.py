import sys
from types import SimpleNamespace
from unittest.mock import MagicMock
from unittest.mock import patch

from alpha_miner.modules.idea_optimizer import IdeaOptimizer
from alpha_miner.modules.llm_router import LLMProvider
from alpha_miner.modules.m1_knowledge_base import KnowledgeBase


def _openai_provider() -> LLMProvider:
    return LLMProvider(
        name="mock-openai",
        role="idea",
        model_id="gpt-5.4-mini",
        client_type="openai_compat",
        api_key_env="OPENAI_API_KEY",
    )


def _openai_response(text: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20),
    )


def _patch_openai(openai_cls):
    return patch.dict(
        sys.modules,
        {"openai": SimpleNamespace(OpenAI=openai_cls)},
    )


def test_optimize_returns_structured_direction():
    mock_router = MagicMock()
    mock_router.pick.return_value = _openai_provider()
    response_text = (
        '{"objective":"supply chain alpha","category":"QUALITY",'
        '"hypothesis":"stable suppliers reduce volatility",'
        '"constraints":[],"suggested_data_fields":["accounts_payable"]}'
    )

    openai_cls = MagicMock()
    with _patch_openai(openai_cls):
        openai_cls.return_value.chat.completions.create.return_value = _openai_response(response_text)
        optimizer = IdeaOptimizer(router=mock_router)
        result = optimizer.optimize("供应链因子")

    assert result["category"] == "QUALITY"
    assert "hypothesis" in result


def test_optimize_fallback_on_llm_error():
    mock_router = MagicMock()
    mock_router.pick.return_value = _openai_provider()

    openai_cls = MagicMock(side_effect=Exception("LLM unavailable"))
    with _patch_openai(openai_cls):
        optimizer = IdeaOptimizer(router=mock_router)
        result = optimizer.optimize("any idea")

    assert result["objective"] == "any idea"
    assert "category" in result
    mock_router.record_result.assert_called_once()
    call = mock_router.record_result.call_args[0]
    assert call[1] == "idea"
    assert call[2] is False


def test_optimize_with_kb_context(tmp_path):
    kb = KnowledgeBase(tmp_path / "kb.db")
    kb.record_failure_pattern(
        reason="LOW_SHARPE",
        expression="rank(close)",
        suggested_fix="add peer-relative normalisation",
    )
    mock_router = MagicMock()
    mock_router.pick.return_value = _openai_provider()
    response_text = (
        '{"objective":"supply chain alpha","category":"QUALITY",'
        '"hypothesis":"stable suppliers reduce volatility",'
        '"constraints":[],"suggested_data_fields":["accounts_payable"]}'
    )

    openai_cls = MagicMock()
    with _patch_openai(openai_cls):
        openai_cls.return_value.chat.completions.create.return_value = _openai_response(response_text)
        optimizer = IdeaOptimizer(router=mock_router, kb=kb)
        optimizer.optimize("供应链因子")

    call_args = openai_cls.return_value.chat.completions.create.call_args
    assert "LOW_SHARPE" in str(call_args)
