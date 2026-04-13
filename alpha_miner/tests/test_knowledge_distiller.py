from unittest.mock import MagicMock
from unittest.mock import patch

from alpha_miner.modules.m1_knowledge_base import KnowledgeBase
from alpha_miner.modules.m9_knowledge_distiller import KnowledgeDistiller


def test_distill_records_failure_patterns_from_batch(tmp_path):
    kb = KnowledgeBase(tmp_path / "kb.db")
    mock_router = MagicMock()
    batch = {
        "records": [
            {
                "candidate": {"expression": "rank(close)", "category": "QUALITY"},
                "backtest": {"status": "completed", "sharpe": 0.3},
                "scorecard": {"gate": "FAIL", "reason": "LOW_SHARPE"},
            }
        ]
    }

    with patch(
        "alpha_miner.modules.m9_knowledge_distiller._call_distill_llm",
        return_value={
            "failure_patterns": [
                {
                    "reason": "LOW_SHARPE",
                    "expression": "rank(close)",
                    "suggested_fix": "use ts_rank",
                }
            ],
            "operator_stats": [],
            "market_regime": None,
        },
    ):
        distiller = KnowledgeDistiller(kb=kb, router=mock_router)
        distiller.distill(batch)

    patterns = kb.get_failure_patterns()
    assert len(patterns) == 1 and patterns[0]["reason"] == "LOW_SHARPE"


def test_distill_records_strategy_stat_for_pass(tmp_path):
    kb = KnowledgeBase(tmp_path / "kb.db")
    mock_router = MagicMock()
    batch = {
        "records": [
            {
                "candidate": {"expression": "ts_rank(close, 5)", "category": "MOMENTUM"},
                "backtest": {"status": "completed", "sharpe": 1.2},
                "scorecard": {"gate": "PASS"},
            }
        ]
    }

    distiller = KnowledgeDistiller(kb=kb, router=mock_router)
    distiller.distill(batch)

    stats = kb.get_strategy_stats("MOMENTUM")
    assert stats["wins"] == 1


def test_distill_skips_when_router_none(tmp_path):
    kb = KnowledgeBase(tmp_path / "kb.db")

    distiller = KnowledgeDistiller(kb=kb, router=None)
    distiller.distill({"records": []})
