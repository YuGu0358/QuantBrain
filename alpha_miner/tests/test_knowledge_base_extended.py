from alpha_miner.modules.m1_knowledge_base import KnowledgeBase


def test_record_strategy_stat_and_query(tmp_path):
    kb = KnowledgeBase(tmp_path / "kb.db")

    kb.record_strategy_stat(
        category="QUALITY",
        gate_result="PASS",
        operator_skeleton="rank(ts_rank(X,N))",
    )
    stats = kb.get_strategy_stats("QUALITY")

    assert stats["attempts"] == 1 and stats["wins"] == 1


def test_record_failure_pattern(tmp_path):
    kb = KnowledgeBase(tmp_path / "kb.db")

    kb.record_failure_pattern(
        reason="LOW_SHARPE",
        expression="rank(close)",
        suggested_fix="add peer-relative normalisation",
    )
    patterns = kb.get_failure_patterns(limit=5)

    assert len(patterns) == 1
    assert patterns[0]["reason"] == "LOW_SHARPE"


def test_record_operator_stat(tmp_path):
    kb = KnowledgeBase(tmp_path / "kb.db")

    kb.record_operator_stat(operator="ts_rank", category="QUALITY", passed=True)
    kb.record_operator_stat(operator="ts_rank", category="QUALITY", passed=False)
    stats = kb.get_operator_stats("QUALITY")

    assert stats["ts_rank"]["attempts"] == 2 and stats["ts_rank"]["wins"] == 1


def test_upsert_market_regime(tmp_path):
    kb = KnowledgeBase(tmp_path / "kb.db")

    kb.upsert_market_regime(
        regime_key="current",
        summary="momentum leading, quality lagging",
        top_categories=["MOMENTUM", "REVERSAL"],
    )
    regime = kb.get_market_regime("current")

    assert regime["summary"] == "momentum leading, quality lagging"
    assert "MOMENTUM" in regime["top_categories"]


def test_enriched_rag_context_includes_patterns(tmp_path):
    kb = KnowledgeBase(tmp_path / "kb.db")

    kb.record_failure_pattern(
        reason="LOW_SHARPE",
        expression="rank(close)",
        suggested_fix="add peer-relative normalisation",
    )
    kb.record_strategy_stat(
        category="QUALITY",
        gate_result="PASS",
        operator_skeleton="rank(ts_rank(X,N))",
    )
    ctx = kb.rag_context("QUALITY")

    assert hasattr(ctx, "failure_patterns")
    assert hasattr(ctx, "strategy_stats")
