from pathlib import Path

from alpha_miner.modules.m6_alpha_pool import AlphaPool, expression_jaccard, save_pnl_series


def test_orthogonality_rejects_highly_correlated_pnl(tmp_path: Path):
    existing = tmp_path / "existing.json"
    save_pnl_series(existing, [0.01, 0.02, -0.01, 0.03, -0.02])
    pool = AlphaPool(tmp_path / "pool.db", threshold=0.5)
    pool.add_alpha("a1", "rank(close)", "MOMENTUM", existing, holdout_used=True)
    result = pool.check_orthogonality([0.011, 0.021, -0.009, 0.029, -0.021])
    assert not result.passed
    assert result.nearest_alpha_id == "a1"


def test_holdout_used_blocks_retest(tmp_path: Path):
    existing = tmp_path / "existing.json"
    save_pnl_series(existing, [0.01, 0.02, -0.01])
    pool = AlphaPool(tmp_path / "pool.db")
    pool.add_alpha("a1", "rank(close)", "MOMENTUM", existing, holdout_used=True)
    try:
        pool.assert_holdout_available("a1")
    except ValueError as error:
        assert "already used holdout" in str(error)
    else:
        raise AssertionError("Expected holdout guard to raise")


def test_expression_similarity_proxy_rejects_near_duplicate(tmp_path: Path):
    existing = tmp_path / "existing.json"
    save_pnl_series(existing, [0.01, 0.02, -0.01])
    pool = AlphaPool(tmp_path / "pool.db")
    pool.add_alpha("a1", "rank(close)", "MOMENTUM", existing, holdout_used=False)
    result = pool.check_expression_similarity("ts_mean(rank(close), 5)", threshold=0.5)
    assert not result.passed
    assert result.nearest_alpha_id == "a1"
    assert result.source == "expression_jaccard_proxy"


def test_expression_jaccard_separates_distinct_families():
    assert expression_jaccard("rank(close)", "rank(volume)") < 1.0
