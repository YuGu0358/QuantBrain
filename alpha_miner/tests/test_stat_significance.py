from alpha_miner.modules.m7_stat_significance import StatSignificance, purged_kfold_indices


def test_deflated_sharpe_is_probability():
    stat = StatSignificance()
    value = stat.deflated_sharpe(1.4, [0.1, 0.2, 0.4, 1.0, 1.4], 252)
    assert 0 <= value <= 1
    assert value > 0.5


def test_pbo_bounds():
    stat = StatSignificance()
    value = stat.pbo([1.5, 0.7, 0.2, -0.1], [1.3, 0.6, 0.1, -0.2])
    assert 0 <= value <= 1


def test_purged_kfold_has_no_leakage():
    folds = purged_kfold_indices(100, n_splits=5, purge=3, embargo=2)
    assert len(folds) == 5
    for fold in folds:
        assert set(fold.train_indices).isdisjoint(fold.test_indices)
        assert fold.test_indices
