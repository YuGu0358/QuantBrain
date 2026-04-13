from pathlib import Path

from alpha_miner.modules.m4_brain_backtester import BrainBacktester, extract_max_correlation


def test_brain_backtester_mock_writes_snapshot_and_pnl(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("WQB_EMAIL", raising=False)
    monkeypatch.delenv("WQB_PASSWORD", raising=False)
    backtester = BrainBacktester(tmp_path, {"brain": {"rate_limit_per_minute": 8, "rate_limit_per_day": 500}})
    result = backtester.submit_alpha("rank(close)", "IS")
    assert result.status == "completed"
    assert result.pnl_path
    assert Path(result.pnl_path).exists()
    assert Path(result.raw_path).exists()
    assert result.has_daily_pnl


def test_brain_check_parser_extracts_self_correlation():
    payload = {
        "checks": [
            {"name": "LOW_SHARPE", "value": 0.3},
            {"name": "SELF_CORRELATION", "value": 0.72, "result": "FAIL"},
        ]
    }
    assert extract_max_correlation(payload) == 0.72


def test_brain_check_mock_returns_passed(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("WQB_EMAIL", raising=False)
    monkeypatch.delenv("WQB_PASSWORD", raising=False)
    backtester = BrainBacktester(tmp_path, {"pool": {"brain_check_correlation_threshold": 0.7}})
    result = backtester.check_alpha("mock-alpha")
    assert result.passed is True
    assert Path(result.raw_path).exists()
