from alpha_miner.modules.m4_brain_backtester import BacktestResult
from alpha_miner.modules.m_quality_guardrails import (
    economic_logic_prescreen,
    should_try_sign_flip,
    sign_flip_expression,
)


def test_economic_logic_prescreen_rejects_function_style_divide():
    result = economic_logic_prescreen(
        "group_rank(ts_rank(ts_mean(divide(volume, adv20), 21), 126), industry)"
    )

    assert not result.is_valid
    assert "function_style_divide_unverified" in result.reasons


def test_economic_logic_prescreen_rejects_price_volume_unit_mismatch():
    result = economic_logic_prescreen(
        "group_rank(ts_mean(close, 21) / ts_mean(volume, 63), industry)"
    )

    assert not result.is_valid
    assert "price_volume_ratio_unit_mismatch" in result.reasons


def test_economic_logic_prescreen_allows_simple_quality_signal():
    result = economic_logic_prescreen(
        "group_rank(ts_rank(cashflow_op / assets, 252), industry)"
    )

    assert result.is_valid
    assert result.reasons == []


def test_should_try_sign_flip_for_materially_negative_sharpe_with_daily_pnl():
    result = BacktestResult(
        alpha_id="a1",
        expression="group_rank(ts_rank(returns, 63), industry)",
        period="IS",
        status="completed",
        sharpe=-0.31,
        fitness=0.42,
        turnover=0.18,
        pnl_path="/tmp/pnl.json",
        net_sharpe=-0.36,
        raw_path="/tmp/raw.json",
        simulation_id="sim-1",
        has_daily_pnl=True,
        test_sharpe=None,
    )

    assert should_try_sign_flip(result)


def test_should_not_try_sign_flip_when_loss_is_small_or_daily_pnl_missing():
    mild = BacktestResult(
        alpha_id="a1",
        expression="group_rank(ts_rank(returns, 63), industry)",
        period="IS",
        status="completed",
        sharpe=-0.09,
        fitness=0.42,
        turnover=0.18,
        pnl_path="/tmp/pnl.json",
        net_sharpe=-0.11,
        raw_path="/tmp/raw.json",
        simulation_id="sim-1",
        has_daily_pnl=True,
        test_sharpe=None,
    )
    no_pnl = BacktestResult(
        alpha_id="a2",
        expression="group_rank(ts_rank(returns, 63), industry)",
        period="IS",
        status="completed",
        sharpe=-0.31,
        fitness=0.42,
        turnover=0.18,
        pnl_path=None,
        net_sharpe=-0.36,
        raw_path="/tmp/raw.json",
        simulation_id="sim-2",
        has_daily_pnl=False,
        test_sharpe=None,
    )

    assert not should_try_sign_flip(mild)
    assert not should_try_sign_flip(no_pnl)


def test_sign_flip_expression_wraps_parent_expression():
    expression = "group_rank(ts_rank(returns, 63), industry)"

    assert sign_flip_expression(expression) == "multiply(-1, group_rank(ts_rank(returns, 63), industry))"
