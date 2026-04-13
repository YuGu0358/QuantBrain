from alpha_miner.modules.m3_validator import ExpressionValidator


def test_validator_accepts_known_expression():
    result = ExpressionValidator().validate("group_rank(ts_rank(operating_income / assets, 252), industry)")
    assert result.is_valid
    assert "group_rank" in result.operators
    assert "operating_income" in result.fields


def test_validator_rejects_hallucinated_operator():
    result = ExpressionValidator().validate("magic_alpha(close, 20)")
    assert not result.is_valid
    assert any("Unknown operators" in error for error in result.errors)


def test_validator_rejects_unbalanced_parentheses():
    result = ExpressionValidator().validate("rank(ts_mean(close, 20)")
    assert not result.is_valid
    assert "Parentheses are not balanced." in result.errors
