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


def test_validator_profile_rejects_field_absent_from_manifest_even_when_config_allows(tmp_path):
    operator_config_path = tmp_path / "brain_operators.yaml"
    operator_config_path.write_text(
        """
operators:
  unary:
    - rank
  time_series:
    - ts_mean
fields:
  - close
  - sales
""".strip()
        + "\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "asset_manifest.yaml"
    manifest_path.write_text(
        """
default_profile: test_profile
profiles:
  test_profile:
    settings:
      instrumentType: EQUITY
      region: USA
      universe: TOP3000
      delay: 1
      language: FASTEXPR
    status: local_offline_verified
    provenance:
      verification_mode: offline_config
      source: test manifest
      live_probe: false
    verified_fields:
      - close
    verified_operators:
      - rank
      - ts_mean
    group_fields: []
""".strip()
        + "\n",
        encoding="utf-8",
    )

    result = ExpressionValidator(
        profile_name="test_profile",
        manifest_path=manifest_path,
        operator_config_path=operator_config_path,
    ).validate("rank(sales)")

    assert not result.is_valid
    assert "Unknown fields: sales." in result.errors


def test_validator_rejects_account_blocked_operator(monkeypatch):
    monkeypatch.setenv("BRAIN_OPERATOR_DENYLIST", "delay, ts_sum")

    result = ExpressionValidator().validate("rank(delay(close, 1))")

    assert not result.is_valid
    assert "Blocked operators for active account: delay." in result.errors


def test_validator_rejects_operator_from_constraint_file(tmp_path, monkeypatch):
    constraints_path = tmp_path / "operator-constraints.json"
    constraints_path.write_text(
        '{"blockedOperators":["delay"],"history":[{"operator":"delay","reason":"Attempted to use inaccessible or unknown operator \\"delay\\"","at":"2026-04-21T09:40:00Z"}]}\n',
        encoding="utf-8",
    )
    monkeypatch.delenv("BRAIN_OPERATOR_DENYLIST", raising=False)
    monkeypatch.setenv("BRAIN_OPERATOR_DENYLIST_PATH", str(constraints_path))

    result = ExpressionValidator().validate("rank(delay(close, 1))")

    assert not result.is_valid
    assert "Blocked operators for active account: delay." in result.errors
