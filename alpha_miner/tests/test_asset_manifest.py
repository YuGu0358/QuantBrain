from __future__ import annotations

import json
import os
import subprocess
import sys

from alpha_miner.modules.asset_manifest import (
    DEFAULT_PROFILE_NAME,
    get_asset_profile,
    load_asset_manifest,
)
from alpha_miner.modules.common import PROJECT_ROOT
from alpha_miner.modules.config_loader import load_operator_config, load_yaml


def _configured_operators() -> set[str]:
    config = load_operator_config()
    operators: set[str] = set()
    for values in (config.get("operators") or {}).values():
        operators.update(values or [])
    return operators


def test_default_profile_loads_safe_brain_settings_and_symbols():
    manifest = load_asset_manifest()
    profile = get_asset_profile(DEFAULT_PROFILE_NAME, manifest=manifest)

    assert profile.name == "equity_usa_top3000"
    assert profile.settings == {
        "instrumentType": "EQUITY",
        "region": "USA",
        "universe": "TOP3000",
        "delay": 1,
        "language": "FASTEXPR",
    }
    assert profile.status == "local_offline_verified"
    assert profile.provenance["verification_mode"] == "offline_config"
    assert profile.provenance["live_probe"] is False
    assert set(profile.group_fields) == {"industry", "subindustry"}
    assert set(profile.verified_fields) == set(load_operator_config().get("fields") or [])
    assert set(profile.verified_operators) == _configured_operators()


def test_dataset_probe_emits_manifest_json_without_credentials():
    env = {**os.environ}
    env.pop("WQB_EMAIL", None)
    env.pop("WQB_PASSWORD", None)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "alpha_miner.scripts.dataset_probe",
            "--profile",
            DEFAULT_PROFILE_NAME,
            "--format",
            "json",
        ],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    payload = json.loads(result.stdout)
    profile = payload["profiles"][DEFAULT_PROFILE_NAME]
    assert profile["provenance"]["live_probe"] is False
    assert profile["settings"]["region"] == "USA"
    assert "operating_income" in profile["verified_fields"]


def test_dataset_probe_writes_manifest_yaml_without_credentials(tmp_path):
    output_path = tmp_path / "asset_manifest.yaml"
    env = {**os.environ}
    env.pop("WQB_EMAIL", None)
    env.pop("WQB_PASSWORD", None)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "alpha_miner.scripts.dataset_probe",
            "--profile",
            DEFAULT_PROFILE_NAME,
            "--format",
            "yaml",
            "--output",
            str(output_path),
        ],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    payload = load_yaml(output_path)
    profile = payload["profiles"][DEFAULT_PROFILE_NAME]
    assert profile["provenance"]["verification_mode"] == "offline_config"
    assert profile["provenance"]["live_probe"] is False
