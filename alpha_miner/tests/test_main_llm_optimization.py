from pathlib import Path

import pytest

from alpha_miner.main import DEFAULT_GENERATION_MODEL
from alpha_miner.main import DEFAULT_REPAIR_MODEL
from alpha_miner.main import DEFAULT_REPAIR_ROUTER_PROVIDER_NAME
from alpha_miner.main import _diagnosis_summary
from alpha_miner.main import _initialize_router
from alpha_miner.main import _primary_llm_api_key
from alpha_miner.main import _resolve_router_provider_name
from alpha_miner.main import _router_stage_metrics
from alpha_miner.main import _should_run_cadence
from alpha_miner.modules.llm_router import LLMProvider
from alpha_miner.modules.llm_router import LLMRouter
from alpha_miner.modules.m_diagnoser import DiagnosisReport
from alpha_miner.modules.m_repair_chain import DEFAULT_REPAIR_CHAIN_MODEL


def test_router_stage_metrics_aggregates_by_role():
    router = LLMRouter(
        providers=[
            LLMProvider(
                name="g1",
                role="generate",
                model_id="m",
                total_tokens_in=100,
                total_tokens_out=50,
                total_latency_ms=300.0,
                calls=2,
                cost_per_1k_tokens_usd=1.0,
            ),
            LLMProvider(
                name="g2",
                role="generate",
                model_id="m",
                total_tokens_in=40,
                total_tokens_out=10,
                total_latency_ms=100.0,
                calls=1,
                cost_per_1k_tokens_usd=2.0,
            ),
            LLMProvider(
                name="d1",
                role="distill",
                model_id="m",
                total_tokens_in=20,
                total_tokens_out=5,
                total_latency_ms=60.0,
                calls=1,
                cost_per_1k_tokens_usd=1.0,
            ),
        ]
    )

    metrics = _router_stage_metrics(router)

    assert metrics["tokens"]["generate"] == {"prompt": 140, "completion": 60}
    assert metrics["calls"]["generate"] == 3
    assert metrics["latency_ms"]["generate"]["total"] == 400.0
    assert metrics["latency_ms"]["generate"]["avg_per_call"] == round(400.0 / 3, 3)
    assert "distill" in metrics["tokens"]


def test_resolve_router_provider_name_prefers_requested_name():
    router = LLMRouter(
        providers=[
            LLMProvider(name="repair_light", role="repair"),
            LLMProvider(name="repair_hq", role="repair"),
        ]
    )
    assert _resolve_router_provider_name(router, "repair", preferred_name="repair_hq") == "repair_hq"
    assert _resolve_router_provider_name(router, "repair", preferred_name="missing") == "repair_light"


def test_should_run_cadence_every_n(tmp_path: Path):
    shared_dir = tmp_path / "runs"
    shared_dir.mkdir(parents=True, exist_ok=True)
    key = "repair_distill"
    every_n = 3
    outcomes = [_should_run_cadence(shared_dir, key, every_n) for _ in range(6)]
    assert outcomes == [False, False, True, False, False, True]


def test_initialize_router_loads_shared_state_when_run_state_missing(tmp_path: Path, monkeypatch):
    output_dir = tmp_path / "runs" / "scheduled-001"
    output_dir.mkdir(parents=True, exist_ok=True)
    shared_state_path = output_dir.parent / "llm_router_state.json"

    persisted_router = LLMRouter(
        providers=[
            LLMProvider(
                name="gpt_generate",
                role="generate",
                model_id="persisted-model",
                calls=5,
                wins=3,
                win_rate=0.6,
            ),
            LLMProvider(
                name="stale_provider",
                role="generate",
                model_id="stale-model",
                calls=2,
                wins=1,
                win_rate=0.5,
            ),
        ],
        daily_budget_usd=1.2,
    )
    persisted_router.spent_usd = 0.42
    persisted_router.save_state(shared_state_path)

    yaml_router = LLMRouter(
        providers=[
            LLMProvider(name="gpt_generate", role="generate", model_id="yaml-model"),
            LLMProvider(name="gpt_judge", role="judge", model_id="judge-model"),
        ]
    )

    monkeypatch.setattr("alpha_miner.main.LLMRouter.from_yaml", lambda: yaml_router)
    monkeypatch.setenv("LLM_BUDGET_DAILY_USD", "3.60")

    router = _initialize_router(output_dir)

    assert router is not None
    assert router._state_path == output_dir / "llm_router_state.json"
    assert ("gpt_generate", "generate") in router._providers
    assert router._providers[("gpt_generate", "generate")].calls == 5
    assert ("gpt_judge", "judge") in router._providers
    assert ("stale_provider", "generate") not in router._providers
    assert router.spent_usd == pytest.approx(0.42)
    assert router.daily_budget_usd == pytest.approx(3.60)


def test_router_yaml_avoids_legacy_anthropic_model_aliases():
    router = LLMRouter.from_yaml()

    role_to_model = {provider.role: provider.model_id for provider in router._providers.values()}

    assert role_to_model["distill"] != "claude-sonnet-4"
    assert role_to_model["diagnose"] != "claude-sonnet-4"
    assert role_to_model["retrieve_summary"] != "claude-sonnet-4"


def test_router_yaml_defaults_to_openai_stack():
    router = LLMRouter.from_yaml()

    providers = list(router._providers.values())

    assert providers
    assert all(provider.api_key_env == "OPENAI_API_KEY" for provider in providers)
    assert all(provider.client_type == "openai_compat" for provider in providers)
    assert all(provider.model_id.startswith("gpt-") for provider in providers)


def test_runtime_defaults_are_gpt_first():
    assert DEFAULT_GENERATION_MODEL.startswith("gpt-")
    assert DEFAULT_REPAIR_MODEL.startswith("gpt-")
    assert DEFAULT_REPAIR_ROUTER_PROVIDER_NAME.startswith("gpt_")
    assert DEFAULT_REPAIR_MODEL == DEFAULT_REPAIR_CHAIN_MODEL


@pytest.mark.parametrize(
    ("model_id", "openai_key", "anthropic_key", "expected"),
    [
        ("gpt-5.4-2026-03-05", "openai-key", "anthropic-key", "openai-key"),
        ("claude-sonnet-4-6", "openai-key", "anthropic-key", "anthropic-key"),
    ],
)
def test_primary_llm_api_key_routes_by_model_family(
    monkeypatch,
    model_id: str,
    openai_key: str,
    anthropic_key: str,
    expected: str,
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", anthropic_key)

    assert _primary_llm_api_key(model_id, openai_key) == expected


def test_diagnosis_summary_preserves_fallback_error():
    diagnosis = DiagnosisReport(
        primary_symptom="low_sharpe",
        secondary_symptoms=["high_turnover"],
        root_causes=["rule_based_fallback:low_sharpe"],
        repair_priorities=[{"symptom": "low_sharpe"}],
        do_not_change=["industry neutralization"],
        raw={"fallback": True, "error": "diagnose timeout"},
    )

    assert _diagnosis_summary(diagnosis) == {
        "primary_symptom": "low_sharpe",
        "secondary_symptoms": ["high_turnover"],
        "fallback": True,
        "error": "diagnose timeout",
    }
