import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from alpha_miner.main import DEFAULT_GENERATION_MODEL
from alpha_miner.main import DEFAULT_REPAIR_MODEL
from alpha_miner.main import DEFAULT_REPAIR_ROUTER_PROVIDER_NAME
from alpha_miner.main import _apply_idea_optimization
from alpha_miner.main import _diagnosis_summary
from alpha_miner.main import _quality_stage_summary
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


def test_router_yaml_pins_all_roles_to_gpt54():
    router = LLMRouter.from_yaml()

    assert {provider.model_id for provider in router._providers.values()} == {"gpt-5.4-2026-03-05"}


def test_service_openai_defaults_pin_idea_and_optimize_to_gpt54():
    source = (Path(__file__).resolve().parents[2] / "service" / "server.mjs").read_text(encoding="utf-8")

    assert 'const OPENAI_IDEA_MODEL = process.env.OPENAI_IDEA_MODEL ?? "gpt-5.4-2026-03-05";' in source
    assert 'const OPENAI_OPTIMIZE_MODEL = process.env.OPENAI_OPTIMIZE_MODEL ?? "gpt-5.4-2026-03-05";' in source


def test_runtime_defaults_are_gpt_first():
    assert DEFAULT_GENERATION_MODEL.startswith("gpt-")
    assert DEFAULT_REPAIR_MODEL.startswith("gpt-")
    assert DEFAULT_REPAIR_ROUTER_PROVIDER_NAME.startswith("gpt_")
    assert DEFAULT_REPAIR_MODEL == DEFAULT_REPAIR_CHAIN_MODEL


def test_apply_idea_optimization_uses_optimizer_output(monkeypatch):
    class _Optimizer:
        def __init__(self, router=None, kb=None):
            self.router = router
            self.kb = kb

        def optimize(self, raw_idea: str) -> dict:
            return {
                "objective": "focus on volatility compression",
                "category": "VOLATILITY",
                "hypothesis": "dispersion compresses before cross-sectional mean reversion",
                "constraints": ["avoid direct volume dependence"],
                "suggested_data_fields": ["returns", "high", "low"],
            }

    monkeypatch.setattr("alpha_miner.main.IdeaOptimizer", _Optimizer)

    result = _apply_idea_optimization(
        objective="original objective",
        router=object(),
        kb=object(),
        explicit_category=None,
    )

    assert result["category"] == "VOLATILITY"
    assert result["objective"].startswith("focus on volatility compression")
    assert "Hypothesis: dispersion compresses before cross-sectional mean reversion" in result["objective"]
    assert "Preferred data fields: returns, high, low" in result["objective"]


def test_apply_idea_optimization_respects_explicit_category(monkeypatch):
    class _Optimizer:
        def __init__(self, router=None, kb=None):
            pass

        def optimize(self, raw_idea: str) -> dict:
            return {
                "objective": "refined objective",
                "category": "VOLATILITY",
                "hypothesis": "refined hypothesis",
                "constraints": [],
                "suggested_data_fields": [],
            }

    monkeypatch.setattr("alpha_miner.main.IdeaOptimizer", _Optimizer)

    result = _apply_idea_optimization(
        objective="original objective",
        router=object(),
        kb=object(),
        explicit_category="QUALITY",
    )

    assert result["category"] == "QUALITY"


def test_main_invokes_idea_optimizer_for_non_repair_runs(tmp_path, monkeypatch):
    import alpha_miner.main as main_module

    output_dir = tmp_path / "runs" / "idea-run"
    fake_router = SimpleNamespace(
        _providers={},
        spent_usd=0.0,
        save_state=lambda path: None,
    )
    seen = {}

    class _Optimizer:
        def __init__(self, router=None, kb=None):
            seen["optimizer_router"] = router
            seen["optimizer_kb"] = kb

        def optimize(self, raw_idea: str) -> dict:
            seen["raw_idea"] = raw_idea
            return {
                "objective": "optimized objective",
                "category": "VOLATILITY",
                "hypothesis": "dispersion compresses before reversal",
                "constraints": ["avoid direct volume dependence"],
                "suggested_data_fields": ["returns", "high", "low"],
            }

    def fake_generate_batch(self, objective, category, n=10, use_llm=False, repair_context=None, diagnosis=None):
        seen["objective"] = objective
        seen["category"] = category
        seen["repair_context"] = repair_context
        return []

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(main_module, "IdeaOptimizer", _Optimizer)
    monkeypatch.setattr(main_module, "_initialize_router", lambda output_dir: fake_router)
    monkeypatch.setattr(main_module.HypothesisAgent, "generate_batch", fake_generate_batch)
    monkeypatch.setattr(
        main_module,
        "parse_args",
        lambda: Namespace(
            mode="generate",
            objective="raw objective",
            output_dir=str(output_dir),
            batch_size=1,
            rounds=1,
            category=None,
            config=None,
            concurrency=1,
            resume_from=None,
            repair_context=None,
            use_llm=False,
            sim_settings=None,
            verbose="true",
        ),
    )

    main_module.main()

    assert seen["raw_idea"] == "raw objective"
    assert seen["objective"].startswith("optimized objective")
    assert "Hypothesis: dispersion compresses before reversal" in seen["objective"]
    assert seen["category"] == "VOLATILITY"
    progress_lines = [
        line.strip()
        for line in (output_dir / "progress.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any('"stage": "idea_optimized"' in line for line in progress_lines)
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["kb_retrieval_mode"] == "uninitialized"
    assert summary["repair_retrieval_mode"] == "not_applicable"


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
        "llm_fallback": False,
        "primary_error": None,
        "primary_provider": None,
        "fallback_provider": None,
    }


def test_diagnosis_summary_preserves_llm_fallback_metadata():
    diagnosis = DiagnosisReport(
        primary_symptom="low_sharpe",
        secondary_symptoms=["high_turnover"],
        root_causes=["weak residual signal"],
        repair_priorities=[{"symptom": "low_sharpe"}],
        do_not_change=["industry neutralization"],
        raw={
            "llm_fallback": True,
            "primary_error": "Invalid primary_symptom: 'sharpe_low'",
            "primary_provider": "gpt_diagnose",
            "fallback_provider": "gpt_repair",
        },
    )

    assert _diagnosis_summary(diagnosis) == {
        "primary_symptom": "low_sharpe",
        "secondary_symptoms": ["high_turnover"],
        "fallback": False,
        "error": None,
        "llm_fallback": True,
        "primary_error": "Invalid primary_symptom: 'sharpe_low'",
        "primary_provider": "gpt_diagnose",
        "fallback_provider": "gpt_repair",
    }


def test_quality_stage_summary_reads_agent_quality_metrics():
    class _Agent:
        last_generation_quality = {
            "passed_count": 2,
            "rejected_count": 1,
            "sample_confidence": 0.91,
        }
        last_repair_quality = {
            "route": "hybrid",
            "passed_count": 1,
            "sample_confidence": 0.74,
        }

    summary = _quality_stage_summary(_Agent())

    assert summary["generation"]["passed_count"] == 2
    assert summary["generation"]["sample_confidence"] == 0.91
    assert summary["repair"]["route"] == "hybrid"
    assert summary["repair"]["sample_confidence"] == 0.74
