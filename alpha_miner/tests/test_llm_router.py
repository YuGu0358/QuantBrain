from alpha_miner.modules.llm_router import LLMRouter


def test_pick_returns_provider_for_role():
    router = LLMRouter(
        providers=[
            {"name": "deepseek", "role": "generate"},
            {"name": "haiku", "role": "repair"},
        ]
    )

    assert router.pick("generate").name == "deepseek"
    assert router.pick("repair").name == "haiku"


def test_epsilon_greedy_explores():
    router = LLMRouter(
        providers=[
            {"name": "deepseek", "role": "generate", "win_rate": 0.9},
            {"name": "gemini", "role": "generate", "win_rate": 0.1},
        ],
        epsilon=1.0,
    )

    seen = {router.pick("generate").name for _ in range(20)}

    assert {"deepseek", "gemini"}.issubset(seen)


def test_record_result_updates_win_rate():
    router = LLMRouter(providers=[{"name": "deepseek", "role": "generate"}])

    router.record_result(
        "deepseek",
        "generate",
        passed=True,
        latency_ms=100,
        tokens_in=500,
        tokens_out=200,
    )

    assert router.get_state()["providers"]["deepseek"]["generate"]["win_rate"] == 1.0


def test_budget_check_blocks_when_exceeded():
    daily_budget_usd = 1.0
    router = LLMRouter(
        providers=[
            {
                "name": "cheap_provider",
                "role": "generate",
                "cost_per_1k_tokens_usd": 0.0001,
            },
            {
                "name": "expensive_provider",
                "role": "generate",
                "cost_per_1k_tokens_usd": 1.0,
            },
        ],
        daily_budget_usd=daily_budget_usd,
    )

    # Spend >80% of budget using the expensive provider (1.0 USD/1k tokens * 900 tokens = 0.9 USD = 90%)
    router.record_result(
        "expensive_provider",
        "generate",
        passed=True,
        latency_ms=100,
        tokens_in=900,
        tokens_out=0,
    )

    # After >80% budget used, pick() must route to the cheapest provider
    assert router.pick("generate").name == "cheap_provider"
    # Budget was consumed (remaining < daily budget)
    assert router.budget_remaining_usd() < daily_budget_usd
    # Also verify over-budget scenario still works (budget_remaining < 0 when fully exceeded)
    router.record_result(
        "expensive_provider",
        "generate",
        passed=True,
        latency_ms=100,
        tokens_in=100000,
        tokens_out=100000,
    )
    assert router.budget_remaining_usd() < 0


def test_state_roundtrip(tmp_path):
    router = LLMRouter(providers=[{"name": "deepseek", "role": "generate"}])
    router.record_result(
        "deepseek",
        "generate",
        passed=True,
        latency_ms=100,
        tokens_in=500,
        tokens_out=200,
    )
    state_path = tmp_path / "llm_router_state.json"

    router.save_state(state_path)
    loaded = LLMRouter.load_state(state_path)

    original_win_rate = router.get_state()["providers"]["deepseek"]["generate"]["win_rate"]
    loaded_win_rate = loaded.get_state()["providers"]["deepseek"]["generate"]["win_rate"]
    assert loaded_win_rate == original_win_rate
