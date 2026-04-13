# Multi-LLM Agentic Alpha Mining — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **⚠️ ALL implementation steps must be executed via `codex "<prompt>"` — do NOT write code directly.**

**Goal:** Replace QuantBrain's disabled/template LLM path with a parallel multi-LLM system (Deepseek V3 + Gemini Flash for generation, Claude Haiku for repair, GPT-4o-mini for judging/distillation) that self-optimises via ε-greedy routing and post-run knowledge distillation.

**Architecture:** A new `LLMRouter` manages all provider connections, tracks per-role win rates, and applies ε-greedy selection. After every run, `KnowledgeDistiller` calls GPT-4o-mini to extract success/failure patterns into four new SQLite tables, enriching the RAG context used by the next generation cycle. A redesigned Apple-style Dashboard and a new `/ideas/optimize` endpoint expose both auto and user-guided modes.

**Tech Stack:** Python 3.11+, sqlite3, `openai` SDK (also used for Deepseek/Gemini OpenAI-compat endpoints), `anthropic` SDK, `concurrent.futures` for parallel generation, Node.js ESM for server changes.

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| **Create** | `alpha_miner/config/llm_providers.yaml` | Provider configs: model IDs, API bases, cost/token |
| **Create** | `alpha_miner/modules/llm_router.py` | Provider registry, ε-greedy selection, budget tracking, state persistence |
| **Create** | `alpha_miner/modules/m9_knowledge_distiller.py` | Post-run GPT-4o-mini distillation → KnowledgeBase |
| **Create** | `alpha_miner/modules/idea_optimizer.py` | Natural language idea → structured research direction via Claude Sonnet |
| **Create** | `alpha_miner/tests/test_llm_router.py` | Router unit tests |
| **Create** | `alpha_miner/tests/test_knowledge_distiller.py` | Distiller unit tests |
| **Create** | `alpha_miner/tests/test_idea_optimizer.py` | Optimizer unit tests |
| **Modify** | `alpha_miner/modules/m1_knowledge_base.py` | Add 4 new tables + write/query interfaces |
| **Modify** | `alpha_miner/modules/m2_hypothesis_agent.py` | Enable `_call_llm()` via router; inject enriched RAG context |
| **Modify** | `alpha_miner/main.py` | Parallel generation; post-run distillation hook; router wiring |
| **Modify** | `service/server.mjs` | `POST /ideas/optimize` endpoint; LLM Router state in `/runs`; new Dashboard HTML |
| **Modify** | `alpha_miner/requirements.txt` | Add `google-generativeai` if needed (Gemini Flash via OpenAI-compat doesn't need it) |
| **Modify** | `.env.example` | Add 6 new env vars |

---

## Task 1: LLM Providers Config + Router Foundation

**Files:**
- Create: `alpha_miner/config/llm_providers.yaml`
- Create: `alpha_miner/modules/llm_router.py`
- Create: `alpha_miner/tests/test_llm_router.py`

- [ ] **Step 1.1 — Write failing tests for LLMRouter**

```bash
codex "In /Users/yugu/Downloads/QuantBrain_claud/alpha_miner/tests/test_llm_router.py write pytest tests for a LLMRouter class (to be created at alpha_miner/modules/llm_router.py).

Tests to write:

1. test_pick_returns_provider_for_role
   - Create a LLMRouter with two providers: name='deepseek', role='generate'; name='haiku', role='repair'
   - router.pick('generate') must return the deepseek provider
   - router.pick('repair') must return the haiku provider

2. test_epsilon_greedy_explores
   - Create a LLMRouter with two generate providers: 'deepseek' (win_rate=0.9) and 'gemini' (win_rate=0.1)
   - With epsilon=1.0 (always explore), pick('generate') must sometimes return 'gemini'
   - Run 20 picks; assert both providers appear at least once

3. test_record_result_updates_win_rate
   - Create a LLMRouter with one provider name='deepseek', role='generate'
   - Call router.record_result('deepseek', 'generate', passed=True, latency_ms=100, tokens_in=500, tokens_out=200)
   - Assert router.get_state()['providers']['deepseek']['generate']['win_rate'] == 1.0

4. test_budget_check_blocks_when_exceeded
   - Create a LLMRouter with daily_budget_usd=0.01
   - Call router.record_result(..., tokens_in=100000, tokens_out=100000) with a provider costing $1/1k tokens in+out
   - Assert router.budget_remaining_usd() < 0

5. test_state_roundtrip
   - Create router, record one result, save state to tmp_path
   - Load a new router from the same state file
   - Assert win_rate matches original

Use tmp_path pytest fixture. The LLMRouter class does not exist yet — these tests should all fail with ImportError or AttributeError when run."
```

- [ ] **Step 1.2 — Run tests to verify they fail**

```bash
cd /Users/yugu/Downloads/QuantBrain_claud && PYTHONPATH=. pytest alpha_miner/tests/test_llm_router.py -v 2>&1 | head -30
```
Expected: `ImportError` — `llm_router` module not found.

- [ ] **Step 1.3 — Create llm_providers.yaml**

```bash
codex "Create /Users/yugu/Downloads/QuantBrain_claud/alpha_miner/config/llm_providers.yaml with the following provider definitions:

providers:
  deepseek_v3:
    role: generate
    model_id: deepseek-chat
    api_base: https://api.deepseek.com/v1
    api_key_env: DEEPSEEK_API_KEY
    client_type: openai_compat
    cost_per_1k_input: 0.00027
    cost_per_1k_output: 0.00110
    weight_initial: 0.5

  gemini_flash:
    role: generate
    model_id: gemini-1.5-flash
    api_base: https://generativelanguage.googleapis.com/v1beta/openai/
    api_key_env: GEMINI_API_KEY
    client_type: openai_compat
    cost_per_1k_input: 0.000075
    cost_per_1k_output: 0.00030
    weight_initial: 0.5

  claude_haiku:
    role: repair
    model_id: claude-haiku-4-5-20251001
    api_base: null
    api_key_env: ANTHROPIC_API_KEY
    client_type: anthropic
    cost_per_1k_input: 0.00025
    cost_per_1k_output: 0.00125
    weight_initial: 0.75

  gpt4o_mini_repair:
    role: repair
    model_id: gpt-4o-mini
    api_base: null
    api_key_env: OPENAI_API_KEY
    client_type: openai_compat
    cost_per_1k_input: 0.00015
    cost_per_1k_output: 0.00060
    weight_initial: 0.25

  gpt4o_mini_judge:
    role: judge
    model_id: gpt-4o-mini
    api_base: null
    api_key_env: OPENAI_API_KEY
    client_type: openai_compat
    cost_per_1k_input: 0.00015
    cost_per_1k_output: 0.00060
    weight_initial: 1.0

  gpt4o_mini_distill:
    role: distill
    model_id: gpt-4o-mini
    api_base: null
    api_key_env: OPENAI_API_KEY
    client_type: openai_compat
    cost_per_1k_input: 0.00015
    cost_per_1k_output: 0.00060
    weight_initial: 1.0

  claude_sonnet_idea:
    role: idea
    model_id: claude-sonnet-4-6
    api_base: null
    api_key_env: ANTHROPIC_API_KEY
    client_type: anthropic
    cost_per_1k_input: 0.003
    cost_per_1k_output: 0.015
    weight_initial: 1.0"
```

- [ ] **Step 1.4 — Implement LLMRouter**

```bash
codex "Create /Users/yugu/Downloads/QuantBrain_claud/alpha_miner/modules/llm_router.py

This module manages LLM provider selection using ε-greedy routing. Study the existing codebase style:
- /Users/yugu/Downloads/QuantBrain_claud/alpha_miner/modules/common.py (read_json, write_json, sha256_json)
- /Users/yugu/Downloads/QuantBrain_claud/alpha_miner/modules/config_loader.py (load_yaml)
- /Users/yugu/Downloads/QuantBrain_claud/alpha_miner/config/llm_providers.yaml (provider schema)

Implement the following:

@dataclass
class ProviderConfig:
    name: str
    role: str
    model_id: str
    api_base: str | None
    api_key_env: str
    client_type: str   # 'openai_compat' | 'anthropic'
    cost_per_1k_input: float
    cost_per_1k_output: float

@dataclass
class ProviderStats:
    attempts: int = 0
    wins: int = 0
    total_cost_usd: float = 0.0
    total_latency_ms: float = 0.0

    @property
    def win_rate(self) -> float:
        return self.wins / self.attempts if self.attempts > 0 else 0.5

class LLMRouter:
    def __init__(
        self,
        providers_yaml_path: Path | None = None,
        state_path: Path | None = None,
        daily_budget_usd: float = 3.60,
        epsilon: float = 0.10,
        rebalance_every: int = 10,
    ):
        # Load provider configs from llm_providers.yaml
        # Load or initialise per-provider stats from state_path (JSON)
        # state shape: {'providers': {name: {'role': role, 'attempts': n, 'wins': n, 'total_cost_usd': f, 'total_latency_ms': f}}, 'day': 'YYYY-MM-DD', 'day_cost_usd': f}

    def pick(self, role: str) -> ProviderConfig:
        # Get all providers matching role
        # With probability epsilon: pick uniformly at random from role providers
        # With probability 1-epsilon: pick provider with highest win_rate
        # If daily budget > 80% used: always pick cheapest provider for this role
        # Raise ValueError if no provider found for role

    def record_result(
        self,
        provider_name: str,
        role: str,
        passed: bool,
        latency_ms: float,
        tokens_in: int,
        tokens_out: int,
    ) -> None:
        # Update stats for provider_name
        # Calculate cost: (tokens_in/1000)*cost_per_1k_input + (tokens_out/1000)*cost_per_1k_output
        # Accumulate day_cost_usd (reset if new day)
        # After every rebalance_every total attempts for this role, recompute weights
        # Save state to state_path

    def budget_remaining_usd(self) -> float:
        # Returns daily_budget_usd - day_cost_usd for today

    def get_state(self) -> dict:
        # Returns full state dict (for dashboard and tests)

    def _save_state(self) -> None: ...
    def _load_state(self) -> None: ...

Use from __future__ import annotations. Use random.random() for epsilon check. Use PACKAGE_ROOT from common.py to default providers_yaml_path to alpha_miner/config/llm_providers.yaml. Default state_path is None (no persistence if not set)."
```

- [ ] **Step 1.5 — Run tests**

```bash
cd /Users/yugu/Downloads/QuantBrain_claud && PYTHONPATH=. pytest alpha_miner/tests/test_llm_router.py -v
```
Expected: all 5 tests PASS.

- [ ] **Step 1.6 — Commit**

```bash
cd /Users/yugu/Downloads/QuantBrain_claud && git add alpha_miner/config/llm_providers.yaml alpha_miner/modules/llm_router.py alpha_miner/tests/test_llm_router.py && git commit -m "feat: add LLMRouter with epsilon-greedy routing and budget tracking"
```

---

## Task 2: KnowledgeBase Schema Extension

**Files:**
- Modify: `alpha_miner/modules/m1_knowledge_base.py`

- [ ] **Step 2.1 — Write failing tests**

```bash
codex "In /Users/yugu/Downloads/QuantBrain_claud/alpha_miner/tests/test_knowledge_base_extended.py write pytest tests for new methods to be added to KnowledgeBase in m1_knowledge_base.py.

Read the current KnowledgeBase at /Users/yugu/Downloads/QuantBrain_claud/alpha_miner/modules/m1_knowledge_base.py first.

Tests to write:

1. test_record_strategy_stat_and_query
   - kb.record_strategy_stat(category='QUALITY', gate_result='PASS', operator_skeleton='rank(ts_rank(X,N))')
   - stats = kb.get_strategy_stats('QUALITY')
   - assert stats['attempts'] == 1 and stats['wins'] == 1

2. test_record_failure_pattern
   - kb.record_failure_pattern(reason='LOW_SHARPE', expression='rank(close)', suggested_fix='add peer-relative normalisation')
   - patterns = kb.get_failure_patterns(limit=5)
   - assert len(patterns) == 1
   - assert patterns[0]['reason'] == 'LOW_SHARPE'

3. test_record_operator_stat
   - kb.record_operator_stat(operator='ts_rank', category='QUALITY', passed=True)
   - kb.record_operator_stat(operator='ts_rank', category='QUALITY', passed=False)
   - stats = kb.get_operator_stats('QUALITY')
   - assert stats['ts_rank']['attempts'] == 2 and stats['ts_rank']['wins'] == 1

4. test_upsert_market_regime
   - kb.upsert_market_regime(regime_key='current', summary='momentum leading, quality lagging', top_categories=['MOMENTUM', 'REVERSAL'])
   - regime = kb.get_market_regime('current')
   - assert regime['summary'] == 'momentum leading, quality lagging'
   - assert 'MOMENTUM' in regime['top_categories']

5. test_enriched_rag_context_includes_patterns
   - After recording one failure pattern and one strategy stat (QUALITY, PASS)
   - ctx = kb.rag_context('QUALITY')
   - assert hasattr(ctx, 'failure_patterns')
   - assert hasattr(ctx, 'strategy_stats')

All tests use tmp_path. The new methods do not exist yet — tests must fail."
```

- [ ] **Step 2.2 — Run to verify failure**

```bash
cd /Users/yugu/Downloads/QuantBrain_claud && PYTHONPATH=. pytest alpha_miner/tests/test_knowledge_base_extended.py -v 2>&1 | head -20
```
Expected: `AttributeError` on new methods.

- [ ] **Step 2.3 — Implement schema extension**

```bash
codex "Modify /Users/yugu/Downloads/QuantBrain_claud/alpha_miner/modules/m1_knowledge_base.py

Read the full file first. Do NOT remove any existing code. Add:

1. Extend RagContext dataclass with two new optional fields:
   failure_patterns: list[dict] = field(default_factory=list)
   strategy_stats: dict[str, Any] = field(default_factory=dict)

2. In _init_db(), add CREATE TABLE IF NOT EXISTS for these 4 tables:

   strategy_stats (category TEXT, gate_result TEXT, operator_skeleton TEXT, ts INTEGER)
   failure_patterns (id INTEGER PRIMARY KEY AUTOINCREMENT, reason TEXT, expression TEXT, suggested_fix TEXT, ts INTEGER)
   operator_stats (operator TEXT, category TEXT, passed INTEGER, ts INTEGER)
   market_regime (regime_key TEXT PRIMARY KEY, summary TEXT, top_categories_json TEXT, ts INTEGER)

3. Add these methods to KnowledgeBase:

   def record_strategy_stat(self, category: str, gate_result: str, operator_skeleton: str = '') -> None
     # INSERT into strategy_stats with int(time.time()) as ts

   def get_strategy_stats(self, category: str) -> dict[str, Any]
     # SELECT from strategy_stats WHERE category=?
     # Return {'attempts': N, 'wins': N (gate_result='PASS'), 'win_rate': float}

   def record_failure_pattern(self, reason: str, expression: str, suggested_fix: str) -> None
     # INSERT into failure_patterns

   def get_failure_patterns(self, limit: int = 10) -> list[dict[str, Any]]
     # SELECT from failure_patterns ORDER BY ts DESC LIMIT ?
     # Return list of dicts with keys: reason, expression, suggested_fix

   def record_operator_stat(self, operator: str, category: str, passed: bool) -> None
     # INSERT into operator_stats

   def get_operator_stats(self, category: str) -> dict[str, dict[str, Any]]
     # SELECT from operator_stats WHERE category=?
     # Return {operator: {'attempts': N, 'wins': N, 'win_rate': float}}

   def upsert_market_regime(self, regime_key: str, summary: str, top_categories: list[str]) -> None
     # INSERT OR REPLACE into market_regime

   def get_market_regime(self, regime_key: str) -> dict[str, Any] | None
     # Returns {'summary': str, 'top_categories': list[str]} or None

4. In rag_context(), after building positive/negative lists, also fetch:
   - failure_patterns = self.get_failure_patterns(limit=5)
   - strategy_stats = self.get_strategy_stats(category)
   Then return RagContext(positive=positive, negative=negative, failure_patterns=failure_patterns, strategy_stats=strategy_stats)

Import time at the top if not already present."
```

- [ ] **Step 2.4 — Run all KB tests**

```bash
cd /Users/yugu/Downloads/QuantBrain_claud && PYTHONPATH=. pytest alpha_miner/tests/test_knowledge_base_extended.py -v
```
Expected: all 5 PASS.

- [ ] **Step 2.5 — Run existing tests to confirm no regression**

```bash
cd /Users/yugu/Downloads/QuantBrain_claud && PYTHONPATH=. pytest alpha_miner/tests/ -v --ignore=alpha_miner/tests/test_llm_router.py 2>&1 | tail -20
```
Expected: all existing tests still PASS.

- [ ] **Step 2.6 — Commit**

```bash
cd /Users/yugu/Downloads/QuantBrain_claud && git add alpha_miner/modules/m1_knowledge_base.py alpha_miner/tests/test_knowledge_base_extended.py && git commit -m "feat: extend KnowledgeBase with strategy_stats, failure_patterns, operator_stats, market_regime tables"
```

---

## Task 3: Multi-LLM Generation (enable `_call_llm` + parallel)

**Files:**
- Modify: `alpha_miner/modules/m2_hypothesis_agent.py`
- Modify: `alpha_miner/main.py`

- [ ] **Step 3.1 — Write failing test for multi-LLM generation**

```bash
codex "In /Users/yugu/Downloads/QuantBrain_claud/alpha_miner/tests/test_hypothesis_agent_multi_llm.py write pytest tests for the updated HypothesisAgent.

Read /Users/yugu/Downloads/QuantBrain_claud/alpha_miner/modules/m2_hypothesis_agent.py first.

Tests:

1. test_generate_batch_calls_router_when_use_llm_true
   - Create a mock LLMRouter that returns a fake ProviderConfig(name='mock', role='generate', model_id='mock', ...)
   - Patch the HTTP call so it returns {'candidates': [{'id':'c1','category':'QUALITY','hypothesis':'h','expression':'rank(close)','origin_refs':[]}]}
   - agent = HypothesisAgent(kb, cache, taxonomy, router=mock_router, use_llm=True)
   - result = agent.generate_batch('test objective', category='QUALITY', n=1)
   - assert len(result) == 1
   - assert mock_router.pick was called with 'generate'

2. test_prompt_includes_rag_failure_patterns
   - Create a KB with one recorded failure_pattern (reason='LOW_SHARPE', expression='rank(close)', suggested_fix='add ts_rank')
   - Build request_payload via agent._request_payload('obj', 'QUALITY', 1)
   - assert 'LOW_SHARPE' in str(request_payload)

3. test_judge_selects_best_candidates
   - Create 4 Candidate objects with different expressions
   - Patch the judge call to return indices [0, 2] as the best
   - result = agent._judge_candidates(candidates, n=2, router=mock_router)
   - assert len(result) == 2

Use unittest.mock.patch and tmp_path. Methods don't exist yet — tests must fail."
```

- [ ] **Step 3.2 — Verify tests fail**

```bash
cd /Users/yugu/Downloads/QuantBrain_claud && PYTHONPATH=. pytest alpha_miner/tests/test_hypothesis_agent_multi_llm.py -v 2>&1 | head -20
```

- [ ] **Step 3.3 — Implement multi-LLM generation**

```bash
codex "Modify /Users/yugu/Downloads/QuantBrain_claud/alpha_miner/modules/m2_hypothesis_agent.py

Read the full file first. Keep all existing code. Make these changes:

1. Update __init__ signature to accept two new optional parameters:
   - router: LLMRouter | None = None  (import from .llm_router)
   - use_llm: bool = False

2. Implement _call_llm(self, request_payload: dict) -> dict:
   - If self.router is None: raise NotImplementedError (existing behaviour)
   - provider = self.router.pick('generate')
   - import time; t0 = time.time()
   - If provider.client_type == 'openai_compat':
       from openai import OpenAI
       client = OpenAI(api_key=os.environ.get(provider.api_key_env, ''), base_url=provider.api_base)
       response = client.chat.completions.create(
           model=provider.model_id,
           messages=request_payload['messages'],
           temperature=request_payload.get('temperature', 0.4),
           max_tokens=request_payload.get('max_tokens', 800),
       )
       raw_text = response.choices[0].message.content or ''
       tokens_in = response.usage.prompt_tokens if response.usage else 0
       tokens_out = response.usage.completion_tokens if response.usage else 0
   - If provider.client_type == 'anthropic':
       from anthropic import Anthropic
       client = Anthropic(api_key=os.environ.get(provider.api_key_env, ''))
       msgs = [m for m in request_payload['messages'] if m['role'] != 'system']
       system_msg = next((m['content'] for m in request_payload['messages'] if m['role'] == 'system'), '')
       response = client.messages.create(
           model=provider.model_id,
           max_tokens=request_payload.get('max_tokens', 800),
           system=system_msg,
           messages=msgs,
       )
       raw_text = response.content[0].text if response.content else ''
       tokens_in = response.usage.input_tokens if response.usage else 0
       tokens_out = response.usage.output_tokens if response.usage else 0
   - latency_ms = (time.time() - t0) * 1000
   - Parse raw_text as JSON; extract candidates list
   - passed = len(candidates) > 0
   - self.router.record_result(provider.name, 'generate', passed, latency_ms, tokens_in, tokens_out)
   - Return {'candidates': candidates}

3. Add _judge_candidates(self, candidates: list[Candidate], n: int, router: LLMRouter) -> list[Candidate]:
   - If router is None or len(candidates) <= n: return candidates[:n]
   - provider = router.pick('judge')
   - Build a prompt asking GPT-4o-mini to select the best n candidates from the list
   - Prompt format (JSON user message):
     {'task': 'select_best_candidates', 'n': n, 'candidates': [{'index': i, 'expression': c.expression, 'hypothesis': c.hypothesis} for i,c in enumerate(candidates)]}
   - System: 'You are a quant alpha judge. Select the n best candidates for WorldQuant BRAIN submission. Favour: (1) expression-hypothesis alignment, (2) operator diversity vs other candidates, (3) avoidance of known failure patterns. Return JSON: {\"selected_indices\": [list of int]}'
   - Call provider via same openai_compat/anthropic pattern as _call_llm
   - Parse selected_indices; return [candidates[i] for i in selected_indices if i < len(candidates)][:n]
   - On any parse error: return candidates[:n] as fallback

4. Update generate_batch to:
   - If self.use_llm and self.router:
       Use concurrent.futures.ThreadPoolExecutor(max_workers=2) to call _call_llm in parallel for two separate request payloads (same objective/category/n, but vary the 'seed' field to get different outputs)
       Merge the two candidate lists (deduplicate by expression)
       Call _judge_candidates(merged, n, self.router) to pick best n
   - Else: existing deterministic/cache path (no change)

5. Update _request_payload to include rag failure_patterns and strategy_stats from self.kb.rag_context():
   - After building base payload, add to the user message content dict:
     'failure_patterns_to_avoid': [p['reason'] + ': ' + p['expression'] for p in context.failure_patterns]
     'strategy_stats': context.strategy_stats

Import os at the top. Import LLMRouter with TYPE_CHECKING guard to avoid circular import."
```

- [ ] **Step 3.4 — Run tests**

```bash
cd /Users/yugu/Downloads/QuantBrain_claud && PYTHONPATH=. pytest alpha_miner/tests/test_hypothesis_agent_multi_llm.py -v
```
Expected: all 3 PASS.

- [ ] **Step 3.5 — Commit**

```bash
cd /Users/yugu/Downloads/QuantBrain_claud && git add alpha_miner/modules/m2_hypothesis_agent.py alpha_miner/tests/test_hypothesis_agent_multi_llm.py && git commit -m "feat: enable multi-LLM parallel generation with router and judge in HypothesisAgent"
```

---

## Task 4: Knowledge Distiller

**Files:**
- Create: `alpha_miner/modules/m9_knowledge_distiller.py`
- Create: `alpha_miner/tests/test_knowledge_distiller.py`

- [ ] **Step 4.1 — Write failing tests**

```bash
codex "In /Users/yugu/Downloads/QuantBrain_claud/alpha_miner/tests/test_knowledge_distiller.py write pytest tests for KnowledgeDistiller (to be created at alpha_miner/modules/m9_knowledge_distiller.py).

Tests:

1. test_distill_records_failure_patterns_from_batch
   - Create a batch_result dict simulating a completed run:
     {'records': [{'candidate': {'expression': 'rank(close)', 'category': 'QUALITY'}, 'backtest': {'status': 'completed', 'sharpe': 0.3}, 'scorecard': {'gate': 'FAIL', 'reason': 'LOW_SHARPE'}}]}
   - Patch the LLM call to return:
     {'failure_patterns': [{'reason': 'LOW_SHARPE', 'expression': 'rank(close)', 'suggested_fix': 'use ts_rank for persistence'}], 'operator_stats': [], 'market_regime': None}
   - distiller = KnowledgeDistiller(kb=kb, router=mock_router)
   - distiller.distill(batch_result)
   - patterns = kb.get_failure_patterns()
   - assert len(patterns) == 1 and patterns[0]['reason'] == 'LOW_SHARPE'

2. test_distill_records_strategy_stat_for_pass
   - batch_result with one PASS candidate (category='MOMENTUM', gate='PASS')
   - distiller.distill(batch_result)
   - stats = kb.get_strategy_stats('MOMENTUM')
   - assert stats['wins'] == 1

3. test_distill_skips_when_router_none
   - distiller = KnowledgeDistiller(kb=kb, router=None)
   - distiller.distill({'records': []})  # must not raise

Use tmp_path and unittest.mock.patch."
```

- [ ] **Step 4.2 — Verify failure**

```bash
cd /Users/yugu/Downloads/QuantBrain_claud && PYTHONPATH=. pytest alpha_miner/tests/test_knowledge_distiller.py -v 2>&1 | head -20
```

- [ ] **Step 4.3 — Implement KnowledgeDistiller**

```bash
codex "Create /Users/yugu/Downloads/QuantBrain_claud/alpha_miner/modules/m9_knowledge_distiller.py

Read these files first:
- alpha_miner/modules/m1_knowledge_base.py
- alpha_miner/modules/llm_router.py
- alpha_miner/modules/common.py

Implement:

class KnowledgeDistiller:
    def __init__(self, kb: KnowledgeBase, router: LLMRouter | None = None):
        self.kb = kb
        self.router = router

    def distill(self, batch_result: dict) -> None:
        # 1. Always record strategy_stat for each candidate based on gate result (no LLM needed):
        for record in batch_result.get('records', []):
            candidate = record.get('candidate', {})
            category = candidate.get('category', '')
            gate_result = record.get('scorecard', {}).get('gate', 'UNKNOWN')
            expression = candidate.get('expression', '')
            operator_skeleton = _extract_operator_skeleton(expression)
            self.kb.record_strategy_stat(category=category, gate_result=gate_result, operator_skeleton=operator_skeleton)
            # record per-operator stats too
            for op in _extract_operators(expression):
                self.kb.record_operator_stat(operator=op, category=category, passed=(gate_result == 'PASS'))

        # 2. If router available, call GPT-4o-mini to extract failure patterns and market regime:
        if self.router is None:
            return
        failed_records = [r for r in batch_result.get('records', []) if r.get('scorecard', {}).get('gate') == 'FAIL']
        if not failed_records:
            return
        provider = self.router.pick('distill')
        prompt_data = {
            'task': 'extract_failure_patterns',
            'failed_candidates': [
                {'expression': r['candidate']['expression'],
                 'reason': r.get('scorecard', {}).get('reason', 'UNKNOWN'),
                 'sharpe': r.get('backtest', {}).get('sharpe')}
                for r in failed_records[:10]
            ]
        }
        system = ('You are a quant alpha research assistant. Given failed alpha candidates, '
                  'extract: (1) failure_patterns: list of {reason, expression, suggested_fix}, '
                  '(2) operator_stats: list of {operator, passed: bool}, '
                  '(3) market_regime: {summary, top_categories} or null. '
                  'Return strict JSON with keys: failure_patterns, operator_stats, market_regime.')
        result = _call_distill_llm(provider, system, prompt_data)
        import time
        for pattern in result.get('failure_patterns', []):
            self.kb.record_failure_pattern(
                reason=pattern.get('reason', ''),
                expression=pattern.get('expression', ''),
                suggested_fix=pattern.get('suggested_fix', ''),
            )
        regime = result.get('market_regime')
        if regime and isinstance(regime, dict):
            self.kb.upsert_market_regime(
                regime_key='current',
                summary=regime.get('summary', ''),
                top_categories=regime.get('top_categories', []),
            )

def _call_distill_llm(provider, system: str, data: dict) -> dict:
    # openai_compat or anthropic — same pattern as m2_hypothesis_agent._call_llm
    # Return parsed JSON dict; on error return {}

def _extract_operator_skeleton(expression: str) -> str:
    # Replace all numeric literals and identifiers that aren't function names with X/N
    # e.g. 'rank(ts_rank(operating_income/assets,252))' -> 'rank(ts_rank(X,N))'
    import re
    result = re.sub(r'\b\d+\b', 'N', expression)
    result = re.sub(r'\b[a-z][a-z0-9_]*\b(?!\s*\()', 'X', result)
    return result

def _extract_operators(expression: str) -> list[str]:
    import re
    return re.findall(r'\b([a-z][a-z0-9_]*)\s*\(', expression)"
```

- [ ] **Step 4.4 — Run tests**

```bash
cd /Users/yugu/Downloads/QuantBrain_claud && PYTHONPATH=. pytest alpha_miner/tests/test_knowledge_distiller.py -v
```
Expected: all 3 PASS.

- [ ] **Step 4.5 — Commit**

```bash
cd /Users/yugu/Downloads/QuantBrain_claud && git add alpha_miner/modules/m9_knowledge_distiller.py alpha_miner/tests/test_knowledge_distiller.py && git commit -m "feat: add KnowledgeDistiller for post-run LLM-powered pattern extraction"
```

---

## Task 5: Wire Router + Distiller into main.py

**Files:**
- Modify: `alpha_miner/main.py`

- [ ] **Step 5.1 — Implement wiring**

```bash
codex "Modify /Users/yugu/Downloads/QuantBrain_claud/alpha_miner/main.py

Read the full file first.

Make these changes:

1. At the top, add imports:
   import os
   from .modules.llm_router import LLMRouter
   from .modules.m9_knowledge_distiller import KnowledgeDistiller

2. In main(), after loading config and before constructing agent, add:
   router = None
   if os.environ.get('LLM_ROUTER_ENABLED', 'false').lower() == 'true':
       router = LLMRouter(
           state_path=output_dir / 'llm_router_state.json',
           daily_budget_usd=float(os.environ.get('LLM_BUDGET_DAILY_USD', '3.60')),
       )

3. Update the HypothesisAgent constructor call to pass router=router and use_llm=(router is not None):
   agent = HypothesisAgent(
       kb=kb,
       cache=cache,
       taxonomy=taxonomy,
       model=os.environ.get('OPENAI_IDEA_MODEL', 'gpt-5.4-mini'),
       temperature=float(generation_cfg.get('temperature', 0.4)),
       top_p=float(generation_cfg.get('top_p', 0.9)),
       max_tokens=int(generation_cfg.get('max_tokens', 800)),
       seed=int(generation_cfg.get('seed', 42)),
       router=router,
       use_llm=(router is not None),
   )

4. After write_json(output_dir / 'summary.json', summary) and before print(), add post-run distillation:
   if os.environ.get('KNOWLEDGE_DISTILL_ENABLED', 'false').lower() == 'true' and router is not None:
       distiller = KnowledgeDistiller(kb=kb, router=router)
       pool_data = read_json(output_dir / 'pool.json', default={})
       distiller.distill(pool_data)
       append_jsonl(progress_path, {'stage': 'distillation_complete'})

No other changes."
```

- [ ] **Step 5.2 — Smoke test (generate mode, no LLM keys)**

```bash
cd /Users/yugu/Downloads/QuantBrain_claud && PYTHONPATH=. python3 -m alpha_miner.main \
  --mode generate \
  --objective 'test multi-llm wiring' \
  --batch-size 2 \
  --output-dir /tmp/quantbrain-wiring-test 2>&1 | tail -10
```
Expected: completes without error, `summary.json` written.

- [ ] **Step 5.3 — Commit**

```bash
cd /Users/yugu/Downloads/QuantBrain_claud && git add alpha_miner/main.py && git commit -m "feat: wire LLMRouter and KnowledgeDistiller into main.py pipeline"
```

---

## Task 6: Idea Optimizer

**Files:**
- Create: `alpha_miner/modules/idea_optimizer.py`
- Create: `alpha_miner/tests/test_idea_optimizer.py`

- [ ] **Step 6.1 — Write failing tests**

```bash
codex "In /Users/yugu/Downloads/QuantBrain_claud/alpha_miner/tests/test_idea_optimizer.py write pytest tests for IdeaOptimizer (to be created at alpha_miner/modules/idea_optimizer.py).

Tests:

1. test_optimize_returns_structured_direction
   - Patch the Anthropic call to return JSON:
     {'objective': 'supply chain stability alpha', 'category': 'QUALITY', 'hypothesis': 'firms with stable supplier relationships show lower earnings volatility', 'constraints': ['avoid raw turnover > 0.5', 'prefer industry-neutral signals'], 'suggested_data_fields': ['accounts_payable', 'cost_of_goods_sold']}
   - optimizer = IdeaOptimizer(router=mock_router)
   - result = optimizer.optimize('想挖供应链稳定性相关的因子')
   - assert result['category'] == 'QUALITY'
   - assert 'hypothesis' in result

2. test_optimize_fallback_on_llm_error
   - Patch the Anthropic call to raise an exception
   - result = optimizer.optimize('any idea')
   - assert result['objective'] == 'any idea'  # fallback: return raw idea as objective
   - assert 'category' in result  # must have a default

3. test_optimize_with_kb_context
   - kb has one failure_pattern (reason='LOW_SHARPE') and one strategy_stat (QUALITY, PASS)
   - optimizer = IdeaOptimizer(router=mock_router, kb=kb)
   - During optimize(), the prompt sent to the LLM must include 'LOW_SHARPE' (from failure_patterns)
   - Assert this by capturing the call args

Use tmp_path and unittest.mock.patch."
```

- [ ] **Step 6.2 — Verify failure**

```bash
cd /Users/yugu/Downloads/QuantBrain_claud && PYTHONPATH=. pytest alpha_miner/tests/test_idea_optimizer.py -v 2>&1 | head -20
```

- [ ] **Step 6.3 — Implement IdeaOptimizer**

```bash
codex "Create /Users/yugu/Downloads/QuantBrain_claud/alpha_miner/modules/idea_optimizer.py

Read these files:
- alpha_miner/modules/llm_router.py
- alpha_miner/modules/m1_knowledge_base.py
- alpha_miner/config/llm_providers.yaml (for model name)

Implement:

@dataclass
class OptimizedDirection:
    objective: str
    category: str
    hypothesis: str
    constraints: list[str]
    suggested_data_fields: list[str]
    raw_idea: str

class IdeaOptimizer:
    CATEGORIES = ['QUALITY', 'MOMENTUM', 'REVERSAL', 'LIQUIDITY', 'VOLATILITY', 'MICROSTRUCTURE', 'SENTIMENT']

    def __init__(self, router: LLMRouter | None = None, kb: KnowledgeBase | None = None):
        self.router = router
        self.kb = kb

    def optimize(self, raw_idea: str) -> dict:
        # Build context from KB if available
        failure_patterns = []
        strategy_stats = {}
        if self.kb:
            failure_patterns = self.kb.get_failure_patterns(limit=5)
            # get stats for all categories
            strategy_stats = {cat: self.kb.get_strategy_stats(cat) for cat in self.CATEGORIES}

        system_prompt = '''You are a WorldQuant BRAIN alpha research specialist.
Convert the user's natural language idea into a precise structured research direction.
Available categories: QUALITY, MOMENTUM, REVERSAL, LIQUIDITY, VOLATILITY, MICROSTRUCTURE, SENTIMENT
Available BRAIN data fields include: close, open, high, low, volume, returns, vwap, adv20,
  operating_income, assets, cashflow_op, revenue, accounts_payable, cost_of_goods_sold,
  net_income, debt, equity, capex, dividends, shares.

Return strict JSON:
{
  \"objective\": \"precise research objective\",
  \"category\": \"one of the categories above\",
  \"hypothesis\": \"economic hypothesis sentence\",
  \"constraints\": [\"list of expression constraints\"],
  \"suggested_data_fields\": [\"list of BRAIN field names\"]
}'''

        user_content = {
            'idea': raw_idea,
            'known_failure_patterns': [p['reason'] + ': ' + p.get('suggested_fix', '') for p in failure_patterns],
            'category_win_rates': {cat: stats.get('win_rate', 0.5) for cat, stats in strategy_stats.items() if stats},
        }

        if self.router is None:
            return {'objective': raw_idea, 'category': 'QUALITY', 'hypothesis': raw_idea, 'constraints': [], 'suggested_data_fields': []}

        provider = self.router.pick('idea')
        import time, os, json
        t0 = time.time()
        try:
            if provider.client_type == 'anthropic':
                from anthropic import Anthropic
                client = Anthropic(api_key=os.environ.get(provider.api_key_env, ''))
                response = client.messages.create(
                    model=provider.model_id,
                    max_tokens=600,
                    system=system_prompt,
                    messages=[{'role': 'user', 'content': json.dumps(user_content, ensure_ascii=False)}],
                )
                raw_text = response.content[0].text if response.content else '{}'
                tokens_in = response.usage.input_tokens if response.usage else 0
                tokens_out = response.usage.output_tokens if response.usage else 0
            else:
                from openai import OpenAI
                client = OpenAI(api_key=os.environ.get(provider.api_key_env, ''), base_url=provider.api_base)
                response = client.chat.completions.create(
                    model=provider.model_id, max_tokens=600,
                    messages=[{'role': 'system', 'content': system_prompt},
                              {'role': 'user', 'content': json.dumps(user_content)}],
                )
                raw_text = response.choices[0].message.content or '{}'
                tokens_in = response.usage.prompt_tokens if response.usage else 0
                tokens_out = response.usage.completion_tokens if response.usage else 0

            result = json.loads(raw_text)
            latency_ms = (time.time() - t0) * 1000
            self.router.record_result(provider.name, 'idea', True, latency_ms, tokens_in, tokens_out)
            return result
        except Exception:
            return {'objective': raw_idea, 'category': 'QUALITY', 'hypothesis': raw_idea, 'constraints': [], 'suggested_data_fields': []}"
```

- [ ] **Step 6.4 — Run tests**

```bash
cd /Users/yugu/Downloads/QuantBrain_claud && PYTHONPATH=. pytest alpha_miner/tests/test_idea_optimizer.py -v
```
Expected: all 3 PASS.

- [ ] **Step 6.5 — Commit**

```bash
cd /Users/yugu/Downloads/QuantBrain_claud && git add alpha_miner/modules/idea_optimizer.py alpha_miner/tests/test_idea_optimizer.py && git commit -m "feat: add IdeaOptimizer for natural language idea to structured research direction"
```

---

## Task 7: server.mjs — `/ideas/optimize` Endpoint + Router State

**Files:**
- Modify: `service/server.mjs`

- [ ] **Step 7.1 — Implement endpoint**

```bash
codex "Modify /Users/yugu/Downloads/QuantBrain_claud/service/server.mjs

Read the full file first. Study how existing POST endpoints like POST /runs and POST /ideas/analyze are implemented. Follow the exact same pattern.

Add a new endpoint: POST /ideas/optimize

Placement: add it near the existing POST /ideas/analyze handler.

Logic:
1. Require authentication (same as /ideas/analyze — check authContext)
2. Parse request body JSON: { idea: string }
3. Validate: idea must be a non-empty string, max 500 chars; return 400 if invalid
4. Spawn a child process running:
   PYTHONPATH=. python3 -m alpha_miner.modules.idea_optimizer_cli --idea '<idea>' --runs-dir <RUNS_DIR> --output-file <tmpfile>
   (We will create this CLI wrapper in a moment — for now just stub the endpoint to call the python process)
   
   Actually: use a simpler approach. Call the existing OPENAI API directly from Node for the idea optimizer, matching the existing /ideas/analyze pattern already in server.mjs.
   
   Look at how /ideas/analyze calls OpenAI in server.mjs and replicate the same pattern for /ideas/optimize, but use this system prompt:
   'You are a WorldQuant BRAIN alpha research specialist. Convert the user idea into a structured research direction. Return JSON with keys: objective, category (one of QUALITY/MOMENTUM/REVERSAL/LIQUIDITY/VOLATILITY/MICROSTRUCTURE/SENTIMENT), hypothesis, constraints (array), suggested_data_fields (array).'
   
   User message: the raw idea string.
   
5. On success, return 200 JSON: { ideaId: uuid, optimized: <parsed json from llm>, rawIdea: idea }
6. Save the result to IDEAS_DIR/<ideaId>.json for history
7. On LLM error or missing OPENAI_API_KEY, return 200 with fallback: { ideaId: uuid, optimized: { objective: idea, category: 'QUALITY', hypothesis: idea, constraints: [], suggested_data_fields: [] }, rawIdea: idea }

Also add GET /ideas/optimize/:ideaId that reads IDEAS_DIR/<ideaId>.json and returns it (return 404 if not found).

Do NOT modify any other existing endpoint."
```

- [ ] **Step 7.2 — Also expose LLM Router state in GET /runs response**

```bash
codex "Modify /Users/yugu/Downloads/QuantBrain_claud/service/server.mjs

Read the full file first. Find the GET /runs handler. 

Add llmRouterState to the response object:
- Try to read RUNS_DIR/llm_router_state.json
- If it exists, include it as llmRouterState in the GET /runs response JSON
- If it does not exist, include llmRouterState: null

This is a read-only addition. Do not modify any write paths."
```

- [ ] **Step 7.3 — Smoke test server starts**

```bash
cd /Users/yugu/Downloads/QuantBrain_claud && node --input-type=module <<'EOF'
import { readFileSync } from 'fs'
const src = readFileSync('service/server.mjs', 'utf8')
// Check endpoint exists
if (src.includes('/ideas/optimize')) console.log('✓ /ideas/optimize found')
else console.error('✗ endpoint missing')
EOF
```
Expected: `✓ /ideas/optimize found`

- [ ] **Step 7.4 — Commit**

```bash
cd /Users/yugu/Downloads/QuantBrain_claud && git add service/server.mjs && git commit -m "feat: add POST /ideas/optimize endpoint and expose LLM router state in GET /runs"
```

---

## Task 8: Dashboard Redesign

**Files:**
- Modify: `service/server.mjs` (replace `dashboardHtml()` function)

- [ ] **Step 8.1 — Replace dashboard HTML**

```bash
codex "Modify /Users/yugu/Downloads/QuantBrain_claud/service/server.mjs

Read the full file. Find the dashboardHtml() function — it returns a large HTML string.

Replace the entire body of dashboardHtml() with the HTML content from:
/Users/yugu/Downloads/QuantBrain_claud/.superpowers/brainstorm/65705-1776081758/content/dashboard-mockup-v3.html

Read that file first to get the exact HTML.

The dashboardHtml() function must return a string containing that HTML. Convert it to a template literal:
function dashboardHtml() {
  return \`<full html content here>\`
}

After replacing, add dynamic data injection: find the hardcoded placeholder values in the HTML and replace them with template literal expressions reading from the auto-loop state and scheduler state that the Node server already tracks.

Specifically replace:
- '24 alphas' big number → actual submitted count from auto-loop state (or 0 if unavailable)
- '38:22' countdown → actual next run countdown computed from schedulerState.nextRunAt
- The repair queue rows → actual repair queue items from autoLoopState.repairQueue (max 3 shown)
- The activity feed rows → actual recent events from autoLoopState.events (max 4 shown)
- The LLM router bars → actual data from llmRouterState if available, else show placeholder values

For the Research Idea textarea and Optimize & Run button: wire them to call POST /ideas/optimize via fetch() and show the result inline. Add a small <script> block at the bottom of the HTML for this interactivity (similar to how the existing dashboard already has inline JS).

Do not break any existing functionality in server.mjs."
```

- [ ] **Step 8.2 — Verify server starts with new dashboard**

```bash
cd /Users/yugu/Downloads/QuantBrain_claud && timeout 5 node service/server.mjs 2>&1 | head -5 || true
```
Expected: server starts (may show port binding message), no syntax errors.

- [ ] **Step 8.3 — Commit**

```bash
cd /Users/yugu/Downloads/QuantBrain_claud && git add service/server.mjs && git commit -m "feat: replace dashboard with Apple-style light UI showing LLM router state and live pipeline"
```

---

## Task 9: Environment Variables + Final Wiring

**Files:**
- Modify: `.env.example`
- Modify: `alpha_miner/requirements.txt`

- [ ] **Step 9.1 — Update .env.example**

```bash
codex "Append these lines to /Users/yugu/Downloads/QuantBrain_claud/.env.example (read the file first, add after the last line):

# Multi-LLM Router
DEEPSEEK_API_KEY=your-deepseek-api-key
GEMINI_API_KEY=your-gemini-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
LLM_ROUTER_ENABLED=true
LLM_BUDGET_DAILY_USD=3.60
KNOWLEDGE_DISTILL_ENABLED=true
IDEA_OPTIMIZER_MODEL=claude-sonnet-4-6"
```

- [ ] **Step 9.2 — Run full test suite**

```bash
cd /Users/yugu/Downloads/QuantBrain_claud && PYTHONPATH=. pytest alpha_miner/tests/ -v 2>&1 | tail -30
```
Expected: all tests PASS (or skip if missing API keys).

- [ ] **Step 9.3 — Final smoke test**

```bash
cd /Users/yugu/Downloads/QuantBrain_claud && PYTHONPATH=. python3 -m alpha_miner.main \
  --mode generate \
  --objective 'supply chain stability alphas' \
  --batch-size 3 \
  --output-dir /tmp/quantbrain-final-smoke 2>&1 | tail -15
```
Expected: `summary.json` written, no errors.

- [ ] **Step 9.4 — Final commit**

```bash
cd /Users/yugu/Downloads/QuantBrain_claud && git add .env.example alpha_miner/requirements.txt && git commit -m "chore: add multi-LLM env vars to .env.example"
```

---

## Self-Review Checklist

| Spec Requirement | Covered in Task |
|---|---|
| LLM Router with ε-greedy, budget tracking | Task 1 |
| KnowledgeBase schema extension (4 new tables) | Task 2 |
| `_call_llm()` enabled, parallel generation, Judge | Task 3 |
| Post-run KnowledgeDistiller | Task 4 |
| Router + Distiller wired into main.py | Task 5 |
| Idea Optimizer (Claude Sonnet) | Task 6 |
| `POST /ideas/optimize` endpoint | Task 7 |
| LLM Router state in Dashboard | Task 7 |
| Apple-style Dashboard redesign | Task 8 |
| `.env.example` updated | Task 9 |
| Feature flags (`LLM_ROUTER_ENABLED`, `KNOWLEDGE_DISTILL_ENABLED`) | Task 5, 9 |
| Budget hard cap (€100/month) | Task 1 (daily_budget_usd) |
| Deepseek V3, Gemini Flash, Claude Haiku, GPT-4o-mini wired | Tasks 1, 3, 4, 6 |
| Existing BRAIN layer untouched | ✓ m4 not in any task |
| Submission gate untouched | ✓ server.mjs gate logic not touched |
