# Multi-LLM Agentic Alpha Mining — Design Spec

**Date:** 2026-04-13  
**Status:** Approved  
**Priority:** Effect & efficiency first — no constraint on code minimality

---

## 1. Goal

Transform QuantBrain from a single-LLM templated system into a self-evolving multi-LLM alpha mining platform. The system runs fully automatically without human intervention, and also accepts optional user-provided research directions that an AI Idea Optimizer refines before execution. All results feed a shared knowledge base that continuously improves generation quality, repair precision, and model routing decisions.

---

## 2. Budget

| Constraint | Value |
|---|---|
| Monthly hard limit | ≤ €100/month |
| Expected actual spend | ~€12/month (~$0.44/day) |
| Daily budget cap (env var) | `LLM_BUDGET_DAILY_USD=3.60` |
| Headroom for premium calls | ~€88/month available for escalation |

---

## 3. Two Operating Modes

### 3.1 Fully Automatic Mode
- Scheduler fires every 60 minutes (configurable)
- The evolution layer (Knowledge + Strategy) autonomously selects the next research direction based on historical performance
- No human input required at any stage
- System self-improves across runs indefinitely

### 3.2 User-Guided Mode
- User inputs a natural language idea via Dashboard ("想挖供应链稳定性相关的因子")
- **Idea Optimizer** (Claude Sonnet) reads the idea + current knowledge base + available BRAIN fields and outputs a structured research direction with category, constraints, and hypothesis
- The structured direction enters the same Multi-LLM pipeline as automatic mode
- Results feed back into the shared knowledge base — both modes evolve together

---

## 4. Multi-LLM Architecture

### 4.1 LLM Role Assignments

| Role | Model | Rationale | Est. cost |
|---|---|---|---|
| Primary generation | Deepseek V3 | Strong formula/code generation, very low cost | ~$0.15/day |
| Diversity challenger | Gemini Flash | Wide associative range, different reasoning style | ~$0.05/day |
| Judge (select best) | GPT-4o-mini | Stable structured output, reliable scoring | ~$0.04/day |
| Standard repair | Claude Haiku | Precise instruction following, ideal for targeted fixes | ~$0.10/day |
| Complex repair escalation | GPT-4o-mini | Fallback when Haiku fails after 2 rounds | ~$0.05/day |
| Knowledge distillation | GPT-4o-mini | Low cost, good at structured extraction | ~$0.05/day |
| Idea optimizer | Claude Sonnet | Strong reasoning, used only on user trigger | ~$0.00/day avg |

### 4.2 Generation Flow (Parallel)
```
Objective + category + RAG context
  ├── Deepseek V3 → 3 candidates (primary objective, exploit known patterns)
  └── Gemini Flash → 3 candidates (same objective, explore alternative operators/data families)
        ↓ parallel
  GPT-4o-mini Judge → scores each candidate on:
      1. Expression validity (no banned operators, correct syntax)
      2. Hypothesis-expression alignment (does expression match stated hypothesis)
      3. Pairwise diversity (penalise candidates with similar operator skeletons)
      → outputs ranked list, top N proceed
        ↓
  ExpressionValidator (existing m3)
        ↓
  BRAIN Simulation (existing m4)
```

### 4.3 Repair Flow (Escalating)
```
Gate failure
  → Claude Haiku (rounds 1-3, targeted fix based on failure reason)
  → GPT-4o-mini (rounds 4-5, if Haiku still failing)
  → Budget exhausted → mark as deprecated
```

### 4.4 LLM Router — ε-greedy
- Tracks per-LLM × per-role: gate pass rate, avg testSharpe, latency, cost
- Routing weight updated every 10 rounds
- 90% exploits current best performer, 10% explores other models
- Hard fallback: if daily budget > 80% used, downgrade all calls to cheapest model
- State persisted in `llm_router_state.json`

---

## 5. Three-Layer Evolution

### 5.1 Strategy Evolution
**What it tracks:** category win rates, data family gate results, operator skeleton success rates, hypothesis→expression trajectories

**Actions:**
- Increase generation weight for high-performing categories
- Apply cooldown to failing data families (extends existing `ALPHA_FAMILY_COOLDOWN_ROUNDS`)
- Block high-frequency failing operator skeletons
- Dynamically adjust batch composition ratios

**Storage:** `strategy_weights.json` + new `strategy_stats` table in KnowledgeBase

### 5.2 Model Evolution
**What it tracks:** per-LLM gate pass rate, testSharpe distribution, response latency, cost per successful alpha

**Actions:** ε-greedy weight rebalancing (see 4.4)

**Storage:** `llm_router_state.json`

### 5.3 Knowledge Evolution
**Trigger:** after every completed run (generate, evaluate, or loop)

**Process:** GPT-4o-mini distiller reads batch results and extracts:
- Successful alpha expressions → `positive_examples` table
- Failure reason patterns → `failure_patterns` table  
- Effective operator combinations → `operator_stats` table
- Current market regime signals → `market_regime` table

**Impact on generation:** RAG context injected into every generation prompt includes:
- Recent N successful examples
- Current known failure patterns to avoid
- High-performing operator combinations for this category
- Market regime summary

---

## 6. New & Modified Components

### New Files
| File | Purpose |
|---|---|
| `alpha_miner/modules/llm_router.py` | LLM provider management, ε-greedy routing, budget tracking, state persistence |
| `alpha_miner/modules/m9_knowledge_distiller.py` | Post-run distillation via GPT-4o-mini, writes to KnowledgeBase |
| `alpha_miner/modules/idea_optimizer.py` | User idea → structured research direction via Claude Sonnet |
| `alpha_miner/config/llm_providers.yaml` | API keys config, model IDs, cost-per-token for budget tracking |

### Modified Files
| File | Changes |
|---|---|
| `alpha_miner/modules/m2_hypothesis_agent.py` | Implement `_call_llm()` with multi-LLM routing; inject RAG context from KnowledgeBase into prompts |
| `alpha_miner/modules/m1_knowledge_base.py` | Add 4 new tables: `strategy_stats`, `failure_patterns`, `operator_stats`, `market_regime`; add write/query interfaces |
| `alpha_miner/main.py` | Integrate LLM Router and post-run distillation call; support parallel generation |
| `service/server.mjs` | Add `POST /ideas/optimize` endpoint; add LLM Router state panel to Dashboard; serve new Dashboard HTML |

### New Environment Variables
```
DEEPSEEK_API_KEY=
GEMINI_API_KEY=
ANTHROPIC_API_KEY=
LLM_ROUTER_ENABLED=true
LLM_BUDGET_DAILY_USD=3.60
KNOWLEDGE_DISTILL_ENABLED=true
IDEA_OPTIMIZER_MODEL=claude-sonnet-4-5
```

---

## 7. Dashboard Redesign

**Style:** Apple-inspired light mode — white cards, frosted glass nav, system font, generous whitespace. Colors limited to functional use: green (success), orange (warning), blue (system).

**Layout:** Three-column card grid

**Card 1 — Metrics + Pipeline**
- Hero stat: Alphas Submitted Today
- Mini stats: Gate Pass Rate, Repair Success, Simulations, Daily Spend
- Pipeline flow: Gen → Valid → Sim → Gate → Sub with live counts

**Card 2 — LLM Router + Repair**
- Per-LLM weight bar + gate pass rate (live updating)
- Repair queue: expression, failure reason tag, repair depth

**Card 3 — Control + Activity**
- Countdown to next run
- Research Idea input + "Optimize & Run" button
- Recent activity feed (submitted / repair / failed)

---

## 8. What Is NOT Changed
- BRAIN API interaction layer (`m4_brain_backtester.py`) — stable, untouched
- Legacy JS Runner (`agentic_alpha_lab.mjs`, `agentic_alpha_library.mjs`) — kept as fallback engine
- Expression Validator (`m3_validator.py`) — untouched
- Submission gate logic — untouched, all safety rules preserved
- Auto-repair budget caps (5 rounds × 5 candidates) — preserved

---

## 9. Rollout Safety

- `LLM_ROUTER_ENABLED=false` → system behaves exactly as today
- `KNOWLEDGE_DISTILL_ENABLED=false` → distillation skipped, no side effects
- New Dashboard served only when new `server.mjs` is deployed; old behavior on rollback
- All new state files (`llm_router_state.json`, `strategy_weights.json`) are optional at startup — system initializes with uniform defaults if missing
