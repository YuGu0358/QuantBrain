from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

from alpha_miner.modules.llm_router import LLMRouter
from alpha_miner.modules.m9_knowledge_distiller import KnowledgeDistiller
from alpha_miner.modules.m_diagnoser import Diagnoser
from alpha_miner.modules.m_distiller import Distiller
from alpha_miner.modules.m_repair_memory import RepairMemory
from alpha_miner.modules.m_retriever import Retriever
from alpha_miner.modules.m_planner import Planner
from alpha_miner.modules.m_scheduler import BanditScheduler
from alpha_miner.modules.m_repair_chain import DEFAULT_REPAIR_CHAIN_MODEL, RepairChain

from .modules.common import PACKAGE_ROOT, append_jsonl, read_json, set_seed, write_json
from .modules.config_loader import load_config, load_taxonomy
from .modules.llm_cache import LLMCache
from .modules.m1_knowledge_base import KnowledgeBase
from .modules.m2_hypothesis_agent import HypothesisAgent
from .modules.m3_validator import ExpressionValidator
from .modules.m4_brain_backtester import BrainBacktester, QuotaWaiting
from .modules.m6_alpha_pool import AlphaPool, load_pnl_series
from .modules.m7_stat_significance import StatSignificance
from .modules.m8_portfolio_optimizer import PortfolioOptimizer
from .modules.m_quality_guardrails import economic_logic_prescreen, should_try_sign_flip, sign_flip_expression


DEFAULT_GENERATION_MODEL = "gpt-5.4-2026-03-05"
DEFAULT_REPAIR_MODEL = DEFAULT_REPAIR_CHAIN_MODEL
DEFAULT_REPAIR_ROUTER_PROVIDER_NAME = "gpt_repair"


def _ts() -> str:
    """Return current UTC time as ISO-8601 string for progress.jsonl 'at' field."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _primary_llm_api_key(model_id: str, openai_key: str) -> str:
    if model_id.startswith(("claude",)):
        return os.environ.get("ANTHROPIC_API_KEY", "")
    return openai_key


def _diagnosis_summary(diagnosis: Any) -> dict[str, Any] | None:
    if diagnosis is None:
        return None
    raw = diagnosis.raw if isinstance(getattr(diagnosis, "raw", None), dict) else {}
    return {
        "primary_symptom": diagnosis.primary_symptom,
        "secondary_symptoms": diagnosis.secondary_symptoms,
        "fallback": bool(raw.get("fallback")),
        "error": raw.get("error"),
    }


def main() -> None:
    args = parse_args()
    started = time.time()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for dirname in ["backtest_snapshots", "pnl_series", "llm_cache"]:
        (output_dir / dirname).mkdir(parents=True, exist_ok=True)

    config = load_config(Path(args.config).resolve() if args.config else None)
    generation_cfg = config.get("generation", {})
    set_seed(int(generation_cfg.get("seed", 42)))
    progress_path = output_dir / "progress.jsonl"
    append_jsonl(progress_path, {"at": _ts(), "stage": "started", "mode": args.mode, "objective": args.objective, "engine": "python-v2", "repairContext": args.repair_context})

    # KB and AlphaPool use shared paths under RUNS_DIR so knowledge accumulates
    # across runs. Per-run output_dir is only for ephemeral artefacts (snapshots, logs).
    shared_dir = output_dir.parent
    kb_embedder = None
    _openai_key = os.environ.get("OPENAI_API_KEY", "")
    if _openai_key:
        try:
            from langchain_openai import OpenAIEmbeddings
            kb_embedder = OpenAIEmbeddings(api_key=_openai_key, model="text-embedding-3-small")
            print("[kb] embedder initialized (text-embedding-3-small)", flush=True)
        except Exception as _emb_exc:
            print(f"[kb] embedder unavailable, falling back to SQL search: {_emb_exc}", flush=True)
    kb = KnowledgeBase(shared_dir / "knowledge_base.db", embedder=kb_embedder)
    kb.import_wq101_negative_examples(PACKAGE_ROOT / "seeds" / "wq101_alphas.json")
    _import_submitted_feedback(kb, shared_dir / "submitted_alphas.jsonl")
    cache = LLMCache(output_dir / "llm_cache")
    taxonomy = load_taxonomy()
    validator = ExpressionValidator()
    router = _initialize_router(output_dir)
    agent = HypothesisAgent(
        kb=kb,
        cache=cache,
        taxonomy=taxonomy,
        model=DEFAULT_GENERATION_MODEL,
        temperature=float(generation_cfg.get("temperature", 0.4)),
        top_p=float(generation_cfg.get("top_p", 0.9)),
        max_tokens=int(generation_cfg.get("max_tokens", 2000)),
        seed=int(generation_cfg.get("seed", 42)),
        router=router,
        use_llm=(router is not None),
        validator=validator,
    )
    stat = StatSignificance(
        dsr_threshold=float(config.get("stat", {}).get("dsr_threshold", 0.95)),
        pbo_threshold=float(config.get("stat", {}).get("pbo_threshold", 0.30)),
    )
    alpha_pool = AlphaPool(shared_dir / "alpha_pool.db", threshold=float(config.get("pool", {}).get("orthogonality_threshold", 0.5)))
    backtester = BrainBacktester(output_dir, config)
    optimizer = PortfolioOptimizer(
        max_weight=float(config.get("pool", {}).get("max_weight", 0.15)),
        mcr_cap=float(config.get("pool", {}).get("mcr_cap", 0.25)),
    )

    batch_size = int(args.batch_size or generation_cfg.get("batch_size", 10))
    category = args.category or agent.sample_underweight_category({})
    sim_settings: dict[str, Any] = {}
    if args.sim_settings:
        try:
            sim_settings = json.loads(args.sim_settings)
        except Exception:
            pass

    # If repair context is provided, pass it as structured data to the generation agent
    effective_objective = args.objective
    repair_ctx: dict | None = None
    if args.repair_context:
        try:
            repair_ctx = read_json(Path(args.repair_context))
            if repair_ctx:
                if repair_ctx.get("_category"):
                    category = repair_ctx["_category"]
                parent_expr = repair_ctx.get("expression") or ""
                failed = repair_ctx.get("failedChecks") or []
                # Keep a short human-readable note in the objective for logging
                if parent_expr:
                    effective_objective = f"Repair alpha with failed checks {failed}: {parent_expr[:80]}"
        except Exception:
            repair_ctx = None

    repair_memory: RepairMemory | None = None
    if repair_ctx is not None:
        repair_memory_path = output_dir.parent / "repair_memory.db"
        repair_memory = RepairMemory(repair_memory_path)
        agent.repair_memory = repair_memory
        # LangChain repair chain (primary path) — GPT by default, Claude still supported via override.
        repair_model = os.environ.get("REPAIR_MODEL", DEFAULT_REPAIR_MODEL)
        agent.repair_chain = RepairChain(
            memory=repair_memory,
            api_key=_primary_llm_api_key(repair_model, _openai_key),
            model_id=repair_model,
            openai_api_key=_openai_key,
        )
        # Legacy fallback (kept for compatibility)
        agent.retriever = Retriever(memory=repair_memory, router=router)
        scheduler = BanditScheduler(output_dir.parent / "repair_scheduler.db")
        agent.planner = Planner(router=router)
        agent.scheduler = scheduler

    diagnosis = None
    if repair_ctx is not None and router is not None:
        diagnoser = Diagnoser(router)
        metrics = {
            "sharpe": repair_ctx.get("metrics", {}).get("isSharpe") or repair_ctx.get("metrics", {}).get("sharpe"),
            "fitness": repair_ctx.get("metrics", {}).get("isFitness") or repair_ctx.get("metrics", {}).get("fitness"),
            "turnover": repair_ctx.get("metrics", {}).get("turnover"),
            "max_abs_correlation": repair_ctx.get("metrics", {}).get("max_abs_correlation"),
        }
        diagnosis = diagnoser.diagnose(
            expression=repair_ctx.get("expression", ""),
            metrics=metrics,
            failed_checks=repair_ctx.get("failedChecks") or [],
            gate_reasons=(repair_ctx.get("gate") or {}).get("reasons") or [],
        )
        append_jsonl(
            progress_path,
            {
                "stage": "diagnosis",
                "primary_symptom": diagnosis.primary_symptom,
                "secondary_symptoms": diagnosis.secondary_symptoms,
                "repair_priorities": diagnosis.repair_priorities,
                "fallback": bool((diagnosis.raw or {}).get("fallback")) if isinstance(diagnosis.raw, dict) else False,
                "error": (diagnosis.raw or {}).get("error") if isinstance(diagnosis.raw, dict) else None,
            },
        )

    _t_gen_start = time.time()
    candidates = agent.generate_batch(
        effective_objective,
        category=category,
        n=batch_size,
        use_llm=args.use_llm,
        repair_context=repair_ctx,
        diagnosis=diagnosis,
    )
    _gen_latency_ms = (time.time() - _t_gen_start) * 1000
    # Record repair-chain usage in LLM router using real token usage metadata.
    if repair_ctx is not None and router is not None and candidates:
        try:
            _usage = getattr(getattr(agent, "repair_chain", None), "last_usage", {}) or {}
            _repair_ti = int(_usage.get("input_tokens") or 0)
            _repair_to = int(_usage.get("output_tokens") or 0)
            _repair_calls = int(_usage.get("calls") or 0)
            if _repair_calls > 0 or (_repair_ti + _repair_to) > 0:
                _repair_latency_ms = float(_usage.get("latency_ms") or _gen_latency_ms)
                _provider_name = _resolve_router_provider_name(
                    router,
                    "repair",
                    preferred_name=os.environ.get(
                        "REPAIR_ROUTER_PROVIDER_NAME",
                        DEFAULT_REPAIR_ROUTER_PROVIDER_NAME,
                    ),
                )
                if _provider_name:
                    router.record_result(
                        _provider_name,
                        "repair",
                        len(candidates) > 0,
                        _repair_latency_ms,
                        _repair_ti,
                        _repair_to,
                    )
        except Exception:
            pass

    validator_records = []
    valid_candidates = []
    rejected_by_stage = {
        "validator": 0,
        "economic_prescreen": 0,
        "dsr": 0,
        "pbo": 0,
        "oos": 0,
        "orthogonality": 0,
        "no_pnl": 0,
    }
    for candidate in candidates:
        validation = validator.validate(candidate.expression)
        prescreen = economic_logic_prescreen(candidate.expression)
        validator_records.append(
            {
                "candidate": asdict(candidate),
                "validation": asdict(validation),
                "economicPrescreen": asdict(prescreen),
            }
        )
        if validation.is_valid and prescreen.is_valid:
            valid_candidates.append(candidate)
        else:
            if not validation.is_valid:
                rejected_by_stage["validator"] += 1
                print(
                    f"[validator] rejected candidate_id={candidate.id} expression={candidate.expression[:120]!r} errors={validation.errors}",
                    flush=True,
                )
                append_jsonl(
                    progress_path,
                    {
                        "at": _ts(),
                        "stage": "rejected",
                        "reason": "validator",
                        "candidate_id": candidate.id,
                        "errors": validation.errors,
                    },
                )
            else:
                rejected_by_stage["economic_prescreen"] += 1
                print(
                    f"[economic_prescreen] rejected candidate_id={candidate.id} expression={candidate.expression[:120]!r} reasons={prescreen.reasons}",
                    flush=True,
                )
                append_jsonl(
                    progress_path,
                    {
                        "at": _ts(),
                        "stage": "rejected",
                        "reason": "economic_prescreen",
                        "candidate_id": candidate.id,
                        "errors": prescreen.reasons,
                        "warnings": prescreen.warnings,
                    },
                )

    round_payload = {
        "round": 1,
        "mode": args.mode,
        "category": category,
        "objective": args.objective,
        "candidates": validator_records,
    }
    write_json(output_dir / "batch-round-1.json", round_payload)

    evaluated_records: list[dict[str, Any]] = []
    portfolio_pool: list[dict[str, Any]] = []
    phase0_status = phase0_mode(args.mode)
    blocked_reason = phase0_status if phase0_status in {"phase0_brain_probe_report_missing", "phase0_brain_probe_not_actionable"} else None
    degraded_mode = phase0_status == "regular_tier_degraded_no_daily_pnl"
    companion_sign_flip_count = 0
    companion_sign_flip_promotions = 0
    if blocked_reason:
        append_jsonl(progress_path, {"at": _ts(), "stage": "blocked", "reason": blocked_reason})
    elif args.mode in {"evaluate", "loop"}:
        # --- parallel BRAIN submission -----------------------------------
        # All candidates are submitted concurrently; the thread-safe RateLimiter
        # enforces the 8/min ceiling. Results are collected in original order and
        # then processed sequentially so shared state (alpha_pool, etc.) stays safe.
        _quota_hit: bool = False

        def _submit_one(_idx_cand: tuple) -> tuple:
            _i, _cand = _idx_cand
            try:
                _res = backtester.submit_alpha(_cand.expression, period="IS", settings=sim_settings or None)
                return _i, _cand, _res, None
            except QuotaWaiting as _e:
                return _i, _cand, None, _e

        _n_workers = min(len(valid_candidates), 8)
        _backtest_pairs: list = [None] * len(valid_candidates)
        if valid_candidates:
            with ThreadPoolExecutor(max_workers=_n_workers) as _pool:
                _futs = [_pool.submit(_submit_one, (i, c)) for i, c in enumerate(valid_candidates)]
                for _fut in as_completed(_futs):
                    _i, _cand, _res, _err = _fut.result()
                    if _err is not None:
                        if not _quota_hit:
                            append_jsonl(progress_path, {"at": _ts(), "stage": "waiting", "reason": str(_err)})
                        _quota_hit = True
                    else:
                        _backtest_pairs[_i] = (_cand, _res)

        # --- sequential result processing --------------------------------
        for _pair in _backtest_pairs:
            if _pair is None:
                continue  # slot was quota-hit or not submitted
            candidate, result = _pair
            candidate, result, companion = _maybe_promote_sign_flip_companion(
                backtester,
                validator,
                candidate,
                result,
                sim_settings,
            )
            record = {"candidate": asdict(candidate), "backtest": asdict(result)}
            if companion is not None:
                companion_sign_flip_count += 1
                companion_sign_flip_promotions += 1 if companion.get("promoted") else 0
                record["companion"] = companion
                append_jsonl(
                    progress_path,
                    {
                        "at": _ts(),
                        "stage": "sign_flip_companion",
                        "candidate_id": candidate.id,
                        "status": companion.get("status"),
                        "promoted": companion.get("promoted", False),
                        "original_sharpe": companion.get("originalSharpe"),
                        "flipped_sharpe": companion.get("flippedSharpe"),
                    },
                )
            if result.pnl_path:
                pnl = load_pnl_series(Path(result.pnl_path))
                dsr = stat.deflated_sharpe(result.sharpe or 0.0, [result.sharpe or 0.0], max(len(pnl), 2))
                orthogonality = alpha_pool.check_orthogonality(pnl)
                record["dsr"] = dsr
                record["orthogonality"] = asdict(orthogonality)
                if dsr < stat.dsr_threshold:
                    rejected_by_stage["dsr"] += 1
                elif not orthogonality.passed:
                    rejected_by_stage["orthogonality"] += 1
                else:
                    metadata_json = json.dumps({"metrics": asdict(result), "candidate": asdict(candidate)}, ensure_ascii=False, sort_keys=True)
                    alpha_pool.add_alpha(result.alpha_id or candidate.id, candidate.expression, candidate.category, Path(result.pnl_path), holdout_used=False, metadata_json=metadata_json)
                    portfolio_pool.append({"alpha_id": result.alpha_id or candidate.id, "pnl": pnl})
            else:
                rejected_by_stage["no_pnl"] += 1
                record["status"] = "no_daily_pnl"
                record["dailyPnlPolicy"] = "blocked_for_dsr_mvo"
                record["expressionProxyOrthogonality"] = asdict(
                    alpha_pool.check_expression_similarity(
                        candidate.expression,
                        threshold=float(config.get("pool", {}).get("expression_similarity_threshold", 0.80)),
                    )
                )
                brain_check_passed: bool | None = None
                if degraded_mode and result.alpha_id:
                    try:
                        brain_check = backtester.check_alpha(result.alpha_id)
                        record["brainCheckOrthogonality"] = asdict(brain_check)
                        brain_check_passed = brain_check.passed
                        if brain_check.passed is False:
                            rejected_by_stage["orthogonality"] += 1
                    except QuotaWaiting as error:
                        append_jsonl(progress_path, {"at": _ts(), "stage": "waiting", "reason": str(error), "alpha_id": result.alpha_id})
                        record["brainCheckOrthogonality"] = {"status": "waiting", "reason": str(error)}
                c_score, c_qualified = case_c_robust_score(
                    sharpe=result.sharpe,
                    fitness=result.fitness,
                    turnover=result.turnover,
                    test_sharpe=result.test_sharpe,
                    brain_check_passed=brain_check_passed,
                )
                if degraded_mode and result.alpha_id and c_qualified:
                    rejected_by_stage["no_pnl"] -= 1  # undo the initial increment
                    metadata_json = json.dumps({"metrics": asdict(result), "candidate": asdict(candidate)}, ensure_ascii=False, sort_keys=True)
                    dummy_pnl_path = Path(backtester.snapshot_dir / f"degraded_pnl_{result.alpha_id}.json")
                    import alpha_miner.modules.common as _common
                    _common.write_json(dummy_pnl_path, {"pnl": [], "source": "degraded_no_daily_pnl", "sharpe": result.sharpe})
                    alpha_pool.add_alpha(result.alpha_id, candidate.expression, candidate.category, dummy_pnl_path, holdout_used=False, metadata_json=metadata_json)
                    portfolio_pool.append({"alpha_id": result.alpha_id, "pnl": []})
                    record["degradedQualified"] = True
                append_jsonl(
                    progress_path,
                    {
                        "at": _ts(),
                        "stage": "degraded_evaluation",
                        "candidate_id": candidate.id,
                        "alpha_id": result.alpha_id,
                        "sharpe": result.sharpe,
                        "fitness": result.fitness,
                        "turnover": result.turnover,
                        "test_sharpe": result.test_sharpe,
                        "case_c_score": c_score,
                        "degraded_qualified": degraded_mode and result.alpha_id is not None and c_qualified,
                        "reason": "regular tier exposes aggregate metrics but no daily pnl",
                    },
                )
            evaluated_records.append(record)
            _accepted = (
                (record.get("dsr") is not None and record.get("orthogonality", {}).get("passed", False))
                or record.get("degradedQualified", False)
            )
            try:
                _kb_write_back(kb, category, candidate, result, _accepted)
            except Exception as _kb_exc:
                print(f"[kb] write-back failed: {_kb_exc}", flush=True)
            append_jsonl(progress_path, {"at": _ts(), "stage": "evaluated", "candidate_id": candidate.id, "status": result.status})

    if repair_ctx is not None and repair_memory is not None:
        tried = [
            {
                "expression": r["candidate"]["expression"],
                "sharpe": r.get("backtest", {}).get("sharpe"),
                "fitness": r.get("backtest", {}).get("fitness"),
                "turnover": r.get("backtest", {}).get("turnover"),
                "accepted": r.get("dsr") is not None and r.get("orthogonality", {}).get("passed", False),
            }
            for r in evaluated_records
        ]
        accepted_expr = next((item["expression"] for item in tried if item["accepted"]), None)
        # Feed repair outcomes back into LangChain RepairChain memory (closes the FAISS learning loop)
        if agent.repair_chain is not None:
            _rc_diag = {}
            if diagnosis is not None:
                _rc_diag = {
                    "primary_symptom": diagnosis.primary_symptom,
                    "secondary_symptoms": diagnosis.secondary_symptoms,
                    "root_causes": diagnosis.root_causes,
                    "repair_priorities": diagnosis.repair_priorities,
                    "do_not_change": diagnosis.do_not_change,
                }
                if isinstance(diagnosis.raw, dict):
                    _rc_diag.update(diagnosis.raw)
            for r in evaluated_records:
                _cand = r.get("candidate", {})
                _accepted = bool(
                    r.get("degradedQualified")
                    or (r.get("dsr") is not None and r.get("orthogonality", {}).get("passed", False))
                )
                _status = str(r.get("status") or "")
                agent.repair_chain.record_outcome(
                    expression=repair_ctx.get("expression", ""),
                    diagnosis=_rc_diag if isinstance(_rc_diag, dict) else {},
                    candidates=[_cand] if _cand else [],
                    accepted=_accepted,
                    candidate_metrics=r.get("backtest", {}) or {},
                    gate=repair_ctx.get("gate") if isinstance(repair_ctx.get("gate"), dict) else {},
                    platform_outcome={
                        "outcome": "accepted" if _accepted else ("degraded_no_daily_pnl" if _status == "no_daily_pnl" else _status),
                        "status": _status,
                        "alpha_id": r.get("backtest", {}).get("alpha_id"),
                        "degradedQualified": r.get("degradedQualified", False),
                        "repairDepth": repair_ctx.get("repairDepth", 0),
                    },
                    category=_cand.get("category") or repair_ctx.get("_category"),
                )

    if repair_ctx is not None and diagnosis is not None and repair_memory is not None:
        tried = [
            {
                "expression": r["candidate"]["expression"],
                "sharpe": r.get("backtest", {}).get("sharpe"),
                "fitness": r.get("backtest", {}).get("fitness"),
                "turnover": r.get("backtest", {}).get("turnover"),
                "accepted": r.get("dsr") is not None and r.get("orthogonality", {}).get("passed", False),
            }
            for r in evaluated_records
        ]
        accepted_expr = next((item["expression"] for item in tried if item["accepted"]), None)

        if router is not None:
            # Distill cadence (default every run). Sampling only affects memory learning,
            # never quality gates.
            _repair_distill_every_n = max(1, int(os.environ.get("REPAIR_DISTILL_EVERY_N", "1")))
            if _should_run_cadence(shared_dir, "repair_distill", _repair_distill_every_n):
                distiller = Distiller(router, repair_memory)
                distiller.distill(
                    original_expression=repair_ctx.get("expression", ""),
                    diagnosis=diagnosis,
                    tried_candidates=tried,
                    accepted_expression=accepted_expr,
                )
                append_jsonl(progress_path, {"at": _ts(), "stage": "distilled", "accepted": accepted_expr is not None})
            else:
                append_jsonl(
                    progress_path,
                    {
                        "at": _ts(),
                        "stage": "distill_skipped",
                        "reason": "repair_distill_sampling",
                        "every_n": _repair_distill_every_n,
                    },
                )

        # Bandit scheduler: record action outcomes
        if scheduler is not None:
            metrics_orig = repair_ctx.get("metrics", {})
            s_old = float(metrics_orig.get("isSharpe") or metrics_orig.get("sharpe") or 0)
            f_old = float(metrics_orig.get("isFitness") or metrics_orig.get("fitness") or 0)
            t_old = float(metrics_orig.get("turnover") or 0)
            j_old = s_old * f_old - 0.5 * max(0.0, t_old - 0.7)
            sched_outcomes = []
            for r in evaluated_records:
                bt = r.get("backtest", {})
                action_type = next(
                    (ref for ref in (r.get("candidate", {}).get("origin_refs") or [])
                     if ref in {"param_tune", "struct_mutation", "template_retrieval", "llm_mutation"}),
                    "llm_mutation",
                )
                j_new = (float(bt.get("sharpe") or 0) * float(bt.get("fitness") or 0)
                         - 0.5 * max(0.0, float(bt.get("turnover") or 0) - 0.7))
                accepted_r = r.get("dsr") is not None and r.get("orthogonality", {}).get("passed", False)
                sched_outcomes.append({"action_type": action_type, "accepted": accepted_r, "j_old": j_old, "j_new": j_new})
            scheduler.record_batch_outcomes(sched_outcomes)
            append_jsonl(progress_path, {"at": _ts(), "stage": "scheduler_updated", "weights": scheduler.get_weights()})

    portfolio = optimizer.optimize(portfolio_pool)
    write_json(output_dir / "portfolio.json", asdict(portfolio))
    pool_payload = {
        "qualified": len(portfolio_pool),
        "blockedReason": blocked_reason,
        "phase0Status": phase0_status,
        "degradedMode": degraded_mode,
        "records": evaluated_records,
    }
    write_json(output_dir / "pool.json", pool_payload)
    memory = {"effectivePool": portfolio_pool, "deprecatedPool": [], "trajectories": validator_records, "objectiveHistory": [{"objective": args.objective}]}
    write_json(output_dir / "memory.json", memory)

    _llm_stage_metrics = _router_stage_metrics(router)
    summary = {
        "engine": "python-v2",
        "mode": args.mode,
        "objective": args.objective,
        "category": category,
        "diagnosis": _diagnosis_summary(diagnosis),
        "phase0Status": phase0_status,
        "degradedMode": degraded_mode,
        "generatedCandidates": len(candidates),
        "validCandidates": len(valid_candidates),
        "qualified_alphas_count": len(portfolio_pool),
        "companion_sign_flip_count": companion_sign_flip_count,
        "companion_sign_flip_promotions": companion_sign_flip_promotions,
        "degraded_candidates_count": sum(1 for record in evaluated_records if record.get("status") == "no_daily_pnl"),
        "total_llm_tokens": _router_token_totals(router),
        "total_llm_cost_usd": round(router.spent_usd, 6) if router is not None else 0.0,
        "llm_stage_tokens": _llm_stage_metrics["tokens"],
        "llm_stage_calls": _llm_stage_metrics["calls"],
        "llm_stage_cost_usd": _llm_stage_metrics["cost_usd"],
        "llm_stage_latency_ms": _llm_stage_metrics["latency_ms"],
        "total_brain_simulations": len(evaluated_records) + companion_sign_flip_count,
        "total_runtime_seconds": round(time.time() - started, 4),
        "rejected_by_stage": rejected_by_stage,
        "topCandidates": [
            {
                "id": record["candidate"]["id"],
                "family": record["candidate"]["category"],
                "expression": record["candidate"]["expression"],
                "alphaId": record.get("backtest", {}).get("alpha_id"),
                "totalScore": record.get("dsr"),
                "scorecard": {
                    "submission": "blocked" if blocked_reason else ("degraded_no_daily_pnl" if record.get("status") == "no_daily_pnl" else "evaluated"),
                    "brainCheck": record.get("brainCheckOrthogonality", {}).get("passed"),
                },
                "metrics": record.get("backtest", {}),
            }
            for record in evaluated_records[:10]
        ],
    }
    write_json(output_dir / "summary.json", summary)
    if os.environ.get("KNOWLEDGE_DISTILL_ENABLED", "false").lower() == "true" and router is not None:
        _knowledge_every_n = max(1, int(os.environ.get("KNOWLEDGE_DISTILL_EVERY_N", "1")))
        if _should_run_cadence(shared_dir, "knowledge_distill", _knowledge_every_n):
            try:
                pool_data = read_json(output_dir / "pool.json", default={})
                distiller = KnowledgeDistiller(kb=kb, router=router)
                distiller.distill(pool_data)
            except Exception as exc:
                print(f"[distiller] skipped: {exc}")
        else:
            append_jsonl(
                progress_path,
                {
                    "at": _ts(),
                    "stage": "knowledge_distill_skipped",
                    "reason": "knowledge_distill_sampling",
                    "every_n": _knowledge_every_n,
                },
            )
    if router is not None:
        router.save_state(output_dir / "llm_router_state.json")
        # Also write to RUNS_DIR root so the dashboard can find it
        global_state_path = output_dir.parent / "llm_router_state.json"
        try:
            router.save_state(global_state_path)
        except Exception:
            pass
    # Embed any new KB examples added during this run (batch, non-blocking on failure)
    if kb_embedder is not None:
        try:
            n_new = kb.backfill_embeddings()
            if n_new:
                print(f"[kb] embedded {n_new} new examples after run", flush=True)
        except Exception as _emb_e:
            print(f"[kb] post-run embedding skipped: {_emb_e}", flush=True)
    append_jsonl(progress_path, {"at": _ts(), "stage": "finished", "summary": summary})
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def _import_submitted_feedback(kb: KnowledgeBase, feedback_path: Path) -> int:
    """Load previously submitted alphas from shared JSONL and add as positive KB examples."""
    if not feedback_path.exists():
        return 0
    count = 0
    try:
        for line in feedback_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except Exception:
                continue
            expression = entry.get("expression")
            alpha_id = entry.get("alphaId")
            i_sharpe = entry.get("isSharpe")
            if not expression or not alpha_id:
                continue
            # Only import alphas with decent IS quality; grade "A"/"B" = strong positive
            grade = entry.get("grade") or ""
            is_negative = grade in ("D", "F")
            sharpe_ok = isinstance(i_sharpe, (int, float)) and i_sharpe >= 0.5
            if not sharpe_ok and not grade:
                continue
            category = entry.get("category") or "QUALITY"
            sharpe_str = f"{i_sharpe:.3f}" if isinstance(i_sharpe, float) else str(i_sharpe)
            test_str = f"{entry.get('testSharpe'):.3f}" if isinstance(entry.get("testSharpe"), float) else "pending"
            hypothesis = (
                f"Previously submitted to BRAIN with IS Sharpe {sharpe_str}, "
                f"test Sharpe {test_str}, grade {grade or 'pending'}. "
                f"Use as {'negative' if is_negative else 'positive'} reference."
            )
            kb.upsert_example(
                item_id=f"submitted_{alpha_id}",
                expression=expression,
                category=category,
                hypothesis=hypothesis,
                is_negative_example=is_negative,
                metadata=entry,
            )
            count += 1
    except Exception as exc:
        print(f"[feedback] failed to load submitted feedback: {exc}", flush=True)
    if count:
        print(f"[feedback] loaded {count} submitted alpha(s) into KB", flush=True)
    return count


def _router_token_totals(router) -> dict[str, int]:
    if router is None:
        return {"prompt": 0, "completion": 0}
    prompt_total = sum(p.total_tokens_in for p in router._providers.values())
    completion_total = sum(p.total_tokens_out for p in router._providers.values())
    return {"prompt": prompt_total, "completion": completion_total}


def _router_stage_metrics(router) -> dict[str, dict[str, Any]]:
    if router is None:
        return {
            "tokens": {},
            "calls": {},
            "cost_usd": {},
            "latency_ms": {},
        }
    tokens: dict[str, dict[str, int]] = {}
    calls: dict[str, int] = {}
    cost_usd: dict[str, float] = {}
    latency_ms: dict[str, dict[str, float]] = {}
    for provider in router._providers.values():
        role = str(provider.role)
        role_tokens = tokens.setdefault(role, {"prompt": 0, "completion": 0})
        role_tokens["prompt"] += int(provider.total_tokens_in)
        role_tokens["completion"] += int(provider.total_tokens_out)
        calls[role] = calls.get(role, 0) + int(provider.calls)
        cost_usd[role] = round(
            cost_usd.get(role, 0.0)
            + float(provider.cost_usd(provider.total_tokens_in, provider.total_tokens_out)),
            6,
        )
        total_latency = latency_ms.get(role, {}).get("total", 0.0) + float(provider.total_latency_ms)
        role_calls = calls[role]
        latency_ms[role] = {
            "total": round(total_latency, 3),
            "avg_per_call": round(total_latency / role_calls, 3) if role_calls > 0 else 0.0,
        }
    return {
        "tokens": dict(sorted(tokens.items())),
        "calls": dict(sorted(calls.items())),
        "cost_usd": dict(sorted(cost_usd.items())),
        "latency_ms": dict(sorted(latency_ms.items())),
    }


def _sync_router_with_yaml(router: LLMRouter, yaml_router: LLMRouter) -> LLMRouter:
    for key, provider in yaml_router._providers.items():
        if key not in router._providers:
            router._providers[key] = provider
            router._providers_by_role.setdefault(provider.role, []).append(provider)
    for key in list(router._providers.keys()):
        if key not in yaml_router._providers:
            stale = router._providers.pop(key)
            role_list = router._providers_by_role.get(stale.role, [])
            router._providers_by_role[stale.role] = [item for item in role_list if item.name != stale.name]
    return router


def _initialize_router(output_dir: Path) -> LLMRouter | None:
    if os.environ.get("LLM_ROUTER_ENABLED", "true").lower() == "false":
        return None

    run_state_path = output_dir / "llm_router_state.json"
    shared_state_path = output_dir.parent / "llm_router_state.json"
    yaml_router = LLMRouter.from_yaml()
    router = yaml_router

    for state_path in (run_state_path, shared_state_path):
        if not state_path.exists():
            continue
        try:
            router = LLMRouter.load_state(state_path)
            break
        except Exception:
            router = yaml_router

    router = _sync_router_with_yaml(router, yaml_router)
    router._state_path = run_state_path
    router.daily_budget_usd = float(os.environ.get("LLM_BUDGET_DAILY_USD", "3.60"))
    return router


def _resolve_router_provider_name(router, role: str, preferred_name: str | None = None) -> str | None:
    if router is None:
        return None
    providers_by_role = getattr(router, "_providers_by_role", None)
    providers = providers_by_role.get(role, []) if isinstance(providers_by_role, dict) else []
    if not providers:
        return None
    if preferred_name:
        for provider in providers:
            if provider.name == preferred_name:
                return provider.name
    return providers[0].name


def _should_run_cadence(shared_dir: Path, key: str, every_n: int) -> bool:
    if every_n <= 1:
        return True
    cadence_path = shared_dir / "llm_cadence_state.json"
    try:
        state = read_json(cadence_path, default={}) or {}
        count = int(state.get(key, 0)) + 1
        state[key] = count
        write_json(cadence_path, state)
        return count % every_n == 0
    except Exception:
        # Fail-open to avoid silently disabling learning loops.
        return True


def _maybe_promote_sign_flip_companion(
    backtester: BrainBacktester,
    validator: ExpressionValidator,
    candidate: Any,
    result: Any,
    sim_settings: dict[str, Any] | None,
) -> tuple[Any, Any, dict[str, Any] | None]:
    if not should_try_sign_flip(result):
        return candidate, result, None

    flipped_expression = sign_flip_expression(candidate.expression)
    flipped_validation = validator.validate(flipped_expression)
    flipped_prescreen = economic_logic_prescreen(flipped_expression)
    companion_info: dict[str, Any] = {
        "type": "auto_sign_flip",
        "originalExpression": candidate.expression,
        "flippedExpression": flipped_expression,
        "originalSharpe": result.sharpe,
        "promoted": False,
    }
    if not flipped_validation.is_valid:
        companion_info["status"] = "invalid"
        companion_info["validationErrors"] = flipped_validation.errors
        return candidate, result, companion_info
    if not flipped_prescreen.is_valid:
        companion_info["status"] = "economic_prescreen_rejected"
        companion_info["economicPrescreen"] = asdict(flipped_prescreen)
        return candidate, result, companion_info

    try:
        flipped_result = backtester.submit_alpha(flipped_expression, period="IS", settings=sim_settings or None)
    except QuotaWaiting as error:
        companion_info["status"] = "quota_waiting"
        companion_info["error"] = str(error)
        return candidate, result, companion_info
    except Exception as error:
        companion_info["status"] = "error"
        companion_info["error"] = str(error)
        return candidate, result, companion_info

    companion_info["status"] = "tested"
    companion_info["flippedSharpe"] = flipped_result.sharpe
    companion_info["flippedResult"] = asdict(flipped_result)
    base_sharpe = float(result.sharpe) if result.sharpe is not None else float("-inf")
    flipped_sharpe = float(flipped_result.sharpe) if flipped_result.sharpe is not None else float("-inf")
    if flipped_sharpe <= base_sharpe:
        return candidate, result, companion_info

    flipped_candidate = candidate.__class__(
        id=f"{candidate.id}__signflip",
        category=candidate.category,
        hypothesis=f"Sign-flipped companion of: {candidate.hypothesis}",
        expression=flipped_expression,
        origin_refs=[*(candidate.origin_refs or []), "auto_sign_flip"],
        metadata={**getattr(candidate, "metadata", {}), "auto_sign_flip": "true"},
        opt_rounds=getattr(candidate, "opt_rounds", 0),
    )
    companion_info["promoted"] = True
    companion_info["promotedCandidateId"] = flipped_candidate.id
    return flipped_candidate, flipped_result, companion_info


_CASE_C_THRESHOLD = 0.20


def case_c_robust_score(
    sharpe: float | None,
    fitness: float | None,
    turnover: float | None,
    test_sharpe: float | None,
    brain_check_passed: bool | None,
) -> tuple[float, bool]:
    """Composite quality score for Case C (degraded mode, no daily PnL).

    Combines IS sharpe (dominant) + fitness + turnover + OOS consistency into a
    single score. Replaces the three separate boolean flags
    (degraded_sharpe_ok / turnover_ok / ortho_ok) with one ranked gate.

    Returns:
        (score, qualified) where score ∈ [0, 1] and qualified = score >= _CASE_C_THRESHOLD
        and brain_check_passed is not False.

    Calibration (threshold = 0.20):
        sharpe=0.50, fitness=None, turnover=None → 0.200 → qualified  (same as old gate)
        sharpe=0.45, fitness=0.70, turnover=0.30 → 0.295 → qualified  (old gate would reject)
        sharpe=0.40, fitness=None, turnover=None → 0.170 → rejected    (correct)
        sharpe=0.80, fitness=0.70, turnover=0.25 → 0.437 → qualified   (strong alpha)
    """
    if brain_check_passed is False:
        return 0.0, False
    if sharpe is None:
        return 0.0, False
    if turnover is not None and turnover > 0.75:
        return 0.0, False

    # IS Sharpe: weight 0.60 (dominant factor)
    # Maps [0, 2.0] → [0, 0.60]; sharpe=0.50 → 0.15, sharpe=1.0 → 0.30
    sharpe_score = min(0.60, max(0.0, sharpe / 2.0 * 0.60))

    # IS Fitness: weight 0.25 (secondary signal)
    # Maps [0, 1.5] → [0, 0.25]; fitness=0.50 → 0.083, fitness=1.0 → 0.167
    fitness_score = min(0.25, max(0.0, (fitness or 0.0) / 1.5 * 0.25)) if fitness is not None else 0.0

    # Turnover: weight 0.10 (prefer low/moderate; neutral when unknown)
    if turnover is None:
        turnover_score = 0.05
    else:
        turnover_score = min(0.10, max(0.0, (0.75 - turnover) / 0.75 * 0.10))

    # OOS consistency: up to 0.05 bonus — rewards stable IS→OOS transfer
    oos_score = 0.0
    if test_sharpe is not None and sharpe > 0:
        oos_ratio = min(1.0, test_sharpe / sharpe)
        oos_score = max(0.0, oos_ratio * 0.05)

    score = sharpe_score + fitness_score + turnover_score + oos_score
    return round(score, 4), score >= _CASE_C_THRESHOLD


def phase0_mode(mode: str) -> str:
    if mode == "generate":
        return "not_required_for_generate"
    # Use absolute path so it resolves correctly regardless of CWD (Railway, local, etc.)
    report = Path(__file__).parent.parent / "docs" / "phase0_brain_probe_report.md"
    if not report.exists():
        # Default to degraded mode when report is missing (e.g., Railway deploy without docs/)
        return "regular_tier_degraded_no_daily_pnl"
    text = report.read_text(encoding="utf-8")
    if "- Decision: Case A" in text or "- Decision: Case B" in text:
        return "daily_pnl_available"
    if "- Decision: Case C" in text:
        return "regular_tier_degraded_no_daily_pnl"
    return "phase0_brain_probe_not_actionable"


def _extract_operator_skeleton(expression: str) -> str:
    import re

    ops = list(dict.fromkeys(re.findall(r"[a-z_]+(?=\()", expression)))
    return ",".join(ops)


def _kb_write_back(kb, category, candidate, result, accepted: bool) -> None:
    skeleton = _extract_operator_skeleton(candidate.expression)
    kb.record_strategy_stat(category, "PASS" if accepted else "FAIL", skeleton)
    for op in skeleton.split(","):
        if op:
            kb.record_operator_stat(op, category, accepted)
    if accepted:
        kb.upsert_example(
            item_id=f"eval_{candidate.id}",
            expression=candidate.expression,
            category=category,
            hypothesis=candidate.hypothesis,
            is_negative_example=False,
            metadata={"sharpe": result.sharpe, "fitness": result.fitness, "turnover": result.turnover},
        )
    elif result.sharpe is not None:
        turnover = result.turnover or 0.0
        if result.sharpe < 0.4:
            reason = "LOW_SHARPE"
            suggested_fix = "Add group_rank neutralization and blend an independent second signal to boost Sharpe"
        elif result.fitness is not None and result.fitness < 0.5 and turnover > 0.60:
            reason = "LOW_FITNESS_HIGH_TURNOVER"
            suggested_fix = "Turnover >60% is crushing fitness — wrap fastest sub-expression with ts_decay_linear(x, 15)"
        elif result.fitness is not None and result.fitness < 0.5:
            reason = "LOW_FITNESS_WEAK_SIGNAL"
            suggested_fix = "Signal is weak despite acceptable turnover — add a second independent factor or switch category"
        else:
            reason = "FAILED_GATE"
            suggested_fix = "Check orthogonality or DSR; try different operator structure"
        if reason in ("LOW_FITNESS_HIGH_TURNOVER", "LOW_FITNESS_WEAK_SIGNAL", "LOW_SHARPE"):
            kb.upsert_example(
                item_id=f"fail_{candidate.id}",
                expression=candidate.expression,
                category=category,
                hypothesis=candidate.hypothesis,
                is_negative_example=True,
                metadata={"sharpe": result.sharpe, "fitness": result.fitness, "turnover": result.turnover},
            )
        kb.record_failure_pattern(reason=reason, expression=candidate.expression, suggested_fix=suggested_fix)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QuantBrain LLM alpha mining v2.1 runner")
    parser.add_argument("--mode", choices=["generate", "evaluate", "loop"], default="generate")
    parser.add_argument("--objective", default="discover robust US equity alphas")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--category", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--repair-context", default=None)
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--sim-settings", default=None, help="JSON string of BRAIN simulation settings overrides")
    parser.add_argument("--verbose", default="true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
