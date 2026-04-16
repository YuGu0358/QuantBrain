from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from alpha_miner.modules.llm_router import LLMRouter
from alpha_miner.modules.m9_knowledge_distiller import KnowledgeDistiller

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
    append_jsonl(progress_path, {"stage": "started", "mode": args.mode, "objective": args.objective, "engine": "python-v2", "repairContext": args.repair_context})

    kb = KnowledgeBase(output_dir / "alpha_pool.db")
    kb.import_wq101_negative_examples(PACKAGE_ROOT / "seeds" / "wq101_alphas.json")
    _import_submitted_feedback(kb, output_dir.parent / "submitted_alphas.jsonl")
    cache = LLMCache(output_dir / "llm_cache")
    taxonomy = load_taxonomy()
    router = None
    if os.environ.get("LLM_ROUTER_ENABLED", "true").lower() != "false":
        router = LLMRouter.from_yaml()
        router._state_path = output_dir / "llm_router_state.json"
        # load existing state if available
        state_file = output_dir / "llm_router_state.json"
        if state_file.exists():
            router = LLMRouter.load_state(state_file)
            router._state_path = state_file
        router.daily_budget_usd = float(os.environ.get("LLM_BUDGET_DAILY_USD", "3.60"))
    agent = HypothesisAgent(
        kb=kb,
        cache=cache,
        taxonomy=taxonomy,
        model=os.environ.get("OPENAI_IDEA_MODEL", "gpt-4o-mini"),
        temperature=float(generation_cfg.get("temperature", 0.4)),
        top_p=float(generation_cfg.get("top_p", 0.9)),
        max_tokens=int(generation_cfg.get("max_tokens", 800)),
        seed=int(generation_cfg.get("seed", 42)),
        router=router,
        use_llm=(router is not None),
    )
    validator = ExpressionValidator()
    stat = StatSignificance(
        dsr_threshold=float(config.get("stat", {}).get("dsr_threshold", 0.95)),
        pbo_threshold=float(config.get("stat", {}).get("pbo_threshold", 0.30)),
    )
    alpha_pool = AlphaPool(output_dir / "alpha_pool.db", threshold=float(config.get("pool", {}).get("orthogonality_threshold", 0.5)))
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

    candidates = agent.generate_batch(effective_objective, category=category, n=batch_size, use_llm=args.use_llm, repair_context=repair_ctx)

    validator_records = []
    valid_candidates = []
    rejected_by_stage = {"validator": 0, "dsr": 0, "pbo": 0, "oos": 0, "orthogonality": 0, "no_pnl": 0}
    for candidate in candidates:
        result = validator.validate(candidate.expression)
        validator_records.append({"candidate": asdict(candidate), "validation": asdict(result)})
        if result.is_valid:
            valid_candidates.append(candidate)
        else:
            rejected_by_stage["validator"] += 1
            append_jsonl(progress_path, {"stage": "rejected", "reason": "validator", "candidate_id": candidate.id, "errors": result.errors})

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
    if blocked_reason:
        append_jsonl(progress_path, {"stage": "blocked", "reason": blocked_reason})
    elif args.mode in {"evaluate", "loop"}:
        for candidate in valid_candidates:
            try:
                result = backtester.submit_alpha(candidate.expression, period="IS", settings=sim_settings or None)
            except QuotaWaiting as error:
                append_jsonl(progress_path, {"stage": "waiting", "reason": str(error)})
                break
            record = {"candidate": asdict(candidate), "backtest": asdict(result)}
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
                        append_jsonl(progress_path, {"stage": "waiting", "reason": str(error), "alpha_id": result.alpha_id})
                        record["brainCheckOrthogonality"] = {"status": "waiting", "reason": str(error)}
                # Degraded gate: accept alphas with sharpe >= 0.5 and reasonable turnover into
                # the pool so the repair loop has candidates to work on and the dashboard
                # shows progress. DSR cannot be computed without daily PnL, so we use a
                # looser proxy: sharpe >= 0.5, turnover in [0.01, 0.70], no self-correlation.
                degraded_sharpe_ok = (result.sharpe is not None) and (result.sharpe >= 0.5)
                degraded_turnover_ok = (result.turnover is None) or (0.01 <= result.turnover <= 0.70)
                degraded_ortho_ok = brain_check_passed is not False
                if degraded_mode and result.alpha_id and degraded_sharpe_ok and degraded_turnover_ok and degraded_ortho_ok:
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
                        "stage": "degraded_evaluation",
                        "candidate_id": candidate.id,
                        "alpha_id": result.alpha_id,
                        "sharpe": result.sharpe,
                        "turnover": result.turnover,
                        "degraded_qualified": degraded_mode and result.alpha_id is not None and degraded_sharpe_ok and degraded_turnover_ok and degraded_ortho_ok,
                        "reason": "regular tier exposes aggregate metrics but no daily pnl",
                    },
                )
            evaluated_records.append(record)
            append_jsonl(progress_path, {"stage": "evaluated", "candidate_id": candidate.id, "status": result.status})

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

    summary = {
        "engine": "python-v2",
        "mode": args.mode,
        "objective": args.objective,
        "category": category,
        "phase0Status": phase0_status,
        "degradedMode": degraded_mode,
        "generatedCandidates": len(candidates),
        "validCandidates": len(valid_candidates),
        "qualified_alphas_count": len(portfolio_pool),
        "degraded_candidates_count": sum(1 for record in evaluated_records if record.get("status") == "no_daily_pnl"),
        "total_llm_tokens": _router_token_totals(router),
        "total_llm_cost_usd": round(router.spent_usd, 6) if router is not None else 0.0,
        "total_brain_simulations": len(evaluated_records),
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
        try:
            from alpha_miner.modules.common import read_json

            pool_data = read_json(output_dir / "pool.json", default={})
            distiller = KnowledgeDistiller(kb=kb, router=router)
            distiller.distill(pool_data)
        except Exception as exc:
            print(f"[distiller] skipped: {exc}")
    if router is not None:
        router.save_state(output_dir / "llm_router_state.json")
        # Also write to RUNS_DIR root so the dashboard can find it
        global_state_path = output_dir.parent / "llm_router_state.json"
        try:
            router.save_state(global_state_path)
        except Exception:
            pass
    append_jsonl(progress_path, {"stage": "finished", "summary": summary})
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
