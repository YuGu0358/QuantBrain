"""Phase 3 tests: BanditScheduler UCB1 + Planner overrides."""
import tempfile
from pathlib import Path

from alpha_miner.modules.m_diagnoser import DiagnosisReport
from alpha_miner.modules.m_planner import Planner, RepairPlan
from alpha_miner.modules.m_scheduler import BanditScheduler


def _diagnosis(symptom: str) -> DiagnosisReport:
    return DiagnosisReport(
        primary_symptom=symptom, secondary_symptoms=[], root_causes=[],
        repair_priorities=[], do_not_change=[], raw={}
    )


# ── BanditScheduler ──────────────────────────────────────────────────────────

def test_scheduler_uniform_at_start():
    with tempfile.TemporaryDirectory() as d:
        s = BanditScheduler(Path(d) / "s.db")
        w = s.get_weights()
        assert set(w) == {"param_tune", "struct_mutation", "template_retrieval", "llm_mutation"}
        assert abs(sum(w.values()) - 1.0) < 1e-6
        for v in w.values():
            assert abs(v - 0.25) < 1e-6


def test_scheduler_shifts_toward_best_action():
    with tempfile.TemporaryDirectory() as d:
        s = BanditScheduler(Path(d) / "s.db")
        for _ in range(6):
            s.record_batch_outcomes([
                {"action_type": "param_tune",       "accepted": True,  "j_old": 1.0, "j_new": 1.6},
                {"action_type": "struct_mutation",   "accepted": False, "j_old": 1.0, "j_new": 0.9},
                {"action_type": "template_retrieval","accepted": False, "j_old": 1.0, "j_new": 1.0},
                {"action_type": "llm_mutation",      "accepted": False, "j_old": 1.0, "j_new": 1.0},
            ])
        w = s.get_weights()
        assert w["param_tune"] > w["struct_mutation"]
        assert w["param_tune"] > w["template_retrieval"]


def test_scheduler_get_stats():
    with tempfile.TemporaryDirectory() as d:
        s = BanditScheduler(Path(d) / "s.db")
        s.record_outcome("param_tune", accepted=True, j_improvement=0.5)
        stats = s.get_stats()
        pt = next(x for x in stats if x["action_type"] == "param_tune")
        assert pt["total_attempts"] == 1
        assert pt["total_accepted"] == 1
        assert abs(pt["total_j_improvement"] - 0.5) < 1e-6


# ── Planner ──────────────────────────────────────────────────────────────────

def test_planner_default_mix_sums_to_budget():
    p = Planner(router=None)
    plan = p.plan(_diagnosis("low_sharpe"), {}, total_budget=10)
    assert sum(plan.candidate_mix.values()) == 10
    assert set(plan.candidate_mix) == {"param_tune", "struct_mutation", "template_retrieval", "llm_mutation"}


def test_planner_high_turnover_override():
    p = Planner(router=None)
    plan = p.plan(_diagnosis("high_turnover"), {}, total_budget=10)
    assert plan.candidate_mix["param_tune"] == 4
    assert plan.candidate_mix["llm_mutation"] == 1
    assert sum(plan.candidate_mix.values()) == 10


def test_planner_high_corrlib_override():
    p = Planner(router=None)
    plan = p.plan(_diagnosis("high_corrlib"), {}, total_budget=10)
    assert plan.candidate_mix["struct_mutation"] >= 4
    assert sum(plan.candidate_mix.values()) == 10


def test_planner_family_saturated_zeroes_template():
    p = Planner(router=None)
    plan = p.plan(_diagnosis("low_sharpe"), {"family_saturated": True}, total_budget=10)
    assert plan.candidate_mix["template_retrieval"] == 0
    assert sum(plan.candidate_mix.values()) == 10


def test_planner_scheduler_weights_respected():
    p = Planner(router=None)
    weights = {"param_tune": 0.8, "struct_mutation": 0.1, "template_retrieval": 0.05, "llm_mutation": 0.05}
    plan = p.plan(_diagnosis("low_sharpe"), {}, total_budget=10, scheduler_weights=weights)
    assert plan.candidate_mix["param_tune"] >= 7
    assert sum(plan.candidate_mix.values()) == 10
