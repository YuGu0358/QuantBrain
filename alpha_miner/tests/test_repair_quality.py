from alpha_miner.modules.m_repair_quality import assess_repair_candidate_quality
from alpha_miner.modules.m2_hypothesis_agent import Candidate


def test_recursive_repair_requires_material_change():
    candidate = Candidate(
        id="repair_001",
        category="QUALITY",
        hypothesis="only extends lookback",
        expression="rank(ts_mean(returns, 20))",
        origin_refs=["langchain_agent", "param_tune"],
    )

    assessment = assess_repair_candidate_quality(
        parent_expression="rank(ts_mean(returns, 10))",
        candidate=candidate,
        diagnosis={"primary_symptom": "low_sharpe", "do_not_change": []},
        repair_context={
            "repairDepth": 1,
            "failedChecks": ["SHARPE"],
            "gate": {"reasons": ["still low sharpe"]},
        },
    )

    assert assessment.passed is False
    assert "recursive_repair_requires_material_change" in assessment.reasons


def test_repair_quality_accepts_structural_cross_family_escape():
    candidate = Candidate(
        id="repair_002",
        category="QUALITY",
        hypothesis="switch to quality plus liquidity blend",
        expression="group_rank(ts_rank(cashflow_op / assets, 252) + ts_rank(volume / adv20, 63), industry)",
        origin_refs=["langchain_agent", "struct_mutation"],
    )

    assessment = assess_repair_candidate_quality(
        parent_expression="rank(ts_mean(returns, 10))",
        candidate=candidate,
        diagnosis={"primary_symptom": "low_sharpe", "do_not_change": []},
        repair_context={
            "repairDepth": 1,
            "failedChecks": ["SHARPE"],
            "gate": {"reasons": ["still low sharpe"]},
        },
    )

    assert assessment.passed is True
    assert assessment.score > 0
