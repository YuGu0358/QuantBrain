from alpha_miner.modules.m_generation_quality import assess_generation_candidate_quality
from alpha_miner.modules.m2_hypothesis_agent import Candidate
from alpha_miner.modules.m2_hypothesis_agent import HypothesisAgent
from alpha_miner.modules.llm_cache import LLMCache
from alpha_miner.modules.m1_knowledge_base import KnowledgeBase


def test_generation_quality_accepts_structured_category_native_single_family_candidate():
    assessment = assess_generation_candidate_quality(
        "group_rank(ts_zscore(returns, 20) - ts_zscore(returns, 120), subindustry)",
        category="REVERSAL",
        candidate_id="rev_001",
    )

    assert assessment.passed is True
    assert "single_family_exposure" not in assessment.reasons


def test_generate_batch_keeps_structured_single_family_reversal_candidate(tmp_path):
    kb = KnowledgeBase(tmp_path / "kb.db")
    cache = LLMCache(tmp_path / "cache")
    agent = HypothesisAgent(kb=kb, cache=cache, taxonomy={"REVERSAL": {}})

    candidate = Candidate(
        id="rev_001",
        category="REVERSAL",
        hypothesis="mean reversion after medium-horizon instability compression",
        expression="group_rank(ts_zscore(returns, 20) - ts_zscore(returns, 120), subindustry)",
        origin_refs=[],
    )

    selected = agent._select_generation_candidates(
        "discover reversal alphas",
        "REVERSAL",
        [candidate],
        n=1,
        judge_applied=False,
    )

    assert [item.expression for item in selected] == [candidate.expression]
    assert float(selected[0].metadata["generation_quality_score"]) > 0
