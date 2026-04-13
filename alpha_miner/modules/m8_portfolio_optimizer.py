from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .common import mean, pearson, stddev


@dataclass(frozen=True)
class PortfolioResult:
    weights: dict[str, float]
    optimizer: str
    risk_contribution: dict[str, float]
    mcr_pass: bool


class PortfolioOptimizer:
    def __init__(self, max_weight: float = 0.15, mcr_cap: float = 0.25):
        self.max_weight = max_weight
        self.mcr_cap = mcr_cap

    def optimize(self, pool: list[dict]) -> PortfolioResult:
        if not pool:
            return PortfolioResult(weights={}, optimizer="fallback", risk_contribution={}, mcr_pass=True)
        try:
            return self._cvxpy_optimize(pool)
        except Exception:
            return self._fallback_optimize(pool)

    def _cvxpy_optimize(self, pool: list[dict]) -> PortfolioResult:
        import cvxpy as cp  # type: ignore
        import numpy as np  # type: ignore

        returns = np.array([item["pnl"] for item in pool], dtype=float).T
        mu = returns.mean(axis=0)
        cov = np.cov(returns, rowvar=False)
        weights = cp.Variable(len(pool))
        objective = cp.Maximize(mu @ weights - 2.0 * cp.quad_form(weights, cov))
        constraints = [cp.sum(weights) == 1, weights >= 0, weights <= self.max_weight]
        problem = cp.Problem(objective, constraints)
        problem.solve(solver="SCS", verbose=False)
        if weights.value is None:
            raise RuntimeError("cvxpy returned no solution")
        result_weights = {pool[index]["alpha_id"]: float(max(0, weights.value[index])) for index in range(len(pool))}
        return self._with_risk(pool, result_weights, "cvxpy")

    def _fallback_optimize(self, pool: list[dict]) -> PortfolioResult:
        scores = {}
        for item in pool:
            pnl = item.get("pnl", [])
            sigma = stddev(pnl)
            scores[item["alpha_id"]] = max(0.0, mean(pnl) / sigma if sigma else 0.0)
        if sum(scores.values()) <= 0:
            scores = {item["alpha_id"]: 1.0 for item in pool}
        raw = {key: value / sum(scores.values()) for key, value in scores.items()}
        clipped = {key: min(self.max_weight, value) for key, value in raw.items()}
        total = sum(clipped.values()) or 1.0
        weights = {key: value / total for key, value in clipped.items()}
        return self._with_risk(pool, weights, "fallback")

    def _with_risk(self, pool: list[dict], weights: dict[str, float], optimizer: str) -> PortfolioResult:
        risk = self.risk_budget_check(weights, pool)
        return PortfolioResult(weights=weights, optimizer=optimizer, risk_contribution=risk, mcr_pass=all(v <= self.mcr_cap for v in risk.values()))

    def risk_budget_check(self, weights: dict[str, float], pool: list[dict]) -> dict[str, float]:
        if not weights:
            return {}
        weighted_risk = {}
        for item in pool:
            alpha_id = item["alpha_id"]
            weighted_risk[alpha_id] = abs(weights.get(alpha_id, 0.0)) * stddev(item.get("pnl", []))
        total = sum(weighted_risk.values()) or 1.0
        return {key: value / total for key, value in weighted_risk.items()}
