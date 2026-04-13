from __future__ import annotations

from dataclasses import replace

from .m2_hypothesis_agent import Candidate


class OptimizationLoop:
    def __init__(self, max_rounds: int = 3):
        self.max_rounds = max_rounds

    def optimize(self, candidate: Candidate, failure_reason: str) -> Candidate | None:
        if candidate.opt_rounds >= self.max_rounds:
            return None
        expression = candidate.expression
        if "group_rank(" not in expression:
            expression = f"group_rank({expression}, industry)"
        elif "ts_mean(" not in expression:
            expression = f"ts_mean({expression}, 20)"
        else:
            expression = f"winsorize({expression})"
        return replace(
            candidate,
            id=f"{candidate.id}_opt{candidate.opt_rounds + 1}",
            expression=expression,
            hypothesis=f"{candidate.hypothesis} Optimization targeted failure: {failure_reason}.",
            opt_rounds=candidate.opt_rounds + 1,
        )
