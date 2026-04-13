from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Sequence

from .common import mean, normal_cdf, normal_ppf, stddev


@dataclass(frozen=True)
class PurgedFold:
    train_indices: list[int]
    test_indices: list[int]


class StatSignificance:
    def __init__(self, dsr_threshold: float = 0.95, pbo_threshold: float = 0.30):
        self.dsr_threshold = dsr_threshold
        self.pbo_threshold = pbo_threshold

    def sharpe(self, returns: Sequence[float], periods_per_year: int = 252) -> float:
        sigma = stddev(returns)
        if sigma == 0:
            return 0.0
        return mean(returns) / sigma * math.sqrt(periods_per_year)

    def deflated_sharpe(
        self,
        sr_observed: float,
        sr_trials: Sequence[float],
        t: int,
        skew: float = 0.0,
        kurt: float = 3.0,
    ) -> float:
        if t <= 1:
            return 0.0
        trials = [float(value) for value in sr_trials if math.isfinite(float(value))]
        n_trials = max(len(trials), 1)
        variance = stddev(trials) ** 2 if len(trials) > 1 else 1.0
        gamma = 0.5772156649015329
        expected_max = math.sqrt(max(variance, 1e-12)) * (
            (1 - gamma) * normal_ppf(1 - 1 / max(n_trials, 2)) + gamma * normal_ppf(1 - 1 / (max(n_trials, 2) * math.e))
        )
        denominator = math.sqrt(max(1e-12, 1 - skew * sr_observed + ((kurt - 1) / 4) * sr_observed**2))
        statistic = (sr_observed - expected_max) * math.sqrt(t - 1) / denominator
        return normal_cdf(statistic)

    def deflated_sharpe_pass(self, sr_observed: float, sr_trials: Sequence[float], t: int) -> bool:
        return self.deflated_sharpe(sr_observed, sr_trials, t) >= self.dsr_threshold

    def pbo(self, is_sharpes: Sequence[float], oos_sharpes: Sequence[float]) -> float:
        if len(is_sharpes) != len(oos_sharpes) or len(is_sharpes) < 2:
            return 1.0
        losses = 0
        total = 0
        indices = range(len(is_sharpes))
        half = max(1, len(is_sharpes) // 2)
        for combo in itertools.combinations(indices, half):
            best = max(combo, key=lambda index: is_sharpes[index])
            sorted_oos = sorted(oos_sharpes)
            rank = sorted_oos.index(oos_sharpes[best])
            if rank < len(sorted_oos) / 2:
                losses += 1
            total += 1
        return losses / total if total else 1.0

    def pbo_pass(self, is_sharpes: Sequence[float], oos_sharpes: Sequence[float]) -> bool:
        return self.pbo(is_sharpes, oos_sharpes) <= self.pbo_threshold


def purged_kfold_indices(size: int, n_splits: int = 5, purge: int = 10, embargo: int = 5) -> list[PurgedFold]:
    if size <= 0 or n_splits <= 1:
        raise ValueError("size must be positive and n_splits must exceed 1")
    fold_size = max(1, size // n_splits)
    folds: list[PurgedFold] = []
    for fold in range(n_splits):
        start = fold * fold_size
        end = size if fold == n_splits - 1 else min(size, start + fold_size)
        test = list(range(start, end))
        blocked_start = max(0, start - purge)
        blocked_end = min(size, end + embargo)
        train = [index for index in range(size) if index < blocked_start or index >= blocked_end]
        folds.append(PurgedFold(train_indices=train, test_indices=test))
    return folds
