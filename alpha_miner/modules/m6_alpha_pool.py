from __future__ import annotations

import sqlite3
import re
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .common import pearson, read_json, write_json


@dataclass(frozen=True)
class OrthogonalityResult:
    passed: bool
    max_abs_correlation: float
    nearest_alpha_id: str | None
    source: str = "pnl_pearson"


class AlphaPool:
    def __init__(self, db_path: Path, threshold: float = 0.5):
        self.db_path = db_path
        self.threshold = threshold
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as db:
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS alphas (
                  alpha_id TEXT PRIMARY KEY,
                  expression TEXT NOT NULL,
                  category TEXT,
                  pnl_path TEXT NOT NULL,
                  holdout_used INTEGER NOT NULL DEFAULT 0,
                  metadata_json TEXT NOT NULL DEFAULT '{}'
                )
                """
            )

    def add_alpha(
        self,
        alpha_id: str,
        expression: str,
        category: str,
        pnl_path: Path,
        holdout_used: bool,
        metadata_json: str = "{}",
    ) -> None:
        if not pnl_path.exists():
            raise ValueError(f"PnL path does not exist: {pnl_path}")
        with sqlite3.connect(self.db_path) as db:
            db.execute(
                """
                INSERT OR REPLACE INTO alphas(alpha_id, expression, category, pnl_path, holdout_used, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (alpha_id, expression, category, str(pnl_path), int(holdout_used), metadata_json),
            )

    def assert_holdout_available(self, alpha_id: str) -> None:
        with sqlite3.connect(self.db_path) as db:
            row = db.execute("SELECT holdout_used FROM alphas WHERE alpha_id = ?", (alpha_id,)).fetchone()
        if row and row[0]:
            raise ValueError(f"Alpha {alpha_id} already used holdout and cannot be modified/retested.")

    def check_orthogonality(self, new_pnl: Sequence[float], threshold: float | None = None) -> OrthogonalityResult:
        limit = self.threshold if threshold is None else threshold
        nearest_alpha_id = None
        max_abs = 0.0
        for alpha_id, pnl_path in self._iter_pnl_paths():
            existing = load_pnl_series(Path(pnl_path))
            rho = abs(pearson(new_pnl, existing))
            if rho > max_abs:
                max_abs = rho
                nearest_alpha_id = alpha_id
        return OrthogonalityResult(passed=max_abs < limit, max_abs_correlation=max_abs, nearest_alpha_id=nearest_alpha_id)

    def check_expression_similarity(self, expression: str, threshold: float = 0.80) -> OrthogonalityResult:
        nearest_alpha_id = None
        max_similarity = 0.0
        for alpha_id, existing_expression in self._iter_expressions():
            similarity = expression_jaccard(expression, existing_expression)
            if similarity > max_similarity:
                max_similarity = similarity
                nearest_alpha_id = alpha_id
        return OrthogonalityResult(
            passed=max_similarity < threshold,
            max_abs_correlation=max_similarity,
            nearest_alpha_id=nearest_alpha_id,
            source="expression_jaccard_proxy",
        )

    def check_metric_similarity(
        self,
        metrics: dict[str, Any],
        threshold: float = 0.95,
    ) -> OrthogonalityResult:
        nearest_alpha_id = None
        max_similarity = 0.0
        for alpha_id, metadata_json in self._iter_metadata():
            existing = read_metadata(metadata_json).get("metrics", {})
            similarity = metric_cosine_similarity(metrics, existing)
            if similarity > max_similarity:
                max_similarity = similarity
                nearest_alpha_id = alpha_id
        return OrthogonalityResult(
            passed=max_similarity < threshold,
            max_abs_correlation=max_similarity,
            nearest_alpha_id=nearest_alpha_id,
            source="aggregate_metric_cosine_proxy",
        )

    def _iter_pnl_paths(self) -> list[tuple[str, str]]:
        with sqlite3.connect(self.db_path) as db:
            return list(db.execute("SELECT alpha_id, pnl_path FROM alphas"))

    def _iter_expressions(self) -> list[tuple[str, str]]:
        with sqlite3.connect(self.db_path) as db:
            return list(db.execute("SELECT alpha_id, expression FROM alphas"))

    def _iter_metadata(self) -> list[tuple[str, str]]:
        with sqlite3.connect(self.db_path) as db:
            return list(db.execute("SELECT alpha_id, metadata_json FROM alphas"))


def save_pnl_series(path: Path, values: Sequence[float]) -> None:
    write_json(path, {"pnl": [float(value) for value in values]})


def load_pnl_series(path: Path) -> list[float]:
    payload = read_json(path, default={}) or {}
    values = payload.get("pnl", payload if isinstance(payload, list) else [])
    return [float(value) for value in values]


def expression_tokens(expression: str) -> set[str]:
    return {token.lower() for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expression)}


def expression_jaccard(left: str, right: str) -> float:
    left_tokens = expression_tokens(left)
    right_tokens = expression_tokens(right)
    if not left_tokens and not right_tokens:
        return 1.0
    union = left_tokens | right_tokens
    if not union:
        return 0.0
    return len(left_tokens & right_tokens) / len(union)


def metric_cosine_similarity(left: dict[str, Any], right: dict[str, Any]) -> float:
    fields = ["sharpe", "fitness", "turnover", "returns", "drawdown", "margin"]
    left_values = [to_float(left.get(field)) for field in fields]
    right_values = [to_float(right.get(field)) for field in fields]
    numerator = sum(a * b for a, b in zip(left_values, right_values))
    left_norm = sum(value * value for value in left_values) ** 0.5
    right_norm = sum(value * value for value in right_values) ** 0.5
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


def read_metadata(metadata_json: str) -> dict[str, Any]:
    try:
        payload = json.loads(metadata_json)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
