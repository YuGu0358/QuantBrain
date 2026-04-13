from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any

from .config_loader import load_json


@dataclass(frozen=True)
class RagContext:
    positive: list[dict[str, Any]]
    negative: list[dict[str, Any]]
    failure_patterns: list = field(default_factory=list)
    strategy_stats: dict = field(default_factory=dict)


class KnowledgeBase:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as db:
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS examples (
                  id TEXT PRIMARY KEY,
                  category TEXT,
                  expression TEXT NOT NULL,
                  hypothesis TEXT,
                  is_negative_example INTEGER NOT NULL DEFAULT 0,
                  metadata_json TEXT NOT NULL DEFAULT '{}'
                )
                """
            )
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS strategy_stats (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  category TEXT NOT NULL,
                  gate_result TEXT NOT NULL,
                  operator_skeleton TEXT DEFAULT '',
                  ts INTEGER NOT NULL
                )
                """
            )
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS failure_patterns (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  reason TEXT NOT NULL,
                  expression TEXT NOT NULL,
                  suggested_fix TEXT NOT NULL,
                  ts INTEGER NOT NULL
                )
                """
            )
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS operator_stats (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  operator TEXT NOT NULL,
                  category TEXT NOT NULL,
                  passed INTEGER NOT NULL,
                  ts INTEGER NOT NULL
                )
                """
            )
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS market_regime (
                  regime_key TEXT PRIMARY KEY,
                  summary TEXT NOT NULL,
                  top_categories_json TEXT NOT NULL,
                  ts INTEGER NOT NULL
                )
                """
            )

    def import_wq101_negative_examples(self, seed_path: Path) -> int:
        seeds = load_json(seed_path)
        count = 0
        for seed in seeds:
            self.upsert_example(
                item_id=seed["id"],
                expression=seed["expression"],
                category=seed.get("category", "WQ101"),
                hypothesis=seed.get("note", "WQ101 baseline negative example."),
                is_negative_example=True,
                metadata=seed,
            )
            count += 1
        return count

    def upsert_example(
        self,
        item_id: str,
        expression: str,
        category: str,
        hypothesis: str,
        is_negative_example: bool,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        with sqlite3.connect(self.db_path) as db:
            db.execute(
                """
                INSERT OR REPLACE INTO examples(id, category, expression, hypothesis, is_negative_example, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (item_id, category, expression, hypothesis, int(is_negative_example), json.dumps(metadata or {}, ensure_ascii=False)),
            )

    def record_strategy_stat(self, category: str, gate_result: str, operator_skeleton: str = '') -> None:
        with sqlite3.connect(self.db_path) as db:
            db.execute(
                """
                INSERT INTO strategy_stats(category, gate_result, operator_skeleton, ts)
                VALUES (?, ?, ?, ?)
                """,
                (category, gate_result, operator_skeleton, int(time.time())),
            )

    def get_strategy_stats(self, category: str) -> dict:
        with sqlite3.connect(self.db_path) as db:
            row = db.execute(
                """
                SELECT COUNT(*), SUM(CASE WHEN gate_result = 'PASS' THEN 1 ELSE 0 END)
                FROM strategy_stats
                WHERE category = ?
                """,
                (category,),
            ).fetchone()
        attempts = int(row[0] or 0)
        wins = int(row[1] or 0)
        return {"attempts": attempts, "wins": wins, "win_rate": wins / attempts if attempts else 0.0}

    def record_failure_pattern(self, reason: str, expression: str, suggested_fix: str) -> None:
        with sqlite3.connect(self.db_path) as db:
            db.execute(
                """
                INSERT INTO failure_patterns(reason, expression, suggested_fix, ts)
                VALUES (?, ?, ?, ?)
                """,
                (reason, expression, suggested_fix, int(time.time())),
            )

    def get_failure_patterns(self, limit: int = 10) -> list:
        with sqlite3.connect(self.db_path) as db:
            rows = db.execute(
                """
                SELECT reason, expression, suggested_fix
                FROM failure_patterns
                ORDER BY ts DESC
                LIMIT ?
                """,
                (limit,),
            )
            return [
                {"reason": row[0], "expression": row[1], "suggested_fix": row[2]}
                for row in rows
            ]

    def record_operator_stat(self, operator: str, category: str, passed: bool) -> None:
        with sqlite3.connect(self.db_path) as db:
            db.execute(
                """
                INSERT INTO operator_stats(operator, category, passed, ts)
                VALUES (?, ?, ?, ?)
                """,
                (operator, category, int(passed), int(time.time())),
            )

    def get_operator_stats(self, category: str) -> dict:
        with sqlite3.connect(self.db_path) as db:
            rows = db.execute(
                """
                SELECT operator, COUNT(*), SUM(passed)
                FROM operator_stats
                WHERE category = ?
                GROUP BY operator
                """,
                (category,),
            )
            return {
                row[0]: {
                    "attempts": int(row[1] or 0),
                    "wins": int(row[2] or 0),
                    "win_rate": int(row[2] or 0) / int(row[1] or 0) if row[1] else 0.0,
                }
                for row in rows
            }

    def upsert_market_regime(self, regime_key: str, summary: str, top_categories: list) -> None:
        with sqlite3.connect(self.db_path) as db:
            db.execute(
                """
                INSERT OR REPLACE INTO market_regime(regime_key, summary, top_categories_json, ts)
                VALUES (?, ?, ?, ?)
                """,
                (regime_key, summary, json.dumps(top_categories), int(time.time())),
            )

    def get_market_regime(self, regime_key: str) -> dict | None:
        with sqlite3.connect(self.db_path) as db:
            row = db.execute(
                """
                SELECT summary, top_categories_json
                FROM market_regime
                WHERE regime_key = ?
                """,
                (regime_key,),
            ).fetchone()
        if row is None:
            return None
        return {"summary": row[0], "top_categories": json.loads(row[1])}

    def rag_context(self, category: str, limit: int = 4) -> RagContext:
        with sqlite3.connect(self.db_path) as db:
            positive = [
                _row_to_dict(row)
                for row in db.execute(
                    """
                    SELECT id, category, expression, hypothesis, is_negative_example, metadata_json
                    FROM examples
                    WHERE is_negative_example = 0 AND (category = ? OR ? = '')
                    LIMIT ?
                    """,
                    (category, category, limit),
                )
            ]
            negative = [
                _row_to_dict(row)
                for row in db.execute(
                    """
                    SELECT id, category, expression, hypothesis, is_negative_example, metadata_json
                    FROM examples
                    WHERE is_negative_example = 1
                    LIMIT ?
                    """,
                    (limit,),
                )
            ]
        failure_patterns = self.get_failure_patterns(limit=5)
        strategy_stats = self.get_strategy_stats(category)
        return RagContext(positive=positive, negative=negative, failure_patterns=failure_patterns, strategy_stats=strategy_stats)


def _row_to_dict(row: tuple[Any, ...]) -> dict[str, Any]:
    return {
        "id": row[0],
        "category": row[1],
        "expression": row[2],
        "hypothesis": row[3],
        "is_negative_example": bool(row[4]),
        "metadata": json.loads(row[5] or "{}"),
    }
