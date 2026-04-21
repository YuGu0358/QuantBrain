from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .config_loader import load_json


@dataclass(frozen=True)
class RagContext:
    positive: list[dict[str, Any]]
    negative: list[dict[str, Any]]
    failure_patterns: list = field(default_factory=list)
    strategy_stats: dict = field(default_factory=dict)


class KnowledgeBase:
    """SQLite-backed knowledge store with optional semantic (vector) retrieval.

    Pass an *embedder* object (duck-typed: must expose ``embed_query(text) ->
    list[float]`` and ``embed_documents(texts) -> list[list[float]]``) to enable
    cosine-similarity search in ``rag_context()``.  Without an embedder the
    method falls back to ``ORDER BY rowid DESC LIMIT k`` (original behaviour).

    Embeddings are computed once per example and persisted in the ``examples``
    table so API calls are only made for new/updated rows.
    """

    def __init__(self, db_path: Path, embedder: Any = None):
        self.db_path = db_path
        self.embedder = embedder
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

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
                  metadata_json TEXT NOT NULL DEFAULT '{}',
                  embedding_json TEXT
                )
                """
            )
            # Migration: add embedding_json to pre-existing databases
            existing_cols = {row[1] for row in db.execute("PRAGMA table_info(examples)")}
            if "embedding_json" not in existing_cols:
                db.execute("ALTER TABLE examples ADD COLUMN embedding_json TEXT")

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

    # ------------------------------------------------------------------
    # Example management
    # ------------------------------------------------------------------

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
                defer_embedding=True,
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
        defer_embedding: bool = False,
    ) -> None:
        # Keep existing embedding if present. Bulk imports can opt into deferred
        # embedding so backfill_embeddings() can batch API calls.
        embedding_json: str | None = None
        if self.embedder is not None:
            with sqlite3.connect(self.db_path) as db:
                row = db.execute(
                    "SELECT embedding_json FROM examples WHERE id = ?", (item_id,)
                ).fetchone()
            if row and row[0]:
                embedding_json = row[0]  # keep existing vector
            elif not defer_embedding:
                text = f"{category} {expression} {hypothesis or ''}"
                try:
                    embedding_json = json.dumps(self.embedder.embed_query(text))
                except Exception as exc:
                    print(f"[kb] embedding skipped for {item_id}: {exc}", flush=True)

        with sqlite3.connect(self.db_path) as db:
            db.execute(
                """
                INSERT OR REPLACE INTO examples(
                  id, category, expression, hypothesis,
                  is_negative_example, metadata_json, embedding_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item_id, category, expression, hypothesis,
                    int(is_negative_example),
                    json.dumps(metadata or {}, ensure_ascii=False),
                    embedding_json,
                ),
            )

    def backfill_embeddings(self, batch_size: int = 50) -> int:
        """Compute and store embeddings for any examples that don't have one yet.

        Uses ``embed_documents()`` for batched API calls.
        Returns the number of examples that were updated.
        """
        if self.embedder is None:
            return 0
        with sqlite3.connect(self.db_path) as db:
            rows = db.execute(
                "SELECT id, category, expression, hypothesis FROM examples WHERE embedding_json IS NULL"
            ).fetchall()
        if not rows:
            return 0

        updated = 0
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            texts = [
                f"{r[1]} {r[2]} {r[3] or ''}" for r in batch
            ]
            try:
                vecs = self.embedder.embed_documents(texts)
            except Exception as exc:
                print(f"[kb] batch embedding failed: {exc}", flush=True)
                continue
            with sqlite3.connect(self.db_path) as db:
                for (item_id, *_), vec in zip(batch, vecs):
                    db.execute(
                        "UPDATE examples SET embedding_json = ? WHERE id = ?",
                        (json.dumps(vec), item_id),
                    )
            updated += len(batch)

        if updated:
            print(f"[kb] backfilled embeddings for {updated} examples", flush=True)
        return updated

    # ------------------------------------------------------------------
    # RAG context  (semantic if embedder available, SQL fallback otherwise)
    # ------------------------------------------------------------------

    def rag_context(self, category: str, limit: int = 4, query: str | None = None) -> RagContext:
        if self.embedder is not None and query:
            try:
                return self._semantic_rag_context(category, limit, query)
            except Exception as exc:
                print(f"[kb] semantic search failed ({exc}), falling back to SQL", flush=True)

        # ── SQL fallback ──────────────────────────────────────────────
        with sqlite3.connect(self.db_path) as db:
            positive = [
                _row_to_dict(row)
                for row in db.execute(
                    """
                    SELECT id, category, expression, hypothesis, is_negative_example, metadata_json
                    FROM examples
                    WHERE is_negative_example = 0 AND (category = ? OR ? = '')
                    ORDER BY rowid DESC
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
                    ORDER BY rowid DESC
                    LIMIT ?
                    """,
                    (limit,),
                )
            ]
        failure_patterns = self.get_failure_patterns(limit=5)
        strategy_stats = self.get_strategy_stats(category)
        return RagContext(
            positive=positive,
            negative=negative,
            failure_patterns=failure_patterns,
            strategy_stats=strategy_stats,
        )

    def _semantic_rag_context(self, category: str, limit: int, query: str) -> RagContext:
        """Cosine-similarity retrieval over stored embeddings."""
        with sqlite3.connect(self.db_path) as db:
            rows = db.execute(
                """
                SELECT id, category, expression, hypothesis,
                       is_negative_example, metadata_json, embedding_json
                FROM examples
                WHERE embedding_json IS NOT NULL
                """
            ).fetchall()

        if not rows:
            return self.rag_context(category, limit, query=None)

        ids = [r[0] for r in rows]
        is_neg = [bool(r[4]) for r in rows]
        vecs = np.array([json.loads(r[6]) for r in rows], dtype=np.float32)
        cats = [r[1] for r in rows]

        # Query vector
        q_vec = np.array(self.embedder.embed_query(query), dtype=np.float32)

        # Cosine similarity
        row_norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        row_norms[row_norms == 0] = 1.0
        q_norm_val = float(np.linalg.norm(q_vec)) or 1.0
        scores = (vecs / row_norms) @ (q_vec / q_norm_val)  # shape (N,)

        # Category affinity bonus: same category gets +0.15 boost
        affinity = np.array([0.15 if c == category else 0.0 for c in cats], dtype=np.float32)
        final_scores = scores + affinity

        pos_idxs = [i for i, neg in enumerate(is_neg) if not neg]
        neg_idxs = [i for i, neg in enumerate(is_neg) if neg]

        def top_k(idxs: list[int], k: int) -> list[dict[str, Any]]:
            ranked = sorted(idxs, key=lambda i: final_scores[i], reverse=True)[:k]
            return [_row_to_dict(rows[i][:6]) for i in ranked]

        positive = top_k(pos_idxs, limit)
        negative = top_k(neg_idxs, limit)
        failure_patterns = self.get_failure_patterns(limit=5)
        strategy_stats = self.get_strategy_stats(category)
        return RagContext(
            positive=positive,
            negative=negative,
            failure_patterns=failure_patterns,
            strategy_stats=strategy_stats,
        )

    # ------------------------------------------------------------------
    # Strategy / operator / failure stats
    # ------------------------------------------------------------------

    def record_strategy_stat(self, category: str, gate_result: str, operator_skeleton: str = '') -> None:
        with sqlite3.connect(self.db_path) as db:
            db.execute(
                "INSERT INTO strategy_stats(category, gate_result, operator_skeleton, ts) VALUES (?, ?, ?, ?)",
                (category, gate_result, operator_skeleton, int(time.time())),
            )

    def get_strategy_stats(self, category: str) -> dict:
        with sqlite3.connect(self.db_path) as db:
            row = db.execute(
                """
                SELECT COUNT(*), SUM(CASE WHEN gate_result = 'PASS' THEN 1 ELSE 0 END)
                FROM strategy_stats WHERE category = ?
                """,
                (category,),
            ).fetchone()
        attempts = int(row[0] or 0)
        wins = int(row[1] or 0)
        return {"attempts": attempts, "wins": wins, "win_rate": wins / attempts if attempts else 0.0}

    def record_failure_pattern(self, reason: str, expression: str, suggested_fix: str) -> None:
        with sqlite3.connect(self.db_path) as db:
            db.execute(
                "INSERT INTO failure_patterns(reason, expression, suggested_fix, ts) VALUES (?, ?, ?, ?)",
                (reason, expression, suggested_fix, int(time.time())),
            )

    def get_failure_patterns(self, limit: int = 10) -> list:
        with sqlite3.connect(self.db_path) as db:
            rows = db.execute(
                "SELECT reason, expression, suggested_fix FROM failure_patterns ORDER BY ts DESC LIMIT ?",
                (limit,),
            )
            return [{"reason": r[0], "expression": r[1], "suggested_fix": r[2]} for r in rows]

    def record_operator_stat(self, operator: str, category: str, passed: bool) -> None:
        with sqlite3.connect(self.db_path) as db:
            db.execute(
                "INSERT INTO operator_stats(operator, category, passed, ts) VALUES (?, ?, ?, ?)",
                (operator, category, int(passed), int(time.time())),
            )

    def get_operator_stats(self, category: str) -> dict:
        with sqlite3.connect(self.db_path) as db:
            rows = db.execute(
                "SELECT operator, COUNT(*), SUM(passed) FROM operator_stats WHERE category = ? GROUP BY operator",
                (category,),
            )
            return {
                r[0]: {
                    "attempts": int(r[1] or 0),
                    "wins": int(r[2] or 0),
                    "win_rate": int(r[2] or 0) / int(r[1] or 0) if r[1] else 0.0,
                }
                for r in rows
            }

    def upsert_market_regime(self, regime_key: str, summary: str, top_categories: list) -> None:
        with sqlite3.connect(self.db_path) as db:
            db.execute(
                "INSERT OR REPLACE INTO market_regime(regime_key, summary, top_categories_json, ts) VALUES (?, ?, ?, ?)",
                (regime_key, summary, json.dumps(top_categories), int(time.time())),
            )

    def get_market_regime(self, regime_key: str) -> dict | None:
        with sqlite3.connect(self.db_path) as db:
            row = db.execute(
                "SELECT summary, top_categories_json FROM market_regime WHERE regime_key = ?",
                (regime_key,),
            ).fetchone()
        if row is None:
            return None
        return {"summary": row[0], "top_categories": json.loads(row[1])}


def _row_to_dict(row: tuple[Any, ...]) -> dict[str, Any]:
    return {
        "id": row[0],
        "category": row[1],
        "expression": row[2],
        "hypothesis": row[3],
        "is_negative_example": bool(row[4]),
        "metadata": json.loads(row[5] or "{}"),
    }
