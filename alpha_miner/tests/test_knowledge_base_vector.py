"""Tests for KnowledgeBase vector (semantic) search path."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from alpha_miner.modules.m1_knowledge_base import KnowledgeBase, RagContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_embedder(dim: int = 8):
    """Return a mock embedder that produces deterministic vectors."""
    call_count = 0

    def embed_query(text: str):
        nonlocal call_count
        call_count += 1
        rng = np.random.default_rng(abs(hash(text)) % (2**31))
        vec = rng.random(dim).tolist()
        return vec

    def embed_documents(texts):
        return [embed_query(t) for t in texts]

    emb = MagicMock()
    emb.embed_query.side_effect = embed_query
    emb.embed_documents.side_effect = embed_documents
    return emb


# ---------------------------------------------------------------------------
# Schema migration — embedding_json column
# ---------------------------------------------------------------------------

def test_embedding_column_created(tmp_path):
    """embedding_json column must exist after init."""
    import sqlite3
    kb = KnowledgeBase(tmp_path / "kb.db")
    with sqlite3.connect(tmp_path / "kb.db") as conn:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(examples)")}
    assert "embedding_json" in cols


def test_migration_adds_column_to_existing_db(tmp_path):
    """If an old DB without embedding_json exists, migration adds the column."""
    import sqlite3
    # Create DB without embedding_json
    db_path = tmp_path / "old.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE examples (
              id TEXT PRIMARY KEY,
              category TEXT,
              expression TEXT NOT NULL,
              hypothesis TEXT,
              is_negative_example INTEGER NOT NULL DEFAULT 0,
              metadata_json TEXT NOT NULL DEFAULT '{}'
            )
        """)
        conn.execute("INSERT INTO examples VALUES ('x1','CAT','expr','hyp',0,'{}')")
    # Init should migrate without error
    kb = KnowledgeBase(db_path)
    with sqlite3.connect(db_path) as conn:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(examples)")}
    assert "embedding_json" in cols


# ---------------------------------------------------------------------------
# upsert_example — embedding stored once, not re-computed
# ---------------------------------------------------------------------------

def test_upsert_stores_embedding(tmp_path):
    """After upsert with embedder, embedding_json is non-null."""
    import sqlite3
    emb = _make_embedder()
    kb = KnowledgeBase(tmp_path / "kb.db", embedder=emb)
    kb.upsert_example("e1", "rank(returns)", "MOMENTUM", "test hyp", False)
    with sqlite3.connect(tmp_path / "kb.db") as conn:
        row = conn.execute("SELECT embedding_json FROM examples WHERE id='e1'").fetchone()
    assert row[0] is not None
    vec = json.loads(row[0])
    assert len(vec) == 8


def test_upsert_reuses_existing_embedding(tmp_path):
    """Second upsert for same id must NOT call embed_query again."""
    emb = _make_embedder()
    kb = KnowledgeBase(tmp_path / "kb.db", embedder=emb)
    kb.upsert_example("e1", "rank(returns)", "MOMENTUM", "test hyp", False)
    first_call_count = emb.embed_query.call_count
    # Second upsert — same id, same content
    kb.upsert_example("e1", "rank(returns)", "MOMENTUM", "updated hyp", False)
    assert emb.embed_query.call_count == first_call_count  # no extra API call


def test_upsert_without_embedder_skips_embedding(tmp_path):
    """Without embedder, embedding_json stays NULL."""
    import sqlite3
    kb = KnowledgeBase(tmp_path / "kb.db")
    kb.upsert_example("e1", "rank(returns)", "MOMENTUM", "hyp", False)
    with sqlite3.connect(tmp_path / "kb.db") as conn:
        row = conn.execute("SELECT embedding_json FROM examples WHERE id='e1'").fetchone()
    assert row[0] is None


# ---------------------------------------------------------------------------
# backfill_embeddings
# ---------------------------------------------------------------------------

def test_backfill_fills_missing_embeddings(tmp_path):
    """backfill_embeddings should embed rows that have NULL embedding_json."""
    import sqlite3
    # Insert rows without embedder first
    kb_no_emb = KnowledgeBase(tmp_path / "kb.db")
    kb_no_emb.upsert_example("e1", "rank(returns)", "MOMENTUM", "h1", False)
    kb_no_emb.upsert_example("e2", "ts_mean(close, 21)", "MOMENTUM", "h2", False)
    # Confirm no embeddings yet
    with sqlite3.connect(tmp_path / "kb.db") as conn:
        nulls = conn.execute("SELECT COUNT(*) FROM examples WHERE embedding_json IS NULL").fetchone()[0]
    assert nulls == 2

    # Now attach embedder and backfill
    emb = _make_embedder()
    kb = KnowledgeBase(tmp_path / "kb.db", embedder=emb)
    updated = kb.backfill_embeddings()
    assert updated == 2
    with sqlite3.connect(tmp_path / "kb.db") as conn:
        nulls_after = conn.execute("SELECT COUNT(*) FROM examples WHERE embedding_json IS NULL").fetchone()[0]
    assert nulls_after == 0


def test_backfill_returns_zero_without_embedder(tmp_path):
    kb = KnowledgeBase(tmp_path / "kb.db")
    assert kb.backfill_embeddings() == 0


def test_backfill_skips_already_embedded_rows(tmp_path):
    """backfill_embeddings should not re-embed rows that already have vectors."""
    emb = _make_embedder()
    kb = KnowledgeBase(tmp_path / "kb.db", embedder=emb)
    kb.upsert_example("e1", "rank(returns)", "MOMENTUM", "h1", False)
    count_before = emb.embed_documents.call_count
    kb.backfill_embeddings()
    # No new rows to backfill → embed_documents NOT called again
    assert emb.embed_documents.call_count == count_before


# ---------------------------------------------------------------------------
# rag_context — semantic path
# ---------------------------------------------------------------------------

def _populate_kb(kb: KnowledgeBase, n: int = 6):
    for i in range(n):
        neg = i % 2 == 0
        kb.upsert_example(
            f"ex{i}",
            f"rank(ts_mean(returns, {21 + i * 5}))",
            "MOMENTUM",
            f"Hypothesis {i}",
            is_negative_example=neg,
        )


def test_rag_context_uses_semantic_path_when_query_provided(tmp_path):
    emb = _make_embedder()
    kb = KnowledgeBase(tmp_path / "kb.db", embedder=emb)
    _populate_kb(kb, 6)

    ctx = kb.rag_context("MOMENTUM", limit=2, query="momentum signal with smoothing")
    assert isinstance(ctx, RagContext)
    assert len(ctx.positive) <= 2
    assert len(ctx.negative) <= 2
    # semantic path was used → embed_query called (at least once for upserts + once for query)
    assert emb.embed_query.call_count > 0
    assert kb.last_rag_mode == "semantic"
    assert kb.last_rag_error is None


def test_rag_context_falls_back_to_sql_without_query(tmp_path):
    """When query=None, fall back to SQL regardless of embedder presence."""
    emb = _make_embedder()
    kb = KnowledgeBase(tmp_path / "kb.db", embedder=emb)
    _populate_kb(kb, 4)
    query_calls_before = emb.embed_query.call_count
    ctx = kb.rag_context("MOMENTUM", limit=2, query=None)
    # No new embed_query calls for the rag_context call itself
    assert emb.embed_query.call_count == query_calls_before
    assert isinstance(ctx, RagContext)


def test_rag_context_falls_back_to_sql_without_embedder(tmp_path):
    kb = KnowledgeBase(tmp_path / "kb.db")
    _populate_kb(kb, 4)
    ctx = kb.rag_context("MOMENTUM", limit=2, query="some query")
    assert isinstance(ctx, RagContext)
    assert len(ctx.positive) + len(ctx.negative) <= 4
    assert kb.last_rag_mode == "sql_no_embedder"
    assert kb.last_rag_error is None


def test_semantic_rag_context_respects_limit(tmp_path):
    emb = _make_embedder()
    kb = KnowledgeBase(tmp_path / "kb.db", embedder=emb)
    _populate_kb(kb, 10)
    ctx = kb.rag_context("MOMENTUM", limit=3, query="momentum returns")
    assert len(ctx.positive) <= 3
    assert len(ctx.negative) <= 3


def test_semantic_rag_context_category_affinity(tmp_path):
    """Results should include the queried category when enough rows exist."""
    emb = _make_embedder()
    kb = KnowledgeBase(tmp_path / "kb.db", embedder=emb)
    # Insert both MOMENTUM and VALUE positives
    for i in range(3):
        kb.upsert_example(f"m{i}", f"rank(returns_{i})", "MOMENTUM", f"mom{i}", False)
        kb.upsert_example(f"v{i}", f"rank(book_to_price_{i})", "VALUE", f"val{i}", False)

    ctx = kb.rag_context("MOMENTUM", limit=3, query="momentum signal")
    # At least one result should be MOMENTUM category
    categories = [r["category"] for r in ctx.positive]
    assert "MOMENTUM" in categories


def test_rag_context_semantic_fallback_on_no_embeddings(tmp_path):
    """If embedder present but no rows have embeddings yet, falls back to SQL."""
    emb = _make_embedder()
    # Insert rows WITHOUT embedder so no embeddings stored
    kb_no_emb = KnowledgeBase(tmp_path / "kb.db")
    _populate_kb(kb_no_emb, 4)
    # Now use KB with embedder — no embeddings in DB
    kb = KnowledgeBase(tmp_path / "kb.db", embedder=emb)
    ctx = kb.rag_context("MOMENTUM", limit=2, query="test query")
    # Should not raise; SQL fallback or empty result
    assert isinstance(ctx, RagContext)


# ---------------------------------------------------------------------------
# rag_context includes failure_patterns and strategy_stats
# ---------------------------------------------------------------------------

def test_rag_context_always_includes_enriched_fields(tmp_path):
    emb = _make_embedder()
    kb = KnowledgeBase(tmp_path / "kb.db", embedder=emb)
    _populate_kb(kb, 4)
    kb.record_failure_pattern("LOW_SHARPE", "rank(close)", "add neutralization")
    kb.record_strategy_stat("MOMENTUM", "PASS", "rank(ts_rank(X,N))")

    ctx = kb.rag_context("MOMENTUM", limit=2, query="momentum signal")
    assert len(ctx.failure_patterns) == 1
    assert ctx.strategy_stats["attempts"] == 1
