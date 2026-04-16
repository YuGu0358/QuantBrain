from __future__ import annotations
import math, sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

__all__ = ["BanditScheduler", "ActionStats"]

ACTION_TYPES = ["param_tune", "struct_mutation", "template_retrieval", "llm_mutation"]

_CREATE = """
CREATE TABLE IF NOT EXISTS scheduler_stats (
    action_type        TEXT PRIMARY KEY,
    total_attempts     INTEGER NOT NULL DEFAULT 0,
    total_accepted     INTEGER NOT NULL DEFAULT 0,
    total_j_improvement REAL NOT NULL DEFAULT 0.0,
    last_updated       TEXT NOT NULL DEFAULT ''
)
"""


@dataclass
class ActionStats:
    action_type: str
    total_attempts: int = 0
    total_accepted: int = 0
    total_j_improvement: float = 0.0
    last_updated: str = ""


class BanditScheduler:
    """UCB1 bandit over the 4 repair action types."""

    def __init__(self, db_path: Path):
        self._db = Path(db_path)
        self._db.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as cx:
            cx.execute(_CREATE)
            for a in ACTION_TYPES:
                cx.execute(
                    "INSERT OR IGNORE INTO scheduler_stats (action_type) VALUES (?)", (a,)
                )

    def _conn(self):
        return sqlite3.connect(self._db)

    def get_weights(self) -> dict:
        stats = {s.action_type: s for s in self._load_all()}
        n_total = sum(s.total_attempts for s in stats.values())
        scores: dict[str, float] = {}
        for a in ACTION_TYPES:
            s = stats.get(a, ActionStats(action_type=a))
            n_a = max(1, s.total_attempts)
            avg_j = s.total_j_improvement / n_a
            if n_total > 0:
                exploration = math.sqrt(2 * math.log(max(1, n_total)) / n_a)
            else:
                exploration = 0.0
            scores[a] = avg_j + exploration
        # uniform fallback when all scores are zero
        if all(v == 0.0 for v in scores.values()):
            return {a: 0.25 for a in ACTION_TYPES}
        # shift all scores positive then normalise
        min_s = min(scores.values())
        if min_s < 0:
            scores = {k: v - min_s + 1e-6 for k, v in scores.items()}
        total = sum(scores.values()) or 1.0
        return {k: v / total for k, v in scores.items()}

    def record_outcome(self, action_type: str, accepted: bool, j_improvement: float) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as cx:
            cx.execute(
                """UPDATE scheduler_stats
                   SET total_attempts = total_attempts + 1,
                       total_accepted = total_accepted + ?,
                       total_j_improvement = total_j_improvement + ?,
                       last_updated = ?
                   WHERE action_type = ?""",
                (1 if accepted else 0, j_improvement, now, action_type),
            )

    def record_batch_outcomes(self, outcomes: list[dict]) -> None:
        for o in outcomes:
            j_imp = float(o.get("j_new", 0)) - float(o.get("j_old", 0))
            self.record_outcome(o["action_type"], bool(o.get("accepted", False)), j_imp)

    def get_stats(self) -> list[dict]:
        return [
            {
                "action_type": s.action_type,
                "total_attempts": s.total_attempts,
                "total_accepted": s.total_accepted,
                "total_j_improvement": round(s.total_j_improvement, 4),
                "last_updated": s.last_updated,
            }
            for s in self._load_all()
        ]

    def _load_all(self) -> list[ActionStats]:
        with self._conn() as cx:
            rows = cx.execute(
                "SELECT action_type, total_attempts, total_accepted, total_j_improvement, last_updated "
                "FROM scheduler_stats"
            ).fetchall()
        return [ActionStats(*r) for r in rows]
