from __future__ import annotations

import json
import math
import sqlite3
import uuid
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any


class RepairMemory:
    def __init__(self, db_path: Path, embedder: Any | None = None):
        self.db_path = Path(db_path)
        self.embedder = embedder
        self.last_retrieval_mode = "uninitialized"
        self.last_retrieval_error: str | None = None
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def set_embedder(self, embedder: Any | None) -> None:
        self.embedder = embedder

    def add_record(self, record: dict) -> None:
        record = dict(record or {})
        if self.embedder is not None and not record.get("embedding_json"):
            record["embedding_json"] = self._embed_text(self._record_text(record))
        values = _record_values(record)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO repair_records (
                    record_id, timestamp, expression, symptom_tags, repair_actions,
                    accept_decision, rejection_reason, recommended_directions,
                    forbidden_directions, metrics, family_tag, notes,
                    math_profile, economic_profile, repair_delta, outcome_score,
                    platform_outcome, embedding_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    values["record_id"],
                    values["timestamp"],
                    values["expression"],
                    values["symptom_tags"],
                    values["repair_actions"],
                    values["accept_decision"],
                    values["rejection_reason"],
                    values["recommended_directions"],
                    values["forbidden_directions"],
                    values["metrics"],
                    values["family_tag"],
                    values["notes"],
                    values["math_profile"],
                    values["economic_profile"],
                    values["repair_delta"],
                    values["outcome_score"],
                    values["platform_outcome"],
                    values["embedding_json"],
                ),
            )

    def get_forbidden_for_symptoms(self, symptom_tags: list[str], limit: int = 10) -> list[str]:
        matching_records = self._records_for_symptoms("rejected", symptom_tags, limit)
        flattened: list[str] = []
        for record in matching_records:
            flattened.extend(str(item) for item in record.get("forbidden_directions", []))
        return flattened

    def get_positive_for_symptoms(self, symptom_tags: list[str], limit: int = 5) -> list[dict]:
        matching_records = self._records_for_symptoms("accepted", symptom_tags, limit)
        return [
            {
                "expression": record.get("expression", ""),
                "recommended_directions": record.get("recommended_directions", []),
            }
            for record in matching_records
        ]

    def retrieve(
        self,
        symptom_tags: list[str],
        expression: str,
        family_tag: str | None = None,
        topk: int = 5,
        math_profile: dict | None = None,
        economic_profile: dict | None = None,
    ) -> dict:
        records = self.get_recent(limit=10000)
        semantic_scores, retrieval_mode, retrieval_error = self._semantic_scores(records, expression, symptom_tags)
        self.last_retrieval_mode = retrieval_mode
        self.last_retrieval_error = retrieval_error
        scored_records = [
            {
                **record,
                "symptom_tags": _string_list(record.get("symptom_tags", [])),
                "semantic_score": round(float(semantic_scores.get(str(record.get("record_id")), 0.0)), 6),
                "score": _retrieval_score(record, symptom_tags, family_tag, math_profile, economic_profile)
                + float(semantic_scores.get(str(record.get("record_id")), 0.0)) * 0.45,
            }
            for record in records
        ]
        scored_records.sort(key=lambda record: record["score"], reverse=True)

        positive = [
            record
            for record in scored_records
            if record.get("accept_decision") == "accepted"
        ][:topk]
        negative = [
            record
            for record in scored_records
            if record.get("accept_decision") == "rejected"
        ][:topk]
        saturated_themes = _saturated_themes(records, family_tag, economic_profile)
        saturated_math_signatures = _saturated_math_signatures(records, family_tag, math_profile)

        return {
            "positive": positive,
            "negative": negative,
            "forbidden_directions": _flatten_unique(negative, "forbidden_directions"),
            "recommended_directions": _flatten_unique(positive, "recommended_directions"),
            "theme_saturated": bool(saturated_themes),
            "math_saturated": bool(saturated_math_signatures),
            "saturated_themes": saturated_themes,
            "saturated_math_signatures": saturated_math_signatures,
        }

    def family_saturation(self, family_tag: str, threshold: int = 5) -> bool:
        if not family_tag:
            return False
        accepted_count = 0
        for record in self.get_recent(limit=10000):
            if record.get("accept_decision") == "accepted" and record.get("family_tag") == family_tag:
                accepted_count += 1
                if accepted_count >= threshold:
                    return True
        return False

    def get_recent(self, limit: int = 20) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM repair_records
                ORDER BY timestamp DESC, rowid DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
        return [_decode_row(row) for row in rows]

    def _records_for_symptoms(self, accept_decision: str, symptom_tags: list[str], limit: int) -> list[dict]:
        target_tags = set(symptom_tags)
        if not target_tags:
            return []
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM repair_records
                WHERE accept_decision = ?
                ORDER BY timestamp DESC, rowid DESC
                """,
                (accept_decision,),
            ).fetchall()

        matches = []
        for row in rows:
            record = _decode_row(row)
            if target_tags.intersection(record.get("symptom_tags", [])):
                matches.append(record)
            if len(matches) >= limit:
                break
        return matches

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS repair_records (
                    record_id TEXT PRIMARY KEY,
                    timestamp TEXT,
                    expression TEXT,
                    symptom_tags TEXT,
                    repair_actions TEXT,
                    accept_decision TEXT,
                    rejection_reason TEXT,
                    recommended_directions TEXT,
                    forbidden_directions TEXT,
                    metrics TEXT,
                    family_tag TEXT,
                    notes TEXT,
                    math_profile TEXT,
                    economic_profile TEXT,
                    repair_delta TEXT,
                    outcome_score REAL,
                    platform_outcome TEXT
                )
                """
            )
            existing = {
                row[1]
                for row in conn.execute("PRAGMA table_info(repair_records)").fetchall()
            }
            migrations = {
                "math_profile": "ALTER TABLE repair_records ADD COLUMN math_profile TEXT",
                "economic_profile": "ALTER TABLE repair_records ADD COLUMN economic_profile TEXT",
                "repair_delta": "ALTER TABLE repair_records ADD COLUMN repair_delta TEXT",
                "outcome_score": "ALTER TABLE repair_records ADD COLUMN outcome_score REAL",
                "platform_outcome": "ALTER TABLE repair_records ADD COLUMN platform_outcome TEXT",
                "embedding_json": "ALTER TABLE repair_records ADD COLUMN embedding_json TEXT",
            }
            for column, statement in migrations.items():
                if column not in existing:
                    conn.execute(statement)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _semantic_scores(
        self,
        records: list[dict[str, Any]],
        expression: str,
        symptom_tags: list[str],
    ) -> tuple[dict[str, float], str, str | None]:
        if not records:
            return {}, "empty", None
        if self.embedder is None:
            return {}, "scored_no_embedder", None
        try:
            self._ensure_embeddings(records)
            available_vectors = [
                _float_list(record.get("embedding_json"))
                for record in records
                if _float_list(record.get("embedding_json"))
            ]
            if not available_vectors:
                return {}, "scored_no_embeddings", None
            query_text = f"{expression} {' '.join(_string_list(symptom_tags))}".strip()
            query_embedding = self._embed_text(query_text)
            if not query_embedding:
                return {}, "scored_embedder_error", "query_embedding_unavailable"
            scores: dict[str, float] = {}
            for record in records:
                vector = _float_list(record.get("embedding_json"))
                if not vector:
                    continue
                scores[str(record.get("record_id"))] = _cosine_similarity(query_embedding, vector)
            return scores, "semantic", None
        except Exception as exc:
            return {}, "scored_embedder_error", str(exc)

    def _ensure_embeddings(self, records: list[dict[str, Any]]) -> None:
        if self.embedder is None:
            return
        missing = [record for record in records if not _float_list(record.get("embedding_json"))]
        if not missing:
            return
        texts = [self._record_text(record) for record in missing]
        vectors = self.embedder.embed_documents(texts)
        with self._connect() as conn:
            for record, vector in zip(missing, vectors):
                record["embedding_json"] = _float_list(vector)
                conn.execute(
                    "UPDATE repair_records SET embedding_json = ? WHERE record_id = ?",
                    (json.dumps(record["embedding_json"], ensure_ascii=False), record.get("record_id")),
                )

    def _embed_text(self, text: str) -> list[float]:
        if self.embedder is None:
            return []
        try:
            return _float_list(self.embedder.embed_query(text))
        except Exception:
            return []

    def _record_text(self, record: dict[str, Any]) -> str:
        return " ".join(
            part
            for part in [
                str(record.get("expression") or ""),
                " ".join(_string_list(record.get("symptom_tags", []))),
                " ".join(_string_list(record.get("recommended_directions", []))),
                " ".join(_string_list(record.get("forbidden_directions", []))),
                str((record.get("economic_profile") or {}).get("theme") or ""),
            ]
            if part
        ).strip()


_JSON_FIELDS = {
    "symptom_tags",
    "repair_actions",
    "recommended_directions",
    "forbidden_directions",
    "metrics",
    "math_profile",
    "economic_profile",
    "repair_delta",
    "platform_outcome",
    "embedding_json",
}

_JSON_DEFAULTS = {
    "symptom_tags": [],
    "repair_actions": [],
    "recommended_directions": [],
    "forbidden_directions": [],
    "metrics": {},
    "math_profile": {},
    "economic_profile": {},
    "repair_delta": {},
    "platform_outcome": {},
    "embedding_json": [],
}


def _record_values(record: dict[str, Any]) -> dict[str, str]:
    timestamp = record.get("timestamp") or datetime.now(timezone.utc).isoformat()
    values: dict[str, Any] = {
        "record_id": record.get("record_id") or str(uuid.uuid4()),
        "timestamp": timestamp,
        "expression": record.get("expression", ""),
        "symptom_tags": record.get("symptom_tags", []),
        "repair_actions": record.get("repair_actions", []),
        "accept_decision": record.get("accept_decision", ""),
        "rejection_reason": record.get("rejection_reason", ""),
        "recommended_directions": record.get("recommended_directions", []),
        "forbidden_directions": record.get("forbidden_directions", []),
        "metrics": record.get("metrics", {}),
        "family_tag": record.get("family_tag", ""),
        "notes": record.get("notes", ""),
        "math_profile": record.get("math_profile", {}),
        "economic_profile": record.get("economic_profile", {}),
        "repair_delta": record.get("repair_delta", {}),
        "outcome_score": _float_or_default(record.get("outcome_score"), 0.0),
        "platform_outcome": record.get("platform_outcome", {}),
        "embedding_json": _float_list(record.get("embedding_json")),
    }
    for field in _JSON_FIELDS:
        values[field] = json.dumps(values[field], ensure_ascii=False)
    return values


def _decode_row(row: sqlite3.Row) -> dict[str, Any]:
    record = dict(row)
    for field in _JSON_FIELDS:
        record[field] = _loads_json(record.get(field), _JSON_DEFAULTS[field])
    record["outcome_score"] = _float_or_default(record.get("outcome_score"), 0.0)
    return record


def _loads_json(value: Any, default: Any) -> Any:
    if value in (None, ""):
        return default
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


def _retrieval_score(
    record: dict[str, Any],
    query_tags: list[str],
    family_tag: str | None,
    math_profile: dict | None = None,
    economic_profile: dict | None = None,
) -> float:
    overlap_score = _symptom_overlap_score(
        _string_list(record.get("symptom_tags", [])),
        _string_list(query_tags),
    )
    family_bonus = 0.3 if family_tag and record.get("family_tag") == family_tag else 0.0
    economic_bonus = 0.0
    if economic_profile and isinstance(record.get("economic_profile"), dict):
        query_theme = str(economic_profile.get("theme") or "")
        record_theme = str(record.get("economic_profile", {}).get("theme") or "")
        if query_theme and query_theme == record_theme:
            economic_bonus = 0.35
    math_bonus = 0.35 * _math_profile_similarity(record.get("math_profile"), math_profile)
    outcome_bonus = max(min(_float_or_default(record.get("outcome_score"), 0.0), 2.0), -2.0) * 0.03
    age_penalty = min(_days_since(record.get("timestamp")) * 0.01, 0.3)
    return overlap_score + family_bonus + economic_bonus + math_bonus + outcome_bonus - age_penalty


def _math_profile_similarity(left: Any, right: Any) -> float:
    if not isinstance(left, dict) or not isinstance(right, dict) or not left or not right:
        return 0.0
    scores = []
    for key in ("operators", "fields", "windows", "group_fields"):
        scores.append(_jaccard(left.get(key), right.get(key)))
    if left.get("dominant_structure") and left.get("dominant_structure") == right.get("dominant_structure"):
        scores.append(1.0)
    return sum(scores) / len(scores) if scores else 0.0


def _jaccard(left: Any, right: Any) -> float:
    left_set = {str(item) for item in (left or [])}
    right_set = {str(item) for item in (right or [])}
    if not left_set and not right_set:
        return 0.0
    union = left_set.union(right_set)
    if not union:
        return 0.0
    return len(left_set.intersection(right_set)) / len(union)


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _float_or_default(value: Any, default: float) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _symptom_overlap_score(record_tags: list[str], query_tags: list[str]) -> float:
    if not record_tags or not query_tags:
        return 0.0
    denominator = max(len(set(record_tags)), len(set(query_tags)))
    if denominator == 0:
        return 0.0
    return len(set(record_tags).intersection(query_tags)) / denominator


def _days_since(timestamp: Any) -> float:
    if not timestamp:
        return 0.0
    try:
        parsed = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
    except ValueError:
        return 0.0
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    elapsed = datetime.now(timezone.utc) - parsed.astimezone(timezone.utc)
    return max(elapsed.total_seconds() / 86400, 0.0)


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        decoded = _loads_json(value, None)
        if isinstance(decoded, list):
            value = decoded
        else:
            return [value] if value else []
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item not in (None, "")]


def _float_list(value: Any) -> list[float]:
    if isinstance(value, str):
        decoded = _loads_json(value, None)
        if isinstance(decoded, list):
            value = decoded
        else:
            return []
    if not isinstance(value, list):
        return []
    floats: list[float] = []
    for item in value:
        try:
            floats.append(float(item))
        except (TypeError, ValueError):
            return []
    return floats


def _flatten_unique(records: list[dict[str, Any]], field: str) -> list[str]:
    values: list[str] = []
    for record in records:
        values.extend(_string_list(record.get(field, [])))
    return list(dict.fromkeys(value for value in values if value))


def _saturated_themes(
    records: list[dict[str, Any]],
    family_tag: str | None,
    economic_profile: dict[str, Any] | None,
    threshold: int = 3,
) -> list[str]:
    query_theme = str((economic_profile or {}).get("theme") or "").strip()
    if not query_theme:
        return []
    accepted = [
        record for record in records
        if record.get("accept_decision") == "accepted"
        and (not family_tag or record.get("family_tag") == family_tag)
        and str((record.get("economic_profile") or {}).get("theme") or "").strip() == query_theme
    ]
    return [query_theme] if len(accepted) >= threshold else []


def _saturated_math_signatures(
    records: list[dict[str, Any]],
    family_tag: str | None,
    math_profile: dict[str, Any] | None,
    threshold: int = 3,
) -> list[str]:
    query_signature = str((math_profile or {}).get("family_signature") or "").strip()
    if not query_signature:
        return []
    accepted = [
        record for record in records
        if record.get("accept_decision") == "accepted"
        and (not family_tag or record.get("family_tag") == family_tag)
        and str((record.get("math_profile") or {}).get("family_signature") or "").strip() == query_signature
    ]
    return [query_signature] if len(accepted) >= threshold else []


__all__ = ["RepairMemory"]
