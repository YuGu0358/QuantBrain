from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any


class RepairMemory:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def add_record(self, record: dict) -> None:
        values = _record_values(record)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO repair_records (
                    record_id, timestamp, expression, symptom_tags, repair_actions,
                    accept_decision, rejection_reason, recommended_directions,
                    forbidden_directions, metrics, family_tag, notes
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    notes TEXT
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn


_JSON_FIELDS = {
    "symptom_tags",
    "repair_actions",
    "recommended_directions",
    "forbidden_directions",
    "metrics",
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
    }
    for field in _JSON_FIELDS:
        values[field] = json.dumps(values[field], ensure_ascii=False)
    return values


def _decode_row(row: sqlite3.Row) -> dict[str, Any]:
    record = dict(row)
    for field in _JSON_FIELDS:
        record[field] = _loads_json(record.get(field), [] if field != "metrics" else {})
    return record


def _loads_json(value: Any, default: Any) -> Any:
    if value in (None, ""):
        return default
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


__all__ = ["RepairMemory"]
