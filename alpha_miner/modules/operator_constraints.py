from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path


_UNSUPPORTED_OPERATOR = re.compile(
    r'(?:inaccessible\s+or\s+unknown|unknown)\s+operator\s*["\']([a-zA-Z_][a-zA-Z0-9_]*)["\']',
    re.IGNORECASE,
)


def parse_operator_list(value: str | None) -> set[str]:
    tokens = re.split(r"[\s,;]+", str(value or "").strip())
    return {token for token in (item.strip() for item in tokens) if token}


def extract_unsupported_operator(message: str | None) -> str | None:
    match = _UNSUPPORTED_OPERATOR.search(str(message or ""))
    return match.group(1) if match else None


def load_blocked_operators(path: Path | None = None) -> set[str]:
    blocked = set(parse_operator_list(os.environ.get("BRAIN_OPERATOR_DENYLIST")))
    raw_path = path or _env_constraint_path()
    if raw_path is None:
        return blocked
    try:
        payload = json.loads(raw_path.read_text(encoding="utf-8"))
    except Exception:
        return blocked
    blocked.update(str(item).strip() for item in payload.get("blockedOperators") or [] if str(item).strip())
    return blocked


def persist_blocked_operator(
    operator: str,
    reason: str,
    *,
    path: Path | None = None,
) -> dict:
    operator = str(operator or "").strip()
    if not operator:
        return {"blockedOperators": [], "history": []}

    payload = {"blockedOperators": [], "history": []}
    target = path or _env_constraint_path()
    if target is not None:
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                payload = {"blockedOperators": [], "history": []}
        except Exception:
            payload = {"blockedOperators": [], "history": []}
        blocked = {str(item).strip() for item in payload.get("blockedOperators") or [] if str(item).strip()}
        blocked.add(operator)
        payload["blockedOperators"] = sorted(blocked)
        history = list(payload.get("history") or [])
        history.append(
            {
                "operator": operator,
                "reason": str(reason or "").strip(),
                "at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            }
        )
        payload["history"] = history[-20:]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def _env_constraint_path() -> Path | None:
    raw = str(os.environ.get("BRAIN_OPERATOR_DENYLIST_PATH") or "").strip()
    return Path(raw) if raw else None
