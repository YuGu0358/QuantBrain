from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .common import PACKAGE_ROOT


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml

        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    except ModuleNotFoundError:
        return _parse_simple_yaml(path.read_text(encoding="utf-8"))


def load_config(path: Path | None = None) -> dict[str, Any]:
    return load_yaml(path or PACKAGE_ROOT / "config" / "config.yaml")


def load_taxonomy(path: Path | None = None) -> dict[str, Any]:
    return load_yaml(path or PACKAGE_ROOT / "config" / "taxonomy.yaml")


def load_operator_config(path: Path | None = None) -> dict[str, Any]:
    return load_yaml(path or PACKAGE_ROOT / "config" / "brain_operators.yaml")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    """Small fallback parser for this repo's config shape when PyYAML is absent."""

    result: dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(-1, result)]
    last_key_at_indent: dict[int, str] = {}
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if line.startswith("- "):
            if not isinstance(parent, list):
                key = last_key_at_indent.get(stack[-1][0])
                if key and isinstance(stack[-2][1], dict):
                    stack[-2][1][key] = []
                    parent = stack[-2][1][key]
                    stack[-1] = (stack[-1][0], parent)
            parent.append(_coerce_scalar(line[2:].strip()))
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        if value == "":
            parent[key] = {}
            last_key_at_indent[indent] = key
            stack.append((indent, parent[key]))
        else:
            parent[key] = _coerce_scalar(value.split("#", 1)[0].strip())
            last_key_at_indent[indent] = key
    return result


def _coerce_scalar(value: str) -> Any:
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value.strip('"').strip("'")
