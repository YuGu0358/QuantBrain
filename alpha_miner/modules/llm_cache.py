from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .common import sha256_json, write_json, read_json


@dataclass(frozen=True)
class CacheResult:
    key: str
    hit: bool
    payload: dict[str, Any] | None
    path: Path


class LLMCache:
    def __init__(self, root: Path):
        self.root = root

    def key_for(self, request_payload: dict[str, Any]) -> str:
        return sha256_json(request_payload)

    def path_for_key(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, request_payload: dict[str, Any]) -> CacheResult:
        key = self.key_for(request_payload)
        path = self.path_for_key(key)
        payload = read_json(path)
        return CacheResult(key=key, hit=payload is not None, payload=payload, path=path)

    def put(self, request_payload: dict[str, Any], response_payload: dict[str, Any]) -> CacheResult:
        key = self.key_for(request_payload)
        path = self.path_for_key(key)
        payload = {"request": request_payload, "response": response_payload}
        write_json(path, payload)
        return CacheResult(key=key, hit=False, payload=payload, path=path)
