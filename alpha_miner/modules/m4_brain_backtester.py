from __future__ import annotations

import base64
import json
import math
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .common import append_jsonl, sha256_json, write_json


API_ROOT = "https://api.worldquantbrain.com"


@dataclass
class BacktestResult:
    alpha_id: str | None
    expression: str
    period: str
    status: str
    sharpe: float | None
    fitness: float | None
    turnover: float | None
    pnl_path: str | None
    net_sharpe: float | None
    raw_path: str
    simulation_id: str | None = None
    has_daily_pnl: bool = False
    test_sharpe: float | None = None


@dataclass
class BrainCheckResult:
    alpha_id: str
    status: str
    passed: bool | None
    max_abs_correlation: float | None
    raw_path: str


class QuotaWaiting(Exception):
    pass


class RateLimiter:
    def __init__(self, max_per_minute: int = 8, max_per_day: int = 500):
        self.max_per_minute = max_per_minute
        self.max_per_day = max_per_day
        self.minute_events: list[float] = []
        self.day_count = 0
        self.day_epoch = time.strftime("%Y-%m-%d")
        self._lock = threading.Lock()

    def wait(self) -> None:
        """Thread-safe rate limiter: sleep outside the lock to avoid blocking other threads."""
        while True:
            with self._lock:
                today = time.strftime("%Y-%m-%d")
                if today != self.day_epoch:
                    self.day_epoch = today
                    self.day_count = 0
                if self.day_count >= self.max_per_day:
                    raise QuotaWaiting("Daily BRAIN simulation quota exhausted.")
                now = time.time()
                self.minute_events = [e for e in self.minute_events if now - e < 60]
                if len(self.minute_events) < self.max_per_minute:
                    self.minute_events.append(time.time())
                    self.day_count += 1
                    return
                sleep_for = 60 - (now - self.minute_events[0])
            time.sleep(max(0.1, sleep_for))


class BrainBacktester:
    def __init__(self, output_dir: Path, config: dict[str, Any]):
        self.output_dir = output_dir
        self.config = config
        brain_cfg = config.get("brain", {})
        self.rate_limiter = RateLimiter(
            max_per_minute=int(brain_cfg.get("rate_limit_per_minute", 8)),
            max_per_day=int(brain_cfg.get("rate_limit_per_day", 500)),
        )
        self.poll_timeout_seconds = int(brain_cfg.get("poll_timeout_seconds", 900))
        self.poll_interval_seconds = float(brain_cfg.get("poll_interval_seconds", 5))
        self.correlation_threshold = float(config.get("pool", {}).get("brain_check_correlation_threshold", 0.7))
        self._cookie: str | None = None
        self._cookie_expires_at = 0.0
        self.progress_path = output_dir / "progress.jsonl"
        self.snapshot_dir = output_dir / "backtest_snapshots"
        self.pnl_dir = output_dir / "pnl_series"
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.pnl_dir.mkdir(parents=True, exist_ok=True)

    def submit_alpha(self, expression: str, period: str = "IS", settings: dict[str, Any] | None = None) -> BacktestResult:
        self.rate_limiter.wait()
        if not os.environ.get("WQB_EMAIL") or not os.environ.get("WQB_PASSWORD"):
            return self._mock_result(expression, period)
        cookie = self._authenticate()
        payload = {
            "type": "REGULAR",
            "settings": {
                "instrumentType": "EQUITY",
                "region": "USA",
                "universe": "TOP3000",
                "delay": 1,
                "decay": 4,
                "neutralization": "INDUSTRY",
                "truncation": 0.08,
                "pasteurization": "ON",
                "unitHandling": "VERIFY",
                "nanHandling": "OFF",
                "language": "FASTEXPR",
                "visualization": False,
                **(settings or {}),
            },
            "regular": expression,
        }
        snapshot_path = self.snapshot_dir / f"{sha256_json({'expression': expression, 'period': period, 'settings': payload['settings']})}.json"
        append_jsonl(self.progress_path, {"stage": "submitted", "period": period, "expression": expression})
        status, headers, response_body = self._request_json("POST", "/simulations", cookie, payload=payload, timeout=60)
        simulation_id = extract_simulation_id(headers.get("Location"))
        snapshot = {
            "request": payload,
            "period": period,
            "submitStatus": status,
            "submitHeaders": headers,
            "submitBody": response_body,
            "simulationId": simulation_id,
        }
        if status == 429:
            write_json(snapshot_path, snapshot)
            raise QuotaWaiting("BRAIN returned HTTP 429 while submitting simulation.")
        if status >= 400:
            snapshot["status"] = f"failed_http_{status}"
            write_json(snapshot_path, snapshot)
            return BacktestResult(None, expression, period, snapshot["status"], None, None, None, None, None, str(snapshot_path), simulation_id)

        poll_payload = self._poll_simulation(cookie, headers.get("Location") or (f"/simulations/{simulation_id}" if simulation_id else None))
        alpha_id = poll_payload.get("alpha") if isinstance(poll_payload, dict) else None
        alpha_payload = self.get_alpha(alpha_id, cookie=cookie) if isinstance(alpha_id, str) else None
        snapshot.update({"poll": poll_payload, "alpha": alpha_payload})
        series_path = None
        series = find_series({"simulation": poll_payload, "alpha": alpha_payload})
        if series:
            series_path = self.pnl_dir / f"{alpha_id or simulation_id or sha256_json(payload)}.json"
            write_json(series_path, {"pnl": series, "source": "brain_api"})
        write_json(snapshot_path, snapshot)

        # Prefer alpha-level IS metrics; fall back to poll payload when no alpha was created
        alpha_is = alpha_payload.get("is", {}) if isinstance(alpha_payload, dict) else {}
        poll_is = poll_payload.get("is", {}) if isinstance(poll_payload, dict) else {}
        metrics = alpha_is if alpha_is else poll_is
        sharpe = safe_float(metrics.get("sharpe")) or safe_float(poll_payload.get("sharpe") if isinstance(poll_payload, dict) else None)
        turnover = safe_float(metrics.get("turnover")) or safe_float(poll_payload.get("turnover") if isinstance(poll_payload, dict) else None)
        fitness = safe_float(metrics.get("fitness")) or safe_float(poll_payload.get("fitness") if isinstance(poll_payload, dict) else None)
        # OOS (test period) Sharpe — BRAIN exposes alpha.test.sharpe separately from IS
        alpha_test = (alpha_payload.get("test") or {}) if isinstance(alpha_payload, dict) else {}
        test_sharpe = safe_float(alpha_test.get("sharpe"))
        append_jsonl(
            self.progress_path,
            {
                "stage": "backtest_completed",
                "period": period,
                "alpha_id": alpha_id,
                "has_daily_pnl": bool(series_path),
                "sharpe": sharpe,
                "fitness": fitness,
                "turnover": turnover,
            },
        )
        return BacktestResult(
            alpha_id=alpha_id,
            expression=expression,
            period=period,
            status="completed" if alpha_id else "incomplete",
            sharpe=sharpe,
            fitness=fitness,
            turnover=turnover,
            pnl_path=str(series_path) if series_path else None,
            net_sharpe=estimate_net_sharpe(sharpe, turnover),
            raw_path=str(snapshot_path),
            simulation_id=simulation_id,
            has_daily_pnl=bool(series_path),
            test_sharpe=test_sharpe,
        )

    def get_alpha(self, alpha_id: str | None, cookie: str | None = None) -> dict[str, Any] | None:
        if not alpha_id:
            return None
        active_cookie = cookie or self._authenticate()
        status, _, payload = self._request_json("GET", f"/alphas/{alpha_id}", active_cookie, timeout=60)
        if status >= 400:
            return {"error": f"HTTP {status}", "payload": payload}
        return payload if isinstance(payload, dict) else {"payload": payload}

    def check_alpha(self, alpha_id: str) -> BrainCheckResult:
        if not os.environ.get("WQB_EMAIL") or not os.environ.get("WQB_PASSWORD") or alpha_id.startswith("mock-"):
            raw_path = self.snapshot_dir / f"check_{alpha_id}.json"
            payload = {"mock": True, "alpha_id": alpha_id, "max_abs_correlation": 0.0}
            write_json(raw_path, payload)
            return BrainCheckResult(alpha_id, "mock", True, 0.0, str(raw_path))
        cookie = self._authenticate()
        status, headers, payload = self._request_json("POST", f"/alphas/{alpha_id}/check", cookie, payload={}, timeout=60)
        if status == 429:
            raise QuotaWaiting("BRAIN returned HTTP 429 while running alpha check.")
        if status < 400:
            location = headers.get("Location")
            if location:
                payload = self._poll_check(cookie, location)
        max_corr = extract_max_correlation(payload)
        passed = None if max_corr is None else max_corr < self.correlation_threshold
        raw_path = self.snapshot_dir / f"check_{alpha_id}.json"
        write_json(raw_path, {"status": status, "headers": headers, "payload": payload, "max_abs_correlation": max_corr, "passed": passed})
        append_jsonl(self.progress_path, {"stage": "brain_check", "alpha_id": alpha_id, "max_abs_correlation": max_corr, "passed": passed})
        return BrainCheckResult(alpha_id, "completed" if status < 400 else f"failed_http_{status}", passed, max_corr, str(raw_path))

    def _authenticate(self) -> str:
        if self._cookie and time.time() < self._cookie_expires_at:
            return self._cookie
        raw = f"{os.environ['WQB_EMAIL']}:{os.environ['WQB_PASSWORD']}".encode("utf-8")
        auth = base64.b64encode(raw).decode("ascii")
        request = urllib.request.Request(
            f"{API_ROOT}/authentication",
            data=b"{}",
            method="POST",
            headers={
                "Accept": "application/json;version=2.0",
                "Authorization": f"Basic {auth}",
                "Content-Type": "application/json",
                "Origin": "https://platform.worldquantbrain.com",
                "Referer": "https://platform.worldquantbrain.com/",
            },
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            cookies = response.headers.get_all("Set-Cookie", [])
        self._cookie = "; ".join(cookie.split(";", 1)[0] for cookie in cookies)
        self._cookie_expires_at = time.time() + 23 * 60 * 60
        return self._cookie

    def _request_json(
        self,
        method: str,
        path_or_url: str,
        cookie: str,
        payload: dict[str, Any] | None = None,
        timeout: int = 60,
    ) -> tuple[int, dict[str, str], Any]:
        url = path_or_url if path_or_url.startswith("http") else f"{API_ROOT}{path_or_url}"
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            method=method,
            headers={
                "Accept": "application/json;version=2.0",
                "Content-Type": "application/json",
                "Cookie": cookie,
                "Origin": "https://platform.worldquantbrain.com",
                "Referer": "https://platform.worldquantbrain.com/",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                text = response.read().decode("utf-8", errors="replace")
                return response.status, dict(response.headers), safe_json(text)
        except urllib.error.HTTPError as error:
            text = error.read().decode("utf-8", errors="replace")
            return error.code, dict(error.headers), safe_json(text) or text

    def _poll_simulation(self, cookie: str, location: str | None) -> dict[str, Any]:
        if not location:
            return {"status": "missing_simulation_location"}
        deadline = time.time() + self.poll_timeout_seconds
        last_payload: Any = None
        while time.time() < deadline:
            status, _, payload = self._request_json("GET", location, cookie, timeout=60)
            last_payload = payload
            if status >= 400:
                return {"status": f"failed_http_{status}", "payload": payload}
            if not is_pending_payload(payload):
                return payload if isinstance(payload, dict) else {"payload": payload}
            time.sleep(self.poll_interval_seconds)
        return {"status": "timeout", "payload": last_payload}

    def _poll_check(self, cookie: str, location: str) -> Any:
        deadline = time.time() + min(self.poll_timeout_seconds, 300)
        last_payload: Any = None
        while time.time() < deadline:
            status, _, payload = self._request_json("GET", location, cookie, timeout=60)
            last_payload = payload
            if status >= 400:
                return {"status": f"failed_http_{status}", "payload": payload}
            if not is_pending_payload(payload):
                return payload
            time.sleep(self.poll_interval_seconds)
        return {"status": "timeout", "payload": last_payload}

    def _mock_result(self, expression: str, period: str) -> BacktestResult:
        mock_key = sha256_json({"expression": expression, "period": period})
        seed = int(mock_key[:8], 16) % 1000
        pnl = [math.sin((index + seed) / 11) / 100 for index in range(252)]
        pnl_path = self.pnl_dir / f"mock_{mock_key}.json"
        write_json(pnl_path, {"pnl": pnl})
        sharpe = round((sum(pnl) / len(pnl)) / (max(1e-9, _std(pnl))) * math.sqrt(252), 4)
        snapshot_path = self.snapshot_dir / f"mock_{mock_key}.json"
        payload = {"mock": True, "period": period, "expression": expression, "sharpe": sharpe, "pnl_path": str(pnl_path)}
        write_json(snapshot_path, payload)
        append_jsonl(self.progress_path, {"stage": "mock_backtest", "period": period, "sharpe": sharpe, "expression": expression})
        return BacktestResult(
            alpha_id=f"mock-{mock_key[:12]}",
            expression=expression,
            period=period,
            status="completed",
            sharpe=sharpe,
            fitness=1.0,
            turnover=0.35,
            pnl_path=str(pnl_path),
            net_sharpe=sharpe - 0.05,
            raw_path=str(snapshot_path),
            simulation_id=f"mock-sim-{mock_key[:12]}",
            has_daily_pnl=True,
        )


def _std(values: list[float]) -> float:
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / max(1, len(values) - 1))


def safe_json(text: str) -> Any:
    try:
        return json.loads(text) if text else {}
    except Exception:
        return text


def extract_simulation_id(location: str | None) -> str | None:
    if not location:
        return None
    return location.rstrip("/").rsplit("/", 1)[-1]


def is_pending_payload(payload: Any) -> bool:
    return isinstance(payload, dict) and set(payload.keys()) == {"progress"}


def safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def estimate_net_sharpe(sharpe: float | None, turnover: float | None) -> float | None:
    if sharpe is None:
        return None
    if turnover is None:
        return sharpe
    # Approximate the v2.1 cost gate when only aggregate metrics are available.
    cost_penalty = 0.05 + 0.02 * math.sqrt(max(turnover, 0.0) / 0.5)
    return sharpe - cost_penalty


def find_series(value: Any) -> list[float] | None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"pnl", "daily_pnl", "returns", "pnlSeries"} and is_numeric_series(item):
                return [float(number) for number in item]
            nested = find_series(item)
            if nested:
                return nested
    if isinstance(value, list):
        if is_numeric_series(value):
            return [float(number) for number in value]
        for item in value[:10]:
            nested = find_series(item)
            if nested:
                return nested
    return None


def is_numeric_series(value: Any) -> bool:
    return isinstance(value, list) and len(value) >= 20 and all(isinstance(item, (int, float)) for item in value[:20])


def extract_max_correlation(payload: Any) -> float | None:
    values = list(extract_correlation_values(payload))
    return max((abs(value) for value in values), default=None)


def extract_correlation_values(payload: Any):
    if isinstance(payload, dict):
        keys_text = " ".join(str(key).lower() for key in payload.keys())
        values_text = " ".join(str(payload.get(key, "")).lower() for key in ("name", "check", "code", "test", "message", "result"))
        is_correlation_node = "correlation" in keys_text or "correlation" in values_text or "self_corr" in values_text
        if is_correlation_node:
            for key in ("correlation", "value", "score"):
                number = safe_float(payload.get(key))
                if number is not None and -1.0 <= number <= 1.0:
                    yield number
        for item in payload.values():
            yield from extract_correlation_values(item)
    elif isinstance(payload, list):
        for item in payload:
            yield from extract_correlation_values(item)
