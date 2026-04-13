from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


API_ROOT = "https://api.worldquantbrain.com"
REPORT_PATH = Path("docs/phase0_brain_probe_report.md")
SPECULATIVE_ALPHA_ENDPOINTS = [
    "/alphas/{alpha_id}/recordsets",
    "/alphas/{alpha_id}/pnl",
    "/alphas/{alpha_id}/returns",
    "/alphas/{alpha_id}/performance",
    "/alphas/{alpha_id}/chart",
    "/alphas/{alpha_id}/visualization",
    "/alphas/{alpha_id}/yearly",
]


def main() -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not os.environ.get("WQB_EMAIL") or not os.environ.get("WQB_PASSWORD"):
        if REPORT_PATH.exists() and os.environ.get("PHASE0_OVERWRITE_WITHOUT_CREDENTIALS", "0") != "1":
            print(f"Skipped live probe because WQB_EMAIL/WQB_PASSWORD are missing; kept existing {REPORT_PATH}")
            return
        write_report("UNKNOWN", "Missing WQB_EMAIL/WQB_PASSWORD; live probe was not run.", {})
        print(f"Wrote {REPORT_PATH}")
        return

    expression = os.environ.get("PHASE0_PROBE_EXPRESSION", "rank(close)")
    try:
        cookie = authenticate()
        is_payload = existing_or_submit(cookie, expression, "IS", "2015-01-01", "2021-12-31")
        oos_payload = existing_or_submit(cookie, expression, "OOS", "2023-01-01", "2025-12-31")
        is_payload["poll"] = poll_simulation(cookie, is_payload.get("simulationId"))
        oos_payload["poll"] = poll_simulation(cookie, oos_payload.get("simulationId"))
        visualization_payload = None
        if os.environ.get("PHASE0_VISUALIZATION_PROBE", "0") == "1":
            visualization_payload = existing_or_submit(
                cookie,
                expression,
                "VIS",
                "2015-01-01",
                "2021-12-31",
                visualization=True,
            )
            visualization_payload["poll"] = poll_simulation(cookie, visualization_payload.get("simulationId"))
        decision, detail = decide_case(is_payload, oos_payload)
        payload = {"is": is_payload, "oos": oos_payload}
        if visualization_payload:
            payload["visualization"] = visualization_payload
            visualization_series = visualization_payload.get("poll", {}).get("pnlSeriesPaths") or []
            if visualization_series and decision == "Case C":
                decision = "Case B"
                detail = "Date splits appear fixed, but visualization:true exposed a time-series-like payload that may be splittable locally."
        write_report(decision, detail, payload)
    except Exception as error:
        write_report("UNKNOWN", f"Probe failed: {error}", {})
        raise
    finally:
        print(f"Wrote {REPORT_PATH}")


def authenticate() -> str:
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
    cookie = "; ".join(item.split(";", 1)[0] for item in cookies)
    if not cookie:
        raise RuntimeError("Authentication returned no cookie.")
    return cookie


def existing_or_submit(cookie: str, expression: str, label: str, start: str, end: str, visualization: bool = False) -> dict[str, Any]:
    existing = os.environ.get(f"PHASE0_EXISTING_{label}_SIM_ID")
    if existing:
        return {
            "request": {"regular": expression, "settings": {"startDate": start, "endDate": end, "visualization": visualization}},
            "status": "existing",
            "simulationId": existing,
        }
    return submit_probe(cookie, expression, start, end, visualization=visualization)


def submit_probe(cookie: str, expression: str, start: str, end: str, visualization: bool = False) -> dict[str, Any]:
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
            "visualization": visualization,
            "startDate": start,
            "endDate": end,
        },
        "regular": expression,
    }
    request = urllib.request.Request(
        f"{API_ROOT}/simulations",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Accept": "application/json;version=2.0",
            "Content-Type": "application/json",
            "Cookie": cookie,
            "Origin": "https://platform.worldquantbrain.com",
            "Referer": "https://platform.worldquantbrain.com/",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            text = response.read().decode("utf-8")
            headers = dict(response.headers)
            return {
                "request": payload,
                "status": response.status,
                "headers": headers,
                "body": safe_json(text) or text,
                "simulationId": extract_simulation_id(headers.get("Location")),
            }
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        headers = dict(error.headers)
        return {
            "request": payload,
            "status": error.code,
            "headers": headers,
            "body": safe_json(body) or body,
            "simulationId": extract_simulation_id(headers.get("Location")),
        }


def poll_simulation(cookie: str, simulation_id: str | None) -> dict[str, Any]:
    if not simulation_id:
        return {"status": "missing_simulation_id"}
    timeout_seconds = int(os.environ.get("PHASE0_POLL_TIMEOUT_SECONDS", "900"))
    deadline = time.time() + timeout_seconds
    observations: list[dict[str, Any]] = []
    last_payload: Any = None
    while time.time() < deadline:
        payload = get_json(cookie, f"/simulations/{simulation_id}")
        observations.append({"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "payload": payload})
        last_payload = payload
        if not is_pending_payload(payload):
            alpha_id = payload.get("alpha") if isinstance(payload, dict) else None
            details = {
                "simulation": payload,
                "alpha": get_json(cookie, f"/alphas/{alpha_id}") if isinstance(alpha_id, str) else None,
                "details": try_get_json(cookie, f"/simulations/{simulation_id}/details"),
                "pnl": try_get_json(cookie, f"/simulations/{simulation_id}/pnl"),
            }
            if isinstance(alpha_id, str) and os.environ.get("PHASE0_SPECULATIVE_ENDPOINT_PROBE", "0") == "1":
                details["speculativeAlphaEndpoints"] = probe_alpha_endpoints(cookie, alpha_id)
            return {
                "status": "completed",
                "simulationId": simulation_id,
                "observations": observations[-5:],
                "details": details,
                "pnlSeriesPaths": find_series_paths(
                    details,
                    {"pnl", "daily_pnl", "returns", "pnlSeries", "chart", "visualization", "recordsets", "records", "points", "data"},
                ),
            }
        time.sleep(float(os.environ.get("PHASE0_POLL_INTERVAL_SECONDS", "5")))
    return {
        "status": "timeout",
        "simulationId": simulation_id,
        "observations": observations[-5:],
        "lastPayload": last_payload,
    }


def get_json(cookie: str, path: str) -> Any:
    request = urllib.request.Request(
        f"{API_ROOT}{path}",
        method="GET",
        headers={
            "Accept": "application/json;version=2.0",
            "Content-Type": "application/json",
            "Cookie": cookie,
            "Origin": "https://platform.worldquantbrain.com",
            "Referer": "https://platform.worldquantbrain.com/",
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return safe_json(response.read().decode("utf-8")) or {}


def try_get_json(cookie: str, path: str) -> Any:
    try:
        return get_json(cookie, path)
    except Exception as error:
        return {"error": str(error)}


def probe_alpha_endpoints(cookie: str, alpha_id: str) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for template in SPECULATIVE_ALPHA_ENDPOINTS:
        path = template.format(alpha_id=alpha_id)
        try:
            results[path] = try_get_json_with_status(cookie, path)
        except Exception as error:
            results[path] = {"error": str(error)}
    return results


def try_get_json_with_status(cookie: str, path: str) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{API_ROOT}{path}",
        method="GET",
        headers={
            "Accept": "application/json;version=2.0",
            "Content-Type": "application/json",
            "Cookie": cookie,
            "Origin": "https://platform.worldquantbrain.com",
            "Referer": "https://platform.worldquantbrain.com/",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            text = response.read().decode("utf-8", errors="replace")
            return {"status": response.status, "body": safe_json(text) or text[:500]}
    except urllib.error.HTTPError as error:
        text = error.read().decode("utf-8", errors="replace")
        return {"status": error.code, "body": safe_json(text) or text[:500]}


def decide_case(is_payload: dict[str, Any], oos_payload: dict[str, Any]) -> tuple[str, str]:
    if isinstance(is_payload["status"], int) and is_payload["status"] >= 400:
        return "Case B/C UNKNOWN", "Date parameters may have been rejected or response shape requires manual inspection."
    if isinstance(oos_payload["status"], int) and oos_payload["status"] >= 400:
        return "Case B/C UNKNOWN", "Date parameters may have been rejected or response shape requires manual inspection."
    if is_payload.get("poll", {}).get("status") == "timeout" or oos_payload.get("poll", {}).get("status") == "timeout":
        return "UNKNOWN_WAITING", "At least one simulation did not finish during the probe timeout."
    series_paths = (is_payload.get("poll", {}).get("pnlSeriesPaths") or []) + (oos_payload.get("poll", {}).get("pnlSeriesPaths") or [])
    dates_honored = dates_match_request(is_payload) and dates_match_request(oos_payload)
    if dates_honored and series_paths:
        return "Case A", "BRAIN honored dated submissions and a daily PnL-like series was found after polling."
    if not dates_honored and series_paths:
        return "Case B", "BRAIN used a fixed window but returned a daily PnL-like series that can be split locally."
    if not dates_honored:
        return "Case C", "BRAIN appears to ignore submitted date splits and no daily PnL series was found."
    return "Case C", "BRAIN returned metrics but no daily PnL series was found after polling/details probes."


def write_report(case: str, detail: str, payload: dict[str, Any]) -> None:
    body = [
        "# Phase 0 BRAIN API Probe Report",
        "",
        f"- Generated at: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
        f"- Decision: {case}",
        f"- Detail: {detail}",
        "",
        "## Follow-up Policy",
        "",
        "- Case A/B: daily PnL is available and M7/M6/M8 can use the v2.1 PnL-native path.",
        "- Case C: regular-tier degraded mode is required. Do not fabricate daily PnL; use BRAIN alpha check for platform-native correlation and aggregate/expression proxies for pre-filtering. DSR and mean-variance optimization remain blocked until a real PnL source exists.",
        "- Optional probes are controlled by `PHASE0_VISUALIZATION_PROBE=1` and `PHASE0_SPECULATIVE_ENDPOINT_PROBE=1` to avoid accidental extra simulations or undocumented endpoint traffic.",
        "",
        "## Raw Probe Snapshot",
        "",
        "```json",
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)[:20000],
        "```",
        "",
    ]
    REPORT_PATH.write_text("\n".join(body), encoding="utf-8")


def safe_json(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return None


def extract_simulation_id(location: str | None) -> str | None:
    if not location:
        return None
    return location.rstrip("/").rsplit("/", 1)[-1]


def is_pending_payload(payload: Any) -> bool:
    return isinstance(payload, dict) and set(payload.keys()) == {"progress"}


def dates_match_request(payload: dict[str, Any]) -> bool:
    requested = payload.get("request", {}).get("settings", {})
    observed = payload.get("poll", {}).get("details", {}).get("alpha", {}).get("settings", {})
    return bool(
        requested.get("startDate")
        and requested.get("endDate")
        and requested.get("startDate") == observed.get("startDate")
        and requested.get("endDate") == observed.get("endDate")
    )


def find_series_paths(value: Any, names: set[str], prefix: str = "$") -> list[str]:
    paths: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            next_prefix = f"{prefix}.{key}"
            if key in names and is_series_like(item):
                paths.append(next_prefix)
            paths.extend(find_series_paths(item, names, next_prefix))
    elif isinstance(value, list):
        for index, item in enumerate(value[:5]):
            paths.extend(find_series_paths(item, names, f"{prefix}[{index}]"))
    return paths


def is_numeric_series(value: Any) -> bool:
    return isinstance(value, list) and len(value) >= 20 and all(isinstance(item, (int, float)) for item in value[:20])


def is_series_like(value: Any) -> bool:
    if is_numeric_series(value):
        return True
    if not isinstance(value, list) or len(value) < 20:
        return False
    sample = value[:20]
    if not all(isinstance(item, dict) for item in sample):
        return False
    has_numeric = all(any(isinstance(cell, (int, float)) for cell in item.values()) for item in sample)
    has_dateish = any(any("date" in str(key).lower() or "time" in str(key).lower() for key in item.keys()) for item in sample)
    return has_numeric and has_dateish


if __name__ == "__main__":
    main()
