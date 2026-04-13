#!/usr/bin/env bash

set -euo pipefail

API_URL="${API_URL:-https://worldquant-agentic-alpha-api-production.up.railway.app}"
TOKEN_FILE="${TOKEN_FILE:-/Users/yugu/Downloads/QuantBrain/.dashboard-token}"
ADMIN_TOKEN="${ADMIN_TOKEN:-}"
OBJECTIVE="${OBJECTIVE:-robust operating-income quality with low crowding and positive test stability}"
MODE="${MODE:-evaluate}"
ROUNDS="${ROUNDS:-1}"
BATCH_SIZE="${BATCH_SIZE:-3}"
WAIT_FOR_COMPLETION="${WAIT_FOR_COMPLETION:-false}"
POLL_SECONDS="${POLL_SECONDS:-20}"
MAX_POLLS="${MAX_POLLS:-60}"
RUN_ID="${RUN_ID:-formal-mining-$(date -u +%Y-%m-%dT%H-%M-%SZ | tr ':' '-')}"

if [ -z "${ADMIN_TOKEN}" ] && [ -f "${TOKEN_FILE}" ]; then
  ADMIN_TOKEN="$(cat "${TOKEN_FILE}")"
fi

auth_args=()
if [ -n "${ADMIN_TOKEN}" ]; then
  auth_args=(-H "authorization: Bearer ${ADMIN_TOKEN}")
fi

active_runs_json="$(curl -fsS "${auth_args[@]}" "${API_URL}/runs")"
active_count="$(
  printf '%s' "${active_runs_json}" | node -e '
    const fs = require("fs");
    const payload = JSON.parse(fs.readFileSync(0, "utf8"));
    process.stdout.write(String((payload.active || []).filter((run) => run.status === "running").length));
  '
)"

if [ "${active_count}" != "0" ]; then
  echo "Another factor-mining run is still active. Skipping a new trigger."
  printf '%s\n' "${active_runs_json}"
  exit 0
fi

payload="$(
  node -e '
    const payload = {
      runId: process.env.RUN_ID,
      mode: process.env.MODE,
      objective: process.env.OBJECTIVE,
      rounds: Number(process.env.ROUNDS),
      batchSize: Number(process.env.BATCH_SIZE),
    };
    process.stdout.write(JSON.stringify(payload));
  '
)"

response="$(curl -fsS -X POST "${API_URL}/runs" "${auth_args[@]}" -H 'content-type: application/json' -d "${payload}")"
printf '%s\n' "${response}"

if [ "${WAIT_FOR_COMPLETION}" != "true" ]; then
  exit 0
fi

for _ in $(seq 1 "${MAX_POLLS}"); do
  status_json="$(curl -fsS "${auth_args[@]}" "${API_URL}/runs/${RUN_ID}")"
  status="$(
    printf '%s' "${status_json}" | node -e '
      const fs = require("fs");
      const payload = JSON.parse(fs.readFileSync(0, "utf8"));
      process.stdout.write(String(payload?.state?.status || "unknown"));
    '
  )"

  if [ "${status}" = "completed" ] || [ "${status}" = "failed" ] || [ "${status}" = "unknown" ]; then
    printf '%s\n' "${status_json}"
    exit 0
  fi

  sleep "${POLL_SECONDS}"
done

echo "Timed out waiting for ${RUN_ID}."
curl -fsS "${auth_args[@]}" "${API_URL}/runs/${RUN_ID}"
