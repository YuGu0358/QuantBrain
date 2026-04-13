# QuantBrain Agentic Alpha Lab

This workspace now contains three runnable layers:

- local research runners for direct CLI execution
- a Railway-deployable HTTP service wrapper for remote execution
- a browser dashboard with scheduler controls and run monitoring
- a Python v2 research-core scaffold for hypothesis generation, validation, BRAIN aggregate evaluation, and regular-tier degraded-mode handling

## Local files

- [brain_discovery.mjs](/Users/yugu/Downloads/QuantBrain/brain_discovery.mjs)
- [candidate_library.mjs](/Users/yugu/Downloads/QuantBrain/candidate_library.mjs)
- [trigger_formal_round.sh](/Users/yugu/Downloads/QuantBrain/scripts/trigger_formal_round.sh)

## Cloud service files

- [service/agentic_alpha_lab.mjs](/Users/yugu/Downloads/QuantBrain/service/agentic_alpha_lab.mjs)
- [service/agentic_alpha_library.mjs](/Users/yugu/Downloads/QuantBrain/service/agentic_alpha_library.mjs)
- [service/server.mjs](/Users/yugu/Downloads/QuantBrain/service/server.mjs)
- [package.json](/Users/yugu/Downloads/QuantBrain/package.json)
- [Dockerfile](/Users/yugu/Downloads/QuantBrain/Dockerfile)
- [docker-compose.yml](/Users/yugu/Downloads/QuantBrain/docker-compose.yml)
- [.env.example](/Users/yugu/Downloads/QuantBrain/.env.example)
- [railway.json](/Users/yugu/Downloads/QuantBrain/railway.json)

## Local usage

Python v2 generation-only smoke:

```bash
PYTHONPATH=. python3 -m alpha_miner.main \
  --mode generate \
  --objective 'US equity momentum alphas' \
  --batch-size 3 \
  --output-dir /tmp/quantbrain-python-smoke
```

Legacy JS diversity-v2 generation smoke:

```bash
ALPHA_GENERATOR_STRATEGY=diversity-v2 \
ALPHA_EXPERIMENTAL_FIELDS=false \
node service/agentic_alpha_lab.mjs \
  --mode generate \
  --objective 'robust operating-income quality with low crowding and positive test stability' \
  --batch-size 7 \
  --output-dir /tmp/quantbrain-diversity-v2-smoke
```

Run the local diversity-v2 regression smoke:

```bash
npm run test:diversity-v2
```

Python v2 live evaluation is intentionally constrained by [docs/phase0_brain_probe_report.md](/Users/yugu/Downloads/QuantBrain/docs/phase0_brain_probe_report.md). If the report remains `Case C`, regular-tier mode does not fabricate daily PnL: DSR and mean-variance optimization are blocked, while BRAIN `/alphas/{id}/check` and aggregate/expression proxies are used only as degraded filters.

Optional BRAIN PnL probes:

```bash
PHASE0_VISUALIZATION_PROBE=1 \
PHASE0_SPECULATIVE_ENDPOINT_PROBE=1 \
PYTHONPATH=. python3 -m alpha_miner.scripts.phase0_brain_probe
```

Legacy JS runner:

```bash
export WQB_EMAIL='your-email@example.com'
export WQB_PASSWORD='your-password'

node service/agentic_alpha_lab.mjs \
  --mode evaluate \
  --objective 'robust fundamental quality with low crowding' \
  --rounds 1 \
  --batch-size 3 \
  --output-dir ./agentic-runs/local-test
```

Trigger a formal run on the deployed Railway service:

```bash
bash /Users/yugu/Downloads/QuantBrain/scripts/trigger_formal_round.sh
```

## HTTP service

Start the server:

```bash
export WQB_EMAIL='your-email@example.com'
export WQB_PASSWORD='your-password'
export PORT=3000
export ADMIN_TOKEN='replace-with-a-random-dashboard-token'
export CREDENTIALS_SECRET='replace-with-a-long-random-secret'
export DASHBOARD_USERS='teammate-a:replace-with-random-token,teammate-b:replace-with-random-token'

npm start
```

Open the dashboard:

```bash
open http://localhost:3000/
```

### Endpoints

Health:

```bash
curl http://localhost:3000/health
```

Create a run:

```bash
curl -X POST http://localhost:3000/runs \
  -H 'content-type: application/json' \
  -H 'authorization: Bearer <ADMIN_TOKEN>' \
  -d '{
    "mode": "evaluate",
    "objective": "robust fundamental quality with low crowding",
    "engine": "python-v2",
    "rounds": 1,
    "batchSize": 3
  }'
```

Inspect a run:

```bash
curl -H 'authorization: Bearer <ADMIN_TOKEN>' http://localhost:3000/runs/<run-id>
```

Inspect the current dashboard user and BRAIN credential status:

```bash
curl -H 'authorization: Bearer <TEAM_OR_ADMIN_TOKEN>' http://localhost:3000/account
```

Save the current dashboard user's own BRAIN credentials:

```bash
curl -X POST http://localhost:3000/account/brain-credentials \
  -H 'content-type: application/json' \
  -H 'authorization: Bearer <TEAM_OR_ADMIN_TOKEN>' \
  -d '{"email":"teammate@example.com","password":"..."}'
```

Submit an alpha only if the server-side gate allows it:

```bash
curl -X POST http://localhost:3000/alphas/<alpha-id>/submit \
  -H 'content-type: application/json' \
  -H 'authorization: Bearer <ADMIN_TOKEN>' \
  -d '{}'
```

Scheduler:

```bash
curl -H 'authorization: Bearer <ADMIN_TOKEN>' http://localhost:3000/scheduler
```

## Docker

Build:

```bash
docker build -t worldquant-agentic-alpha-lab .
```

Run:

```bash
docker run --rm -p 3000:3000 \
  -e WQB_EMAIL='your-email@example.com' \
  -e WQB_PASSWORD='your-password' \
  worldquant-agentic-alpha-lab
```

Compose on a cloud VM:

```bash
cp .env.example .env
docker compose up -d --build
```

## Deployment notes

- This is suitable for a VPS, bare cloud VM, Docker host, Render worker/web service, Railway, Fly.io, or Kubernetes.
- For single-user rollback, `WQB_EMAIL` and `WQB_PASSWORD` can stay in environment variables and are only used by the default/admin owner.
- For team sharing, use `DASHBOARD_USERS=user_id:token,user_id_2:token_2`; each teammate logs in with their own dashboard token and saves their own BRAIN credentials in the dashboard Account panel.
- Per-user BRAIN credentials are encrypted under `CREDENTIALS_DIR` using `CREDENTIALS_SECRET` and are injected only into that user's run/submit process. They are not returned by the API and are not written into run artifacts.
- Set `AUTO_RUN_OWNER_ID=default` or another configured user id to decide which BRAIN account owns scheduled server-side runs.
- Run artifacts are written under `RUNS_DIR` and can be mounted to persistent storage.
- The service is intentionally stateless except for run, idea, auto-loop, and encrypted credential directories.
- `GET /runs/:id` now returns either `generated`, `batch`, or `summary` artifacts depending on the run mode.
- On Railway, if a volume is attached and `RUNS_DIR` is not set, the service automatically uses `RAILWAY_VOLUME_MOUNT_PATH/runs`.
- `ADMIN_TOKEN` protects run and scheduler APIs. The dashboard stores the token only in browser local storage.
- `ADMIN_TOKEN` is the administrator/default-owner token. It can update the global scheduler and see all runs. `DASHBOARD_USERS` tokens can start their own runs, save their own credentials, and see only their own run/idea records.
- `AUTO_RUN_ENABLED=true` runs the scheduler inside the Railway service, so the mining loop does not depend on a local machine.
- `AUTO_RUN_MODE=loop` enables the full generate, test, optimize loop for scheduled runs. Use `AUTO_RUN_ROUNDS=2` or higher if you want mutations to execute after the seed batch.
- `ALPHA_MINER_ENGINE=python-v2` routes runs through the Python core. Set it to `legacy-js` if you want the older JS loop as the live fallback while the regular-tier PnL limitation remains unresolved.
- `AUTO_REPAIR_ENABLED=true` and `AUTO_SUBMIT_ENABLED=true` enable the legacy-JS repair loop: blocked alphas are repaired from gate feedback and only submitted with the normal non-forced gate.
- `ALPHA_GENERATOR_STRATEGY=legacy|diversity-v2` controls the live JS generator. Use `legacy` for rollback and `diversity-v2` for data-family rotation, operator-pattern crowding control, and preflight filtering.
- `ALPHA_EXPERIMENTAL_FIELDS=false` keeps account-dependent news/options/model fields out of default live batches. Set it to `true` only after verifying those fields are available in your BRAIN account.
- `ALPHA_CROWDING_PATTERN_THRESHOLD=2` blocks repeated operator skeletons before they consume BRAIN simulation quota.
- `ALPHA_FAMILY_COOLDOWN_ROUNDS=3` blocks a data family after it dominates recent trajectories under `diversity-v2`.
- `GET /auto-loop` shows the current repair queue, active repair run, recent submissions, and auto-loop events.
- The dashboard submit button is gated. The service refuses submission unless the alpha is `UNSUBMITTED`, every available IS check is `PASS`, and `testSharpe` is positive.
- The mining loop now ranks candidates with an added submission-readiness component and favors OOS/test-stable, low-crowding, hypothesis-preserving mutations over pure IS score chasing.
- The `Mining Progress` dashboard panel shows diversity status from the latest run: active strategy, data-family counts, preflight accepted/rejected counts, and crowded operator patterns.
