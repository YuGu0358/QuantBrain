import { createServer } from "node:http";
import * as fs from "node:fs";
import { mkdir, readFile, readdir, unlink, writeFile } from "node:fs/promises";
import path from "node:path";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import * as crypto from "node:crypto";
import { createCipheriv, createDecipheriv, randomBytes, randomUUID, scryptSync, timingSafeEqual } from "node:crypto";

const API_ROOT = "https://api.worldquantbrain.com";
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DEFAULT_RUNS_DIR = process.env.RAILWAY_VOLUME_MOUNT_PATH
  ? path.resolve(process.env.RAILWAY_VOLUME_MOUNT_PATH, "runs")
  : path.resolve(__dirname, "../cloud-runs");
const RUNS_DIR = process.env.RUNS_DIR
  ? path.resolve(process.env.RUNS_DIR)
  : DEFAULT_RUNS_DIR;
const IDEAS_DIR = process.env.IDEAS_DIR
  ? path.resolve(process.env.IDEAS_DIR)
  : path.resolve(RUNS_DIR, "../ideas");
const CREDENTIALS_DIR = process.env.CREDENTIALS_DIR
  ? path.resolve(process.env.CREDENTIALS_DIR)
  : path.resolve(RUNS_DIR, "../credentials");
const PORT = Number(process.env.PORT ?? 3000);
const ADMIN_TOKEN = process.env.ADMIN_TOKEN ?? "";
const CREDENTIALS_SECRET = process.env.CREDENTIALS_SECRET ?? "";
const DASHBOARD_USERS = parseDashboardUsers(process.env.DASHBOARD_USERS ?? "");
const REGISTRATION_CODE = process.env.REGISTRATION_CODE ?? "";
const OPENAI_IDEA_MODEL = process.env.OPENAI_IDEA_MODEL ?? "gpt-5.4-mini";
const OPENAI_OPTIMIZE_MODEL = "gpt-4o-mini";
const OPTIMIZE_IDEA_SYSTEM_PROMPT =
  'You are a WorldQuant BRAIN alpha research specialist. Convert the user idea into a structured research direction. Return strict JSON: {"objective":string,"category":one of QUALITY/MOMENTUM/REVERSAL/LIQUIDITY/VOLATILITY/MICROSTRUCTURE/SENTIMENT,"hypothesis":string,"constraints":[string],"suggested_data_fields":[string]}';
const OPTIMIZE_IDEA_CATEGORIES = new Set([
  "QUALITY",
  "MOMENTUM",
  "REVERSAL",
  "LIQUIDITY",
  "VOLATILITY",
  "MICROSTRUCTURE",
  "SENTIMENT",
]);
const DEFAULT_ENGINE = process.env.ALPHA_MINER_ENGINE ?? "python-v2";
const PYTHON_BIN = process.env.PYTHON_BIN ?? "python3";
const AUTO_RUN_OWNER_ID = sanitizeOwnerId(process.env.AUTO_RUN_OWNER_ID ?? "default");
const DEFAULT_MODE = process.env.AUTO_RUN_MODE ?? "loop";
const DEFAULT_OBJECTIVE =
  process.env.AUTO_RUN_OBJECTIVE ??
  "robust operating-income quality with low crowding and positive test stability";
const DEFAULT_INTERVAL_MINUTES = Number(process.env.AUTO_RUN_INTERVAL_MINUTES ?? 60);
const DEFAULT_BATCH_SIZE = Number(process.env.AUTO_RUN_BATCH_SIZE ?? 3);
const DEFAULT_ROUNDS = Number(process.env.AUTO_RUN_ROUNDS ?? 1);
const DEFAULT_CONCURRENCY = clampInt(process.env.AUTO_RUN_CONCURRENCY, 3, 1, 3);
const AUTO_LOOP_STATE_PATH = path.join(RUNS_DIR, "auto-loop-state.json");
const AUTO_REPAIR_ENABLED = process.env.AUTO_REPAIR_ENABLED === "true";
const AUTO_REPAIR_ENGINE = normalizeEngine(process.env.AUTO_REPAIR_ENGINE ?? "legacy-js");
const AUTO_REPAIR_MAX_ROUNDS = clampInt(process.env.AUTO_REPAIR_MAX_ROUNDS, 5, 1, 10);
const AUTO_REPAIR_BATCH_SIZE = clampInt(process.env.AUTO_REPAIR_BATCH_SIZE, 5, 1, 10);
const AUTO_SUBMIT_ENABLED = process.env.AUTO_SUBMIT_ENABLED === "true";
const ALPHA_GENERATOR_STRATEGY = ["legacy", "diversity-v2"].includes(process.env.ALPHA_GENERATOR_STRATEGY)
  ? process.env.ALPHA_GENERATOR_STRATEGY
  : "legacy";
const ALPHA_EXPERIMENTAL_FIELDS = process.env.ALPHA_EXPERIMENTAL_FIELDS === "true";
const ALPHA_CROWDING_PATTERN_THRESHOLD = clampInt(process.env.ALPHA_CROWDING_PATTERN_THRESHOLD, 2, 1, 20);
const ALPHA_FAMILY_COOLDOWN_ROUNDS = clampInt(process.env.ALPHA_FAMILY_COOLDOWN_ROUNDS, 3, 1, 20);

const activeRuns = new Map();
const schedulerState = {
  enabled: process.env.AUTO_RUN_ENABLED === "true",
  engine: normalizeEngine(DEFAULT_ENGINE),
  mode: DEFAULT_MODE,
  objective: DEFAULT_OBJECTIVE,
  intervalMinutes: DEFAULT_INTERVAL_MINUTES,
  rounds: DEFAULT_ROUNDS,
  batchSize: DEFAULT_BATCH_SIZE,
  concurrency: DEFAULT_CONCURRENCY,
  lastTickAt: null,
  lastRunId: null,
  lastSkipReason: null,
  nextRunAt: null,
};

await mkdir(RUNS_DIR, { recursive: true });
await mkdir(IDEAS_DIR, { recursive: true });
await mkdir(CREDENTIALS_DIR, { recursive: true });
const userRegistry = await loadUserRegistry();
let autoLoopState = await loadAutoLoopState();
scheduleNextRun("startup");

const server = createServer(async (req, res) => {
  try {
    const url = new URL(req.url, `http://${req.headers.host}`);
    const authContext = getAuthContext(req, url);

    if (req.method === "GET" && url.pathname === "/") {
      return sendHtml(res, 200, dashboardHtml());
    }

    if (req.method === "GET" && url.pathname === "/setup") {
      return sendHtml(res, 200, setupHtml());
    }

    if (req.method === "POST" && url.pathname === "/account/register") {
      const body = await readJson(req);
      const result = await registerUser(body);
      return sendJson(res, result.ok ? 201 : 400, result);
    }

    if (req.method === "GET" && url.pathname === "/health") {
      return sendJson(res, 200, {
        ok: true,
        service: "worldquant-agentic-alpha-lab",
        scheduler: publicSchedulerState(),
      });
    }

    if (
      url.pathname.startsWith("/runs") ||
      url.pathname.startsWith("/scheduler") ||
      url.pathname.startsWith("/alphas") ||
      url.pathname.startsWith("/ideas") ||
      url.pathname.startsWith("/auto-loop") ||
      url.pathname.startsWith("/account")
    ) {
      if (!authContext.ok) {
        return sendJson(res, 401, {
          error: "Unauthorized",
          detail: "Set a dashboard token in the Authorization header or x-admin-token header.",
        });
      }
    }

    if (req.method === "GET" && url.pathname === "/account") {
      return sendJson(res, 200, await publicAccountState(authContext));
    }

    if (req.method === "POST" && url.pathname === "/account/brain-credentials") {
      const body = await readJson(req);
      const result = await saveBrainCredentials(authContext.userId, body);
      return sendJson(res, 200, result);
    }

    if (req.method === "DELETE" && url.pathname === "/account/brain-credentials") {
      const result = await deleteBrainCredentials(authContext.userId);
      return sendJson(res, 200, result);
    }

    if (req.method === "GET" && url.pathname === "/runs") {
      const response = await buildRunsIndex(authContext);
      response.llmRouterState = await readLlmRouterState();
      return sendJson(res, 200, response);
    }

    if (req.method === "POST" && url.pathname === "/runs") {
      const body = await readJson(req);
      if (hasRunningRun() && body.force !== true) {
        return sendJson(res, 409, {
          error: "A run is already active.",
          active: runningRuns(authContext),
        });
      }
      const state = await createRun(body, "manual", authContext);
      return sendJson(res, 202, {
        runId: state.runId,
        pid: state.pid,
        engine: state.engine,
        status: state.status,
        outputDir: state.outputDir,
      });
    }

    if (req.method === "GET" && url.pathname === "/ideas") {
      return sendJson(res, 200, await buildIdeasIndex(authContext));
    }

    if (req.method === "POST" && url.pathname === "/ideas/analyze") {
      const body = await readJson(req);
      const idea = await analyzeIdea(body.description, authContext);
      await saveIdea(idea);
      return sendJson(res, 201, idea);
    }

    if (req.method === "POST" && url.pathname === "/ideas/optimize") {
      const body = await readJson(req);
      const idea = (typeof body?.idea === "string" ? body.idea : "").trim().slice(0, 500);
      if (!idea) return sendJson(res, 400, { error: "Idea is required." });
      return sendJson(res, 200, await optimizeIdea(idea));
    }

    const optimizedIdeaMatch = url.pathname.match(/^\/ideas\/optimize\/([^/]+)$/);
    if (req.method === "GET" && optimizedIdeaMatch) {
      const ideaId = sanitizeIdeaId(decodeURIComponent(optimizedIdeaMatch[1]));
      const idea = await readOptimizedIdea(ideaId);
      if (!idea) return sendJson(res, 404, { error: "Optimized idea not found." });
      return sendJson(res, 200, idea);
    }

    const ideaMatch = url.pathname.match(/^\/ideas\/([^/]+)$/);
    if (req.method === "GET" && ideaMatch) {
      const ideaId = sanitizeIdeaId(decodeURIComponent(ideaMatch[1]));
      return sendJson(res, 200, await readIdea(ideaId, authContext));
    }

    const ideaRunMatch = url.pathname.match(/^\/ideas\/([^/]+)\/run$/);
    if (req.method === "POST" && ideaRunMatch) {
      const body = await readJson(req);
      const ideaId = sanitizeIdeaId(decodeURIComponent(ideaRunMatch[1]));
      const result = await createRunFromIdea(ideaId, body, authContext);
      return sendJson(res, result.error ? 409 : 202, result);
    }

    if (req.method === "GET" && url.pathname === "/scheduler") {
      return sendJson(res, 200, publicSchedulerState());
    }

    if (req.method === "GET" && url.pathname === "/auto-loop") {
      return sendJson(res, 200, publicAutoLoopState(authContext));
    }

    if (req.method === "POST" && url.pathname === "/scheduler") {
      if (!isAdminAuth(authContext)) return sendJson(res, 403, { error: "Only the admin token can update the global scheduler." });
      const body = await readJson(req);
      if (typeof body.enabled === "boolean") schedulerState.enabled = body.enabled;
      if (typeof body.engine === "string") schedulerState.engine = normalizeEngine(body.engine);
      if (typeof body.mode === "string" && ["evaluate", "loop", "generate"].includes(body.mode)) {
        schedulerState.mode = body.mode;
      }
      if (typeof body.objective === "string" && body.objective.trim()) {
        schedulerState.objective = body.objective.trim();
      }
      if (Number.isFinite(Number(body.intervalMinutes)) && Number(body.intervalMinutes) >= 15) {
        schedulerState.intervalMinutes = Number(body.intervalMinutes);
      }
      if (Number.isFinite(Number(body.rounds)) && Number(body.rounds) >= 1) {
        schedulerState.rounds = Number(body.rounds);
      }
      if (Number.isFinite(Number(body.batchSize)) && Number(body.batchSize) >= 1) {
        schedulerState.batchSize = Number(body.batchSize);
      }
      if (Number.isFinite(Number(body.concurrency)) && Number(body.concurrency) >= 1) {
        schedulerState.concurrency = clampInt(body.concurrency, DEFAULT_CONCURRENCY, 1, 3);
      }
      scheduleNextRun("scheduler-update");
      return sendJson(res, 200, publicSchedulerState());
    }

    const submitMatch = url.pathname.match(/^\/alphas\/([^/]+)\/submit$/);
    if (req.method === "POST" && submitMatch) {
      await readJson(req);
      const alphaId = decodeURIComponent(submitMatch[1]);
      const result = await submitAlpha(alphaId, { force: false, source: "manual-submit", ownerId: authContext.userId });
      return sendJson(res, result.submitted ? 200 : 422, result);
    }

    const runMatch = url.pathname.match(/^\/runs\/([^/]+)$/);
    if (req.method === "GET" && runMatch) {
      const runId = sanitizeRunId(decodeURIComponent(runMatch[1]));
      return sendJson(res, 200, await readRun(runId, false, authContext));
    }

    return sendJson(res, 404, { error: "Not found" });
  } catch (error) {
    return sendJson(res, 500, { error: error instanceof Error ? error.message : String(error) });
  }
});

server.listen(PORT, () => {
  console.log(`worldquant-agentic-alpha-lab listening on ${PORT}`);
});

setInterval(() => {
  void tickScheduler().catch((error) => {
    schedulerState.lastSkipReason = `Scheduler error: ${error instanceof Error ? error.message : String(error)}`;
    schedulerState.lastErrorAt = new Date().toISOString();
    scheduleNextRun("scheduler-error");
    pushAutoLoopEvent({
      type: "scheduler-error",
      ownerId: AUTO_RUN_OWNER_ID,
      error: error instanceof Error ? error.message : String(error),
    });
    void saveAutoLoopState();
  });
}, 60_000).unref();

async function createRun(input, source, authContext = systemAuthContext()) {
  const now = new Date();
  const runId = sanitizeRunId(
    input.runId ??
      `${source}-mining-${now.toISOString().replaceAll(":", "-").replace(/\.\d{3}Z$/, "Z")}`,
  );
  const mode = normalizeMode(input.mode ?? schedulerState.mode);
  const engine = normalizeEngine(input.engine ?? schedulerState.engine);
  const objective = input.objective ?? schedulerState.objective;
  const rounds = Number(input.rounds ?? schedulerState.rounds);
  const batchSize = Number(input.batchSize ?? schedulerState.batchSize);
  const concurrency = clampInt(input.concurrency ?? schedulerState.concurrency, DEFAULT_CONCURRENCY, 1, 3);
  const ownerId = sanitizeOwnerId(input.ownerId ?? authContext?.userId ?? "default");
  const outputDir = path.join(RUNS_DIR, runId);
  const credentialEnv = mode === "generate"
    ? { ok: true, source: "not-required", env: {} }
    : await brainCredentialEnv(ownerId);
  if (mode !== "generate" && !credentialEnv.ok) {
    throw new Error(
      `BRAIN credentials are not configured for owner ${ownerId}. Save account credentials in the dashboard before running ${mode}.`,
    );
  }

  await mkdir(outputDir, { recursive: true });
  let repairContextPath = null;
  if (input.repairContext) {
    repairContextPath = path.join(outputDir, "repair-context.json");
    await writeFile(repairContextPath, `${JSON.stringify(input.repairContext, null, 2)}\n`, "utf8");
  }
  await writeRunMeta(outputDir, {
    runId,
    ownerId,
    source,
    engine,
    mode,
    objective,
    rounds,
    batchSize,
    concurrency,
    generatorStrategy: ALPHA_GENERATOR_STRATEGY,
    credentialSource: credentialEnv.source,
    startedAt: now.toISOString(),
  });
  const command = buildRunCommand({ engine, mode, objective, rounds, batchSize, concurrency, outputDir, repairContextPath });
  const child = spawn(
    command.bin,
    command.args,
    {
      cwd: path.resolve(__dirname, ".."),
      env: {
        ...process.env,
        PYTHONPATH: path.resolve(__dirname, ".."),
        ...credentialEnv.env,
      },
      stdio: ["ignore", "pipe", "pipe"],
    },
  );

  const state = {
    runId,
    pid: child.pid,
    engine,
    mode,
    ownerId,
    source,
    generatorStrategy: ALPHA_GENERATOR_STRATEGY,
    objective,
    rounds,
    batchSize,
    concurrency,
    parentAlphaId: input.parentAlphaId ?? input.repairContext?.parentAlphaId ?? null,
    repairDepth: input.repairContext?.repairDepth ?? null,
    outputDir,
    status: "running",
    logs: [],
    startedAt: new Date().toISOString(),
  };
  activeRuns.set(runId, state);

  child.stdout.on("data", (chunk) => appendLog(state, chunk.toString()));
  child.stderr.on("data", (chunk) => appendLog(state, chunk.toString()));
  child.on("exit", (code) => {
    state.status = code === 0 ? "completed" : "failed";
    state.exitCode = code;
    state.finishedAt = new Date().toISOString();
    if (source === "scheduler") scheduleNextRun("run-finished");
    void handleRunFinished(state).catch((error) => {
      pushAutoLoopEvent({
        type: "run-finish-handler-error",
        runId: state.runId,
        ownerId: state.ownerId,
        error: error instanceof Error ? error.message : String(error),
      });
      void saveAutoLoopState();
    });
  });

  return state;
}

async function tickScheduler() {
  schedulerState.lastTickAt = new Date().toISOString();
  if (!schedulerState.enabled) return;
  if (!schedulerState.nextRunAt) scheduleNextRun("missing-next-run");
  if (Date.now() < Date.parse(schedulerState.nextRunAt)) return;
  if (hasRunningRun()) {
    schedulerState.lastSkipReason = "A run is still active.";
    scheduleNextRun("active-run-skip");
    return;
  }

  const state = await createRun(
    {
      runId: `scheduled-mining-${new Date().toISOString().replaceAll(":", "-").replace(/\.\d{3}Z$/, "Z")}`,
      mode: schedulerState.mode,
      engine: schedulerState.engine,
      objective: schedulerState.objective,
      rounds: schedulerState.rounds,
      batchSize: schedulerState.batchSize,
      concurrency: schedulerState.concurrency,
      ownerId: AUTO_RUN_OWNER_ID,
    },
    "scheduler",
    systemAuthContext(AUTO_RUN_OWNER_ID),
  );
  schedulerState.lastRunId = state.runId;
  schedulerState.lastSkipReason = null;
  scheduleNextRun("scheduled-run-started");
}

function scheduleNextRun(reason) {
  const next = new Date(Date.now() + schedulerState.intervalMinutes * 60_000);
  schedulerState.nextRunAt = schedulerState.enabled ? next.toISOString() : null;
  schedulerState.lastScheduleReason = reason;
}

async function buildRunsIndex(authContext = systemAuthContext()) {
  const dirs = await visibleRunDirs(await listRunDirs(RUNS_DIR), authContext);
  const diversityStats = await latestDiversityStats(dirs);
  return {
    active: [...activeRuns.values()].filter((run) => run.status === "running" && canAccessOwner(authContext, run.ownerId)),
    recent: await Promise.all(dirs.sort().reverse().slice(0, 30).map((runId) => readRun(runId, true))),
    storedRuns: dirs.sort().reverse(),
    scheduler: publicSchedulerState(),
    autoLoop: publicAutoLoopState(authContext),
    diversityStats,
  };
}

async function readLlmRouterState() {
  return await maybeReadJson(path.join(RUNS_DIR, "llm_router_state.json"));
}

async function readRun(runId, compact = false, authContext = systemAuthContext()) {
  const state = activeRuns.get(runId);
  const outputDir = state?.outputDir ?? path.join(RUNS_DIR, runId);
  const meta = state ? { ownerId: state.ownerId ?? "default" } : await readRunMeta(outputDir);
  if (!canAccessOwner(authContext, meta?.ownerId ?? "default")) {
    throw new Error(`Run ${runId} was not found.`);
  }
  const summary = await maybeReadJson(path.join(outputDir, "summary.json"));
  const batch = compact ? null : await maybeReadJson(path.join(outputDir, "batch-round-1.json"));
  const generated = compact ? null : await maybeReadJson(path.join(outputDir, "generated-batch.json"));
  const memory = await maybeReadJson(path.join(outputDir, "memory.json"));
  const pool = compact ? null : await maybeReadJson(path.join(outputDir, "pool.json"));
  const portfolio = compact ? null : await maybeReadJson(path.join(outputDir, "portfolio.json"));
  const progressTail = await readProgressTail(path.join(outputDir, "progress.jsonl"), compact ? 8 : 80);
  const diversityStats = summary?.diversityStats ?? batch?.diversityStats ?? generated?.diversityStats ?? null;
  const artifacts = (await safeReadDir(outputDir))
    .filter((name) => name.endsWith(".json") || name.endsWith(".jsonl"))
    .sort();
  const inferredStatus =
    state?.status ??
    (summary || batch || generated ? "completed" : "unknown");

  return {
    runId,
    state: state ?? { runId, status: inferredStatus, outputDir, engine: summary?.engine ?? "unknown", ownerId: meta?.ownerId ?? "default" },
    meta,
    summary,
    batch,
    generated,
    diversityStats,
    pool,
    portfolio,
    progressTail,
    memorySummary: memory
      ? {
          effectivePool: memory.effectivePool?.length ?? 0,
          deprecatedPool: memory.deprecatedPool?.length ?? 0,
          trajectories: memory.trajectories?.length ?? 0,
        }
      : null,
    artifacts,
  };
}

async function optimizeIdea(idea) {
  const ideaId = crypto.randomUUID();
  let optimized;
  try {
    optimized = await optimizeIdeaWithOpenAI(idea);
  } catch {
    optimized = fallbackOptimizedIdea(idea);
  }
  const payload = {
    ideaId,
    optimized,
    rawIdea: idea,
    ts: Date.now(),
  };
  await saveOptimizedIdea(payload);
  return {
    ideaId,
    optimized,
    rawIdea: idea,
  };
}

async function optimizeIdeaWithOpenAI(idea) {
  if (!process.env.OPENAI_API_KEY) {
    throw new Error("OPENAI_API_KEY is not configured.");
  }

  const response = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: OPENAI_OPTIMIZE_MODEL,
      response_format: { type: "json_object" },
      messages: [
        { role: "system", content: OPTIMIZE_IDEA_SYSTEM_PROMPT },
        { role: "user", content: idea },
      ],
    }),
  });

  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    throw new Error(`OpenAI idea optimization failed with ${response.status}: ${JSON.stringify(payload)}`);
  }
  const content = payload?.choices?.[0]?.message?.content;
  if (!content) {
    throw new Error("OpenAI idea optimization returned no message content.");
  }
  return parseOptimizedIdea(content);
}

function parseOptimizedIdea(content) {
  const parsed = JSON.parse(content);
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error("Optimized idea JSON must be an object.");
  }
  if (typeof parsed.objective !== "string" || !parsed.objective.trim()) {
    throw new Error("Optimized idea JSON is missing objective.");
  }
  if (!OPTIMIZE_IDEA_CATEGORIES.has(parsed.category)) {
    throw new Error("Optimized idea JSON has an invalid category.");
  }
  if (typeof parsed.hypothesis !== "string" || !parsed.hypothesis.trim()) {
    throw new Error("Optimized idea JSON is missing hypothesis.");
  }
  if (!Array.isArray(parsed.constraints) || !Array.isArray(parsed.suggested_data_fields)) {
    throw new Error("Optimized idea JSON is missing array fields.");
  }
  return parsed;
}

function fallbackOptimizedIdea(idea) {
  return {
    objective: idea,
    category: "QUALITY",
    hypothesis: idea,
    constraints: [],
    suggested_data_fields: [],
  };
}

async function saveOptimizedIdea(payload) {
  fs.mkdirSync(IDEAS_DIR, { recursive: true });
  await writeFile(path.join(IDEAS_DIR, `${sanitizeIdeaId(payload.ideaId)}.json`), `${JSON.stringify(payload, null, 2)}\n`, "utf8");
}

async function readOptimizedIdea(ideaId) {
  try {
    return JSON.parse(await readFile(path.join(IDEAS_DIR, `${sanitizeIdeaId(ideaId)}.json`), "utf8"));
  } catch (error) {
    if (error?.code === "ENOENT") return null;
    throw error;
  }
}

async function latestDiversityStats(runDirs) {
  for (const runId of [...runDirs].sort().reverse()) {
    const outputDir = path.join(RUNS_DIR, runId);
    const summary = await maybeReadJson(path.join(outputDir, "summary.json"));
    if (summary?.diversityStats) return summary.diversityStats;
    const generated = await maybeReadJson(path.join(outputDir, "generated-batch.json"));
    if (generated?.diversityStats) return generated.diversityStats;
    const batch = await maybeReadJson(path.join(outputDir, "batch-round-1.json"));
    if (batch?.diversityStats) return batch.diversityStats;
  }
  return null;
}

async function readRunDiversityStats(outputDir) {
  const summary = await maybeReadJson(path.join(outputDir, "summary.json"));
  if (summary?.diversityStats) return summary.diversityStats;
  const generated = await maybeReadJson(path.join(outputDir, "generated-batch.json"));
  if (generated?.diversityStats) return generated.diversityStats;
  const batch = await maybeReadJson(path.join(outputDir, "batch-round-1.json"));
  return batch?.diversityStats ?? null;
}

function buildRunCommand({ engine, mode, objective, rounds, batchSize, concurrency, outputDir, repairContextPath }) {
  if (engine === "legacy-js") {
    const args = [
      path.resolve(__dirname, "./agentic_alpha_lab.mjs"),
      "--mode",
      mode,
      "--objective",
      objective,
      "--rounds",
      String(rounds),
      "--batch-size",
      String(batchSize),
      "--concurrency",
      String(concurrency),
      "--output-dir",
      outputDir,
      "--verbose",
      "true",
    ];
    if (repairContextPath) args.push("--repair-context", repairContextPath);
    return {
      bin: process.execPath,
      args,
    };
  }
  return {
    bin: PYTHON_BIN,
    args: [
      "-m",
      "alpha_miner.main",
      "--mode",
      mode,
      "--objective",
      objective,
      "--rounds",
      String(rounds),
      "--batch-size",
      String(batchSize),
      "--concurrency",
      String(concurrency),
      "--output-dir",
      outputDir,
      "--verbose",
      "true",
    ],
  };
}

async function analyzeIdea(description, authContext = systemAuthContext()) {
  const normalizedDescription = String(description ?? "").trim();
  if (!normalizedDescription) {
    throw new Error("Please enter a research idea before asking AI to improve it.");
  }

  const ideaId = sanitizeIdeaId(
    `idea-${new Date().toISOString().replaceAll(":", "-").replace(/\.\d{3}Z$/, "Z")}-${randomUUID().slice(0, 8)}`,
  );
  const createdAt = new Date().toISOString();
  let draft;
  let usedOpenAI = false;
  let fallbackReason = null;

  if (process.env.OPENAI_API_KEY) {
    try {
      draft = await analyzeIdeaWithOpenAI(normalizedDescription);
      usedOpenAI = true;
    } catch (error) {
      fallbackReason = error instanceof Error ? error.message : String(error);
      draft = fallbackIdeaDraft(normalizedDescription);
      draft.riskWarnings = [
        "OpenAI 当前不可用，本草案由本地规则生成；建议人工检查后再启动 run。",
        ...draft.riskWarnings,
      ];
    }
  } else {
    fallbackReason = "OPENAI_API_KEY is not configured.";
    draft = fallbackIdeaDraft(normalizedDescription);
  }

  const idea = normalizeIdeaDraft({
    ...draft,
    ideaId,
    rawIdea: normalizedDescription,
    usedOpenAI,
    fallbackReason,
    ownerId: authContext.userId,
    linkedRunIds: [],
    createdAt,
    updatedAt: createdAt,
  });
  return idea;
}

async function analyzeIdeaWithOpenAI(description) {
  const response = await fetch("https://api.openai.com/v1/responses", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: OPENAI_IDEA_MODEL,
      input: [
        {
          role: "system",
          content: [
            "你是 WorldQuant BRAIN 因子研究方向编辑助手。",
            "你的任务是把用户的自然语言想法规范化为可执行的因子挖掘方向，而不是给出投资建议或承诺收益。",
            "请保留用户假设，补全可测试方向，强调低相关性、低换手、样本外稳定性、提交闸门和概念多样性。",
            "输出必须适合传给自动因子挖掘 agent 的 objective。",
          ].join(""),
        },
        {
          role: "user",
          content: buildIdeaPrompt(description),
        },
      ],
      text: {
        format: {
          type: "json_schema",
          name: "worldquant_brain_idea_draft",
          schema: ideaDraftSchema(),
          strict: true,
        },
      },
    }),
  });

  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    throw new Error(`OpenAI idea analysis failed with ${response.status}: ${JSON.stringify(payload)}`);
  }

  const text = extractResponseText(payload);
  if (!text) {
    throw new Error("OpenAI idea analysis returned no output text.");
  }
  return JSON.parse(text);
}

function buildIdeaPrompt(description) {
  return [
    "请把下面的自由描述想法，整理成 QuantBrain 可用于 WorldQuant BRAIN 自动挖掘公式化 Alpha 的研究方向草案。",
    "",
    "要求：",
    "1. 用中文输出，researchObjective 可以中英混合，但必须清晰可执行。",
    "2. 不要直接生成完整 Alpha 公式；重点是改进挖掘方向、目标数据族、算子倾向和风险约束。",
    "3. targetFamilies 从这些概念里选择或组合：price-volume, fundamental-quality, fundamental-efficiency, estimate-revision, sentiment-news, risk-volatility, liquidity-turnover, valuation。",
    "4. operatorBias 写适合 BRAIN 表达式探索的算子偏好，例如 rank, ts_rank, ts_mean, ts_decay_linear, group_neutralize, winsorize, zscore, regression_neut。",
    "5. suggestedRun 默认保守：mode=loop, rounds=2, batchSize=3；如果想法很宽泛可以建议 rounds=1。",
    "6. guardrails 必须包含低换手、低拥挤、概念多样性、正 test stability、不能只追求 IS Sharpe。",
    "7. riskWarnings 必须指出该想法可能过拟合、同质化或需要 BRAIN 模拟验证的地方。",
    "",
    "用户原始想法：",
    description,
  ].join("\n");
}

function ideaDraftSchema() {
  return {
    type: "object",
    additionalProperties: false,
    properties: {
      researchObjective: { type: "string" },
      targetFamilies: {
        type: "array",
        items: { type: "string" },
      },
      operatorBias: {
        type: "array",
        items: { type: "string" },
      },
      guardrails: {
        type: "array",
        items: { type: "string" },
      },
      improvementPoints: {
        type: "array",
        items: { type: "string" },
      },
      riskWarnings: {
        type: "array",
        items: { type: "string" },
      },
      suggestedRun: {
        type: "object",
        additionalProperties: false,
        properties: {
          mode: { type: "string", enum: ["generate", "evaluate", "loop"] },
          rounds: { type: "integer" },
          batchSize: { type: "integer" },
          objective: { type: "string" },
        },
        required: ["mode", "rounds", "batchSize", "objective"],
      },
    },
    required: [
      "researchObjective",
      "targetFamilies",
      "operatorBias",
      "guardrails",
      "improvementPoints",
      "riskWarnings",
      "suggestedRun",
    ],
  };
}

function extractResponseText(payload) {
  if (typeof payload?.output_text === "string") return payload.output_text;
  const chunks = [];
  for (const item of payload?.output ?? []) {
    for (const content of item.content ?? []) {
      if (typeof content.text === "string") chunks.push(content.text);
    }
  }
  return chunks.join("").trim();
}

function fallbackIdeaDraft(description) {
  const families = inferTargetFamilies(description);
  const operatorBias = inferOperatorBias(families);
  const familyText = families.join(", ");
  const objective = truncateText(
    [
      `User hypothesis: ${description}`,
      `Focus families: ${familyText}.`,
      "Generate diverse WorldQuant BRAIN formula alphas with low crowding, low turnover, robust test stability, and explicit hypothesis-to-expression trace.",
      `Prefer operators: ${operatorBias.join(", ")}.`,
      "Avoid minor rewrites of the same concept and do not promote candidates using IS Sharpe alone.",
    ].join(" "),
    1400,
  );

  return {
    researchObjective: objective,
    targetFamilies: families,
    operatorBias,
    guardrails: defaultIdeaGuardrails(),
    improvementPoints: [
      "将自然语言想法改写为可传给因子挖掘 agent 的 objective。",
      "补充低相关性、低换手和样本外稳定性约束，降低只拟合 IS 表现的风险。",
      "保留人工确认步骤，草案不会自动提交 Alpha。",
    ],
    riskWarnings: [
      "本地 fallback 只能做关键词归类，不能深度理解复杂经济假设。",
      "该方向必须通过 BRAIN 模拟和提交检查验证，不能凭研究假设直接提交。",
      "如果生成结果集中在同一概念变体上，需要扩大 targetFamilies 或加入负向样本约束。",
    ],
    suggestedRun: {
      mode: "loop",
      rounds: 2,
      batchSize: 3,
      objective,
    },
  };
}

function inferTargetFamilies(description) {
  const text = description.toLowerCase();
  const rules = [
    ["fundamental-quality", ["质量", "盈利", "利润", "毛利", "roe", "roa", "income", "earnings", "quality"]],
    ["fundamental-efficiency", ["效率", "周转", "库存", "现金流", "营运", "turnover", "inventory", "cash flow", "efficiency"]],
    ["estimate-revision", ["预期", "一致预期", "分析师", "上调", "下调", "revision", "estimate", "analyst"]],
    ["sentiment-news", ["新闻", "情绪", "舆情", "社媒", "sentiment", "news", "social"]],
    ["risk-volatility", ["波动", "风险", "回撤", "volatility", "risk", "drawdown", "beta"]],
    ["liquidity-turnover", ["流动性", "换手", "成交量", "volume", "liquidity", "turnover"]],
    ["valuation", ["估值", "市盈率", "市净率", "便宜", "贵", "valuation", "pe", "pb"]],
    ["price-volume", ["价量", "动量", "反转", "趋势", "momentum", "reversal", "trend", "price"]],
  ];
  const matches = rules
    .filter(([, keywords]) => keywords.some((keyword) => text.includes(keyword)))
    .map(([family]) => family);
  return matches.length ? [...new Set(matches)].slice(0, 4) : ["fundamental-quality", "price-volume"];
}

function inferOperatorBias(families) {
  const operators = new Set(["rank", "ts_rank", "ts_mean", "group_neutralize", "winsorize"]);
  if (families.includes("price-volume") || families.includes("liquidity-turnover")) {
    ["ts_decay_linear", "ts_delta", "ts_zscore"].forEach((item) => operators.add(item));
  }
  if (families.includes("fundamental-quality") || families.includes("fundamental-efficiency")) {
    ["zscore", "group_rank", "ts_backfill"].forEach((item) => operators.add(item));
  }
  if (families.includes("estimate-revision") || families.includes("sentiment-news")) {
    ["ts_mean", "ts_sum", "hump"].forEach((item) => operators.add(item));
  }
  if (families.includes("risk-volatility")) {
    ["ts_std_dev", "regression_neut", "scale"].forEach((item) => operators.add(item));
  }
  return [...operators].slice(0, 10);
}

function defaultIdeaGuardrails() {
  return [
    "保持概念多样性，避免只生成同一类公式的窗口参数变体。",
    "优先选择低拥挤、低自相关、低换手的候选方向。",
    "必须同时观察 IS 与 test stability，不因单次 IS Sharpe 高就提交。",
    "保留假设到表达式的轨迹记录，优化时不要让公式偏离原始经济假设。",
    "候选 Alpha 仍需通过 WorldQuant BRAIN 官方模拟和提交检查。",
  ];
}

function normalizeIdeaDraft(raw) {
  const suggestedRun = raw.suggestedRun ?? {};
  const researchObjective = truncateText(
    String(raw.researchObjective || suggestedRun.objective || raw.rawIdea || DEFAULT_OBJECTIVE).trim(),
    1800,
  );
  const mode = ["generate", "evaluate", "loop"].includes(suggestedRun.mode) ? suggestedRun.mode : "loop";
  const rounds = clampInt(suggestedRun.rounds, 2, 1, 6);
  const batchSize = clampInt(suggestedRun.batchSize, 3, 1, 10);
  return {
    ideaId: sanitizeIdeaId(raw.ideaId),
    rawIdea: String(raw.rawIdea ?? "").trim(),
    researchObjective,
    targetFamilies: cleanStringArray(raw.targetFamilies, ["fundamental-quality", "price-volume"], 6),
    operatorBias: cleanStringArray(raw.operatorBias, ["rank", "ts_rank", "group_neutralize"], 12),
    guardrails: cleanStringArray(raw.guardrails, defaultIdeaGuardrails(), 10),
    improvementPoints: cleanStringArray(raw.improvementPoints, [], 10),
    riskWarnings: cleanStringArray(raw.riskWarnings, [], 10),
    suggestedRun: {
      mode,
      rounds,
      batchSize,
      objective: truncateText(String(suggestedRun.objective || researchObjective).trim(), 1800),
    },
    usedOpenAI: raw.usedOpenAI === true,
    fallbackReason: raw.fallbackReason ?? null,
    ownerId: sanitizeOwnerId(raw.ownerId ?? "default"),
    linkedRunIds: Array.isArray(raw.linkedRunIds) ? raw.linkedRunIds.map(sanitizeRunId) : [],
    createdAt: raw.createdAt ?? new Date().toISOString(),
    updatedAt: raw.updatedAt ?? new Date().toISOString(),
  };
}

function cleanStringArray(values, fallback, limit) {
  const source = Array.isArray(values) ? values : fallback;
  const seen = new Set();
  const output = [];
  for (const value of source) {
    const item = String(value ?? "").trim();
    if (!item || seen.has(item)) continue;
    seen.add(item);
    output.push(item);
    if (output.length >= limit) break;
  }
  return output;
}

async function buildIdeasIndex(authContext = systemAuthContext()) {
  const files = (await safeReadDir(IDEAS_DIR)).filter((name) => name.endsWith(".json")).sort().reverse();
  const items = [];
  for (const file of files.slice(0, 50)) {
    const idea = await maybeReadJson(path.join(IDEAS_DIR, file));
    if (!idea?.ideaId) continue;
    if (!canAccessOwner(authContext, idea.ownerId ?? "default")) continue;
    items.push({
      ideaId: idea.ideaId,
      ownerId: idea.ownerId ?? "default",
      rawIdea: idea.rawIdea,
      researchObjective: idea.researchObjective,
      targetFamilies: idea.targetFamilies,
      suggestedRun: idea.suggestedRun,
      usedOpenAI: idea.usedOpenAI,
      fallbackReason: idea.fallbackReason,
      linkedRunIds: idea.linkedRunIds ?? [],
      createdAt: idea.createdAt,
      updatedAt: idea.updatedAt,
    });
  }
  return {
    ideasDir: IDEAS_DIR,
    total: files.length,
    items,
  };
}

async function readIdea(ideaId, authContext = systemAuthContext()) {
  const idea = await maybeReadJson(ideaPath(ideaId));
  if (!idea) {
    throw new Error(`Idea ${ideaId} was not found.`);
  }
  if (!canAccessOwner(authContext, idea.ownerId ?? "default")) {
    throw new Error(`Idea ${ideaId} was not found.`);
  }
  return idea;
}

async function saveIdea(idea) {
  const normalized = normalizeIdeaDraft(idea);
  await mkdir(IDEAS_DIR, { recursive: true });
  await writeFile(ideaPath(normalized.ideaId), `${JSON.stringify(normalized, null, 2)}\n`);
  return normalized;
}

async function createRunFromIdea(ideaId, input = {}, authContext = systemAuthContext()) {
  const idea = await readIdea(ideaId, authContext);
  if (hasRunningRun() && input.force !== true) {
    return {
      error: "A run is already active.",
      active: runningRuns(authContext),
    };
  }

  const suggestedRun = idea.suggestedRun ?? {};
  const state = await createRun(
    {
      runId:
        input.runId ??
        `idea-${ideaId}-${new Date().toISOString().replaceAll(":", "-").replace(/\.\d{3}Z$/, "Z")}`,
      mode: normalizeMode(input.mode ?? suggestedRun.mode ?? "loop"),
      engine: normalizeEngine(input.engine ?? suggestedRun.engine ?? DEFAULT_ENGINE),
      objective: String(input.objective ?? suggestedRun.objective ?? idea.researchObjective ?? DEFAULT_OBJECTIVE),
      rounds: clampInt(input.rounds ?? suggestedRun.rounds, 2, 1, 6),
      batchSize: clampInt(input.batchSize ?? suggestedRun.batchSize, 3, 1, 10),
      ownerId: authContext.userId,
    },
    "idea",
    authContext,
  );
  idea.linkedRunIds = [...new Set([...(idea.linkedRunIds ?? []), state.runId])];
  idea.updatedAt = new Date().toISOString();
  await saveIdea(idea);
  return {
    ideaId,
    runId: state.runId,
    pid: state.pid,
    engine: state.engine,
    status: state.status,
    outputDir: state.outputDir,
  };
}

function ideaPath(ideaId) {
  return path.join(IDEAS_DIR, `${sanitizeIdeaId(ideaId)}.json`);
}

async function handleRunFinished(runState) {
  if (!AUTO_REPAIR_ENABLED && !AUTO_SUBMIT_ENABLED) return;
  autoLoopState.diversityStats = (await readRunDiversityStats(runState.outputDir)) ?? autoLoopState.diversityStats ?? null;
  if (!["evaluate", "loop"].includes(runState.mode)) {
    if (runState.source === "repair") await clearActiveRepair(runState.runId);
    await saveAutoLoopState();
    return;
  }
  if (runState.engine !== "legacy-js") {
    if (runState.source === "repair") await clearActiveRepair(runState.runId);
    await saveAutoLoopState();
    return;
  }

  const candidates = await loadRunScoredCandidates(runState.outputDir);
  const bestRepairCandidate = chooseRepairCandidate(candidates);
  const submitted = await tryAutoSubmitCandidates(candidates, runState.runId, runState.ownerId);
  if (submitted) {
    if (runState.source === "repair") await clearActiveRepair(runState.runId);
    await startNextAutoRepairRun();
    return;
  }

  if (bestRepairCandidate) {
    const priorDepth = Number(runState.repairDepth ?? -1);
    const repairDepth = runState.source === "repair" ? priorDepth + 1 : 0;
    const gate = evaluateCandidateGate(bestRepairCandidate);
    await enqueueRepairFromGate({
      alpha: summarizeCandidateAsAlpha(bestRepairCandidate),
      gate,
      source: runState.source === "repair" ? "repair-run-finished" : "run-finished",
      sourceRunId: runState.runId,
      repairDepth,
      rootAlphaId: runState.parentAlphaId ?? bestRepairCandidate.alphaId,
      ownerId: runState.ownerId,
    });
  } else {
    pushAutoLoopEvent({ type: "no-repair-candidate", runId: runState.runId, ownerId: runState.ownerId });
  }

  if (runState.source === "repair") await clearActiveRepair(runState.runId);
  await saveAutoLoopState();
  await startNextAutoRepairRun();
}

async function tryAutoSubmitCandidates(candidates, sourceRunId, ownerId = "default") {
  if (!AUTO_SUBMIT_ENABLED) return null;
  const ready = candidates
    .filter((candidate) => candidate.alphaId && evaluateCandidateGate(candidate).ok)
    .sort((left, right) => (right.totalScore ?? -Infinity) - (left.totalScore ?? -Infinity));
  for (const candidate of ready) {
    const result = await submitAlpha(candidate.alphaId, {
      force: false,
      queueRepairOnGateFailure: false,
      source: "auto-run-complete",
      sourceRunId,
      ownerId,
    });
    if (result.submitted) {
      pushAutoLoopEvent({ type: "auto-submitted", runId: sourceRunId, alphaId: candidate.alphaId, ownerId });
      await saveAutoLoopState();
      return result;
    }
  }
  return null;
}

async function enqueueRepairFromGate({ alpha, gate, source, sourceRunId, repairDepth = 0, rootAlphaId = null, ownerId = "default" }) {
  if (!AUTO_REPAIR_ENABLED) {
    pushAutoLoopEvent({ type: "repair-disabled", alphaId: alpha?.id ?? gate?.alphaId ?? null, ownerId, source });
    await saveAutoLoopState();
    return { queued: false, runId: null, reason: "auto repair disabled" };
  }
  const alphaId = alpha?.id ?? gate?.alphaId ?? null;
  if (!alphaId) return { queued: false, runId: null, reason: "missing alpha id" };
  if (repairDepth >= AUTO_REPAIR_MAX_ROUNDS) {
    pushAutoLoopEvent({ type: "repair-budget-exhausted", alphaId, ownerId, repairDepth, source });
    await saveAutoLoopState();
    return { queued: false, runId: null, reason: "repair budget exhausted" };
  }
  if (autoLoopState.queue.some((item) => item.parentAlphaId === alphaId) || autoLoopState.activeRepair?.parentAlphaId === alphaId) {
    return { queued: false, runId: autoLoopState.activeRepair?.runId ?? null, reason: "repair already queued" };
  }

  const item = buildRepairQueueItem({ alpha, gate, source, sourceRunId, repairDepth, rootAlphaId, ownerId });
  autoLoopState.queue.push(item);
  autoLoopState.lastAction = `Queued repair for ${alphaId}`;
  pushAutoLoopEvent({ type: "repair-queued", alphaId, ownerId, source, repairDepth, failedChecks: item.failedChecks });
  await saveAutoLoopState();
  return { queued: true, runId: null, reason: "queued" };
}

function buildRepairQueueItem({ alpha, gate, source, sourceRunId, repairDepth, rootAlphaId, ownerId = "default" }) {
  const alphaId = alpha?.id ?? gate?.alphaId;
  const failedChecks = failedCheckNames(gate);
  const createdAt = new Date().toISOString();
  return {
    id: sanitizeRunId(`repair-item-${alphaId}-${createdAt}`),
    parentAlphaId: alphaId,
    rootAlphaId: rootAlphaId ?? alphaId,
    expression: alpha?.expression ?? null,
    ownerId: sanitizeOwnerId(ownerId),
    gate,
    alpha,
    failedChecks,
    repairDepth,
    source,
    sourceRunId,
    status: "queued",
    nextAction: repairNextAction(failedChecks, alpha),
    createdAt,
    updatedAt: createdAt,
  };
}

async function startNextAutoRepairRun() {
  if (!AUTO_REPAIR_ENABLED) return null;
  if (hasRunningRun()) return null;
  if (autoLoopState.activeRepair) return null;
  const item = autoLoopState.queue.shift();
  if (!item) {
    await saveAutoLoopState();
    return null;
  }
  const runId = sanitizeRunId(
    `repair-${item.parentAlphaId}-${new Date().toISOString().replaceAll(":", "-").replace(/\.\d{3}Z$/, "Z")}`,
  );
  item.status = "running";
  item.runId = runId;
  item.updatedAt = new Date().toISOString();
  autoLoopState.activeRepair = item;
  autoLoopState.lastAction = `Started repair run ${runId}`;
  pushAutoLoopEvent({ type: "repair-started", runId, alphaId: item.parentAlphaId, ownerId: item.ownerId, repairDepth: item.repairDepth });
  await saveAutoLoopState();
  await createRun(
    {
      runId,
      mode: "evaluate",
      engine: AUTO_REPAIR_ENGINE,
      objective: buildRepairObjective(item),
      rounds: 1,
      batchSize: AUTO_REPAIR_BATCH_SIZE,
      repairContext: item,
      parentAlphaId: item.rootAlphaId ?? item.parentAlphaId,
      ownerId: item.ownerId ?? "default",
    },
    "repair",
    systemAuthContext(item.ownerId ?? "default"),
  );
  return runId;
}

async function clearActiveRepair(runId) {
  if (!autoLoopState.activeRepair || autoLoopState.activeRepair.runId !== runId) return;
  autoLoopState.activeRepair = null;
  await saveAutoLoopState();
}

async function loadRunScoredCandidates(outputDir) {
  const files = (await safeReadDir(outputDir)).filter((name) => /^batch-round-\d+\.json$/.test(name)).sort();
  const candidates = [];
  for (const file of files) {
    const payload = await maybeReadJson(path.join(outputDir, file));
    for (const candidate of payload?.scored ?? []) {
      if (candidate?.alphaId) candidates.push(candidate);
    }
  }
  return candidates;
}

function chooseRepairCandidate(candidates) {
  return candidates
    .filter((candidate) => candidate.alphaId && !evaluateCandidateGate(candidate).ok)
    .sort((left, right) => repairPriority(right) - repairPriority(left))[0] ?? null;
}

function repairPriority(candidate) {
  const metrics = candidate.metrics ?? {};
  const gate = evaluateCandidateGate(candidate);
  const failures = new Set(failedCheckNames(gate));
  let score = Number(candidate.totalScore ?? 0);
  if (metrics.testSharpe !== null && metrics.testSharpe > 0) score += 1.5;
  if (failures.has("LOW_SHARPE") || failures.has("LOW_FITNESS")) score += 1.0;
  if (failures.has("SELF_CORRELATION")) score += 0.6;
  if (failures.has("LOW_SUB_UNIVERSE_SHARPE")) score += 0.4;
  if (metrics.turnover !== null && metrics.turnover > 0.7) score -= 1.0;
  if (metrics.testSharpe !== null && metrics.testSharpe <= 0) score -= 0.5;
  return score;
}

function evaluateCandidateGate(candidate) {
  const checks = candidate.checks ?? [];
  const nonPassChecks = checks.filter((check) => check.result !== "PASS");
  const testSharpe = toNumber(candidate.metrics?.testSharpe);
  const reasons = [];
  if (checks.length === 0) reasons.push("No IS submission checks are available yet.");
  if (nonPassChecks.length > 0) {
    reasons.push(`Checks not all PASS: ${nonPassChecks.map((check) => `${check.name}:${check.result}`).join(", ")}.`);
  }
  if (!Number.isFinite(testSharpe) || testSharpe <= 0) {
    reasons.push(`testSharpe must be positive, got ${Number.isFinite(testSharpe) ? testSharpe : "missing"}.`);
  }
  return {
    ok: reasons.length === 0,
    reasons,
    alphaId: candidate.alphaId ?? null,
    status: "UNSUBMITTED",
    stage: candidate.stage ?? "IS",
    testSharpe: Number.isFinite(testSharpe) ? testSharpe : null,
    checks: checks.map((check) => ({
      name: check.name,
      result: check.result,
      limit: check.limit,
      value: check.value,
    })),
  };
}

function summarizeCandidateAsAlpha(candidate) {
  const metrics = candidate.metrics ?? {};
  return {
    id: candidate.alphaId ?? null,
    status: "UNSUBMITTED",
    stage: candidate.stage ?? "IS",
    dateSubmitted: null,
    grade: null,
    expression: candidate.expression ?? null,
    isSharpe: toNumber(metrics.isSharpe),
    isFitness: toNumber(metrics.isFitness),
    turnover: toNumber(metrics.turnover),
    testSharpe: toNumber(metrics.testSharpe),
    testFitness: toNumber(metrics.testFitness),
  };
}

function buildRepairObjective(item) {
  const failed = item.failedChecks?.length ? item.failedChecks.join(", ") : "submission gate failure";
  return [
    `Repair blocked BRAIN alpha ${item.parentAlphaId}.`,
    `Original expression: ${item.expression ?? "unknown"}.`,
    `Failed checks: ${failed}.`,
    `Gate reasons: ${(item.gate?.reasons ?? []).join(" ")}`,
    "Generate targeted variants only: preserve the economic thesis where useful, materially change concept if self-correlation fails, improve Sharpe/Fitness without chasing IS only, keep turnover inside 1%-70%, and require positive test stability before submission.",
  ].join(" ");
}

function repairNextAction(failedChecks, alpha) {
  const names = new Set(failedChecks);
  const actions = [];
  if (names.has("LOW_SHARPE") || names.has("LOW_FITNESS")) actions.push("strength repair via peer-relative blend / denominator swap");
  if (names.has("HIGH_TURNOVER")) actions.push("turnover repair via decay and slower windows");
  if (names.has("LOW_SUB_UNIVERSE_SHARPE")) actions.push("sub-universe repair via group/subindustry relative transform");
  if (names.has("CONCENTRATED_WEIGHT")) actions.push("concentration repair via rank/group_rank/truncation-safe transforms");
  if (names.has("SELF_CORRELATION")) actions.push("crowding repair via material concept/horizon/group change");
  if (alpha?.testSharpe !== null && alpha?.testSharpe <= 0) actions.push("robustness repair because testSharpe is non-positive");
  return actions.join("; ") || "targeted repair from gate feedback";
}

function failedCheckNames(gate) {
  return [...new Set((gate?.checks ?? []).filter((check) => check.result !== "PASS").map((check) => check.name))];
}

function recordAutoSubmission(alphaId, alpha, gate, source, ownerId = "default") {
  autoLoopState.submitted.push({
    alphaId,
    ownerId: sanitizeOwnerId(ownerId),
    alpha,
    gate,
    source,
    at: new Date().toISOString(),
  });
  autoLoopState.submitted = autoLoopState.submitted.slice(-50);
  autoLoopState.lastAction = `Submitted ${alphaId}`;
  pushAutoLoopEvent({ type: "submitted", alphaId, ownerId: sanitizeOwnerId(ownerId), source });
}

function publicAutoLoopState(authContext = systemAuthContext()) {
  const queue = autoLoopState.queue.filter((item) => canAccessOwner(authContext, item.ownerId ?? "default"));
  const submitted = autoLoopState.submitted.filter((item) => canAccessOwner(authContext, item.ownerId ?? "default"));
  const events = autoLoopState.events.filter((event) => canAccessOwner(authContext, event.ownerId ?? "default"));
  const activeRepair = autoLoopState.activeRepair && canAccessOwner(authContext, autoLoopState.activeRepair.ownerId ?? "default")
    ? autoLoopState.activeRepair
    : null;
  return {
    enabled: AUTO_REPAIR_ENABLED,
    autoSubmitEnabled: AUTO_SUBMIT_ENABLED,
    engine: AUTO_REPAIR_ENGINE,
    generatorStrategy: ALPHA_GENERATOR_STRATEGY,
    experimentalFieldsEnabled: ALPHA_EXPERIMENTAL_FIELDS,
    crowdingPatternThreshold: ALPHA_CROWDING_PATTERN_THRESHOLD,
    familyCooldownRounds: ALPHA_FAMILY_COOLDOWN_ROUNDS,
    maxRepairRounds: AUTO_REPAIR_MAX_ROUNDS,
    repairBatchSize: AUTO_REPAIR_BATCH_SIZE,
    queueLength: queue.length,
    queue: queue.slice(0, 10),
    activeRepair,
    submitted: submitted.slice(-10).reverse(),
    events: events.slice(-30).reverse(),
    diversityStats: autoLoopState.diversityStats ?? null,
    lastAction: autoLoopState.lastAction,
    statePath: AUTO_LOOP_STATE_PATH,
  };
}

async function loadAutoLoopState() {
  const persisted = await maybeReadJson(AUTO_LOOP_STATE_PATH);
  return {
    queue: Array.isArray(persisted?.queue) ? persisted.queue : [],
    activeRepair: null,
    submitted: Array.isArray(persisted?.submitted) ? persisted.submitted : [],
    events: Array.isArray(persisted?.events) ? persisted.events.slice(-100) : [],
    diversityStats: persisted?.diversityStats ?? null,
    lastAction: persisted?.lastAction ?? null,
  };
}

async function saveAutoLoopState() {
  await writeFile(AUTO_LOOP_STATE_PATH, `${JSON.stringify(autoLoopState, null, 2)}\n`, "utf8");
}

function pushAutoLoopEvent(event) {
  autoLoopState.events.push({ ...event, at: new Date().toISOString() });
  autoLoopState.events = autoLoopState.events.slice(-100);
}

async function submitAlpha(alphaId, options = {}) {
  const ownerId = sanitizeOwnerId(options.ownerId ?? "default");
  const session = await authenticateBrain(ownerId);
  const before = await fetchAlpha(session, alphaId);
  const gate = evaluateSubmissionGate(before);
  if (!gate.ok && !options.force) {
    let repairResult = { queued: false, runId: null, reason: "auto repair disabled" };
    if (options.queueRepairOnGateFailure !== false) {
      repairResult = await enqueueRepairFromGate({
        alpha: summarizeAlpha(before),
        gate,
        source: options.source ?? "submit-gate",
        sourceRunId: options.sourceRunId ?? null,
        repairDepth: options.repairDepth ?? 0,
        ownerId,
      });
      const startedRepairRunId = await startNextAutoRepairRun();
      if (startedRepairRunId) repairResult.runId = startedRepairRunId;
    }
    return {
      submitted: false,
      alphaId,
      gate,
      alpha: summarizeAlpha(before),
      repairQueued: repairResult.queued,
      repairRunId: repairResult.runId ?? null,
      repairReason: repairResult.reason ?? null,
      autoLoopState: publicAutoLoopState(systemAuthContext(ownerId, "user")),
    };
  }

  if (options.force) {
    throw new Error("Forced alpha submission is disabled by the auto-loop safety policy.");
  }
  const response = await brainFetch(session, `/alphas/${encodeURIComponent(alphaId)}/submit`, {
    method: "POST",
    body: "{}",
  });
  const text = await response.text();
  const submitResponse = text ? safeJson(text) ?? text : null;
  const after = await fetchAlpha(session, alphaId);
  recordAutoSubmission(alphaId, summarizeAlpha(after), gate, options.source ?? "manual-submit", ownerId);
  await saveAutoLoopState();
  return {
    submitted: true,
    alphaId,
    gate,
    submitResponse,
    alpha: summarizeAlpha(after),
  };
}

async function authenticateBrain(ownerId = "default") {
  const credential = await resolveBrainCredentials(ownerId);
  if (!credential) {
    throw new Error(`BRAIN credentials are not configured for owner ${ownerId}.`);
  }
  const { email, password } = credential;
  const authorization = `Basic ${Buffer.from(`${email}:${password}`).toString("base64")}`;
  const response = await fetch(`${API_ROOT}/authentication`, {
    method: "POST",
    headers: {
      Accept: "application/json;version=2.0",
      Authorization: authorization,
      "Content-Type": "application/json",
      Origin: "https://platform.worldquantbrain.com",
      Referer: "https://platform.worldquantbrain.com/",
    },
    body: "{}",
    redirect: "manual",
  });
  if (!response.ok) {
    throw new Error(`Authentication failed with ${response.status}: ${await response.text()}`);
  }
  const cookieJar = setCookieHeader(getSetCookie(response.headers));
  if (!cookieJar) {
    throw new Error("Authentication succeeded but no session cookie was returned.");
  }
  return { cookieJar };
}

async function fetchAlpha(session, alphaId) {
  const response = await brainFetch(session, `/alphas/${encodeURIComponent(alphaId)}`, {
    method: "GET",
  });
  return response.json();
}

async function brainFetch(session, pathName, init = {}) {
  const response = await fetch(`${API_ROOT}${pathName}`, {
    ...init,
    headers: {
      Accept: "application/json;version=2.0",
      "Content-Type": "application/json",
      Cookie: session.cookieJar,
      Origin: "https://platform.worldquantbrain.com",
      Referer: "https://platform.worldquantbrain.com/",
      ...(init.headers ?? {}),
    },
  });
  if (response.ok) return response;
  const text = await response.text();
  throw new Error(`${init.method ?? "GET"} ${pathName} failed with ${response.status}: ${text}`);
}

function evaluateSubmissionGate(alpha) {
  const status = alpha?.status ?? "UNKNOWN";
  const stage = alpha?.stage ?? "UNKNOWN";
  const checks = alpha?.is?.checks ?? [];
  const nonPassChecks = checks.filter((check) => check.result !== "PASS");
  const testSharpe = toNumber(alpha?.test?.sharpe);
  const reasons = [];

  if (status !== "UNSUBMITTED") {
    reasons.push(`Alpha status is ${status}, expected UNSUBMITTED.`);
  }
  if (checks.length === 0) {
    reasons.push("No IS submission checks are available yet.");
  }
  if (nonPassChecks.length > 0) {
    reasons.push(
      `Checks not all PASS: ${nonPassChecks.map((check) => `${check.name}:${check.result}`).join(", ")}.`,
    );
  }
  if (!Number.isFinite(testSharpe) || testSharpe <= 0) {
    reasons.push(`testSharpe must be positive, got ${Number.isFinite(testSharpe) ? testSharpe : "missing"}.`);
  }

  return {
    ok: reasons.length === 0,
    reasons,
    alphaId: alpha?.id ?? null,
    status,
    stage,
    testSharpe: Number.isFinite(testSharpe) ? testSharpe : null,
    checks: checks.map((check) => ({
      name: check.name,
      result: check.result,
      limit: check.limit,
      value: check.value,
    })),
  };
}

function summarizeAlpha(alpha) {
  return {
    id: alpha?.id ?? null,
    status: alpha?.status ?? null,
    stage: alpha?.stage ?? null,
    dateSubmitted: alpha?.dateSubmitted ?? null,
    grade: alpha?.grade ?? null,
    expression: alpha?.regular?.code ?? null,
    isSharpe: toNumber(alpha?.is?.sharpe),
    isFitness: toNumber(alpha?.is?.fitness),
    turnover: toNumber(alpha?.is?.turnover),
    testSharpe: toNumber(alpha?.test?.sharpe),
    testFitness: toNumber(alpha?.test?.fitness),
  };
}

function publicSchedulerState() {
  return {
    enabled: schedulerState.enabled,
    engine: schedulerState.engine,
    mode: schedulerState.mode,
    objective: schedulerState.objective,
    intervalMinutes: schedulerState.intervalMinutes,
    rounds: schedulerState.rounds,
    batchSize: schedulerState.batchSize,
    lastTickAt: schedulerState.lastTickAt,
    lastRunId: schedulerState.lastRunId,
    lastSkipReason: schedulerState.lastSkipReason,
    nextRunAt: schedulerState.nextRunAt,
    lastScheduleReason: schedulerState.lastScheduleReason,
    authRequired: Boolean(ADMIN_TOKEN || DASHBOARD_USERS.size),
    multiUserConfigured: DASHBOARD_USERS.size > 0,
    schedulerOwnerId: AUTO_RUN_OWNER_ID,
    runsDir: RUNS_DIR,
    ideasDir: IDEAS_DIR,
    credentialsDir: CREDENTIALS_DIR,
    openAIConfigured: Boolean(process.env.OPENAI_API_KEY),
    ideaModel: OPENAI_IDEA_MODEL,
    generatorStrategy: ALPHA_GENERATOR_STRATEGY,
    experimentalFieldsEnabled: ALPHA_EXPERIMENTAL_FIELDS,
    crowdingPatternThreshold: ALPHA_CROWDING_PATTERN_THRESHOLD,
    familyCooldownRounds: ALPHA_FAMILY_COOLDOWN_ROUNDS,
  };
}

function appendLog(state, text) {
  const lines = text.split(/\r?\n/).filter(Boolean);
  for (const line of lines) {
    state.logs.push({ at: new Date().toISOString(), line });
  }
  if (state.logs.length > 200) {
    state.logs = state.logs.slice(-200);
  }
}

function hasRunningRun() {
  return runningRuns().length > 0;
}

function runningRuns(authContext = null) {
  return [...activeRuns.values()].filter((run) => {
    if (run.status !== "running") return false;
    return authContext ? canAccessOwner(authContext, run.ownerId) : true;
  });
}

async function loadUserRegistry() {
  const registryPath = path.resolve(CREDENTIALS_DIR, "user-registry.json");
  const data = await maybeReadJson(registryPath);
  return Array.isArray(data) ? data : [];
}

async function saveUserRegistry(registry) {
  const registryPath = path.resolve(CREDENTIALS_DIR, "user-registry.json");
  await writeFile(registryPath, JSON.stringify(registry, null, 2) + "\n", { mode: 0o600 });
}

async function registerUser(body) {
  const userId = sanitizeOwnerId(String(body?.userId ?? "").trim());
  const email = String(body?.email ?? "").trim();
  const password = String(body?.password ?? "");
  const code = String(body?.registrationCode ?? "").trim();

  if (!userId) return { ok: false, error: "Username is required." };
  if (!email || !email.includes("@")) return { ok: false, error: "Valid email is required." };
  if (!password) return { ok: false, error: "Password is required." };
  if (REGISTRATION_CODE && !safeTokenEquals(code, REGISTRATION_CODE)) {
    return { ok: false, error: "Invalid registration code." };
  }
  // Check if userId already taken
  const existing = userRegistry.find(u => u.userId === userId);
  if (existing) return { ok: false, error: "Username already taken. Choose another." };

  const token = randomBytes(24).toString("hex");
  const entry = { userId, token, createdAt: new Date().toISOString() };
  userRegistry.push(entry);
  await saveUserRegistry(userRegistry);

  // Save BRAIN credentials
  await saveBrainCredentials(userId, { email, password });

  return { ok: true, userId, token };
}

function parseDashboardUsers(value) {
  const users = new Map();
  for (const rawEntry of String(value ?? "").split(/[\n,]+/)) {
    const entry = rawEntry.trim();
    if (!entry) continue;
    const separator = entry.includes("=") ? "=" : ":";
    const [rawUserId, ...rest] = entry.split(separator);
    const token = rest.join(separator).trim();
    const userId = sanitizeOwnerId(rawUserId);
    if (!userId || !token) continue;
    users.set(userId, { token, role: "user" });
  }
  return users;
}

function getAuthContext(req, url) {
  const authRequired = Boolean(ADMIN_TOKEN || DASHBOARD_USERS.size);
  if (!authRequired) return systemAuthContext("default", "admin", false);

  const token = extractAccessToken(req, url);
  if (ADMIN_TOKEN && safeTokenEquals(token, ADMIN_TOKEN)) {
    return systemAuthContext("default", "admin", true);
  }
  for (const [userId, config] of DASHBOARD_USERS.entries()) {
    if (safeTokenEquals(token, config.token)) {
      return { ok: true, userId, role: config.role ?? "user", authRequired: true };
    }
  }
  // Check file-based user registry
  for (const entry of userRegistry) {
    if (safeTokenEquals(token, entry.token)) {
      return { ok: true, userId: entry.userId, role: "user", authRequired: true };
    }
  }
  return { ok: false, userId: null, role: "anonymous", authRequired: true };
}

function extractAccessToken(req, url) {
  const authHeader = req.headers.authorization ?? "";
  const bearer = authHeader.startsWith("Bearer ") ? authHeader.slice("Bearer ".length) : "";
  const headerToken = Array.isArray(req.headers["x-admin-token"])
    ? req.headers["x-admin-token"][0]
    : req.headers["x-admin-token"] ?? "";
  const queryToken = url.searchParams.get("token") ?? "";
  return String(bearer || headerToken || queryToken || "").trim();
}

function safeTokenEquals(actual, expected) {
  const left = Buffer.from(String(actual ?? ""));
  const right = Buffer.from(String(expected ?? ""));
  if (!left.length || left.length !== right.length) return false;
  return timingSafeEqual(left, right);
}

function systemAuthContext(userId = "default", role = "system", authRequired = Boolean(ADMIN_TOKEN || DASHBOARD_USERS.size)) {
  return { ok: true, userId: sanitizeOwnerId(userId), role, authRequired };
}

function isAdminAuth(authContext) {
  return authContext?.role === "admin" || authContext?.role === "system";
}

function canAccessOwner(authContext, ownerId = "default") {
  if (isAdminAuth(authContext)) return true;
  return sanitizeOwnerId(authContext?.userId ?? "") === sanitizeOwnerId(ownerId);
}

async function publicAccountState(authContext) {
  const ownerId = sanitizeOwnerId(authContext.userId ?? "default");
  const credential = await resolveBrainCredentials(ownerId);
  return {
    userId: ownerId,
    role: authContext.role,
    hasBrainCredentials: Boolean(credential),
    usingGlobalFallback: credential?.source === "global-env",
    credentialSource: credential?.source ?? "missing",
    brainEmail: credential?.email ? maskEmail(credential.email) : null,
    credentialsDir: CREDENTIALS_DIR,
    credentialsSecretConfigured: Boolean(CREDENTIALS_SECRET || ADMIN_TOKEN),
    multiUserConfigured: DASHBOARD_USERS.size > 0,
  };
}

async function saveBrainCredentials(ownerId, body) {
  const email = String(body?.email ?? "").trim();
  const password = String(body?.password ?? "");
  if (!email || !password) {
    throw new Error("BRAIN email and password are required.");
  }
  if (!email.includes("@")) {
    throw new Error("BRAIN email does not look valid.");
  }
  const payload = encryptCredentialPayload({
    email,
    password,
    updatedAt: new Date().toISOString(),
  });
  await mkdir(CREDENTIALS_DIR, { recursive: true });
  await writeFile(credentialsPath(ownerId), `${JSON.stringify(payload, null, 2)}\n`, { mode: 0o600 });
  return {
    ok: true,
    userId: sanitizeOwnerId(ownerId),
    hasBrainCredentials: true,
    brainEmail: maskEmail(email),
    credentialSource: "stored",
  };
}

async function deleteBrainCredentials(ownerId) {
  try {
    await unlink(credentialsPath(ownerId));
  } catch {
    // Missing credentials are already deleted.
  }
  return {
    ok: true,
    userId: sanitizeOwnerId(ownerId),
    hasBrainCredentials: Boolean(await resolveBrainCredentials(ownerId)),
  };
}

async function brainCredentialEnv(ownerId) {
  const credential = await resolveBrainCredentials(ownerId);
  if (!credential) return { ok: false, source: "missing", env: {} };
  return {
    ok: true,
    source: credential.source,
    env: {
      WQB_EMAIL: credential.email,
      WQB_PASSWORD: credential.password,
    },
  };
}

async function resolveBrainCredentials(ownerId) {
  const stored = await readStoredBrainCredentials(ownerId);
  if (stored) return { ...stored, source: "stored" };
  if (sanitizeOwnerId(ownerId) === "default" && process.env.WQB_EMAIL && process.env.WQB_PASSWORD) {
    return {
      email: process.env.WQB_EMAIL,
      password: process.env.WQB_PASSWORD,
      source: "global-env",
    };
  }
  return null;
}

async function readStoredBrainCredentials(ownerId) {
  const payload = await maybeReadJson(credentialsPath(ownerId));
  if (!payload) return null;
  return decryptCredentialPayload(payload);
}

function encryptCredentialPayload(value) {
  const key = credentialKey();
  const iv = randomBytes(12);
  const cipher = createCipheriv("aes-256-gcm", key, iv);
  const plaintext = Buffer.from(JSON.stringify(value), "utf8");
  const ciphertext = Buffer.concat([cipher.update(plaintext), cipher.final()]);
  const tag = cipher.getAuthTag();
  return {
    version: 1,
    algorithm: "aes-256-gcm",
    iv: iv.toString("base64"),
    tag: tag.toString("base64"),
    ciphertext: ciphertext.toString("base64"),
    updatedAt: value.updatedAt,
  };
}

function decryptCredentialPayload(payload) {
  const key = credentialKey();
  const decipher = createDecipheriv("aes-256-gcm", key, Buffer.from(payload.iv, "base64"));
  decipher.setAuthTag(Buffer.from(payload.tag, "base64"));
  const plaintext = Buffer.concat([
    decipher.update(Buffer.from(payload.ciphertext, "base64")),
    decipher.final(),
  ]);
  return JSON.parse(plaintext.toString("utf8"));
}

function credentialKey() {
  const secret = CREDENTIALS_SECRET || ADMIN_TOKEN;
  if (!secret || secret.length < 12) {
    throw new Error("Set CREDENTIALS_SECRET before saving per-user BRAIN credentials.");
  }
  return scryptSync(secret, "quantbrain-credentials-v1", 32);
}

function credentialsPath(ownerId) {
  return path.join(CREDENTIALS_DIR, `${sanitizeOwnerId(ownerId)}.json`);
}

function maskEmail(email) {
  const [name, domain] = String(email ?? "").split("@");
  if (!name || !domain) return null;
  return `${name.slice(0, 2)}***@${domain}`;
}

async function writeRunMeta(outputDir, meta) {
  await writeFile(path.join(outputDir, "run-meta.json"), `${JSON.stringify(meta, null, 2)}\n`, "utf8");
}

async function readRunMeta(outputDir) {
  return (await maybeReadJson(path.join(outputDir, "run-meta.json"))) ?? { ownerId: "default" };
}

async function visibleRunDirs(runIds, authContext) {
  if (isAdminAuth(authContext)) return runIds;
  const visible = [];
  for (const runId of runIds) {
    const meta = await readRunMeta(path.join(RUNS_DIR, runId));
    if (canAccessOwner(authContext, meta.ownerId ?? "default")) visible.push(runId);
  }
  return visible;
}

function sanitizeRunId(value) {
  const fallback = `run-${randomUUID()}`;
  const clean = String(value ?? fallback)
    .replace(/[^a-zA-Z0-9._:-]/g, "-")
    .replace(/\.+/g, ".")
    .slice(0, 120);
  return clean && clean !== "." && clean !== ".." ? clean : fallback;
}

function sanitizeIdeaId(value) {
  const fallback = `idea-${randomUUID()}`;
  const clean = String(value ?? fallback)
    .replace(/[^a-zA-Z0-9._:-]/g, "-")
    .replace(/\.+/g, ".")
    .slice(0, 140);
  return clean && clean !== "." && clean !== ".." ? clean : fallback;
}

function sanitizeOwnerId(value) {
  const clean = String(value ?? "default")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9._-]/g, "-")
    .replace(/-+/g, "-")
    .slice(0, 64);
  return clean && clean !== "." && clean !== ".." ? clean : "default";
}

function normalizeMode(value) {
  return ["generate", "evaluate", "loop"].includes(value) ? value : "loop";
}

function normalizeEngine(value) {
  return ["python-v2", "legacy-js"].includes(value) ? value : "python-v2";
}

function clampInt(value, defaultValue, low, high) {
  const number = Number(value);
  if (!Number.isFinite(number)) return defaultValue;
  return Math.max(low, Math.min(high, Math.round(number)));
}

function truncateText(value, limit) {
  const text = String(value ?? "").trim();
  return text.length > limit ? `${text.slice(0, limit - 1)}…` : text;
}

async function readJson(req) {
  const chunks = [];
  for await (const chunk of req) {
    chunks.push(chunk);
  }
  if (chunks.length === 0) return {};
  return JSON.parse(Buffer.concat(chunks).toString("utf8"));
}

async function maybeReadJson(filePath) {
  try {
    return JSON.parse(await readFile(filePath, "utf8"));
  } catch {
    return null;
  }
}

async function readProgressTail(filePath, maxLines) {
  try {
    const text = await readFile(filePath, "utf8");
    return text.split(/\r?\n/).filter(Boolean).slice(-maxLines).map((line) => safeJson(line) ?? { line });
  } catch {
    return [];
  }
}

async function safeReadDir(dirPath) {
  try {
    return await readdir(dirPath);
  } catch {
    return [];
  }
}

async function listRunDirs(dirPath) {
  try {
    const entries = await readdir(dirPath, { withFileTypes: true });
    return entries.filter((entry) => entry.isDirectory()).map((entry) => entry.name);
  } catch {
    return [];
  }
}

function getSetCookie(headers) {
  if (typeof headers.getSetCookie === "function") {
    return headers.getSetCookie();
  }
  const value = headers.get("set-cookie");
  return value ? [value] : [];
}

function setCookieHeader(cookies) {
  return cookies
    .flatMap((cookie) => cookie.split(/,(?=\s*[^;,]+=)/g))
    .map((cookie) => cookie.split(";")[0].trim())
    .filter(Boolean)
    .join("; ");
}

function safeJson(text) {
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}

function toNumber(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function sendJson(res, statusCode, payload) {
  res.writeHead(statusCode, { "Content-Type": "application/json; charset=utf-8" });
  res.end(`${JSON.stringify(payload, null, 2)}\n`);
}

function sendHtml(res, statusCode, html) {
  res.writeHead(statusCode, {
    "Content-Type": "text/html; charset=utf-8",
    "Cache-Control": "no-store",
  });
  res.end(html);
}

function setupHtml() {
  const regCodeRequired = Boolean(REGISTRATION_CODE);
  return `<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>QuantBrain — Setup</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', sans-serif;
  background: #f5f5f7;
  min-height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  -webkit-font-smoothing: antialiased;
}
.card {
  background: #fff;
  border-radius: 18px;
  padding: 40px 44px;
  width: 100%;
  max-width: 420px;
  box-shadow: 0 4px 24px rgba(0,0,0,0.08);
}
.logo {
  width: 44px; height: 44px;
  background: #1d1d1f;
  border-radius: 12px;
  display: flex; align-items: center; justify-content: center;
  font-size: 18px; font-weight: 800; color: #fff;
  margin: 0 auto 20px;
}
h1 { font-size: 22px; font-weight: 700; text-align: center; color: #1d1d1f; margin-bottom: 6px; }
.sub { font-size: 14px; color: #86868b; text-align: center; margin-bottom: 32px; }
.field { display: flex; flex-direction: column; gap: 6px; margin-bottom: 16px; }
label { font-size: 13px; font-weight: 500; color: #3a3a3c; }
input {
  padding: 11px 14px;
  border: 1px solid #d2d2d7;
  border-radius: 10px;
  font-size: 14px;
  outline: none;
  transition: border-color 0.15s;
}
input:focus { border-color: #007aff; box-shadow: 0 0 0 3px rgba(0,122,255,0.12); }
.divider { height: 1px; background: #f2f2f7; margin: 20px 0; }
.section-label { font-size: 11px; font-weight: 600; color: #86868b; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 14px; }
.btn {
  width: 100%; padding: 13px;
  background: #1d1d1f; color: #fff;
  border: none; border-radius: 12px;
  font-size: 15px; font-weight: 600;
  cursor: pointer; margin-top: 8px;
  transition: opacity 0.15s;
}
.btn:hover { opacity: 0.85; }
.btn:disabled { opacity: 0.4; cursor: not-allowed; }
.status { font-size: 13px; margin-top: 14px; text-align: center; min-height: 18px; }
.status.error { color: #ff3b30; }
.status.success { color: #34c759; }
.login-link { text-align: center; margin-top: 20px; font-size: 13px; color: #86868b; }
.login-link a { color: #007aff; text-decoration: none; }
</style>
</head>
<body>
<div class="card">
  <div class="logo">Q</div>
  <h1>Welcome to QuantBrain</h1>
  <p class="sub">Create your account to start mining alphas</p>

  <div class="section-label">Your Account</div>
  <div class="field"><label>Username</label><input id="userId" placeholder="e.g. alice" autocomplete="username"></div>
  ${regCodeRequired ? '<div class="field"><label>Registration Code</label><input id="regCode" type="password" placeholder="Ask your admin for the code"></div>' : ''}

  <div class="divider"></div>
  <div class="section-label">WorldQuant BRAIN Credentials</div>
  <div class="field"><label>BRAIN Email</label><input id="email" type="email" placeholder="your@email.com" autocomplete="email"></div>
  <div class="field"><label>BRAIN Password</label><input id="password" type="password" placeholder="••••••••" autocomplete="current-password"></div>

  <button class="btn" id="registerBtn">Create Account &amp; Enter</button>
  <div class="status" id="status"></div>
  <div class="login-link">Already have an account? <a href="#" id="switchToLogin">Sign in</a></div>

  <div id="loginSection" style="display:none">
    <div class="divider"></div>
    <div class="section-label">Sign In</div>
    <div class="field"><label>Username</label><input id="loginUserId" placeholder="Your username" autocomplete="username"></div>
    <div class="field"><label>Token</label><input id="loginToken" type="password" placeholder="Paste your token" autocomplete="current-password"></div>
    <button class="btn" id="loginBtn" style="background:#007aff">Sign In</button>
    <div class="status" id="loginStatus"></div>
    <div class="login-link"><a href="#" id="switchToRegister">← Create new account</a></div>
  </div>
</div>
<script>
(function(){
  const reg = document.getElementById('registerBtn');
  const st = document.getElementById('status');

  reg.addEventListener('click', async function() {
    st.className = 'status'; st.textContent = '';
    const userId = document.getElementById('userId').value.trim();
    const email = document.getElementById('email').value.trim();
    const password = document.getElementById('password').value;
    const regCode = document.getElementById('regCode')?.value ?? '';
    if (!userId || !email || !password) { st.className='status error'; st.textContent='All fields are required.'; return; }
    reg.disabled = true; reg.textContent = 'Creating account…';
    try {
      const r = await fetch('/account/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ userId, email, password, registrationCode: regCode })
      });
      const d = await r.json();
      if (!d.ok) { st.className='status error'; st.textContent = d.error ?? 'Registration failed.'; reg.disabled=false; reg.textContent='Create Account & Enter'; return; }
      localStorage.setItem('qb_token', d.token);
      localStorage.setItem('qb_user', d.userId);
      st.className='status success'; st.textContent = 'Account created! Entering dashboard…';
      setTimeout(() => location.href = '/', 800);
    } catch(e) { st.className='status error'; st.textContent='Network error: ' + e.message; reg.disabled=false; reg.textContent='Create Account & Enter'; }
  });

  document.getElementById('switchToLogin').addEventListener('click', function(e) {
    e.preventDefault();
    document.getElementById('loginSection').style.display = '';
    document.querySelector('.card > *:not(#loginSection)');
  });
  document.getElementById('switchToRegister').addEventListener('click', function(e) {
    e.preventDefault();
    document.getElementById('loginSection').style.display = 'none';
  });

  document.getElementById('loginBtn').addEventListener('click', async function() {
    const st2 = document.getElementById('loginStatus');
    const token = document.getElementById('loginToken').value.trim();
    const userId = document.getElementById('loginUserId').value.trim();
    if (!token) { st2.className='status error'; st2.textContent='Token is required.'; return; }
    this.disabled = true; this.textContent = 'Signing in…';
    try {
      const r = await fetch('/account', { headers: { 'Authorization': 'Bearer ' + token } });
      if (r.ok) {
        localStorage.setItem('qb_token', token);
        if (userId) localStorage.setItem('qb_user', userId);
        location.href = '/';
      } else {
        st2.className='status error'; st2.textContent='Invalid token.';
        this.disabled=false; this.textContent='Sign In';
      }
    } catch(e) { st2.className='status error'; st2.textContent='Network error.'; this.disabled=false; this.textContent='Sign In'; }
  });
})();
</script>
</body>
</html>`;
}

function dashboardHtml() {
  return `<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="dashboard-token" content="${process.env.ADMIN_TOKEN ?? process.env.DASHBOARD_TOKEN ?? ''}">
<meta name="registration-code-required" content="${REGISTRATION_CODE ? 'true' : 'false'}">
<title>QuantBrain Dashboard v3</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
html, body { height: 100%; overflow: hidden; }

body {
  font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'SF Pro Text', sans-serif;
  background: #f5f5f7;
  color: #1d1d1f;
  display: flex;
  flex-direction: column;
  -webkit-font-smoothing: antialiased;
}

/* ── Nav ── */
.nav {
  background: rgba(255,255,255,0.85);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  border-bottom: 1px solid rgba(0,0,0,0.08);
  height: 52px;
  display: flex;
  align-items: center;
  padding: 0 28px;
  gap: 32px;
  flex-shrink: 0;
}
.nav-brand {
  display: flex; align-items: center; gap: 9px;
  font-size: 15px; font-weight: 600; color: #1d1d1f;
  letter-spacing: -0.3px;
}
.nav-logo {
  width: 26px; height: 26px;
  background: #1d1d1f;
  border-radius: 7px;
  display: flex; align-items: center; justify-content: center;
  font-size: 12px; font-weight: 800; color: #fff;
}
.nav-links { display: flex; gap: 2px; flex: 1; }
.nav-link {
  font-size: 13px; color: #86868b;
  padding: 5px 12px; border-radius: 7px; cursor: pointer;
  transition: all 0.15s;
}
.nav-link:hover { color: #1d1d1f; background: rgba(0,0,0,0.05); }
.nav-link.active { color: #1d1d1f; background: rgba(0,0,0,0.07); font-weight: 500; }
.nav-right { display: flex; align-items: center; gap: 12px; }
.live-badge {
  display: flex; align-items: center; gap: 6px;
  font-size: 12px; color: #86868b;
}
.live-dot {
  width: 6px; height: 6px;
  background: #34c759;
  border-radius: 50%;
  box-shadow: 0 0 5px rgba(52,199,89,0.6);
  animation: breathe 2.5s ease-in-out infinite;
}
@keyframes breathe { 0%,100%{opacity:1} 50%{opacity:0.4} }

/* ── Layout ── */
.layout {
  flex: 1;
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 16px;
  padding: 20px 24px;
  overflow: hidden;
}

/* ── Card ── */
.card {
  background: #ffffff;
  border-radius: 16px;
  padding: 24px;
  overflow-y: auto;
  box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 4px 16px rgba(0,0,0,0.04);
}

/* ── Eyebrow ── */
.eyebrow {
  font-size: 11px;
  font-weight: 600;
  color: #86868b;
  text-transform: uppercase;
  letter-spacing: 0.07em;
  margin-bottom: 6px;
}

/* ── Big Number ── */
.big-num {
  font-size: 52px;
  font-weight: 700;
  letter-spacing: -2px;
  line-height: 1;
  color: #1d1d1f;
  margin-bottom: 6px;
}
.big-num .unit {
  font-size: 22px; font-weight: 400;
  color: #86868b; margin-left: 4px;
}
.trend {
  font-size: 13px; color: #34c759;
  display: flex; align-items: center; gap: 4px;
}
.trend.warn { color: #ff9f0a; }

/* ── Divider ── */
.divider { height: 1px; background: #f0f0f2; margin: 20px 0; }

/* ── Mini stats ── */
.mini-stats { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
.mini-stat .ms-label { font-size: 11px; color: #86868b; margin-bottom: 3px; }
.mini-stat .ms-value { font-size: 22px; font-weight: 600; letter-spacing: -0.5px; color: #1d1d1f; }
.mini-stat .ms-sub { font-size: 11px; color: #86868b; margin-top: 1px; }
.ms-sub.up { color: #34c759; }

/* ── Pipeline ── */
.pipeline { display: flex; gap: 6px; margin-top: 4px; }
.p-stage {
  flex: 1;
  background: #f5f5f7;
  border-radius: 10px;
  padding: 12px 8px;
  text-align: center;
}
.p-num {
  font-size: 24px; font-weight: 600;
  letter-spacing: -0.5px; margin-bottom: 3px;
}
.p-label { font-size: 10px; color: #86868b; text-transform: uppercase; letter-spacing: 0.04em; }
.p-num.blue   { color: #007aff; }
.p-num.yellow { color: #ff9f0a; }
.p-num.purple { color: #af52de; }
.p-num.red    { color: #ff3b30; }
.p-num.green  { color: #34c759; }

/* ── LLM bars ── */
.llm-list { display: flex; flex-direction: column; gap: 16px; margin-top: 4px; }
.llm-header {
  display: flex; align-items: baseline;
  justify-content: space-between; margin-bottom: 7px;
}
.llm-name { font-size: 13px; font-weight: 500; color: #1d1d1f; }
.llm-role-tag { font-size: 11px; color: #86868b; margin-left: 7px; }
.llm-rate { font-size: 13px; font-weight: 600; color: #1d1d1f; }
.bar-track {
  height: 4px;
  background: #f0f0f2;
  border-radius: 2px;
  overflow: hidden;
}
.bar-fill { height: 100%; border-radius: 2px; }
.bar-green  { background: #34c759; }
.bar-yellow { background: #ff9f0a; }
.bar-blue   { background: #007aff; }

/* ── Repair Queue ── */
.repair-list { display: flex; flex-direction: column; gap: 8px; }
.repair-row {
  display: flex; align-items: center; gap: 10px;
  padding: 10px 12px;
  background: #f9f9f9;
  border-radius: 10px;
}
.r-expr {
  font-family: 'SF Mono', 'Fira Code', monospace;
  font-size: 11px; color: #1d1d1f; flex: 1;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.r-tag {
  font-size: 10px; font-weight: 600;
  color: #ff9f0a; background: rgba(255,159,10,0.1);
  padding: 2px 7px; border-radius: 5px; flex-shrink: 0;
}
.r-depth { font-size: 10px; color: #86868b; flex-shrink: 0; width: 30px; text-align: right; }

/* ── Right panel ── */
.countdown {
  text-align: center; padding: 8px 0 20px;
}
.cd-label { font-size: 11px; color: #86868b; text-transform: uppercase; letter-spacing: 0.07em; margin-bottom: 6px; }
.cd-time { font-size: 44px; font-weight: 300; letter-spacing: -1px; color: #1d1d1f; font-variant-numeric: tabular-nums; }
.cd-sub { font-size: 11px; color: #86868b; margin-top: 4px; }

.idea-label { font-size: 11px; color: #86868b; text-transform: uppercase; letter-spacing: 0.07em; margin-bottom: 8px; }
.idea-field {
  width: 100%;
  background: #f5f5f7;
  border: 1px solid rgba(0,0,0,0.08);
  border-radius: 10px;
  padding: 11px 13px;
  color: #1d1d1f;
  font-size: 13px;
  font-family: inherit;
  resize: none;
  height: 76px;
  line-height: 1.5;
  outline: none;
  transition: border-color 0.15s;
}
.idea-field:focus { border-color: rgba(0,0,0,0.2); background: #fff; }
.idea-field::placeholder { color: #aeaeb2; }
.run-btn {
  margin-top: 8px; width: 100%;
  background: #1d1d1f; color: #fff;
  border: none; border-radius: 10px;
  padding: 11px; font-size: 13px;
  font-weight: 600; cursor: pointer;
  font-family: inherit; letter-spacing: -0.2px;
  transition: opacity 0.15s;
}
.run-btn:hover { opacity: 0.82; }

/* ── Activity ── */
.activity-feed { display: flex; flex-direction: column; gap: 14px; }
.act-item { display: flex; align-items: flex-start; gap: 11px; }
.act-icon {
  width: 28px; height: 28px; border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-size: 11px; font-weight: 700; flex-shrink: 0;
}
.act-icon.submit { background: rgba(52,199,89,0.12); color: #34c759; }
.act-icon.repair { background: rgba(255,159,10,0.12); color: #ff9f0a; }
.act-icon.fail   { background: rgba(255,59,48,0.10);  color: #ff3b30; }
.act-body { flex: 1; min-width: 0; }
.act-expr {
  font-family: 'SF Mono', 'Fira Code', monospace;
  font-size: 11px; color: #1d1d1f;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.act-meta { font-size: 11px; color: #86868b; margin-top: 2px; }
</style>
</head>
<body>

<!-- Nav -->
<div class="nav">
  <div class="nav-brand">
    <div class="nav-logo">Q</div>
    QuantBrain
  </div>
  <div class="nav-links">
    <div class="nav-link active" data-tab="overview">Overview</div>
    <div class="nav-link" data-tab="pipeline">Pipeline</div>
    <div class="nav-link" data-tab="knowledge">Knowledge</div>
    <div class="nav-link" data-tab="runs">Runs</div>
    <div class="nav-link" data-tab="settings">Settings</div>
  </div>
  <div class="nav-right">
    <div class="live-badge">
      <div class="live-dot"></div>
      Mining Active
    </div>
  </div>
</div>

<!-- Panels -->
<div id="panel-overview" class="panel" style="display:flex;flex:1;overflow:hidden">
<div class="layout">

  <!-- Card 1: Metrics + Pipeline -->
  <div class="card">
    <div class="eyebrow">Submitted Today</div>
    <div class="big-num">24 <span class="unit">alphas</span></div>
    <div class="trend">↑ 12% vs last week</div>

    <div class="divider"></div>

    <div class="mini-stats">
      <div class="mini-stat">
        <div class="ms-label">Gate Pass Rate</div>
        <div class="ms-value">38%</div>
        <div class="ms-sub up">↑ trending up</div>
      </div>
      <div class="mini-stat">
        <div class="ms-label">Repair Success</div>
        <div class="ms-value">61%</div>
        <div class="ms-sub">3 in queue</div>
      </div>
      <div class="mini-stat">
        <div class="ms-label">Simulations</div>
        <div class="ms-value">47</div>
        <div class="ms-sub">453 remaining</div>
      </div>
      <div class="mini-stat">
        <div class="ms-label">Daily Spend</div>
        <div class="ms-value">$0.38</div>
        <div class="ms-sub">of $3.60 limit</div>
      </div>
    </div>

    <div class="divider"></div>

    <div class="eyebrow" style="margin-bottom:12px">Alpha Pipeline — Now</div>
    <div class="pipeline">
      <div class="p-stage"><div class="p-num blue">6</div><div class="p-label">Gen</div></div>
      <div class="p-stage"><div class="p-num yellow">5</div><div class="p-label">Valid</div></div>
      <div class="p-stage"><div class="p-num purple">5</div><div class="p-label">Sim</div></div>
      <div class="p-stage"><div class="p-num red">2</div><div class="p-label">Gate</div></div>
      <div class="p-stage"><div class="p-num green">1</div><div class="p-label">Sub</div></div>
    </div>
  </div>

  <!-- Card 2: LLM Router + Repair -->
  <div class="card">
    <div class="eyebrow" style="margin-bottom:14px">LLM Router — Live</div>
    <div class="llm-list">
      <div>
        <div class="llm-header">
          <div><span class="llm-name">Deepseek V3</span><span class="llm-role-tag">Generate · 68%</span></div>
          <div class="llm-rate" style="color:#34c759">42% pass</div>
        </div>
        <div class="bar-track"><div class="bar-fill bar-green" style="width:68%"></div></div>
      </div>
      <div>
        <div class="llm-header">
          <div><span class="llm-name">Gemini Flash</span><span class="llm-role-tag">Generate · 32%</span></div>
          <div class="llm-rate" style="color:#ff9f0a">31% pass</div>
        </div>
        <div class="bar-track"><div class="bar-fill bar-yellow" style="width:32%"></div></div>
      </div>
      <div>
        <div class="llm-header">
          <div><span class="llm-name">Claude Haiku</span><span class="llm-role-tag">Repair · 75%</span></div>
          <div class="llm-rate" style="color:#34c759">58% pass</div>
        </div>
        <div class="bar-track"><div class="bar-fill bar-green" style="width:75%"></div></div>
      </div>
      <div>
        <div class="llm-header">
          <div><span class="llm-name">GPT-4o-mini</span><span class="llm-role-tag">Judge / Distill</span></div>
          <div class="llm-rate" style="color:#007aff">Infra</div>
        </div>
        <div class="bar-track"><div class="bar-fill bar-blue" style="width:100%"></div></div>
      </div>
    </div>

    <div class="divider"></div>

    <div class="eyebrow" style="margin-bottom:12px">Repair Queue</div>
    <div class="repair-list">
      <div class="repair-row">
        <div class="r-expr">rank(ts_rank(op_inc/assets, 252))</div>
        <div class="r-tag">LOW_SHARPE</div>
        <div class="r-depth">2/5</div>
      </div>
      <div class="repair-row">
        <div class="r-expr">group_rank(cashflow_op/rev, sector)</div>
        <div class="r-tag">SELF_CORR</div>
        <div class="r-depth">1/5</div>
      </div>
      <div class="repair-row">
        <div class="r-expr">rank(-ts_std_dev(returns, 60))</div>
        <div class="r-tag">LOW_FIT</div>
        <div class="r-depth">—</div>
      </div>
    </div>
  </div>

  <!-- Card 3: Right Panel -->
  <div class="card">
    <div class="countdown">
      <div class="cd-label">Next Run In</div>
      <div class="cd-time">38:22</div>
      <div class="cd-sub">loop · legacy-js · batch 3</div>
    </div>

    <div class="divider"></div>

    <div style="margin-bottom:24px">
      <div class="idea-label">Research Idea</div>
      <textarea class="idea-field" placeholder="Describe your alpha idea in plain language..."></textarea>
      <button class="run-btn">Optimize &amp; Run</button>
    </div>

    <div class="divider"></div>

    <div class="eyebrow" style="margin-bottom:14px">Recent Activity</div>
    <div class="activity-feed">
      <div class="act-item">
        <div class="act-icon submit">↑</div>
        <div class="act-body">
          <div class="act-expr">group_rank(ts_rank(op_inc/assets,252),ind)</div>
          <div class="act-meta">Submitted · Sharpe 1.42 · 14:35</div>
        </div>
      </div>
      <div class="act-item">
        <div class="act-icon repair">⟳</div>
        <div class="act-body">
          <div class="act-expr">rank(ts_delta(cashflow_op/assets, 252))</div>
          <div class="act-meta">Repair round 3 · LOW_SHARPE · 13:58</div>
        </div>
      </div>
      <div class="act-item">
        <div class="act-icon fail">✕</div>
        <div class="act-body">
          <div class="act-expr">rank(vwap - close) * rank(-returns)</div>
          <div class="act-meta">Budget exhausted · 13:12</div>
        </div>
      </div>
      <div class="act-item">
        <div class="act-icon submit">↑</div>
        <div class="act-body">
          <div class="act-expr">rank(ts_mean(returns,60)-ts_mean(returns,252))</div>
          <div class="act-meta">Submitted · Sharpe 1.28 · 12:41</div>
        </div>
      </div>
    </div>
  </div>

</div>
</div><!-- /panel-overview -->

<!-- Panel: Pipeline -->
<div id="panel-pipeline" class="panel" style="display:none;padding:28px;overflow-y:auto;flex:1">
  <div class="eyebrow" style="margin-bottom:20px">Active Runs</div>
  <div id="pipeline-active" style="font-size:13px;color:#86868b">No active runs</div>
  <div class="divider" style="margin:24px 0"></div>
  <div class="eyebrow" style="margin-bottom:16px">Pipeline Stages</div>
  <div class="pipeline" id="pipeline-stages">
    <div class="p-stage"><div class="p-num blue" id="ps-gen">–</div><div class="p-label">Generate</div></div>
    <div class="p-stage"><div class="p-num yellow" id="ps-val">–</div><div class="p-label">Validate</div></div>
    <div class="p-stage"><div class="p-num purple" id="ps-sim">–</div><div class="p-label">Simulate</div></div>
    <div class="p-stage"><div class="p-num red" id="ps-gate">–</div><div class="p-label">Gate</div></div>
    <div class="p-stage"><div class="p-num green" id="ps-sub">–</div><div class="p-label">Submit</div></div>
  </div>
  <div class="divider" style="margin:24px 0"></div>
  <div class="eyebrow" style="margin-bottom:12px">Progress Log</div>
  <div id="pipeline-log" style="font-family:monospace;font-size:11px;color:#3a3a3c;background:#f5f5f7;border-radius:8px;padding:12px;max-height:280px;overflow-y:auto;white-space:pre-wrap;word-break:break-all"></div>
</div>

<!-- Panel: Knowledge -->
<div id="panel-knowledge" class="panel" style="display:none;padding:28px;overflow-y:auto;flex:1">
  <div class="eyebrow" style="margin-bottom:20px">LLM Router — All Providers</div>
  <div id="knowledge-llm" style="font-size:13px;color:#86868b">Loading…</div>
  <div class="divider" style="margin:24px 0"></div>
  <div class="eyebrow" style="margin-bottom:16px">Budget</div>
  <div id="knowledge-budget" style="font-size:13px;color:#3a3a3c">–</div>
</div>

<!-- Panel: Runs -->
<div id="panel-runs" class="panel" style="display:none;padding:28px;overflow-y:auto;flex:1">
  <div class="eyebrow" style="margin-bottom:16px">Recent Runs</div>
  <div id="runs-list" style="font-size:13px;color:#86868b">Loading…</div>
</div>

<!-- Panel: Settings -->
<div id="panel-settings" class="panel" style="display:none;padding:28px;overflow-y:auto;flex:1;max-width:520px">
  <div class="eyebrow" style="margin-bottom:20px">Scheduler Settings</div>
  <div style="display:flex;flex-direction:column;gap:16px" id="settings-form">
    <label style="display:flex;flex-direction:column;gap:6px;font-size:13px;color:#3a3a3c">
      Enabled
      <select id="s-enabled" style="padding:8px 10px;border-radius:8px;border:1px solid #d2d2d7;font-size:13px;background:#fff">
        <option value="true">Yes</option><option value="false">No</option>
      </select>
    </label>
    <label style="display:flex;flex-direction:column;gap:6px;font-size:13px;color:#3a3a3c">
      Mode
      <select id="s-mode" style="padding:8px 10px;border-radius:8px;border:1px solid #d2d2d7;font-size:13px;background:#fff">
        <option value="evaluate">evaluate</option>
        <option value="loop">loop</option>
        <option value="generate">generate</option>
      </select>
    </label>
    <label style="display:flex;flex-direction:column;gap:6px;font-size:13px;color:#3a3a3c">
      Interval (minutes)
      <input id="s-interval" type="number" min="15" style="padding:8px 10px;border-radius:8px;border:1px solid #d2d2d7;font-size:13px">
    </label>
    <label style="display:flex;flex-direction:column;gap:6px;font-size:13px;color:#3a3a3c">
      Batch Size
      <input id="s-batch" type="number" min="1" style="padding:8px 10px;border-radius:8px;border:1px solid #d2d2d7;font-size:13px">
    </label>
    <label style="display:flex;flex-direction:column;gap:6px;font-size:13px;color:#3a3a3c">
      Rounds
      <input id="s-rounds" type="number" min="1" style="padding:8px 10px;border-radius:8px;border:1px solid #d2d2d7;font-size:13px">
    </label>
    <label style="display:flex;flex-direction:column;gap:6px;font-size:13px;color:#3a3a3c">
      Objective
      <input id="s-objective" type="text" style="padding:8px 10px;border-radius:8px;border:1px solid #d2d2d7;font-size:13px">
    </label>
    <button id="s-save" style="margin-top:8px;padding:10px 24px;border-radius:20px;border:none;background:#1d1d1f;color:#fff;font-size:14px;font-weight:500;cursor:pointer">Save Settings</button>
    <div id="s-status" style="font-size:12px;color:#86868b;min-height:16px"></div>
  </div>
</div>

</body>
<script>
(function(){
  // Prefer localStorage token (registered users), fall back to server-injected admin token
  const adminToken = document.querySelector('meta[name=dashboard-token]').content;
  const storedToken = localStorage.getItem('qb_token') ?? '';
  const token = storedToken || adminToken;

  // If no token at all, redirect to setup
  if (!token) { location.href = '/setup'; return; }

  const h = { 'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json' };

  // Verify token is valid on load; redirect to setup if 401
  fetch('/account', { headers: h }).then(r => {
    if (r.status === 401) { localStorage.removeItem('qb_token'); location.href = '/setup'; }
  }).catch(() => {});

  // Show username in nav if available
  const userName = localStorage.getItem('qb_user');
  const navRight = document.querySelector('.nav-right');
  if (navRight && userName) {
    const userBadge = document.createElement('div');
    userBadge.style.cssText = 'font-size:13px;color:#3a3a3c;font-weight:500;margin-right:12px;display:flex;align-items:center;gap:6px';
    userBadge.innerHTML = '<span style="width:24px;height:24px;background:#007aff;border-radius:50%;display:inline-flex;align-items:center;justify-content:center;color:#fff;font-size:11px;font-weight:700">' +
      (userName[0] ?? '?').toUpperCase() + '</span>' + userName;
    navRight.insertBefore(userBadge, navRight.firstChild);
  }

  function esc(s) {
    return String(s ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  // ── Countdown ──
  let nextRunAt = null;
  function tickCd() {
    const el = document.querySelector('.cd-time');
    if (!el) return;
    if (!nextRunAt) { el.textContent = '–'; return; }
    const ms = new Date(nextRunAt) - Date.now();
    if (ms <= 0) { el.textContent = '00:00'; return; }
    el.textContent = String(Math.floor(ms / 60000)).padStart(2, '0') + ':' +
                     String(Math.floor((ms % 60000) / 1000)).padStart(2, '0');
  }
  setInterval(tickCd, 1000);
  async function fetchSched() {
    try {
      const r = await fetch('/scheduler', { headers: h });
      const d = await r.json();
      nextRunAt = d.nextRunAt ?? null;
    } catch (e) {}
  }
  fetchSched();
  setInterval(fetchSched, 30000);

  // ── Poll /runs every 10s ──
  async function poll() {
    try {
      const r = await fetch('/runs', { headers: h });
      const d = await r.json();
      const al = d.autoLoop ?? {};
      const lr = d.llmRouterState ?? null;

      // Stat cards
      const sv = document.querySelectorAll('.stat-val');
      if (sv[0]) sv[0].textContent = al.submitted?.length ?? 0;
      const recent = d.recent ?? [];
      const passed = recent.filter(x => x.summary?.passedGate).length;
      if (sv[1]) sv[1].textContent = recent.length ? Math.round(passed / recent.length * 100) + '%' : '–';
      if (sv[2] && lr) sv[2].textContent = '$' + (lr.spent_usd ?? 0).toFixed(2) + ' / $' + (lr.daily_budget_usd ?? 3.6).toFixed(2);

      // ── Pipeline panel ──
      const activeEl = document.getElementById('pipeline-active');
      if (activeEl) {
        const activeRuns = d.active ?? [];
        if (activeRuns.length) {
          activeEl.innerHTML = activeRuns.map(r =>
            '<div style="background:#fff;border-radius:10px;padding:12px 14px;margin-bottom:8px;border:1px solid rgba(0,0,0,0.06)">' +
            '<div style="font-weight:500;font-size:13px">' + esc(r.runId) + '</div>' +
            '<div style="font-size:11px;color:#86868b;margin-top:3px">' + esc(r.status) + ' · ' + esc(r.engine ?? '') + '</div>' +
            '</div>'
          ).join('');
        } else {
          activeEl.textContent = 'No active runs';
        }
        // progress log from most recent run
        const logEl = document.getElementById('pipeline-log');
        if (logEl) {
          const recent0 = d.recent?.[0];
          if (recent0?.progressTail?.length) {
            logEl.textContent = recent0.progressTail.map(l => l.msg ?? JSON.stringify(l)).join('\\n');
          } else {
            logEl.textContent = 'No progress data';
          }
        }
      }

      // ── Knowledge panel ──
      const kllm = document.getElementById('knowledge-llm');
      if (kllm && lr?.providers) {
        kllm.innerHTML = '<table style="width:100%;border-collapse:collapse;font-size:12px">' +
          '<tr style="color:#86868b;border-bottom:1px solid #e5e5ea"><th style="text-align:left;padding:6px 8px">Provider</th><th style="padding:6px 8px">Role</th><th style="padding:6px 8px">Win Rate</th><th style="padding:6px 8px">Calls</th><th style="padding:6px 8px">Wins</th></tr>' +
          Object.entries(lr.providers).map(([n, p]) => {
            const wr = ((p.win_rate ?? 0.5) * 100).toFixed(1);
            const col = (p.win_rate ?? 0.5) >= 0.5 ? '#34c759' : '#ff9f0a';
            return '<tr style="border-bottom:1px solid #f2f2f7">' +
              '<td style="padding:8px;font-weight:500">' + esc(n) + '</td>' +
              '<td style="padding:8px;text-align:center;color:#86868b">' + esc(p.role ?? '–') + '</td>' +
              '<td style="padding:8px;text-align:center;color:' + col + ';font-weight:600">' + wr + '%</td>' +
              '<td style="padding:8px;text-align:center">' + (p.calls ?? 0) + '</td>' +
              '<td style="padding:8px;text-align:center">' + (p.wins ?? 0) + '</td>' +
              '</tr>';
          }).join('') + '</table>';
        const kb = document.getElementById('knowledge-budget');
        if (kb) kb.innerHTML =
          '<div style="display:flex;gap:24px">' +
          '<div><div style="font-size:11px;color:#86868b;margin-bottom:4px">Spent Today</div><div style="font-size:20px;font-weight:600">$' + (lr.spent_usd ?? 0).toFixed(3) + '</div></div>' +
          '<div><div style="font-size:11px;color:#86868b;margin-bottom:4px">Daily Budget</div><div style="font-size:20px;font-weight:600">$' + (lr.daily_budget_usd ?? 3.6).toFixed(2) + '</div></div>' +
          '<div><div style="font-size:11px;color:#86868b;margin-bottom:4px">Remaining</div><div style="font-size:20px;font-weight:600;color:#34c759">$' + Math.max(0, (lr.daily_budget_usd ?? 3.6) - (lr.spent_usd ?? 0)).toFixed(3) + '</div></div>' +
          '</div>';
      }

      // ── Runs panel ──
      const rl2 = document.getElementById('runs-list');
      if (rl2) {
        const runs = d.recent ?? [];
        if (runs.length) {
          rl2.innerHTML = '<table style="width:100%;border-collapse:collapse;font-size:12px">' +
            '<tr style="color:#86868b;border-bottom:1px solid #e5e5ea"><th style="text-align:left;padding:6px 8px">Run ID</th><th style="padding:6px 8px">Status</th><th style="padding:6px 8px">Engine</th><th style="padding:6px 8px">Submitted</th><th style="padding:6px 8px">Sharpe</th></tr>' +
            runs.slice(0, 20).map(r => {
              const s = r.state ?? {};
              const sum = r.summary ?? {};
              const statusCol = s.status === 'completed' ? '#34c759' : s.status === 'running' ? '#007aff' : '#86868b';
              return '<tr style="border-bottom:1px solid #f2f2f7">' +
                '<td style="padding:8px;font-family:monospace;font-size:11px">' + esc(r.runId?.slice(-12) ?? '–') + '</td>' +
                '<td style="padding:8px;text-align:center;color:' + statusCol + '">' + esc(s.status ?? '–') + '</td>' +
                '<td style="padding:8px;text-align:center;color:#86868b">' + esc(s.engine ?? '–') + '</td>' +
                '<td style="padding:8px;text-align:center">' + (sum.submitted ?? '–') + '</td>' +
                '<td style="padding:8px;text-align:center">' + (sum.bestSharpe != null ? sum.bestSharpe.toFixed(2) : '–') + '</td>' +
                '</tr>';
            }).join('') + '</table>';
        } else {
          rl2.textContent = 'No runs found';
        }
      }

      // LLM rows
      const ll = document.querySelector('.llm-list');
      if (ll && lr?.providers) {
        ll.innerHTML = Object.entries(lr.providers).slice(0, 4).map(([n, p]) => {
          const wr = ((p.win_rate ?? 0.5) * 100).toFixed(0);
          const col = (p.win_rate ?? 0.5) >= 0.5 ? '#34c759' : '#ff9f0a';
          return '<div>' +
            '<div class="llm-header">' +
              '<div><span class="llm-name">' + esc(n) + '</span>' +
              '<span class="llm-role-tag">' + esc(p.role ?? '') + ' · ' + wr + '%</span></div>' +
              '<div class="llm-rate" style="color:' + col + '">' + wr + '% pass</div>' +
            '</div>' +
            '<div class="bar-track"><div class="bar-fill" style="background:' + col + ';width:' + wr + '%"></div></div>' +
          '</div>';
        }).join('');
      }

      // Repair queue
      const rl = document.querySelector('.repair-list');
      if (rl) {
        const q = al.queue?.slice(0, 3) ?? [];
        rl.innerHTML = q.length
          ? q.map(x => '<div class="repair-row"><div class="r-expr">' + esc(x.expression) + '</div>' +
              '<div class="r-tag">' + esc(x.reason) + '</div>' +
              '<div class="r-depth">' + esc(x.depth) + '</div></div>').join('')
          : '<div class="repair-row"><div class="r-expr" style="color:#86868b">Queue empty</div></div>';
      }

      // Activity feed
      const af = document.querySelector('.activity-feed');
      if (af) {
        const ev = (al.events ?? []).slice(0, 4);
        af.innerHTML = ev.length
          ? ev.map(e => {
              const cls = e.type === 'submitted' ? 'submit' : e.type === 'repair' ? 'repair' : 'fail';
              const icon = e.type === 'submitted' ? '↑' : e.type === 'repair' ? '⟳' : '✕';
              return '<div class="act-item">' +
                '<div class="act-icon ' + cls + '">' + icon + '</div>' +
                '<div class="act-body">' +
                  '<div class="act-expr">' + esc(e.alphaId ?? e.expression ?? '–') + '</div>' +
                  '<div class="act-meta">' + esc(e.message ?? e.type ?? '') + '</div>' +
                '</div></div>';
            }).join('')
          : '<div class="act-item"><div class="act-body"><div class="act-meta" style="color:#86868b">No recent activity</div></div></div>';
      }
    } catch (e) {}
  }
  poll();
  setInterval(poll, 10000);

  // ── Nav tabs ──
  function showPanel(tab) {
    document.querySelectorAll('.panel').forEach(p => p.style.display = 'none');
    const el = document.getElementById('panel-' + tab);
    if (el) el.style.display = tab === 'overview' ? 'flex' : 'flex';
  }
  document.querySelectorAll('.nav-link').forEach(a => a.addEventListener('click', function(e) {
    e.preventDefault();
    document.querySelectorAll('.nav-link').forEach(x => x.classList.remove('active'));
    this.classList.add('active');
    showPanel(this.dataset.tab ?? 'overview');
  }));

  // ── Settings panel load + save ──
  let settingsLoaded = false;
  async function loadSettings() {
    if (settingsLoaded) return;
    try {
      const r = await fetch('/scheduler', { headers: h });
      const d = await r.json();
      const g = id => document.getElementById(id);
      if (g('s-enabled')) g('s-enabled').value = String(d.enabled ?? false);
      if (g('s-mode')) g('s-mode').value = d.mode ?? 'evaluate';
      if (g('s-interval')) g('s-interval').value = d.intervalMinutes ?? 60;
      if (g('s-batch')) g('s-batch').value = d.batchSize ?? 5;
      if (g('s-rounds')) g('s-rounds').value = d.rounds ?? 3;
      if (g('s-objective')) g('s-objective').value = d.objective ?? '';
      settingsLoaded = true;
    } catch (e) {}
  }
  document.getElementById('s-save')?.addEventListener('click', async function() {
    const g = id => document.getElementById(id);
    const st = document.getElementById('s-status');
    this.disabled = true; this.textContent = 'Saving…';
    try {
      const body = {
        enabled: g('s-enabled')?.value === 'true',
        mode: g('s-mode')?.value,
        intervalMinutes: Number(g('s-interval')?.value),
        batchSize: Number(g('s-batch')?.value),
        rounds: Number(g('s-rounds')?.value),
        objective: g('s-objective')?.value?.trim(),
      };
      const r = await fetch('/scheduler', { method: 'POST', headers: h, body: JSON.stringify(body) });
      if (r.ok) { if (st) st.textContent = 'Saved ✓'; }
      else { if (st) st.textContent = 'Error: ' + r.status; }
    } catch (e) { if (st) st.textContent = 'Error: ' + e.message; }
    this.disabled = false; this.textContent = 'Save Settings';
    setTimeout(() => { if (st) st.textContent = ''; }, 3000);
  });
  // Load settings when user clicks the tab
  document.querySelectorAll('.nav-link').forEach(a => {
    if (a.dataset.tab === 'settings') a.addEventListener('click', loadSettings);
  });

  // ── Optimize & Run button ──
  const btn = document.querySelector('.run-btn');
  if (btn) {
    const statusEl = document.createElement('div');
    statusEl.style.cssText = 'font-size:12px;color:#86868b;margin-top:8px;min-height:16px;';
    btn.parentNode.insertBefore(statusEl, btn.nextSibling);

    btn.addEventListener('click', async function() {
      const idea = document.querySelector('.idea-field')?.value?.trim();
      if (!idea) { statusEl.textContent = 'Please enter an idea'; return; }
      btn.disabled = true;
      btn.textContent = 'Optimizing…';
      statusEl.textContent = '';
      try {
        const or = await fetch('/ideas/optimize', { method: 'POST', headers: h, body: JSON.stringify({ idea }) });
        const od = await or.json();
        statusEl.textContent = 'Direction: ' + esc(od.direction ?? od.target_family ?? '');
        btn.textContent = 'Starting run…';
        const rr = await fetch('/runs', { method: 'POST', headers: h, body: JSON.stringify({ force: false }) });
        if (rr.status === 409) {
          statusEl.textContent = 'Run already active';
          btn.disabled = false;
          btn.textContent = 'Optimize & Run';
          return;
        }
        btn.textContent = 'Run Started ✓';
        setTimeout(() => { btn.disabled = false; btn.textContent = 'Optimize & Run'; }, 3000);
      } catch (e) {
        statusEl.textContent = 'Error: ' + e.message;
        btn.disabled = false;
        btn.textContent = 'Optimize & Run';
      }
    });
  }
})();
</script>
</html>
`;
}
