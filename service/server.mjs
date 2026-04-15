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
const OPENAI_IDEA_MODEL = process.env.OPENAI_IDEA_MODEL ?? "gpt-4o-mini";
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
const AUTO_REPAIR_ENABLED = process.env.AUTO_REPAIR_ENABLED !== "false";
const AUTO_REPAIR_ENGINE = normalizeEngine(process.env.AUTO_REPAIR_ENGINE ?? "python-v2");
const AUTO_REPAIR_MAX_ROUNDS = clampInt(process.env.AUTO_REPAIR_MAX_ROUNDS, 3, 1, 10);
const AUTO_REPAIR_BATCH_SIZE = clampInt(process.env.AUTO_REPAIR_BATCH_SIZE, 5, 1, 10);
const REPAIR_TARGET_SHARPE = parseFloat(process.env.REPAIR_TARGET_SHARPE ?? "1.25");
const REPAIR_TARGET_FITNESS = parseFloat(process.env.REPAIR_TARGET_FITNESS ?? "1.0");
const REPAIR_TARGET_TURNOVER = parseFloat(process.env.REPAIR_TARGET_TURNOVER ?? "0.40");
const AUTO_SUBMIT_ENABLED = process.env.AUTO_SUBMIT_ENABLED === "true";
const ALPHA_GENERATOR_STRATEGY = ["legacy", "diversity-v2"].includes(process.env.ALPHA_GENERATOR_STRATEGY)
  ? process.env.ALPHA_GENERATOR_STRATEGY
  : "legacy";
const ALPHA_EXPERIMENTAL_FIELDS = process.env.ALPHA_EXPERIMENTAL_FIELDS === "true";
const ALPHA_CROWDING_PATTERN_THRESHOLD = clampInt(process.env.ALPHA_CROWDING_PATTERN_THRESHOLD, 2, 1, 20);
const ALPHA_FAMILY_COOLDOWN_ROUNDS = clampInt(process.env.ALPHA_FAMILY_COOLDOWN_ROUNDS, 3, 1, 20);

// Dynamic objective: category → relevant BRAIN fields
const CATEGORY_FIELDS = {
  QUALITY:        ["operating_income", "cashflow_op", "assets", "est_eps", "returns"],
  MOMENTUM:       ["returns", "close", "vwap", "volume", "adv20"],
  REVERSAL:       ["returns", "close", "open", "high", "low", "vwap"],
  VOLATILITY:     ["returns", "high", "low", "close", "adv20", "volume"],
  LIQUIDITY:      ["volume", "adv20", "vwap", "close", "returns"],
  MICROSTRUCTURE: ["volume", "adv20", "high", "low", "open", "close", "vwap"],
  SENTIMENT:      ["news_sentiment", "est_eps", "returns", "volume", "operating_income"],
};
const OBJECTIVE_HISTORY_PATH = path.join(RUNS_DIR, "objective-history.json");
// Loaded after RUNS_DIR is created; max 200 entries kept
let objectiveHistory = [];

const activeRuns = new Map();
const DEFAULT_SIM_SETTINGS = {
  region: "USA",
  universe: "TOP3000",
  delay: 1,
  decay: 4,
  neutralization: "INDUSTRY",
  truncation: 0.08,
  pasteurization: "ON",
  unitHandling: "VERIFY",
};

const schedulerState = {
  enabled: process.env.AUTO_RUN_ENABLED === "true",
  engine: normalizeEngine(DEFAULT_ENGINE),
  mode: DEFAULT_MODE,
  objective: DEFAULT_OBJECTIVE,
  intervalMinutes: DEFAULT_INTERVAL_MINUTES,
  rounds: DEFAULT_ROUNDS,
  batchSize: DEFAULT_BATCH_SIZE,
  concurrency: DEFAULT_CONCURRENCY,
  simulationSettings: { ...DEFAULT_SIM_SETTINGS },
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
objectiveHistory = (await maybeReadJson(OBJECTIVE_HISTORY_PATH)) ?? [];
scheduleNextRun("startup");
// On startup: if repair queue has items from a previous session, drain immediately
if (AUTO_REPAIR_ENABLED && autoLoopState.queue.length > 0) {
  setTimeout(() => void startNextAutoRepairRun(), 3000);
}

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
      url.pathname.startsWith("/account") ||
      url.pathname === "/alpha-library"
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

    if (req.method === "GET" && url.pathname === "/scheduler/objective-history") {
      return sendJson(res, 200, { history: objectiveHistory.slice(-50).reverse() });
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
      if (Number.isFinite(Number(body.intervalMinutes)) && Number(body.intervalMinutes) >= 1) {
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
      if (body.simulationSettings && typeof body.simulationSettings === "object") {
        const s = body.simulationSettings;
        const merged = { ...schedulerState.simulationSettings };
        if (["USA","EUROPE","ASIA"].includes(s.region)) merged.region = s.region;
        if (["TOP500","TOP1000","TOP2000","TOP3000"].includes(s.universe)) merged.universe = s.universe;
        if ([0, 1].includes(Number(s.delay))) merged.delay = Number(s.delay);
        if (Number.isFinite(Number(s.decay)) && Number(s.decay) >= 0 && Number(s.decay) <= 13) merged.decay = Number(s.decay);
        if (["NONE","SUBINDUSTRY","INDUSTRY","SECTOR","MARKET"].includes(s.neutralization)) merged.neutralization = s.neutralization;
        if (Number.isFinite(Number(s.truncation)) && Number(s.truncation) >= 0.01 && Number(s.truncation) <= 0.1) merged.truncation = Number(s.truncation);
        if (["ON","OFF"].includes(s.pasteurization)) merged.pasteurization = s.pasteurization;
        if (["VERIFY","CASH"].includes(s.unitHandling)) merged.unitHandling = s.unitHandling;
        schedulerState.simulationSettings = merged;
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

    if (req.method === "GET" && url.pathname === "/alpha-library") {
      return sendJson(res, 200, await buildAlphaLibrary());
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
  const simulationSettings = input.simulationSettings ?? schedulerState.simulationSettings ?? DEFAULT_SIM_SETTINGS;
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
  const command = buildRunCommand({ engine, mode, objective, rounds, batchSize, concurrency, simulationSettings, outputDir, repairContextPath });
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

// Returns {category, field} choosing least-recently-used category first,
// then a random field from that category's field list.
function pickNextTarget() {
  const categories = Object.keys(CATEGORY_FIELDS);
  // Count recent uses per category (last 20 entries)
  const recent = objectiveHistory.slice(-20);
  const counts = {};
  for (const cat of categories) counts[cat] = 0;
  for (const entry of recent) {
    if (entry.category && counts[entry.category] !== undefined) counts[entry.category]++;
  }
  // Sort ascending by count; ties broken by shuffle
  const sorted = categories.slice().sort((a, b) => counts[a] - counts[b]);
  // Pick the least used; if tied, pick randomly among tied
  const minCount = counts[sorted[0]];
  const tied = sorted.filter(c => counts[c] === minCount);
  const category = tied[Math.floor(Math.random() * tied.length)];
  const fields = CATEGORY_FIELDS[category];
  // Avoid repeating the last-used field for this category
  const lastField = objectiveHistory.slice().reverse().find(e => e.category === category)?.field;
  const available = fields.filter(f => f !== lastField);
  const fieldPool = available.length ? available : fields;
  const field = fieldPool[Math.floor(Math.random() * fieldPool.length)];
  return { category, field };
}

// Save persistent objective history file (fire-and-forget)
function saveObjectiveHistory() {
  const trimmed = objectiveHistory.slice(-200);
  objectiveHistory = trimmed;
  writeFile(OBJECTIVE_HISTORY_PATH, JSON.stringify(trimmed, null, 2)).catch(() => {});
}

// Generate a specific, creative mining objective using GPT-4o-mini.
// Falls back to a template string if OpenAI is unavailable.
async function generateScheduledObjective() {
  const { category, field } = pickNextTarget();
  const recentObjectives = objectiveHistory.slice(-5).map(e => e.objective).join("; ");

  let objective;
  if (process.env.OPENAI_API_KEY) {
    try {
      const systemPrompt =
        "You are a WorldQuant BRAIN quantitative researcher. Generate a concise, specific alpha mining objective (1-2 sentences) for the given category and data field. " +
        "The objective should suggest a testable hypothesis about return predictability. " +
        "Avoid repeating recent objectives. Return ONLY the objective string, no JSON, no commentary.";
      const userPrompt =
        `Category: ${category}\nPrimary field: ${field}\n` +
        (recentObjectives ? `Recent objectives to avoid repeating: ${recentObjectives}\n` : "") +
        "Write a specific mining objective for this category and field.";

      const resp = await fetch("https://api.openai.com/v1/chat/completions", {
        method: "POST",
        headers: {
          Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model: OPENAI_OPTIMIZE_MODEL,
          messages: [
            { role: "system", content: systemPrompt },
            { role: "user", content: userPrompt },
          ],
          max_tokens: 120,
          temperature: 0.9,
        }),
      });
      const payload = await resp.json().catch(() => null);
      const content = payload?.choices?.[0]?.message?.content?.trim();
      if (content) objective = content;
    } catch (_) {}
  }

  // Fallback template when OpenAI is unavailable
  if (!objective) {
    const templates = {
      QUALITY:        `Discover robust ${field}-based quality signals with low crowding and positive out-of-sample stability`,
      MOMENTUM:       `Identify short-to-medium horizon momentum patterns using ${field} with sector neutralization`,
      REVERSAL:       `Mine mean-reversion alphas from ${field} anomalies with low turnover and high Sharpe`,
      VOLATILITY:     `Explore volatility-adjusted ${field} signals that predict cross-sectional return dispersion`,
      LIQUIDITY:      `Find ${field}-driven liquidity premium signals with stable risk-adjusted returns`,
      MICROSTRUCTURE: `Discover intraday microstructure patterns in ${field} that predict next-day returns`,
      SENTIMENT:      `Extract sentiment-driven mispricings using ${field} with earnings surprise confirmation`,
    };
    objective = templates[category] ?? `Discover robust ${category.toLowerCase()} alphas using ${field}`;
  }

  // Record to history
  objectiveHistory.push({ at: new Date().toISOString(), category, field, objective });
  saveObjectiveHistory();

  return objective;
}

async function tickScheduler() {
  schedulerState.lastTickAt = new Date().toISOString();

  // Heal any stuck activeRepair first (e.g. createRun threw before spawning the process)
  if (AUTO_REPAIR_ENABLED) await recoverStuckActiveRepair();

  // Drain repair queue opportunistically: if items are waiting and nothing is
  // running, kick off the next repair without waiting for a mining run to finish.
  if (AUTO_REPAIR_ENABLED && autoLoopState.queue.length > 0 && !autoLoopState.activeRepair && !hasRunningRun()) {
    await startNextAutoRepairRun();
    return;
  }

  if (!schedulerState.enabled) return;
  if (!schedulerState.nextRunAt) scheduleNextRun("missing-next-run");
  if (Date.now() < Date.parse(schedulerState.nextRunAt)) return;
  if (hasRunningRun()) {
    schedulerState.lastSkipReason = "A run is still active.";
    scheduleNextRun("active-run-skip");
    return;
  }

  // Generate a fresh, evolving objective for each scheduled run
  const dynamicObjective = await generateScheduledObjective();
  schedulerState.objective = dynamicObjective;

  const state = await createRun(
    {
      runId: `scheduled-mining-${new Date().toISOString().replaceAll(":", "-").replace(/\.\d{3}Z$/, "Z")}`,
      mode: schedulerState.mode,
      engine: schedulerState.engine,
      objective: dynamicObjective,
      rounds: schedulerState.rounds,
      batchSize: schedulerState.batchSize,
      concurrency: schedulerState.concurrency,
      simulationSettings: schedulerState.simulationSettings,
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
  // startup / error / skip → 2-min warmup so first run fires quickly after deploy
  // run-finished → 1-min cooldown for continuous 24/7 mining
  // everything else (scheduler-update, active-run-skip, etc.) → full interval
  let cooldown;
  if (reason === "run-finished") cooldown = 1;
  else if (reason === "startup" || reason === "scheduler-error" || reason === "missing-next-run") cooldown = 2;
  else cooldown = schedulerState.intervalMinutes;
  const next = new Date(Date.now() + cooldown * 60_000);
  schedulerState.nextRunAt = schedulerState.enabled ? next.toISOString() : null;
  schedulerState.lastScheduleReason = reason;
}

async function buildRunsIndex(authContext = systemAuthContext()) {
  const dirs = await visibleRunDirs(await listRunDirs(RUNS_DIR), authContext);
  const diversityStats = await latestDiversityStats(dirs);
  const activeList = [...activeRuns.values()].filter((run) => run.status === "running" && canAccessOwner(authContext, run.ownerId));
  // Annotate python-v2 active runs with live progress stats from progress.jsonl
  await Promise.all(activeList.map(async (run) => {
    if (run.engine !== "legacy-js") {
      run.progressStats = await readRunProgressStats(run.outputDir);
    }
  }));
  return {
    active: activeList,
    recent: await Promise.all(dirs.sort().reverse().slice(0, 30).map((runId) => readRun(runId, true))),
    storedRuns: dirs.sort().reverse(),
    scheduler: publicSchedulerState(),
    autoLoop: publicAutoLoopState(authContext),
    diversityStats,
  };
}

async function buildAlphaLibrary() {
  const dirs = await listRunDirs(RUNS_DIR);
  const seen = new Set();
  const entries = [];
  for (const runId of dirs.sort().reverse()) {
    const poolPath = path.join(RUNS_DIR, runId, "pool.json");
    const pool = await maybeReadJson(poolPath);
    if (!pool || !Array.isArray(pool.records)) continue;
    const runTs = runId.match(/\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z/)?.[0]?.replace(/-/g, (m, o, s) => o > 10 ? ':' : m) ?? null;
    for (const rec of pool.records) {
      const cand = rec.candidate || {};
      const bt = rec.backtest || {};
      const alphaId = bt.alpha_id || cand.id;
      if (!alphaId || seen.has(alphaId)) continue;
      const qualifiedByDsr = typeof rec.dsr === 'number' && rec.dsr >= 0;
      const qualifiedDegraded = rec.degradedQualified === true;
      if (!qualifiedByDsr && !qualifiedDegraded) continue;
      seen.add(alphaId);
      entries.push({
        alphaId,
        runId,
        qualifiedAt: runTs,
        category: cand.category || bt.category || 'UNKNOWN',
        expression: cand.expression || bt.expression || '',
        hypothesis: cand.hypothesis || '',
        originRefs: cand.origin_refs || [],
        sharpe: bt.sharpe ?? null,
        fitness: bt.fitness ?? null,
        turnover: bt.turnover ?? null,
        dsr: rec.dsr ?? null,
        degraded: qualifiedDegraded && !qualifiedByDsr,
        optRounds: cand.opt_rounds ?? 0,
      });
    }
  }
  entries.sort((a, b) => (b.sharpe ?? -Infinity) - (a.sharpe ?? -Infinity));
  return { total: entries.length, entries };
}

async function readLlmRouterState() {
  // Try global state file first (written by Python after each run)
  const global = await maybeReadJson(path.join(RUNS_DIR, "llm_router_state.json"));
  if (global) return global;
  // Fall back: scan most recent run directories for per-run state
  try {
    const entries = await safeReadDir(RUNS_DIR);
    const runDirs = entries
      .filter(f => f.startsWith("scheduled-") || f.startsWith("manual-") || f.startsWith("repair-"))
      .sort()
      .reverse()
      .slice(0, 10);
    for (const d of runDirs) {
      const s = await maybeReadJson(path.join(RUNS_DIR, d, "llm_router_state.json"));
      if (s) return s;
    }
  } catch (_) {}
  return null;
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

function buildRunCommand({ engine, mode, objective, rounds, batchSize, concurrency, simulationSettings, outputDir, repairContextPath }) {
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
  const pythonArgs = [
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
  ];
  if (repairContextPath) pythonArgs.push("--repair-context", repairContextPath);
  if (simulationSettings && Object.keys(simulationSettings).length > 0) {
    pythonArgs.push("--sim-settings", JSON.stringify(simulationSettings));
  }
  return { bin: PYTHON_BIN, args: pythonArgs };
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
  const candidates = await loadRunScoredCandidates(runState.outputDir);
  const best = bestRepairCandidate(candidates);
  const bestMetrics = extractBestMetrics(best);
  const targetMet = best ? candidateMeetsRepairTarget(best) : false;

  const submitted = await tryAutoSubmitCandidates(candidates, runState.runId, runState.ownerId);
  if (submitted) {
    if (runState.source === "repair") {
      pushRepairHistory({ alphaId: runState.parentAlphaId, expression: autoLoopState.activeRepair?.expression ?? null, failedChecks: autoLoopState.activeRepair?.failedChecks ?? [], repairDepth: runState.repairDepth ?? 0, outcome: "submitted", runId: runState.runId, bestMetrics });
      await clearActiveRepair(runState.runId);
    }
    await startNextAutoRepairRun();
    return;
  }

  // Target met (Sharpe ≥ 1.25, fitness ≥ 1.0, turnover ≤ 40%) — stop repairing, count as success
  if (targetMet) {
    if (runState.source === "repair") {
      pushRepairHistory({ alphaId: runState.parentAlphaId, expression: autoLoopState.activeRepair?.expression ?? null, failedChecks: autoLoopState.activeRepair?.failedChecks ?? [], repairDepth: runState.repairDepth ?? 0, outcome: "target-met", runId: runState.runId, bestMetrics });
      await clearActiveRepair(runState.runId);
    }
    await saveAutoLoopState();
    await startNextAutoRepairRun();
    return;
  }

  const repairDepth = runState.source === "repair" ? Number(runState.repairDepth ?? -1) + 1 : 0;

  // Max rounds reached without hitting target — abandon this candidate
  if (runState.source === "repair" && repairDepth >= AUTO_REPAIR_MAX_ROUNDS) {
    pushAutoLoopEvent({ type: "repair-abandoned", runId: runState.runId, ownerId: runState.ownerId });
    pushRepairHistory({ alphaId: runState.parentAlphaId, expression: autoLoopState.activeRepair?.expression ?? null, failedChecks: autoLoopState.activeRepair?.failedChecks ?? [], repairDepth: runState.repairDepth ?? 0, outcome: "abandoned", runId: runState.runId, bestMetrics });
    await clearActiveRepair(runState.runId);
    await saveAutoLoopState();
    await startNextAutoRepairRun();
    return;
  }

  // Still within repair budget — re-queue repair-worthy candidates
  const repairCandidates = chooseRepairCandidates(candidates, 3);
  if (repairCandidates.length > 0) {
    const source = runState.source === "repair" ? "repair-run-finished" : "run-finished";
    for (const candidate of repairCandidates) {
      const gate = evaluateCandidateGate(candidate);
      await enqueueRepairFromGate({
        alpha: summarizeCandidateAsAlpha(candidate),
        gate,
        source,
        sourceRunId: runState.runId,
        repairDepth,
        rootAlphaId: runState.parentAlphaId ?? candidate.alphaId,
        ownerId: runState.ownerId,
        _category: candidate._category ?? null,
      });
    }
    if (runState.source === "repair") {
      pushRepairHistory({ alphaId: runState.parentAlphaId, expression: autoLoopState.activeRepair?.expression ?? null, failedChecks: autoLoopState.activeRepair?.failedChecks ?? [], repairDepth: runState.repairDepth ?? 0, outcome: "re-queued", runId: runState.runId, bestMetrics });
    }
  } else {
    pushAutoLoopEvent({ type: "no-repair-candidate", runId: runState.runId, ownerId: runState.ownerId });
    if (runState.source === "repair") {
      pushRepairHistory({ alphaId: runState.parentAlphaId, expression: autoLoopState.activeRepair?.expression ?? null, failedChecks: autoLoopState.activeRepair?.failedChecks ?? [], repairDepth: runState.repairDepth ?? 0, outcome: "no-candidate", runId: runState.runId, bestMetrics });
    }
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

async function enqueueRepairFromGate({ alpha, gate, source, sourceRunId, repairDepth = 0, rootAlphaId = null, ownerId = "default", _category = null }) {
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

  const item = buildRepairQueueItem({ alpha, gate, source, sourceRunId, repairDepth, rootAlphaId, ownerId, _category: _category ?? alpha?._category ?? null });
  autoLoopState.queue.push(item);
  autoLoopState.lastAction = `Queued repair for ${alphaId}`;
  pushAutoLoopEvent({ type: "repair-queued", alphaId, ownerId, source, repairDepth, failedChecks: item.failedChecks });
  await saveAutoLoopState();
  return { queued: true, runId: null, reason: "queued" };
}

function buildRepairQueueItem({ alpha, gate, source, sourceRunId, repairDepth, rootAlphaId, ownerId = "default", _category = null }) {
  const alphaId = alpha?.id ?? gate?.alphaId;
  const failedChecks = failedCheckNames(gate);
  const createdAt = new Date().toISOString();
  return {
    id: sanitizeRunId(`repair-item-${alphaId}-${createdAt}`),
    parentAlphaId: alphaId,
    rootAlphaId: rootAlphaId ?? alphaId,
    expression: alpha?.expression ?? null,
    _category: _category ?? alpha?._category ?? null,
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

// Recover from stuck activeRepair: if it points to a runId that is no longer running
// (e.g., createRun threw before spawning, or server restarted mid-repair), clear it.
async function recoverStuckActiveRepair() {
  if (!autoLoopState.activeRepair) return;
  const runId = autoLoopState.activeRepair.runId;
  if (runId && activeRuns.has(runId)) return; // run is in memory — still live
  // activeRepair points to nothing running — clear it
  console.log(`[repair] Clearing stuck activeRepair ${runId ?? "(no runId)"} — no matching active run`);
  pushAutoLoopEvent({ type: "repair-stuck-cleared", runId: runId ?? null, alphaId: autoLoopState.activeRepair.parentAlphaId });
  autoLoopState.activeRepair = null;
  await saveAutoLoopState();
}

async function startNextAutoRepairRun() {
  if (!AUTO_REPAIR_ENABLED) return null;
  await recoverStuckActiveRepair();
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
  try {
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
  } catch (err) {
    // createRun failed — clear activeRepair so the next tick can retry or move on
    console.error(`[repair] createRun failed for ${runId}: ${err?.message ?? err}`);
    pushAutoLoopEvent({ type: "repair-start-failed", runId, alphaId: item.parentAlphaId, error: err?.message ?? String(err) });
    pushRepairHistory({ alphaId: item.parentAlphaId, expression: item.expression ?? null, failedChecks: item.failedChecks ?? [], repairDepth: item.repairDepth ?? 0, outcome: "start-failed", runId, bestMetrics: null });
    autoLoopState.activeRepair = null;
    await saveAutoLoopState();
    return null;
  }
  return runId;
}

async function clearActiveRepair(runId) {
  if (!autoLoopState.activeRepair || autoLoopState.activeRepair.runId !== runId) return;
  autoLoopState.activeRepair = null;
  await saveAutoLoopState();
}

async function loadRunScoredCandidates(outputDir) {
  // python-v2: read pool.json records
  const pool = await maybeReadJson(path.join(outputDir, "pool.json"));
  if (pool?.records) {
    return pool.records
      .filter((r) => r.backtest?.alpha_id)
      .map((r) => normalizePythonV2Candidate(r));
  }
  // legacy-js: read batch-round-N.json scored arrays
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

function normalizePythonV2Candidate(record) {
  const bt = record.backtest ?? {};
  const cand = record.candidate ?? {};
  const ortho = record.orthogonality ?? {};
  const checks = [];
  const sharpe = bt.sharpe ?? null;
  const fitness = bt.fitness ?? null;
  const turnover = bt.turnover ?? null;

  // Case A/B: daily PnL available — use DSR as primary quality gate
  if (record.dsr != null) {
    const dsr = Number(record.dsr);
    if (dsr < 0.95) {
      checks.push({ name: "DSR", result: "FAIL", value: dsr, limit: 0.95 });
    } else {
      checks.push({ name: "DSR", result: "PASS", value: dsr, limit: 0.95 });
    }
    if (ortho.passed === false) {
      checks.push({ name: "ORTHOGONALITY", result: "FAIL", value: ortho.correlation ?? null, limit: 0.5 });
    } else if (ortho.passed === true) {
      checks.push({ name: "ORTHOGONALITY", result: "PASS", value: ortho.correlation ?? null, limit: 0.5 });
    }
  } else if (record.status === "no_daily_pnl") {
    // Case C (degraded): no daily PnL — use BRAIN aggregate metrics as gate proxy.
    // Threshold is intentionally looser than Case A/B (0.5 vs DSR 0.95) because
    // we want weak-signal alphas to enter the pool and be improved by the repair loop.
    if (sharpe != null) {
      checks.push({ name: "SHARPE", result: sharpe >= 0.5 ? "PASS" : "FAIL", value: sharpe, limit: 0.5 });
    }
    if (fitness != null) {
      checks.push({ name: "FITNESS", result: fitness >= 0.5 ? "PASS" : "FAIL", value: fitness, limit: 0.5 });
    }
    // TURNOVER check: must be in BRAIN's acceptable range
    if (turnover != null) {
      const turnoverOk = turnover >= 0.01 && turnover <= 0.7;
      checks.push({ name: "TURNOVER", result: turnoverOk ? "PASS" : "FAIL", value: turnover, limit: 0.7 });
    }
    // Brain orthogonality check result if available
    const bc = record.brainCheckOrthogonality ?? {};
    if (bc.passed === false) {
      checks.push({ name: "SELF_CORRELATION", result: "FAIL", value: bc.max_abs_correlation ?? null, limit: 0.7 });
    } else if (bc.passed === true) {
      checks.push({ name: "SELF_CORRELATION", result: "PASS", value: bc.max_abs_correlation ?? null, limit: 0.7 });
    }
  }

  const dsr = record.dsr ?? null;
  return {
    alphaId: bt.alpha_id,
    expression: cand.expression ?? null,
    stage: "IS",
    totalScore: (dsr != null ? dsr : (sharpe ?? 0)),
    checks,
    metrics: {
      isSharpe: sharpe,
      testSharpe: sharpe,
      fitness,
      turnover,
      netSharpe: bt.net_sharpe ?? null,
    },
    _engine: "python-v2",
    _category: cand.category ?? null,
    _degraded: record.status === "no_daily_pnl",
  };
}

function chooseRepairCandidate(candidates) {
  return candidates
    .filter((candidate) => candidate.alphaId && !evaluateCandidateGate(candidate).ok)
    .sort((left, right) => repairPriority(right) - repairPriority(left))[0] ?? null;
}

function chooseRepairCandidates(candidates, maxItems = 3) {
  return candidates
    .filter((candidate) => candidate.alphaId && !evaluateCandidateGate(candidate).ok)
    .sort((left, right) => repairPriority(right) - repairPriority(left))
    .slice(0, maxItems);
}

function repairPriority(candidate) {
  const metrics = candidate.metrics ?? {};
  const gate = evaluateCandidateGate(candidate);
  const failures = new Set(failedCheckNames(gate));
  let score = Number(candidate.totalScore ?? 0);
  if (metrics.testSharpe !== null && metrics.testSharpe > 0) score += 1.5;
  // Accept both legacy names (LOW_SHARPE) and python-v2 names (SHARPE / FITNESS)
  if (failures.has("LOW_SHARPE") || failures.has("SHARPE") || failures.has("LOW_FITNESS") || failures.has("FITNESS")) score += 1.0;
  // Fitness is equally important: low fitness is repaired with same priority as low Sharpe
  const fitness = toNumber(metrics.isFitness ?? metrics.fitness);
  if (Number.isFinite(fitness) && fitness < 0.5) score += 0.8;
  if (failures.has("SELF_CORRELATION")) score += 0.6;
  if (failures.has("LOW_SUB_UNIVERSE_SHARPE")) score += 0.4;
  if (failures.has("HIGH_TURNOVER") || failures.has("TURNOVER") || (metrics.turnover !== null && metrics.turnover > 0.7)) score += 0.8;
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
    // python-v2 stores fitness as metrics.fitness; legacy-js uses metrics.isFitness
    isFitness: toNumber(metrics.isFitness ?? metrics.fitness),
    turnover: toNumber(metrics.turnover),
    testSharpe: toNumber(metrics.testSharpe),
    testFitness: toNumber(metrics.testFitness ?? metrics.fitness),
    _category: candidate._category ?? null,
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
  // Accept both legacy names and python-v2 names
  if (names.has("LOW_SHARPE") || names.has("SHARPE") || names.has("LOW_FITNESS") || names.has("FITNESS") || names.has("DSR")) actions.push("strength repair via peer-relative blend / denominator swap");
  if (names.has("HIGH_TURNOVER") || names.has("TURNOVER")) actions.push("turnover repair via decay and slower windows");
  if (names.has("LOW_SUB_UNIVERSE_SHARPE")) actions.push("sub-universe repair via group/subindustry relative transform");
  if (names.has("CONCENTRATED_WEIGHT")) actions.push("concentration repair via rank/group_rank/truncation-safe transforms");
  if (names.has("SELF_CORRELATION") || names.has("ORTHOGONALITY")) actions.push("crowding repair via material concept/horizon/group change");
  if (alpha?.testSharpe !== null && alpha?.testSharpe <= 0) actions.push("robustness repair because testSharpe is non-positive");
  return actions.join("; ") || "targeted repair from gate feedback";
}

function failedCheckNames(gate) {
  return [...new Set((gate?.checks ?? []).filter((check) => check.result !== "PASS").map((check) => check.name))];
}

function candidateMeetsRepairTarget(candidate) {
  const m = candidate.metrics ?? {};
  const sharpe = toNumber(m.isSharpe);
  const fitness = toNumber(m.isFitness ?? m.fitness);
  const turnover = toNumber(m.turnover);
  return (
    Number.isFinite(sharpe) && sharpe >= REPAIR_TARGET_SHARPE &&
    Number.isFinite(fitness) && fitness >= REPAIR_TARGET_FITNESS &&
    Number.isFinite(turnover) && turnover <= REPAIR_TARGET_TURNOVER
  );
}

function bestRepairCandidate(candidates) {
  return (candidates ?? []).reduce((best, c) => {
    const s = toNumber(c.metrics?.isSharpe);
    const b = toNumber(best?.metrics?.isSharpe);
    return (Number.isFinite(s) && (!Number.isFinite(b) || s > b)) ? c : best;
  }, null);
}

function extractBestMetrics(candidate) {
  if (!candidate) return null;
  const m = candidate.metrics ?? {};
  const sharpe = toNumber(m.isSharpe);
  const fitness = toNumber(m.isFitness ?? m.fitness);
  const turnover = toNumber(m.turnover);
  return {
    isSharpe: Number.isFinite(sharpe) ? sharpe : null,
    fitness: Number.isFinite(fitness) ? fitness : null,
    turnover: Number.isFinite(turnover) ? turnover : null,
    expression: candidate.expression ?? null,
  };
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
    repairTargets: { sharpe: REPAIR_TARGET_SHARPE, fitness: REPAIR_TARGET_FITNESS, turnover: REPAIR_TARGET_TURNOVER },
    queueLength: queue.length,
    queue: queue.slice(0, 20),
    activeRepair,
    submitted: submitted.slice(-10).reverse(),
    events: events.slice(-30).reverse(),
    repairHistory: (autoLoopState.repairHistory ?? []).slice(-20).reverse(),
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
    repairHistory: Array.isArray(persisted?.repairHistory) ? persisted.repairHistory.slice(-50) : [],
    diversityStats: persisted?.diversityStats ?? null,
    lastAction: persisted?.lastAction ?? null,
  };
}

function pushRepairHistory(entry) {
  autoLoopState.repairHistory.push({ ...entry, completedAt: new Date().toISOString() });
  autoLoopState.repairHistory = autoLoopState.repairHistory.slice(-50);
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
  const summary = summarizeAlpha(after);
  recordAutoSubmission(alphaId, summary, gate, options.source ?? "manual-submit", ownerId);
  await saveAutoLoopState();
  // Write to shared feedback file so future Python runs can use this as a positive example
  void appendSubmittedAlphaFeedback({
    alphaId,
    expression: summary.expression ?? after?.regular?.code ?? null,
    category: options._category ?? null,
    isSharpe: summary.isSharpe,
    isFitness: summary.isFitness,
    testSharpe: summary.testSharpe,
    grade: summary.grade,
    submittedAt: new Date().toISOString(),
    source: options.source ?? "manual-submit",
  });
  return {
    submitted: true,
    alphaId,
    gate,
    submitResponse,
    alpha: summary,
  };
}

async function appendSubmittedAlphaFeedback(entry) {
  try {
    const feedbackPath = path.join(RUNS_DIR, "submitted_alphas.jsonl");
    await writeFile(feedbackPath, JSON.stringify(entry) + "\n", { flag: "a", encoding: "utf8" });
  } catch (_) {
    // non-fatal
  }
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
  const isSharpe = toNumber(alpha?.is?.sharpe);
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
  if (Number.isFinite(testSharpe)) {
    // Case A/B: daily PnL available — require positive OOS test Sharpe
    if (testSharpe <= 0) {
      reasons.push(`testSharpe must be positive, got ${testSharpe}.`);
    }
  } else {
    // Case C (degraded): BRAIN regular tier does not provide test.sharpe before
    // submission. Use IS Sharpe >= 0.5 as proxy gate to allow the factor to be
    // submitted so BRAIN can evaluate it and return real OOS results.
    if (!Number.isFinite(isSharpe) || isSharpe < 0.5) {
      reasons.push(`Degraded mode: IS Sharpe must be >= 0.5, got ${Number.isFinite(isSharpe) ? isSharpe.toFixed(3) : "missing"}.`);
    }
  }

  const turnover = toNumber(alpha?.is?.turnover);
  if (Number.isFinite(turnover)) {
    if (turnover < 0.01 || turnover > 0.70) {
      reasons.push(`Turnover ${(turnover * 100).toFixed(1)}% is outside valid range 1%-70%.`);
    }
  }

  const isFitness = toNumber(alpha?.is?.fitness);
  if (Number.isFinite(isFitness) && isFitness < 0.5) {
    reasons.push(`IS Fitness ${isFitness.toFixed(3)} is below minimum 0.5.`);
  }

  return {
    ok: reasons.length === 0,
    reasons,
    alphaId: alpha?.id ?? null,
    status,
    stage,
    testSharpe: Number.isFinite(testSharpe) ? testSharpe : null,
    isSharpe: Number.isFinite(isSharpe) ? isSharpe : null,
    degradedMode: !Number.isFinite(testSharpe),
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
    simulationSettings: schedulerState.simulationSettings,
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
    recentObjectives: objectiveHistory.slice(-10).reverse(),
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

async function readRunProgressStats(outputDir) {
  try {
    const text = await readFile(path.join(outputDir, "progress.jsonl"), "utf8");
    const lines = text.trim().split(/\r?\n/).filter(Boolean);
    let nGen = 0, nSim = 0, nGate = 0;
    const recentEvents = [];
    for (const line of lines) {
      const e = safeJson(line);
      if (!e) continue;
      if (e.stage === "evaluated") nSim++;
      if (e.stage === "rejected") nGen++;
      // Build human-readable log line for dashboard
      const at = e.at ? String(e.at).slice(11, 19) : "";
      let msg = "";
      if (e.stage === "started") msg = `启动 mode=${e.mode ?? ""} engine=python-v2`;
      else if (e.stage === "submitted") msg = `提交仿真 ${e.expression ? e.expression.slice(0, 60) : ""}`;
      else if (e.stage === "backtest_completed") msg = `回测完成 alpha_id=${e.alpha_id ?? "?"} sharpe=${e.sharpe ?? "?"}`;
      else if (e.stage === "mock_backtest") msg = `模拟回测 sharpe=${e.sharpe ?? "?"}`;
      else if (e.stage === "evaluated") msg = `已评估 ${e.candidate_id ?? ""} status=${e.status ?? ""}`;
      else if (e.stage === "degraded_evaluation") {
        const qual = e.degraded_qualified ? "✅ 入池" : "❌ 未达标";
        msg = `降级评估 ${e.candidate_id ?? ""} sharpe=${e.sharpe ?? "?"} ${qual}`;
      }
      else if (e.stage === "rejected") msg = `验证拒绝 ${e.candidate_id ?? ""} ${(e.errors ?? []).slice(0,2).join("; ")}`;
      else if (e.stage === "blocked") msg = `⚠️ 阻塞: ${e.reason ?? ""}`;
      else if (e.stage === "waiting") msg = `⏳ 等待配额: ${e.reason ?? ""}`;
      else if (e.stage === "finished") msg = `完成 gen=${e.summary?.generatedCandidates ?? 0} sim=${e.summary?.total_brain_simulations ?? 0}`;
      else msg = `${e.stage}`;
      if (msg) recentEvents.push(`${at} ${msg}`);
    }
    nGen += nSim; // total generated = simulated + rejected
    // pick up qualified count from pool.json if already written
    const pool = await maybeReadJson(path.join(outputDir, "pool.json"));
    if (pool) nGate = pool.qualified ?? 0;
    return { nGen, nSim, nGate, recentEvents: recentEvents.slice(-30) };
  } catch {
    return null;
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
  <h1>欢迎使用 QuantBrain</h1>
  <p class="sub">创建账户以开始挖掘因子</p>

  <div class="section-label">您的账户</div>
  <div class="field"><label>用户名</label><input id="userId" placeholder="例如 alice" autocomplete="username"></div>
  ${regCodeRequired ? '<div class="field"><label>注册码</label><input id="regCode" type="password" placeholder="请向管理员索取注册码"></div>' : ''}

  <div class="divider"></div>
  <div class="section-label">WorldQuant BRAIN 凭据</div>
  <div class="field"><label>BRAIN 邮箱</label><input id="email" type="email" placeholder="your@email.com" autocomplete="email"></div>
  <div class="field"><label>BRAIN 密码</label><input id="password" type="password" placeholder="••••••••" autocomplete="current-password"></div>

  <button class="btn" id="registerBtn">创建账户并进入</button>
  <div class="status" id="status"></div>
  <div class="login-link">已有账户？<a href="#" id="switchToLogin">登录</a></div>

  <div id="loginSection" style="display:none">
    <div class="divider"></div>
    <div class="section-label">登录</div>
    <div class="field"><label>用户名</label><input id="loginUserId" placeholder="您的用户名" autocomplete="username"></div>
    <div class="field"><label>令牌</label><input id="loginToken" type="password" placeholder="粘贴您的令牌" autocomplete="current-password"></div>
    <button class="btn" id="loginBtn" style="background:#007aff">登录</button>
    <div class="status" id="loginStatus"></div>
    <div class="login-link"><a href="#" id="switchToRegister">← 创建新账户</a></div>
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
    if (!userId || !email || !password) { st.className='status error'; st.textContent='所有字段均为必填项。'; return; }
    reg.disabled = true; reg.textContent = '创建中…';
    try {
      const r = await fetch('/account/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ userId, email, password, registrationCode: regCode })
      });
      const d = await r.json();
      if (!d.ok) { st.className='status error'; st.textContent = d.error ?? '注册失败。'; reg.disabled=false; reg.textContent='创建账户并进入'; return; }
      localStorage.setItem('qb_token', d.token);
      localStorage.setItem('qb_user', d.userId);
      st.className='status success'; st.textContent = '账户已创建！正在进入仪表盘…';
      setTimeout(() => location.href = '/', 800);
    } catch(e) { st.className='status error'; st.textContent='网络错误：' + e.message; reg.disabled=false; reg.textContent='创建账户并进入'; }
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
    if (!token) { st2.className='status error'; st2.textContent='令牌为必填项。'; return; }
    this.disabled = true; this.textContent = '登录中…';
    try {
      const r = await fetch('/account', { headers: { 'Authorization': 'Bearer ' + token } });
      if (r.ok) {
        localStorage.setItem('qb_token', token);
        if (userId) localStorage.setItem('qb_user', userId);
        location.href = '/';
      } else {
        st2.className='status error'; st2.textContent='令牌无效。';
        this.disabled=false; this.textContent='登录';
      }
    } catch(e) { st2.className='status error'; st2.textContent='网络错误。'; this.disabled=false; this.textContent='登录'; }
  });
})();
</script>
</body>
</html>`;
}

function dashboardHtml() {
  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="dashboard-token" content="${process.env.ADMIN_TOKEN ?? process.env.DASHBOARD_TOKEN ?? ''}">
<meta name="registration-code-required" content="${REGISTRATION_CODE ? 'true' : 'false'}">
<title>QuantBrain</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root{
  --bg:#080c14;--surface:#0d1117;--card:#111927;--card2:#151e2e;
  --border:#1e2d42;--border2:#243347;
  --blue:#3b82f6;--blue-dim:rgba(59,130,246,0.12);
  --green:#10b981;--green-dim:rgba(16,185,129,0.12);
  --amber:#f59e0b;--amber-dim:rgba(245,158,11,0.12);
  --red:#ef4444;--red-dim:rgba(239,68,68,0.12);
  --purple:#8b5cf6;--purple-dim:rgba(139,92,246,0.12);
  --t1:#f1f5f9;--t2:#94a3b8;--t3:#475569;--t4:#2d3f55;
}
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%;overflow:hidden;font-family:'Inter',-apple-system,sans-serif;background:var(--bg);color:var(--t1);-webkit-font-smoothing:antialiased}
body{display:flex}

/* Sidebar */
.sidebar{width:220px;min-width:220px;background:var(--surface);border-right:1px solid var(--border);display:flex;flex-direction:column;padding:0}
.sb-brand{padding:20px 18px 16px;display:flex;align-items:center;gap:10px;border-bottom:1px solid var(--border)}
.sb-logo{width:32px;height:32px;background:var(--blue);border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:15px;font-weight:800;color:#fff;flex-shrink:0}
.sb-name{font-size:15px;font-weight:700;color:var(--t1);letter-spacing:-0.3px}
.sb-nav{flex:1;padding:10px 10px;display:flex;flex-direction:column;gap:2px}
.sb-item{display:flex;align-items:center;gap:10px;padding:9px 12px;border-radius:8px;cursor:pointer;font-size:13px;color:var(--t3);transition:all .15s;border:1px solid transparent}
.sb-item:hover{background:var(--card);color:var(--t2);border-color:var(--border)}
.sb-item.active{background:var(--blue-dim);color:var(--blue);border-color:rgba(59,130,246,0.25);font-weight:500}
.sb-icon{font-size:15px;width:20px;text-align:center;flex-shrink:0}
.sb-bottom{padding:12px 10px;border-top:1px solid var(--border)}
.sb-user{display:flex;align-items:center;gap:10px;padding:9px 12px;border-radius:8px;font-size:12px;color:var(--t3)}
.sb-avatar{width:28px;height:28px;background:var(--blue);border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:700;color:#fff;flex-shrink:0}

/* Main */
.main{flex:1;display:flex;flex-direction:column;overflow:hidden}
.topbar{height:52px;min-height:52px;display:flex;align-items:center;justify-content:space-between;padding:0 24px;border-bottom:1px solid var(--border);background:var(--surface)}
.tb-title{font-size:15px;font-weight:600;color:var(--t1)}
.tb-right{display:flex;align-items:center;gap:12px}
.status-pill{display:flex;align-items:center;gap:6px;font-size:12px;color:var(--t2);background:var(--card);border:1px solid var(--border);padding:5px 12px;border-radius:20px}
.pulse{width:7px;height:7px;background:var(--green);border-radius:50%;box-shadow:0 0 6px rgba(16,185,129,0.7);animation:pulse 2.5s ease-in-out infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
.panel{flex:1;overflow-y:auto;padding:24px;display:none}
.panel.active{display:block}

/* Mining Status Banner */
.mining-banner{display:flex;align-items:center;gap:12px;padding:14px 18px;border-radius:12px;margin-bottom:20px;border:1px solid var(--border);transition:all .4s ease}
.mining-banner.active-mining{background:rgba(16,185,129,0.06);border-color:rgba(16,185,129,0.35)}
.mining-banner.idle-mining{background:var(--card);border-color:var(--border)}
.mining-indicator{width:10px;height:10px;border-radius:50%;flex-shrink:0;transition:all .4s}
.mining-indicator.active-ind{background:var(--green);box-shadow:0 0 10px rgba(16,185,129,0.8);animation:pulse 2s ease-in-out infinite}
.mining-indicator.idle-ind{background:var(--t4)}
.mining-label{font-size:13px;font-weight:700;letter-spacing:.02em}
.mining-label.active-lbl{color:var(--green)}
.mining-label.idle-lbl{color:var(--t3)}
.mining-task{font-size:12px;color:var(--t2);font-family:'JetBrains Mono',monospace;flex:1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.mining-elapsed{font-size:11px;color:var(--t3);font-family:'JetBrains Mono',monospace;flex-shrink:0;margin-left:auto}
.mining-sep{width:1px;height:20px;background:var(--border2);flex-shrink:0}

/* KPI row */
.kpi-row{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:20px}
.kpi{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:18px 20px}
.kpi-label{font-size:11px;font-weight:500;color:var(--t3);text-transform:uppercase;letter-spacing:.06em;margin-bottom:8px}
.kpi-val{font-size:28px;font-weight:700;color:var(--t1);letter-spacing:-1px;font-family:'JetBrains Mono',monospace}
.kpi-sub{font-size:11px;color:var(--t3);margin-top:4px}
.kpi-sub.up{color:var(--green)}
.kpi-sub.down{color:var(--red)}

/* Grid panels */
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px}
.grid3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;margin-bottom:16px}
.box{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:20px}
.box-title{font-size:11px;font-weight:600;color:var(--t3);text-transform:uppercase;letter-spacing:.06em;margin-bottom:14px}

/* Countdown */
.cd-wrap{text-align:center;padding:8px 0 20px}
.cd-time{font-size:42px;font-weight:300;font-family:'JetBrains Mono',monospace;color:var(--t1);letter-spacing:-1px;font-variant-numeric:tabular-nums}
.cd-label{font-size:11px;color:var(--t3);text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px}
.cd-sub{font-size:11px;color:var(--t3);margin-top:6px}

/* Pipeline stages */
.stages{display:flex;gap:8px;margin-top:4px}
.stage{flex:1;background:var(--card2);border:1px solid var(--border);border-radius:10px;padding:14px 8px;text-align:center;cursor:default;position:relative;transition:border-color .2s}
.stage:hover{border-color:var(--border2)}
.stage-n{font-size:22px;font-weight:700;font-family:'JetBrains Mono',monospace;margin-bottom:4px}
.stage-l{font-size:10px;color:var(--t3);text-transform:uppercase;letter-spacing:.05em}
.stage-full{font-size:9px;color:var(--t4);margin-top:2px;letter-spacing:.03em}
.c-blue{color:var(--blue)}.c-amber{color:var(--amber)}.c-purple{color:var(--purple)}.c-red{color:var(--red)}.c-green{color:var(--green)}

/* Stage tooltip */
.stage-tip{position:absolute;bottom:calc(100% + 8px);left:50%;transform:translateX(-50%);background:#1a2840;border:1px solid var(--border2);color:var(--t1);font-size:10px;padding:8px 12px;border-radius:8px;white-space:nowrap;z-index:200;font-family:'Inter',sans-serif;pointer-events:none;opacity:0;transition:opacity .15s;box-shadow:0 4px 16px rgba(0,0,0,0.4);min-width:160px;text-align:left}
.stage-tip .tip-title{font-weight:600;font-size:11px;margin-bottom:4px}
.stage-tip .tip-body{color:var(--t2);font-size:10px;line-height:1.5}
.stage:hover .stage-tip{opacity:1}

/* Current task strip */
.current-task-strip{margin-top:14px;padding:10px 12px;background:var(--card2);border:1px solid var(--border);border-radius:8px;display:flex;align-items:center;gap:8px}
.ct-dot{width:6px;height:6px;border-radius:50%;flex-shrink:0}
.ct-dot.running{background:var(--blue);animation:pulse 1.5s ease-in-out infinite}
.ct-dot.idle{background:var(--t4)}
.ct-label{font-size:10px;color:var(--t3);text-transform:uppercase;letter-spacing:.05em;flex-shrink:0}
.ct-text{font-size:11px;color:var(--t2);font-family:'JetBrains Mono',monospace;flex:1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}

/* LLM bars */
.llm-list{display:flex;flex-direction:column;gap:14px}
.llm-row-head{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:6px}
.llm-name{font-size:13px;font-weight:500;color:var(--t1)}
.llm-tag{font-size:10px;color:var(--t3);margin-left:6px;background:var(--card2);padding:2px 7px;border-radius:4px;border:1px solid var(--border)}
.llm-rate{font-size:12px;font-weight:600;font-family:'JetBrains Mono',monospace}
.bar-track{height:3px;background:var(--border2);border-radius:2px;overflow:hidden}
.bar-fill{height:100%;border-radius:2px;transition:width .5s ease}
.llm-active-badge{font-size:9px;font-weight:700;color:var(--green);background:var(--green-dim);border:1px solid rgba(16,185,129,0.25);padding:1px 6px;border-radius:4px;margin-left:6px;letter-spacing:.04em;vertical-align:middle}

/* Repair queue */
.queue-list{display:flex;flex-direction:column;gap:6px}
.queue-row{display:flex;align-items:center;gap:10px;padding:10px 12px;background:var(--card2);border:1px solid var(--border);border-radius:8px}
.q-expr{font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--t2);flex:1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.q-tag{font-size:10px;font-weight:600;color:var(--amber);background:var(--amber-dim);padding:2px 8px;border-radius:4px;flex-shrink:0;border:1px solid rgba(245,158,11,0.2)}
.q-depth{font-size:10px;color:var(--t3);flex-shrink:0;font-family:'JetBrains Mono',monospace;width:28px;text-align:right}

/* Activity */
.act-list{display:flex;flex-direction:column;gap:12px}
.act-row{display:flex;align-items:flex-start;gap:10px}
.act-dot{width:26px;height:26px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:11px;flex-shrink:0;font-weight:700}
.act-dot.s{background:var(--green-dim);color:var(--green)}
.act-dot.r{background:var(--amber-dim);color:var(--amber)}
.act-dot.f{background:var(--red-dim);color:var(--red)}
.act-expr{font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--t2);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.act-meta{font-size:10px;color:var(--t3);margin-top:3px}

/* Alpha status chips */
.alpha-status-row{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:16px}
.alpha-chip{display:flex;flex-direction:column;align-items:center;background:var(--card2);border:1px solid var(--border);border-radius:8px;padding:10px 14px;flex:1;min-width:80px}
.alpha-chip-val{font-size:20px;font-weight:700;font-family:'JetBrains Mono',monospace}
.alpha-chip-label{font-size:9px;color:var(--t3);text-transform:uppercase;letter-spacing:.05em;margin-top:3px}

/* Factor library */
.lib-filter{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:16px}
.lib-pill{padding:4px 12px;border-radius:20px;font-size:11px;font-weight:600;cursor:pointer;border:1px solid var(--border);background:var(--card2);color:var(--t2);transition:all .15s;letter-spacing:.03em}
.lib-pill.active,.lib-pill:hover{background:var(--blue);border-color:var(--blue);color:#fff}
.lib-card{background:var(--card2);border:1px solid var(--border);border-radius:10px;padding:14px 16px;margin-bottom:10px;transition:border-color .15s}
.lib-card:hover{border-color:var(--blue)}
.lib-card-header{display:flex;align-items:center;gap:8px;margin-bottom:8px}
.lib-badge{padding:2px 8px;border-radius:4px;font-size:9px;font-weight:700;letter-spacing:.07em;text-transform:uppercase}
.lib-badge-QUALITY{background:#1a472a;color:#4ade80}.lib-badge-MOMENTUM{background:#1e3a5f;color:#60a5fa}
.lib-badge-REVERSAL{background:#4a1942;color:#e879f9}.lib-badge-LIQUIDITY{background:#422006;color:#fb923c}
.lib-badge-VOLATILITY{background:#1a1a2e;color:#a78bfa}.lib-badge-MICROSTRUCTURE{background:#0f2724;color:#34d399}
.lib-badge-SENTIMENT{background:#3b1a1a;color:#f87171}.lib-badge-UNKNOWN{background:var(--card);color:var(--t3)}
.lib-alphaid{font-size:10px;color:var(--t3);font-family:'JetBrains Mono',monospace;flex:1;text-align:right}
.lib-origin{font-size:9px;padding:2px 7px;border-radius:4px;background:var(--card);border:1px solid var(--border);color:var(--t3)}
.lib-hypothesis{font-size:12px;color:var(--t1);line-height:1.55;margin-bottom:10px}
.lib-expr{font-family:'JetBrains Mono',monospace;font-size:10px;color:var(--t2);background:var(--card);border:1px solid var(--border);border-radius:6px;padding:7px 10px;margin-bottom:10px;word-break:break-all;line-height:1.6;max-height:48px;overflow:hidden;cursor:pointer;transition:max-height .2s}
.lib-expr.expanded{max-height:300px}
.lib-metrics{display:flex;gap:16px}
.lib-metric{display:flex;flex-direction:column;gap:2px}
.lib-metric-val{font-size:13px;font-weight:700;font-family:'JetBrains Mono',monospace}
.lib-metric-lbl{font-size:9px;color:var(--t3);text-transform:uppercase;letter-spacing:.05em}
.lib-empty{text-align:center;padding:48px 0;color:var(--t3);font-size:13px}
.lib-stats{display:flex;gap:24px;margin-bottom:16px;flex-wrap:wrap}
.lib-stat{display:flex;flex-direction:column;gap:3px}
.lib-stat-val{font-size:22px;font-weight:700;font-family:'JetBrains Mono',monospace;color:var(--t1)}
.lib-stat-lbl{font-size:9px;color:var(--t3);text-transform:uppercase;letter-spacing:.05em}

/* Idea box */
.idea-field{width:100%;background:var(--card2);border:1px solid var(--border);border-radius:8px;padding:10px 13px;color:var(--t1);font-size:13px;font-family:inherit;resize:none;height:72px;line-height:1.5;outline:none;transition:border-color .15s}
.idea-field:focus{border-color:var(--blue)}
.idea-field::placeholder{color:var(--t4)}
.run-btn{margin-top:8px;width:100%;background:var(--blue);color:#fff;border:none;border-radius:8px;padding:10px;font-size:13px;font-weight:600;cursor:pointer;font-family:inherit;transition:opacity .15s}
.run-btn:hover{opacity:.85}
.run-btn:disabled{opacity:.45;cursor:not-allowed}
.run-status{font-size:11px;color:var(--t3);margin-top:6px;min-height:14px}

/* Divider */
.div{height:1px;background:var(--border);margin:16px 0}

/* Runs table */
.data-table{width:100%;border-collapse:collapse;font-size:12px}
.data-table th{text-align:left;padding:8px 12px;font-size:10px;font-weight:600;color:var(--t3);text-transform:uppercase;letter-spacing:.06em;border-bottom:1px solid var(--border)}
.data-table td{padding:10px 12px;border-bottom:1px solid var(--border);color:var(--t2);font-family:'JetBrains Mono',monospace;font-size:11px}
.data-table tr:last-child td{border-bottom:none}
.data-table tr:hover td{background:var(--card2)}
.jb{font-family:'JetBrains Mono',monospace}
.badge{display:inline-flex;align-items:center;padding:2px 8px;border-radius:4px;font-size:10px;font-weight:600}
.badge.run{background:var(--blue-dim);color:var(--blue);border:1px solid rgba(59,130,246,.2)}
.badge.done{background:var(--green-dim);color:var(--green);border:1px solid rgba(16,185,129,.2)}
.badge.fail{background:var(--red-dim);color:var(--red);border:1px solid rgba(239,68,68,.2)}
.badge.pass{background:var(--green-dim);color:var(--green);border:1px solid rgba(16,185,129,.2)}
.badge.no{background:var(--red-dim);color:var(--red);border:1px solid rgba(239,68,68,.2)}

/* Settings form */
.form-row{display:flex;flex-direction:column;gap:6px;margin-bottom:16px}
.form-label{font-size:11px;font-weight:600;color:var(--t3);text-transform:uppercase;letter-spacing:.06em}
.form-input,.form-select{background:var(--card2);border:1px solid var(--border);border-radius:8px;padding:9px 12px;color:var(--t1);font-size:13px;font-family:inherit;outline:none;width:100%;transition:border-color .15s}
.form-input:focus,.form-select:focus{border-color:var(--blue)}
.form-select option{background:var(--card)}
.save-btn{background:var(--blue);color:#fff;border:none;border-radius:8px;padding:10px 24px;font-size:13px;font-weight:600;cursor:pointer;font-family:inherit;transition:opacity .15s}
.save-btn:hover{opacity:.85}
</style>
</head>
<body>

<!-- Sidebar -->
<div class="sidebar">
  <div class="sb-brand">
    <div class="sb-logo">Q</div>
    <span class="sb-name">QuantBrain</span>
  </div>
  <nav class="sb-nav">
    <div class="sb-item active" data-tab="overview"><span class="sb-icon">⬡</span>总览</div>
    <div class="sb-item" data-tab="pipeline"><span class="sb-icon">◈</span>流水线</div>
    <div class="sb-item" data-tab="alphas"><span class="sb-icon">◆</span>因子</div>
    <div class="sb-item" data-tab="library"><span class="sb-icon">▣</span>因子库</div>
    <div class="sb-item" data-tab="repair"><span class="sb-icon">⟳</span>修复工作区</div>
    <div class="sb-item" data-tab="knowledge"><span class="sb-icon">◎</span>知识库</div>
    <div class="sb-item" data-tab="runs"><span class="sb-icon">≡</span>运行记录</div>
    <div class="sb-item" data-tab="settings"><span class="sb-icon">⚙</span>设置</div>
  </nav>
  <div class="sb-bottom">
    <div class="sb-user" id="sb-user">
      <div class="sb-avatar" id="sb-av">?</div>
      <span id="sb-uname" style="font-size:12px;color:var(--t3)">访客</span>
    </div>
  </div>
</div>

<!-- Main -->
<div class="main">
  <div class="topbar">
    <span class="tb-title" id="topbar-title">总览</span>
    <div class="tb-right">
      <div class="status-pill" id="sched-status">
        <div class="pulse"></div>
        <span id="sched-label">加载中…</span>
      </div>
      <div class="status-pill" style="font-family:'JetBrains Mono',monospace;font-size:12px" id="topbar-cd">–:––</div>
    </div>
  </div>

  <!-- Overview -->
  <div class="panel active" id="panel-overview">

    <!-- Mining Status Banner -->
    <div class="mining-banner idle-mining" id="mining-banner">
      <div class="mining-indicator idle-ind" id="mining-ind"></div>
      <span class="mining-label idle-lbl" id="mining-label">系统空闲</span>
      <div class="mining-sep"></div>
      <span class="mining-task" id="mining-task">当前无挖掘任务</span>
      <span class="mining-elapsed" id="mining-elapsed"></span>
    </div>

    <div class="kpi-row">
      <div class="kpi"><div class="kpi-label">今日提交</div><div class="kpi-val" id="k-submitted">–</div><div class="kpi-sub up" id="k-sub1"></div></div>
      <div class="kpi"><div class="kpi-label">闸门通过率</div><div class="kpi-val" id="k-gatepass">–</div><div class="kpi-sub" id="k-sub2">近期运行</div></div>
      <div class="kpi"><div class="kpi-label">修复队列</div><div class="kpi-val" id="k-repair">–</div><div class="kpi-sub" id="k-sub3"></div></div>
      <div class="kpi"><div class="kpi-label">今日花费</div><div class="kpi-val" id="k-spend">–</div><div class="kpi-sub" id="k-sub4">上限 $3.60</div></div>
    </div>
    <div class="grid3">
      <div class="box">
        <div class="box-title">Alpha 流水线 — 实时</div>
        <div class="stages">
          <div class="stage">
            <div class="stage-n c-blue" id="ps-gen">–</div>
            <div class="stage-l">生成</div>
            <div class="stage-full">Generate</div>
            <div class="stage-tip">
              <div class="tip-title">生成</div>
              <div class="tip-body">LLM 根据研究方向生成新的因子表达式。每轮生成多条候选 Alpha。</div>
            </div>
          </div>
          <div class="stage">
            <div class="stage-n c-amber" id="ps-val">–</div>
            <div class="stage-l">验证</div>
            <div class="stage-full">Validate</div>
            <div class="stage-tip">
              <div class="tip-title">验证</div>
              <div class="tip-body">语法检查 + 基本逻辑过滤。不合格的表达式在此阶段被剔除，节省仿真资源。</div>
            </div>
          </div>
          <div class="stage">
            <div class="stage-n c-purple" id="ps-sim">–</div>
            <div class="stage-l">仿真</div>
            <div class="stage-full">Simulate</div>
            <div class="stage-tip">
              <div class="tip-title">仿真</div>
              <div class="tip-body">提交至 WorldQuant BRAIN 进行历史回测。计算 Sharpe、Fitness 等绩效指标。</div>
            </div>
          </div>
          <div class="stage">
            <div class="stage-n c-red" id="ps-gate">–</div>
            <div class="stage-l">闸门</div>
            <div class="stage-full">Gate Check</div>
            <div class="stage-tip">
              <div class="tip-title">闸门检查</div>
              <div class="tip-body">用 Sharpe 比率、Fitness 阈值过滤因子。只有通过闸门的 Alpha 才会被提交。</div>
            </div>
          </div>
          <div class="stage">
            <div class="stage-n c-green" id="ps-sub">–</div>
            <div class="stage-l">提交</div>
            <div class="stage-full">Submit</div>
            <div class="stage-tip">
              <div class="tip-title">提交</div>
              <div class="tip-body">通过闸门的因子提交至 WorldQuant 生产池，等待人工评审与最终录用。</div>
            </div>
          </div>
        </div>
        <!-- Current task strip -->
        <div class="current-task-strip" id="current-task-strip">
          <div class="ct-dot idle" id="ct-dot"></div>
          <span class="ct-label">任务</span>
          <span class="ct-text" id="ct-text">空闲</span>
        </div>
      </div>
      <div class="box">
        <div class="box-title">LLM 路由器</div>
        <div class="llm-list" id="llm-list">
          <div style="font-size:12px;color:var(--t3)">加载中…</div>
        </div>
      </div>
      <div class="box">
        <div class="cd-wrap">
          <div class="cd-label">距下次运行</div>
          <div class="cd-time" id="cd-time">–:––</div>
          <div class="cd-sub" id="cd-sub">–</div>
        </div>
        <div class="div"></div>
        <div class="box-title">研究想法</div>
        <textarea class="idea-field" id="idea-field" placeholder="用自然语言描述你的因子想法…"></textarea>
        <button class="run-btn" id="run-btn">优化并运行</button>
        <div class="run-status" id="run-status"></div>
      </div>
    </div>
    <div class="grid2">
      <div class="box">
        <div class="box-title">修复队列</div>
        <div class="queue-list" id="repair-list">
          <div style="font-size:12px;color:var(--t3)">加载中…</div>
        </div>
      </div>
      <div class="box">
        <div class="box-title">近期活动</div>
        <div class="act-list" id="act-list">
          <div style="font-size:12px;color:var(--t3)">加载中…</div>
        </div>
      </div>
    </div>
  </div>

  <!-- Pipeline -->
  <div class="panel" id="panel-pipeline">
    <div class="box" style="margin-bottom:16px">
      <div class="box-title">活跃运行</div>
      <div id="pipeline-active" style="font-size:12px;color:var(--t3)">无活跃运行</div>
    </div>
    <div class="box">
      <div class="box-title">进度日志</div>
      <div id="pipeline-log" style="font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--t2);background:var(--card2);border:1px solid var(--border);border-radius:8px;padding:12px;max-height:400px;overflow-y:auto;white-space:pre-wrap;word-break:break-all;line-height:1.6"></div>
    </div>
  </div>

  <!-- Alphas -->
  <div class="panel" id="panel-alphas">
    <div class="box" style="margin-bottom:16px">
      <div class="box-title" style="margin-bottom:16px">Alpha 因子状态汇总</div>
      <div class="alpha-status-row">
        <div class="alpha-chip"><div class="alpha-chip-val c-blue" id="as-gen">–</div><div class="alpha-chip-label">已生成</div></div>
        <div class="alpha-chip"><div class="alpha-chip-val c-amber" id="as-sim">–</div><div class="alpha-chip-label">已仿真</div></div>
        <div class="alpha-chip"><div class="alpha-chip-val c-green" id="as-pass">–</div><div class="alpha-chip-label">通过闸门</div></div>
        <div class="alpha-chip"><div class="alpha-chip-val c-red" id="as-fail">–</div><div class="alpha-chip-label">未通过</div></div>
        <div class="alpha-chip"><div class="alpha-chip-val c-green" id="as-sub">–</div><div class="alpha-chip-label">已提交</div></div>
        <div class="alpha-chip"><div class="alpha-chip-val c-amber" id="as-rep">–</div><div class="alpha-chip-label">修复中</div></div>
      </div>
    </div>
    <div class="box">
      <div class="box-title">已提交因子</div>
      <div id="alphas-wrap" style="overflow-x:auto">
        <table class="data-table">
          <thead><tr><th>Alpha ID / 表达式</th><th>状态</th><th>IS Sharpe</th><th>适应度</th><th>引擎</th><th>时间</th></tr></thead>
          <tbody id="alphas-body"><tr><td colspan="6" style="color:var(--t3);font-family:inherit">加载中…</td></tr></tbody>
        </table>
      </div>
    </div>
  </div>

  <!-- Factor Library -->
  <div class="panel" id="panel-library">
    <div class="box">
      <div class="box-title" style="margin-bottom:16px">因子库 — 已通过闸门的全部因子</div>
      <div class="lib-stats" id="lib-stats"></div>
      <div class="lib-filter" id="lib-filter"></div>
      <div id="lib-list"><div class="lib-empty">加载中…</div></div>
    </div>
  </div>

  <!-- Repair Workspace -->
  <div class="panel" id="panel-repair">
    <div class="kpi-row" style="margin-bottom:16px">
      <div class="kpi"><div class="kpi-label">队列中</div><div class="kpi-val" id="rp-queue">–</div></div>
      <div class="kpi"><div class="kpi-label">正在修复</div><div class="kpi-val" id="rp-active">–</div></div>
      <div class="kpi"><div class="kpi-label">已完成</div><div class="kpi-val" id="rp-done">–</div></div>
      <div class="kpi"><div class="kpi-label">修复成功率</div><div class="kpi-val" id="rp-rate">–</div></div>
    </div>
    <div style="font-size:11px;color:var(--t3);margin-bottom:12px;display:flex;gap:16px;flex-wrap:wrap">
      <span>目标阈值：</span>
      <span id="rp-target-sharpe" style="color:var(--t2)">夏普率 ≥ 1.25</span>
      <span id="rp-target-fitness" style="color:var(--t2)">Fitness ≥ 1.0</span>
      <span id="rp-target-turnover" style="color:var(--t2)">换手率 ≤ 40%</span>
      <span id="rp-max-rounds" style="color:var(--t2)">最多修复 3 次</span>
    </div>
    <div class="box" style="margin-bottom:16px" id="rp-active-box">
      <div class="box-title">当前修复任务</div>
      <div id="rp-active-card"><div style="font-size:12px;color:var(--t3)">当前无修复任务</div></div>
    </div>
    <div class="box" style="margin-bottom:16px">
      <div class="box-title">修复队列</div>
      <div style="overflow-x:auto">
        <table class="data-table">
          <thead><tr><th>#</th><th>Alpha ID</th><th>表达式</th><th>失败检查</th><th>深度</th><th>入队时间</th></tr></thead>
          <tbody id="rp-queue-body"><tr><td colspan="6" style="color:var(--t3);font-family:inherit">加载中…</td></tr></tbody>
        </table>
      </div>
    </div>
    <div class="box">
      <div class="box-title">修复历史</div>
      <div style="overflow-x:auto">
        <table class="data-table">
          <thead><tr><th>Alpha ID</th><th>表达式</th><th>夏普率</th><th>Fitness</th><th>换手率</th><th>结果</th><th>深度</th><th>完成时间</th></tr></thead>
          <tbody id="rp-history-body"><tr><td colspan="8" style="color:var(--t3);font-family:inherit">暂无记录</td></tr></tbody>
        </table>
      </div>
    </div>
  </div>

  <!-- Knowledge -->
  <div class="panel" id="panel-knowledge">
    <div class="box" style="margin-bottom:16px">
      <div class="box-title">LLM 路由器 — 全部提供商</div>
      <div id="knowledge-llm"><div style="font-size:12px;color:var(--t3)">加载中…</div></div>
    </div>
    <div class="box">
      <div class="box-title">预算明细</div>
      <div id="knowledge-budget" style="display:flex;gap:32px">
        <div><div style="font-size:10px;color:var(--t3);margin-bottom:6px;text-transform:uppercase;letter-spacing:.06em">今日花费</div><div style="font-size:28px;font-weight:700;font-family:'JetBrains Mono',monospace" id="kb-spent">$–</div></div>
        <div><div style="font-size:10px;color:var(--t3);margin-bottom:6px;text-transform:uppercase;letter-spacing:.06em">日限额</div><div style="font-size:28px;font-weight:700;font-family:'JetBrains Mono',monospace" id="kb-limit">$3.60</div></div>
        <div><div style="font-size:10px;color:var(--t3);margin-bottom:6px;text-transform:uppercase;letter-spacing:.06em">剩余</div><div style="font-size:28px;font-weight:700;font-family:'JetBrains Mono',monospace;color:var(--green)" id="kb-rem">$–</div></div>
      </div>
    </div>
    <div class="box">
      <div class="box-title">目标演化</div>
      <div style="font-size:11px;color:var(--t3);margin-bottom:10px">每次运行自动生成不同类别与字段的挖掘目标</div>
      <div id="obj-history"><div style="font-size:12px;color:var(--t3)">加载中…</div></div>
    </div>
  </div>

  <!-- Runs -->
  <div class="panel" id="panel-runs">
    <div class="box">
      <div class="box-title">近期运行</div>
      <div id="runs-wrap" style="overflow-x:auto">
        <table class="data-table">
          <thead><tr><th>运行ID</th><th>状态</th><th>引擎</th><th>合格数</th><th>最佳Sharpe</th><th>开始时间</th></tr></thead>
          <tbody id="runs-body"><tr><td colspan="6" style="color:var(--t3);font-family:inherit">加载中…</td></tr></tbody>
        </table>
      </div>
    </div>
  </div>

  <!-- Settings -->
  <div class="panel" id="panel-settings" style="max-width:520px">
    <div class="box">
      <div class="box-title" style="margin-bottom:20px">调度器设置</div>
      <div class="form-row"><div class="form-label">启用</div><select class="form-select" id="s-enabled"><option value="true">是</option><option value="false">否</option></select></div>
      <div class="form-row"><div class="form-label">模式</div><select class="form-select" id="s-mode"><option value="evaluate">evaluate</option><option value="loop">loop</option><option value="generate">generate</option></select></div>
      <div class="form-row"><div class="form-label">间隔（分钟）<span style="font-weight:400;color:var(--t3);font-size:10px">— 运行结束后固定等待1分钟继续</span></div><input class="form-input" id="s-interval" type="number" min="1"></div>
      <div class="form-row"><div class="form-label">批次大小</div><input class="form-input" id="s-batch" type="number" min="1"></div>
      <div class="form-row"><div class="form-label">轮数</div><input class="form-input" id="s-rounds" type="number" min="1"></div>
      <div class="form-row"><div class="form-label">挖掘目标</div><input class="form-input" id="s-objective" type="text"></div>
      <div style="font-size:11px;font-weight:600;color:var(--t3);text-transform:uppercase;letter-spacing:0.5px;margin:16px 0 8px">BRAIN 仿真参数</div>
      <div class="form-row"><div class="form-label">市场区域</div><select class="form-select" id="s-region"><option value="USA">USA</option><option value="EUROPE">EUROPE</option><option value="ASIA">ASIA</option></select></div>
      <div class="form-row"><div class="form-label">交易宇宙</div><select class="form-select" id="s-universe"><option value="TOP500">TOP500</option><option value="TOP1000">TOP1000</option><option value="TOP2000">TOP2000</option><option value="TOP3000">TOP3000</option></select></div>
      <div class="form-row"><div class="form-label">交易延迟</div><select class="form-select" id="s-delay"><option value="1">1（默认）</option><option value="0">0</option></select></div>
      <div class="form-row"><div class="form-label">信号衰减 (0-13)</div><input class="form-input" id="s-decay" type="number" min="0" max="13"></div>
      <div class="form-row"><div class="form-label">中性化方式</div><select class="form-select" id="s-neutralization"><option value="INDUSTRY">INDUSTRY</option><option value="NONE">NONE</option><option value="SUBINDUSTRY">SUBINDUSTRY</option><option value="SECTOR">SECTOR</option><option value="MARKET">MARKET</option></select></div>
      <div class="form-row"><div class="form-label">仓位截断 (0.01-0.1)</div><input class="form-input" id="s-truncation" type="number" min="0.01" max="0.1" step="0.01"></div>
      <div class="form-row"><div class="form-label">Pasteurization</div><select class="form-select" id="s-pasteurization"><option value="ON">ON</option><option value="OFF">OFF</option></select></div>
      <div class="form-row"><div class="form-label">Unit Handling</div><select class="form-select" id="s-unithandling"><option value="VERIFY">VERIFY</option><option value="CASH">CASH</option></select></div>
      <button class="save-btn" id="s-save">保存设置</button>
      <div style="font-size:12px;color:var(--t3);margin-top:10px;min-height:16px" id="s-status"></div>
    </div>
  </div>

</div><!-- /main -->

<script>
(function(){
  const adminToken = document.querySelector('meta[name=dashboard-token]').content;
  const storedToken = localStorage.getItem('qb_token') || '';
  const token = storedToken || adminToken;
  if (!token) { location.href = '/setup'; return; }
  const h = { 'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json' };

  fetch('/account', { headers: h }).then(r => {
    if (r.status === 401) { localStorage.removeItem('qb_token'); location.href = '/setup'; }
  }).catch(function(){});

  const userName = localStorage.getItem('qb_user') || 'Admin';
  const av = document.getElementById('sb-av');
  const un = document.getElementById('sb-uname');
  if (av) av.textContent = userName[0].toUpperCase();
  if (un) un.textContent = userName;

  function esc(s) { return String(s ?? '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
  function $(id) { return document.getElementById(id); }

  // Nav tabs
  const TITLES = { overview:'总览', pipeline:'流水线', alphas:'因子', library:'因子库', repair:'修复工作区', knowledge:'知识库', runs:'运行记录', settings:'设置' };
  document.querySelectorAll('.sb-item').forEach(function(el) {
    el.addEventListener('click', function() {
      document.querySelectorAll('.sb-item').forEach(function(x){ x.classList.remove('active'); });
      document.querySelectorAll('.panel').forEach(function(p){ p.classList.remove('active'); });
      this.classList.add('active');
      var tab = this.dataset.tab;
      var panel = document.getElementById('panel-' + tab);
      if (panel) panel.classList.add('active');
      var tt = $('topbar-title');
      if (tt) tt.textContent = TITLES[tab] || tab;
      if (tab === 'settings' && !settingsLoaded) loadSettings();
      if (tab === 'library' && !libLoaded) loadLibrary();
    });
  });

  // Factor Library
  var libLoaded = false;
  var libData = [];
  var libFilter = 'ALL';
  var CAT_COLORS = { QUALITY:'c-green', MOMENTUM:'c-blue', REVERSAL:'c-purple', LIQUIDITY:'c-amber', VOLATILITY:'c-purple', MICROSTRUCTURE:'c-green', SENTIMENT:'c-red' };
  function renderLibrary() {
    var filtered = libFilter === 'ALL' ? libData : libData.filter(function(e){ return e.category === libFilter; });
    var list = $('lib-list');
    if (!list) return;
    if (!filtered.length) {
      list.innerHTML = '<div class="lib-empty">暂无因子。运行挖掘并通过闸门后将显示在这里。</div>';
      return;
    }
    list.innerHTML = filtered.map(function(e, idx) {
      var badge = 'lib-badge-' + (e.category || 'UNKNOWN');
      var origin = (e.originRefs || []).indexOf('llm') >= 0 ? 'AI' : '模板';
      var sharpe = e.sharpe !== null ? e.sharpe.toFixed(3) : '–';
      var fitness = e.fitness !== null ? e.fitness.toFixed(3) : '–';
      var turnover = e.turnover !== null ? (e.turnover * 100).toFixed(1) + '%' : '–';
      var sharpeColor = e.sharpe !== null ? (e.sharpe >= 1 ? 'c-green' : e.sharpe >= 0.5 ? 'c-amber' : 'c-red') : '';
      var hyp = e.hypothesis || '（无假设记录）';
      var degradedTag = e.degraded ? ' <span style="font-size:9px;color:var(--amber);margin-left:6px">降级模式</span>' : '';
      return '<div class="lib-card">' +
        '<div class="lib-card-header">' +
          '<span class="lib-badge ' + badge + '">' + (e.category || 'UNKNOWN') + '</span>' +
          '<span class="lib-origin">' + origin + '</span>' +
          degradedTag +
          '<span class="lib-alphaid">' + e.alphaId + '</span>' +
        '</div>' +
        '<div class="lib-hypothesis">' + hyp + '</div>' +
        '<div class="lib-expr" id="libexpr-' + idx + '" onclick="this.classList.toggle(&apos;expanded&apos;)" title="点击展开/折叠">' + e.expression + '</div>' +
        '<div class="lib-metrics">' +
          '<div class="lib-metric"><div class="lib-metric-val ' + sharpeColor + '">' + sharpe + '</div><div class="lib-metric-lbl">IS Sharpe</div></div>' +
          '<div class="lib-metric"><div class="lib-metric-val">' + fitness + '</div><div class="lib-metric-lbl">Fitness</div></div>' +
          '<div class="lib-metric"><div class="lib-metric-val">' + turnover + '</div><div class="lib-metric-lbl">换手率</div></div>' +
          (e.qualifiedAt ? '<div class="lib-metric" style="margin-left:auto"><div class="lib-metric-val" style="font-size:10px">' + (e.qualifiedAt || '').replace('T',' ').slice(0,16) + '</div><div class="lib-metric-lbl">挖掘时间</div></div>' : '') +
        '</div>' +
      '</div>';
    }).join('');
  }
  function renderLibStats(data) {
    var stats = $('lib-stats');
    if (!stats) return;
    var cats = {};
    data.entries.forEach(function(e){ cats[e.category] = (cats[e.category]||0) + 1; });
    var catKeys = Object.keys(cats).sort();
    var avgSharpe = data.entries.filter(function(e){ return e.sharpe !== null; });
    var avg = avgSharpe.length ? (avgSharpe.reduce(function(s,e){ return s + e.sharpe; }, 0) / avgSharpe.length).toFixed(3) : '–';
    stats.innerHTML =
      '<div class="lib-stat"><div class="lib-stat-val c-blue">' + data.total + '</div><div class="lib-stat-lbl">因子总数</div></div>' +
      '<div class="lib-stat"><div class="lib-stat-val c-green">' + avg + '</div><div class="lib-stat-lbl">平均Sharpe</div></div>' +
      catKeys.map(function(c){ return '<div class="lib-stat"><div class="lib-stat-val">' + cats[c] + '</div><div class="lib-stat-lbl">' + c + '</div></div>'; }).join('');
  }
  function renderLibFilter(data) {
    var f = $('lib-filter');
    if (!f) return;
    var cats = {};
    data.entries.forEach(function(e){ cats[e.category] = (cats[e.category]||0) + 1; });
    var catKeys = Object.keys(cats).sort();
    f.innerHTML = '<span class="lib-pill' + (libFilter==='ALL'?' active':'') + '" onclick="setLibFilter(&apos;ALL&apos;)">全部 (' + data.total + ')</span>' +
      catKeys.map(function(c){ return '<span class="lib-pill' + (libFilter===c?' active':'') + '" onclick="setLibFilter(&apos;' + c + '&apos;)">' + c + ' (' + cats[c] + ')</span>'; }).join('');
  }
  window.setLibFilter = function(cat) {
    libFilter = cat;
    renderLibFilter({ entries: libData, total: libData.length });
    renderLibrary();
  };
  async function loadLibrary() {
    libLoaded = true;
    try {
      var r = await fetch('/alpha-library', { headers: h });
      var data = await r.json();
      libData = data.entries || [];
      renderLibStats(data);
      renderLibFilter(data);
      renderLibrary();
    } catch(e) {
      var list = $('lib-list');
      if (list) list.innerHTML = '<div class="lib-empty">加载失败: ' + e.message + '</div>';
    }
  }

  // Countdown
  var nextRunAt = null;
  function tickCd() {
    if (!nextRunAt) return;
    var ms = new Date(nextRunAt) - Date.now();
    var fmt = ms > 0
      ? String(Math.floor(ms/60000)).padStart(2,'0') + ':' + String(Math.floor((ms%60000)/1000)).padStart(2,'0')
      : '00:00';
    var el = $('cd-time');
    var el2 = $('topbar-cd');
    if (el) el.textContent = fmt;
    if (el2) el2.textContent = fmt;
  }
  setInterval(tickCd, 1000);

  async function fetchSched() {
    try {
      var r = await fetch('/scheduler', { headers: h });
      var d = await r.json();
      nextRunAt = d.nextRunAt || null;
      var sl = $('sched-label');
      var cs = $('cd-sub');
      if (sl) sl.textContent = d.enabled ? (d.mode + ' \u00b7 ' + d.engine) : '调度器已关闭';
      var subText = (d.mode || '') + ' \u00b7 ' + (d.engine || '') + ' \u00b7 批次 ' + (d.batchSize || '\u2013');
      if (d.lastSkipReason) subText += ' \u00b7 \u26a0\ufe0f ' + d.lastSkipReason.slice(0, 60);
      if (cs) cs.textContent = subText;
    } catch(e) {}
  }
  fetchSched();
  setInterval(fetchSched, 30000);

  // Update mining status banner
  function updateMiningBanner(activeRun) {
    var banner = $('mining-banner');
    var ind = $('mining-ind');
    var lbl = $('mining-label');
    var task = $('mining-task');
    var elapsed = $('mining-elapsed');
    var ctDot = $('ct-dot');
    var ctText = $('ct-text');

    if (activeRun) {
      if (banner) { banner.className = 'mining-banner active-mining'; }
      if (ind) { ind.className = 'mining-indicator active-ind'; }
      if (lbl) { lbl.className = 'mining-label active-lbl'; lbl.textContent = '挖掘中'; }
      var engine = activeRun.engine || 'legacy-js';
      var status = activeRun.status || 'running';
      var currentLlm = activeRun.currentLlm || activeRun.llmProvider || engine;
      var logs = (activeRun.logs || []);
      var latestLog = logs.length ? (logs[logs.length-1].line || '') : '';
      var taskStr = latestLog ? latestLog.slice(0,90) : (currentLlm + ' \u00b7 ' + status);
      if (task) task.textContent = taskStr;
      if (elapsed && activeRun.startedAt) {
        var sec = Math.floor((Date.now() - new Date(activeRun.startedAt)) / 1000);
        var em = Math.floor(sec/60);
        var es = sec % 60;
        elapsed.textContent = em + ':' + String(es).padStart(2,'0') + ' 已运行';
      }
      if (ctDot) { ctDot.className = 'ct-dot running'; }
      if (ctText) { ctText.textContent = taskStr || '运行中'; }
    } else {
      if (banner) { banner.className = 'mining-banner idle-mining'; }
      if (ind) { ind.className = 'mining-indicator idle-ind'; }
      if (lbl) { lbl.className = 'mining-label idle-lbl'; lbl.textContent = '系统空闲'; }
      if (task) task.textContent = '当前无挖掘任务';
      if (elapsed) elapsed.textContent = '';
      if (ctDot) { ctDot.className = 'ct-dot idle'; }
      if (ctText) { ctText.textContent = '空闲'; }
    }
  }

  // Main poll
  async function poll() {
    try {
      var r = await fetch('/runs', { headers: h });
      var d = await r.json();
      var al = d.autoLoop || {};
      var lr = d.llmRouterState || null;
      var recent = d.recent || [];
      var active = d.active || [];

      // Mining banner
      updateMiningBanner(active[0] || null);

      // KPIs
      var ks = $('k-submitted');
      if (ks) ks.textContent = al.submitted ? al.submitted.length : 0;
      // Bug fix: sum.passedGate never exists; use qualified_alphas_count (Python) as proxy
      var passed = recent.filter(function(x){
        var sum = x.summary || {};
        return sum.qualified_alphas_count > 0 || sum.passedGate;
      }).length;
      var kg = $('k-gatepass');
      if (kg) kg.textContent = recent.length ? Math.round(passed/recent.length*100) + '%' : '\u2013';
      var kr = $('k-repair');
      if (kr) kr.textContent = al.queueLength != null ? al.queueLength : (al.queue ? al.queue.length : 0);
      var ksp = $('k-spend');
      if (ksp && lr) ksp.textContent = '$' + (lr.spent_usd ?? 0).toFixed(2);
      var ks4 = $('k-sub4');
      if (ks4 && lr) ks4.textContent = '上限 $' + (lr.daily_budget_usd ?? 3.6).toFixed(2);

      // Pipeline stages — parse from active run logs
      // Note: [gate-pass]/[gate-fail] are never emitted; python-v2 stdout has no stage lines
      // For python-v2 active runs: read stage counts from summary (for completed part)
      var activeRun = active[0] || null;
      var runLogs = (activeRun && activeRun.logs) ? activeRun.logs : [];
      var nGen = 0, nSim = 0, nVal = 0, nGate = 0, nSub = 0;
      var lastLogLine = '';
      runLogs.forEach(function(l) {
        var line = l.line || '';
        if (line) lastLogLine = line;
        // legacy-js log prefixes
        if (line.indexOf('[submit]') === 0) nGen++;
        else if (line.indexOf('[simulation]') === 0) nSim++;
        else if (line.indexOf('[complete]') === 0) nVal++;
        else if (line.indexOf('[gate-pass]') === 0 || line.indexOf('[passed]') === 0) { nGate++; nSub++; }
        else if (line.indexOf('[gate-fail]') === 0 || line.indexOf('[failed]') === 0) nGate++;
      });
      // For python-v2 active run: use live progressStats injected by server
      if (activeRun && activeRun.progressStats) {
        var ps = activeRun.progressStats;
        nGen = ps.nGen || 0;
        nSim = ps.nSim || 0;
        nGate = ps.nGate || 0;
      }
      // Fallback: use last completed run's summary when no active-run data available
      if (!nGen && recent.length) {
        var latestSum = recent[0] ? (recent[0].summary || {}) : {};
        if (latestSum.generatedCandidates) nGen = latestSum.generatedCandidates;
        if (latestSum.total_brain_simulations) nSim = latestSum.total_brain_simulations;
        if (latestSum.validCandidates) nVal = latestSum.validCandidates;
        if (latestSum.qualified_alphas_count != null) { nGate = latestSum.total_brain_simulations || 0; nSub = latestSum.qualified_alphas_count; }
      }
      var stageMap = { gen: nGen, val: nVal, sim: nSim, gate: nGate || '\u2013', sub: nSub };
      Object.keys(stageMap).forEach(function(s) {
        var el = $('ps-' + s);
        if (el) el.textContent = stageMap[s];
      });
      // Update current task strip with latest log line
      var ctDot2 = $('ct-dot'); var ctText2 = $('ct-text');
      if (activeRun && lastLogLine) {
        if (ctDot2) ctDot2.className = 'ct-dot running';
        if (ctText2) ctText2.textContent = lastLogLine.length > 80 ? lastLogLine.slice(0,80) + '\u2026' : lastLogLine;
      } else {
        if (ctDot2) ctDot2.className = 'ct-dot idle';
        if (ctText2) ctText2.textContent = '空闲';
      }

      // LLM rows
      var ll = $('llm-list');
      if (ll && lr && lr.providers) {
        // Flatten nested {name:{role:{...provider}}} into flat list
        var flatProviders = [];
        Object.entries(lr.providers).forEach(function(kv) {
          var n = kv[0]; var roleMap = kv[1];
          if (roleMap && typeof roleMap === 'object' && !('win_rate' in roleMap)) {
            // nested: {role: providerObj}
            Object.values(roleMap).forEach(function(p) { flatProviders.push(Object.assign({}, p, {name: n})); });
          } else {
            flatProviders.push(Object.assign({}, roleMap, {name: n}));
          }
        });
        var entries5 = flatProviders.slice(0, 5);
        var activeLlm = activeRun ? (activeRun.currentLlm || activeRun.llmProvider || '') : '';
        ll.innerHTML = entries5.map(function(p) {
          var n = p.name || '';
          var wr = ((p.win_rate ?? 0.5) * 100).toFixed(0);
          var col = (p.win_rate ?? 0.5) >= 0.5 ? 'var(--green)' : 'var(--amber)';
          var activeBadge = (activeLlm && n.toLowerCase().indexOf(activeLlm.toLowerCase()) >= 0) ? '<span class="llm-active-badge">ACTIVE</span>' : '';
          return '<div>' +
            '<div class="llm-row-head"><div><span class="llm-name">' + esc(n) + '</span><span class="llm-tag">' + esc(p.role || '') + '</span>' + activeBadge + '</div>' +
            '<span class="llm-rate" style="color:' + col + '">' + wr + '%</span></div>' +
            '<div class="bar-track"><div class="bar-fill" style="width:' + wr + '%;background:' + col + '"></div></div>' +
            '</div>';
        }).join('<div style="height:12px"></div>');
      }

      // Repair queue (overview panel mini list)
      var rl = $('repair-list');
      if (rl) {
        var q = (al.queue || []).slice(0,4);
        rl.innerHTML = q.length
          ? q.map(function(x){ return '<div class="queue-row"><div class="q-expr">' + esc(x.expression || '\u2013') + '</div><div class="q-tag">' + esc((x.failedChecks||[]).join(', ') || '\u2013') + '</div><div class="q-depth">D' + esc(x.repairDepth ?? 0) + '</div></div>'; }).join('')
          : '<div style="font-size:12px;color:var(--t3)">队列为空</div>';
      }

      // Repair workspace panel
      var rHistory = al.repairHistory || [];
      var rQueue = al.queue || [];
      var rActive = al.activeRepair || null;
      var rDone = rHistory.length;
      var rSuccess = rHistory.filter(function(h){return h.outcome==='submitted'||h.outcome==='target-met';}).length;
      var $rpQ = $('rp-queue'); if ($rpQ) $rpQ.textContent = rQueue.length + (rActive ? ' (+1 运行中)' : '');
      var $rpA = $('rp-active'); if ($rpA) $rpA.textContent = rActive ? '运行中' : '空闲';
      var $rpD = $('rp-done'); if ($rpD) $rpD.textContent = rDone;
      var $rpR = $('rp-rate'); if ($rpR) $rpR.textContent = rDone ? Math.round(rSuccess/rDone*100) + '%' : '–';
      // Update target labels from server config
      var rt = al.repairTargets || {};
      var $rts = $('rp-target-sharpe'); if ($rts && rt.sharpe != null) $rts.textContent = '\u590f\u666e\u7387 \u2265 ' + rt.sharpe;
      var $rtf = $('rp-target-fitness'); if ($rtf && rt.fitness != null) $rtf.textContent = 'Fitness \u2265 ' + rt.fitness;
      var $rtt = $('rp-target-turnover'); if ($rtt && rt.turnover != null) $rtt.textContent = '\u6362\u624b\u7387 \u2264 ' + Math.round(rt.turnover*100) + '%';
      var $rmr = $('rp-max-rounds'); if ($rmr && al.maxRepairRounds != null) $rmr.textContent = '\u6700\u591a\u4fee\u590d ' + al.maxRepairRounds + ' \u6b21';
      // Active repair card
      var $ac = $('rp-active-card');
      if ($ac) {
        if (rActive) {
          var acChecks = (rActive.failedChecks||[]).map(function(c){return '<span style="background:rgba(239,68,68,.15);color:var(--red);border:1px solid rgba(239,68,68,.3);padding:2px 7px;border-radius:4px;font-size:11px;font-weight:600">' + esc(c) + '</span>';}).join(' ');
          $ac.innerHTML = '<div style="display:flex;flex-direction:column;gap:10px">' +
            '<div style="display:flex;align-items:center;gap:8px"><div style="width:8px;height:8px;border-radius:50%;background:var(--green);box-shadow:0 0 6px var(--green)"></div><span style="font-size:12px;font-weight:600;color:var(--green)">正在修复</span><span style="font-size:11px;color:var(--t3)">深度 D' + (rActive.repairDepth||0) + '</span></div>' +
            '<div style="font-size:11px;color:var(--t3)">Alpha ID</div><div style="font-family:monospace;font-size:12px;color:var(--t1)">' + esc(rActive.parentAlphaId||'–') + '</div>' +
            '<div style="font-size:11px;color:var(--t3)">原始表达式</div><div style="font-family:monospace;font-size:11px;color:var(--t2);word-break:break-all;background:var(--card2);border:1px solid var(--border);border-radius:6px;padding:8px">' + esc(rActive.expression||'–') + '</div>' +
            '<div style="font-size:11px;color:var(--t3)">失败检查</div><div style="display:flex;flex-wrap:wrap;gap:6px">' + (acChecks||'<span style="color:var(--t3);font-size:12px">–</span>') + '</div>' +
            '</div>';
        } else {
          $ac.innerHTML = '<div style="font-size:12px;color:var(--t3)">当前无修复任务</div>';
        }
      }
      // Queue table
      var $rqb = $('rp-queue-body');
      if ($rqb) {
        $rqb.innerHTML = rQueue.length ? rQueue.map(function(x,i){
          var checks = (x.failedChecks||[]).map(function(c){return '<span style="background:rgba(239,68,68,.1);color:var(--red);border:1px solid rgba(239,68,68,.2);padding:1px 5px;border-radius:3px;font-size:10px">' + esc(c) + '</span>';}).join(' ');
          return '<tr><td style="color:var(--t3)">' + (i+1) + '</td>' +
            '<td style="color:var(--blue);font-family:monospace;font-size:10px">' + esc((x.parentAlphaId||'–').slice(0,16)) + '</td>' +
            '<td style="font-family:monospace;font-size:10px;max-width:260px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="' + esc(x.expression||'') + '">' + esc((x.expression||'–').slice(0,60)) + '</td>' +
            '<td>' + (checks||'<span style="color:var(--t3)">–</span>') + '</td>' +
            '<td style="color:var(--t2)">D' + (x.repairDepth||0) + '</td>' +
            '<td style="color:var(--t3)">' + esc(x.createdAt ? new Date(x.createdAt).toLocaleTimeString() : '–') + '</td></tr>';
        }).join('') : '<tr><td colspan="6" style="color:var(--t3);font-family:inherit">队列为空</td></tr>';
      }
      // History table
      var $rhb = $('rp-history-body');
      if ($rhb) {
        var outcomeLabel = {
          submitted:'<span style="color:var(--green);font-weight:600">\u2713 \u5df2\u63d0\u4ea4</span>',
          'target-met':'<span style="color:var(--green)">\u2713 \u8fbe\u5230\u76ee\u6807</span>',
          're-queued':'<span style="color:var(--amber)">\u27f3 \u518d\u6b21\u4fee\u590d</span>',
          abandoned:'<span style="color:var(--red)">\u2717 \u653e\u5f03</span>',
          'no-candidate':'<span style="color:var(--t3)">\u2014 \u65e0\u5019\u9009</span>',
          'start-failed':'<span style="color:var(--red)">\u26a0 \u542f\u52a8\u5931\u8d25</span>'
        };
        $rhb.innerHTML = rHistory.length ? rHistory.map(function(h){
          var bm = h.bestMetrics || {};
          var sharpeCell = bm.isSharpe != null ? '<span style="color:' + (bm.isSharpe >= 1.25 ? 'var(--green)' : 'var(--red)') + ';font-weight:600">' + bm.isSharpe.toFixed(2) + '</span>' : '<span style="color:var(--t3)">–</span>';
          var fitCell = bm.fitness != null ? '<span style="color:' + (bm.fitness >= 1.0 ? 'var(--green)' : 'var(--red)') + '">' + bm.fitness.toFixed(2) + '</span>' : '<span style="color:var(--t3)">–</span>';
          var toCell = bm.turnover != null ? '<span style="color:' + (bm.turnover <= 0.40 ? 'var(--green)' : 'var(--red)') + '">' + (bm.turnover*100).toFixed(1) + '%</span>' : '<span style="color:var(--t3)">–</span>';
          return '<tr>' +
            '<td style="color:var(--blue);font-family:monospace;font-size:10px">' + esc((h.alphaId||'–').slice(0,16)) + '</td>' +
            '<td style="font-family:monospace;font-size:10px;max-width:180px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="' + esc(h.expression||'') + '">' + esc((h.expression||'–').slice(0,40)) + '</td>' +
            '<td style="text-align:right">' + sharpeCell + '</td>' +
            '<td style="text-align:right">' + fitCell + '</td>' +
            '<td style="text-align:right">' + toCell + '</td>' +
            '<td>' + (outcomeLabel[h.outcome]||esc(h.outcome||'–')) + '</td>' +
            '<td style="color:var(--t2)">D' + (h.repairDepth||0) + '</td>' +
            '<td style="color:var(--t3)">' + esc(h.completedAt ? new Date(h.completedAt).toLocaleTimeString() : '–') + '</td></tr>';
        }).join('') : '<tr><td colspan="8" style="color:var(--t3);font-family:inherit">\u6682\u65e0\u8bb0\u5f55</td></tr>';
      }

      // Activity
      var af = $('act-list');
      if (af) {
        var ev = (al.events || []).slice(0, 5);
        af.innerHTML = ev.length
          ? ev.map(function(e) {
              var cls = e.type==='submitted'?'s':e.type==='repair'?'r':'f';
              var icon = e.type==='submitted'?'\u2191':e.type==='repair'?'\u27f3':'\u2715';
              return '<div class="act-row"><div class="act-dot ' + cls + '">' + icon + '</div><div style="flex:1;min-width:0"><div class="act-expr">' + esc(e.alphaId || e.expression || '\u2013') + '</div><div class="act-meta">' + esc(e.message || e.type || '') + '</div></div></div>';
            }).join('')
          : '<div style="font-size:12px;color:var(--t3)">暂无近期活动</div>';
      }

      // Pipeline panel
      var pa = $('pipeline-active');
      if (pa) {
        pa.innerHTML = active.length
          ? active.map(function(x){ return '<div style="background:var(--card2);border:1px solid var(--border);border-radius:8px;padding:12px 14px;margin-bottom:8px"><div class="jb" style="font-size:12px;color:var(--t1)">' + esc(x.runId || '\u2013') + '</div><div style="font-size:11px;color:var(--t3);margin-top:4px">' + esc(x.status || '\u2013') + ' \u00b7 ' + esc(x.engine || '\u2013') + '</div></div>'; }).join('')
          : '<div style="font-size:12px;color:var(--t3)">无活跃运行</div>';
        var pl = $('pipeline-log');
        if (pl) {
          var logLines = runLogs.slice(-60).map(function(l){ return (l.at ? l.at.slice(11,19) : '') + ' ' + (l.line || ''); });
          // For python-v2: fall back to progress.jsonl events when stdout logs are empty
          if (!logLines.length && activeRun && activeRun.progressStats && activeRun.progressStats.recentEvents) {
            logLines = activeRun.progressStats.recentEvents;
          }
          pl.textContent = logLines.join('\\n') || '暂无进度数据';
        }
      }

      // Alphas panel
      // Bug fix: use correct field names for both engines:
      // Python: generatedCandidates, total_brain_simulations, qualified_alphas_count
      // Legacy-js: no direct equivalents, infer from topCandidates
      var totalGen = 0, totalSim = 0, totalPass = 0, totalFail = 0;
      var totalRep = (al.queue ? al.queue.length : (al.queueLength || 0)) + (al.activeRepair ? 1 : 0);
      // totalSub comes from autoLoopState submissions, not summary
      var totalSub = (al.submitted || []).length;
      recent.forEach(function(x) {
        var sum = x.summary || {};
        // generatedCandidates (python) or generated (legacy) or topCandidates count
        totalGen += (sum.generatedCandidates || sum.generated ||
                     (sum.topCandidates ? sum.topCandidates.length : 0));
        // total_brain_simulations (python) or simulated (legacy)
        totalSim += (sum.total_brain_simulations || sum.simulated || 0);
        // qualified_alphas_count (python) or passedGate bool (legacy)
        var qc = sum.qualified_alphas_count;
        if (qc != null) { if (qc > 0) totalPass++; }
        else if (sum.passedGate) totalPass++;
        // rejected count: python has rejected_by_stage totals
        var rej = sum.rejected_by_stage;
        if (rej) totalFail += Object.values(rej).reduce(function(a,b){ return a + (b||0); }, 0);
      });
      var setAl = function(id, v) { var el = $(id); if (el) el.textContent = v != null ? v : '\u2013'; };
      setAl('as-gen', totalGen || '\u2013');
      setAl('as-sim', totalSim || '\u2013');
      setAl('as-pass', totalPass || '\u2013');
      setAl('as-fail', totalFail || '\u2013');
      setAl('as-sub', totalSub || '\u2013');
      setAl('as-rep', totalRep || '\u2013');

      // Bug fix: submitted alphas come from al.submitted (autoLoopState), not from summary.submitted
      var ab = $('alphas-body');
      if (ab) {
        var subs = (al.submitted || []).slice().reverse().slice(0, 20);
        ab.innerHTML = subs.length
          ? subs.map(function(sub) {
              var alpha = sub.alpha || {};
              var isSharpe = alpha.isSharpe != null ? Number(alpha.isSharpe).toFixed(3) : '\u2013';
              var iFitness = alpha.isFitness != null ? Number(alpha.isFitness).toFixed(2) : '\u2013';
              var ts = sub.at ? new Date(sub.at).toLocaleTimeString() : '\u2013';
              return '<tr><td style="color:var(--t1)">' + esc((sub.alphaId||'').slice(0,22)) + '</td>' +
                '<td><span class="badge pass">SUBMITTED</span></td>' +
                '<td>' + isSharpe + '</td><td>' + iFitness + '</td>' +
                '<td>' + esc(sub.source || '\u2013') + '</td>' +
                '<td>' + ts + '</td></tr>';
            }).join('')
          : '<tr><td colspan="6" style="color:var(--t3);font-family:inherit">暂无已提交因子</td></tr>';
      }

      // Objective history (from scheduler.recentObjectives)
      var sched = d.scheduler || {};
      var objHistory = sched.recentObjectives || [];
      var oh = $('obj-history');
      if (oh) {
        if (objHistory.length) {
          var catColors = {
            QUALITY:'#34d399',MOMENTUM:'#60a5fa',REVERSAL:'#f472b6',
            VOLATILITY:'#fbbf24',LIQUIDITY:'#a78bfa',MICROSTRUCTURE:'#fb923c',SENTIMENT:'#38bdf8'
          };
          oh.innerHTML = objHistory.slice(0,10).map(function(e) {
            var col = catColors[e.category] || 'var(--t2)';
            var ts = e.at ? new Date(e.at).toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'}) : '';
            return '<div style="padding:10px 0;border-bottom:1px solid var(--border)">' +
              '<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px">' +
              '<span style="font-size:10px;font-weight:700;color:' + col + ';text-transform:uppercase;letter-spacing:.08em;background:' + col + '22;padding:2px 8px;border-radius:4px">' + esc(e.category||'') + '</span>' +
              '<span style="font-size:10px;color:var(--t3);font-family:monospace">' + esc(e.field||'') + '</span>' +
              '<span style="font-size:10px;color:var(--t3);margin-left:auto">' + ts + '</span>' +
              '</div>' +
              '<div style="font-size:12px;color:var(--t2);line-height:1.5">' + esc(e.objective||'') + '</div>' +
              '</div>';
          }).join('');
        } else {
          oh.innerHTML = '<div style="font-size:12px;color:var(--t3)">暂无生成目标，下次定时运行后开始</div>';
        }
      }
      // Show current objective in mining banner task line
      var curObj = sched.objective || '';
      if (curObj && !activeRun) {
        var taskEl = $('mining-task');
        if (taskEl && taskEl.textContent === '当前无挖掘任务') {
          taskEl.textContent = '下一个：' + curObj.slice(0, 90) + (curObj.length > 90 ? '\u2026' : '');
        }
      }

      // Knowledge panel
      var kl = $('knowledge-llm');
      if (kl && lr && lr.providers) {
        // Flatten nested {name:{role:{...provider}}} into flat list
        var allProviders = [];
        Object.entries(lr.providers).forEach(function(kv) {
          var n = kv[0]; var roleMap = kv[1];
          if (roleMap && typeof roleMap === 'object' && !('win_rate' in roleMap)) {
            Object.values(roleMap).forEach(function(p) { allProviders.push(Object.assign({}, p, {name: n})); });
          } else {
            allProviders.push(Object.assign({}, roleMap, {name: n}));
          }
        });
        kl.innerHTML = '<table class="data-table"><thead><tr><th>提供商</th><th>角色</th><th>胜率</th><th>调用</th><th>成功</th></tr></thead><tbody>' +
          allProviders.map(function(p) {
            var wr = ((p.win_rate ?? 0)*100).toFixed(1);
            var col = (p.win_rate ?? 0) >= 0.5 ? 'var(--green)' : 'var(--amber)';
            return '<tr><td style="color:var(--t1);font-weight:500">' + esc(p.name || '\u2013') + '</td><td>' + esc(p.role || '\u2013') + '</td><td style="color:' + col + ';font-weight:600">' + wr + '%</td><td>' + (p.calls || 0) + '</td><td>' + (p.wins || 0) + '</td></tr>';
          }).join('') + '</tbody></table>';
        var sp = lr.spent_usd ?? 0;
        var bud = lr.daily_budget_usd ?? 3.6;
        var ks2 = $('kb-spent'); if (ks2) ks2.textContent = '$' + sp.toFixed(3);
        var kl2 = $('kb-limit'); if (kl2) kl2.textContent = '$' + bud.toFixed(2);
        var kr2 = $('kb-rem'); if (kr2) kr2.textContent = '$' + Math.max(0,bud-sp).toFixed(3);
      }

      // Runs panel
      // Bug fix: sum.submitted/sum.bestSharpe don't exist; derive from topCandidates + qualified_alphas_count
      // Bug fix: s.startedAt undefined for completed runs; fall back to sum.generatedAt
      var rb = $('runs-body');
      if (rb) {
        rb.innerHTML = recent.slice(0,20).map(function(x) {
          var s = x.state || {};
          var sum = x.summary || {};
          var cls = s.status==='running'?'run':s.status==='completed'?'done':'fail';
          var dt = x.runId ? x.runId.slice(0,12) : '\u2013';
          // bestSharpe: scan topCandidates for max sharpe or DSR score
          var bestSharpe = null;
          if (sum.topCandidates && sum.topCandidates.length) {
            sum.topCandidates.forEach(function(c) {
              var sh = (c.metrics && c.metrics.sharpe != null) ? c.metrics.sharpe
                      : (c.totalScore != null ? c.totalScore : null);
              if (sh != null && (bestSharpe == null || +sh > bestSharpe)) bestSharpe = +sh;
            });
          }
          // qualified count: python qualified_alphas_count, legacy n/a
          var qualCount = sum.qualified_alphas_count != null ? sum.qualified_alphas_count : '\u2013';
          // startedAt: in-memory state has it; completed disk-read uses generatedAt from summary
          var startedAt = s.startedAt || sum.generatedAt || null;
          return '<tr><td>' + esc(dt) + '</td>' +
            '<td><span class="badge ' + cls + '">' + esc(s.status || '\u2013') + '</span></td>' +
            '<td>' + esc(s.engine || '\u2013') + '</td>' +
            '<td>' + qualCount + '</td>' +
            '<td>' + (bestSharpe != null ? bestSharpe.toFixed(2) : '\u2013') + '</td>' +
            '<td>' + esc(startedAt ? new Date(startedAt).toLocaleTimeString() : '\u2013') + '</td></tr>';
        }).join('') || '<tr><td colspan="6" style="color:var(--t3);font-family:inherit">暂无运行记录</td></tr>';
      }

    } catch(e) {}
  }
  poll();
  setInterval(poll, 10000);

  // Settings
  var settingsLoaded = false;
  async function loadSettings() {
    try {
      var r = await fetch('/scheduler', { headers: h });
      var d = await r.json();
      if ($('s-enabled')) $('s-enabled').value = String(d.enabled ?? false);
      if ($('s-mode')) $('s-mode').value = d.mode || 'evaluate';
      if ($('s-interval')) $('s-interval').value = d.intervalMinutes || 60;
      if ($('s-batch')) $('s-batch').value = d.batchSize || 5;
      if ($('s-rounds')) $('s-rounds').value = d.rounds || 3;
      if ($('s-objective')) $('s-objective').value = d.objective || '';
      var ss = d.simulationSettings || {};
      if ($('s-region')) $('s-region').value = ss.region || 'USA';
      if ($('s-universe')) $('s-universe').value = ss.universe || 'TOP3000';
      if ($('s-delay')) $('s-delay').value = String(ss.delay ?? 1);
      if ($('s-decay')) $('s-decay').value = ss.decay ?? 4;
      if ($('s-neutralization')) $('s-neutralization').value = ss.neutralization || 'INDUSTRY';
      if ($('s-truncation')) $('s-truncation').value = ss.truncation ?? 0.08;
      if ($('s-pasteurization')) $('s-pasteurization').value = ss.pasteurization || 'ON';
      if ($('s-unithandling')) $('s-unithandling').value = ss.unitHandling || 'VERIFY';
      settingsLoaded = true;
    } catch(e) {}
  }
  var saveBtn = $('s-save');
  if (saveBtn) saveBtn.addEventListener('click', async function() {
    var st = $('s-status');
    this.disabled = true; this.textContent = '保存中\u2026';
    try {
      var body = {
        enabled: $('s-enabled') ? $('s-enabled').value === 'true' : false,
        mode: $('s-mode') ? $('s-mode').value : 'evaluate',
        intervalMinutes: Number($('s-interval') ? $('s-interval').value : 60),
        batchSize: Number($('s-batch') ? $('s-batch').value : 5),
        rounds: Number($('s-rounds') ? $('s-rounds').value : 3),
        objective: $('s-objective') ? ($('s-objective').value || '').trim() : '',
        simulationSettings: {
          region: $('s-region') ? $('s-region').value : 'USA',
          universe: $('s-universe') ? $('s-universe').value : 'TOP3000',
          delay: Number($('s-delay') ? $('s-delay').value : 1),
          decay: Number($('s-decay') ? $('s-decay').value : 4),
          neutralization: $('s-neutralization') ? $('s-neutralization').value : 'INDUSTRY',
          truncation: Number($('s-truncation') ? $('s-truncation').value : 0.08),
          pasteurization: $('s-pasteurization') ? $('s-pasteurization').value : 'ON',
          unitHandling: $('s-unithandling') ? $('s-unithandling').value : 'VERIFY'
        }
      };
      var r = await fetch('/scheduler', { method:'POST', headers: h, body: JSON.stringify(body) });
      if (st) st.textContent = r.ok ? '已保存 \u2713' : '错误 ' + r.status;
    } catch(e) { if (st) st.textContent = '错误：' + e.message; }
    this.disabled = false; this.textContent = '保存设置';
    setTimeout(function(){ if (st) st.textContent = ''; }, 3000);
  });

  // Run button
  var btn = $('run-btn');
  var runSt = $('run-status');
  if (btn) btn.addEventListener('click', async function() {
    var ideaEl = $('idea-field');
    var idea = ideaEl ? ideaEl.value.trim() : '';
    if (!idea) { if (runSt) runSt.textContent = '请输入想法'; return; }
    btn.disabled = true; btn.textContent = '优化中\u2026'; if (runSt) runSt.textContent = '';
    try {
      var or = await fetch('/ideas/optimize', { method:'POST', headers: h, body: JSON.stringify({ idea: idea }) });
      var od = await or.json();
      if (runSt) runSt.textContent = '方向：' + esc(od.direction || od.target_family || '');
      btn.textContent = '启动中\u2026';
      var rr = await fetch('/runs', { method:'POST', headers: h, body: JSON.stringify({ force: false }) });
      if (rr.status === 409) { if (runSt) runSt.textContent = '已有运行在进行中'; btn.disabled=false; btn.textContent='优化并运行'; return; }
      btn.textContent = '运行已启动 \u2713';
      setTimeout(function(){ btn.disabled=false; btn.textContent='优化并运行'; }, 3000);
    } catch(e) { if (runSt) runSt.textContent='错误：'+e.message; btn.disabled=false; btn.textContent='优化并运行'; }
  });
})();
</script>
</body>
</html>
`;
}
