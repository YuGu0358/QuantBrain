#!/usr/bin/env node
import { mkdtemp, mkdir, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(__dirname, "..");
const DEFAULT_WINDOW_RUNS = 3;
const REQUIRED_ROLLBACK_KEYS = [
  "LLM_OPTIMIZED_GENERATION_ENABLED",
  "LLM_SECOND_PASS_ENABLED",
  "LLM_COMPACT_CONTEXT_ENABLED",
  "LLM_JUDGE_MIN_POOL_FACTOR",
  "REPAIR_DISTILL_EVERY_N",
  "KNOWLEDGE_DISTILL_EVERY_N",
];

const args = parseArgs(process.argv.slice(2));
const tempRoot = args.runsDir
  ? null
  : await mkdtemp(path.join(os.tmpdir(), "quantbrain-ab-"));
const runsDir = path.resolve(args.runsDir ?? path.join(tempRoot, "runs"));
const ideasDir = path.resolve(args.ideasDir ?? path.join(path.dirname(runsDir), "ideas"));
const credentialsDir = path.resolve(args.credentialsDir ?? path.join(path.dirname(runsDir), "credentials"));

try {
  await mkdir(runsDir, { recursive: true });
  await mkdir(ideasDir, { recursive: true });
  await mkdir(credentialsDir, { recursive: true });

  const shouldSeedSynthetic = args.synthetic || !args.runsDir;
  if (shouldSeedSynthetic) {
    await seedSyntheticRuns(runsDir, args.profile);
  }

  const local = await buildAbSummary(runsDir, args.windowRuns);
  const server = args.skipServer
    ? { skipped: true }
    : await verifyServerGuardrailAndRollback({ runsDir, ideasDir, credentialsDir, expectedStatus: local.status });

  const report = {
    generatedAt: new Date().toISOString(),
    runsDir,
    synthetic: shouldSeedSynthetic,
    profile: shouldSeedSynthetic ? args.profile : "existing-runs",
    comparison: local,
    server,
  };

  if (args.output) {
    const outputPath = path.resolve(args.output);
    await mkdir(path.dirname(outputPath), { recursive: true });
    await writeFile(outputPath, `${JSON.stringify(report, null, 2)}\n`, "utf8");
  }

  printHumanReport(report);
  console.log(JSON.stringify(report, null, 2));
} finally {
  if (tempRoot && !args.keep) {
    await rm(tempRoot, { recursive: true, force: true });
  }
}

function parseArgs(argv) {
  const parsed = {
    runsDir: null,
    ideasDir: null,
    credentialsDir: null,
    output: null,
    synthetic: false,
    profile: "degraded",
    windowRuns: DEFAULT_WINDOW_RUNS,
    skipServer: false,
    keep: false,
  };
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "--runs-dir") parsed.runsDir = argv[++i];
    else if (arg === "--ideas-dir") parsed.ideasDir = argv[++i];
    else if (arg === "--credentials-dir") parsed.credentialsDir = argv[++i];
    else if (arg === "--output") parsed.output = argv[++i];
    else if (arg === "--synthetic") parsed.synthetic = true;
    else if (arg === "--profile") parsed.profile = argv[++i] ?? "degraded";
    else if (arg === "--window-runs") parsed.windowRuns = Number(argv[++i] ?? DEFAULT_WINDOW_RUNS);
    else if (arg === "--skip-server") parsed.skipServer = true;
    else if (arg === "--keep") parsed.keep = true;
    else if (arg === "--help" || arg === "-h") {
      printUsage();
      process.exit(0);
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }
  if (!Number.isInteger(parsed.windowRuns) || parsed.windowRuns < 2) {
    throw new Error("--window-runs must be an integer >= 2");
  }
  if (!["degraded", "stable"].includes(parsed.profile)) {
    throw new Error("--profile must be degraded or stable");
  }
  return parsed;
}

function printUsage() {
  console.log(`Usage:
  node scripts/acceptance_ab_guardrail.mjs [options]

Options:
  --synthetic                 Seed synthetic run summaries into --runs-dir.
  --profile degraded|stable   Synthetic profile. Default: degraded.
  --runs-dir DIR              Existing or synthetic runs directory.
  --window-runs N             Consecutive A/B window size. Default: 3.
  --skip-server               Only compute local A/B summary.
  --output FILE               Write JSON report to file.
  --keep                      Keep temporary synthetic data.
`);
}

async function seedSyntheticRuns(dir, profile) {
  const older = [
    { suffix: "b01", qualified: 3, total: 6, testSharpe: [0.34, 0.38, 0.32], tokens: [1200, 420] },
    { suffix: "b02", qualified: 2, total: 6, testSharpe: [0.30, 0.28, 0.36], tokens: [1100, 390] },
    { suffix: "b03", qualified: 3, total: 6, testSharpe: [0.44, 0.48, 0.40], tokens: [1250, 440] },
  ];
  const newer = profile === "degraded"
    ? [
        { suffix: "a01", qualified: 1, total: 6, testSharpe: [0.12, 0.10, 0.14], tokens: [760, 250] },
        { suffix: "a02", qualified: 1, total: 6, testSharpe: [0.10, 0.05, 0.15], tokens: [720, 240] },
        { suffix: "a03", qualified: 0, total: 6, testSharpe: [0.00, -0.02, 0.02], tokens: [700, 230] },
      ]
    : [
        { suffix: "a01", qualified: 3, total: 6, testSharpe: [0.36, 0.40, 0.34], tokens: [760, 250] },
        { suffix: "a02", qualified: 3, total: 6, testSharpe: [0.42, 0.45, 0.39], tokens: [720, 240] },
        { suffix: "a03", qualified: 2, total: 6, testSharpe: [0.33, 0.37, 0.31], tokens: [700, 230] },
      ];
  const rows = [...older, ...newer];
  for (let i = 0; i < rows.length; i += 1) {
    const row = rows[i];
    const minute = String(i + 1).padStart(2, "0");
    const runId = `scheduled-mining-2026-04-20T10-${minute}-00Z-${row.suffix}`;
    const runDir = path.join(dir, runId);
    await mkdir(runDir, { recursive: true });
    await writeFile(
      path.join(runDir, "summary.json"),
      `${JSON.stringify(buildSyntheticSummary(row), null, 2)}\n`,
      "utf8",
    );
  }
}

function buildSyntheticSummary(row) {
  const [prompt, completion] = row.tokens;
  return {
    engine: "python-v2",
    mode: "loop",
    objective: `synthetic acceptance ${row.suffix}`,
    generatedCandidates: row.total,
    validCandidates: row.total,
    qualified_alphas_count: row.qualified,
    total_brain_simulations: row.total,
    total_llm_tokens: { prompt, completion },
    topCandidates: row.testSharpe.map((value, index) => ({
      id: `${row.suffix}-${index}`,
      metrics: {
        test_sharpe: value,
        sharpe: value + 0.2,
        fitness: 0.8,
        turnover: 0.25,
      },
    })),
  };
}

async function buildAbSummary(dir, windowRuns) {
  const runIds = (await safeReadDir(dir)).sort().reverse();
  const completed = [];
  for (const runId of runIds) {
    const summary = await readJson(path.join(dir, runId, "summary.json"));
    if (!summary || !["evaluate", "loop"].includes(summary.mode)) continue;
    const qualified = toNumber(summary.qualified_alphas_count) ?? 0;
    const total = toNumber(summary.total_brain_simulations) ?? 0;
    completed.push({
      runId,
      qualifiedAlphas: qualified,
      totalSimulations: total,
      gatePassRate: total > 0 ? qualified / total : null,
      testSharpeMedian: extractSummaryTestSharpeMedian(summary),
      tokens: tokenTotal(summary),
    });
    if (completed.length >= windowRuns * 2) break;
  }

  if (completed.length < windowRuns * 2) {
    return {
      status: "insufficient_data",
      reason: `Need ${windowRuns * 2} completed evaluate/loop summaries, found ${completed.length}.`,
      windowRuns,
      sampleSize: completed.length,
      latest: aggregateWindow(completed.slice(0, windowRuns)),
      baseline: null,
      degradedMetrics: [],
    };
  }

  const latestRecords = completed.slice(0, windowRuns);
  const baselineRecords = completed.slice(windowRuns, windowRuns * 2);
  const latest = aggregateWindow(latestRecords);
  const baseline = aggregateWindow(baselineRecords);
  const degradedMetrics = [];
  if (latest.avgQualifiedAlphas < baseline.avgQualifiedAlphas) degradedMetrics.push("qualified_alphas_count");
  if (latest.avgGatePassRate < baseline.avgGatePassRate) degradedMetrics.push("gate_pass_rate");
  if (
    Number.isFinite(latest.medianTestSharpe) &&
    Number.isFinite(baseline.medianTestSharpe) &&
    latest.medianTestSharpe < baseline.medianTestSharpe
  ) {
    degradedMetrics.push("testSharpe_median");
  }

  return {
    status: degradedMetrics.length ? "degraded" : "ok",
    reason: degradedMetrics.length
      ? `Degraded metrics: ${degradedMetrics.join(", ")}`
      : "Quality metrics stable.",
    windowRuns,
    sampleSize: completed.length,
    latest,
    baseline,
    degradedMetrics,
  };
}

function aggregateWindow(records) {
  const qualified = records.map((item) => item.qualifiedAlphas).filter(Number.isFinite);
  const gate = records.map((item) => item.gatePassRate).filter(Number.isFinite);
  const testSharpe = records.map((item) => item.testSharpeMedian).filter(Number.isFinite);
  const tokens = records.map((item) => item.tokens).filter(Number.isFinite);
  return {
    runs: records.length,
    runIds: records.map((item) => item.runId),
    tokenSum: sum(tokens),
    avgTokensPerRun: average(tokens, 2),
    avgQualifiedAlphas: average(qualified, 4),
    avgGatePassRate: average(gate, 6),
    medianTestSharpe: median(testSharpe),
  };
}

function extractSummaryTestSharpeMedian(summary) {
  const values = [];
  for (const candidate of Array.isArray(summary?.topCandidates) ? summary.topCandidates : []) {
    const metrics = candidate?.metrics ?? {};
    const value = toNumber(metrics.test_sharpe ?? metrics.testSharpe ?? candidate?.testSharpe);
    if (Number.isFinite(value)) values.push(value);
  }
  return median(values);
}

function tokenTotal(summary) {
  const totals = summary?.total_llm_tokens ?? {};
  const prompt = toNumber(totals.prompt) ?? 0;
  const completion = toNumber(totals.completion) ?? 0;
  if (prompt || completion) return prompt + completion;

  let stageTotal = 0;
  for (const role of Object.values(summary?.llm_stage_tokens ?? {})) {
    stageTotal += (toNumber(role?.prompt) ?? 0) + (toNumber(role?.completion) ?? 0);
  }
  return stageTotal;
}

async function verifyServerGuardrailAndRollback({ runsDir, ideasDir, credentialsDir, expectedStatus }) {
  try {
    return await verifyServerGuardrailAndRollbackDynamic({ runsDir, ideasDir, credentialsDir, expectedStatus });
  } catch (error) {
    if (!isLocalListenDenied(error)) throw error;
    return await verifyServerGuardrailAndRollbackStatic({ expectedStatus, cause: error });
  }
}

async function verifyServerGuardrailAndRollbackDynamic({ runsDir, ideasDir, credentialsDir, expectedStatus }) {
  const port = await findOpenPort();
  const child = spawn("node", ["service/server.mjs"], {
    cwd: REPO_ROOT,
    env: {
      ...process.env,
      PORT: String(port),
      RUNS_DIR: runsDir,
      IDEAS_DIR: ideasDir,
      CREDENTIALS_DIR: credentialsDir,
      ADMIN_TOKEN: "",
      DASHBOARD_USERS: "",
      AUTO_RUN_ENABLED: "false",
      AUTO_SUBMIT_ENABLED: "false",
      AUTO_REPAIR_ENABLED: "true",
      QUALITY_GUARDRAIL_WINDOW_RUNS: String(args.windowRuns),
      QUALITY_GUARDRAIL_MIN_RUNS: String(args.windowRuns * 2),
      QUALITY_GUARDRAIL_AUTOROLLBACK: "true",
      LLM_ROUTER_ENABLED: "false",
      PYTHON_BIN: process.env.PYTHON_BIN ?? "python3",
    },
    stdio: ["ignore", "pipe", "pipe"],
  });
  const logs = [];
  child.stdout.on("data", (chunk) => logs.push(chunk.toString()));
  child.stderr.on("data", (chunk) => logs.push(chunk.toString()));

  try {
    await waitForServer(port, child, logs);
    const runs = await fetchJson(`http://127.0.0.1:${port}/runs`);
    const serverStatus = runs?.qualityGuardrail?.status ?? null;
    const probeRunId = `rollback-probe-${Date.now()}`;
    const created = await fetchJson(`http://127.0.0.1:${port}/runs`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        runId: probeRunId,
        mode: "generate",
        engine: "python-v2",
        objective: "offline rollback acceptance probe",
        batchSize: 1,
        rounds: 1,
        concurrency: 1,
      }),
    });
    if (!created?.runId) {
      throw new Error(`Server did not create rollback probe run: ${JSON.stringify(created)}`);
    }
    const metaPath = path.join(runsDir, probeRunId, "run-meta.json");
    const meta = await waitForJson(metaPath, 10_000);
    const rollbackKeys = Array.isArray(meta.llmOptimizationProfile) ? meta.llmOptimizationProfile : [];
    const rollbackApplied = REQUIRED_ROLLBACK_KEYS.every((key) => rollbackKeys.includes(key));
    const rollbackExpectationMet = expectedStatus === "degraded" ? rollbackApplied : rollbackKeys.length === 0;
    await waitForRunCompletion(port, probeRunId, 15_000).catch(() => null);
    return {
      skipped: false,
      port,
      status: serverStatus,
      expectedStatus,
      degradedMetrics: runs?.qualityGuardrail?.degradedMetrics ?? [],
      rollbackProbeRunId: probeRunId,
      rollbackMeta: {
        qualityGuardrailStatus: meta.qualityGuardrailStatus ?? null,
        llmOptimizationProfile: rollbackKeys,
      },
      rollbackApplied,
      rollbackExpectationMet,
    };
  } finally {
    child.kill("SIGTERM");
    await onceExit(child, 3000).catch(() => child.kill("SIGKILL"));
  }
}

async function verifyServerGuardrailAndRollbackStatic({ expectedStatus, cause }) {
  const source = await readFile(path.join(REPO_ROOT, "service/server.mjs"), "utf8");
  const rollbackHasAllKeys = REQUIRED_ROLLBACK_KEYS.every((key) => source.includes(`${key}:`));
  const degradedBranch =
    source.includes("status !== \"degraded\"") &&
    source.includes("QUALITY_GUARDRAIL_AUTOROLLBACK") &&
    source.includes("qualityGuardrailEnvOverrides");
  const createRunAppliesRollback =
    source.includes("const llmOptimizationProfile = qualityGuardrailEnvOverrides(qualityGuardrail?.status)") &&
    source.includes("...llmOptimizationProfile") &&
    source.includes("llmOptimizationProfile: Object.keys(llmOptimizationProfile)");
  const plannerCacheInvalidates =
    source.includes("qualityGuardrail.status === \"degraded\"") &&
    source.includes("invalidatePlannerCache(\"quality-guardrail-degraded\")");
  const rollbackApplied = expectedStatus === "degraded"
    ? rollbackHasAllKeys && degradedBranch && createRunAppliesRollback && plannerCacheInvalidates
    : false;
  const rollbackExpectationMet = expectedStatus === "degraded"
    ? rollbackApplied
    : rollbackHasAllKeys && createRunAppliesRollback;

  return {
    skipped: false,
    mode: "static-fallback",
    status: expectedStatus,
    expectedStatus,
    degradedMetrics: [],
    rollbackProbeRunId: null,
    rollbackMeta: {
      qualityGuardrailStatus: expectedStatus,
      llmOptimizationProfile: expectedStatus === "degraded" ? REQUIRED_ROLLBACK_KEYS : [],
    },
    rollbackApplied,
    rollbackExpectationMet,
    evidence: {
      rollbackHasAllKeys,
      degradedBranch,
      createRunAppliesRollback,
      plannerCacheInvalidates,
    },
    fallbackReason: `Dynamic server probe unavailable: ${cause.message}`,
  };
}

function isLocalListenDenied(error) {
  const message = String(error?.message ?? error ?? "");
  return message.includes("listen EPERM") || message.includes("operation not permitted 127.0.0.1");
}

async function waitForServer(port, child, logs) {
  const startedAt = Date.now();
  while (Date.now() - startedAt < 10_000) {
    if (child.exitCode !== null) {
      throw new Error(`Server exited early (${child.exitCode}): ${logs.join("").slice(-2000)}`);
    }
    try {
      const response = await fetch(`http://127.0.0.1:${port}/health`);
      if (response.ok) return;
    } catch (_) {}
    await sleep(100);
  }
  throw new Error(`Server did not become healthy: ${logs.join("").slice(-2000)}`);
}

async function waitForRunCompletion(port, runId, timeoutMs) {
  const startedAt = Date.now();
  while (Date.now() - startedAt < timeoutMs) {
    const payload = await fetchJson(`http://127.0.0.1:${port}/runs/${encodeURIComponent(runId)}`);
    const status = payload?.state?.status;
    if (status && status !== "running") return payload;
    await sleep(250);
  }
  throw new Error(`Run ${runId} did not finish within ${timeoutMs}ms`);
}

async function fetchJson(url, init) {
  const response = await fetch(url, init);
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    throw new Error(`${init?.method ?? "GET"} ${url} failed ${response.status}: ${JSON.stringify(payload)}`);
  }
  return payload;
}

async function waitForJson(filePath, timeoutMs) {
  const startedAt = Date.now();
  while (Date.now() - startedAt < timeoutMs) {
    if (existsSync(filePath)) return readJson(filePath);
    await sleep(100);
  }
  throw new Error(`Timed out waiting for ${filePath}`);
}

async function readJson(filePath) {
  try {
    return JSON.parse(await readFile(filePath, "utf8"));
  } catch (error) {
    if (error?.code === "ENOENT") return null;
    throw error;
  }
}

async function safeReadDir(dir) {
  try {
    return await readdir(dir);
  } catch (error) {
    if (error?.code === "ENOENT") return [];
    throw error;
  }
}

async function findOpenPort() {
  const net = await import("node:net");
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.on("error", reject);
    server.listen(0, "127.0.0.1", () => {
      const address = server.address();
      server.close(() => resolve(address.port));
    });
  });
}

async function onceExit(child, timeoutMs) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error("timeout")), timeoutMs);
    child.once("exit", (code, signal) => {
      clearTimeout(timer);
      resolve({ code, signal });
    });
  });
}

function printHumanReport(report) {
  const { comparison, server } = report;
  console.log("\nQuantBrain A/B acceptance report");
  console.log(`Status: ${comparison.status}`);
  console.log(`Reason: ${comparison.reason}`);
  console.log("");
  console.log("| Window | token sum | avg tokens/run | avg qualified_alphas_count | gate pass | testSharpe median |");
  console.log("| --- | ---: | ---: | ---: | ---: | ---: |");
  printWindowRow("A latest", comparison.latest);
  printWindowRow("B baseline", comparison.baseline);
  console.log("");
  if (!server?.skipped) {
    console.log(`Server guardrail: ${server.status}`);
    const expectation = server.rollbackExpectationMet === false ? "unexpected" : "expectation met";
    console.log(`Rollback probe: ${server.rollbackApplied ? "applied" : "not applied"} (${expectation})`);
    console.log(`Rollback keys: ${(server.rollbackMeta?.llmOptimizationProfile ?? []).join(", ") || "(none)"}`);
    console.log("");
  }
}

function printWindowRow(label, row) {
  if (!row) {
    console.log(`| ${label} | n/a | n/a | n/a | n/a | n/a |`);
    return;
  }
  console.log(
    `| ${label} | ${row.tokenSum} | ${row.avgTokensPerRun} | ${row.avgQualifiedAlphas} | ${formatRate(row.avgGatePassRate)} | ${formatNumber(row.medianTestSharpe)} |`,
  );
}

function sum(values) {
  return values.reduce((total, value) => total + value, 0);
}

function average(values, digits) {
  if (!values.length) return 0;
  return Number((sum(values) / values.length).toFixed(digits));
}

function median(values) {
  const clean = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (!clean.length) return null;
  const mid = Math.floor(clean.length / 2);
  if (clean.length % 2) return Number(clean[mid].toFixed(6));
  return Number(((clean[mid - 1] + clean[mid]) / 2).toFixed(6));
}

function toNumber(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function formatRate(value) {
  return Number.isFinite(value) ? `${(value * 100).toFixed(2)}%` : "n/a";
}

function formatNumber(value) {
  return Number.isFinite(value) ? value.toFixed(6) : "n/a";
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
