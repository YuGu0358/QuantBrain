import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import {
  buildPlan,
  buildSeedBatch,
  buildRepairBatch,
  buildMutationBatch,
  choosePromotionBucket,
  applyPreflight,
  summarizeDiversity,
} from "./agentic_alpha_library.mjs";

const API_ROOT = "https://api.worldquantbrain.com";
const DEFAULT_TIMEOUT_MS = 14 * 60 * 1000;
const DEFAULT_POLL_INTERVAL_MS = 15_000;
const DEFAULT_SIMULATION_CONCURRENCY = clampInteger(process.env.WQB_SIM_CONCURRENCY ?? 3, 1, 3);

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const mode = args.mode ?? "generate";
  const verbose = args.verbose !== "false";
  const objective = args.objective ?? "discover robust WorldQuant BRAIN alphas";
  const outputDir = path.resolve(args.outputDir ?? "./runs/service");
  const memoryPath = path.join(outputDir, "memory.json");

  await mkdir(outputDir, { recursive: true });
  const memory = await loadMemory(memoryPath);
  const repairContext = args.repairContext ? await loadRepairContext(args.repairContext) : null;
  const plan = plannerAgent(objective, args, memory);

  if (mode === "generate") {
    const batch = repairContext ? repairGeneratorAgent(plan, memory, repairContext) : generatorAgent(plan, memory);
    const payload = {
      mode,
      plan,
      repairContext: summarizeRepairContext(repairContext),
      generatedAt: new Date().toISOString(),
      diversityStats: summarizeDiversity(batch, memory, plan),
      batch,
    };
    await writeJson(path.join(outputDir, "generated-batch.json"), payload);
    console.log(JSON.stringify(payload, null, 2));
    return;
  }

  const email = process.env.WQB_EMAIL;
  const password = process.env.WQB_PASSWORD;
  if (!email || !password) {
    throw new Error("Set WQB_EMAIL and WQB_PASSWORD for live evaluation modes.");
  }

  const session = await authenticate(email, password);
  const rounds = Number(args.rounds ?? plan.rounds);
  let batch = repairContext ? repairGeneratorAgent(plan, memory, repairContext) : generatorAgent(plan, memory);
  const roundsOutput = [];

  for (let round = 1; round <= rounds; round += 1) {
    const evaluated = await evaluatorAgent(session, batch, {
      timeoutMs: Number(args.timeoutMs ?? DEFAULT_TIMEOUT_MS),
      pollIntervalMs: Number(args.pollIntervalMs ?? DEFAULT_POLL_INTERVAL_MS),
      concurrency: clampInteger(args.concurrency ?? DEFAULT_SIMULATION_CONCURRENCY, 1, 3),
      verbose,
    });
    const scored = scoredCandidates(evaluated, memory);
    librarianAgent(memory, plan, scored);

    const roundPayload = {
      round,
      objective: plan.objective,
      repairContext: summarizeRepairContext(repairContext),
      generatedAt: new Date().toISOString(),
      diversityStats: summarizeDiversity(scored, memory, plan),
      scored,
      memorySummary: summarizeMemory(memory),
    };

    roundsOutput.push(roundPayload);
    await writeJson(path.join(outputDir, `batch-round-${round}.json`), roundPayload);
    await writeJson(memoryPath, memory);

    if (mode === "evaluate") break;
    batch = optimizerAgent(scored, memory, plan);
    if (batch.length === 0) break;
  }

  const summary = {
    mode,
    objective: plan.objective,
    generatedAt: new Date().toISOString(),
    rounds: roundsOutput.length,
    topCandidates: topCandidates(roundsOutput),
    diversityStats: roundsOutput.at(-1)?.diversityStats ?? null,
    repairContext: summarizeRepairContext(repairContext),
    memorySummary: summarizeMemory(memory),
  };

  await writeJson(path.join(outputDir, "summary.json"), summary);
  console.log(JSON.stringify(summary, null, 2));
}

function plannerAgent(objective, args, memory) {
  const plan = buildPlan(objective, args);
  return {
    ...plan,
    avoidedSignatures: [
      ...(memory.deprecatedPool ?? []).map((candidate) =>
        [...new Set(candidate.signature ?? [])].sort().join("|"),
      ),
    ],
  };
}

function generatorAgent(plan, memory) {
  const batch = buildSeedBatch(plan, memory, plan.batchSize).map((candidate, index) => ({
    ...candidate,
    trajectoryId: candidate.trajectoryId ?? `trajectory-${Date.now().toString(36)}-${index}`,
    plannerObjective: plan.objective,
    stage: "generated",
  }));
  return applyPreflight(batch, memory, plan, { phase: "generate" });
}

function repairGeneratorAgent(plan, memory, repairContext) {
  const batch = buildRepairBatch(repairContext, plan, memory, plan.batchSize).map((candidate, index) => ({
    ...candidate,
    trajectoryId: candidate.trajectoryId ?? `repair-trajectory-${Date.now().toString(36)}-${index}`,
    plannerObjective: plan.objective,
    stage: "repair-generated",
  }));
  return applyPreflight(batch, memory, plan, { phase: "repair" });
}

async function evaluatorAgent(session, batch, options) {
  const results = new Array(batch.length);
  const executable = [];
  for (const [index, candidate] of batch.entries()) {
    if (candidate.preflightStatus?.ok === false) {
      if (options.verbose) {
        console.error(`[preflight-rejected] ${candidate.id} :: ${candidate.preflightStatus.reasons.join(", ")}`);
      }
      results[index] = {
        ...candidate,
        stage: "preflight-rejected",
        payload: {
          preflightRejected: true,
          reasons: candidate.preflightStatus.reasons,
        },
      };
      continue;
    }
    executable.push({ index, candidate });
  }
  const concurrency = clampInteger(options.concurrency ?? DEFAULT_SIMULATION_CONCURRENCY, 1, 3);
  for (let start = 0; start < executable.length; start += concurrency) {
    const chunk = executable.slice(start, start + concurrency);
    await Promise.all(chunk.map(async ({ index, candidate }) => {
      results[index] = await evaluateCandidate(session, candidate, options);
    }));
  }
  return results.filter(Boolean);
}

async function evaluateCandidate(session, candidate, options) {
  try {
    if (options.verbose) console.error(`[submit] ${candidate.id} :: ${candidate.expression}`);
    const submission = await submitSimulation(session, candidate);
    if (options.verbose) console.error(`[simulation] ${candidate.id} -> ${submission.simulationId}`);
    const finished = await waitForSimulation(session, submission.simulationId, options, candidate.id);
    return {
      ...candidate,
      simulationId: submission.simulationId,
      submittedAt: submission.submittedAt,
      payload: finished.payload,
      stage: finished.status,
    };
  } catch (error) {
    if (options.verbose) {
      console.error(`[error] ${candidate.id} -> ${error instanceof Error ? error.message : String(error)}`);
    }
    return {
      ...candidate,
      stage: "error",
      payload: {
        error: error instanceof Error ? error.message : String(error),
      },
    };
  }
}

function optimizerAgent(scored, memory, plan) {
  const batch = buildMutationBatch(scored, memory, plan.batchSize, plan).map((candidate) => ({
    ...candidate,
    plannerObjective: plan.objective,
    stage: "mutated",
  }));
  return applyPreflight(batch, memory, plan, { phase: "optimize" });
}

function librarianAgent(memory, plan, scored) {
  memory.objectiveHistory.push({ objective: plan.objective, at: new Date().toISOString() });

  for (const candidate of scored) {
    const bucket = choosePromotionBucket(candidate);
    const record = {
      id: candidate.id,
      trajectoryId: candidate.trajectoryId,
      family: candidate.family,
      horizon: candidate.horizon,
      hypothesis: candidate.hypothesis,
      expression: candidate.expression,
      settings: candidate.settings,
      signature: candidate.signature,
      strategy: candidate.strategy ?? plan.generatorStrategy,
      dataFamily: candidate.dataFamily ?? null,
      settingPolicy: candidate.settingPolicy ?? null,
      operatorPattern: candidate.operatorPattern ?? null,
      crowdingPenalty: candidate.crowdingPenalty ?? null,
      preflightStatus: candidate.preflightStatus ?? null,
      failureClass: candidate.failureClass ?? null,
      parentIds: candidate.parentIds ?? [],
      metrics: candidate.metrics,
      checks: candidate.checks,
      scorecard: candidate.scorecard,
      totalScore: candidate.totalScore,
      alphaId: candidate.alphaId,
      simulationId: candidate.simulationId,
      mutationReason: candidate.mutationReason ?? null,
      promotedTo: bucket,
      updatedAt: new Date().toISOString(),
    };

    memory.trajectories.push(record);
    if (bucket === "effective") upsertByExpression(memory.effectivePool, record);
    else if (bucket === "deprecated") upsertByExpression(memory.deprecatedPool, record);
  }
}

async function authenticate(email, password) {
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
  if (!response.ok) throw new Error(`Authentication failed with ${response.status}: ${await response.text()}`);
  const cookieJar = setCookieHeader(getSetCookie(response.headers));
  if (!cookieJar) throw new Error("Authentication succeeded but no session cookie was returned.");
  return { cookieJar };
}

async function submitSimulation(session, candidate) {
  const response = await apiFetch(session, "/simulations", {
    method: "POST",
    body: JSON.stringify({ type: "REGULAR", settings: candidate.settings, regular: candidate.expression }),
  });
  const text = await response.text();
  const body = text ? safeJson(text) : null;
  const simulationId =
    body?.id ??
    body?.simulation ??
    extractSimulationId(response.headers.get("location")) ??
    extractSimulationId(text);
  if (!simulationId) throw new Error(`Could not determine simulation id from response: ${text}`);
  return { simulationId, submittedAt: new Date().toISOString() };
}

async function waitForSimulation(session, simulationId, options, label = simulationId) {
  const deadline = Date.now() + options.timeoutMs;
  let lastProgress = null;
  while (Date.now() < deadline) {
    const response = await apiFetch(session, `/simulations/${simulationId}`, { method: "GET" });
    const payload = await response.json();
    if (!isPendingPayload(payload)) {
      const enriched = await attachAlphaDetails(session, payload);
      if (options.verbose) {
        const alphaId = enriched?.alpha?.id ?? payload?.alpha ?? "unknown";
        console.error(`[complete] ${label} -> alpha ${alphaId}`);
      }
      return { status: inferCompletionStatus(enriched), payload: enriched };
    }
    if (options.verbose && payload.progress !== lastProgress) {
      lastProgress = payload.progress;
      console.error(`[progress] ${label} -> ${payload.progress}`);
    }
    await sleep(options.pollIntervalMs);
  }
  if (options.verbose) console.error(`[timeout] ${label} -> ${simulationId}`);
  return { status: "timeout", payload: { detail: `Timed out after ${options.timeoutMs} ms.` } };
}

async function attachAlphaDetails(session, payload) {
  if (!payload?.alpha || typeof payload.alpha !== "string") return payload;
  const response = await apiFetch(session, `/alphas/${payload.alpha}`, { method: "GET" });
  const alphaPayload = await response.json();
  return { simulation: payload, alpha: alphaPayload };
}

function scoredCandidates(candidates, memory) {
  return candidates.map((candidate) => {
    if (candidate.stage === "preflight-rejected") {
      return {
        ...candidate,
        alphaId: null,
        metrics: extractMetrics(null),
        checks: [],
        scorecard: null,
        totalScore: null,
        failureClass: "preflight",
      };
    }
    if (candidate.stage === "error") {
      return {
        ...candidate,
        alphaId: null,
        metrics: extractMetrics(null),
        checks: [],
        scorecard: null,
        totalScore: null,
        failureClass: "simulation-error",
      };
    }
    const alpha = candidate.payload?.alpha ?? null;
    const metrics = extractMetrics(alpha);
    const checks = alpha?.is?.checks ?? [];
    const scorecard = buildScorecard(candidate, metrics, checks, memory);
    return {
      ...candidate,
      alphaId: alpha?.id ?? candidate.payload?.alpha ?? null,
      metrics,
      checks,
      scorecard,
      totalScore: scorecard.total,
      failureClass: classifyFailure(metrics, checks),
    };
  });
}

function buildScorecard(candidate, metrics, checks, memory) {
  const strength = clamp01(average([scaledPositive(metrics.isSharpe, 2), scaledPositive(metrics.isFitness, 1.5)]));
  const consistency = clamp01(average([
    signAgreement(metrics.trainSharpe, metrics.testSharpe),
    stabilityScore(metrics.isSharpe, metrics.trainSharpe, metrics.testSharpe),
    positiveTestScore(metrics.testSharpe),
  ]));
  const efficiency = clamp01(average([turnoverScore(metrics.turnover), drawdownScore(metrics.drawdown)]));
  const diversity = clamp01(1 - maxSimilarity(candidate, memory.effectivePool ?? []));
  const submission = submissionReadinessScore(metrics, checks);
  const total = round4(0.3 * strength + 0.3 * consistency + 0.15 * efficiency + 0.15 * diversity + 0.1 * submission);
  return {
    strength: round4(strength),
    consistency: round4(consistency),
    efficiency: round4(efficiency),
    diversity: round4(diversity),
    submission: round4(submission),
    total,
  };
}

function extractMetrics(alpha) {
  const isMetrics = alpha?.is ?? {};
  const train = alpha?.train ?? {};
  const test = alpha?.test ?? {};
  return {
    isSharpe: toNumber(isMetrics.sharpe),
    isFitness: toNumber(isMetrics.fitness),
    turnover: toNumber(isMetrics.turnover),
    returns: toNumber(isMetrics.returns),
    drawdown: toNumber(isMetrics.drawdown),
    trainSharpe: toNumber(train.sharpe),
    trainFitness: toNumber(train.fitness),
    testSharpe: toNumber(test.sharpe),
    testFitness: toNumber(test.fitness),
  };
}

async function apiFetch(session, pathName, init) {
  const response = await fetch(`${API_ROOT}${pathName}`, {
    ...init,
    headers: {
      Accept: "application/json;version=2.0",
      "Content-Type": "application/json",
      Cookie: session.cookieJar,
      Origin: "https://platform.worldquantbrain.com",
      Referer: "https://platform.worldquantbrain.com/",
      ...(init?.headers ?? {}),
    },
  });
  if (response.ok) return response;
  const errorText = await response.text();
  throw new Error(`${init?.method ?? "GET"} ${pathName} failed with ${response.status}: ${errorText}`);
}

async function loadMemory(filePath) {
  const defaultState = { effectivePool: [], deprecatedPool: [], trajectories: [], objectiveHistory: [] };
  try {
    const raw = await readFile(filePath, "utf8");
    return { ...defaultState, ...JSON.parse(raw) };
  } catch {
    return defaultState;
  }
}

async function loadRepairContext(filePath) {
  const raw = await readFile(path.resolve(filePath), "utf8");
  return JSON.parse(raw);
}

function summarizeRepairContext(context) {
  if (!context) return null;
  return {
    parentAlphaId: context.parentAlphaId,
    rootAlphaId: context.rootAlphaId,
    failedChecks: context.failedChecks ?? [],
    repairDepth: context.repairDepth ?? 0,
    expression: context.expression ?? null,
    nextAction: context.nextAction ?? null,
  };
}

async function writeJson(filePath, value) {
  await writeFile(filePath, `${JSON.stringify(value, null, 2)}\n`, "utf8");
}

function summarizeMemory(memory) {
  return {
    effectivePool: memory.effectivePool.length,
    deprecatedPool: memory.deprecatedPool.length,
    trajectories: memory.trajectories.length,
    familyCounts: countBy(memory.trajectories, (candidate) => candidate.dataFamily ?? "UNKNOWN"),
    recentObjectives: memory.objectiveHistory.slice(-5),
  };
}

function topCandidates(roundsOutput) {
  return roundsOutput
    .flatMap((round) => round.scored ?? [])
    .filter((candidate) => candidate.totalScore !== null)
    .sort((left, right) => (right.totalScore ?? -Infinity) - (left.totalScore ?? -Infinity))
    .slice(0, 10)
    .map((candidate) => ({
      id: candidate.id,
      family: candidate.family,
      expression: candidate.expression,
      alphaId: candidate.alphaId,
      strategy: candidate.strategy ?? null,
      dataFamily: candidate.dataFamily ?? null,
      settingPolicy: candidate.settingPolicy ?? null,
      operatorPattern: candidate.operatorPattern ?? null,
      preflightStatus: candidate.preflightStatus ?? null,
      failureClass: candidate.failureClass ?? null,
      totalScore: candidate.totalScore,
      scorecard: candidate.scorecard,
      metrics: candidate.metrics,
    }));
}

function classifyFailure(metrics, checks) {
  const nonPass = checks.filter((check) => check.result !== "PASS").map((check) => check.name);
  if (nonPass.includes("LOW_SHARPE") || nonPass.includes("LOW_FITNESS")) return "weak-strength";
  if (nonPass.includes("SELF_CORRELATION")) return "crowding";
  if (nonPass.includes("LOW_SUB_UNIVERSE_SHARPE")) return "sub-universe-instability";
  if (metrics?.testSharpe !== null && metrics?.testSharpe <= 0) return "oos-instability";
  if (nonPass.length) return "gate";
  return null;
}

function countBy(items, getKey) {
  const counts = {};
  for (const item of items ?? []) {
    const key = getKey(item);
    counts[key] = (counts[key] ?? 0) + 1;
  }
  return counts;
}

function upsertByExpression(pool, record) {
  const index = pool.findIndex((item) => item.expression === record.expression);
  if (index === -1) pool.push(record);
  else pool[index] = record;
}

function maxSimilarity(candidate, pool) {
  if (pool.length === 0) return 0;
  return Math.max(...pool.map((other) => expressionSimilarity(candidate.expression, other.expression, candidate.signature, other.signature)));
}

function expressionSimilarity(leftExpression, rightExpression, leftSignature = [], rightSignature = []) {
  const leftTokens = new Set(tokenize(leftExpression).concat(leftSignature));
  const rightTokens = new Set(tokenize(rightExpression).concat(rightSignature));
  const intersection = [...leftTokens].filter((token) => rightTokens.has(token)).length;
  const union = new Set([...leftTokens, ...rightTokens]).size;
  return union === 0 ? 0 : intersection / union;
}

function tokenize(expression) {
  return expression.toLowerCase().split(/[^a-z0-9_]+/g).filter(Boolean);
}

function isPendingPayload(payload) {
  return payload && typeof payload === "object" && Object.keys(payload).length === 1 && typeof payload.progress === "number";
}

function inferCompletionStatus(payload) {
  const alpha = payload.alpha ?? null;
  if (alpha?.status) return String(alpha.status).toLowerCase();
  return "completed";
}

function getSetCookie(headers) {
  if (typeof headers.getSetCookie === "function") return headers.getSetCookie();
  const single = headers.get("set-cookie");
  return single ? single.split(/,(?=[^;]+?=)/g) : [];
}

function setCookieHeader(setCookies) {
  return setCookies.map((cookie) => cookie.split(";", 1)[0]).filter(Boolean).join("; ");
}

function extractSimulationId(input) {
  if (!input) return null;
  const match = input.match(/simulations\/([A-Za-z0-9]+)/);
  return match?.[1] ?? null;
}

function safeJson(text) {
  try { return JSON.parse(text); } catch { return null; }
}

function parseArgs(argv) {
  const args = {};
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    if (!token.startsWith("--")) continue;
    const [key, inlineValue] = token.slice(2).split("=");
    const normalizedKey = camelizeKey(key);
    if (inlineValue !== undefined) {
      args[key] = inlineValue;
      args[normalizedKey] = inlineValue;
      continue;
    }
    const value = argv[index + 1] && !argv[index + 1].startsWith("--") ? argv[++index] : true;
    args[key] = value;
    args[normalizedKey] = value;
  }
  return args;
}

function camelizeKey(key) {
  return key.replace(/-([a-z])/g, (_, char) => char.toUpperCase());
}

function toNumber(value) {
  return Number.isFinite(Number(value)) ? Number(value) : null;
}

function clampInteger(value, low, high) {
  const number = Number(value);
  if (!Number.isFinite(number)) return low;
  return Math.max(low, Math.min(high, Math.round(number)));
}

function average(values) {
  const filtered = values.filter((value) => value !== null && Number.isFinite(value));
  if (filtered.length === 0) return 0;
  return filtered.reduce((sum, value) => sum + value, 0) / filtered.length;
}

function scaledPositive(value, goodThreshold) {
  if (value === null) return 0;
  return clamp01(value / goodThreshold);
}

function signAgreement(left, right) {
  if (left === null || right === null) return 0.5;
  if (left === 0 || right === 0) return 0.4;
  return Math.sign(left) === Math.sign(right) ? 1 : 0;
}

function stabilityScore(isSharpe, trainSharpe, testSharpe) {
  if (isSharpe === null || trainSharpe === null) return 0.5;
  if (testSharpe === null) return 0.5;
  const spread = Math.abs(isSharpe - trainSharpe) + Math.abs(trainSharpe - testSharpe);
  return clamp01(1 - spread / 4);
}

function positiveTestScore(testSharpe) {
  if (testSharpe === null) return 0.4;
  if (testSharpe <= 0) return 0;
  return clamp01(testSharpe / 0.5);
}

function submissionReadinessScore(metrics, checks) {
  const testScore = positiveTestScore(metrics.testSharpe);
  if (!checks.length) return 0.25 * testScore;
  const passCount = checks.filter((check) => check.result === "PASS").length;
  const failCount = checks.filter((check) => check.result === "FAIL").length;
  const pendingCount = checks.length - passCount - failCount;
  const checkScore = clamp01((passCount - failCount - 0.35 * pendingCount) / checks.length);
  const strengthGate = numberGate(metrics.isSharpe, 1.25) * numberGate(metrics.isFitness, 1.0);
  return clamp01(0.45 * checkScore + 0.35 * testScore + 0.2 * strengthGate);
}

function numberGate(value, threshold) {
  if (value === null) return 0;
  return value >= threshold ? 1 : clamp01(value / threshold);
}

function turnoverScore(turnover) {
  if (turnover === null) return 0;
  if (turnover < 0.01) return 0.2;
  if (turnover <= 0.2) return 1;
  if (turnover <= 0.7) return clamp01(1 - (turnover - 0.2) / 0.8);
  return 0;
}

function drawdownScore(drawdown) {
  if (drawdown === null) return 0.4;
  return clamp01(1 - drawdown / 0.25);
}

function clamp01(value) {
  return Math.max(0, Math.min(1, value));
}

function round4(value) {
  return Number.isFinite(value) ? Number(value.toFixed(4)) : null;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exitCode = 1;
});
