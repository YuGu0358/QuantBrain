import { mkdir, writeFile } from "node:fs/promises";
import { buildInitialBatch, buildMutationBatch } from "./candidate_library.mjs";

const API_ROOT = "https://api.worldquantbrain.com";
const DEFAULT_TIMEOUT_MS = 14 * 60 * 1000;
const DEFAULT_POLL_INTERVAL_MS = 15_000;

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const email = process.env.WQB_EMAIL;
  const password = process.env.WQB_PASSWORD;

  if (!email || !password) {
    throw new Error("Set WQB_EMAIL and WQB_PASSWORD before running this script.");
  }

  const rounds = Number(args.rounds ?? 1);
  const batchSize = Number(args.batchSize ?? 5);
  const mutationLimit = Number(args.mutationLimit ?? 3);
  const session = await authenticate(email, password);
  const runId = new Date().toISOString().replaceAll(":", "-");
  const runDir = new URL(`./runs/${runId}/`, import.meta.url);
  await mkdir(runDir, { recursive: true });

  const allRounds = [];
  let batch = buildInitialBatch(batchSize);

  for (let round = 1; round <= rounds; round += 1) {
    const roundResults = await runBatch(session, batch, {
      pollIntervalMs: Number(args.pollIntervalMs ?? DEFAULT_POLL_INTERVAL_MS),
      timeoutMs: Number(args.timeoutMs ?? DEFAULT_TIMEOUT_MS),
    });

    allRounds.push({
      round,
      startedAt: new Date().toISOString(),
      candidates: roundResults,
    });

    await writeJson(new URL(`round-${round}.json`, runDir), roundResults);

    const scored = roundResults
      .filter((candidate) => candidate.status === "completed")
      .sort((left, right) => (right.score ?? -Infinity) - (left.score ?? -Infinity));

    if (round >= rounds) {
      break;
    }

    batch = buildMutationBatch(scored, mutationLimit);
    if (batch.length === 0) {
      break;
    }
  }

  const summary = summarizeRun(allRounds);
  await writeJson(new URL("summary.json", runDir), summary);
  console.log(JSON.stringify(summary, null, 2));
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

  if (!response.ok) {
    throw new Error(`Authentication failed with ${response.status}: ${await response.text()}`);
  }

  const cookieJar = setCookieHeader(getSetCookie(response.headers));
  if (!cookieJar) {
    throw new Error("Authentication succeeded but no session cookie was returned.");
  }

  return { cookieJar };
}

async function runBatch(session, candidates, options) {
  const results = [];

  for (const candidate of candidates) {
    const submitted = await submitSimulation(session, candidate);
    const finished = await waitForSimulation(session, submitted.simulationId, options);
    const flattened = flattenObject(finished.payload);
    const metrics = extractMetrics(flattened);
    const score = scoreMetrics(metrics);

    results.push({
      ...candidate,
      submittedAt: submitted.submittedAt,
      simulationId: submitted.simulationId,
      status: finished.status,
      payload: finished.payload,
      metrics,
      score,
      metricPaths: metrics.paths,
      nextMutation: suggestNextMutation(candidate, metrics),
    });
  }

  return results;
}

async function submitSimulation(session, candidate) {
  const payload = {
    type: "REGULAR",
    settings: candidate.settings,
    regular: candidate.expression,
  };

  const response = await apiFetch(session, "/simulations", {
    method: "POST",
    body: JSON.stringify(payload),
  });

  const rawText = await response.text();
  const body = rawText ? safeJson(rawText) : null;
  const simulationId =
    body?.id ??
    body?.simulation ??
    extractSimulationId(response.headers.get("location")) ??
    extractSimulationId(rawText);

  if (!simulationId) {
    throw new Error(`Could not determine simulation id from response: ${rawText}`);
  }

  return {
    simulationId,
    submittedAt: new Date().toISOString(),
  };
}

async function waitForSimulation(session, simulationId, options) {
  const deadline = Date.now() + options.timeoutMs;

  while (Date.now() < deadline) {
    const response = await apiFetch(session, `/simulations/${simulationId}`, {
      method: "GET",
    });
    const payload = await response.json();

    if (!isPendingPayload(payload)) {
      const enrichedPayload = await attachAlphaDetails(session, payload);
      return {
        status: inferCompletionStatus(enrichedPayload),
        payload: enrichedPayload,
      };
    }

    await sleep(options.pollIntervalMs);
  }

  return {
    status: "timeout",
    payload: { detail: `Timed out after ${options.timeoutMs} ms.` },
  };
}

async function attachAlphaDetails(session, payload) {
  if (!payload?.alpha || typeof payload.alpha !== "string") {
    return payload;
  }

  try {
    const response = await apiFetch(session, `/alphas/${payload.alpha}`, {
      method: "GET",
    });
    const alphaPayload = await response.json();
    return {
      simulation: payload,
      alpha: alphaPayload,
    };
  } catch {
    return payload;
  }
}

function isPendingPayload(payload) {
  return (
    payload &&
    typeof payload === "object" &&
    Object.keys(payload).length === 1 &&
    typeof payload.progress === "number"
  );
}

function inferCompletionStatus(payload) {
  if (!payload || typeof payload !== "object") {
    return "unknown";
  }

  const flat = flattenObject(payload);
  const failure = Object.entries(flat).find(([path, value]) => {
    if (typeof value !== "string") {
      return false;
    }
    return /error|failed|invalid/i.test(path) || /error|failed|invalid/i.test(value);
  });

  return failure ? "failed" : "completed";
}

async function apiFetch(session, path, init) {
  const response = await fetch(`${API_ROOT}${path}`, {
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

  if (response.ok) {
    return response;
  }

  const errorText = await response.text();
  throw new Error(`${init?.method ?? "GET"} ${path} failed with ${response.status}: ${errorText}`);
}

function flattenObject(value, prefix = "", flat = {}) {
  if (Array.isArray(value)) {
    value.forEach((entry, index) => {
      flattenObject(entry, `${prefix}[${index}]`, flat);
    });
    return flat;
  }

  if (value && typeof value === "object") {
    for (const [key, nested] of Object.entries(value)) {
      const nextPrefix = prefix ? `${prefix}.${key}` : key;
      flattenObject(nested, nextPrefix, flat);
    }
    return flat;
  }

  flat[prefix] = value;
  return flat;
}

function extractMetrics(flat) {
  const sharpe = pickMetric(flat, ["sharpe"]);
  const fitness = pickMetric(flat, ["fitness"]);
  const turnover = pickMetric(flat, ["turnover"]);
  const returns = pickMetric(flat, ["returns", "annualizedReturn"]);
  const drawdown = pickMetric(flat, ["drawdown", "maxDrawdown"]);
  const margin = pickMetric(flat, ["margin"]);

  return {
    sharpe: sharpe.value,
    fitness: fitness.value,
    turnover: turnover.value,
    returns: returns.value,
    drawdown: drawdown.value,
    margin: margin.value,
    paths: {
      sharpe: sharpe.path,
      fitness: fitness.path,
      turnover: turnover.path,
      returns: returns.path,
      drawdown: drawdown.path,
      margin: margin.path,
    },
  };
}

function pickMetric(flat, aliases) {
  for (const alias of aliases) {
    const exact = Object.entries(flat).find(([path, value]) =>
      isNumericValue(value) && path.toLowerCase().endsWith(alias.toLowerCase()),
    );
    if (exact) {
      return { path: exact[0], value: Number(exact[1]) };
    }
  }

  for (const alias of aliases) {
    const loose = Object.entries(flat).find(([path, value]) =>
      isNumericValue(value) && path.toLowerCase().includes(alias.toLowerCase()),
    );
    if (loose) {
      return { path: loose[0], value: Number(loose[1]) };
    }
  }

  return { path: null, value: null };
}

function scoreMetrics(metrics) {
  let score = 0;

  if (Number.isFinite(metrics.sharpe)) {
    score += metrics.sharpe * 4;
  }
  if (Number.isFinite(metrics.fitness)) {
    score += metrics.fitness * 3;
  }
  if (Number.isFinite(metrics.returns)) {
    score += metrics.returns;
  }
  if (Number.isFinite(metrics.turnover)) {
    const turnoverPenalty =
      metrics.turnover < 0.01
        ? 1.5
        : metrics.turnover > 0.7
          ? (metrics.turnover - 0.7) * 10
          : 0;
    score -= turnoverPenalty;
  }
  if (Number.isFinite(metrics.drawdown)) {
    score -= Math.max(0, metrics.drawdown - 0.1) * 10;
  }

  return Number.isFinite(score) ? Number(score.toFixed(4)) : null;
}

function suggestNextMutation(candidate, metrics) {
  if (Number.isFinite(metrics.turnover) && metrics.turnover > 0.7) {
    return "Increase decay to 3-5 or use a slower horizon.";
  }
  if (Number.isFinite(metrics.sharpe) && metrics.sharpe < 1.0) {
    return candidate.family.startsWith("fundamental")
      ? "Try peer-relative comparison or a ratio form."
      : "Try a slower window or different operator class.";
  }
  if (Number.isFinite(metrics.fitness) && metrics.fitness < 1.0) {
    return "Simplify the mechanism or change the scaling.";
  }
  return "Promising candidate. Keep and mutate one knob at a time.";
}

function summarizeRun(rounds) {
  const completed = rounds
    .flatMap((round) => round.candidates)
    .filter((candidate) => candidate.status === "completed");

  const ranked = [...completed].sort(
    (left, right) => (right.score ?? -Infinity) - (left.score ?? -Infinity),
  );

  return {
    generatedAt: new Date().toISOString(),
    rounds: rounds.length,
    totalCandidates: rounds.reduce((count, round) => count + round.candidates.length, 0),
    completedCandidates: completed.length,
    bestCandidate: ranked[0] ?? null,
    leaderboard: ranked.slice(0, 10).map((candidate) => ({
      id: candidate.id,
      family: candidate.family,
      expression: candidate.expression,
      score: candidate.score,
      metrics: candidate.metrics,
      nextMutation: candidate.nextMutation,
    })),
  };
}

function parseArgs(argv) {
  const args = {};
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    if (!token.startsWith("--")) {
      continue;
    }
    const [key, inlineValue] = token.slice(2).split("=");
    if (inlineValue !== undefined) {
      args[key] = inlineValue;
      continue;
    }
    args[key] = argv[index + 1] && !argv[index + 1].startsWith("--") ? argv[++index] : true;
  }
  return args;
}

function getSetCookie(headers) {
  if (typeof headers.getSetCookie === "function") {
    return headers.getSetCookie();
  }
  const single = headers.get("set-cookie");
  return single ? single.split(/,(?=[^;]+?=)/g) : [];
}

function setCookieHeader(setCookies) {
  const entries = setCookies
    .map((cookie) => cookie.split(";", 1)[0])
    .filter(Boolean);
  return entries.join("; ");
}

function extractSimulationId(input) {
  if (!input) {
    return null;
  }
  const match = input.match(/simulations\/([A-Za-z0-9]+)/);
  return match?.[1] ?? null;
}

function safeJson(text) {
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}

function isNumericValue(value) {
  return value !== null && value !== "" && Number.isFinite(Number(value));
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function writeJson(url, value) {
  await writeFile(url, `${JSON.stringify(value, null, 2)}\n`, "utf8");
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exitCode = 1;
});
