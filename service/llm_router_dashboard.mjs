function asNumber(value, fallback = 0) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : fallback;
}

function asNonEmptyString(value) {
  if (typeof value !== "string") return null;
  const trimmed = value.trim();
  return trimmed ? trimmed : null;
}

function pickRecentRetrievalMode(recentRuns, config) {
  const ignoredModes = new Set(config.ignoredModes || []);
  for (const run of recentRuns || []) {
    const summary = run?.summary;
    if (!summary || typeof summary !== "object") continue;
    const mode = asNonEmptyString(summary[config.modeKey]);
    if (!mode || ignoredModes.has(mode)) continue;
    return {
      runId: run?.runId || null,
      mode,
      error: summary[config.errorKey] ?? null,
      [config.statusField]: asNonEmptyString(summary[config.statusKey]),
    };
  }
  return null;
}

export function flattenProviderEntries(providers) {
  if (!providers || typeof providers !== "object") return [];
  const flat = [];
  for (const [name, roleMap] of Object.entries(providers)) {
    if (roleMap && typeof roleMap === "object" && !("win_rate" in roleMap)) {
      for (const provider of Object.values(roleMap)) {
        flat.push({ ...provider, name: provider?.name || name });
      }
      continue;
    }
    flat.push({ ...roleMap, name });
  }
  return flat;
}

export function formatProviderWinRate(provider, options = {}) {
  const minCallsForRate = asNumber(options.minCallsForRate, 3);
  const calls = asNumber(provider?.calls, 0);
  const wins = asNumber(provider?.wins, 0);
  const winRate = asNumber(provider?.win_rate, 0);
  if (calls < minCallsForRate) {
    return {
      label: "N/A",
      width: 0,
      colorToken: "var(--t3)",
      detail: `样本不足（${calls} 次）`,
      lowSample: true,
    };
  }
  const width = Math.max(0, Math.min(100, Math.round(winRate * 100)));
  return {
    label: `${width}%`,
    width,
    colorToken: winRate >= 0.5 ? "var(--green)" : "var(--amber)",
    detail: `${wins}/${calls} 成功`,
    lowSample: false,
  };
}

export function summarizeDiagnoseProvider(providers, recentRuns = [], options = {}) {
  const minCallsForRate = asNumber(options.minCallsForRate, 3);
  const diagnose = (providers || []).find((provider) => provider?.role === "diagnose");
  if (!diagnose) return null;
  const formatted = formatProviderWinRate(diagnose, { minCallsForRate });
  let lastFailureReason = null;
  let lastFailureRunId = null;
  for (const run of recentRuns || []) {
    const diagnosis = run?.summary?.diagnosis;
    if (diagnosis?.fallback && diagnosis?.error) {
      lastFailureReason = String(diagnosis.error);
      lastFailureRunId = run?.runId || null;
      break;
    }
    if (diagnosis?.llm_fallback && diagnosis?.primary_error) {
      lastFailureReason = String(diagnosis.primary_error);
      lastFailureRunId = run?.runId || null;
      break;
    }
  }
  return {
    name: diagnose.name || "diagnose",
    role: diagnose.role || "diagnose",
    calls: asNumber(diagnose.calls, 0),
    wins: asNumber(diagnose.wins, 0),
    lowSample: formatted.lowSample,
    rateLabel: formatted.label,
    rateWidth: formatted.width,
    rateColorToken: formatted.colorToken,
    rateDetail: formatted.detail,
    lastFailureReason,
    lastFailureRunId,
  };
}

export function summarizeRetrievalStatus(recentRuns = []) {
  return {
    generation: pickRecentRetrievalMode(recentRuns, {
      modeKey: "kb_retrieval_mode",
      errorKey: "kb_retrieval_error",
      statusKey: "kb_embedder_status",
      statusField: "embedderStatus",
      ignoredModes: ["uninitialized"],
    }),
    repair: pickRecentRetrievalMode(recentRuns, {
      modeKey: "repair_retrieval_mode",
      errorKey: "repair_retrieval_error",
      statusKey: "repair_semantic_memory_status",
      statusField: "semanticMemoryStatus",
      ignoredModes: ["not_applicable", "uninitialized", "unused"],
    }),
  };
}
