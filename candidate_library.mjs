export const DEFAULT_SETTINGS = {
  instrumentType: "EQUITY",
  region: "USA",
  universe: "TOP3000",
  delay: 1,
  decay: 0,
  neutralization: "SUBINDUSTRY",
  truncation: 0.08,
  pasteurization: "ON",
  unitHandling: "VERIFY",
  nanHandling: "OFF",
  language: "FASTEXPR",
  visualization: false,
  testPeriod: "P1Y",
};

export const SEED_CANDIDATES = [
  {
    id: "fundamental_income_strength",
    family: "fundamental-quality",
    hypothesis:
      "Companies with operating income strong versus their own trailing history tend to keep outperforming.",
    expression: "ts_rank(operating_income, 252)",
    expectedStrength: "persistent fundamental drift with relatively low turnover",
    expectedFailureMode: "slow update cadence can dilute short-horizon signal",
    firstMetric: "Sharpe",
  },
  {
    id: "balance_sheet_quality",
    family: "fundamental-balance-sheet",
    hypothesis:
      "Lower liabilities relative to assets should reward stronger balance sheets.",
    expression: "-rank(liabilities / assets)",
    expectedStrength: "stable cross-sectional quality spread",
    expectedFailureMode: "coverage gaps in recent listings",
    firstMetric: "Fitness",
  },
  {
    id: "price_mean_reversion",
    family: "price-volume",
    hypothesis:
      "Yesterday's winners underperform and losers rebound on the next day.",
    expression: "rank(-returns)",
    expectedStrength: "simple and broad coverage",
    expectedFailureMode: "often too crowded or too self-correlated",
    firstMetric: "Turnover",
  },
  {
    id: "volume_shock_reversion",
    family: "price-volume",
    hypothesis:
      "Large positive volume shocks followed by weak price action mean short-term exhaustion.",
    expression: "rank(ts_delta(volume, 20)) * rank(-returns)",
    expectedStrength: "captures attention bursts with fast reaction",
    expectedFailureMode: "turnover can spike",
    firstMetric: "Turnover",
  },
  {
    id: "profitability_efficiency",
    family: "fundamental-efficiency",
    hypothesis:
      "Peer-relative profitability efficiency should work better than raw levels.",
    expression: "group_rank(ts_rank(operating_income / assets, 252), subindustry)",
    expectedStrength: "better concentration control inside peer groups",
    expectedFailureMode: "group field or coverage mismatch",
    firstMetric: "Sharpe",
  },
];

export function buildInitialBatch(size = 5) {
  return SEED_CANDIDATES.slice(0, size).map((candidate) => ({
    ...candidate,
    settings: candidate.settings ? { ...DEFAULT_SETTINGS, ...candidate.settings } : { ...DEFAULT_SETTINGS },
  }));
}

export function buildMutationBatch(scoredCandidates, limit = 3) {
  const followUps = [];

  for (const candidate of scoredCandidates.slice(0, limit)) {
    const metrics = candidate.metrics ?? {};
    const turnover = metrics.turnover;
    const sharpe = metrics.sharpe;
    const fitness = metrics.fitness;

    if (Number.isFinite(turnover) && turnover > 0.7) {
      followUps.push({
        ...candidate,
        id: `${candidate.id}_decay4`,
        hypothesis: `${candidate.hypothesis} Smooth the signal to control turnover.`,
        expression: candidate.expression,
        settings: { ...candidate.settings, decay: 4 },
        mutationReason: "High turnover",
      });
    }

    if (
      candidate.family.startsWith("fundamental") &&
      Number.isFinite(sharpe) &&
      sharpe < 1.0
    ) {
      followUps.push({
        ...candidate,
        id: `${candidate.id}_peer_relative`,
        hypothesis: `${candidate.hypothesis} Compare firms inside peer groups instead of market-wide.`,
        expression: `group_rank(${candidate.expression}, subindustry)`,
        settings: { ...candidate.settings, neutralization: "SUBINDUSTRY" },
        mutationReason: "Weak Sharpe",
      });
    }

    if (
      candidate.family === "price-volume" &&
      Number.isFinite(turnover) &&
      turnover > 0.4
    ) {
      followUps.push({
        ...candidate,
        id: `${candidate.id}_slower_window`,
        hypothesis: `${candidate.hypothesis} Use a slower horizon to trade less noise.`,
        expression:
          candidate.id === "price_mean_reversion"
            ? "rank(-ts_mean(returns, 5))"
            : "rank(ts_mean(ts_delta(volume, 20), 5)) * rank(-ts_mean(returns, 3))",
        settings: { ...candidate.settings, decay: Math.max(candidate.settings.decay, 2) },
        mutationReason: "High turnover",
      });
    }

    if (
      Number.isFinite(fitness) &&
      fitness < 1.0 &&
      candidate.id === "fundamental_income_strength"
    ) {
      followUps.push({
        ...candidate,
        id: `${candidate.id}_scaled_by_price`,
        hypothesis:
          "Scaling operating income by price may improve cross-sectional comparability.",
        expression: "ts_rank(operating_income / close, 252)",
        mutationReason: "Weak fitness",
      });
    }
  }

  return dedupeCandidates(followUps).slice(0, limit);
}

function dedupeCandidates(candidates) {
  const seen = new Set();
  return candidates.filter((candidate) => {
    const key = `${candidate.expression}::${JSON.stringify(candidate.settings)}`;
    if (seen.has(key)) {
      return false;
    }
    seen.add(key);
    return true;
  });
}
