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

export const GENERATOR_STRATEGIES = ["legacy", "diversity-v2"];
export const DEFAULT_GENERATOR_STRATEGY = GENERATOR_STRATEGIES.includes(process.env.ALPHA_GENERATOR_STRATEGY)
  ? process.env.ALPHA_GENERATOR_STRATEGY
  : "legacy";
export const DEFAULT_EXPERIMENTAL_FIELDS_ENABLED = process.env.ALPHA_EXPERIMENTAL_FIELDS === "true";
export const DEFAULT_CROWDING_PATTERN_THRESHOLD = clampInt(process.env.ALPHA_CROWDING_PATTERN_THRESHOLD, 2, 1, 20);
export const DEFAULT_FAMILY_COOLDOWN_ROUNDS = clampInt(process.env.ALPHA_FAMILY_COOLDOWN_ROUNDS, 3, 1, 20);

const DATA_FAMILY_ORDER = [
  "FUNDAMENTAL_QUALITY",
  "FUNDAMENTAL_VALUE",
  "PRICE_VOLUME",
  "EARNINGS_ANALYST",
  "MICROSTRUCTURE",
  "VOLATILITY",
  "SENTIMENT",
];

export const TEMPLATE_LIBRARY = [
  {
    id: "income_history",
    family: "fundamental-quality",
    dataFamily: "FUNDAMENTAL_QUALITY",
    horizon: "slow",
    hypothesis: "Operating income strength versus its own history predicts relative outperformance.",
    expression: "ts_rank(operating_income, 252)",
    signature: ["operating_income", "ts_rank", "252", "history"],
    settingPolicy: "fundamental-subindustry-slow",
  },
  {
    id: "income_price_scale",
    family: "fundamental-quality",
    dataFamily: "FUNDAMENTAL_QUALITY",
    horizon: "slow",
    hypothesis: "Scaling operating income by price improves cross-sectional comparability.",
    expression: "ts_rank(operating_income / close, 252)",
    signature: ["operating_income", "close", "ts_rank", "252", "scale"],
    settingPolicy: "fundamental-subindustry-slow",
  },
  {
    id: "income_asset_scale",
    family: "fundamental-efficiency",
    dataFamily: "FUNDAMENTAL_QUALITY",
    horizon: "slow",
    hypothesis: "Operating income scaled by assets captures profitability efficiency.",
    expression: "ts_rank(operating_income / assets, 252)",
    signature: ["operating_income", "assets", "ts_rank", "252", "efficiency"],
    settingPolicy: "fundamental-subindustry-slow",
  },
  {
    id: "income_asset_peer",
    family: "fundamental-efficiency",
    dataFamily: "FUNDAMENTAL_QUALITY",
    horizon: "slow",
    hypothesis: "Peer-relative profitability efficiency should outperform raw cross-sectional ranking.",
    expression: "group_rank(ts_rank(operating_income / assets, 252), industry)",
    signature: ["operating_income", "assets", "group_rank", "industry", "252"],
    settings: { neutralization: "INDUSTRY" },
    settingPolicy: "fundamental-industry-peer",
  },
  {
    id: "income_asset_subindustry_peer",
    family: "fundamental-efficiency",
    dataFamily: "FUNDAMENTAL_QUALITY",
    horizon: "slow",
    hypothesis: "Subindustry-relative profitability efficiency reduces broad sector structure and crowding.",
    expression: "group_rank(ts_rank(operating_income / assets, 252), subindustry)",
    signature: ["operating_income", "assets", "group_rank", "subindustry", "252"],
    settings: { neutralization: "SUBINDUSTRY" },
    settingPolicy: "fundamental-subindustry-peer",
  },
  {
    id: "income_asset_history_blend",
    family: "fundamental-efficiency",
    dataFamily: "FUNDAMENTAL_QUALITY",
    horizon: "slow",
    hypothesis: "Combining profitability efficiency with raw operating-income persistence improves stability without changing the economic thesis.",
    expression: "rank(ts_rank(operating_income / assets, 252) + ts_rank(operating_income, 252))",
    signature: ["operating_income", "assets", "ts_rank", "rank", "252", "blend", "persistence"],
    settingPolicy: "fundamental-subindustry-slow",
  },
  {
    id: "income_asset_price_blend",
    family: "fundamental-efficiency",
    dataFamily: "FUNDAMENTAL_QUALITY",
    horizon: "slow",
    hypothesis: "Asset-scaled operating-income quality is the anchor; price-scaled quality is used only as a secondary strength signal.",
    expression: "rank(ts_rank(operating_income / assets, 252) + ts_rank(operating_income / close, 252))",
    signature: ["operating_income", "assets", "close", "ts_rank", "rank", "252", "blend"],
    settingPolicy: "fundamental-subindustry-slow",
  },
  {
    id: "income_asset_price_peer_blend",
    family: "fundamental-efficiency",
    dataFamily: "FUNDAMENTAL_QUALITY",
    horizon: "slow",
    hypothesis: "A peer-relative blend of asset-scaled and price-scaled operating income targets stronger IS signal while preserving OOS robustness.",
    expression: "group_rank(rank(ts_rank(operating_income / assets, 252) + ts_rank(operating_income / close, 252)), industry)",
    signature: ["operating_income", "assets", "close", "group_rank", "industry", "rank", "252", "blend"],
    settings: { neutralization: "INDUSTRY" },
    settingPolicy: "fundamental-industry-peer",
  },
  {
    id: "income_price_peer",
    family: "fundamental-quality",
    dataFamily: "FUNDAMENTAL_QUALITY",
    horizon: "slow",
    hypothesis: "Peer-relative operating income scaled by price reduces sector structure noise.",
    expression: "group_rank(ts_rank(operating_income / close, 252), industry)",
    signature: ["operating_income", "close", "group_rank", "industry", "252"],
    settings: { neutralization: "INDUSTRY" },
    settingPolicy: "fundamental-industry-peer",
  },
  {
    id: "cashflow_asset_quality",
    family: "fundamental-quality",
    dataFamily: "FUNDAMENTAL_QUALITY",
    horizon: "slow",
    hypothesis: "Cash-flow backed profitability quality should be less accrual-sensitive than operating-income level alone.",
    expression: "rank(ts_rank(cashflow_op / assets, 252) + ts_rank(operating_income / assets, 252))",
    signature: ["cashflow_op", "assets", "operating_income", "rank", "ts_rank", "quality"],
    settingPolicy: "fundamental-subindustry-slow",
  },
  {
    id: "liability_asset_reversal",
    family: "fundamental-value",
    dataFamily: "FUNDAMENTAL_VALUE",
    horizon: "slow",
    hypothesis: "Lower liability burden relative to assets can proxy balance-sheet quality after peer ranking.",
    expression: "group_rank(-rank(liabilities / assets), industry)",
    signature: ["liabilities", "assets", "group_rank", "rank", "balance-sheet"],
    settings: { neutralization: "INDUSTRY" },
    settingPolicy: "fundamental-industry-peer",
  },
  {
    id: "estimate_price_peer",
    family: "estimate-revision",
    dataFamily: "EARNINGS_ANALYST",
    horizon: "medium",
    hypothesis: "EPS estimates relative to price work best inside industry peers.",
    expression: "group_rank(ts_rank(est_eps / close, 60), industry)",
    signature: ["est_eps", "close", "group_rank", "industry", "60"],
    settings: { neutralization: "INDUSTRY" },
    settingPolicy: "fundamental-industry-peer",
  },
  {
    id: "price_reversal",
    family: "price-volume",
    dataFamily: "PRICE_VOLUME",
    horizon: "fast",
    hypothesis: "Recent winners mean revert and recent losers rebound.",
    expression: "rank(-returns)",
    signature: ["returns", "rank", "reversal", "1d"],
    settings: { neutralization: "MARKET", decay: 12, truncation: 0.08 },
    settingPolicy: "price-volume-market-slowed",
  },
  {
    id: "volume_reversal",
    family: "price-volume",
    dataFamily: "PRICE_VOLUME",
    horizon: "fast",
    hypothesis: "Volume shocks combined with weak price follow-through indicate exhaustion.",
    expression: "rank(ts_delta(volume, 20)) * rank(-returns)",
    signature: ["volume", "ts_delta", "returns", "reversal", "20"],
    settings: { neutralization: "MARKET", decay: 12, truncation: 0.08 },
    settingPolicy: "price-volume-market-slowed",
  },
  {
    id: "vwap_close_pressure",
    family: "microstructure",
    dataFamily: "MICROSTRUCTURE",
    horizon: "fast",
    hypothesis: "VWAP-close pressure can capture short-term intraday exhaustion without reusing fundamental quality exposure.",
    expression: "ts_decay_linear(rank((vwap - close) / close), 4)",
    signature: ["vwap", "close", "ts_decay_linear", "rank", "microstructure"],
    settings: { neutralization: "MARKET", decay: 10, truncation: 0.08 },
    settingPolicy: "price-volume-market-slowed",
  },
  {
    id: "volume_volatility_gate",
    family: "volatility",
    dataFamily: "VOLATILITY",
    horizon: "medium",
    hypothesis: "Volume shocks matter more when recent volatility is elevated, reducing regime mismatch.",
    expression: "rank(ts_rank(ts_std_dev(returns, 22), 252) * ts_delta(volume, 20))",
    signature: ["returns", "volume", "ts_std_dev", "ts_rank", "ts_delta", "volatility"],
    settings: { neutralization: "MARKET", decay: 8, truncation: 0.08 },
    settingPolicy: "price-volume-market-slowed",
  },
  {
    id: "sentiment_news_abnormal",
    family: "sentiment",
    dataFamily: "SENTIMENT",
    horizon: "medium",
    hypothesis: "Abnormal news sentiment can capture delayed repricing, but this field is account-dependent.",
    expression: "rank(ts_zscore(news_sentiment, 20))",
    signature: ["news_sentiment", "ts_zscore", "rank", "sentiment"],
    settings: { neutralization: "SUBINDUSTRY", decay: 8, truncation: 0.1 },
    settingPolicy: "sentiment-subindustry",
    experimental: true,
  },
];

export function buildPlan(objective, constraints = {}) {
  const text = (objective ?? "").toLowerCase();
  const preferredFamilies = [];
  const prioritizeRobustTest = /(robust|oos|out.?of.?sample|near.?oos|test stability|positive test|submit|submission)/.test(text);
  const avoidCrowding = /(low crowding|crowding|divers|orthogonal|self.?correlation|correlation)/.test(text);
  const generatorStrategy = GENERATOR_STRATEGIES.includes(constraints.generatorStrategy)
    ? constraints.generatorStrategy
    : DEFAULT_GENERATOR_STRATEGY;
  const experimentalFieldsEnabled = parseBoolean(
    constraints.experimentalFields ?? constraints.experimentalFieldsEnabled,
    DEFAULT_EXPERIMENTAL_FIELDS_ENABLED,
  );
  const crowdingPatternThreshold = clampInt(
    constraints.crowdingPatternThreshold,
    DEFAULT_CROWDING_PATTERN_THRESHOLD,
    1,
    20,
  );
  const familyCooldownRounds = clampInt(
    constraints.familyCooldownRounds,
    DEFAULT_FAMILY_COOLDOWN_ROUNDS,
    1,
    20,
  );

  if (/(fundamental|quality|profit|income|value|robust)/.test(text)) {
    preferredFamilies.push("fundamental-quality", "fundamental-efficiency");
  }
  if (/(estimate|eps|analyst)/.test(text)) {
    preferredFamilies.push("estimate-revision");
  }
  if (/(price|volume|mean reversion|momentum|turnover)/.test(text)) {
    preferredFamilies.push("price-volume", "microstructure", "volatility");
  }
  if (/(sentiment|news|attention|buzz)/.test(text)) {
    preferredFamilies.push("sentiment");
  }
  if (preferredFamilies.length === 0) {
    preferredFamilies.push("fundamental-quality", "fundamental-efficiency", "price-volume");
  }
  if (generatorStrategy === "diversity-v2" && (avoidCrowding || prioritizeRobustTest)) {
    preferredFamilies.push("fundamental-value", "estimate-revision", "microstructure", "volatility", "price-volume");
    if (experimentalFieldsEnabled) preferredFamilies.push("sentiment");
  }

  return {
    objective: objective ?? "discover robust WorldQuant BRAIN alphas",
    preferredFamilies: dedupeArray(preferredFamilies),
    batchSize: Number(constraints.batchSize ?? 5),
    rounds: Number(constraints.rounds ?? 2),
    focus: prioritizeRobustTest ? "robustness" : "score",
    prioritizeRobustTest,
    avoidCrowding,
    generatorStrategy,
    experimentalFieldsEnabled,
    crowdingPatternThreshold,
    familyCooldownRounds,
  };
}

export function buildSeedBatch(plan, memory, batchSize = 5) {
  if (plan.generatorStrategy === "diversity-v2") {
    return buildDiversityV2SeedBatch(plan, memory, batchSize);
  }
  const candidates = orderTemplatesForPlan(
    enabledTemplates(plan).filter((template) => plan.preferredFamilies.includes(template.family)),
    plan,
  ).map((template) => materializeCandidate(template, { strategy: plan.generatorStrategy }));

  const parentCandidates = selectParents(memory, plan, Math.max(0, Math.floor(batchSize / 2)));
  const merged = dedupeBySignature([...parentCandidates, ...candidates]);
  const filtered = avoidDeprecatedSignatures(merged, memory);
  return diversifyCandidates(filtered.length >= batchSize ? filtered : merged, batchSize);
}

function buildDiversityV2SeedBatch(plan, memory, batchSize = 5) {
  const templates = enabledTemplates(plan)
    .filter((template) => plan.preferredFamilies.includes(template.family))
    .sort((left, right) => templateDiversityWeight(left, memory, plan) - templateDiversityWeight(right, memory, plan));
  const parentCandidates = selectParents(memory, plan, Math.max(0, Math.floor(batchSize / 3)))
    .map((candidate) => ({
      ...candidate,
      strategy: "diversity-v2",
      agentNotes: [...(candidate.agentNotes ?? []), "Selected parent in diversity-v2 cross-sample phase"],
    }));
  const candidates = templates.map((template) => materializeCandidate(template, { strategy: plan.generatorStrategy }));
  const merged = dedupeBySignature([...parentCandidates, ...candidates]);
  const filtered = avoidDeprecatedSignatures(merged, memory);
  return diversifyCandidatesByDataFamily(filtered.length >= batchSize ? filtered : merged, batchSize, memory);
}

export function buildRepairBatch(repairContext, plan, memory, batchSize = 5) {
  const repairTemplates = buildRepairTemplates(repairContext);
  const repairCandidates = repairTemplates.map((template) =>
    materializeCandidate(template, {
      trajectoryId: `repair-${repairContext?.rootAlphaId ?? repairContext?.parentAlphaId ?? "alpha"}`,
      parentIds: [repairContext?.parentAlphaId].filter(Boolean),
      strategy: plan.generatorStrategy,
      agentNotes: [
        `Repair context: ${(repairContext?.failedChecks ?? []).join(", ") || "gate failure"}`,
        repairContext?.nextAction ?? "Targeted repair",
      ],
    }),
  );
  const originalExpression = repairContext?.expression ?? null;
  const fallbackSeeds = buildSeedBatch(plan, memory, Math.max(0, batchSize - repairCandidates.length))
    .filter((candidate) => candidate.expression !== originalExpression)
    .map((candidate) => ({
      ...candidate,
      agentNotes: [...(candidate.agentNotes ?? []), "Fallback seed added only because targeted repair batch had spare capacity"],
    }));
  return diversifyCandidates(dedupeBySignature([...repairCandidates, ...fallbackSeeds]), batchSize);
}

export function applyPreflight(candidates, memory, plan, context = {}) {
  const history = [...(memory?.trajectories ?? []), ...(memory?.effectivePool ?? []), ...(memory?.deprecatedPool ?? [])];
  const currentPatterns = new Map();
  const currentFamilies = new Map();
  return candidates.map((candidate) => {
    const result = preflightCandidate(candidate, history, plan, currentPatterns, currentFamilies);
    currentPatterns.set(result.operatorPattern, (currentPatterns.get(result.operatorPattern) ?? 0) + 1);
    currentFamilies.set(candidate.dataFamily, (currentFamilies.get(candidate.dataFamily) ?? 0) + 1);
    return {
      ...candidate,
      operatorPattern: result.operatorPattern,
      strategy: plan.generatorStrategy,
      preflightStatus: result,
      crowdingPenalty: result.crowdingPenalty,
      failureClass: result.ok ? null : "preflight",
      preflightContext: {
        phase: context.phase ?? null,
        checkedAt: new Date().toISOString(),
      },
    };
  });
}

export function summarizeDiversity(candidates, memory, plan) {
  const familyCounts = countBy(candidates, (candidate) => candidate.dataFamily ?? "UNKNOWN");
  const submittedFamilyCounts = countBy(
    candidates.filter((candidate) => candidate.preflightStatus?.ok !== false),
    (candidate) => candidate.dataFamily ?? "UNKNOWN",
  );
  const preflightRejections = countBy(
    candidates.filter((candidate) => candidate.preflightStatus?.ok === false),
    (candidate) => candidate.preflightStatus?.primaryReason ?? "unknown",
  );
  const patternCounts = countBy(candidates, (candidate) => candidate.operatorPattern ?? extractOperatorPattern(candidate.expression));
  const historyPatternCounts = countBy(
    [...(memory?.trajectories ?? []), ...(memory?.effectivePool ?? []), ...(memory?.deprecatedPool ?? [])],
    (candidate) => candidate.operatorPattern ?? extractOperatorPattern(candidate.expression),
  );
  const crowdedPatterns = Object.entries(mergeCounts(historyPatternCounts, patternCounts))
    .filter(([, count]) => count >= plan.crowdingPatternThreshold)
    .sort((left, right) => right[1] - left[1])
    .slice(0, 8)
    .map(([pattern, count]) => ({ pattern, count }));

  return {
    strategy: plan.generatorStrategy,
    experimentalFieldsEnabled: plan.experimentalFieldsEnabled,
    crowdingPatternThreshold: plan.crowdingPatternThreshold,
    familyCooldownRounds: plan.familyCooldownRounds,
    familyCounts,
    submittedFamilyCounts,
    preflightRejections,
    crowdedPatterns,
    preflightRejected: candidates.filter((candidate) => candidate.preflightStatus?.ok === false).length,
    preflightAccepted: candidates.filter((candidate) => candidate.preflightStatus?.ok !== false).length,
  };
}

function buildRepairTemplates(repairContext) {
  const expression = repairContext?.expression ?? "ts_rank(operating_income / assets, 252)";
  const failed = new Set(repairContext?.failedChecks ?? []);
  const alpha = repairContext?.alpha ?? {};
  const templates = [];
  const add = (id, hypothesis, repairedExpression, signature, settings = {}) => {
    templates.push({
      id,
      family: inferRepairFamily(repairedExpression),
      dataFamily: inferDataFamily(repairedExpression),
      horizon: repairedExpression.includes("252") ? "slow" : "medium",
      hypothesis,
      expression: repairedExpression,
      signature,
      settings,
      settingPolicy: inferSettingPolicy(repairedExpression, settings),
    });
  };

  if (failed.has("LOW_SHARPE") || failed.has("LOW_FITNESS")) {
    if (expression.includes("operating_income") && expression.includes("assets")) {
      add(
        "repair_quality_improvement_delta",
        "Repair weak Sharpe/Fitness by testing profitability improvement instead of only profitability level.",
        "rank(ts_delta(operating_income / assets, 252))",
        ["repair", "operating_income", "assets", "ts_delta", "quality-improvement"],
      );
      add(
        "repair_quality_peer_blend",
        "Repair weak Sharpe/Fitness while preserving the quality thesis through peer-relative strength blending.",
        "group_rank(rank(ts_rank(operating_income / assets, 252) + ts_rank(operating_income / close, 252)), industry)",
        ["repair", "operating_income", "assets", "close", "group_rank", "industry", "blend"],
        { neutralization: "INDUSTRY" },
      );
    } else {
      add(
        "repair_generic_peer_rank",
        "Repair weak Sharpe/Fitness by moving the signal into peer-relative space.",
        `group_rank(${expression}, industry)`,
        ["repair", "group_rank", "industry", ...tokenize(expression)],
        { neutralization: "INDUSTRY" },
      );
    }
  }

  if (failed.has("SELF_CORRELATION") || failed.has("LOW_SUB_UNIVERSE_SHARPE")) {
    if (expression.includes("operating_income")) {
      add(
        "repair_cross_family_price_volume",
        "Repair crowding by leaving the operating-income family and testing a price-volume exhaustion mechanism.",
        "rank(ts_delta(volume, 20)) * rank(-returns)",
        ["repair", "price-volume", "volume", "returns", "cross-family"],
        { neutralization: "MARKET", decay: 12, truncation: 0.08 },
      );
      add(
        "repair_cross_family_vwap_pressure",
        "Repair crowding by changing both data family and horizon to intraday pressure.",
        "ts_decay_linear(rank((vwap - close) / close), 4)",
        ["repair", "microstructure", "vwap", "close", "cross-family"],
        { neutralization: "MARKET", decay: 10, truncation: 0.08 },
      );
      add(
        "repair_cross_family_balance_sheet",
        "Repair crowding by moving to a balance-sheet value mechanism instead of another profitability ratio.",
        "group_rank(-rank(liabilities / assets), industry)",
        ["repair", "fundamental-value", "liabilities", "assets", "cross-family"],
        { neutralization: "INDUSTRY", decay: 2, truncation: 0.05 },
      );
    }
    add(
      "repair_subindustry_escape",
      "Repair crowding or sub-universe instability by changing the comparison group materially.",
      expression.includes("operating_income")
        ? "group_rank(ts_delta(operating_income / assets, 252), subindustry)"
        : `group_rank(${expression}, subindustry)`,
      ["repair", "subindustry", "concept-escape", ...tokenize(expression).slice(0, 6)],
      { neutralization: "SUBINDUSTRY" },
    );
  }

  if (failed.has("HIGH_TURNOVER")) {
    add(
      "repair_slow_smooth",
      "Repair high turnover by smoothing the signal with a slower time-series mean.",
      `rank(ts_mean(${expression}, 20))`,
      ["repair", "ts_mean", "slow-window", ...tokenize(expression).slice(0, 6)],
      { decay: 6 },
    );
  }

  if (failed.has("CONCENTRATED_WEIGHT")) {
    add(
      "repair_concentration_rank",
      "Repair concentrated weight by forcing a bounded cross-sectional ranking transform.",
      `rank(${expression})`,
      ["repair", "rank", "concentration", ...tokenize(expression).slice(0, 6)],
    );
  }

  if (alpha.testSharpe !== null && alpha.testSharpe <= 0) {
    if (expression.includes("operating_income")) {
      add(
        "repair_test_cross_family_volatility",
        "Repair non-positive test behavior by moving away from the crowded profitability regime and adding volatility conditioning.",
        "rank(ts_rank(ts_std_dev(returns, 22), 252) * ts_delta(volume, 20))",
        ["repair", "volatility", "returns", "volume", "test-robust"],
        { neutralization: "MARKET", decay: 8, truncation: 0.08 },
      );
    }
    add(
      "repair_test_robust_peer",
      "Repair non-positive test behavior by using peer-relative robust quality rather than raw level exposure.",
      expression.includes("operating_income")
        ? "group_rank(rank(ts_delta(operating_income / assets, 252)), industry)"
        : `group_rank(rank(${expression}), industry)`,
      ["repair", "test-robust", "group_rank", "industry", ...tokenize(expression).slice(0, 6)],
      { neutralization: "INDUSTRY" },
    );
  }

  if (!templates.length) {
    add(
      "repair_default_peer_delta",
      "Repair gate failure by changing horizon and peer-comparison mechanics instead of making a cosmetic edit.",
      expression.includes("operating_income")
        ? "group_rank(ts_delta(operating_income / assets, 252), industry)"
        : `group_rank(rank(${expression}), industry)`,
      ["repair", "default", "group_rank", "industry", ...tokenize(expression).slice(0, 6)],
      { neutralization: "INDUSTRY" },
    );
  }

  return templates.slice(0, 5);
}

function inferRepairFamily(expression) {
  if (expression.includes("ts_std_dev")) return "volatility";
  if (expression.includes("vwap")) return "microstructure";
  if (expression.includes("operating_income") || expression.includes("assets")) return "fundamental-efficiency";
  if (expression.includes("volume") || expression.includes("returns") || expression.includes("close")) return "price-volume";
  return "fundamental-quality";
}

function inferDataFamily(expression) {
  if (expression.includes("news_sentiment")) return "SENTIMENT";
  if (expression.includes("est_eps")) return "EARNINGS_ANALYST";
  if (expression.includes("vwap")) return "MICROSTRUCTURE";
  if (expression.includes("ts_std_dev")) return "VOLATILITY";
  if (expression.includes("returns") || expression.includes("volume")) return "PRICE_VOLUME";
  if (expression.includes("liabilities")) return "FUNDAMENTAL_VALUE";
  return "FUNDAMENTAL_QUALITY";
}

function inferSettingPolicy(expression, settings = {}) {
  const dataFamily = inferDataFamily(expression);
  if (dataFamily === "PRICE_VOLUME" || dataFamily === "MICROSTRUCTURE" || dataFamily === "VOLATILITY") {
    return "price-volume-market-slowed";
  }
  if (dataFamily === "SENTIMENT") return "sentiment-subindustry";
  if (settings.neutralization === "INDUSTRY") return "fundamental-industry-peer";
  return "fundamental-subindustry-slow";
}

export function buildMutationBatch(scoredCandidates, memory, limit = 4, plan = {}) {
  const mutations = [];
  const promising = scoredCandidates
    .filter((candidate) => candidate.totalScore !== null)
    .sort((left, right) => (right.totalScore ?? -Infinity) - (left.totalScore ?? -Infinity))
    .slice(0, limit);

  for (const candidate of promising) {
    const stats = candidate.metrics ?? {};
    const checks = candidate.checks ?? [];
    const failedNames = new Set(checks.filter((check) => check.result === "FAIL").map((check) => check.name));
    const nonPassNames = new Set(checks.filter((check) => check.result !== "PASS").map((check) => check.name));
    const weakStrength = failedNames.has("LOW_SHARPE") || failedNames.has("LOW_FITNESS") || numberBelow(stats.isSharpe, 1);
    const positiveTest = stats.testSharpe !== null && stats.testSharpe > 0;
    const nonPositiveTest = stats.testSharpe !== null && stats.testSharpe <= 0;

    if (candidate.family.startsWith("fundamental") && positiveTest && weakStrength) {
      mutations.push(makeMutation(candidate, {
        suffix: "asset_anchor_strength_blend",
        hypothesis: `${candidate.hypothesis} Preserve the asset-scaled quality anchor, but add a price-scaled strength channel to target failed Sharpe/Fitness checks.`,
        expression:
          "rank(ts_rank(operating_income / assets, 252) + ts_rank(operating_income / close, 252))",
        signature: [...candidate.signature, "assets", "close", "strength-blend", "positive-test"],
        mutationReason: "Positive test behavior but weak IS submission checks",
      }));

      mutations.push(makeMutation(candidate, {
        suffix: "peer_strength_blend",
        hypothesis: `${candidate.hypothesis} Keep the profitability-efficiency thesis and reduce crowding through peer-relative ranking.`,
        expression:
          "group_rank(rank(ts_rank(operating_income / assets, 252) + ts_rank(operating_income / close, 252)), industry)",
        settings: { ...candidate.settings, neutralization: "INDUSTRY" },
        signature: [...candidate.signature, "assets", "close", "group_rank", "industry", "positive-test"],
        mutationReason: "Positive test behavior with failed checks; add strength and concept distance",
      }));
    }

    if (candidate.family.startsWith("fundamental") && nonPositiveTest) {
      if (plan.generatorStrategy === "diversity-v2" && candidate.expression.includes("operating_income")) {
        mutations.push(makeMutation(candidate, {
          suffix: "oos_cross_family_reversal",
          hypothesis: `${candidate.hypothesis} The original quality signal failed near-OOS; test a separate price-volume exhaustion mechanism instead of tuning the same ratio.`,
          expression: "rank(ts_delta(volume, 20)) * rank(-returns)",
          settings: { ...candidate.settings, neutralization: "MARKET", decay: 12, truncation: 0.08 },
          signature: [...candidate.signature, "cross-family", "volume", "returns", "oos-repair"],
          dataFamily: "PRICE_VOLUME",
          settingPolicy: "price-volume-market-slowed",
          mutationReason: "Non-positive test behavior; force cross-family robustness repair",
        }));
        mutations.push(makeMutation(candidate, {
          suffix: "oos_cross_family_vwap",
          hypothesis: `${candidate.hypothesis} Replace the failed operating-income exposure with intraday pressure to avoid regime-specific fundamental crowding.`,
          expression: "ts_decay_linear(rank((vwap - close) / close), 4)",
          settings: { ...candidate.settings, neutralization: "MARKET", decay: 10, truncation: 0.08 },
          signature: [...candidate.signature, "cross-family", "vwap", "close", "oos-repair"],
          dataFamily: "MICROSTRUCTURE",
          settingPolicy: "price-volume-market-slowed",
          mutationReason: "Non-positive test behavior; force cross-family robustness repair",
        }));
      }

      if (
        candidate.expression.includes("operating_income / close") &&
        !candidate.expression.includes("operating_income / assets")
      ) {
        mutations.push(makeMutation(candidate, {
          suffix: "blend_assets_scale",
          hypothesis: `${candidate.hypothesis} Blend price scaling with asset scaling to improve robustness.`,
          expression:
            "rank(ts_rank(operating_income / close, 252) + ts_rank(operating_income / assets, 252))",
          signature: [...candidate.signature, "blend", "assets"],
          mutationReason: "Strong IS but weak near-OOS",
        }));
      }

      if (!candidate.expression.includes("group_rank(")) {
        mutations.push(makeMutation(candidate, {
          suffix: "peer_relative",
          hypothesis: `${candidate.hypothesis} Compare companies inside industry peers.`,
          expression: `group_rank(${candidate.expression}, industry)`,
          settings: { ...candidate.settings, neutralization: "INDUSTRY" },
          signature: [...candidate.signature, "group_rank", "industry"],
          mutationReason: "Weak robustness",
        }));
      }
    }

    if (
      candidate.family.startsWith("fundamental") &&
      (failedNames.has("LOW_SUB_UNIVERSE_SHARPE") || nonPassNames.has("SELF_CORRELATION")) &&
      !candidate.expression.includes("subindustry")
    ) {
      if (plan.generatorStrategy === "diversity-v2") {
        mutations.push(makeMutation(candidate, {
          suffix: "self_corr_balance_sheet_escape",
          hypothesis: `${candidate.hypothesis} Self-correlation risk requires a material mechanism change into balance-sheet structure.`,
          expression: "group_rank(-rank(liabilities / assets), industry)",
          settings: { ...candidate.settings, neutralization: "INDUSTRY", decay: 2, truncation: 0.05 },
          signature: [...candidate.signature, "cross-family", "liabilities", "assets", "self-correlation"],
          dataFamily: "FUNDAMENTAL_VALUE",
          settingPolicy: "fundamental-industry-peer",
          mutationReason: "Self-correlation risk; force data-family escape",
        }));
      }
      mutations.push(makeMutation(candidate, {
        suffix: "subindustry_escape",
        hypothesis: `${candidate.hypothesis} Move to subindustry-relative comparison to address crowding or sub-universe instability.`,
        expression: "group_rank(ts_rank(operating_income / assets, 252), subindustry)",
        settings: { ...candidate.settings, neutralization: "SUBINDUSTRY" },
        signature: [...candidate.signature, "group_rank", "subindustry", "crowding-escape"],
        mutationReason: "Crowding or sub-universe stability risk",
      }));
    }

    if (stats.turnover !== null && stats.turnover > 0.2) {
      mutations.push(makeMutation(candidate, {
        suffix: "decay4",
        hypothesis: `${candidate.hypothesis} Smooth the signal to reduce trading noise.`,
        settings: { ...candidate.settings, decay: Math.max(candidate.settings.decay ?? 0, 4) },
        signature: [...candidate.signature, "decay4"],
        mutationReason: "High turnover",
      }));
    }

    if (
      candidate.family.startsWith("fundamental") &&
      stats.isSharpe !== null &&
      stats.isSharpe < 1 &&
      !candidate.expression.includes("/ close")
    ) {
      mutations.push(makeMutation(candidate, {
        suffix: "asset_price_blend",
        hypothesis: `${candidate.hypothesis} Add price scaling only as a secondary channel instead of abandoning the original robust denominator.`,
        expression: "rank(ts_rank(operating_income / assets, 252) + ts_rank(operating_income / close, 252))",
        signature: [...candidate.signature, "assets", "close-scale", "blend"],
        mutationReason: "Weak strength",
      }));
    }
  }

  const recombined = buildRecombinations(promising, memory);
  return diversifyCandidates(dedupeBySignature([...mutations, ...recombined]), limit);
}

export function choosePromotionBucket(candidate) {
  const metrics = candidate.metrics ?? {};
  const checks = candidate.checks ?? [];
  const visibleFailures = checks.filter((check) => check.result === "FAIL").map((check) => check.name);
  const nonPassChecks = checks.filter((check) => check.result !== "PASS");

  const strongEnough =
    numberAtLeast(metrics.isSharpe, 1.25) &&
    numberAtLeast(metrics.isFitness, 1.0) &&
    numberBetween(metrics.turnover, 0.01, 0.7);
  const robustEnough = numberAtLeast(metrics.testSharpe, 0.01);
  const checksReady = checks.length > 0 && nonPassChecks.length === 0;

  if (strongEnough && robustEnough && checksReady) {
    return "effective";
  }
  if (metrics.isSharpe !== null && metrics.isSharpe < 0.5) {
    return "deprecated";
  }
  if (visibleFailures.includes("LOW_SUB_UNIVERSE_SHARPE") && metrics.testSharpe !== null && metrics.testSharpe < 0) {
    return "deprecated";
  }
  return "watch";
}

function orderTemplatesForPlan(templates, plan) {
  if (!plan.prioritizeRobustTest && !plan.avoidCrowding) return templates;
  const preferredIds = [
    "income_asset_scale",
    "income_asset_peer",
    "income_asset_subindustry_peer",
    "income_asset_history_blend",
    "income_asset_price_blend",
    "income_asset_price_peer_blend",
    "income_history",
    "income_price_peer",
    "income_price_scale",
  ];
  const weight = new Map(preferredIds.map((id, index) => [id, index]));
  return [...templates].sort((left, right) => (weight.get(left.id) ?? 100) - (weight.get(right.id) ?? 100));
}

function avoidDeprecatedSignatures(candidates, memory) {
  const signaturesToAvoid = new Set(
    (memory?.deprecatedPool ?? []).map((candidate) => signatureKey(candidate.signature ?? [])),
  );
  const expressionsToAvoid = new Set((memory?.deprecatedPool ?? []).map((candidate) => candidate.expression));
  return candidates.filter((candidate) => {
    if (expressionsToAvoid.has(candidate.expression)) return false;
    return !signaturesToAvoid.has(signatureKey(candidate.signature ?? []));
  });
}

export function materializeCandidate(template, extra = {}) {
  const timestamp = Date.now().toString(36);
  const expression = template.expression;
  const settings = { ...DEFAULT_SETTINGS, ...(template.settings ?? {}), ...(extra.settings ?? {}) };
  return {
    id: `${template.id}_${timestamp}`,
    templateId: template.id,
    family: template.family,
    dataFamily: template.dataFamily ?? inferDataFamily(expression),
    horizon: template.horizon,
    hypothesis: template.hypothesis,
    expression,
    settings,
    signature: [...(template.signature ?? [])],
    settingPolicy: template.settingPolicy ?? inferSettingPolicy(expression, settings),
    strategy: extra.strategy ?? DEFAULT_GENERATOR_STRATEGY,
    experimental: Boolean(template.experimental ?? extra.experimental),
    operatorPattern: extractOperatorPattern(expression),
    trajectoryId: extra.trajectoryId ?? `${template.id}-root`,
    parentIds: extra.parentIds ?? [],
    agentNotes: extra.agentNotes ?? [],
  };
}

function selectParents(memory, plan, limit) {
  const pool = (memory?.effectivePool ?? [])
    .filter((candidate) => plan.preferredFamilies.includes(candidate.family))
    .sort((left, right) => (right.totalScore ?? -Infinity) - (left.totalScore ?? -Infinity));

  return diversifyCandidates(pool, limit).map((candidate) => ({
    ...candidate,
    id: `${candidate.id}_reuse_${Date.now().toString(36)}`,
    parentIds: [...(candidate.parentIds ?? []), candidate.id],
    agentNotes: [...(candidate.agentNotes ?? []), "Reused by CrossSampleSelectorAgent"],
  }));
}

function buildRecombinations(promising, memory) {
  if (promising.length < 2) return [];
  const candidates = [];

  for (let index = 0; index < promising.length - 1; index += 1) {
    const left = promising[index];
    const right = promising[index + 1];
    if (
      left.family.startsWith("fundamental") &&
      right.family.startsWith("fundamental") &&
      left.expression !== right.expression
    ) {
      candidates.push({
        ...left,
        id: `${left.id}_x_${right.id}`,
        hypothesis: `${left.hypothesis} Combine it with ${right.hypothesis.toLowerCase()}`,
        expression: `rank(${left.expression} + ${right.expression})`,
        signature: dedupeArray([...left.signature, ...right.signature, "recombine"]),
        parentIds: [left.id, right.id],
        agentNotes: [...(left.agentNotes ?? []), "Recombined by OptimizerAgent"],
        mutationReason: "Trajectory recombination",
      });
    }
  }

  const signaturesToAvoid = new Set(
    (memory?.deprecatedPool ?? []).map((candidate) => signatureKey(candidate.signature ?? [])),
  );
  return candidates.filter((candidate) => !signaturesToAvoid.has(signatureKey(candidate.signature)));
}

function makeMutation(candidate, patch) {
  const expression = patch.expression ?? candidate.expression;
  const settings = patch.settings ?? { ...candidate.settings };
  return {
    ...candidate,
    id: `${candidate.id}_${patch.suffix}`,
    hypothesis: patch.hypothesis ?? candidate.hypothesis,
    expression,
    settings,
    signature: dedupeArray(patch.signature ?? [...candidate.signature]),
    dataFamily: patch.dataFamily ?? inferDataFamily(expression),
    settingPolicy: patch.settingPolicy ?? inferSettingPolicy(expression, settings),
    operatorPattern: extractOperatorPattern(expression),
    trajectoryId: candidate.trajectoryId,
    parentIds: [...(candidate.parentIds ?? []), candidate.id],
    agentNotes: [...(candidate.agentNotes ?? []), patch.mutationReason ?? "Mutated"],
    mutationReason: patch.mutationReason ?? null,
  };
}

function dedupeBySignature(candidates) {
  const seen = new Set();
  return candidates.filter((candidate) => {
    const key = `${candidate.expression}::${signatureKey(candidate.signature ?? [])}::${JSON.stringify(candidate.settings)}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function diversifyCandidates(candidates, limit) {
  const selected = [];
  const usedFamilies = new Set();
  const usedSignatures = new Set();

  for (const candidate of candidates) {
    const family = candidate.family;
    const signature = signatureKey(candidate.signature ?? []);
    if (!usedFamilies.has(family) && !usedSignatures.has(signature)) {
      selected.push(candidate);
      usedFamilies.add(family);
      usedSignatures.add(signature);
    }
    if (selected.length >= limit) return selected;
  }

  for (const candidate of candidates) {
    if (selected.length >= limit) break;
    if (selected.find((item) => item.id === candidate.id)) continue;
    selected.push(candidate);
  }

  return selected.slice(0, limit);
}

function diversifyCandidatesByDataFamily(candidates, limit, memory) {
  const selected = [];
  const usedFamilies = new Set();
  const ordered = [...candidates].sort((left, right) => {
    const familyDiff = dataFamilyUsage(left.dataFamily, memory) - dataFamilyUsage(right.dataFamily, memory);
    if (familyDiff !== 0) return familyDiff;
    return DATA_FAMILY_ORDER.indexOf(left.dataFamily) - DATA_FAMILY_ORDER.indexOf(right.dataFamily);
  });

  for (const candidate of ordered) {
    if (selected.length >= limit) break;
    if (usedFamilies.has(candidate.dataFamily)) continue;
    selected.push(candidate);
    usedFamilies.add(candidate.dataFamily);
  }

  for (const candidate of ordered) {
    if (selected.length >= limit) break;
    if (selected.find((item) => item.expression === candidate.expression)) continue;
    selected.push(candidate);
  }

  return selected.slice(0, limit);
}

function enabledTemplates(plan) {
  return TEMPLATE_LIBRARY.filter((template) => plan.experimentalFieldsEnabled || !template.experimental);
}

function templateDiversityWeight(template, memory, plan) {
  const familyIndex = DATA_FAMILY_ORDER.indexOf(template.dataFamily);
  const familyPenalty = dataFamilyUsage(template.dataFamily, memory) * 10;
  const pattern = extractOperatorPattern(template.expression);
  const patternPenalty = operatorPatternUsage(pattern, memory) >= plan.crowdingPatternThreshold ? 100 : 0;
  const objectivePenalty = plan.preferredFamilies.includes(template.family) ? 0 : 25;
  return familyPenalty + patternPenalty + objectivePenalty + (familyIndex < 0 ? 99 : familyIndex);
}

function dataFamilyUsage(dataFamily, memory) {
  return (memory?.trajectories ?? []).filter((candidate) => candidate.dataFamily === dataFamily).length;
}

function operatorPatternUsage(pattern, memory) {
  const pool = [...(memory?.trajectories ?? []), ...(memory?.effectivePool ?? []), ...(memory?.deprecatedPool ?? [])];
  return pool.filter((candidate) => (candidate.operatorPattern ?? extractOperatorPattern(candidate.expression)) === pattern).length;
}

function preflightCandidate(candidate, history, plan, currentPatterns, currentFamilies) {
  const reasons = [];
  const expression = String(candidate.expression ?? "");
  const fields = extractDataFields(expression);
  const operators = extractOperators(expression);
  const pattern = candidate.operatorPattern ?? extractOperatorPattern(expression);
  const historicalPatternCount = history.filter((item) => (item.operatorPattern ?? extractOperatorPattern(item.expression)) === pattern).length;
  const currentPatternCount = currentPatterns.get(pattern) ?? 0;
  const familyStreak = recentFamilyStreak(history);
  const currentFamilyCount = currentFamilies.get(candidate.dataFamily) ?? 0;
  const historicalExpressions = new Set(history.map((item) => item.expression).filter(Boolean));

  if (plan.generatorStrategy !== "diversity-v2") {
    return {
      ok: true,
      reasons: [],
      primaryReason: null,
      dataFields: fields,
      operators,
      operatorPattern: pattern,
      historicalPatternCount,
      batchPatternCount: currentPatternCount,
      batchFamilyCount: currentFamilyCount,
      familyStreak,
      crowdingPenalty: 0,
    };
  }

  if (candidate.experimental && !plan.experimentalFieldsEnabled) reasons.push("experimental-fields-disabled");
  if (historicalExpressions.has(expression)) reasons.push("duplicate-expression");
  if (fields.length > 3) reasons.push("too-many-data-fields");
  if (maxDepth(expression) > 5) reasons.push("expression-too-deep");
  if (operators.some((operator) => !ALLOWED_OPERATORS.has(operator))) reasons.push("unknown-operator");
  if (historicalPatternCount + currentPatternCount >= plan.crowdingPatternThreshold) reasons.push("crowded-operator-pattern");
  if (
    familyStreak.family === candidate.dataFamily &&
    familyStreak.count >= plan.familyCooldownRounds &&
    plan.generatorStrategy === "diversity-v2"
  ) {
    reasons.push("data-family-cooldown");
  }
  if (plan.generatorStrategy === "diversity-v2" && currentFamilyCount >= 2) {
    reasons.push("batch-family-overuse");
  }

  const ok = reasons.length === 0;
  return {
    ok,
    reasons,
    primaryReason: reasons[0] ?? null,
    dataFields: fields,
    operators,
    operatorPattern: pattern,
    historicalPatternCount,
    batchPatternCount: currentPatternCount,
    batchFamilyCount: currentFamilyCount,
    familyStreak,
    crowdingPenalty: round4(Math.min(1, (historicalPatternCount + currentPatternCount) / Math.max(1, plan.crowdingPatternThreshold))),
  };
}

const ALLOWED_OPERATORS = new Set([
  "rank",
  "zscore",
  "winsorize",
  "group_rank",
  "group_neutralize",
  "group_vector_neut",
  "vector_neut",
  "ts_rank",
  "ts_delta",
  "ts_mean",
  "ts_sum",
  "ts_std_dev",
  "ts_zscore",
  "ts_decay_linear",
  "ts_decay_exp_window",
  "ts_backfill",
  "ts_av_diff",
  "ts_product",
  "ts_delay",
  "ts_regression",
  "trade_when",
  "power",
  "vec_sum",
  "vec_avg",
]);

const GROUP_NAMES = new Set(["industry", "subindustry", "sector", "market", "exchange"]);
const RESERVED_TOKENS = new Set(["if", "else", "true", "false", "nan"]);

function extractOperators(expression) {
  return [...new Set(String(expression ?? "").match(/\b[a-zA-Z_][a-zA-Z0-9_]*\s*\(/g)?.map((item) => item.replace(/\s*\($/, "")) ?? [])];
}

function extractDataFields(expression) {
  const operators = new Set(extractOperators(expression));
  const tokens = String(expression ?? "").match(/\b[a-zA-Z_][a-zA-Z0-9_]*\b/g) ?? [];
  return [...new Set(tokens.filter((token) =>
    !operators.has(token) &&
    !GROUP_NAMES.has(token) &&
    !RESERVED_TOKENS.has(token) &&
    Number.isNaN(Number(token)),
  ))];
}

export function extractOperatorPattern(expression) {
  const operators = new Set(extractOperators(expression));
  return String(expression ?? "").replace(/\b[a-zA-Z_][a-zA-Z0-9_]*\b/g, (token) => {
    if (operators.has(token)) return token;
    if (GROUP_NAMES.has(token)) return token;
    if (RESERVED_TOKENS.has(token)) return token;
    return Number.isNaN(Number(token)) ? "F" : token;
  });
}

function maxDepth(expression) {
  let depth = 0;
  let max = 0;
  for (const char of String(expression ?? "")) {
    if (char === "(") {
      depth += 1;
      max = Math.max(max, depth);
    } else if (char === ")") {
      depth = Math.max(0, depth - 1);
    }
  }
  return max;
}

function recentFamilyStreak(history) {
  const recent = [...history].filter((item) => item.dataFamily).slice(-20).reverse();
  const family = recent[0]?.dataFamily ?? null;
  if (!family) return { family: null, count: 0 };
  let count = 0;
  for (const item of recent) {
    if (item.dataFamily !== family) break;
    count += 1;
  }
  return { family, count };
}

function countBy(items, getKey) {
  const counts = {};
  for (const item of items) {
    const key = getKey(item);
    counts[key] = (counts[key] ?? 0) + 1;
  }
  return counts;
}

function mergeCounts(...countObjects) {
  const merged = {};
  for (const counts of countObjects) {
    for (const [key, value] of Object.entries(counts)) {
      merged[key] = (merged[key] ?? 0) + value;
    }
  }
  return merged;
}

function signatureKey(signature) {
  return dedupeArray(signature).sort().join("|");
}

function dedupeArray(items) {
  return [...new Set(items)];
}

function tokenize(expression) {
  return String(expression ?? "").toLowerCase().split(/[^a-z0-9_]+/g).filter(Boolean);
}

function numberAtLeast(value, limit) {
  return value !== null && value >= limit;
}

function numberBetween(value, min, max) {
  return value !== null && value >= min && value <= max;
}

function numberBelow(value, limit) {
  return value !== null && value < limit;
}

function round4(value) {
  return Number.isFinite(value) ? Number(value.toFixed(4)) : null;
}

function clampInt(value, defaultValue, low, high) {
  const number = Number(value);
  if (!Number.isFinite(number)) return defaultValue;
  return Math.max(low, Math.min(high, Math.round(number)));
}

function parseBoolean(value, defaultValue = false) {
  if (value === undefined || value === null || value === "") return defaultValue;
  if (typeof value === "boolean") return value;
  return ["1", "true", "yes", "on"].includes(String(value).toLowerCase());
}
