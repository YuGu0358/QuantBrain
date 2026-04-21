const DEFAULT_CATEGORY_FIELDS = {
  QUALITY: ["operating_income", "cashflow_op", "assets", "est_eps", "returns"],
  MOMENTUM: ["returns", "close", "vwap", "volume", "adv20"],
  REVERSAL: ["returns", "close", "open", "high", "low", "vwap"],
  VOLATILITY: ["returns", "high", "low", "close", "adv20", "volume"],
  LIQUIDITY: ["volume", "adv20", "vwap", "close", "returns"],
  MICROSTRUCTURE: ["volume", "adv20", "high", "low", "open", "close", "vwap"],
  SENTIMENT: ["news_sentiment", "est_eps", "returns", "volume", "operating_income"],
};

const FIELD_ALIASES = new Map([
  ["operating_income", ["operating income", "operating_income"]],
  ["cashflow_op", ["cashflow_op", "cash flow", "cash-flow"]],
  ["assets", ["assets", "asset"]],
  ["est_eps", ["est_eps", "eps", "estimate", "analyst"]],
  ["returns", ["returns", "return", "price action"]],
  ["close", ["close", "closing price"]],
  ["open", ["open", "opening price"]],
  ["high", ["high price", "daily high"]],
  ["low", ["low price", "daily low"]],
  ["vwap", ["vwap"]],
  ["volume", ["volume"]],
  ["adv20", ["adv20", "average daily volume", "liquidity"]],
  ["news_sentiment", ["news sentiment", "sentiment", "news"]],
]);

const CATEGORY_KEYWORDS = {
  QUALITY: ["quality", "profitability", "fundamental", "earnings quality"],
  MOMENTUM: ["momentum", "trend", "continuation"],
  REVERSAL: ["reversal", "mean reversion", "oversold", "dislocation"],
  VOLATILITY: ["volatility", "dispersion", "risk"],
  LIQUIDITY: ["liquidity", "participation"],
  MICROSTRUCTURE: ["microstructure", "intraday", "vwap", "order flow"],
  SENTIMENT: ["sentiment", "news", "attention", "earnings surprise"],
};

const THEME_KEYWORDS = {
  crowding: ["crowding", "orthogonal", "divers"],
  robustness: ["robust", "stability", "resilient"],
  test_stability: ["test stability", "out-of-sample", "oos", "positive test", "submit", "submission"],
  neutralization: ["neutral", "peer", "industry", "sector"],
  speed: ["short", "fast", "intraday", "next-day"],
};

const OBJECTIVE_VARIANTS = {
  QUALITY: [
    "{fieldLabel}-based quality signals with low crowding and positive test stability",
    "peer-relative quality dislocations anchored on {fieldLabel} with robust out-of-sample behavior",
    "{fieldLabel}-driven quality improvement signals that stay orthogonal to crowded profitability templates",
  ],
  MOMENTUM: [
    "short-to-medium horizon momentum patterns driven by {fieldLabel} with sector-neutral confirmation",
    "{fieldLabel}-led continuation signals that avoid crowded trend-following templates",
    "robust momentum regimes where {fieldLabel} confirms persistent cross-sectional strength",
  ],
  REVERSAL: [
    "{fieldLabel}-based reversal dislocations with fast mean-reversion confirmation",
    "oversold/overbought reversion patterns anchored on {fieldLabel} without reusing crowded reversal templates",
    "{fieldLabel}-driven snapback signals that keep turnover controlled and test stability positive",
  ],
  VOLATILITY: [
    "volatility-adjusted {fieldLabel} signals that predict cross-sectional dispersion",
    "{fieldLabel}-conditioned risk regimes with robust out-of-sample stability",
    "dispersion-sensitive {fieldLabel} alphas that remain orthogonal to crowded risk templates",
  ],
  LIQUIDITY: [
    "{fieldLabel}-driven liquidity-premium signals with stable risk-adjusted returns",
    "{fieldLabel}-based participation regimes that stay robust under crowding pressure",
    "robust liquidity dislocations where {fieldLabel} captures improving market depth",
  ],
  MICROSTRUCTURE: [
    "intraday microstructure patterns in {fieldLabel} that predict next-day returns",
    "{fieldLabel}-based order-flow pressure signals with orthogonal short-horizon edge",
    "robust microstructure dislocations where {fieldLabel} captures execution imbalance",
  ],
  SENTIMENT: [
    "sentiment-driven mispricings using {fieldLabel} with earnings confirmation",
    "{fieldLabel}-based news reactions that stay robust beyond crowded attention spikes",
    "orthogonal sentiment regimes where {fieldLabel} confirms post-news repricing",
  ],
};

function canonicalFieldLabel(field) {
  return String(field ?? "").replace(/_/g, " ");
}

function chooseLeastUsed(items, getCount, rng = Math.random) {
  const sorted = [...items].sort((left, right) => getCount(left) - getCount(right));
  const minCount = getCount(sorted[0]);
  const tied = sorted.filter((item) => getCount(item) === minCount);
  return tied[Math.floor(rng() * tied.length)];
}

export function pickNextTarget(history = [], categoryFields = DEFAULT_CATEGORY_FIELDS, rng = Math.random) {
  const categories = Object.keys(categoryFields);
  const recent = history.slice(-20);
  const categoryCounts = new Map(categories.map((category) => [category, 0]));
  for (const entry of recent) {
    if (entry?.category && categoryCounts.has(entry.category)) {
      categoryCounts.set(entry.category, (categoryCounts.get(entry.category) ?? 0) + 1);
    }
  }
  const category = chooseLeastUsed(categories, (item) => categoryCounts.get(item) ?? 0, rng);
  const fields = categoryFields[category] ?? [];
  const fieldCounts = new Map(fields.map((field) => [field, 0]));
  for (const entry of recent) {
    if (entry?.category === category && fieldCounts.has(entry.field)) {
      fieldCounts.set(entry.field, (fieldCounts.get(entry.field) ?? 0) + 1);
    }
  }
  const lastField = [...history].reverse().find((entry) => entry?.category === category)?.field;
  const field = chooseLeastUsed(
    fields,
    (item) => (fieldCounts.get(item) ?? 0) + (item === lastField ? 0.25 : 0),
    rng,
  );
  return { category, field };
}

export function pickTemplateVariantIndex(category, field, history = [], rng = Math.random) {
  const variants = OBJECTIVE_VARIANTS[category] ?? ["robust {fieldLabel} alpha objective"];
  const recent = history.slice(-20);
  const variantCounts = variants.map((_, index) => {
    const candidate = buildTemplateObjective(category, field, index);
    return recent.filter((entry) => {
      const text = typeof entry === "string" ? entry : entry?.objective;
      return signaturesOverlap(normalizeObjectiveSignature(candidate), normalizeObjectiveSignature(text));
    }).length;
  });
  return chooseLeastUsed(
    variants.map((_, index) => index),
    (index) => variantCounts[index] ?? 0,
    rng,
  );
}

export function buildTemplateObjective(category, field, variantIndex = 0) {
  const variants = OBJECTIVE_VARIANTS[category] ?? ["robust {fieldLabel} alpha objective"];
  const template = variants[Math.max(0, Math.min(Number(variantIndex) || 0, variants.length - 1))];
  const fieldLabel = canonicalFieldLabel(field);
  return template.replaceAll("{fieldLabel}", fieldLabel);
}

export function normalizeObjectiveSignature(text, categoryFields = DEFAULT_CATEGORY_FIELDS) {
  const normalized = String(text ?? "").toLowerCase().replace(/\s+/g, " ").trim();
  const field = detectField(normalized, categoryFields);
  const category = detectCategory(normalized, field, categoryFields);
  const themes = Object.entries(THEME_KEYWORDS)
    .filter(([, keywords]) => keywords.some((keyword) => normalized.includes(keyword)))
    .map(([theme]) => theme)
    .sort();
  return {
    category,
    field,
    themes,
  };
}

export function objectiveNeedsDiversification(candidate, recentObjectives = [], categoryFields = DEFAULT_CATEGORY_FIELDS) {
  const candidateSignature = normalizeObjectiveSignature(candidate, categoryFields);
  if (!candidateSignature.category && !candidateSignature.field) return recentObjectives.length > 0;
  return recentObjectives.some((item) =>
    signaturesOverlap(candidateSignature, normalizeObjectiveSignature(typeof item === "string" ? item : item?.objective, categoryFields)),
  );
}

function detectField(text, categoryFields) {
  const aliasEntries = [...FIELD_ALIASES.entries()]
    .flatMap(([field, aliases]) => aliases.map((alias) => ({ field, alias })))
    .sort((left, right) => right.alias.length - left.alias.length);
  for (const { field, alias } of aliasEntries) {
    const escaped = alias
      .replace(/[.*+?^${}()|[\]\\]/g, "\\$&")
      .replace(/[\s_]+/g, "[-_\\s]+");
    if (new RegExp(`(^|[^a-z0-9_])${escaped}([^a-z0-9_]|$)`, "i").test(text)) return field;
  }
  for (const fields of Object.values(categoryFields)) {
    for (const field of fields) {
      if (field.length <= 4 && !field.includes("_") && /\D/.test(field)) continue;
      const escaped = field
        .replace(/_/g, " ")
        .replace(/[.*+?^${}()|[\]\\]/g, "\\$&")
        .replace(/\s+/g, "[-_\\s]+");
      if (new RegExp(`(^|[^a-z0-9_])${escaped}([^a-z0-9_]|$)`, "i").test(text)) return field;
    }
  }
  return null;
}

function detectCategory(text, field, categoryFields) {
  for (const [category, keywords] of Object.entries(CATEGORY_KEYWORDS)) {
    if (keywords.some((keyword) => text.includes(keyword))) return category;
  }
  if (field) {
    for (const [category, fields] of Object.entries(categoryFields)) {
      if (fields.includes(field)) return category;
    }
  }
  return null;
}

function signaturesOverlap(left, right) {
  if (!left || !right) return false;
  if (left.category && right.category && left.category !== right.category) return false;
  if (left.field && right.field && left.field !== right.field) return false;
  const sharedThemes = left.themes.filter((theme) => right.themes.includes(theme));
  return Boolean(left.category || left.field) && sharedThemes.length >= Math.min(2, Math.max(left.themes.length, 1));
}
