import assert from "node:assert/strict";
import {
  applyPreflight,
  buildPlan,
  buildRepairBatch,
  buildSeedBatch,
  summarizeDiversity,
} from "./agentic_alpha_library.mjs";

const emptyMemory = { effectivePool: [], deprecatedPool: [], trajectories: [], objectiveHistory: [] };

const plan = buildPlan("robust operating-income quality with low crowding and positive test stability", {
  generatorStrategy: "diversity-v2",
  experimentalFields: false,
  batchSize: 7,
});
const seedBatch = applyPreflight(buildSeedBatch(plan, emptyMemory, 7), emptyMemory, plan, { phase: "test" });
const seedStats = summarizeDiversity(seedBatch, emptyMemory, plan);

assert.equal(seedStats.strategy, "diversity-v2");
assert.ok(Object.keys(seedStats.familyCounts).length >= 3, "diversity-v2 should cover at least three data families");
assert.equal(seedStats.familyCounts.SENTIMENT, undefined, "sentiment templates must stay disabled by default");
assert.equal(seedStats.preflightRejected, 0, "fresh diverse seed batch should pass preflight");

const crowdedMemory = {
  ...emptyMemory,
  trajectories: [
    { expression: "ts_rank(operating_income / assets, 252)", dataFamily: "FUNDAMENTAL_QUALITY" },
    { expression: "ts_rank(cashflow_op / assets, 252)", dataFamily: "FUNDAMENTAL_QUALITY" },
  ],
};
const crowdedPlan = buildPlan("robust quality with low crowding", {
  generatorStrategy: "diversity-v2",
  crowdingPatternThreshold: 2,
});
const crowdedBatch = applyPreflight([
  {
    id: "crowded-direct",
    family: "fundamental-efficiency",
    dataFamily: "FUNDAMENTAL_QUALITY",
    horizon: "slow",
    hypothesis: "Direct crowded-pattern test.",
    expression: "ts_rank(operating_income / close, 252)",
    settings: {},
    signature: ["operating_income", "close", "ts_rank", "252"],
  },
], crowdedMemory, crowdedPlan, { phase: "test" });
assert.ok(
  crowdedBatch.some((candidate) => candidate.preflightStatus?.reasons?.includes("crowded-operator-pattern")),
  "crowded operator skeletons should be blocked before BRAIN simulation",
);

const repeatedFieldBundleBatch = applyPreflight([
  {
    id: "field-bundle-a",
    family: "fundamental-efficiency",
    dataFamily: "FUNDAMENTAL_QUALITY",
    horizon: "slow",
    hypothesis: "First use of operating_income/assets bundle.",
    expression: "rank(ts_mean(operating_income / assets, 63))",
    settings: {},
    signature: ["operating_income", "assets", "ts_mean", "63"],
  },
  {
    id: "field-bundle-b",
    family: "fundamental-quality",
    dataFamily: "FUNDAMENTAL_QUALITY",
    horizon: "slow",
    hypothesis: "Second use of the same operating_income/assets bundle with a different operator.",
    expression: "group_rank(ts_delta(operating_income / assets, 126), industry)",
    settings: {},
    signature: ["operating_income", "assets", "ts_delta", "126", "group_rank", "industry"],
  },
], emptyMemory, crowdedPlan, { phase: "test" });
assert.ok(
  repeatedFieldBundleBatch[1]?.preflightStatus?.reasons?.includes("field-bundle-overuse"),
  "diversity-v2 should block repeated field bundles inside the same batch",
);

const sentimentOff = buildPlan("sentiment news attention alpha", {
  generatorStrategy: "diversity-v2",
  experimentalFields: false,
});
const sentimentOffBatch = buildSeedBatch(sentimentOff, emptyMemory, 5);
assert.ok(sentimentOffBatch.every((candidate) => candidate.dataFamily !== "SENTIMENT"));

const sentimentOn = buildPlan("sentiment news attention alpha", {
  generatorStrategy: "diversity-v2",
  experimentalFields: true,
});
const sentimentOnBatch = buildSeedBatch(sentimentOn, emptyMemory, 5);
assert.ok(sentimentOnBatch.some((candidate) => candidate.dataFamily === "SENTIMENT"));

const repairContext = {
  parentAlphaId: "78KWVrb1",
  rootAlphaId: "78KWVrb1",
  expression: "ts_rank(operating_income / assets, 252)",
  failedChecks: ["LOW_SHARPE", "LOW_FITNESS", "SELF_CORRELATION"],
  repairDepth: 0,
  nextAction: "strength and crowding repair",
  alpha: {
    id: "78KWVrb1",
    expression: "ts_rank(operating_income / assets, 252)",
    isSharpe: 0.92,
    isFitness: 0.5,
    turnover: 0.0488,
    testSharpe: 0.29,
    testFitness: 0.09,
  },
};
const repairBatch = applyPreflight(buildRepairBatch(repairContext, plan, emptyMemory, 5), emptyMemory, plan, { phase: "repair" });
assert.equal(repairBatch.length, 5);
assert.ok(repairBatch.every((candidate) => candidate.expression !== repairContext.expression));
assert.ok(repairBatch.every((candidate) => !(candidate.agentNotes ?? []).some((note) => note.includes("Fallback seed"))));
assert.ok(new Set(repairBatch.map((candidate) => candidate.dataFamily)).size >= 3);

console.log("diversity-v2 smoke passed");
