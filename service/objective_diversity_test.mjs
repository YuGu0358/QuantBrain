import assert from "node:assert/strict";
import {
  buildTemplateObjective,
  normalizeObjectiveSignature,
  objectiveNeedsDiversification,
  pickNextTarget,
} from "./objective_diversity.mjs";

const categoryFields = {
  QUALITY: ["operating_income", "cashflow_op"],
  MOMENTUM: ["returns", "volume"],
  REVERSAL: ["returns", "close"],
};

const history = [
  { category: "QUALITY", field: "operating_income", objective: "Discover robust operating_income quality signals with low crowding and positive test stability" },
  { category: "QUALITY", field: "cashflow_op", objective: "Find cashflow_op quality signals with peer stability and low crowding" },
  { category: "MOMENTUM", field: "returns", objective: "Identify returns momentum patterns with sector neutralization" },
];

const target = pickNextTarget(history, categoryFields, () => 0);
assert.deepEqual(target, { category: "REVERSAL", field: "returns" });

const templateObjective = buildTemplateObjective("QUALITY", "operating_income", 0);
assert.match(templateObjective, /operating[_ ]income/i);

const repeatedFamily = "Mine robust operating income quality signals with orthogonal crowding control and positive test stability";
assert.equal(objectiveNeedsDiversification(repeatedFamily, history), true);

const differentFamily = "Explore returns-based reversal dislocations with fast mean reversion confirmation";
assert.equal(objectiveNeedsDiversification(differentFamily, history), false);

const signature = normalizeObjectiveSignature(
  "Discover robust operating-income quality signals with low crowding and positive test stability",
);
assert.deepEqual(signature, {
  category: "QUALITY",
  field: "operating_income",
  themes: ["crowding", "robustness", "test_stability"],
});

console.log("objective diversity smoke passed");
