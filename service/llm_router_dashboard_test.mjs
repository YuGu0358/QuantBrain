import assert from "node:assert/strict";

import {
  flattenProviderEntries,
  formatProviderWinRate,
  summarizeDiagnoseProvider,
} from "./llm_router_dashboard.mjs";

const flattened = flattenProviderEntries({
  gpt_diagnose: {
    diagnose: {
      name: "gpt_diagnose",
      role: "diagnose",
      win_rate: 0,
      calls: 1,
      wins: 0,
    },
  },
  gpt_repair: {
    repair: {
      name: "gpt_repair",
      role: "repair",
      win_rate: 1,
      calls: 4,
      wins: 4,
    },
  },
});

assert.equal(flattened.length, 2);

assert.deepEqual(
  formatProviderWinRate({ win_rate: 0, calls: 1, wins: 0 }),
  {
    label: "N/A",
    width: 0,
    colorToken: "var(--t3)",
    detail: "样本不足（1 次）",
    lowSample: true,
  },
);

assert.deepEqual(
  summarizeDiagnoseProvider(flattened, [
    {
      runId: "repair-001",
      summary: {
        diagnosis: {
          fallback: true,
          error: "diagnose timeout",
        },
      },
    },
  ]),
  {
    name: "gpt_diagnose",
    role: "diagnose",
    calls: 1,
    wins: 0,
    lowSample: true,
    rateLabel: "N/A",
    rateWidth: 0,
    rateColorToken: "var(--t3)",
    rateDetail: "样本不足（1 次）",
    lastFailureReason: "diagnose timeout",
    lastFailureRunId: "repair-001",
  },
);

console.log("llm router dashboard smoke passed");
