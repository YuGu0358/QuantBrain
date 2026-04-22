import assert from "node:assert/strict";

import {
  flattenProviderEntries,
  formatProviderWinRate,
  summarizeDiagnoseProvider,
  summarizeRetrievalStatus,
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

assert.deepEqual(
  summarizeDiagnoseProvider(flattened, [
    {
      runId: "repair-002",
      summary: {
        diagnosis: {
          llm_fallback: true,
          primary_error: "Invalid primary_symptom: 'sharpe_low'",
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
    lastFailureReason: "Invalid primary_symptom: 'sharpe_low'",
    lastFailureRunId: "repair-002",
  },
);

assert.deepEqual(
  summarizeRetrievalStatus([
    {
      runId: "loop-003",
      summary: {
        kb_retrieval_mode: "uninitialized",
        kb_embedder_status: "disabled_no_openai_key",
        repair_retrieval_mode: "not_applicable",
      },
    },
    {
      runId: "loop-002",
      summary: {
        kb_retrieval_mode: "semantic",
        kb_retrieval_error: null,
        kb_embedder_status: "ready",
      },
    },
    {
      runId: "repair-001",
      summary: {
        repair_retrieval_mode: "disabled_no_embedder",
        repair_retrieval_error: "missing_openai_key",
        repair_semantic_memory_status: "missing_openai_key",
      },
    },
  ]),
  {
    generation: {
      runId: "loop-002",
      mode: "semantic",
      error: null,
      embedderStatus: "ready",
    },
    repair: {
      runId: "repair-001",
      mode: "disabled_no_embedder",
      error: "missing_openai_key",
      semanticMemoryStatus: "missing_openai_key",
    },
  },
);

console.log("llm router dashboard smoke passed");
