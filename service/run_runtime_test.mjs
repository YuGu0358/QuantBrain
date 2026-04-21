import assert from "node:assert/strict";

import { finalizeRunExit } from "./run_runtime.mjs";

const activeRuns = new Map();
const state = {
  runId: "scheduled-mining-2026-04-21T12-49-43Z",
  status: "running",
};

activeRuns.set(state.runId, state);

finalizeRunExit(activeRuns, state, 0, "2026-04-21T12:54:02.000Z");

assert.equal(state.status, "completed");
assert.equal(state.exitCode, 0);
assert.equal(state.finishedAt, "2026-04-21T12:54:02.000Z");
assert.equal(activeRuns.has(state.runId), false);

console.log("run runtime smoke passed");
