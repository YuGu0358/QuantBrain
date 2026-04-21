import assert from "node:assert/strict";

import {
  describeRunLiveness,
  isRunStalled,
} from "./run_liveness.mjs";

const now = Date.parse("2026-04-21T09:45:00.000Z");
const stallMs = 10 * 60 * 1000;

assert.equal(
  isRunStalled(
    {
      startedAt: "2026-04-21T09:20:00.000Z",
      progressStats: { lastEventAt: "2026-04-21T09:25:00.000Z" },
    },
    { now, stallMs },
  ),
  true,
);

assert.equal(
  isRunStalled(
    {
      startedAt: "2026-04-21T09:20:00.000Z",
      progressStats: { lastEventAt: "2026-04-21T09:40:30.000Z" },
    },
    { now, stallMs },
  ),
  false,
);

assert.deepEqual(
  describeRunLiveness(
    {
      startedAt: "2026-04-21T09:20:00.000Z",
      progressStats: { lastEventAt: "2026-04-21T09:25:00.000Z" },
    },
    { now, stallMs },
  ),
  {
    stalled: true,
    lastActivityAt: "2026-04-21T09:25:00.000Z",
    inactivityMs: 20 * 60 * 1000,
  },
);

console.log("run liveness smoke passed");
