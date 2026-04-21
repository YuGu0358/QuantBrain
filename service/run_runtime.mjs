export function finalizeRunExit(activeRuns, state, exitCode, finishedAt = new Date().toISOString()) {
  state.status = exitCode === 0 ? "completed" : "failed";
  state.exitCode = exitCode;
  state.finishedAt = finishedAt;
  if (state?.runId) activeRuns.delete(state.runId);
  return state;
}
