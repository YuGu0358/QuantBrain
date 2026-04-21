function toTimeMs(value) {
  if (!value) return null;
  const ms = Date.parse(value);
  return Number.isFinite(ms) ? ms : null;
}

export function describeRunLiveness(run, { now = Date.now(), stallMs } = {}) {
  const lastActivityMs =
    toTimeMs(run?.progressStats?.lastEventAt) ??
    toTimeMs(run?.startedAt);
  if (!Number.isFinite(lastActivityMs)) {
    return {
      stalled: false,
      lastActivityAt: null,
      inactivityMs: null,
    };
  }
  const inactivityMs = Math.max(0, now - lastActivityMs);
  return {
    stalled: Number.isFinite(stallMs) ? inactivityMs >= stallMs : false,
    lastActivityAt: new Date(lastActivityMs).toISOString(),
    inactivityMs,
  };
}

export function isRunStalled(run, options = {}) {
  return describeRunLiveness(run, options).stalled;
}
