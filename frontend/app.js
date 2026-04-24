// API helpers — all endpoints are on the same origin.
// Adds a per-attempt timeout and retries for network/timeout/5xx errors,
// so flaky wifi (e.g. overloaded Churchill network) doesn't hang the UI.
async function api(path, options = {}, { retries = 2, timeoutMs = 8000 } = {}) {
  const headers = {};
  if (options.body) headers["Content-Type"] = "application/json";
  let attempt = 0;
  while (true) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const res = await fetch(path, {
        headers,
        signal: controller.signal,
        ...options,
      });
      if (res.ok) return res.json();
      if (res.status < 500) {
        // client error — don't retry
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || `Request failed (${res.status})`);
      }
      if (attempt >= retries) throw new Error(`Server error (${res.status})`);
      // 5xx falls through to retry
    } catch (e) {
      const isNetworkErr = e.name === "AbortError" || e instanceof TypeError;
      if (!isNetworkErr || attempt >= retries) {
        if (e.name === "AbortError") {
          throw new Error("Request timed out");
        }
        throw e;
      }
    } finally {
      clearTimeout(timer);
    }
    // backoff: 400ms, 1200ms
    await new Promise((r) => setTimeout(r, 400 * (attempt + 1) ** 2));
    attempt++;
  }
}

function fetchRaces() {
  return api("/races");
}

function fetchRace(raceId) {
  return api(`/races/${raceId}`);
}

function updateOdds(raceId, odds) {
  return api(`/races/${raceId}/odds`, {
    method: "PATCH",
    body: JSON.stringify({ odds }),
  });
}

function scratchRunner(raceId, postPosition) {
  return api(`/races/${raceId}/runners/${postPosition}/scratch`, {
    method: "PATCH",
  });
}

function unscratchRunner(raceId, postPosition) {
  return api(`/races/${raceId}/runners/${postPosition}/unscratch`, {
    method: "PATCH",
  });
}

function getPredictions(raceId) {
  return api(`/races/${raceId}/predict`, { method: "POST" });
}

function fetchTwinSpiresOdds(raceId) {
  return api(`/races/${raceId}/fetch-twinspires-odds`, { method: "POST" });
}

// formatting helpers
function fmtProb(p) {
  return (p * 100).toFixed(1) + "%";
}

function fmtEv(ev) {
  const sign = ev >= 0 ? "+" : "";
  return sign + (ev * 100).toFixed(0) + "\u00a2";
}

function fmtOdds(odds) {
  if (odds == null) return "-";
  // whole numbers show as integers (8.0 -> "8"); fractional odds keep one decimal (2.5 -> "2.5")
  return Number.isInteger(odds) ? String(odds) : odds.toFixed(1);
}

function surfaceLabel(s) {
  return s === "D" ? "Dirt" : "Turf";
}

// Format distance in yards as "N furlongs" for sub-mile races or "N M/D mile" for mile+.
function distanceLabel(yards) {
  // under 1 mile -> furlongs, rounded to the nearest half-furlong (110 yards)
  if (yards < 1760) {
    const halves = Math.round(yards / 110);
    const whole = Math.floor(halves / 2);
    const half = halves % 2 === 1;
    return half ? `${whole} 1/2 furlongs` : `${whole} furlongs`;
  }
  // 1+ mile -> express as 16ths of a mile (110 yards per 1/16 mile), reduced
  const sixteenths = Math.round(yards / 110);
  const miles = Math.floor(sixteenths / 16);
  const rem = sixteenths % 16;
  if (rem === 0) return `${miles} mile`;
  const g = gcd(rem, 16);
  return `${miles} ${rem / g}/${16 / g} mile`;
}

function gcd(a, b) {
  return b === 0 ? a : gcd(b, a % b);
}

// Build a short restriction label, e.g. "3yo+ F", "2yo", "3yo F&M", or "" if no restriction.
function restrictionLabel(age, sex) {
  const parts = [];
  if (age) {
    const second = age[1];
    if (second === "U" || second === "+") {
      parts.push(`${age[0]}yo+`);
    } else {
      const n = parseInt(age, 10);
      parts.push(Number.isNaN(n) ? age : `${n}yo`);
    }
  }
  if (sex === "F") parts.push("F");
  else if (sex === "B") parts.push("F&M");
  return parts.join(" ");
}

// "6 furlongs · Dirt · 3yo+ F" — used on list cards and the detail header.
function raceMetaStr(race) {
  const parts = [distanceLabel(race.distance_yards), surfaceLabel(race.surface)];
  const restriction = restrictionLabel(race.age_restriction, race.sex_restriction);
  if (restriction) parts.push(restriction);
  return parts.join(" \u00b7 ");
}

const TRACK_NAMES = {
  CD: "Churchill Downs",
  KEE: "Keeneland",
};

function trackName(code) {
  return TRACK_NAMES[code] || code;
}

// Group races by (race_date, track). Returns a list sorted by date (asc) then track,
// with each group's races sorted by race_number.
function groupRaces(races) {
  const groups = {};
  for (const race of races) {
    const key = `${race.race_date}|${race.track}`;
    if (!groups[key]) {
      groups[key] = {
        key,
        date: race.race_date,
        track: race.track,
        label: `${race.race_date} \u00b7 ${trackName(race.track)}`,
        races: [],
      };
    }
    groups[key].races.push(race);
  }
  const list = Object.values(groups);
  list.sort((a, b) => {
    if (a.date !== b.date) return a.date < b.date ? -1 : 1;
    return a.track < b.track ? -1 : 1;
  });
  for (const g of list) g.races.sort((a, b) => a.race_number - b.race_number);
  return list;
}

// hash-based routing so refreshes restore the current screen
function parseRoute() {
  const hash = window.location.hash.replace(/^#\/?/, "");
  const parts = hash.split("/").filter(Boolean);
  if (parts[0] === "race" && parts[1]) {
    if (parts[2] === "predictions") {
      return { screen: "predictions", raceId: parts[1] };
    }
    return { screen: "detail", raceId: parts[1] };
  }
  return { screen: "list" };
}

function routeHash(screen, raceId) {
  if (screen === "detail" && raceId) return `#/race/${raceId}`;
  if (screen === "predictions" && raceId) return `#/race/${raceId}/predictions`;
  return "";
}

// Alpine.js app data
function horseApp() {
  return {
    screen: "list",
    races: [],
    currentRace: null,
    predictions: null,
    loading: false,
    error: null,
    warning: null,
    online: true,
    refreshingOdds: false,
    lastOddsFetchAt: null,

    async init() {
      this.online = navigator.onLine;
      window.addEventListener("online", () => {
        this.online = true;
      });
      window.addEventListener("offline", () => {
        this.online = false;
      });
      window.addEventListener("error", (e) => {
        this.reportClientError("error", e.message, e.error?.stack);
      });
      window.addEventListener("unhandledrejection", (e) => {
        const reason = e.reason;
        this.reportClientError(
          "unhandledrejection",
          reason?.message || String(reason),
          reason?.stack,
        );
      });
      window.addEventListener("popstate", () => this.syncFromRoute());
      // blank out the default list screen if refreshing into a race URL,
      // so we don't flash the list while loading the race
      if (parseRoute().screen !== "list") {
        this.screen = null;
      }
      await this.loadRaces();
      await this.syncFromRoute();
    },

    pushRoute() {
      const hash = routeHash(this.screen, this.currentRace?.race_id);
      if (hash === window.location.hash) return;
      const url = window.location.pathname + window.location.search + hash;
      history.pushState(null, "", url);
    },

    async syncFromRoute() {
      const route = parseRoute();
      if (route.screen === "list") {
        if (this.screen !== "list") {
          this.screen = "list";
          this.currentRace = null;
          this.predictions = null;
          this.clearError();
          await this.loadRaces();
        }
        return;
      }
      if (route.screen === "detail") {
        this.clearError();
        this.loading = true;
        try {
          if (this.currentRace?.race_id !== route.raceId) {
            this.currentRace = await fetchRace(route.raceId);
            this.predictions = null;
            this.lastOddsFetchAt = null;
          }
          this.screen = "detail";
        } catch (e) {
          this.error = e.message;
          this.screen = "list";
          history.replaceState(
            null,
            "",
            window.location.pathname + window.location.search,
          );
        } finally {
          this.loading = false;
        }
        return;
      }
      // route.screen === 'predictions'
      this.clearError();
      this.loading = true;
      try {
        if (this.currentRace?.race_id !== route.raceId) {
          this.currentRace = await fetchRace(route.raceId);
          this.predictions = null;
          this.lastOddsFetchAt = null;
        }
        if (!this.predictions) {
          this.predictions = await getPredictions(route.raceId);
        }
        this.screen = "predictions";
      } catch (e) {
        this.error = e.message;
        this.screen = this.currentRace ? "detail" : "list";
        history.replaceState(
          null,
          "",
          window.location.pathname +
          window.location.search +
          routeHash(this.screen, this.currentRace?.race_id),
        );
      } finally {
        this.loading = false;
      }
    },

    clearError() {
      this.error = null;
      this.warning = null;
    },

    reportClientError(kind, message, stack) {
      this.error = `${kind}: ${message}`;
      const payload = JSON.stringify({
        kind,
        message,
        stack,
        url: window.location.href,
        userAgent: navigator.userAgent,
        ts: new Date().toISOString(),
      });
      // fire-and-forget; keepalive so it survives a navigation
      fetch("/client-error", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: payload,
        keepalive: true,
      }).catch(() => { });
    },

    async loadRaces() {
      this.clearError();
      this.loading = true;
      try {
        this.races = await fetchRaces();
      } catch (e) {
        this.error = e.message;
      } finally {
        this.loading = false;
      }
    },

    async selectRace(raceId) {
      this.clearError();
      this.loading = true;
      try {
        this.currentRace = await fetchRace(raceId);
        this.predictions = null;
        this.lastOddsFetchAt = null;
        this.screen = "detail";
        this.pushRoute();
      } catch (e) {
        this.error = e.message;
      } finally {
        this.loading = false;
      }
    },

    async toggleScratch(runner) {
      this.clearError();
      try {
        if (runner.scratched) {
          this.currentRace = await unscratchRunner(
            this.currentRace.race_id,
            runner.post_position,
          );
        } else {
          this.currentRace = await scratchRunner(
            this.currentRace.race_id,
            runner.post_position,
          );
        }
      } catch (e) {
        this.error = e.message;
      }
    },

    async refreshLiveOdds() {
      if (!this.currentRace) return;
      this.clearError();
      this.refreshingOdds = true;
      try {
        const resp = await fetchTwinSpiresOdds(this.currentRace.race_id);
        this.currentRace = resp.race;
        this.lastOddsFetchAt = new Date(resp.fetched_at);
        const missing = resp.missing_post_positions || [];
        if (resp.applied_post_positions.length === 0) {
          this.error =
            "TwinSpires has no live odds yet (wagering may not be open).";
        } else if (missing.length > 0) {
          this.warning =
            `TwinSpires returned no odds for post${missing.length > 1 ? "s" : ""} ` +
            `${missing.join(", ")}. Missing odds will use morning-line defaults.`;
        }
      } catch (e) {
        this.error = `Live odds fetch failed: ${e.message}`;
      } finally {
        this.refreshingOdds = false;
      }
    },

    fmtFetchedAt() {
      if (!this.lastOddsFetchAt) return "";
      return this.lastOddsFetchAt.toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
      });
    },

    async submitOdds() {
      this.clearError();
      const active = this.currentRace.runners.filter((r) => !r.scratched);
      if (active.some((r) => !r.live_odds || r.live_odds <= 0)) {
        this.error = "Enter valid odds for all active runners.";
        return;
      }
      this.loading = true;
      try {
        const odds = active.map((r) => ({
          post_position: r.post_position,
          live_odds: r.live_odds,
        }));
        await updateOdds(this.currentRace.race_id, odds);
        this.predictions = await getPredictions(this.currentRace.race_id);
        this.screen = "predictions";
        this.pushRoute();
      } catch (e) {
        this.error = e.message;
      } finally {
        this.loading = false;
      }
    },

    async backToList() {
      this.screen = "list";
      this.currentRace = null;
      this.predictions = null;
      this.lastOddsFetchAt = null;
      this.clearError();
      this.pushRoute();
      await this.loadRaces();
    },

    backToDetail() {
      this.screen = "detail";
      this.predictions = null;
      this.clearError();
      this.pushRoute();
    },

    // template helpers
    fmtProb,
    fmtEv,
    fmtOdds,
    surfaceLabel,
    distanceLabel,
    restrictionLabel,
    raceMetaStr,
    trackName,
    groupRaces,
  };
}
