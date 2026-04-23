// API helpers — all endpoints are on the same origin
async function api(path, options = {}) {
  const headers = {};
  if (options.body) headers["Content-Type"] = "application/json";
  const res = await fetch(path, { headers, ...options });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed (${res.status})`);
  }
  return res.json();
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
  return odds.toFixed(1);
}

function surfaceLabel(s) {
  return s === "D" ? "Dirt" : "Turf";
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

    async init() {
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
  };
}
