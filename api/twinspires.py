"""TwinSpires live-odds fetch + parsing.

Pulls the public entries JSON for a given track/race, extracts the current live odds for
each program number, and returns a post_position -> live_odds mapping. Results are
cached in-process for a short TTL so repeated refreshes from the UI don't hammer
TwinSpires.

Bot-protection notes:
- TwinSpires is fronted by Akamai Bot Manager, which fingerprints the TLS ClientHello
  (JA3) and HTTP/2 SETTINGS frames. Plain httpx/requests get silently rate-limited or
  blocked because their fingerprint doesn't match any real browser.
- We use curl_cffi with `impersonate="chrome"` so the TLS/HTTP2 fingerprint matches
  Chrome exactly. This is the single biggest thing that keeps us unblocked.
- We reuse a single module-level Session so Akamai's bot-protection cookies (_abck,
  bm_sz) persist across fetches.
"""

import os
import re
import threading
from datetime import datetime

from cachetools import TTLCache
from curl_cffi import requests as curl_requests

DEFAULT_BASE_URL = "https://www.twinspires.com/adw/todays-tracks"
DEFAULT_CACHE_TTL_SECONDS = 15
DEFAULT_TIMEOUT_SECONDS = 8.0
BROWSER_IMPERSONATE = "chrome"

# our track code -> TwinSpires track code (lowercase). fallback is track.lower()
TRACK_CODE_MAP: dict[str, str] = {
    "CD": "cd",
}


class TwinSpiresError(Exception):
    """Raised when the TwinSpires fetch or parse fails."""


class TwinSpiresRaceNotFound(TwinSpiresError):
    """Raised when TwinSpires returns 404 for the requested race.

    Typically means the track isn't running that breed today, the race
    number doesn't exist, or the race has already been removed from the
    live entries feed. Distinct from network/parse failures so callers
    can surface a 404 instead of a 502.
    """


def _base_url() -> str:
    return os.environ.get("TWINSPIRES_BASE_URL", DEFAULT_BASE_URL).rstrip("/")


def twinspires_track_code(track: str) -> str:
    return TRACK_CODE_MAP.get(track.upper(), track.lower())


_EVEN_TOKENS = {"EVEN", "EV", "EVS"}


def parse_fractional_odds(raw) -> float | None:
    """Parse a fractional-odds string to decimal. Fallback for string values.

    Examples: "5/2" -> 2.5, "9-5" -> 1.8, "EVEN" -> 1.0.
    Returns None for null, empty, zero-denominator, or unparseable input.
    """
    if raw is None:
        return None
    s = str(raw).strip().upper()
    if not s:
        return None
    if s in _EVEN_TOKENS:
        return 1.0
    parts = re.split(r"[/-]", s, maxsplit=1)
    if len(parts) != 2:
        return None
    try:
        num = float(parts[0])
        den = float(parts[1])
    except ValueError:
        return None
    if den == 0:
        return None
    return num / den


def parse_program_number(raw) -> int | None:
    """Leading-digit parse. '1' -> 1, '1A' -> 1, '' / None -> None."""
    if raw is None:
        return None
    m = re.match(r"\s*(\d+)", str(raw))
    return int(m.group(1)) if m else None


def _extract_live_odds(runner: dict) -> float | None:
    """Pull the current live odds from one TwinSpires runner entry.

    TwinSpires returns `liveOdds` in several shapes: decimal number (6.0),
    fractional string ("5/2", "9-5", "EVEN"), or bare integer string ("9",
    meaning 9/1 = 9.0). Bare-integer strings don't parse as fractions, so
    we fall back to `oddsTrend.current.oddsNumeric` which always carries
    the already-decimal tote value.
    """
    raw = runner.get("liveOdds")
    if isinstance(raw, (int, float)) and raw > 0:
        return float(raw)
    if isinstance(raw, str):
        parsed = parse_fractional_odds(raw)
        if parsed is not None:
            return parsed
    trend = runner.get("oddsTrend") or {}
    current = trend.get("current") or {} if isinstance(trend, dict) else {}
    val = current.get("oddsNumeric") if isinstance(current, dict) else None
    if isinstance(val, (int, float)) and val > 0:
        return float(val)
    return None


# module-level cache: key -> (fetched_at, odds). TTL is baked in at import time.
_cache: TTLCache = TTLCache(
    maxsize=256,
    ttl=float(
        os.environ.get("TWINSPIRES_CACHE_TTL_SECONDS", DEFAULT_CACHE_TTL_SECONDS)
    ),
)
_cache_lock = threading.Lock()

# persistent http client. reusing a single Session (with its cookie jar and
# connection pool) across fetches lets Akamai's bot-protection cookies
# accumulate, which reduces the chance of being challenged or rate-limited.
_client: curl_requests.Session | None = None
_client_lock = threading.Lock()


def _get_client() -> curl_requests.Session:
    global _client
    with _client_lock:
        if _client is None:
            _client = curl_requests.Session(
                timeout=DEFAULT_TIMEOUT_SECONDS,
                impersonate=BROWSER_IMPERSONATE,
            )
        return _client


def _reset_client():
    global _client
    with _client_lock:
        if _client is not None:
            _client.close()
            _client = None


def _cache_get(
    key: tuple[str, str, int],
) -> tuple[datetime, dict[int, float]] | None:
    with _cache_lock:
        hit = _cache.get(key)
    if hit is None:
        return None
    fetched_at, odds = hit
    return fetched_at, dict(odds)


def _cache_put(
    key: tuple[str, str, int],
    fetched_at: datetime,
    odds: dict[int, float],
):
    with _cache_lock:
        _cache[key] = (fetched_at, dict(odds))


def clear_cache():
    """Clear the module-level cache and http client (used by tests)."""
    with _cache_lock:
        _cache.clear()
    _reset_client()


def _parse_entries_payload(payload) -> dict[int, float]:
    """Extract {post_position: live_odds} from a TwinSpires entries response.

    Coupled entries (e.g. programNumber '1' and '1A') collapse to the same
    post_position -- first non-null parsed odds wins.
    """
    runners = _find_runner_list(payload)
    out: dict[int, float] = {}
    for r in runners:
        if not isinstance(r, dict):
            continue
        post = parse_program_number(r.get("programNumber"))
        if post is None or post in out:
            continue
        odds = _extract_live_odds(r)
        if odds is None:
            continue
        out[post] = odds
    return out


def _find_runner_list(payload) -> list:
    """Best-effort extraction of the runners array from the entries payload."""
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("runners", "entries", "raceEntries"):
            val = payload.get(key)
            if isinstance(val, list):
                return val
        race = payload.get("race")
        if isinstance(race, dict):
            for key in ("runners", "entries", "raceEntries"):
                val = race.get(key)
                if isinstance(val, list):
                    return val
    return []


def fetch_twinspires_odds(
    track: str,
    race_number: int,
    *,
    breed: str = "Thoroughbred",
) -> tuple[datetime, dict[int, float]]:
    """Fetch live odds from TwinSpires for one race.

    Returns (fetched_at, {post_position: live_odds_decimal}). Only post positions
    with a non-null, parseable odds value are included. Raises
    TwinSpiresRaceNotFound if TwinSpires has no entry for the race (404), or
    TwinSpiresError on any other network or parse failure.
    """
    ts_code = twinspires_track_code(track)
    key = (ts_code, breed, race_number)

    cached = _cache_get(key)
    if cached is not None:
        return cached

    url = f"{_base_url()}/{ts_code}/{breed}/races/{race_number}/entries"
    params = {"affid": "2800"}

    last_exc: Exception | None = None
    for _ in range(2):
        try:
            resp = _get_client().get(url, params=params)
            if resp.status_code == 404:
                raise TwinSpiresRaceNotFound(
                    f"TwinSpires has no entry for {track} R{race_number} ({breed})"
                )
            resp.raise_for_status()
            payload = resp.json()
            break
        except TwinSpiresRaceNotFound:
            raise
        except Exception as e:
            last_exc = e
    else:
        raise TwinSpiresError(f"TwinSpires fetch failed: {last_exc}") from last_exc

    odds = _parse_entries_payload(payload)
    fetched_at = datetime.now().astimezone()
    _cache_put(key, fetched_at, odds)
    return fetched_at, odds
