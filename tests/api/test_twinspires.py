"""Tests for TwinSpires live-odds fetch + parsers."""

import pytest

from api import twinspires
from api.twinspires import (
    TwinSpiresError,
    TwinSpiresRaceNotFound,
    fetch_twinspires_odds,
    parse_fractional_odds,
    parse_program_number,
    twinspires_track_code,
)


class TestParseFractionalOdds:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("5/2", 2.5),
            ("9-5", 1.8),
            ("1/1", 1.0),
            ("EVEN", 1.0),
            ("ev", 1.0),
            ("EVS", 1.0),
            ("20/1", 20.0),
            (" 3/1 ", 3.0),
        ],
    )
    def test_valid(self, raw, expected):
        assert parse_fractional_odds(raw) == pytest.approx(expected)

    @pytest.mark.parametrize("raw", [None, "", "   ", "garbage", "5/", "/2", "5/0"])
    def test_invalid(self, raw):
        assert parse_fractional_odds(raw) is None


class TestParseProgramNumber:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("1", 1),
            ("7", 7),
            ("1A", 1),
            ("12B", 12),
            ("  3 ", 3),
        ],
    )
    def test_valid(self, raw, expected):
        assert parse_program_number(raw) == expected

    @pytest.mark.parametrize("raw", [None, "", "A", "AB"])
    def test_invalid(self, raw):
        assert parse_program_number(raw) is None


class TestTrackCodeMap:
    def test_known_track(self):
        assert twinspires_track_code("CD") == "cd"

    def test_case_insensitive(self):
        assert twinspires_track_code("cd") == "cd"

    def test_unknown_fallback(self):
        assert twinspires_track_code("XYZ") == "xyz"


class _FakeResponse:
    def __init__(self, payload, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


class _FakeClient:
    """Drop-in stand-in for curl_cffi Session."""

    def __init__(self, *, payload=None, raises=None, record=None, status_code=200):
        self._payload = payload
        self._raises = raises
        self._record = record if record is not None else []
        self._status_code = status_code

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get(self, url, params=None):
        self._record.append((url, params))
        if self._raises:
            raise self._raises
        return _FakeResponse(self._payload, status_code=self._status_code)

    def close(self):
        pass


@pytest.fixture(autouse=True)
def _clear_cache():
    twinspires.clear_cache()
    yield
    twinspires.clear_cache()


class TestFetchTwinSpiresOdds:
    def test_happy_path(self, monkeypatch):
        payload = {
            "runners": [
                {"programNumber": "1", "liveOdds": "5/2"},
                {"programNumber": "2", "liveOdds": "9-5"},
                {"programNumber": "3", "liveOdds": "EVEN"},
                {"programNumber": "4", "liveOdds": None},
            ]
        }
        monkeypatch.setattr(
            twinspires.curl_requests,
            "Session",
            lambda **kw: _FakeClient(payload=payload),
        )

        fetched_at, odds = fetch_twinspires_odds("CD", 5)

        assert odds == {1: 2.5, 2: 1.8, 3: 1.0}
        assert fetched_at is not None

    def test_coupled_entries_dedupe(self, monkeypatch):
        payload = {
            "runners": [
                {"programNumber": "1", "liveOdds": "5/2"},
                {"programNumber": "1A", "liveOdds": "8/1"},  # same post, ignored
                {"programNumber": "2", "liveOdds": "3/1"},
            ]
        }
        monkeypatch.setattr(
            twinspires.curl_requests,
            "Session",
            lambda **kw: _FakeClient(payload=payload),
        )

        _, odds = fetch_twinspires_odds("CD", 1)
        assert odds == {1: 2.5, 2: 3.0}

    def test_all_null_odds(self, monkeypatch):
        payload = {
            "runners": [
                {"programNumber": "1", "liveOdds": None},
                {"programNumber": "2", "liveOdds": None},
            ]
        }
        monkeypatch.setattr(
            twinspires.curl_requests,
            "Session",
            lambda **kw: _FakeClient(payload=payload),
        )

        _, odds = fetch_twinspires_odds("CD", 1)
        assert odds == {}

    def test_cache_hit_within_ttl(self, monkeypatch):
        payload = {"runners": [{"programNumber": "1", "liveOdds": "5/2"}]}
        record: list = []
        monkeypatch.setattr(
            twinspires.curl_requests,
            "Session",
            lambda **kw: _FakeClient(payload=payload, record=record),
        )

        first_at, first_odds = fetch_twinspires_odds("CD", 1)
        second_at, second_odds = fetch_twinspires_odds("CD", 1)

        assert len(record) == 1  # second call hit the cache
        assert first_at == second_at
        assert first_odds == second_odds == {1: 2.5}

    def test_different_keys_not_shared(self, monkeypatch):
        payload = {"runners": [{"programNumber": "1", "liveOdds": "5/2"}]}
        record: list = []
        monkeypatch.setattr(
            twinspires.curl_requests,
            "Session",
            lambda **kw: _FakeClient(payload=payload, record=record),
        )

        fetch_twinspires_odds("CD", 1)
        fetch_twinspires_odds("CD", 2)
        fetch_twinspires_odds("SA", 1)

        assert len(record) == 3

    def test_network_error_retries_then_raises(self, monkeypatch):
        err = RuntimeError("boom")
        record: list = []
        monkeypatch.setattr(
            twinspires.curl_requests,
            "Session",
            lambda **kw: _FakeClient(raises=err, record=record),
        )

        with pytest.raises(TwinSpiresError):
            fetch_twinspires_odds("CD", 1)

        assert len(record) == 2  # original attempt + 1 retry

    def test_404_raises_race_not_found_without_retry(self, monkeypatch):
        record: list = []
        monkeypatch.setattr(
            twinspires.curl_requests,
            "Session",
            lambda **kw: _FakeClient(payload={}, status_code=404, record=record),
        )

        with pytest.raises(TwinSpiresRaceNotFound):
            fetch_twinspires_odds("CD", 1)

        assert len(record) == 1  # no retry on 404

    def test_retry_recovers(self, monkeypatch):
        """First attempt raises, second attempt succeeds."""
        payload = {"runners": [{"programNumber": "1", "liveOdds": "5/2"}]}
        calls = {"n": 0}

        class _FlakyClient(_FakeClient):
            def get(self, url, params=None):
                calls["n"] += 1
                if calls["n"] == 1:
                    raise RuntimeError("flaky wifi")
                return _FakeResponse(payload)

        monkeypatch.setattr(
            twinspires.curl_requests, "Session", lambda **kw: _FlakyClient()
        )

        _, odds = fetch_twinspires_odds("CD", 1)
        assert odds == {1: 2.5}
        assert calls["n"] == 2

    def test_url_uses_mapped_track_code(self, monkeypatch):
        payload = {"runners": []}
        record: list = []
        monkeypatch.setattr(
            twinspires.curl_requests,
            "Session",
            lambda **kw: _FakeClient(payload=payload, record=record),
        )

        fetch_twinspires_odds("CD", 7)

        url, params = record[0]
        assert "/cd/Thoroughbred/races/7/entries" in url
        assert params == {"affid": "2800"}

    def test_handles_numeric_live_odds_and_top_level_list(self, monkeypatch):
        """Real TwinSpires shape: top-level list with numeric `liveOdds`."""
        payload = [
            {"programNumber": "1", "liveOdds": 6.0, "scratched": False},
            {"programNumber": "2", "liveOdds": 2.5, "scratched": False},
            {"programNumber": "3", "liveOdds": None, "scratched": False},
        ]
        monkeypatch.setattr(
            twinspires.curl_requests,
            "Session",
            lambda **kw: _FakeClient(payload=payload),
        )

        _, odds = fetch_twinspires_odds("CD", 1)
        assert odds == {1: 6.0, 2: 2.5}

    def test_falls_back_to_odds_trend(self, monkeypatch):
        """If liveOdds is null, use oddsTrend.current.oddsNumeric."""
        payload = [
            {
                "programNumber": "1",
                "liveOdds": None,
                "oddsTrend": {"current": {"oddsNumeric": 3.5, "oddsText": "7/2"}},
            },
        ]
        monkeypatch.setattr(
            twinspires.curl_requests,
            "Session",
            lambda **kw: _FakeClient(payload=payload),
        )

        _, odds = fetch_twinspires_odds("CD", 1)
        assert odds == {1: 3.5}

    def test_bare_integer_live_odds_string_uses_odds_trend(self, monkeypatch):
        """Real TwinSpires shape for whole-number odds: liveOdds='9' (= 9.0)."""
        payload = [
            {
                "programNumber": "1",
                "liveOdds": "9",
                "oddsTrend": {"current": {"oddsNumeric": 9.0, "oddsText": "9"}},
            },
            {
                "programNumber": "2",
                "liveOdds": "9/2",
                "oddsTrend": {"current": {"oddsNumeric": 4.5, "oddsText": "9/2"}},
            },
        ]
        monkeypatch.setattr(
            twinspires.curl_requests,
            "Session",
            lambda **kw: _FakeClient(payload=payload),
        )

        _, odds = fetch_twinspires_odds("CD", 1)
        assert odds == {1: 9.0, 2: 4.5}
