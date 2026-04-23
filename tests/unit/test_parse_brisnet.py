"""Tests for data.parse_brisnet."""

from datetime import date
from pathlib import Path

import pytest

from data.parse_brisnet import (
    _date,
    _derive_surface_stats,
    _encode_age_restriction,
    _encode_sex_restriction,
    _expand_yob,
    _map_surface,
    _sub_nonneg,
    _validate_sex,
    parse_brisnet_csv,
)

CDX0425_PATH = Path("data/raw/brisnet/CDX0425.csv")


# ---------- pure helpers ----------


@pytest.mark.parametrize(
    "code,expected",
    [
        ("BUM", "3U"),  # 3yo and up, fillies/mares
        ("BON", "03"),  # 3yo only
        ("EON", "34"),  # 3 & 4yo only
        ("FON", "45"),  # 4 & 5yo only
        ("GON", "35"),  # 3,4,5yo only
        ("AOC", "02"),  # 2yo only, colts/geldings
        ("HUN", "2U"),  # all ages and up -> collapse to 2U
        ("HON", "2U"),  # 'all ages only' also collapses to 2U
        ("", None),
        ("XX", None),  # unknown age class
    ],
)
def test_encode_age_restriction(code, expected):
    assert _encode_age_restriction(code) == expected


@pytest.mark.parametrize(
    "code,expected",
    [
        ("BUM", "B"),  # mares + fillies
        ("BOC", "A"),  # colts + geldings
        ("BOF", "F"),  # fillies
        ("BUN", None),  # no sex restriction
        ("XX", None),
        ("", None),
    ],
)
def test_encode_sex_restriction(code, expected):
    assert _encode_sex_restriction(code) == expected


@pytest.mark.parametrize(
    "surface,expected",
    [("D", "D"), ("d", "D"), ("T", "T"), ("t", "T"), ("s", None), ("", None)],
)
def test_map_surface(surface, expected):
    assert _map_surface(surface) == expected


@pytest.mark.parametrize(
    "sex,expected",
    [("F", "F"), ("m", "M"), ("G ", "G"), ("X", None), ("", None)],
)
def test_validate_sex(sex, expected):
    assert _validate_sex(sex) == expected


def test_expand_yob():
    assert _expand_yob(22) == 2022
    assert _expand_yob(None) is None


def test_date_valid():
    assert _date("20260425") == date(2026, 4, 25)


def test_date_invalid():
    assert _date("") is None
    assert _date("2026") is None
    assert _date("20261301") is None  # month 13


def test_sub_nonneg():
    assert _sub_nonneg("10", "3", "2") == 5
    assert _sub_nonneg("10", "", "2") == 8  # blank treated as 0
    assert _sub_nonneg("", "3") is None  # blank minuend -> None
    assert _sub_nonneg("3", "5") == 0  # floor at 0


# ---------- surface stats derivation ----------


def _row_with(field_values: dict[int, str]) -> list[str]:
    """Build a sparse 250-field row with specific 1-indexed values set."""
    max_idx = max(field_values.keys())
    row = [""] * max_idx
    for idx, val in field_values.items():
        row[idx - 1] = val
    return row


def test_derive_surface_stats_turf_today():
    # today = turf -> use lifetime turf record (fields 75-78)
    row = _row_with({75: "12", 76: "3", 77: "2", 78: "4"})
    assert _derive_surface_stats(row, "T") == {
        "surface_starts": 12,
        "surface_wins": 3,
        "surface_seconds": 2,
        "surface_thirds": 4,
    }


def test_derive_surface_stats_dirt_today():
    # today = dirt -> lifetime (97-100) - turf (75-78); all-weather stays folded
    # into dirt to match Equibase's convention
    row = _row_with({
        97: "20",
        98: "5",
        99: "4",
        100: "3",
        75: "5",
        76: "1",
        77: "1",
        78: "0",
        231: "2",
        232: "0",
        233: "1",
        234: "0",
    })
    assert _derive_surface_stats(row, "D") == {
        "surface_starts": 15,  # 20 - 5
        "surface_wins": 4,  # 5 - 1
        "surface_seconds": 3,  # 4 - 1
        "surface_thirds": 3,  # 3 - 0
    }


def test_derive_surface_stats_unknown_surface():
    assert _derive_surface_stats([], None) == {
        "surface_starts": None,
        "surface_wins": None,
        "surface_seconds": None,
        "surface_thirds": None,
    }


# ---------- smoke test against real file ----------


@pytest.mark.skipif(not CDX0425_PATH.exists(), reason="CDX0425.csv not present")
def test_parse_cdx0425_smoke():
    entries, pps, workouts = parse_brisnet_csv(CDX0425_PATH)

    # one entry per CSV row
    assert len(entries) > 0

    # first row of CDX0425 is "TOO MANY MIKES", CD race 1 post 1, 2026-04-25
    first = entries[0]
    assert first["track"] == "CD"
    assert first["race_number"] == 1
    assert first["race_date"] == date(2026, 4, 25)
    assert first["post_position"] == 1
    assert first["horse_name"] == "TOO MANY MIKES"
    assert first["race_id"] == "2026-04-25_CD_R1"
    assert first["surface"] == "D"
    assert first["age_restriction"] == "3U"  # BUM -> 3yo and up
    assert first["sex_restriction"] == "B"  # fillies and mares
    assert first["sex"] == "F"
    assert first["year_of_birth"] == 2022  # Brisnet yob "22"
    assert first["morning_line_odds_float"] == 8.0
    assert first["weight_carried"] == 126

    # PPs use same race_id and horse_name keys, pp_index 1..N
    first_horse_pps = [p for p in pps if p["horse_name"] == "TOO MANY MIKES"]
    assert len(first_horse_pps) >= 1
    assert first_horse_pps[0]["pp_index"] == 1
    assert first_horse_pps[0]["pp_race_date"] == date(2026, 3, 22)
    assert first_horse_pps[0]["pp_distance_yards"] == 1870
    assert first_horse_pps[0]["pp_surface"] == "T"

    # workouts also keyed on race_id + horse_name
    first_horse_workouts = [w for w in workouts if w["horse_name"] == "TOO MANY MIKES"]
    assert len(first_horse_workouts) > 0
