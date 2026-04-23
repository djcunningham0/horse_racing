"""Parse Brisnet single-track/single-day CSV files.

Each CSV row is one horse-in-a-race with ~1435 positional fields (see
data/raw/brisnet/pp_schema.txt). This module parses one CSV into three lists of dicts
matching the schemas in data/ingest.py (entries, past_performances, workouts), so the
output flows into the same model/features.py aggregation used at training time.
"""

import csv
from datetime import date
from pathlib import Path

from data.schema import make_race_id

# Brisnet age-class (1st char of field 10) -> (min_age, max_age)
AGE_CLASS_RANGES: dict[str, tuple[int, int]] = {
    "A": (2, 2),
    "B": (3, 3),
    "C": (4, 4),
    "D": (5, 5),
    "E": (3, 4),
    "F": (4, 5),
    "G": (3, 5),
    "H": (2, 99),
}

# Brisnet sex-restriction char (3rd char of field 10) -> training code
SEX_RESTRICTION_MAP: dict[str, str | None] = {
    "N": None,
    "M": "B",  # fillies and mares
    "C": "A",  # colts and geldings
    "F": "F",  # fillies
}

VALID_SEX_CODES = {"G", "F", "M", "C", "H", "R", "B"}


def parse_brisnet_csv(
    csv_path: Path | str,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Parse a Brisnet CSV into (entries, past_performances, workouts)."""
    entries: list[dict] = []
    pps: list[dict] = []
    workouts: list[dict] = []

    with open(csv_path, newline="") as f:
        for row in csv.reader(f):
            if not row:
                continue
            entry, pp_rows, workout_rows = _parse_row(row)
            entries.append(entry)
            pps.extend(pp_rows)
            workouts.extend(workout_rows)

    return entries, pps, workouts


def _parse_row(row: list[str]) -> tuple[dict, list[dict], list[dict]]:
    """Parse one Brisnet CSV row into an entry dict + PP rows + workout rows."""
    track = _f(row, 1).strip()
    race_number = _int(_f(row, 3))
    race_date = _date(_f(row, 2))
    horse_name = _f(row, 45).strip()
    race_date_str = race_date.isoformat() if race_date else ""
    race_id = make_race_id(race_date_str, track, race_number or 0)

    today_surface = _map_surface(_f(row, 7))
    all_weather = _f(row, 25).strip() == "A"

    entry = {
        # race-level
        "race_id": race_id,
        "race_date": race_date,
        "track": track,
        "race_number": race_number,
        "distance_yards": _abs_int(_f(row, 6)),
        "surface": today_surface,
        "course_desc": "All Weather Track" if all_weather else None,
        "purse": _float(_f(row, 12)),
        "age_restriction": _encode_age_restriction(_f(row, 10)),
        "sex_restriction": _encode_sex_restriction(_f(row, 10)),
        # runner identity
        "horse_name": horse_name,
        "post_position": _int(_f(row, 58)) or _int(_f(row, 4)),
        # odds
        "morning_line_odds": _f(row, 44).strip() or None,
        "morning_line_odds_float": _float(_f(row, 44)),
        # runner attributes
        "weight_carried": _int(_f(row, 51)),
        "year_of_birth": _expand_yob(_int(_f(row, 46))),
        "sex": _validate_sex(_f(row, 49)),
        # lifetime career stats
        "career_starts": _int(_f(row, 97)),
        "career_wins": _int(_f(row, 98)),
        "career_seconds": _int(_f(row, 99)),
        "career_thirds": _int(_f(row, 100)),
        "career_earnings": _float(_f(row, 101)),
        # surface-specific career stats (today's surface)
        **_derive_surface_stats(row, today_surface),
        # scratched flag from Brisnet (informational; not filtered at ingest)
        "brisnet_scratched": _f(row, 5).strip() == "S",
    }

    pp_rows = _parse_pp_rows(row, race_id, horse_name)
    workout_rows = _parse_workout_rows(row, race_id, horse_name)
    return entry, pp_rows, workout_rows


def _parse_pp_rows(row: list[str], race_id: str, horse_name: str) -> list[dict]:
    """Extract up to 10 PP rows. pp_index 1 = most recent."""
    out = []
    for i in range(10):
        pp_date = _date(_f(row, 256 + i))
        if pp_date is None:
            continue
        out.append({
            "race_id": race_id,
            "horse_name": horse_name,
            "pp_index": i + 1,
            "pp_race_date": pp_date,
            "pp_distance_yards": _abs_int(_f(row, 316 + i)),
            "pp_surface": _f(row, 326 + i).strip() or None,
            "pp_num_starters": _int(_f(row, 346 + i)),
            "pp_official_finish": _int(_f(row, 616 + i)),  # None for DNF-code chars
            "pp_speed_figure": _int(_f(row, 846 + i)),  # BRIS Speed Rating
            "pp_odds": _float(_f(row, 516 + i)),
            "pp_class_rating": _int(_f(row, 1167 + i)),  # BRIS Speed Par for class
        })
    return out


def _parse_workout_rows(row: list[str], race_id: str, horse_name: str) -> list[dict]:
    """Extract up to 12 workout rows (schema fields 102/186/198 base)."""
    out = []
    for i in range(12):
        wo_date = _date(_f(row, 102 + i))
        if wo_date is None:
            continue
        out.append({
            "race_id": race_id,
            "horse_name": horse_name,
            "workout_date": wo_date,
            "workout_ranking": _int(_f(row, 198 + i)),
            "workout_num_in_group": _int(_f(row, 186 + i)),
        })
    return out


def _derive_surface_stats(row: list[str], today_surface: str | None) -> dict:
    """Compute surface_starts/wins/seconds/thirds for today's surface.

    Brisnet splits all-weather (231-234) out from its Lifetime Turf (75-78) and
    Lifetime totals (97-100), but Equibase — which our training data comes from —
    lumps AW in with dirt. To match that convention, dirt today = lifetime - turf.
    """
    if today_surface == "T":
        return {
            "surface_starts": _int(_f(row, 75)),
            "surface_wins": _int(_f(row, 76)),
            "surface_seconds": _int(_f(row, 77)),
            "surface_thirds": _int(_f(row, 78)),
        }
    if today_surface == "D":
        return {
            "surface_starts": _sub_nonneg(_f(row, 97), _f(row, 75)),
            "surface_wins": _sub_nonneg(_f(row, 98), _f(row, 76)),
            "surface_seconds": _sub_nonneg(_f(row, 99), _f(row, 77)),
            "surface_thirds": _sub_nonneg(_f(row, 100), _f(row, 78)),
        }
    return {
        "surface_starts": None,
        "surface_wins": None,
        "surface_seconds": None,
        "surface_thirds": None,
    }


def _encode_age_restriction(bris_code: str) -> str | None:
    """Convert Brisnet 3-char age/sex restriction code to the 2-char training code
    that derive_features parses in model/feature_pipeline.py.

    Examples: 'BUM' -> '3U', 'BON' -> '03', 'EON' -> '34', 'HUN' -> '2U'.
    """
    code = bris_code.strip()
    if len(code) < 2:
        return None
    age_class = code[0]
    mode = code[1]
    rng = AGE_CLASS_RANGES.get(age_class)
    if rng is None:
        return None
    min_age, max_age = rng
    # any open-ended age class collapses to {min}U
    if mode == "U" or max_age >= 99:
        return f"{min_age}U"
    if mode == "O":
        if min_age == max_age:
            return f"0{min_age}"
        return f"{min_age}{max_age}"
    return None


def _encode_sex_restriction(bris_code: str) -> str | None:
    """Extract the sex-restriction char (3rd of 3) and map to training code."""
    code = bris_code.strip()
    if len(code) < 3:
        return None
    return SEX_RESTRICTION_MAP.get(code[2])


def _map_surface(bris: str) -> str | None:
    """Collapse D/d -> 'D', T/t -> 'T', else None (filters out steeplechase etc.)."""
    s = bris.strip()
    if s in ("D", "d"):
        return "D"
    if s in ("T", "t"):
        return "T"
    return None


def _validate_sex(bris_sex: str) -> str | None:
    s = bris_sex.strip().upper()
    return s if s in VALID_SEX_CODES else None


def _expand_yob(yy: int | None) -> int | None:
    """Brisnet 2-digit year of birth (e.g. 22 for 2022). Always 20xx in practice."""
    if yy is None:
        return None
    return 2000 + yy


def _sub_nonneg(a: str, *subtract: str) -> int | None:
    """Return a - sum(subtract), treating blanks in subtrahends as 0. Floors at 0."""
    total = _int(a)
    if total is None:
        return None
    for s in subtract:
        total -= _int(s) or 0
    return max(total, 0)


def _f(row: list[str], n: int) -> str:
    """Get 1-indexed field; empty string if out of range."""
    idx = n - 1
    if idx < 0 or idx >= len(row):
        return ""
    return row[idx]


def _int(s: str) -> int | None:
    s = s.strip()
    if not s:
        return None
    try:
        # some numeric fields come through as floats (e.g. "1.0")
        return int(float(s))
    except ValueError:
        return None


def _float(s: str) -> float | None:
    s = s.strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _abs_int(s: str) -> int | None:
    """Negative Brisnet distances denote 'about' — take the magnitude."""
    val = _int(s)
    return abs(val) if val is not None else None


def _date(s: str) -> date | None:
    """Parse YYYYMMDD -> date. Returns None for empty/invalid."""
    s = s.strip()
    if len(s) != 8:
        return None
    try:
        return date(int(s[:4]), int(s[4:6]), int(s[6:8]))
    except ValueError:
        return None
