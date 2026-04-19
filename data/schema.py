"""Shared helpers for data ingestion."""

FURLONGS_TO_YARDS = 220
METERS_TO_YARDS = 1.09361


def make_race_id(date: str, track: str, race_number: int) -> str:
    """Construct a unique race identifier, e.g. '2023-04-29_CD_R5'."""
    return f"{date}_{track}_R{race_number}"


def parse_odds(odds_str: str | None) -> float | None:
    """Parse odds string to decimal (dollars-to-one) float.

    Handles:
      - Fractional: "5/1" → 5.0, "3/2" → 1.5
      - Integer-coded (PPS past performances): "5875" → 58.75
      - Decimal string: "4.50" → 4.5
      - Empty/None/zero → None
    """
    if not odds_str or not odds_str.strip():
        return None
    s = odds_str.strip()
    if "/" in s:
        num, denom = s.split("/", 1)
        val = safe_float(num) / safe_float(denom)
    else:
        val = safe_float(s)
        if val is None:
            return None
        # Integer-coded odds from PPS PastPerformance: stored as int * 100
        # e.g. 500 = 5.00, 5875 = 58.75. These are always >= 1 when decoded.
        # Distinguish from decimal odds: if it looks like a whole number > 99,
        # it's almost certainly integer-coded.
        if val == int(val) and val > 99:
            val = val / 100
    # 0 is never a real odds value (floor is ~0.05); treat as missing
    return val if val > 0 else None


def to_yards(value: int | None, unit: str | None) -> int | None:
    """
    Convert an Equibase distance value + unit to yards (rounded to nearest int).

    Unit encodings in Equibase:
    - F = furlongs, scaled at 1/100 (e.g., "100 F" means 1 furlong
    - Y = yards
    - M = meters
    """
    if value is None or unit is None:
        return None
    if unit == "F":
        return round(value / 100 * FURLONGS_TO_YARDS)
    if unit == "Y":
        return value
    if unit == "M":
        return round(value * METERS_TO_YARDS)
    return None


def safe_int(value: str | None) -> int | None:
    """Convert XML text to int, returning None for missing/empty/invalid."""
    if not value or not value.strip():
        return None
    try:
        return int(value.strip())
    except ValueError:
        return None


def safe_float(value: str | None) -> float | None:
    """Convert XML text to float, returning None for missing/empty/invalid."""
    if not value or not value.strip():
        return None
    try:
        return float(value.strip())
    except ValueError:
        return None


def xml_text(element, path: str) -> str | None:
    """Extract text from an XML subelement, or None if missing."""
    child = element.find(path)
    if child is not None and child.text and child.text.strip():
        return child.text.strip()
    return None
