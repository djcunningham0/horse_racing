"""Parse Equibase Parameters Excel file for track code lookups."""

from pathlib import Path

import polars as pl


def parse_track_codes(xlsx_path: Path) -> pl.DataFrame:
    """Read track codes from the 'Track codes' sheet of the Equibase Parameters file."""
    import openpyxl

    wb = openpyxl.load_workbook(xlsx_path, read_only=True)
    ws = wb["Track codes"]

    rows = []
    for i, row in enumerate(ws.iter_rows(values_only=True)):
        if i == 0:  # skip header
            continue
        country, track_id, track_name, state = row
        if track_id:
            rows.append({
                "country": country,
                "track_id": track_id,
                "track_name": track_name,
                "state": state,
            })

    wb.close()
    schema = {"country": pl.Utf8, "track_id": pl.Utf8, "track_name": pl.Utf8, "state": pl.Utf8}
    return pl.DataFrame(rows, schema=schema)
