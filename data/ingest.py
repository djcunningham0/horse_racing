"""Equibase data ingestion pipeline: raw XML/ZIP → Parquet.

Usage:
    python -m data.ingest [--raw-dir DIR] [--output-dir DIR]
"""

import argparse
import logging
import time
from pathlib import Path

import polars as pl

from data.parse_entries import parse_pps_zip
from data.parse_params import parse_track_codes
from data.parse_results import parse_result_chart

logger = logging.getLogger(__name__)

DEFAULT_RAW_DIR = Path("data/raw/equibase")
DEFAULT_OUTPUT_DIR = Path("data/processed")

# Explicit schemas to avoid inference issues with sparse/null-heavy columns.
# All string-like fields use Utf8; numeric fields that can be null use nullable types.

RESULTS_SCHEMA = {
    "race_id": pl.Utf8,
    "race_date": pl.Date,
    "track": pl.Utf8,
    "race_number": pl.Int64,
    "breed": pl.Utf8,
    "race_type": pl.Utf8,
    "course_id": pl.Utf8,
    "course_desc": pl.Utf8,
    "run_up_distance": pl.Int64,
    "purse": pl.Float64,
    "distance_val": pl.Int64,
    "distance_unit": pl.Utf8,
    "distance_yards": pl.Int64,
    "surface": pl.Utf8,
    "track_condition": pl.Utf8,
    "weather": pl.Utf8,
    "class_rating": pl.Int64,
    "num_runners": pl.Int64,
    "win_time": pl.Utf8,
    "fraction_1": pl.Float64,
    "fraction_2": pl.Float64,
    "fraction_3": pl.Float64,
    "fraction_4": pl.Float64,
    "fraction_5": pl.Float64,
    "pace_call_1": pl.Utf8,
    "pace_call_2": pl.Utf8,
    "pace_final": pl.Utf8,
    "footnotes": pl.Utf8,
    "horse_name": pl.Utf8,
    "program_number": pl.Utf8,
    "post_position": pl.Int64,
    "weight": pl.Int64,
    "age": pl.Int64,
    "sex": pl.Utf8,
    "medication": pl.Utf8,
    "equipment": pl.Utf8,
    "claim_price": pl.Float64,
    "dollar_odds": pl.Float64,
    "official_finish": pl.Int64,
    "finish_time": pl.Utf8,
    "speed_rating": pl.Int64,
    "comment": pl.Utf8,
    "dh_dq_flags": pl.Utf8,
    "jockey_first_name": pl.Utf8,
    "jockey_last_name": pl.Utf8,
    "trainer_first_name": pl.Utf8,
    "trainer_last_name": pl.Utf8,
    "win_payoff": pl.Float64,
    "place_payoff": pl.Float64,
    "show_payoff": pl.Float64,
}

ENTRIES_SCHEMA = {
    "race_id": pl.Utf8,
    "race_date": pl.Date,
    "track": pl.Utf8,
    "race_number": pl.Int64,
    "breed": pl.Utf8,
    "race_type": pl.Utf8,
    "race_type_desc": pl.Utf8,
    "surface": pl.Utf8,
    "distance_val": pl.Int64,
    "distance_unit": pl.Utf8,
    "distance_yards": pl.Int64,
    "purse": pl.Float64,
    "grade": pl.Utf8,
    "num_runners": pl.Int64,
    "condition_text": pl.Utf8,
    "max_claim_price": pl.Float64,
    "age_restriction": pl.Utf8,
    "sex_restriction": pl.Utf8,
    "horse_name": pl.Utf8,
    "registration_number": pl.Utf8,
    "year_of_birth": pl.Int64,
    "sex": pl.Utf8,
    "sire_name": pl.Utf8,
    "dam_sire_name": pl.Utf8,
    "post_position": pl.Int64,
    "program_number": pl.Utf8,
    "morning_line_odds": pl.Utf8,
    "morning_line_odds_float": pl.Float64,
    "weight_carried": pl.Int64,
    "jockey_first_name": pl.Utf8,
    "jockey_last_name": pl.Utf8,
    "trainer_first_name": pl.Utf8,
    "trainer_last_name": pl.Utf8,
    "equipment": pl.Utf8,
    "medication": pl.Utf8,
    "apprentice_weight_allowance": pl.Int64,
    "class_rating": pl.Float64,
    "career_starts": pl.Int64,
    "career_wins": pl.Int64,
    "career_seconds": pl.Int64,
    "career_thirds": pl.Int64,
    "career_earnings": pl.Float64,
    "surface_starts": pl.Int64,
    "surface_wins": pl.Int64,
    "surface_seconds": pl.Int64,
    "surface_thirds": pl.Int64,
}

PAST_PERFORMANCES_SCHEMA = {
    "race_id": pl.Utf8,
    "horse_name": pl.Utf8,
    "registration_number": pl.Utf8,
    "pp_index": pl.Int64,
    "pp_race_date": pl.Date,
    "pp_track": pl.Utf8,
    "pp_race_number": pl.Int64,
    "pp_race_type": pl.Utf8,
    "pp_distance_val": pl.Int64,
    "pp_distance_unit": pl.Utf8,
    "pp_distance_yards": pl.Int64,
    "pp_surface": pl.Utf8,
    "pp_track_condition": pl.Utf8,
    "pp_num_starters": pl.Int64,
    "pp_purse": pl.Float64,
    "pp_post_position": pl.Int64,
    "pp_official_finish": pl.Int64,
    "pp_speed_figure": pl.Int64,
    "pp_odds": pl.Float64,
    "pp_weight_carried": pl.Int64,
    "pp_class_rating": pl.Int64,
    "pp_jockey_last_name": pl.Utf8,
    "pp_trainer_last_name": pl.Utf8,
    "pp_long_comment": pl.Utf8,
    "pp_short_comment": pl.Utf8,
    "pp_pace_figure_1": pl.Int64,
    "pp_pace_figure_2": pl.Int64,
    "pp_pace_figure_3": pl.Int64,
    "pp_poc_start_pos": pl.Int64,
    "pp_poc_1_pos": pl.Int64,
    "pp_poc_1_behind": pl.Int64,
    "pp_poc_2_pos": pl.Int64,
    "pp_poc_2_behind": pl.Int64,
    "pp_poc_3_pos": pl.Int64,
    "pp_poc_3_behind": pl.Int64,
    "pp_poc_4_pos": pl.Int64,
    "pp_poc_4_behind": pl.Int64,
    "pp_poc_5_pos": pl.Int64,
    "pp_poc_5_behind": pl.Int64,
    "pp_poc_final_pos": pl.Int64,
    "pp_poc_final_behind": pl.Int64,
}

WORKOUTS_SCHEMA = {
    "race_id": pl.Utf8,
    "horse_name": pl.Utf8,
    "registration_number": pl.Utf8,
    "workout_date": pl.Date,
    "workout_track": pl.Utf8,
    "workout_distance_val": pl.Int64,
    "workout_distance_unit": pl.Utf8,
    "workout_distance_yards": pl.Int64,
    "workout_time": pl.Utf8,
    "workout_type": pl.Utf8,
    "workout_course": pl.Utf8,
    "workout_track_condition": pl.Utf8,
    "workout_ranking": pl.Int32,
    "workout_num_in_group": pl.Int32,
    "workout_comment": pl.Utf8,
}


def _to_dataframe(rows: list[dict], schema: dict) -> pl.DataFrame:
    """Build a Polars DataFrame from dicts with an explicit schema.

    Point-of-call columns (dynamic keys from result charts) are auto-detected
    across all rows and added as the appropriate type. Each dict is filtered
    to only include keys present in the final schema.
    """
    if not rows:
        return pl.DataFrame(schema=schema)

    # Detect extra columns across all rows (e.g. point_of_call_* from results)
    all_keys: set[str] = set()
    for row in rows:
        all_keys.update(row.keys())

    full_schema = {**schema}
    for key in sorted(all_keys - set(schema.keys())):
        if "position" in key:
            full_schema[key] = pl.Int64
        else:
            full_schema[key] = pl.Utf8

    return pl.DataFrame(rows, schema=full_schema)


def ingest_results(raw_dir: Path, output_dir: Path):
    """Parse all result chart XMLs and write results.parquet."""
    chart_dir = raw_dir / "2023_result_charts"
    xml_files = sorted(chart_dir.glob("*tch.xml"))
    logger.info(f"Found {len(xml_files)} result chart files")

    rows = []
    errors = 0
    for i, path in enumerate(xml_files, 1):
        try:
            rows.extend(parse_result_chart(path))
        except Exception:
            errors += 1
            logger.warning(f"Failed to parse {path.name}", exc_info=True)
        if i % 500 == 0:
            logger.info(
                f"  Parsed {i}/{len(xml_files)} result charts ({len(rows)} rows)"
            )

    df = _to_dataframe(rows, RESULTS_SCHEMA)
    out_path = output_dir / "results.parquet"
    df.write_parquet(out_path)
    logger.info(
        f"Wrote {out_path}: {df.shape[0]:,} rows, {df.shape[1]} cols, "
        f"{out_path.stat().st_size / 1e6:.1f} MB ({errors} parse errors)"
    )


def ingest_entries(raw_dir: Path, output_dir: Path):
    """Parse all PPS ZIPs and write entries, past_performances, workouts parquets."""
    pps_dir = raw_dir / "2023_pps"
    zip_files = sorted(pps_dir.glob("*.zip"))
    logger.info(f"Found {len(zip_files)} PPS ZIP files")

    all_entries = []
    all_pps = []
    all_workouts = []
    errors = 0

    for i, path in enumerate(zip_files, 1):
        try:
            entries, pps, workouts = parse_pps_zip(path)
            all_entries.extend(entries)
            all_pps.extend(pps)
            all_workouts.extend(workouts)
        except Exception:
            errors += 1
            logger.warning(f"Failed to parse {path.name}", exc_info=True)
        if i % 500 == 0:
            logger.info(
                f"  Parsed {i}/{len(zip_files)} PPS files "
                f"({len(all_entries)} entries, {len(all_pps)} PPs, {len(all_workouts)} "
                f"workouts)"
            )

    for name, data, schema in [
        ("entries", all_entries, ENTRIES_SCHEMA),
        ("past_performances", all_pps, PAST_PERFORMANCES_SCHEMA),
        ("workouts", all_workouts, WORKOUTS_SCHEMA),
    ]:
        df = _to_dataframe(data, schema)
        out_path = output_dir / f"{name}.parquet"
        df.write_parquet(out_path)
        logger.info(
            f"Wrote {out_path}: {df.shape[0]:,} rows, {df.shape[1]} cols, "
            f"{out_path.stat().st_size / 1e6:.1f} MB"
        )

    logger.info(f"PPS ingestion complete ({errors} parse errors)")


def ingest_params(raw_dir: Path, output_dir: Path):
    """Parse Equibase Parameters Excel and write track_codes.parquet."""
    xlsx_path = raw_dir / "Equibase Parameters.xlsx"
    if not xlsx_path.exists():
        logger.warning(f"Parameters file not found: {xlsx_path}")
        return

    df = parse_track_codes(xlsx_path)
    out_path = output_dir / "track_codes.parquet"
    df.write_parquet(out_path)
    logger.info(f"Wrote {out_path}: {df.shape[0]:,} rows")


def main():
    parser = argparse.ArgumentParser(
        description="Ingest Equibase data into Parquet files"
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    ingest_results(args.raw_dir, args.output_dir)
    ingest_entries(args.raw_dir, args.output_dir)
    ingest_params(args.raw_dir, args.output_dir)
    elapsed = time.time() - t0

    logger.info(f"Done in {elapsed:.0f}s")

    # Print summary
    print("\n=== Summary ===")
    for f in sorted(args.output_dir.glob("*.parquet")):
        df = pl.read_parquet(f)
        size_mb = f.stat().st_size / 1e6
        print(f"  {f.name}: {df.shape[0]:,} rows, {df.shape[1]} cols, {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
