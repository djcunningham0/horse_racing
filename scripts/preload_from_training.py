"""Pre-load races from the training-data parquet files into a running API.

Picks one date + track from training data, assembles `CreateRaceRequest` payloads
for each race on that card, and POSTs them to `/races`. Useful for end-to-end UX
testing against realistic data. Backtest-only: pulls `race_class_rating` and
`course_desc` from `results.parquet`, which won't exist on real race day.

Usage:
    python -m scripts.preload_from_training --date 2023-05-06 --track CD
"""

import argparse
import logging
import os
from datetime import date
from pathlib import Path

import httpx
import polars as pl

from api.schemas import CreateRaceRequest, StaticRunnerInput
from model.features import (
    EXCLUDED_COURSE_DESCS,
    _workout_features,
    aggregate_pp_features,
)
from model.paths import DEFAULT_PROCESSED_DIR

logger = logging.getLogger(__name__)


def build_preload_df(
    processed_dir: Path,
    race_date: date,
    track: str,
) -> pl.DataFrame:
    """Assemble one row per (race_id, horse_name) with all fields needed to build
    a `CreateRaceRequest`. Filters out excluded course types (jump/timber/downhill)."""
    results = pl.read_parquet(processed_dir / "results.parquet")
    entries = pl.read_parquet(processed_dir / "entries.parquet")
    pp = pl.read_parquet(processed_dir / "past_performances.parquet")
    workouts = pl.read_parquet(processed_dir / "workouts.parquet")

    day_results = results.filter(
        (pl.col("race_date") == race_date)
        & (pl.col("track") == track)
        & ~pl.col("course_desc").is_in(EXCLUDED_COURSE_DESCS)
    )
    race_ids = day_results["race_id"].unique().to_list()
    if not race_ids:
        raise ValueError(f"No races found for {race_date} at {track}")

    day_entries = entries.filter(pl.col("race_id").is_in(race_ids))
    day_pp = pp.filter(pl.col("race_id").is_in(race_ids))
    day_workouts = workouts.filter(pl.col("race_id").is_in(race_ids))

    pp_feats = aggregate_pp_features(day_pp)
    workout_feats = _workout_features(day_workouts)

    # fmt: off
    race_cols = (
        day_results
        .select(
            "race_id", "race_date", "track", "race_number",
            "distance_yards", "surface", "course_desc", "purse",
            pl.col("class_rating").cast(pl.Float64).alias("race_class_rating"),
        )
        .unique(subset=["race_id"])
    )

    entry_cols = day_entries.select(
        "race_id",
        "horse_name",
        "post_position",
        "weight_carried",
        pl.col("morning_line_odds_float").alias("morning_line_odds"),
        pl.col("class_rating").cast(pl.Float64).alias("entry_class_rating"),
        "year_of_birth",
        "sex",
        # race-level (constant within race) — pulled here so they flow through the join
        "age_restriction",
        "sex_restriction",
        "career_starts", "career_wins", "career_seconds", "career_thirds",
        "career_earnings",
        "surface_starts", "surface_wins", "surface_seconds", "surface_thirds",
    )

    return (
        race_cols
        .join(entry_cols, on="race_id", how="inner")
        .join(pp_feats, on=["race_id", "horse_name"], how="left")
        .join(workout_feats, on=["race_id", "horse_name"], how="left")
        .with_columns(
            pl.col("num_prior_starts").fill_null(0),
            pl.col("num_workouts").fill_null(0),
        )
        .sort("race_number", "post_position")
    )
    # fmt: on


def to_create_race_request(race_df: pl.DataFrame) -> CreateRaceRequest:
    """Build a `CreateRaceRequest` from a DataFrame of one race's runners."""
    first = race_df.row(0, named=True)
    runner_fields = set(StaticRunnerInput.model_fields.keys())
    runners = [
        StaticRunnerInput(**{k: v for k, v in row.items() if k in runner_fields})
        for row in race_df.iter_rows(named=True)
    ]
    return CreateRaceRequest(
        track=first["track"],
        race_number=first["race_number"],
        race_date=first["race_date"],
        distance_yards=first["distance_yards"],
        surface=first["surface"],
        course_desc=first["course_desc"],
        race_class_rating=first["race_class_rating"],
        purse=first["purse"],
        age_restriction=first["age_restriction"],
        sex_restriction=first["sex_restriction"],
        runners=runners,
    )


def preload(
    base_url: str,
    processed_dir: Path,
    race_date: date,
    track: str,
):
    df = build_preload_df(processed_dir, race_date, track)
    race_ids = df["race_id"].unique(maintain_order=True).to_list()
    logger.info(f"Found {len(race_ids)} races for {race_date} at {track}")

    user, pwd = os.environ.get("APP_USERNAME"), os.environ.get("APP_PASSWORD")
    auth = (user, pwd) if user and pwd else None
    with httpx.Client(base_url=base_url, timeout=10.0, auth=auth) as client:
        for race_id in race_ids:
            race_df = df.filter(pl.col("race_id") == race_id)
            payload = to_create_race_request(race_df).model_dump(mode="json")
            response = client.post("/races", json=payload)
            if response.status_code == 409:
                logger.warning(f"  skipped {race_id} (already exists)")
                continue
            response.raise_for_status()
            created_id = response.json()["race_id"]
            logger.info(f"  created {created_id} ({race_df.height} runners)")


def main():
    parser = argparse.ArgumentParser(
        description="Pre-load races from training data into a running API.",
    )
    parser.add_argument("--date", type=date.fromisoformat, required=True)
    parser.add_argument("--track", required=True)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    preload(
        base_url=args.base_url,
        processed_dir=args.processed_dir,
        race_date=args.date,
        track=args.track,
    )


if __name__ == "__main__":
    main()
