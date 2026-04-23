"""Parse a Brisnet CSV + apply manual overrides -> flat staged CSV.

The staged CSV has one row per runner with race-level fields repeated across the
runners of the same race. Columns are ordered: race-level first, then every
`StaticRunnerInput` field in declaration order, so the companion `post_staged_races`
script can build `CreateRaceRequest` payloads directly from it.

Overrides CSV: one row per runner, keyed on (race_number, post_position). All
columns below are required; extra columns (e.g. `horse_name` for data entry
convenience) are ignored. `race_class_rating` must be non-null for every runner;
`entry_class_rating` and `speed_fig_L{1,2,3}` cells may be blank, in which case
the blank overwrites the BRIS value with null — the override is always authoritative.

Overrides CSV format:

| race_number | post_position | race_class_rating | entry_class_rating | speed_fig_L1 | speed_fig_L2 | speed_fig_L3 |
|-------------|---------------|-------------------|--------------------|--------------|--------------|--------------|
| 1           | 1             | 85                | 450                | 76           | 72           | 68           |
| 1           | 2             | 85                | 480                | 82           | 78           |              |

Staged CSV format (abbreviated; 52 cols total = 10 race-level + `StaticRunnerInput`):

| track | race_number | race_date  | race_class_rating | horse_name     | post_position | entry_class_rating | avg_speed_fig_L3 |
|-------|-------------|------------|-------------------|----------------|---------------|--------------------|------------------|
| CD    | 1           | 2026-04-25 | 85                | TOO MANY MIKES | 1             | 450                | 72.0             |
| CD    | 1           | 2026-04-25 | 85                | ISLAND GIRL    | 2             | 480                | 80.0             |

Usage:
    python -m scripts.ingest_brisnet \\
        data/raw/brisnet/CDX0425.csv \\
        --overrides data/raw/overrides/CDX0425_overrides.csv
"""

import argparse
import logging
from pathlib import Path

import polars as pl

from api.schemas import StaticRunnerInput
from data.parse_brisnet import parse_brisnet_csv
from model.features import (
    _workout_features,
    aggregate_pp_features,
    derive_pp_rollup_features,
)

logger = logging.getLogger(__name__)

DEFAULT_STAGING_DIR = Path("data/staging")

RACE_LEVEL_COLS: list[str] = [
    "track",
    "race_number",
    "race_date",
    "distance_yards",
    "surface",
    "course_desc",
    "race_class_rating",
    "purse",
    "age_restriction",
    "sex_restriction",
]

REQUIRED_OVERRIDE_COLS = {
    "race_number",
    "post_position",
    "race_class_rating",
    "entry_class_rating",
    "speed_fig_L1",
    "speed_fig_L2",
    "speed_fig_L3",
}


def ingest(csv_path: Path, overrides_path: Path, out_path: Path):
    entries_dicts, pp_dicts, workout_dicts = parse_brisnet_csv(csv_path)
    logger.info(
        f"parsed {len(entries_dicts)} entries, {len(pp_dicts)} PP rows, "
        f"{len(workout_dicts)} workout rows from {csv_path}"
    )

    entries = pl.DataFrame(entries_dicts)
    pp = pl.DataFrame(pp_dicts) if pp_dicts else _empty_pp()
    workouts = pl.DataFrame(workout_dicts) if workout_dicts else _empty_workouts()

    overrides = _read_overrides(overrides_path)
    _validate_one_to_one(entries, overrides)
    entries = _apply_entry_overrides(entries, overrides)
    _validate_required(entries)

    pp_feats = (
        derive_pp_rollup_features(aggregate_pp_features(pp)) if pp.height > 0 else None
    )
    workout_feats = _workout_features(workouts) if workouts.height > 0 else None

    runners = entries
    if pp_feats is not None:
        runners = runners.join(pp_feats, on=["race_id", "horse_name"], how="left")
    if workout_feats is not None:
        runners = runners.join(workout_feats, on=["race_id", "horse_name"], how="left")

    runners = _apply_speed_fig_overrides(runners, overrides)

    # parser emits both a raw-string `morning_line_odds` and float `morning_line_odds_float`;
    # keep only the float, under the StaticRunnerInput name
    if "morning_line_odds_float" in runners.columns:
        runners = runners.drop("morning_line_odds").rename(
            {"morning_line_odds_float": "morning_line_odds"}
        )

    # ensure every column the staged CSV needs exists before selecting
    out_cols = RACE_LEVEL_COLS + list(StaticRunnerInput.model_fields.keys())
    runners = runners.with_columns(
        [pl.lit(None).alias(c) for c in out_cols if c not in runners.columns]
    )

    runners = runners.with_columns(
        pl.col("num_prior_starts").fill_null(0),
        pl.col("num_workouts").fill_null(0),
    )

    staged = runners.select(out_cols).sort("race_number", "post_position")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    staged.write_csv(out_path, null_value="")
    logger.info(f"wrote {staged.height} rows x {staged.width} cols -> {out_path}")


def _empty_pp() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "race_id": pl.Utf8,
            "horse_name": pl.Utf8,
            "pp_index": pl.Int64,
            "pp_race_date": pl.Date,
            "pp_distance_yards": pl.Int64,
            "pp_surface": pl.Utf8,
            "pp_num_starters": pl.Int64,
            "pp_official_finish": pl.Int64,
            "pp_speed_figure": pl.Int64,
            "pp_odds": pl.Float64,
            "pp_class_rating": pl.Int64,
        }
    )


def _empty_workouts() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "race_id": pl.Utf8,
            "horse_name": pl.Utf8,
            "workout_date": pl.Date,
            "workout_ranking": pl.Int64,
            "workout_num_in_group": pl.Int64,
        }
    )


def _read_overrides(path: Path) -> pl.DataFrame:
    overrides = pl.read_csv(path, infer_schema_length=1000)
    missing = REQUIRED_OVERRIDE_COLS - set(overrides.columns)
    if missing:
        raise ValueError(
            f"overrides CSV is missing required columns: {sorted(missing)}"
        )
    return overrides


def _validate_one_to_one(entries: pl.DataFrame, overrides: pl.DataFrame):
    """Raise if entries and overrides aren't 1:1 on (race_number, post_position)."""
    key_cols = ["race_number", "post_position"]
    entry_keys = entries.select(key_cols)
    override_keys = overrides.select(key_cols)

    missing = entry_keys.join(override_keys, on=key_cols, how="anti").sort(key_cols)
    if missing.height > 0:
        raise ValueError(f"no override row for {missing.height} runner(s):\n{missing}")

    extra = override_keys.join(entry_keys, on=key_cols, how="anti").sort(key_cols)
    if extra.height > 0:
        raise ValueError(
            f"{extra.height} override row(s) do not match any runner:\n{extra}"
        )


def _apply_entry_overrides(
    entries: pl.DataFrame, overrides: pl.DataFrame
) -> pl.DataFrame:
    """Merge race_class_rating + entry_class_rating from overrides into entries."""
    return entries.join(
        overrides.select(
            "race_number", "post_position", "race_class_rating", "entry_class_rating"
        ),
        on=["race_number", "post_position"],
        how="left",
    )


def _apply_speed_fig_overrides(
    runners: pl.DataFrame, overrides: pl.DataFrame
) -> pl.DataFrame:
    """Overwrite speed_fig_L{1,2,3} rollups from overrides (authoritative incl. null)
    and recompute avg/max from the overridden values.
    """
    speed_cols = ["speed_fig_L1", "speed_fig_L2", "speed_fig_L3"]
    runners = runners.drop([c for c in speed_cols if c in runners.columns]).join(
        overrides.select(["race_number", "post_position"] + speed_cols),
        on=["race_number", "post_position"],
        how="left",
    )
    return runners.with_columns(
        pl.mean_horizontal(speed_cols).alias("avg_speed_fig_L3"),
        pl.max_horizontal(speed_cols).alias("max_speed_fig_L3"),
    )


def _validate_required(entries: pl.DataFrame):
    """Null race_class_rating cells and within-race rcr disagreements.

    entry_class_rating is allowed to be null (some runners have no rating on the PDF).
    """
    missing = entries.filter(pl.col("race_class_rating").is_null())
    if missing.height > 0:
        rows = missing.select("race_number", "post_position", "horse_name").sort(
            "race_number", "post_position"
        )
        raise ValueError(f"race_class_rating blank for runners:\n{rows}")

    dup = (
        entries.group_by("race_id")
        .agg(pl.col("race_class_rating").n_unique().alias("n_distinct"))
        .filter(pl.col("n_distinct") > 1)
    )
    if dup.height > 0:
        raise ValueError(
            f"race_class_rating disagrees within race(s): {dup['race_id'].to_list()}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Parse a Brisnet single-track CSV + overrides into a staged race-day CSV.",
    )
    parser.add_argument(
        "csv_path",
        type=Path,
        help="Path to Brisnet CSV (e.g. data/raw/brisnet/CDX0425.csv)",
    )
    parser.add_argument(
        "--overrides",
        type=Path,
        required=True,
        help="Path to overrides CSV",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"Output path (default: {DEFAULT_STAGING_DIR}/<csv_stem>.csv)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    out = args.out or (DEFAULT_STAGING_DIR / f"{args.csv_path.stem}.csv")
    ingest(args.csv_path, args.overrides, out)


if __name__ == "__main__":
    main()
