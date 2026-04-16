"""Build the modeling dataset from processed parquet files."""

from pathlib import Path

import numpy as np
import polars as pl

DEFAULT_PROCESSED_DIR = Path("data/processed")

# Surface encoding (results.surface is "D" or "T").
SURFACE_MAP = {"D": 0, "T": 1}

DEFAULT_FEATURE_COLS: list[str] = [
    "morning_line_odds_float",
    "post_position",
    "weight_carried",
    "field_size",
    "distance",
    "surface_int",
    "class_rating",
    "speed_fig_L1",
    "speed_fig_L2",
    "speed_fig_L3",
    "avg_speed_fig_L3",
    "days_since_last",
    "num_prior_starts",
    "is_first_start",
]


def _pp_features(pp: pl.DataFrame) -> pl.DataFrame:
    """Aggregate past-performance rows into one row per (race_id, horse_name).

    pp_index == 1 is the most recent prior race.
    """
    return (
        pp.with_columns(pl.col("pp_race_date").str.to_date())
        .group_by(["race_id", "horse_name"])
        .agg(
            pl.col("pp_speed_figure")
            .filter(pl.col("pp_index") == 1)
            .first()
            .alias("speed_fig_L1"),
            pl.col("pp_speed_figure")
            .filter(pl.col("pp_index") == 2)
            .first()
            .alias("speed_fig_L2"),
            pl.col("pp_speed_figure")
            .filter(pl.col("pp_index") == 3)
            .first()
            .alias("speed_fig_L3"),
            pl.col("pp_speed_figure")
            .filter(pl.col("pp_index") <= 3)
            .mean()
            .alias("avg_speed_fig_L3"),
            pl.col("pp_race_date")
            .filter(pl.col("pp_index") == 1)
            .first()
            .alias("last_pp_date"),
            pl.len().alias("num_prior_starts"),
        )
    )


def build_training_df(
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
) -> pl.DataFrame:
    """Assemble one row per horse-in-a-race with features and label."""
    entries = pl.read_parquet(processed_dir / "entries.parquet")
    results = pl.read_parquet(processed_dir / "results.parquet")
    pp = pl.read_parquet(processed_dir / "past_performances.parquet")

    race_cols = results.select(
        "race_id",
        "race_date",
        "track",
        "race_number",
        "distance",
        "surface",
        "num_runners",
        "horse_name",
        "official_finish",
        "dollar_odds",
    ).with_columns(pl.col("race_date").str.to_date())

    entry_cols = entries.select(
        "race_id",
        "horse_name",
        "morning_line_odds_float",
        "post_position",
        "weight_carried",
        "class_rating",
    )

    pp_feats = _pp_features(pp)

    df = (
        race_cols.join(entry_cols, on=["race_id", "horse_name"], how="inner")
        .join(pp_feats, on=["race_id", "horse_name"], how="left")
        .with_columns(
            (pl.col("race_date") - pl.col("last_pp_date"))
            .dt.total_days()
            .alias("days_since_last"),
            pl.col("num_prior_starts").fill_null(0),
            pl.col("surface")
            .replace_strict(SURFACE_MAP, default=None)
            .alias("surface_int"),
            pl.col("num_runners").alias("field_size"),
            (pl.col("official_finish") == 1).cast(pl.Int8).alias("won"),
        )
        .with_columns(
            (pl.col("num_prior_starts") == 0).cast(pl.Int8).alias("is_first_start"),
        )
        .drop("last_pp_date")
    )
    return df


def split_by_race(
    df: pl.DataFrame,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 0,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Random split by race_id. All rows of a given race stay together.

    See plan: random (not chronological) because only 2023 data is available,
    so a time split would confound seasonal effects with model quality. Safe
    for the current baseline feature set — none of the features are keyed on
    horse/jockey/trainer identity, so no cross-split leakage.
    """
    race_ids = np.sort(df["race_id"].unique().to_numpy())
    rng = np.random.default_rng(seed)
    rng.shuffle(race_ids)
    n = len(race_ids)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    test_ids = race_ids[:n_test]
    val_ids = race_ids[n_test : n_test + n_val]
    train_ids = race_ids[n_test + n_val :]
    return (
        df.filter(pl.col("race_id").is_in(train_ids)),
        df.filter(pl.col("race_id").is_in(val_ids)),
        df.filter(pl.col("race_id").is_in(test_ids)),
    )
