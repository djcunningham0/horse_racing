"""Build the modeling dataset from processed parquet files."""

from pathlib import Path

import numpy as np
import polars as pl

DEFAULT_PROCESSED_DIR = Path("data/processed")

DEFAULT_FEATURE_COLS: list[str] = [
    # odds
    "morning_line_odds_float",
    "dollar_odds_plus_noise",
    # race characteristics
    "field_size",
    "distance",
    "is_turf",
    "race_class_rating",  # scale: 20-100+
    "purse",
    # entry characteristics
    "post_position",
    "weight_carried",
    "entry_class_rating",  # scale: 300-800+
    "entry_class_rating_minus_field_avg",
    "entry_class_rating_to_field_avg_ratio",
    "entry_to_race_class_ratio",
    # PP speed
    "speed_fig_L1",
    "speed_fig_L2",
    "speed_fig_L3",
    "avg_speed_fig_L3",
    "max_speed_fig_L3",
    "speed_fig_trend",
    "speed_fig_minus_field_avg_L1",
    "speed_fig_to_field_avg_ratio_L1",
    # PP class
    "class_rating_L1",  # scale: 20-100+
    "class_rating_L2",
    "class_rating_L3",
    "avg_class_rating_L3",
    "max_class_rating_L3",
    "class_rating_diff_L1",
    "class_rating_diff_avg_L3",
    "class_rating_diff_max_L3",
    # PP finishes
    "official_finish_L1",
    "official_finish_L2",
    "official_finish_L3",
    "relative_finish_L1",
    "relative_finish_L2",
    "relative_finish_L3",
    "avg_relative_finish",
    # PP distance
    "distance_diff_L1",
    "distance_diff_L2",
    "distance_diff_L3",
    # PP prior starts
    "days_since_last",
    "num_prior_starts",
    "is_first_start",
    # workouts
    "best_workout_rank_pct",
    "best_workout_group_size",
    "last_workout_rank_pct",
    "last_workout_group_size",
    "num_workouts",
    "days_since_last_workout",
]


def _workout_features(workouts: pl.DataFrame) -> pl.DataFrame:
    """Aggregate workout rows into one row per (race_id, horse_name)."""
    # fmt: off
    return (
        workouts
        .with_columns(
            (pl.col("workout_ranking") / pl.col("workout_num_in_group")).alias("rank_pct"),
        )
        .group_by(["race_id", "horse_name"])
        .agg(
            pl.col("rank_pct").min().alias("best_workout_rank_pct"),
            pl.col("workout_num_in_group").sort_by("rank_pct").first().alias("best_workout_group_size"),
            pl.col("rank_pct").sort_by("workout_date").last().alias("last_workout_rank_pct"),
            pl.col("workout_num_in_group").sort_by("workout_date").last().alias("last_workout_group_size"),
            pl.len().alias("num_workouts"),
            pl.col("workout_date").max().alias("last_workout_date"),
        )
    )
    # fmt: on


def _pp_features(pp: pl.DataFrame) -> pl.DataFrame:
    """Aggregate past-performance rows into one row per (race_id, horse_name).

    pp_index == 1 is the most recent prior race.
    """
    # fmt: off
    return (
        pp
        .group_by(["race_id", "horse_name"])
        .agg(
            # speed
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
            pl.col("pp_speed_figure")
            .filter(pl.col("pp_index") <= 3)
            .max()
            .alias("max_speed_fig_L3"),

            # class rating
            pl.col("pp_class_rating")
            .filter(pl.col("pp_index") == 1)
            .first()
            .alias("class_rating_L1"),
            pl.col("pp_class_rating")
            .filter(pl.col("pp_index") == 2)
            .first()
            .alias("class_rating_L2"),
            pl.col("pp_class_rating")
            .filter(pl.col("pp_index") == 3)
            .first()
            .alias("class_rating_L3"),
            pl.col("pp_class_rating")
            .filter(pl.col("pp_index") <= 3)
            .mean()
            .alias("avg_class_rating_L3"),
            pl.col("pp_class_rating")
            .filter(pl.col("pp_index") <= 3)
            .max()
            .alias("max_class_rating_L3"),

            # official finish
            pl.col("pp_official_finish")
            .filter(pl.col("pp_index") == 1)
            .first()
            .alias("official_finish_L1"),
            pl.col("pp_official_finish")
            .filter(pl.col("pp_index") == 2)
            .first()
            .alias("official_finish_L2"),
            pl.col("pp_official_finish")
            .filter(pl.col("pp_index") == 3)
            .first()
            .alias("official_finish_L3"),

            # num starters
            pl.col("pp_num_starters")
            .filter(pl.col("pp_index") == 1)
            .first()
            .alias("num_starters_L1"),
            pl.col("pp_num_starters")
            .filter(pl.col("pp_index") == 2)
            .first()
            .alias("num_starters_L2"),
            pl.col("pp_num_starters")
            .filter(pl.col("pp_index") == 3)
            .first()
            .alias("num_starters_L3"),

            # distance
            pl.col("pp_distance_id")
            .filter(pl.col("pp_index") == 1)
            .first()
            .alias("distance_L1"),
            pl.col("pp_distance_id")
            .filter(pl.col("pp_index") == 2)
            .first()
            .alias("distance_L2"),
            pl.col("pp_distance_id")
            .filter(pl.col("pp_index") == 3)
            .first()
            .alias("distance_L3"),

            # date of last race
            pl.col("pp_race_date")
            .filter(pl.col("pp_index") == 1)
            .first()
            .alias("last_pp_date"),

            # count of prior starts
            pl.len().alias("num_prior_starts"),
        )
        .with_columns(
            # relative finish
            (pl.col("official_finish_L1") / pl.col("num_starters_L1")).alias("relative_finish_L1"),
            (pl.col("official_finish_L2") / pl.col("num_starters_L2")).alias("relative_finish_L2"),
            (pl.col("official_finish_L3") / pl.col("num_starters_L3")).alias("relative_finish_L3"),

            # speed trend
            (pl.col("speed_fig_L1") - pl.col("avg_speed_fig_L3")).alias("speed_fig_trend"),
        )
        .with_columns(
            ((pl.col("relative_finish_L1") + pl.col("relative_finish_L2") + pl.col("relative_finish_L3")) / 3).alias("avg_relative_finish")
        )
    )
    # fmt: on


def _dollar_odds_plus_noise(
    dollar_odds: pl.Expr,
    morning_line: pl.Expr,
    p_exact: float = 0.5,
    p_interior: float = 0.75,
) -> pl.Expr:
    """
    Simulate mid-pool odds. With some probability `p_exact` use the final odds exactly.
    Otherwise, add random noise between the final odds and the morning line (interior)
    with probability `p_interior`, or between the final odds and 20% better than the
    morning line (overshoot) with probability `1 - p_interior`.

    Rounded to nearest 0.5 to mimic tote board increments.
    """
    return pl.struct(dollar_odds, morning_line).map_batches(
        lambda s: _compute_odds_noise(s, p_exact, p_interior),
        return_dtype=pl.Float64,
    )


def _compute_odds_noise(s: pl.Series, p_exact: float, p_interior: float) -> pl.Series:
    """
    Row-level random noise simulating non-final tote odds.

    Suppose p_exact = 0.5 and p_interior = 0.75. Then:
    - 50%: use final odds exactly
    - 37.5%: uniform between final and ML (interior)
    - 12.5%: uniform between final and 20% better than ML (overshoot)
    """
    df = s.struct.unnest()
    dollar_odds = df["dollar_odds"].to_numpy(zero_copy_only=False).astype(float)
    ml_odds = df["morning_line_odds_float"].to_numpy(zero_copy_only=False).astype(float)

    n = len(dollar_odds)
    rng = np.random.default_rng()
    roll_1 = rng.random(n)
    roll_2 = rng.random(n)
    noise = rng.random(n)
    diff = ml_odds - dollar_odds

    # p_exact final; otherwise p_interior between final and ML, 1 - p_interior overshoot
    noisy_odds = np.where(
        roll_1 < p_exact,
        dollar_odds,
        np.where(
            roll_2 < p_interior,
            dollar_odds + noise * diff,  # interior
            dollar_odds - noise * 0.2 * diff,  # overshoot past final
        ),
    )
    noisy_odds = np.round(noisy_odds, 1)  # round to nearest 0.1
    noisy_odds = np.maximum(noisy_odds, 0.05)  # floor at lowest observed odds
    return pl.Series(noisy_odds)


def build_training_df(processed_dir: Path = DEFAULT_PROCESSED_DIR) -> pl.DataFrame:
    """Assemble one row per horse-in-a-race with features and label."""
    entries = pl.read_parquet(processed_dir / "entries.parquet")
    results = pl.read_parquet(processed_dir / "results.parquet")
    pp = pl.read_parquet(processed_dir / "past_performances.parquet")
    workouts = pl.read_parquet(processed_dir / "workouts.parquet")

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
        pl.col("class_rating").alias("race_class_rating"),
    )

    entry_cols = entries.select(
        "race_id",
        "horse_name",
        "morning_line_odds_float",
        "post_position",
        "weight_carried",
        pl.col("class_rating").alias("entry_class_rating"),
        "purse",
    )

    pp_feats = _pp_features(pp)
    workout_feats = _workout_features(workouts)

    # fmt: off
    df = (
        race_cols
        .join(entry_cols, on=["race_id", "horse_name"], how="inner")
        .filter(pl.col("dollar_odds") > 0)
        .join(pp_feats, on=["race_id", "horse_name"], how="left")
        .join(workout_feats, on=["race_id", "horse_name"], how="left")
        .with_columns(
            # simple encoding and renaming
            pl.col("num_prior_starts").fill_null(0),
            pl.col("num_workouts").fill_null(0),
            (pl.col("surface") == "T").cast(pl.Int8).alias("is_turf"),
            pl.col("num_runners").alias("field_size"),
            (pl.col("official_finish") == 1).cast(pl.Int8).alias("won"),
        )
        .with_columns(
            (pl.col("num_prior_starts") == 0).cast(pl.Int8).alias("is_first_start"),
        )
        .with_columns(
            # simple derivations
            (pl.col("entry_class_rating") / pl.col("race_class_rating")).alias("entry_to_race_class_ratio"),
        )
        .with_columns(
            ### comparison to previous races/workouts
            # date
            (pl.col("race_date") - pl.col("last_pp_date"))
            .dt.total_days()
            .alias("days_since_last"),
            (pl.col("race_date") - pl.col("last_workout_date"))
            .dt.total_days()
            .alias("days_since_last_workout"),
            # distance
            (pl.col("distance") - pl.col("distance_L1")).alias("distance_diff_L1"),
            (pl.col("distance") - pl.col("distance_L2")).alias("distance_diff_L2"),
            (pl.col("distance") - pl.col("distance_L3")).alias("distance_diff_L3"),
            # race class rating
            (pl.col("race_class_rating") - pl.col("class_rating_L1")).alias("class_rating_diff_L1"),
            (pl.col("race_class_rating") - pl.col("avg_class_rating_L3")).alias("class_rating_diff_avg_L3"),
            (pl.col("race_class_rating") - pl.col("max_class_rating_L3")).alias("class_rating_diff_max_L3"),
        )
        .with_columns(
            ### comparison to field (window functions)
            # entry class rating
            (
                pl.col("entry_class_rating") - pl.col("entry_class_rating").mean().over("race_id")
            ).alias("entry_class_rating_minus_field_avg"),
            (
                pl.col("entry_class_rating") / pl.col("entry_class_rating").mean().over("race_id")
            ).alias("entry_class_rating_to_field_avg_ratio"),
            # speed
            (
                pl.col("speed_fig_L1") - pl.col("speed_fig_L1").mean().over("race_id")
            ).alias("speed_fig_minus_field_avg_L1"),
            (
                pl.col("speed_fig_L1") / pl.col("speed_fig_L1").mean().over("race_id")
            ).alias("speed_fig_to_field_avg_ratio_L1"),
        )
        .with_columns(
            # add noise to final odds to simulate mid-pool odds
            _dollar_odds_plus_noise(
                pl.col("dollar_odds"), pl.col("morning_line_odds_float")
            ).alias("dollar_odds_plus_noise"),
        )
        .drop("last_pp_date", "last_workout_date")
    )
    # fmt: on
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
