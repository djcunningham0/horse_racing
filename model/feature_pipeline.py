"""sklearn Pipeline components for model training and inference.

Two stateless transforms wrapped as ``FunctionTransformer``s:

- ``derive_features``: raw joined DataFrame -> DataFrame with raw + derived columns
- ``select_features``: DataFrame -> numpy ``float32`` matrix in the given order

Future categorical encoders will slot in as a new Pipeline step between
``select`` and the estimator, so these two remain stateless.
"""

import numpy as np
import polars as pl
from sklearn.preprocessing import FunctionTransformer

FEATURE_NAMES: list[str] = [
    # odds
    "morning_line_odds_float",
    "ml_odds_rank",
    "live_odds",
    "live_odds_rank",
    "market_prob",
    # race characteristics
    "field_size",
    "distance_yards",
    "is_dirt",
    "is_turf",
    "is_all_weather",
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
    # PP odds
    "pp_odds_L1",
    "pp_odds_L2",
    "pp_odds_L3",
    "pp_overperformance_L1",
    "pp_overperformance_L2",
    "pp_overperformance_L3",
    "pp_avg_overperformance_L3",
    # PP distance
    "distance_diff_L1",
    "distance_diff_L2",
    "distance_diff_L3",
    # PP surface
    "surface_switch_L1",
    # PP prior starts
    "days_since_last",
    "num_prior_starts",
    # workouts
    "best_workout_rank_pct",
    "best_workout_group_size",
    "last_workout_rank_pct",
    "last_workout_group_size",
    "num_workouts",
    "days_since_last_workout",
    # career stats
    "career_starts",
    "career_win_rate",
    "career_top3_rate",
    "career_earnings_per_start",
    "surface_starts",
    "surface_win_rate",
    "surface_top3_rate",
]


def derive_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute all derived model features from a raw joined DataFrame.

    Input must have the columns produced by ``build_raw_df``: joined entries, results,
    PP aggregates, workout aggregates, plus ``won``, ``field_size``, ``live_odds``.

    Output is the same DataFrame with derived columns added (and temporary helper
    columns dropped).
    """
    # fmt: off
    return (
        df
        .with_columns(
            # derive course type from course description and/or surface
            (
                # results: use course_desc and surface
                pl.when(pl.col("course_desc") == "All Weather Track").then(pl.lit("All Weather Track"))
                .when(pl.col("surface") == "D").then(pl.lit("Dirt"))
                .when(pl.col("surface") == "T").then(pl.lit("Turf"))
                .otherwise(None)
            ).alias("_course_type"),
            (
                # PPs: use pp_surface
                # - T/I/O = turf, D = dirt, E = all-weather
                # - rare codes (M/C/B/S/V/J/U/N) fall through to null
                pl.when(pl.col("pp_surface_L1").is_in(["T", "I", "O"])).then(pl.lit("Turf"))
                .when(pl.col("pp_surface_L1") == "D").then(pl.lit("Dirt"))
                .when(pl.col("pp_surface_L1") == "E").then(pl.lit("All Weather Track"))
                .otherwise(None)
            ).alias("_pp_course_type_L1"),
        )
        .with_columns(
            # course type (continued)
            (pl.col("_course_type") == "All Weather Track").cast(pl.Int8).alias("is_all_weather"),
            (pl.col("_course_type") == "Dirt").cast(pl.Int8).alias("is_dirt"),
            (pl.col("_course_type") == "Turf").cast(pl.Int8).alias("is_turf"),
            (pl.col("_course_type") != pl.col("_pp_course_type_L1")).cast(pl.Int8).alias("surface_switch_L1"),
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
            # distance (yards)
            (pl.col("distance_yards") - pl.col("distance_yards_L1")).alias("distance_diff_L1"),
            (pl.col("distance_yards") - pl.col("distance_yards_L2")).alias("distance_diff_L2"),
            (pl.col("distance_yards") - pl.col("distance_yards_L3")).alias("distance_diff_L3"),
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
            # career stats
            (pl.col("career_wins") / pl.col("career_starts")).alias("career_win_rate"),
            (
                (pl.col("career_wins") + pl.col("career_seconds") + pl.col("career_thirds"))
                / pl.col("career_starts")
            ).alias("career_top3_rate"),
            (pl.col("career_earnings") / pl.col("career_starts")).alias("career_earnings_per_start"),
            (pl.col("surface_wins") / pl.col("surface_starts")).alias("surface_win_rate"),
            (
                (pl.col("surface_wins") + pl.col("surface_seconds") + pl.col("surface_thirds"))
                / pl.col("surface_starts")
            ).alias("surface_top3_rate"),
        )
        .with_columns(
            # odds ordinal rank within field (1 = favorite; ties share the lower rank)
            pl.col("morning_line_odds_float")
            .rank(method="min")
            .over("race_id")
            .alias("ml_odds_rank"),
            pl.col("live_odds")
            .rank(method="min")
            .over("race_id")
            .alias("live_odds_rank"),
        )
        .with_columns(
            # implied win probability from live odds, normalized per race
            # (noisy shares already sum to ~1 by construction, but takeout re-embedding
            # and numerical floors can perturb the sum; normalize for safety)
            (1 / (pl.col("live_odds") + 1)).alias("_market_prob_raw"),
        )
        .with_columns(
            (pl.col("_market_prob_raw") / pl.col("_market_prob_raw").sum().over("race_id"))
            .alias("market_prob"),
        )
        .drop(
            "_course_type",
            "_pp_course_type_L1",
            "_market_prob_raw",
        )
    )
    # fmt: on


def select_features(df: pl.DataFrame, features: list[str]) -> np.ndarray:
    """Select feature columns in order and return a float32 numpy matrix."""
    return df.select(features).to_numpy().astype(np.float32)


def make_column_selector(features: list[str]) -> FunctionTransformer:
    """FunctionTransformer that selects ``features`` as a float32 matrix.

    ``get_feature_names_out`` returns the stored feature list so downstream
    callers (API, calibration) can introspect the model's expected inputs.
    """
    return FunctionTransformer(
        select_features,
        kw_args={"features": features},
        validate=False,
        feature_names_out=_selector_feature_names_out,
    )


def _selector_feature_names_out(transformer, _input_features):
    """Return the stored feature list from a column-selector transformer."""
    return np.asarray(transformer.kw_args["features"], dtype=object)


def make_feature_deriver() -> FunctionTransformer:
    """FunctionTransformer wrapping ``derive_features``."""
    return FunctionTransformer(derive_features, validate=False)
