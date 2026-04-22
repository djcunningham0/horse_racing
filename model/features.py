"""Build the modeling dataset from processed parquet files."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from numpy.typing import NDArray

from model.paths import DEFAULT_PROCESSED_DIR


@dataclass(frozen=True)
class NoisyOddsConfig:
    """Hyperparameters for the mid-pool live-odds simulator."""

    beta_a: float = 2.0
    beta_b: float = 5.0
    kappa: float = 500.0
    max_odds: float = 99.0


DEFAULT_NOISY_ODDS_CONFIG = NoisyOddsConfig()

# settings for random split
DEFAULT_VAL_FRAC = 0.15
DEFAULT_TEST_FRAC = 0.15
DEFAULT_RANDOM_SEED = 0

# exclude jump races and downhill turf — Churchill Downs has neither, and the
# dynamics differ enough to add noise
EXCLUDED_COURSE_DESCS: list[str] = ["Hurdle", "Downhill turf", "Timber"]

DEFAULT_FEATURE_COLS: list[str] = [
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
    "is_first_start",
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


def build_training_df(
    processed_dir: Path | str = DEFAULT_PROCESSED_DIR,
    seed: int | None = None,
) -> pl.DataFrame:
    """Assemble one row per horse-in-a-race with features and label."""
    if isinstance(processed_dir, str):
        processed_dir = Path(processed_dir)

    entries = pl.read_parquet(processed_dir / "entries.parquet")
    results = pl.read_parquet(processed_dir / "results.parquet")
    pp = pl.read_parquet(processed_dir / "past_performances.parquet")
    workouts = pl.read_parquet(processed_dir / "workouts.parquet")

    race_cols = (
        results
        .filter(~pl.col("course_desc").is_in(EXCLUDED_COURSE_DESCS))
        .select(
            "race_id",
            "race_date",
            "track",
            "race_number",
            "distance_yards",
            "surface",
            "course_desc",
            "num_runners",
            "horse_name",
            "official_finish",
            "dollar_odds",
            pl.col("class_rating").alias("race_class_rating"),
        )
    )  # fmt: skip

    entry_cols = entries.select(
        "race_id",
        "horse_name",
        "morning_line_odds_float",
        "post_position",
        "weight_carried",
        pl.col("class_rating").alias("entry_class_rating"),
        "purse",
        "career_starts",
        "career_wins",
        "career_seconds",
        "career_thirds",
        "career_earnings",
        "surface_starts",
        "surface_wins",
        "surface_seconds",
        "surface_thirds",
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
            pl.col("num_runners").alias("field_size"),
            (pl.col("official_finish") == 1).cast(pl.Int8).alias("won"),
        )
        .with_columns(
            # simple encoding (continued)
            (pl.col("num_prior_starts") == 0).cast(pl.Int8).alias("is_first_start"),
        )
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
            # derive course type (continued)
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
            # simulate mid-pool live odds from ML + final
            # TODO: at inference, replace with actual live tote odds
            _noisy_live_odds(
                pl.col("dollar_odds"),
                pl.col("morning_line_odds_float"),
                pl.col("race_id"),
                seed=seed,
            ).alias("live_odds"),
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
            "last_pp_date",
            "last_workout_date",
            "_course_type",
            "_pp_course_type_L1",
            "_market_prob_raw",
        )
    )
    # fmt: on
    return df


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

            # final public odds
            pl.col("pp_odds")
            .filter(pl.col("pp_index") == 1)
            .first()
            .alias("pp_odds_L1"),
            pl.col("pp_odds")
            .filter(pl.col("pp_index") == 2)
            .first()
            .alias("pp_odds_L2"),
            pl.col("pp_odds")
            .filter(pl.col("pp_index") == 3)
            .first()
            .alias("pp_odds_L3"),

            # distance (yards)
            pl.col("pp_distance_yards")
            .filter(pl.col("pp_index") == 1)
            .first()
            .alias("distance_yards_L1"),
            pl.col("pp_distance_yards")
            .filter(pl.col("pp_index") == 2)
            .first()
            .alias("distance_yards_L2"),
            pl.col("pp_distance_yards")
            .filter(pl.col("pp_index") == 3)
            .first()
            .alias("distance_yards_L3"),

            # surface of last race
            pl.col("pp_surface")
            .filter(pl.col("pp_index") == 1)
            .first()
            .alias("pp_surface_L1"),

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
            # relative finish (continued)
            ((pl.col("relative_finish_L1") + pl.col("relative_finish_L2") + pl.col("relative_finish_L3")) / 3).alias("avg_relative_finish"),
        )
        .with_columns(
            # performance vs. market: positive = outperformed implied prob
            (
                (1 - pl.col("relative_finish_L1")) - (1 / (pl.col("pp_odds_L1") + 1))
            ).alias("pp_overperformance_L1"),
            (
                (1 - pl.col("relative_finish_L2")) - (1 / (pl.col("pp_odds_L2") + 1))
            ).alias("pp_overperformance_L2"),
            (
                (1 - pl.col("relative_finish_L3")) - (1 / (pl.col("pp_odds_L3") + 1))
            ).alias("pp_overperformance_L3"),
        )
        .with_columns(
            # performance vs. market (continued)
            (
                (pl.col("pp_overperformance_L1") + pl.col("pp_overperformance_L2") + pl.col("pp_overperformance_L3"))  / 3
            ).alias("pp_avg_overperformance_L3")
        )
    )
    # fmt: on


def base_margin_from_market_prob(df: pl.DataFrame) -> np.ndarray:
    """Logit of ``market_prob``, clipped to avoid infinities at the boundaries.

    Intended for use as ``base_margin`` at train and inference time, so the
    model learns corrections relative to market-implied probabilities.
    """
    p = df["market_prob"].to_numpy()
    p = np.clip(p, 1e-4, 1.0 - 1e-4)
    return np.log(p / (1.0 - p))


def _noisy_live_odds(
    dollar_odds: pl.Expr,
    morning_line: pl.Expr,
    race_id: pl.Expr,
    config: NoisyOddsConfig = DEFAULT_NOISY_ODDS_CONFIG,
    seed: int | None = None,
) -> pl.Expr:
    """
    Simulate mid-pool live odds via pool-share interpolation.

    For each race: sample alpha ~ Beta(a, b) as the pool-maturity parameter, interpolate
    pool shares from ML to final, add Dirichlet jitter with concentration kappa, then
    re-embed the race's takeout and invert back to odds.
    """
    return pl.struct(dollar_odds, morning_line, race_id).map_batches(
        lambda s: _compute_noisy_live_odds(s, config, seed),
        return_dtype=pl.Float64,
    )


def _compute_noisy_live_odds(
    s: pl.Series,
    config: NoisyOddsConfig,
    seed: int | None,
) -> pl.Series:
    """Polars wrapper around _noisy_live_odds_numpy."""
    df = s.struct.unnest()
    final = df["dollar_odds"].to_numpy(zero_copy_only=False).astype(float)
    ml = df["morning_line_odds_float"].to_numpy(zero_copy_only=False).astype(float)
    race_id = df["race_id"].to_numpy(zero_copy_only=False)
    _, race_codes = np.unique(race_id, return_inverse=True)
    odds = _noisy_live_odds_numpy(final, ml, race_codes, config, seed)
    return pl.Series(odds)


def _noisy_live_odds_numpy(
    final: NDArray[np.float64],
    ml: NDArray[np.float64],
    race_codes: NDArray[np.int64],
    config: NoisyOddsConfig,
    seed: int | None,
) -> NDArray[np.float64]:
    """
    Inputs are per-row; race_codes groups rows into races (contiguous integer codes
    0..n_races-1).
    """
    rng = np.random.default_rng(seed)
    n_races = int(race_codes.max()) + 1

    # implied probs
    p_ml = 1.0 / (ml + 1.0)
    p_fin = 1.0 / (final + 1.0)

    # race-level takeout from final odds (overround above 1)
    p_fin_sum = np.bincount(race_codes, weights=p_fin, minlength=n_races)
    takeout = p_fin_sum - 1.0  # one scalar per race

    # normalize to pool shares (sum to 1 per race)
    p_ml_sum = np.bincount(race_codes, weights=p_ml, minlength=n_races)
    ml_share = p_ml / p_ml_sum[race_codes]
    fin_share = p_fin / p_fin_sum[race_codes]

    # one alpha per race, broadcast to rows
    alpha = rng.beta(config.beta_a, config.beta_b, size=n_races)[race_codes]

    # interpolated shares (still sum to 1 per race since both endpoints do)
    interp = (1.0 - alpha) * ml_share + alpha * fin_share

    # Dirichlet jitter via the Gamma trick
    g = rng.gamma(shape=config.kappa * interp, scale=1.0)
    g_sum = np.bincount(race_codes, weights=g, minlength=n_races)
    noisy = g / g_sum[race_codes]

    # re-embed takeout so implied probs sum to (1 + takeout) as in real odds
    noisy_implied = noisy * (1.0 + takeout[race_codes])

    # floor implied probs at p_min = 1/(max_odds+1)
    # Renormalize free horses to keep the per-race sum = (1 + takeout). Iterate since
    # renormalization can push previously-free horses below the floor.
    p_min = 1.0 / (config.max_odds + 1.0)
    target = 1.0 + takeout
    for _ in range(5):
        pegged = noisy_implied < p_min
        noisy_implied = np.where(pegged, p_min, noisy_implied)
        pegged_mass = np.bincount(
            race_codes, weights=np.where(pegged, p_min, 0.0), minlength=n_races
        )
        free_mass = np.bincount(
            race_codes, weights=np.where(pegged, 0.0, noisy_implied), minlength=n_races
        )
        remaining = target - pegged_mass
        scale = np.where(free_mass > 0, remaining / np.maximum(free_mass, 1e-12), 1.0)
        noisy_implied = np.where(pegged, p_min, noisy_implied * scale[race_codes])

    # numerical safety: keep implied prob strictly below 1 so odds stay positive
    noisy_implied = np.minimum(noisy_implied, 1.0 - 1e-6)
    return (1.0 - noisy_implied) / noisy_implied


def split_by_race(
    df: pl.DataFrame,
    val_frac: float = DEFAULT_VAL_FRAC,
    test_frac: float = DEFAULT_TEST_FRAC,
    seed: int = DEFAULT_RANDOM_SEED,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Random split by race_id. All rows of a given race stay together.

    Random (not chronological) because only data for a single year (2023) is available,
    so a time split would confound seasonal effects with model quality. Safe for the
    current baseline feature set — none of the features are keyed on horse, jockey, or
    trainer identity, so no cross-split leakage.
    """
    train_ids, val_ids, test_ids = get_race_id_splits(
        df=df,
        val_frac=val_frac,
        test_frac=test_frac,
        seed=seed,
    )
    return (
        df.filter(pl.col("race_id").is_in(train_ids)),
        df.filter(pl.col("race_id").is_in(val_ids)),
        df.filter(pl.col("race_id").is_in(test_ids)),
    )


def get_race_id_splits(
    df: pl.DataFrame,
    val_frac: float = DEFAULT_VAL_FRAC,
    test_frac: float = DEFAULT_TEST_FRAC,
    seed: int = DEFAULT_RANDOM_SEED,
) -> tuple[NDArray[np.str_], NDArray[np.str_], NDArray[np.str_]]:
    """
    Return the (train_ids, val_ids, test_ids) to split races in to train, validation,
    and test set.
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
    return train_ids, val_ids, test_ids
