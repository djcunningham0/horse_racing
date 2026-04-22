from datetime import date

import numpy as np
import polars as pl
import pytest
from sklearn.pipeline import Pipeline

from model.feature_pipeline import (
    FEATURE_NAMES,
    derive_features,
    make_column_selector,
    make_feature_deriver,
)


@pytest.fixture
def raw_df() -> pl.DataFrame:
    """Minimal raw DataFrame covering the columns ``derive_features`` reads.

    Two races with 3 and 2 entries respectively. Values are chosen so that a
    handful of derived values are easy to hand-verify.
    """
    # fmt: off
    return pl.DataFrame({
        "race_id": ["R1", "R1", "R1", "R2", "R2"],
        "race_date": [date(2023, 6, 10)] * 3 + [date(2023, 6, 11)] * 2,
        "distance_yards": [1320, 1320, 1320, 1760, 1760],
        "surface": ["D", "D", "D", "T", "T"],
        "course_desc": ["Dirt main", "Dirt main", "Dirt main", "Turf main", "Turf main"],
        "pp_surface_L1": ["D", "T", None, "T", "D"],
        "num_prior_starts": [5, 0, 3, 8, 2],
        "num_workouts": [4, 2, 3, 5, 1],
        "field_size": [3, 3, 3, 2, 2],
        "num_runners": [3, 3, 3, 2, 2],
        "won": [1, 0, 0, 1, 0],
        "official_finish": [1, 3, 2, 1, 2],
        "dollar_odds": [2.5, 5.0, 10.0, 1.5, 3.0],
        "horse_name": ["A", "B", "C", "D", "E"],
        "morning_line_odds_float": [3.0, 5.0, 8.0, 2.0, 4.0],
        "live_odds": [2.0, 4.0, 8.0, 1.5, 3.0],
        "post_position": [1, 2, 3, 1, 2],
        "weight_carried": [120.0, 118.0, 122.0, 126.0, 120.0],
        "entry_class_rating": [700.0, 600.0, 500.0, 800.0, 700.0],
        "race_class_rating": [80.0, 80.0, 80.0, 90.0, 90.0],
        "purse": [50000, 50000, 50000, 80000, 80000],
        "career_starts": [20, 0, 10, 30, 5],
        "career_wins": [4, 0, 1, 6, 1],
        "career_seconds": [3, 0, 2, 4, 1],
        "career_thirds": [2, 0, 1, 5, 0],
        "career_earnings": [100_000.0, 0.0, 20_000.0, 200_000.0, 10_000.0],
        "surface_starts": [12, 0, 5, 20, 3],
        "surface_wins": [3, 0, 0, 5, 1],
        "surface_seconds": [2, 0, 1, 3, 0],
        "surface_thirds": [1, 0, 1, 3, 0],
        # PP aggregates (from _pp_features) and workout aggregates (from _workout_features)
        "last_pp_date": [date(2023, 5, 10), None, date(2023, 4, 1), date(2023, 5, 20), date(2023, 5, 1)],
        "last_workout_date": [date(2023, 6, 5), date(2023, 6, 1), date(2023, 6, 3), date(2023, 6, 8), date(2023, 6, 5)],
        "distance_yards_L1": [1320, None, 1320, 1760, 1540],
        "distance_yards_L2": [1320, None, 1100, 1760, None],
        "distance_yards_L3": [1100, None, 1320, 1980, None],
        "class_rating_L1": [75.0, None, 70.0, 88.0, 85.0],
        "class_rating_L2": [72.0, None, 65.0, 90.0, None],
        "class_rating_L3": [70.0, None, 72.0, 85.0, None],
        "avg_class_rating_L3": [72.33, None, 69.0, 87.67, 85.0],
        "max_class_rating_L3": [75.0, None, 72.0, 90.0, 85.0],
        "speed_fig_L1": [85.0, None, 78.0, 92.0, 88.0],
        "speed_fig_L2": [82.0, None, 75.0, 90.0, None],
        "speed_fig_L3": [80.0, None, 77.0, 88.0, None],
        "avg_speed_fig_L3": [82.33, None, 76.67, 90.0, 88.0],
        "max_speed_fig_L3": [85.0, None, 78.0, 92.0, 88.0],
        "speed_fig_trend": [2.67, None, 1.33, 2.0, 0.0],
        "official_finish_L1": [1, None, 4, 2, 3],
        "official_finish_L2": [3, None, 5, 1, None],
        "official_finish_L3": [2, None, 3, 4, None],
        "relative_finish_L1": [0.1, None, 0.5, 0.25, 0.3],
        "relative_finish_L2": [0.3, None, 0.6, 0.1, None],
        "relative_finish_L3": [0.2, None, 0.3, 0.4, None],
        "avg_relative_finish": [0.2, None, 0.467, 0.25, 0.3],
        "pp_odds_L1": [3.0, None, 6.0, 2.5, 4.0],
        "pp_odds_L2": [4.0, None, 5.0, 3.0, None],
        "pp_odds_L3": [2.5, None, 7.0, 2.0, None],
        "pp_overperformance_L1": [0.65, None, 0.36, 0.46, 0.5],
        "pp_overperformance_L2": [0.5, None, 0.23, 0.65, None],
        "pp_overperformance_L3": [0.51, None, 0.57, 0.27, None],
        "pp_avg_overperformance_L3": [0.55, None, 0.38, 0.46, 0.5],
        "num_starters_L1": [10, None, 8, 8, 10],
        "num_starters_L2": [10, None, 9, 9, None],
        "num_starters_L3": [10, None, 10, 10, None],
        "best_workout_rank_pct": [0.2, 0.5, 0.3, 0.1, 0.4],
        "best_workout_group_size": [5, 8, 6, 10, 5],
        "last_workout_rank_pct": [0.25, 0.4, 0.35, 0.15, 0.5],
        "last_workout_group_size": [5, 8, 6, 10, 5],
        "days_since_last_workout": [5, 10, 8, 3, 6],
    })
    # fmt: on


def test_make_column_selector_as_transformer():
    df = pl.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
    selector = make_column_selector(["c", "a"])
    out = selector.fit_transform(df)
    assert out.shape == (2, 2)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out[:, 0], [5, 6])  # c first
    np.testing.assert_allclose(out[:, 1], [1, 2])  # then a


def test_make_column_selector_feature_names_out():
    selector = make_column_selector(["foo", "bar", "baz"])
    names = selector.get_feature_names_out()
    assert list(names) == ["foo", "bar", "baz"]


def test_derive_features_known_values(raw_df: pl.DataFrame):
    out = derive_features(raw_df).sort(["race_id", "horse_name"])

    # surface one-hots: R1 is dirt, R2 is turf
    assert out["is_dirt"].to_list() == [1, 1, 1, 0, 0]
    assert out["is_turf"].to_list() == [0, 0, 0, 1, 1]
    assert out["is_all_weather"].to_list() == [0, 0, 0, 0, 0]

    # is_first_start: horse B has num_prior_starts == 0
    # sorted order: A, B, C, D, E
    assert out["is_first_start"].to_list() == [0, 1, 0, 0, 0]

    # entry_to_race_class_ratio: horse A = 700 / 80 = 8.75
    np.testing.assert_allclose(out["entry_to_race_class_ratio"][0], 700 / 80)

    # career_win_rate: horse A = 4/20 = 0.2; horse B = 0/0 = NaN
    np.testing.assert_allclose(out["career_win_rate"][0], 0.2)
    assert np.isnan(out["career_win_rate"][1])

    # ml_odds_rank within R1: ML odds 3/5/8 -> ranks 1/2/3 for A/B/C
    ranks_r1 = out.filter(pl.col("race_id") == "R1")["ml_odds_rank"].to_list()
    assert ranks_r1 == [1, 2, 3]

    # market_prob in each race should sum to 1
    for rid in ["R1", "R2"]:
        mp = out.filter(pl.col("race_id") == rid)["market_prob"].to_numpy()
        np.testing.assert_allclose(mp.sum(), 1.0, atol=1e-6)


def test_pipeline_composition(raw_df: pl.DataFrame):
    pipeline = Pipeline([
        ("derive", make_feature_deriver()),
        ("select", make_column_selector(FEATURE_NAMES)),
    ])
    X = pipeline.fit_transform(raw_df)
    assert X.shape == (5, len(FEATURE_NAMES))
    assert X.dtype == np.float32

    names = pipeline.named_steps["select"].get_feature_names_out()
    assert list(names) == FEATURE_NAMES
