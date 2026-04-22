import polars as pl
import pytest

from model.train import prepare_df


@pytest.fixture
def race_df() -> pl.DataFrame:
    """Small DataFrame with scrambled race_id order."""
    return pl.DataFrame({
        "race_id": ["B", "A", "A", "B", "A"],
        "won": [0, 1, 0, 1, 0],
        "feat1": [10.0, 20.0, 30.0, 40.0, 50.0],
        "feat2": [1.0, 2.0, 3.0, 4.0, 5.0],
    })


def test_prepare_df_sorts_by_race_id(race_df: pl.DataFrame):
    X, _, _, _ = prepare_df(race_df, ["feat1", "feat2"])

    # all A rows come before B rows
    assert X["feat1"].to_list()[:3] == [20.0, 30.0, 50.0]  # A rows
    assert X["feat1"].to_list()[3:] == [10.0, 40.0]  # B rows


def test_prepare_df_selects_features(race_df: pl.DataFrame):
    X, _, _, _ = prepare_df(race_df, ["feat1"])
    assert X.columns == ["feat1"]


def test_prepare_df_y_matches_sorted_order(race_df: pl.DataFrame):
    _, y, _, _ = prepare_df(race_df, ["feat1"])

    # after sorting: A(1,0,0), B(0,1)
    assert y.tolist() == [1, 0, 0, 0, 1]


def test_prepare_df_group_sizes(race_df: pl.DataFrame):
    _, _, group_sizes, _ = prepare_df(race_df, ["feat1", "feat2"])

    assert group_sizes.tolist() == [3, 2]  # 3 A rows, 2 B rows
    assert group_sizes.sum() == len(race_df)
