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
    data = prepare_df(race_df, use_base_margin=False)

    # all A rows come before B rows
    assert data.X["feat1"].to_list()[:3] == [20.0, 30.0, 50.0]  # A rows
    assert data.X["feat1"].to_list()[3:] == [10.0, 40.0]  # B rows


def test_prepare_df_returns_raw_df(race_df: pl.DataFrame):
    data = prepare_df(race_df, use_base_margin=False)
    # no feature selection happens here — all original columns are preserved
    assert set(data.X.columns) == {"race_id", "won", "feat1", "feat2"}


def test_prepare_df_y_matches_sorted_order(race_df: pl.DataFrame):
    data = prepare_df(race_df, use_base_margin=False)

    # after sorting: A(1,0,0), B(0,1)
    assert data.y.tolist() == [1, 0, 0, 0, 1]


def test_prepare_df_group_sizes(race_df: pl.DataFrame):
    data = prepare_df(race_df, use_base_margin=False)

    assert data.group_sizes.tolist() == [3, 2]  # 3 A rows, 2 B rows
    assert data.group_sizes.sum() == len(race_df)


def test_prepare_df_base_margin_none_when_disabled(race_df: pl.DataFrame):
    data = prepare_df(race_df, use_base_margin=False)
    assert data.base_margin is None
