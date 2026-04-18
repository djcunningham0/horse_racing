import polars as pl
import pytest

from model.betting import summarize_roi


def test_summarize_roi_empty():
    empty = pl.DataFrame({
        "stake": pl.Series([], dtype=pl.Float64),
        "payout": pl.Series([], dtype=pl.Float64),
        "won": pl.Series([], dtype=pl.Int64),
        "ev_per_dollar": pl.Series([], dtype=pl.Float64),
    })
    result = summarize_roi(empty)
    assert result["n_bets"] == 0
    assert result["profit"] == 0.0
    assert result["roi"] == 0.0


def test_summarize_roi_known_values():
    bets = pl.DataFrame({
        "stake": [2.0, 2.0, 2.0],
        "payout": [10.0, 0.0, 0.0],  # first horse wins
        "won": [1, 0, 0],
        "ev_per_dollar": [0.5, -0.2, -0.1],
    })
    result = summarize_roi(bets)

    assert result["n_bets"] == 3
    assert result["total_staked"] == 6.0
    assert result["total_return"] == 10.0
    assert result["profit"] == pytest.approx(4.0)
    assert result["roi"] == pytest.approx(4.0 / 6.0)
    assert result["hit_rate"] == pytest.approx(1.0 / 3.0)
    assert result["avg_ev"] == pytest.approx((0.5 - 0.2 - 0.1) / 3.0)
