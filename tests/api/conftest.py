"""Shared fixtures for API tests."""

from datetime import date
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest
from fastapi.testclient import TestClient


def _fake_predict_from_raw(raw_df: pl.DataFrame, bundle: dict) -> pl.DataFrame:
    """Stand-in for model.inference.predict_from_raw.

    Avoids needing a real fitted pipeline: computes market_prob from live_odds
    (normalized per race) and assigns uniform model_prob within each race.
    Downstream callers only read `model_prob` and `market_prob`.
    """
    rng = np.random.default_rng(0)
    scores = rng.standard_normal(raw_df.height)
    shifted = pl.Series("_s", scores) - pl.Series("_s", scores).max()
    return (
        raw_df.with_columns(
            (1.0 / pl.col("live_odds")).alias("_implied"),
            shifted.exp().alias("_exp_score"),
        )
        .with_columns(
            (pl.col("_implied") / pl.col("_implied").sum().over("race_id")).alias(
                "market_prob"
            ),
            (pl.col("_exp_score") / pl.col("_exp_score").sum().over("race_id")).alias(
                "model_prob"
            ),
        )
        .drop("_implied", "_exp_score")
    )


@pytest.fixture
def client():
    """Create a test client with predict_from_raw mocked at the API boundary."""
    with (
        patch("api.predict.load_model", return_value={"pipeline": None}),
        patch("api.predict.predict_from_raw", side_effect=_fake_predict_from_raw),
    ):
        from api.main import app

        with TestClient(app) as c:
            yield c


def sample_race_body(
    track: str = "CD",
    race_number: int = 1,
    num_runners: int = 3,
) -> dict:
    """Build a minimal valid CreateRaceRequest body."""
    runners = [
        {
            "horse_name": f"Horse {i}",
            "post_position": i,
            "morning_line_odds": 5.0 + i,
            "weight_carried": 122,
            "entry_class_rating": 75.0,
            "year_of_birth": 2022,
            "sex": "G",
            "num_prior_starts": 5,
            "speed_fig_L1": 80.0 + i,
            "speed_fig_L2": 78.0,
            "last_pp_date": "2026-03-01",
        }
        for i in range(1, num_runners + 1)
    ]
    return {
        "track": track,
        "race_number": race_number,
        "race_date": date(2026, 4, 22).isoformat(),
        "distance_yards": 1320,
        "surface": "D",
        "race_class_rating": 80.0,
        "purse": 50000.0,
        "runners": runners,
    }
