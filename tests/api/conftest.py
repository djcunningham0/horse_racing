"""Shared fixtures for API tests."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a test client with a mocked model."""
    mock_model = MagicMock()
    # return arbitrary scores for any input shape
    mock_model.predict = lambda X: np.random.randn(X.shape[0])

    with patch("api.predict.load_model", return_value={"model": mock_model}):
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
            "class_rating": 75.0,
            "speed_fig_last1": 80.0 + i,
            "speed_fig_last2": 78.0,
            "days_since_last": 28,
            "num_prior_starts": 5,
        }
        for i in range(1, num_runners + 1)
    ]
    return {
        "track": track,
        "race_number": race_number,
        "distance": 6.0,
        "surface": "D",
        "runners": runners,
    }
