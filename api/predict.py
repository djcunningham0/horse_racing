"""Prediction logic: load model, build features, score runners."""

from pathlib import Path

import joblib
import numpy as np

from model.features import DEFAULT_FEATURE_COLS, SURFACE_MAP
from model.train import DEFAULT_MODEL_DIR, MODEL_FILENAME

from api.schemas import RaceRequest, RunnerPrediction


def load_model(
    model_dir: Path = DEFAULT_MODEL_DIR,
    model_filename: str = MODEL_FILENAME,
) -> dict:
    """Load the serialized model bundle (model + feature list)."""
    return joblib.load(model_dir / model_filename)


def predict_race(request: RaceRequest, model_bundle: dict) -> list[RunnerPrediction]:
    """Score all runners in a race and return predictions sorted by EV."""
    model = model_bundle["model"]
    field_size = len(request.runners)

    # build feature matrix
    rows = []
    for r in request.runners:
        figs = [r.speed_fig_last1, r.speed_fig_last2, r.speed_fig_last3]
        valid_figs = [f for f in figs if f is not None]
        avg_speed = float(np.mean(valid_figs)) if valid_figs else np.nan

        row = {
            "morning_line_decimal": r.morning_line_decimal,
            "post_position": r.post_position,
            "weight_carried": r.weight_carried,
            "field_size": field_size,
            "distance": request.distance,
            "surface_int": SURFACE_MAP.get(request.surface),
            "class_rating": r.class_rating,
            "speed_fig_L1": _nan_if_none(r.speed_fig_last1),
            "speed_fig_L2": _nan_if_none(r.speed_fig_last2),
            "speed_fig_L3": _nan_if_none(r.speed_fig_last3),
            "avg_speed_fig_L3": avg_speed,
            "days_since_last": _nan_if_none(r.days_since_last),
            "num_prior_starts": r.num_prior_starts,
            "is_first_start": int(r.num_prior_starts == 0),
        }

        # ensure columns the necessary columns are present and in the right order
        rows.append([row[col] for col in DEFAULT_FEATURE_COLS])

    X = np.array(rows, dtype=np.float32)
    scores = model.predict(X)
    probs = _softmax(scores)

    # calculate market probabilities from tote odds, then normalize
    raw_market = np.array([1.0 / (r.tote_odds + 1.0) for r in request.runners])
    market_total = raw_market.sum()
    market_probs = raw_market / market_total if market_total > 0 else raw_market

    # build predictions
    predictions = []
    for i, runner in enumerate(request.runners):
        model_prob = float(probs[i])
        market_prob = float(market_probs[i])
        decimal_odds = runner.tote_odds + 1.0
        ev = model_prob * decimal_odds - 1.0

        predictions.append(
            RunnerPrediction(
                horse_name=runner.horse_name,
                post_position=runner.post_position,
                model_prob=round(model_prob, 4),
                market_prob=round(market_prob, 4),
                edge=round(model_prob - market_prob, 4),
                ev_per_dollar=round(ev, 4),
            )
        )

    predictions.sort(key=lambda p: p.ev_per_dollar, reverse=True)
    return predictions


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def _nan_if_none(val: float | None) -> float:
    return val if val is not None else np.nan
