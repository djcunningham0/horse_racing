"""
Post-training probability calibration.

XGBRanker emits scores that are optimized for *ordering*, not calibrated probabilities.
A per-race softmax over those scores has an arbitrary temperature.  We fit a single
scalar `T` on a held-out split to minimize winner log-loss, and apply
`softmax(score / T)` at inference.
"""

import numpy as np
import polars as pl
from scipy.optimize import minimize_scalar

from model.evaluate import _log_loss_winner, _per_race_softmax

T_SEARCH_BOUNDS = (0.1, 10.0)


def fit_temperature(
    model,
    df: pl.DataFrame,
    features: list[str],
    bounds: tuple[float, float] = T_SEARCH_BOUNDS,
) -> float:
    """
    Find T > 0 minimizing winner log-loss on `df` for softmax(score / T).

    `df` must contain ``race_id`` and ``won``.  Races without a recorded winner are
    dropped before fitting.
    """
    X = df.select(features).to_numpy()
    scores = model.predict(X)
    df = df.with_columns(pl.Series("__score", scores))
    df = df.filter(pl.col("won").max().over("race_id") == 1)

    def objective(T: float) -> float:
        tmp = df.with_columns((pl.col("__score") / T).alias("_s"))
        tmp = _per_race_softmax(tmp, "_s", "_p")
        return _log_loss_winner(tmp, "_p")

    result = minimize_scalar(
        objective,
        bounds=bounds,
        method="bounded",
        options={"xatol": 1e-4},
    )
    return float(result.x)


def log_loss_at_T(
    model,
    df: pl.DataFrame,
    features: list[str],
    temperature: float = 1.0,
) -> float:
    """Convenience: winner log-loss on `df` at a given temperature."""
    X = df.select(features).to_numpy()
    scores = model.predict(X)
    tmp = df.with_columns(pl.Series("_s", scores / temperature))
    tmp = tmp.filter(pl.col("won").max().over("race_id") == 1)
    tmp = _per_race_softmax(tmp, "_s", "_p")
    return _log_loss_winner(tmp, "_p")


def apply_temperature(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Return softmax(scores / T) for a single race (1-D array)."""
    s = scores / temperature
    s = s - s.max()
    e = np.exp(s)
    return e / e.sum()
