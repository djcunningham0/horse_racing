"""
Post-training probability calibration.

XGBRanker emits scores that are optimized for *ordering*, not calibrated probabilities.
A per-race softmax over those scores has an arbitrary temperature.  We fit a single
scalar `T` on a held-out split to minimize winner log-loss, and apply
`softmax(score / T)` at inference.
"""

import polars as pl
from scipy.optimize import minimize_scalar
from sklearn.pipeline import Pipeline

from model.evaluate import _log_loss_winner
from model.inference import compute_scores, per_race_softmax, predict_from_raw

T_SEARCH_BOUNDS = (0.1, 10.0)


def fit_temperature(
    pipeline: Pipeline,
    df: pl.DataFrame,
    bounds: tuple[float, float] = T_SEARCH_BOUNDS,
    use_base_margin: bool = True,
) -> float:
    """
    Find T > 0 minimizing winner log-loss on `df` for softmax(score / T).

    `df` is a raw DataFrame from `build_raw_df` (must contain `race_id` and `won`).
    Races without a recorded winner are dropped before fitting.
    """
    bundle = {
        "pipeline": pipeline,
        "temperature": 1.0,  # unused — we rescore at each T below
        "use_base_margin": use_base_margin,
    }
    derived, scores = compute_scores(df, bundle)
    derived = derived.with_columns(pl.Series("__score", scores))
    derived = derived.filter(pl.col("won").max().over("race_id") == 1)

    def objective(T: float) -> float:
        tmp = per_race_softmax(derived, "__score", "_p", temperature=T)
        return _log_loss_winner(tmp, "_p")

    result = minimize_scalar(
        objective,
        bounds=bounds,
        method="bounded",
        options={"xatol": 1e-4},
    )
    return float(result.x)


def log_loss_at_T(
    pipeline: Pipeline,
    df: pl.DataFrame,
    temperature: float = 1.0,
    use_base_margin: bool = True,
) -> float:
    """Convenience: winner log-loss on `df` at a given temperature."""
    bundle = {
        "pipeline": pipeline,
        "temperature": temperature,
        "use_base_margin": use_base_margin,
    }
    scored = predict_from_raw(df, bundle)
    scored = scored.filter(pl.col("won").max().over("race_id") == 1)
    return _log_loss_winner(scored, "model_prob")
