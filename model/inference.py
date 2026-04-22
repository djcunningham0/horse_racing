"""Model-agnostic inference helpers for the training + serving pipeline.

Consolidates the `derive → select → predict_scores → per-race softmax with T`
chain that evaluation, calibration, diagnostics, and the API all need.
"""

import numpy as np
import polars as pl
from xgboost import XGBClassifier, XGBRanker

from model.features import base_margin_from_market_prob


def predict_from_raw(raw_df: pl.DataFrame, bundle: dict) -> pl.DataFrame:
    """Apply the full pipeline. Returns `raw_df` + `model_score` and `model_prob`.

    `model_prob` is the per-race softmax of `model_score / bundle["temperature"]`.
    """
    derived, scores = compute_scores(raw_df, bundle)
    derived = derived.with_columns(pl.Series("model_score", scores))
    return per_race_softmax(
        df=derived,
        score_col="model_score",
        out_col="model_prob",
        temperature=bundle["temperature"],
    )


def compute_scores(
    raw_df: pl.DataFrame,
    bundle: dict,
) -> tuple[pl.DataFrame, np.ndarray]:
    """Apply the feature pipeline and return `(derived_df, raw margin scores)`.

    Useful when the caller needs raw scores separately — e.g., temperature fitting
    rescoring at different T values without re-deriving features.
    """
    pipeline = bundle["pipeline"]
    derived = pipeline.named_steps["derive"].transform(raw_df)
    X = pipeline.named_steps["select"].transform(derived)
    base_margin = (
        base_margin_from_market_prob(derived) if bundle["use_base_margin"] else None
    )
    scores = predict_scores(pipeline.named_steps["model"], X, base_margin=base_margin)
    return derived, scores


def predict_scores(model, X, base_margin: np.ndarray | None = None) -> np.ndarray:
    """Raw margin-space scores from either XGBRanker or XGBClassifier.

    XGBClassifier.predict returns class labels by default, so we explicitly request
    margin output to keep the downstream per-race softmax identical
    across model types.
    """
    if isinstance(model, XGBClassifier):
        return model.predict(X, output_margin=True, base_margin=base_margin)
    elif isinstance(model, XGBRanker):
        return model.predict(X, base_margin=base_margin)
    else:
        raise NotImplementedError(f"model type {type(model)} not implemented")


def per_race_softmax(
    df: pl.DataFrame,
    score_col: str,
    out_col: str,
    temperature: float = 1.0,
) -> pl.DataFrame:
    """Add `out_col`: softmax of `score_col / temperature` within each race."""
    scaled = pl.col(score_col) / temperature
    shifted = scaled - scaled.max().over("race_id")
    return (
        df
        .with_columns(shifted.exp().alias(out_col))
        .with_columns(pl.col(out_col) / pl.col(out_col).sum().over("race_id"))
    )  # fmt: skip
