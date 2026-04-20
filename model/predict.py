"""Model-agnostic inference helper used at calibration, evaluation, and serving."""

import numpy as np
from xgboost import XGBClassifier, XGBRanker


def predict_scores(model, X, base_margin: np.ndarray | None = None) -> np.ndarray:
    """Raw margin-space scores from either XGBRanker or XGBClassifier.

    XGBClassifier.predict returns class labels by default, so we explicitly
    request margin output to keep the downstream per-race softmax identical
    across model types.
    """
    if isinstance(model, XGBClassifier):
        return model.predict(X, output_margin=True, base_margin=base_margin)
    elif isinstance(model, XGBRanker):
        return model.predict(X, base_margin=base_margin)
    else:
        raise NotImplementedError(f"model type {type(model)} not implemented")
