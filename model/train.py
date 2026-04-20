"""Train an XGBoost ranker for horse racing win prediction.

Usage:
    python -m model.train
"""

import logging

import joblib
import numpy as np
import polars as pl
from xgboost import XGBRanker

from model.calibration import fit_temperature
from model.features import (
    DEFAULT_FEATURE_COLS,
    base_margin_from_market_prob,
    build_training_df,
    split_by_race,
)
from model.paths import DEFAULT_MODEL_DIR, MODEL_FILENAME

logger = logging.getLogger(__name__)

DEFAULT_HYPERPARAMS = {
    "objective": "rank:ndcg",
    "tree_method": "hist",
    "n_estimators": 800,
    "learning_rate": 0.01,
    "max_depth": 4,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 2,
    "early_stopping_rounds": 30,
    "eval_metric": "ndcg@3",
}


def prepare_df(
    df: pl.DataFrame,
    features: list[str],
    use_base_margin: bool = False,
) -> tuple[pl.DataFrame, np.ndarray, np.ndarray, np.ndarray | None]:
    """Return (X, y, group_sizes, base_margin) sorted by race_id.

    ``base_margin`` is ``None`` unless ``use_base_margin`` is set, in which
    case it is ``logit(market_prob)``.
    """
    df = df.sort("race_id")
    X = df.select(features)
    y = df["won"].to_numpy()
    group_sizes = df.group_by("race_id", maintain_order=True).len()["len"].to_numpy()
    base_margin = base_margin_from_market_prob(df) if use_base_margin else None
    return X, y, group_sizes, base_margin


def train(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    features: list[str] = DEFAULT_FEATURE_COLS,
    hyperparameters: dict | None = None,
    use_base_margin: bool = False,
) -> XGBRanker:
    params = {**DEFAULT_HYPERPARAMS, **(hyperparameters or {})}

    X_tr, y_tr, g_tr, m_tr = prepare_df(train_df, features, use_base_margin)
    X_va, y_va, g_va, m_va = prepare_df(val_df, features, use_base_margin)

    logger.info(
        f"train: {len(y_tr):,} rows / {len(g_tr):,} races | "
        f"val: {len(y_va):,} rows / {len(g_va):,} races"
    )

    model = XGBRanker(**params)
    model.fit(
        X_tr,
        y_tr,
        group=g_tr,
        base_margin=m_tr,
        eval_set=[(X_va, y_va)],
        eval_group=[g_va],
        base_margin_eval_set=[m_va] if use_base_margin else None,
        verbose=50,
    )
    return model


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    df = build_training_df()
    train_df, val_df, _ = split_by_race(df)
    model = train(train_df, val_df)

    temperature = fit_temperature(model, val_df, DEFAULT_FEATURE_COLS)
    logger.info(f"fit softmax temperature on val: T={temperature:.4f}")

    DEFAULT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DEFAULT_MODEL_DIR / MODEL_FILENAME
    joblib.dump(
        {
            "model": model,
            "features": DEFAULT_FEATURE_COLS,
            "temperature": temperature,
            "use_base_margin": False,
        },
        out_path,
    )
    logger.info(f"saved model to {out_path}")


if __name__ == "__main__":
    main()
