"""Train an XGBoost ranker for horse racing win prediction.

Usage:
    python -m model.train
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from xgboost import XGBRanker

from model.features import FEATURE_COLUMNS, build_training_frame, split_by_race

logger = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = Path("model/artifacts")
MODEL_FILENAME = "xgb_ranker_v1.joblib"


def _prepare(df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X, y, group_sizes) sorted by race_id so XGBRanker groups align."""
    df = df.sort("race_id")
    X = df.select(FEATURE_COLUMNS).to_numpy()
    y = df["won"].to_numpy()
    group_sizes = df.group_by("race_id", maintain_order=True).len()["len"].to_numpy()
    return X, y, group_sizes


def train(model_dir: Path = DEFAULT_MODEL_DIR) -> XGBRanker:
    df = build_training_frame()
    train_df, val_df, _ = split_by_race(df)

    X_tr, y_tr, g_tr = _prepare(train_df)
    X_va, y_va, g_va = _prepare(val_df)

    logger.info(
        f"train: {len(y_tr):,} rows / {len(g_tr):,} races | "
        f"val: {len(y_va):,} rows / {len(g_va):,} races"
    )

    model = XGBRanker(
        objective="rank:pairwise",
        tree_method="hist",
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=30,
        eval_metric="ndcg@1",
    )
    model.fit(
        X_tr,
        y_tr,
        group=g_tr,
        eval_set=[(X_va, y_va)],
        eval_group=[g_va],
        verbose=50,
    )

    model_dir.mkdir(parents=True, exist_ok=True)
    out_path = model_dir / MODEL_FILENAME
    joblib.dump({"model": model, "features": FEATURE_COLUMNS}, out_path)
    logger.info(f"saved model to {out_path}")
    return model


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    train()


if __name__ == "__main__":
    main()
