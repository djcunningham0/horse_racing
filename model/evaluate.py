"""Evaluate the trained ranker against market and uniform baselines.

Usage:
    python -m model.evaluate
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import polars as pl

from model.features import build_training_frame, split_by_race
from model.train import DEFAULT_MODEL_DIR, MODEL_FILENAME

logger = logging.getLogger(__name__)

EPS = 1e-12


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def _per_race_softmax(df: pl.DataFrame, score_col: str, out_col: str) -> pl.DataFrame:
    """Add `out_col`: softmax of `score_col` within each race."""
    out_frames = []
    for (_race_id,), group in df.group_by("race_id"):
        s = group[score_col].to_numpy().astype(float)
        if np.isnan(s).any():
            mean = np.nanmean(s) if not np.all(np.isnan(s)) else 0.0
            s = np.where(np.isnan(s), mean, s)
        out_frames.append(group.with_columns(pl.Series(out_col, _softmax(s))))
    return pl.concat(out_frames)


def _market_probs(df: pl.DataFrame) -> pl.DataFrame:
    """Add `market_prob` from dollar_odds, renormalized per race to sum to 1."""
    return (
        df.with_columns((1.0 / (pl.col("dollar_odds") + 1.0)).alias("_raw_mp"))
        .with_columns(
            (pl.col("_raw_mp") / pl.col("_raw_mp").sum().over("race_id")).alias(
                "market_prob"
            )
        )
        .drop("_raw_mp")
    )


def _log_loss_winner(df: pl.DataFrame, prob_col: str) -> float:
    """Mean negative log prob on the actual winner (one value per race)."""
    winners = df.filter(pl.col("won") == 1)
    probs = winners[prob_col].to_numpy()
    return float(-np.log(np.clip(probs, EPS, 1.0)).mean())


def _top1_accuracy(df: pl.DataFrame, prob_col: str) -> float:
    """Fraction of races where the horse with highest prob is the winner."""
    top = (
        df.sort([pl.col("race_id"), pl.col(prob_col)], descending=[False, True])
        .group_by("race_id", maintain_order=True)
        .first()
    )
    return float(top["won"].mean())


def _uniform_log_loss(df: pl.DataFrame) -> float:
    sizes = df.group_by("race_id").len()["len"].to_numpy()
    return float(np.log(sizes).mean())


def evaluate(model_dir: Path = DEFAULT_MODEL_DIR) -> dict:
    bundle = joblib.load(model_dir / MODEL_FILENAME)
    model = bundle["model"]
    features = bundle["features"]

    df = build_training_frame()
    _, _, test_df = split_by_race(df)

    X = test_df.select(features).to_numpy()
    scores = model.predict(X)
    test_df = test_df.with_columns(pl.Series("model_score", scores))

    test_df = _per_race_softmax(test_df, "model_score", "model_prob")
    test_df = _market_probs(test_df)

    # Drop races with no winner in the test set (shouldn't happen, but safe).
    test_df = test_df.filter(pl.col("won").max().over("race_id") == 1)

    metrics = {
        "n_races": int(test_df["race_id"].n_unique()),
        "n_rows": int(test_df.shape[0]),
        "model_log_loss": _log_loss_winner(test_df, "model_prob"),
        "market_log_loss": _log_loss_winner(test_df, "market_prob"),
        "uniform_log_loss": _uniform_log_loss(test_df),
        "model_top1_acc": _top1_accuracy(test_df, "model_prob"),
        "favorite_top1_acc": _top1_accuracy(test_df, "market_prob"),
    }

    sums = (
        test_df.group_by("race_id")
        .agg(pl.col("model_prob").sum().alias("s"))["s"]
        .to_numpy()
    )
    metrics["model_prob_sum_mean"] = float(sums.mean())

    print("\n=== Test-set metrics ===")
    print(f"  races:            {metrics['n_races']:,}")
    print(f"  rows:             {metrics['n_rows']:,}")
    print(f"  model log-loss:   {metrics['model_log_loss']:.4f}")
    print(f"  market log-loss:  {metrics['market_log_loss']:.4f}")
    print(f"  uniform log-loss: {metrics['uniform_log_loss']:.4f}")
    print(f"  model top-1 acc:  {metrics['model_top1_acc']:.4f}")
    print(f"  favorite top-1:   {metrics['favorite_top1_acc']:.4f}")
    print(f"  mean Σp per race: {metrics['model_prob_sum_mean']:.4f} (should be ~1.0)")

    return metrics


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    evaluate()


if __name__ == "__main__":
    main()
