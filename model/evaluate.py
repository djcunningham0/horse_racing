"""Evaluate the trained ranker against market and uniform baselines.

Usage:
    python -m model.evaluate
"""

import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from sklearn.pipeline import Pipeline

from model.betting import add_ev_columns, apply_bet_rule, summarize_roi
from model.features import build_raw_df
from model.split import DEFAULT_SPLIT_MODE, SPLIT_MODES, split_by_race
from model.inference import predict_from_raw
from model.paths import DEFAULT_MODEL_DIR, MODEL_FILENAME

logger = logging.getLogger(__name__)

EPS = 1e-12


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


ROI_RULES: list[dict] = [
    {"rule": "top_model_per_race"},
    {"rule": "favorite"},
    {"rule": "ml_favorite"},
    {"rule": "top_speed_fig"},
    {"rule": "top_ev_per_race", "ev_threshold": 0.0},
    {"rule": "ev_threshold", "ev_threshold": 0.10},
    {"rule": "ev_threshold", "ev_threshold": 0.20},
]


def _roi_label(cfg: dict) -> str:
    label = cfg["rule"]
    if "ev_threshold" in cfg:
        label += f" (>{cfg['ev_threshold']:.0%})"
    return label


def _metrics_for_split(
    split_df: pl.DataFrame,
    pipeline: Pipeline,
    temperature: float = 1.0,
    use_base_margin: bool = True,
) -> dict:
    bundle = {
        "pipeline": pipeline,
        "temperature": temperature,
        "use_base_margin": use_base_margin,
    }
    # predict_from_raw adds model_score + model_prob; _market_probs then overwrites
    # the live-odds-based market_prob with one computed from final dollar_odds, which
    # is the fair benchmark for historical evaluation
    split_df = predict_from_raw(split_df, bundle)
    split_df = _market_probs(split_df)

    # drop races with no winner
    split_df = split_df.filter(pl.col("won").max().over("race_id") == 1)

    split_df = add_ev_columns(split_df)

    sums = (
        split_df.group_by("race_id")
        .agg(pl.col("model_prob").sum().alias("s"))["s"]
        .to_numpy()
    )

    roi_by_rule = {}
    for cfg in ROI_RULES:
        bets = apply_bet_rule(split_df, **cfg)
        roi_by_rule[_roi_label(cfg)] = summarize_roi(bets)

    return {
        "n_races": int(split_df["race_id"].n_unique()),
        "n_rows": int(split_df.shape[0]),
        "model_log_loss": _log_loss_winner(split_df, "model_prob"),
        "market_log_loss": _log_loss_winner(split_df, "market_prob"),
        "uniform_log_loss": _uniform_log_loss(split_df),
        "model_top1_acc": _top1_accuracy(split_df, "model_prob"),
        "favorite_top1_acc": _top1_accuracy(split_df, "market_prob"),
        "model_prob_sum_mean": float(sums.mean()),
        "roi": roi_by_rule,
    }


def print_metrics_table(metrics: dict[str, dict]):
    splits = list(metrics.keys())
    header = f"{'metric':<22}" + "".join(f"{s:>12}" for s in splits)
    print("\n=== Metrics by split ===")
    print(header)
    print("-" * len(header))
    rows = [
        ("races", "n_races", "{:>12,}"),
        ("rows", "n_rows", "{:>12,}"),
        ("model log-loss", "model_log_loss", "{:>12.4f}"),
        ("market log-loss", "market_log_loss", "{:>12.4f}"),
        ("uniform log-loss", "uniform_log_loss", "{:>12.4f}"),
        ("model top-1 acc", "model_top1_acc", "{:>12.4f}"),
        ("favorite top-1", "favorite_top1_acc", "{:>12.4f}"),
        ("mean Σp per race", "model_prob_sum_mean", "{:>12.4f}"),
    ]
    for label, key, fmt in rows:
        line = f"{label:<22}" + "".join(fmt.format(metrics[s][key]) for s in splits)
        print(line)

    _print_roi_table(metrics)


def _print_roi_table(metrics: dict[str, dict]):
    splits = list(metrics.keys())
    rule_labels = list(metrics[splits[0]]["roi"].keys())

    print("\n=== ROI by bet rule ($2 flat stake) ===")
    for rule_label in rule_labels:
        header = f"  {'':.<24}" + "".join(f"{s:>12}" for s in splits)
        print(f"\n  {rule_label}")
        print(header)
        roi_rows = [
            ("bets", "n_bets", "{:>12,}"),
            ("staked", "total_staked", "{:>12,.0f}"),
            ("profit", "profit", "{:>12,.0f}"),
            ("roi", "roi", "{:>12.1%}"),
            ("hit rate", "hit_rate", "{:>12.1%}"),
            ("avg EV", "avg_ev", "{:>12.3f}"),
        ]
        for label, key, fmt in roi_rows:
            vals = []
            for s in splits:
                vals.append(fmt.format(metrics[s]["roi"][rule_label][key]))
            print(f"  {label:<24}" + "".join(vals))


def evaluate_splits(
    pipeline: Pipeline,
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    test_df: pl.DataFrame,
    temperature: float = 1.0,
    use_base_margin: bool = True,
) -> dict[str, dict]:
    """Compute metrics on all three splits. Returns dict keyed by split name."""
    return {
        "train": _metrics_for_split(train_df, pipeline, temperature, use_base_margin),
        "val": _metrics_for_split(val_df, pipeline, temperature, use_base_margin),
        "test": _metrics_for_split(test_df, pipeline, temperature, use_base_margin),
    }


def evaluate(
    model_dir: Path = DEFAULT_MODEL_DIR,
    split_mode: str = DEFAULT_SPLIT_MODE,
) -> dict[str, dict]:
    """Load model from disk and evaluate. Convenience entrypoint."""
    bundle = joblib.load(model_dir / MODEL_FILENAME)
    pipeline = bundle["pipeline"]
    temperature = bundle["temperature"]
    use_base_margin = bundle["use_base_margin"]
    logger.info(f"using temperature T={temperature:.4f}")

    df = build_raw_df()
    train_df, val_df, test_df = split_by_race(df, mode=split_mode)

    metrics = evaluate_splits(
        pipeline=pipeline,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        temperature=temperature,
        use_base_margin=use_base_margin,
    )
    print_metrics_table(metrics)
    return metrics


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    parser = argparse.ArgumentParser(description="Evaluate the saved model")
    parser.add_argument(
        "--split-mode",
        choices=SPLIT_MODES,
        default=DEFAULT_SPLIT_MODE,
        help=(
            "How to split races into train/val/test. Must match what was used for "
            "training, otherwise rows leak across splits."
        ),
    )
    args = parser.parse_args()
    evaluate(split_mode=args.split_mode)


if __name__ == "__main__":
    main()
