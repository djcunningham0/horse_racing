"""Train an XGBoost ranker for horse racing win prediction.

Usage:
    python -m model.train
"""

import argparse
import logging
from dataclasses import dataclass

import joblib
import numpy as np
import polars as pl
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier, XGBRanker

from model.calibration import fit_temperature
from model.feature_pipeline import (
    FEATURE_NAMES,
    derive_features,
    make_column_selector,
    make_feature_deriver,
)
from model.features import (
    base_margin_from_market_prob,
    build_raw_df,
    split_by_race,
)
from model.paths import DEFAULT_MODEL_DIR, MODEL_FILENAME

logger = logging.getLogger(__name__)

DEFAULT_RANKER_HYPERPARAMS = {
    "objective": "rank:ndcg",
    "tree_method": "hist",
    "n_estimators": 1000,
    "learning_rate": 0.01,
    "max_depth": 4,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 2,
    "early_stopping_rounds": 30,
    "eval_metric": "ndcg@3",
}

DEFAULT_CLASSIFIER_HYPERPARAMS = {
    "objective": "binary:logistic",
    "tree_method": "hist",
    "n_estimators": 2000,
    "learning_rate": 0.01,
    "max_depth": 4,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 2,
    "early_stopping_rounds": 30,
    "eval_metric": "logloss",
}

DEFAULT_HYPERPARAMS: dict[str, dict] = {
    "ranker": DEFAULT_RANKER_HYPERPARAMS,
    "classifier": DEFAULT_CLASSIFIER_HYPERPARAMS,
}


@dataclass(frozen=True)
class PreparedData:
    """Race-sorted inputs for training or inference.

    Attributes
    ----------
    X
        Raw DataFrame sorted by `race_id`, ready for the sklearn Pipeline
    y
        Binary win labels, aligned with `X`
    group_sizes
        Number of horses per race, in `race_id` order
    base_margin
        `logit(market_prob)` aligned with `X`, or `None` if not requested
    """

    X: pl.DataFrame
    y: np.ndarray
    group_sizes: np.ndarray
    base_margin: np.ndarray | None


def prepare_df(df: pl.DataFrame, use_base_margin: bool = True) -> PreparedData:
    """Sort `df` by race_id and extract aligned labels, group sizes, and base margin.

    Parameters
    ----------
    df
        Raw DataFrame from `build_raw_df`
    use_base_margin
        If True, compute `base_margin` as `logit(market_prob)` via `derive_features`;
        otherwise `base_margin` is `None`
    """
    df = df.sort("race_id")
    y = df["won"].to_numpy()
    group_sizes = df.group_by("race_id", maintain_order=True).len()["len"].to_numpy()
    base_margin = (
        base_margin_from_market_prob(derive_features(df)) if use_base_margin else None
    )
    return PreparedData(X=df, y=y, group_sizes=group_sizes, base_margin=base_margin)


def train(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame | None = None,
    features: list[str] | None = None,
    hyperparameters: dict | None = None,
    use_base_margin: bool = True,
    model_type: str = "classifier",
) -> Pipeline:
    """Fit a feature + model sklearn Pipeline.

    Parameters
    ----------
    train_df
        Raw DataFrame from `build_raw_df`.
    val_df
        Optional validation DataFrame. If `None`, no `eval_set` is passed and early
        stopping is disabled — the model trains for the full `n_estimators`. Used for
        the final retrain on all available data.
    features
        Feature column names selected after the `derive` step (if None, defaults to
        `FEATURE_NAMES`)
    hyperparameters
        Overrides merged on top of `DEFAULT_HYPERPARAMS[model_type]`
    use_base_margin
        If True, pass `logit(market_prob)` as XGBoost `base_margin`
    model_type
        Either "ranker" (XGBRanker) or "classifier" (XGBClassifier)

    Returns
    -------
    Pipeline
        Fitted `derive` + `select` + `model` pipeline, ready to apply to future raw
        DataFrames
    """
    if features is None:
        features = FEATURE_NAMES

    params = {**DEFAULT_HYPERPARAMS[model_type], **(hyperparameters or {})}
    if val_df is None:
        # no val set to early-stop on; train for the full n_estimators
        params.pop("early_stopping_rounds", None)

    train_prepped = prepare_df(train_df, use_base_margin)
    val_prepped = prepare_df(val_df, use_base_margin) if val_df is not None else None

    val_summary = (
        f"val: {len(val_prepped.y):,} rows / {len(val_prepped.group_sizes):,} races"
        if val_prepped is not None
        else "val: none (final retrain)"
    )
    logger.info(
        f"train ({model_type}): {len(train_prepped.y):,} rows / {len(train_prepped.group_sizes):,} races | "
        f"{val_summary}"
    )

    if model_type == "ranker":
        estimator = XGBRanker(**params)
    elif model_type == "classifier":
        estimator = XGBClassifier(**params)
    else:
        raise ValueError(f"unknown model_type: {model_type!r}")

    pipeline = Pipeline([
        ("derive", make_feature_deriver()),
        ("select", make_column_selector(features)),
        ("model", estimator),
    ])

    # Fit feature steps on train, transform val — xgboost needs a pre-transformed
    # eval_set since sklearn Pipelines don't auto-transform fit kwargs.
    feature_pipeline = Pipeline(pipeline.steps[:-1])
    X_tr_numpy = feature_pipeline.fit_transform(train_prepped.X)

    fit_kwargs = {
        "base_margin": train_prepped.base_margin,
        "verbose": 50,
    }
    if val_prepped is not None:
        X_va_numpy = feature_pipeline.transform(val_prepped.X)
        fit_kwargs["eval_set"] = [(X_va_numpy, val_prepped.y)]
        fit_kwargs["base_margin_eval_set"] = (
            [val_prepped.base_margin] if use_base_margin else None
        )
    if model_type == "ranker":
        fit_kwargs["group"] = train_prepped.group_sizes
        if val_prepped is not None:
            fit_kwargs["eval_group"] = [val_prepped.group_sizes]

    # earlier pipeline steps (`feature_pipeline`) were already fit, so just need to fit
    # the estimator here
    estimator.fit(X_tr_numpy, train_prepped.y, **fit_kwargs)

    return pipeline


def _temperature_arg(value: str) -> float | str:
    if value == "auto":
        return "auto"
    return float(value)


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    parser = argparse.ArgumentParser(description="Train a horse racing model")
    parser.add_argument(
        "--model-type",
        choices=["ranker", "classifier"],
        default="classifier",
        help="XGBoost model type",
    )
    parser.add_argument(
        "--use-base-margin",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use logit(market_prob) as XGBoost base_margin",
    )
    parser.add_argument(
        "--temperature",
        type=_temperature_arg,
        default=1.0,
        help="Softmax temperature (float) or 'auto' to fit on the validation set.",
    )
    parser.add_argument(
        "--final-retrain",
        action="store_true",
        help=(
            "Fit on train+val+test combined for deploy. Requires --n-estimators "
            "(copied from the early-stopping run) and a numeric --temperature."
        ),
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=None,
        help="Override n_estimators. Required with --final-retrain.",
    )
    live_odds = parser.add_mutually_exclusive_group()
    live_odds.add_argument(
        "--use-morning-line-as-live",
        action="store_true",
        help="Set live_odds = morning line (no simulator, no leakage). For experimentation.",
    )
    live_odds.add_argument(
        "--use-final-as-live",
        action="store_true",
        help="Set live_odds = final public odds. Leaks future info; upper-bound only.",
    )
    args = parser.parse_args()

    if args.final_retrain:
        if args.n_estimators is None:
            parser.error("--final-retrain requires --n-estimators")
        if args.temperature == "auto":
            parser.error("--final-retrain is incompatible with --temperature auto (no val set)")

    df = build_raw_df(
        use_morning_line_as_live=args.use_morning_line_as_live,
        use_final_as_live=args.use_final_as_live,
    )
    train_df, val_df, test_df = split_by_race(df)

    hyperparameters = (
        {"n_estimators": args.n_estimators} if args.n_estimators is not None else None
    )
    if args.final_retrain:
        combined_df = pl.concat([train_df, val_df, test_df])
        pipeline = train(
            combined_df,
            val_df=None,
            hyperparameters=hyperparameters,
            use_base_margin=args.use_base_margin,
            model_type=args.model_type,
        )
    else:
        pipeline = train(
            train_df,
            val_df,
            hyperparameters=hyperparameters,
            use_base_margin=args.use_base_margin,
            model_type=args.model_type,
        )

    if args.temperature == "auto":
        temperature = fit_temperature(
            pipeline, val_df, use_base_margin=args.use_base_margin
        )
        logger.info(f"fit softmax temperature on val: T={temperature:.4f}")
    else:
        temperature = args.temperature
        logger.info(f"using fixed softmax temperature: T={temperature:.4f}")

    DEFAULT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DEFAULT_MODEL_DIR / MODEL_FILENAME
    joblib.dump(
        {
            "pipeline": pipeline,
            "temperature": temperature,
            "use_base_margin": args.use_base_margin,
            "model_type": args.model_type,
        },
        out_path,
    )
    logger.info(f"saved model to {out_path}")


if __name__ == "__main__":
    main()
