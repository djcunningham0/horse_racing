"""Train an XGBoost ranker for horse racing win prediction.

Usage:
    python -m model.train
"""

import argparse
import logging

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


def prepare_df(
    df: pl.DataFrame,
    use_base_margin: bool = True,
) -> tuple[pl.DataFrame, np.ndarray, np.ndarray, np.ndarray | None]:
    """Return (X_raw, y, group_sizes, base_margin) sorted by race_id.

    Parameters
    ----------
    df
        Raw DataFrame from `build_raw_df`
    use_base_margin
        If True, compute `base_margin` as `logit(market_prob)` via `derive_features`;
        otherwise `base_margin` is `None`

    Returns
    -------
    X_raw
        Sorted raw DataFrame, ready for the sklearn Pipeline
    y
        Binary win labels
    group_sizes
        Number of horses per race, in race_id order
    base_margin
        `logit(market_prob)` if `use_base_margin` else `None`
    """
    df = df.sort("race_id")
    y = df["won"].to_numpy()
    group_sizes = df.group_by("race_id", maintain_order=True).len()["len"].to_numpy()
    base_margin = (
        base_margin_from_market_prob(derive_features(df)) if use_base_margin else None
    )
    return df, y, group_sizes, base_margin


def train(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    features: list[str] | None = None,
    hyperparameters: dict | None = None,
    use_base_margin: bool = True,
    model_type: str = "classifier",
) -> Pipeline:
    """Fit a feature + model sklearn Pipeline.

    Parameters
    ----------
    train_df, val_df
        Raw DataFrames from `build_raw_df`.
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

    X_tr, y_tr, g_tr, m_tr = prepare_df(train_df, use_base_margin)
    X_va, y_va, g_va, m_va = prepare_df(val_df, use_base_margin)

    logger.info(
        f"train ({model_type}): {len(y_tr):,} rows / {len(g_tr):,} races | "
        f"val: {len(y_va):,} rows / {len(g_va):,} races"
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
    X_tr_numpy = feature_pipeline.fit_transform(X_tr)
    X_va_numpy = feature_pipeline.transform(X_va)

    fit_kwargs = {
        "base_margin": m_tr,
        "eval_set": [(X_va_numpy, y_va)],
        "base_margin_eval_set": [m_va] if use_base_margin else None,
        "verbose": 50,
    }
    if model_type == "ranker":
        fit_kwargs["group"] = g_tr
        fit_kwargs["eval_group"] = [g_va]

    # earlier pipeline steps (`feature_pipeline`) were already fit, so just need to fit
    # the estimator here
    estimator.fit(X_tr_numpy, y_tr, **fit_kwargs)

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

    df = build_raw_df(
        use_morning_line_as_live=args.use_morning_line_as_live,
        use_final_as_live=args.use_final_as_live,
    )
    train_df, val_df, _ = split_by_race(df)
    pipeline = train(
        train_df,
        val_df,
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
