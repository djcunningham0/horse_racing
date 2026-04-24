"""
Run and track model experiments with MLflow.

Usage:
    python -m model.experiment --label baseline --description "v1 14-feature XGBoost"
"""

import argparse
import logging
import subprocess
import tempfile
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import shap
from sklearn.pipeline import Pipeline
from xgboost import plot_importance

from model.calibration import fit_temperature
from model.evaluate import print_metrics_table, evaluate_splits
from model.feature_pipeline import FEATURE_NAMES
from model.features import build_raw_df
from model.split import DEFAULT_SPLIT_MODE, SPLIT_MODES, split_by_race
from model.paths import DEFAULT_MODEL_DIR
from model.train import DEFAULT_HYPERPARAMS, _temperature_arg, prepare_df, train

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "horse-racing"


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    parser = argparse.ArgumentParser(description="Run a model experiment")
    parser.add_argument("--label", required=True, help="Short name for this run")
    parser.add_argument("--description", default="", help="Longer description")
    parser.add_argument(
        "--use-base-margin",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use logit(market_prob) as XGBoost base_margin",
    )
    parser.add_argument(
        "--model-type",
        choices=["ranker", "classifier"],
        default="classifier",
        help="XGBoost model type",
    )
    parser.add_argument(
        "--temperature",
        type=_temperature_arg,
        default=1.0,
        help="Softmax temperature (float) or 'auto' to fit on the validation set.",
    )
    parser.add_argument(
        "--split-mode",
        choices=SPLIT_MODES,
        default=DEFAULT_SPLIT_MODE,
        help=(
            "How to split races into train/val/test. 'random' shuffles by race_id; "
            "'chronological' uses earliest races for train and latest for test."
        ),
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

    run_id = run_experiment(
        label=args.label,
        description=args.description,
        split_kwargs={"mode": args.split_mode},
        use_base_margin=args.use_base_margin,
        model_type=args.model_type,
        temperature=args.temperature,
        use_morning_line_as_live=args.use_morning_line_as_live,
        use_final_as_live=args.use_final_as_live,
    )
    logging.info(f"finished run ID: {run_id}")


def run_experiment(
    label: str,
    description: str = "",
    features: list[str] | None = None,
    hyperparameters: dict | None = None,
    split_kwargs: dict | None = None,
    use_base_margin: bool = True,
    model_type: str = "classifier",
    temperature: float | str = 1.0,
    use_morning_line_as_live: bool = False,
    use_final_as_live: bool = False,
) -> str:
    """Train, evaluate, and log an experiment to MLflow. Returns the MLflow run ID."""
    features = features or FEATURE_NAMES
    params = {**DEFAULT_HYPERPARAMS[model_type], **(hyperparameters or {})}
    split_kwargs = split_kwargs or {}

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=label) as run:
        mlflow.set_tag("description", description)
        git_sha = _git_sha()
        if git_sha:
            mlflow.set_tag("git_sha", git_sha)

        # log params
        mlflow.log_params(params)
        mlflow.log_param("n_features", len(features))
        mlflow.log_param("features", ", ".join(features))
        mlflow.log_param("use_base_margin", use_base_margin)
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("use_morning_line_as_live", use_morning_line_as_live)
        mlflow.log_param("use_final_as_live", use_final_as_live)
        for k, v in split_kwargs.items():
            mlflow.log_param(f"split.{k}", v)

        # build data and split
        df = build_raw_df(
            use_morning_line_as_live=use_morning_line_as_live,
            use_final_as_live=use_final_as_live,
        )
        train_df, val_df, test_df = split_by_race(df, **split_kwargs)

        # train
        pipeline = train(
            train_df,
            val_df,
            features=features,
            hyperparameters=params,
            use_base_margin=use_base_margin,
            model_type=model_type,
        )

        # softmax temperature: fixed if a float, else fit on val
        fit_temp = temperature == "auto"
        if fit_temp:
            temperature = fit_temperature(
                pipeline, val_df, use_base_margin=use_base_margin
            )
            logger.info(f"fit softmax temperature on val: T={temperature:.4f}")
        else:
            temperature = float(temperature)
            logger.info(f"using fixed softmax temperature: T={temperature:.4f}")
        mlflow.log_metric("temperature", temperature)
        mlflow.log_param("temperature_fitted", fit_temp)

        # evaluate
        metrics = evaluate_splits(
            pipeline,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            temperature=temperature,
            use_base_margin=use_base_margin,
        )
        print_metrics_table(metrics)
        _log_metrics_to_mlflow(metrics)

        # feature importance
        _log_feature_importance(pipeline, features, val_df)

        # save artifact
        DEFAULT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        artifact_path = DEFAULT_MODEL_DIR / f"{run.info.run_id}.joblib"
        joblib.dump(
            {
                "pipeline": pipeline,
                "features": features,
                "temperature": temperature,
                "use_base_margin": use_base_margin,
                "model_type": model_type,
            },
            artifact_path,
        )
        mlflow.log_artifact(str(artifact_path))

        logger.info(f"experiment '{label}' logged as run {run.info.run_id}")
        return run.info.run_id


def _git_sha() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _log_feature_importance(pipeline: Pipeline, features: list[str], val_df):
    """Log XGBoost gain importance and SHAP values as MLflow artifacts."""
    estimator = pipeline.named_steps["model"]
    feature_pipeline = Pipeline(pipeline.steps[:-1])

    # xgboost uses positional names ("f0", "f1", ...) since the Pipeline feeds it a
    # numpy matrix — remap to real feature names
    booster = estimator.get_booster()
    booster.feature_names = list(features)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # -- XGBoost gain importance --
        importance = booster.get_score(importance_type="gain")
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

        # add any missing features with zero importance
        for x in features:
            if x not in importance:
                importance[x] = 0.0

        mlflow.log_dict(importance, "feature_importance_gain.json")

        fig, ax = plt.subplots(figsize=(8, max(4, len(importance) * 0.3)))
        plot_importance(
            estimator, importance_type="gain", values_format="{v:.2f}", ax=ax
        )
        fig.tight_layout()
        gain_plot = tmp_path / "feature_importance_gain.png"
        fig.savefig(gain_plot, dpi=100)
        plt.close(fig)
        mlflow.log_artifact(str(gain_plot))

        # -- SHAP --
        X_val_raw = prepare_df(val_df).X
        X_val = feature_pipeline.transform(X_val_raw)
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(X_val)
        explanation = shap.Explanation(
            shap_values,
            data=X_val,
            feature_names=features,
        )
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        shap_dict = dict(zip(features, mean_abs_shap.tolist()))
        shap_dict = dict(sorted(shap_dict.items(), key=lambda x: x[1], reverse=True))

        mlflow.log_dict(shap_dict, "feature_importance_shap.json")

        # SHAP bar chart
        fig, ax = plt.subplots()
        shap.plots.bar(explanation, ax=ax, show=False)
        shap_bar_path = tmp_path / "feature_importance_shap_bar.png"
        fig.savefig(shap_bar_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        mlflow.log_artifact(str(shap_bar_path))

        # SHAP beeswarm plot
        fig, ax = plt.subplots()
        shap.plots.beeswarm(explanation, max_display=20, show=False)
        shap_beeswarm_path = tmp_path / "feature_importance_shap_beeswarm.png"
        fig.savefig(shap_beeswarm_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        mlflow.log_artifact(str(shap_beeswarm_path))


def _log_metrics_to_mlflow(metrics: dict[str, dict]):
    """Flatten the nested metrics dict and log to MLflow."""
    for split_name, split_metrics in metrics.items():
        for key, val in split_metrics.items():
            if key == "roi":
                for rule_label, roi_summary in val.items():
                    # clean up label for mlflow metric name
                    clean_label = (
                        rule_label.replace(" ", "_")
                        .replace("(", "")
                        .replace(")", "")
                        .replace(">", "gt")
                        .replace("%", "pct")
                    )
                    for roi_key, roi_value in roi_summary.items():
                        mlflow.log_metric(
                            f"{split_name}.roi.{clean_label}.{roi_key}", roi_value
                        )
            elif isinstance(val, (int, float)):
                mlflow.log_metric(f"{split_name}.{key}", val)


if __name__ == "__main__":
    main()
