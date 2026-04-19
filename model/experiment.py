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
from xgboost import plot_importance

from model.calibration import fit_temperature
from model.evaluate import print_metrics_table, evaluate_splits
from model.features import DEFAULT_FEATURE_COLS, build_training_df, split_by_race
from model.paths import DEFAULT_MODEL_DIR
from model.train import DEFAULT_HYPERPARAMS, prepare_df, train

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "horse-racing"


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    parser = argparse.ArgumentParser(description="Run a model experiment")
    parser.add_argument("--label", required=True, help="Short name for this run")
    parser.add_argument("--description", default="", help="Longer description")
    args = parser.parse_args()

    run_id = run_experiment(label=args.label, description=args.description)
    logging.info(f"finished run ID: {run_id}")


def run_experiment(
    label: str,
    description: str = "",
    features: list[str] | None = None,
    hyperparameters: dict | None = None,
    split_kwargs: dict | None = None,
) -> str:
    """Train, evaluate, and log an experiment to MLflow. Returns the MLflow run ID."""
    features = features or DEFAULT_FEATURE_COLS
    params = {**DEFAULT_HYPERPARAMS, **(hyperparameters or {})}
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
        for k, v in split_kwargs.items():
            mlflow.log_param(f"split.{k}", v)

        # build data and split
        df = build_training_df()
        train_df, val_df, test_df = split_by_race(df, **split_kwargs)

        # train
        model = train(train_df, val_df, features=features, hyperparameters=params)

        # fit softmax temperature on val
        temperature = fit_temperature(model, val_df, features)
        logger.info(f"fit softmax temperature on val: T={temperature:.4f}")
        mlflow.log_metric("temperature", temperature)

        # evaluate
        metrics = evaluate_splits(
            model,
            features=features,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            temperature=temperature,
        )
        print_metrics_table(metrics)
        _log_metrics_to_mlflow(metrics)

        # feature importance
        _log_feature_importance(model, features, val_df)

        # save artifact
        DEFAULT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        artifact_path = DEFAULT_MODEL_DIR / f"{run.info.run_id}.joblib"
        joblib.dump(
            {"model": model, "features": features, "temperature": temperature},
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


def _log_feature_importance(model, features: list[str], val_df):
    """Log XGBoost gain importance and SHAP values as MLflow artifacts."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # -- XGBoost gain importance --
        importance = model.get_booster().get_score(importance_type="gain")
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

        # add any missing features with zero importance
        for x in features:
            if x not in importance:
                importance[x] = 0.0

        mlflow.log_dict(importance, "feature_importance_gain.json")

        fig, ax = plt.subplots(figsize=(8, max(4, len(importance) * 0.3)))
        plot_importance(model, importance_type="gain", values_format="{v:.2f}", ax=ax)
        fig.tight_layout()
        gain_plot = tmp_path / "feature_importance_gain.png"
        fig.savefig(gain_plot, dpi=100)
        plt.close(fig)
        mlflow.log_artifact(str(gain_plot))

        # -- SHAP --
        explainer = shap.TreeExplainer(model)
        X_val, _, _ = prepare_df(val_df, features)
        shap_values = explainer.shap_values(X_val)
        explanation = shap.Explanation(
            shap_values,
            data=X_val.to_pandas(),
            feature_names=X_val.columns,
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
