"""Optuna hyperparameter sweep for the XGBoost classifier.

Each trial trains with early stopping on the val set, fits the softmax temperature,
and reports val winner log-loss. Every trial is a nested MLflow run under a single
sweep parent run so the whole sweep is browsable as a group in the MLflow UI.

After the sweep, a final `run_experiment` call retrains the best config with full
train/val/test evaluation (including ROI simulation) for an honest report.

Usage:
    python -m model.sweep --n-trials 50 --label sweep_v1
"""

import argparse
import logging
import tempfile
from pathlib import Path

import mlflow
import optuna
import polars as pl

from model.calibration import fit_temperature
from model.evaluate import _metrics_for_split
from model.experiment import EXPERIMENT_NAME, run_experiment
from model.features import build_raw_df, split_by_race
from model.train import DEFAULT_HYPERPARAMS, train

logger = logging.getLogger(__name__)

# ceiling for n_estimators — early stopping on val log-loss effectively tunes this
MAX_N_ESTIMATORS = 3000


def sample_params(trial: optuna.Trial) -> dict:
    """Sample an XGBoost hyperparameter config for one trial."""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


def _run_trial(
    trial: optuna.Trial,
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    sweep_label: str,
    model_type: str,
    use_base_margin: bool,
) -> float:
    """Train, fit temperature, return val log-loss. Logs to a nested MLflow run."""
    sampled = sample_params(trial)
    params = {
        **DEFAULT_HYPERPARAMS[model_type],
        **sampled,
        "n_estimators": MAX_N_ESTIMATORS,
    }

    run_name = f"{sweep_label}_trial_{trial.number:03d}"
    with mlflow.start_run(run_name=run_name, nested=True):
        mlflow.set_tag("sweep", sweep_label)
        mlflow.set_tag("trial_number", str(trial.number))
        mlflow.log_params(params)
        mlflow.log_param("use_base_margin", use_base_margin)
        mlflow.log_param("model_type", model_type)

        pipeline = train(
            train_df,
            val_df,
            hyperparameters=params,
            use_base_margin=use_base_margin,
            model_type=model_type,
            verbose=False,
        )
        best_iteration = pipeline.named_steps["model"].best_iteration
        mlflow.log_metric("best_iteration", float(best_iteration))

        temperature = fit_temperature(pipeline, val_df, use_base_margin=use_base_margin)
        mlflow.log_metric("temperature", temperature)

        val_metrics = _metrics_for_split(
            val_df,
            pipeline,
            temperature=temperature,
            use_base_margin=use_base_margin,
        )
        val_log_loss = val_metrics["model_log_loss"]
        mlflow.log_metric("val.model_log_loss", val_log_loss)
        mlflow.log_metric("val.model_top1_acc", val_metrics["model_top1_acc"])
        mlflow.log_metric("val.market_log_loss", val_metrics["market_log_loss"])
        mlflow.log_metric("val.favorite_top1_acc", val_metrics["favorite_top1_acc"])

        logger.info(
            f"trial {trial.number:03d}: val log-loss={val_log_loss:.5f} "
            f"(best_iter={best_iteration}, T={temperature:.3f})"
        )
        return val_log_loss


def run_sweep(
    n_trials: int,
    label: str,
    model_type: str = "classifier",
    use_base_margin: bool = True,
    seed: int = 0,
) -> dict:
    """Run an Optuna TPE sweep and retrain the best config with full evaluation.

    Returns the best-trial hyperparameter dict.
    """
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = build_raw_df()
    train_df, val_df, _ = split_by_race(df)

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    with mlflow.start_run(run_name=f"{label}_parent"):
        mlflow.set_tag("sweep", label)
        mlflow.set_tag("role", "parent")
        mlflow.log_param("n_trials", n_trials)
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("use_base_margin", use_base_margin)
        mlflow.log_param("seed", seed)

        def objective(trial: optuna.Trial) -> float:
            return _run_trial(
                trial,
                train_df,
                val_df,
                sweep_label=label,
                model_type=model_type,
                use_base_margin=use_base_margin,
            )

        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best = study.best_trial
        mlflow.log_metric("best_val_log_loss", best.value)
        mlflow.log_metric("best_trial_number", float(best.number))
        for k, v in best.params.items():
            mlflow.log_param(f"best.{k}", v)

        with tempfile.TemporaryDirectory() as tmp:
            trials_path = Path(tmp) / "sweep_trials.csv"
            study.trials_dataframe().to_csv(trials_path, index=False)
            mlflow.log_artifact(str(trials_path))

    logger.info(f"best trial: {best.number}, val log-loss={best.value:.5f}")
    logger.info(f"best params: {best.params}")
    _print_top_trials(study, n=10)

    # honest report: retrain best config, evaluate all three splits with full metrics.
    # include the same n_estimators ceiling the trial used so early stopping picks the
    # same point on the loss curve.
    best_hyperparams = {**best.params, "n_estimators": MAX_N_ESTIMATORS}
    logger.info(f"retraining best config as '{label}_best' with full evaluation...")
    run_experiment(
        label=f"{label}_best",
        description=f"Best trial from sweep '{label}' (trial {best.number})",
        hyperparameters=best_hyperparams,
        model_type=model_type,
        temperature="auto",
        use_base_margin=use_base_margin,
    )

    return best.params


def _print_top_trials(study: optuna.Study, n: int = 10):
    """Print a compact table of the top-N trials by objective value."""
    completed = [t for t in study.trials if t.value is not None]
    top = sorted(completed, key=lambda t: t.value)[:n]
    print(f"\n=== Top {len(top)} trials (by val log-loss) ===")
    header = f"{'trial':>6} {'log_loss':>10}  params"
    print(header)
    print("-" * len(header))
    for t in top:
        params_str = ", ".join(
            f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
            for k, v in t.params.items()
        )
        print(f"{t.number:>6} {t.value:>10.5f}  {params_str}")


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    parser = argparse.ArgumentParser(
        description="Run an Optuna hyperparameter sweep for the XGBoost classifier"
    )
    parser.add_argument(
        "--n-trials", type=int, required=True, help="Number of Optuna trials"
    )
    parser.add_argument(
        "--label", required=True, help="Short name for this sweep (used in MLflow)"
    )
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
    parser.add_argument("--seed", type=int, default=0, help="TPE sampler seed")
    args = parser.parse_args()

    run_sweep(
        n_trials=args.n_trials,
        label=args.label,
        model_type=args.model_type,
        use_base_margin=args.use_base_margin,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
