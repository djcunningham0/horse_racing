"""Prediction logic: load model, build features, score runners."""

from pathlib import Path

import joblib
import polars as pl

from model.betting import add_ev_columns
from model.features import derive_pp_rollup_features
from model.inference import predict_from_raw
from model.paths import DEFAULT_MODEL_DIR, MODEL_FILENAME

from api.schemas import RaceRequest, RunnerPrediction


def load_model(
    model_dir: Path | str = DEFAULT_MODEL_DIR,
    model_filename: Path | str = MODEL_FILENAME,
) -> dict:
    """Load the serialized model bundle (pipeline + calibration metadata)."""
    if isinstance(model_dir, str):
        model_dir = Path(model_dir)

    return joblib.load(model_dir / model_filename)


def predict_race(request: RaceRequest, model_bundle: dict) -> list[RunnerPrediction]:
    """Score all runners in a race and return predictions sorted by EV."""
    raw_df = derive_pp_rollup_features(_build_raw_df(request))
    scored = predict_from_raw(raw_df, model_bundle)
    scored = add_ev_columns(scored, dollar_odds_col="live_odds")

    return [
        RunnerPrediction(
            horse_name=row["horse_name"],
            post_position=row["post_position"],
            model_prob=round(row["model_prob"], 4),
            market_prob=round(row["market_prob"], 4),
            edge=round(row["model_prob"] - row["market_prob"], 4),
            ev_per_dollar=round(row["ev_per_dollar"], 4),
        )
        for row in scored.sort("ev_per_dollar", descending=True).iter_rows(named=True)
    ]


def _build_raw_df(request: RaceRequest) -> pl.DataFrame:
    """
    Assemble a raw DataFrame matching the columns produced by `build_raw_df` +
    `aggregate_pp_features`. Client-supplied rollups are used directly; per-race and
    derived PP features are layered on downstream.
    """
    race_id = request.race_id or f"api-{request.race_date.isoformat()}"
    field_size = len(request.runners)
    rows = []
    for r in request.runners:
        # rename runner fields as needed, then add race-level fields
        row = r.model_dump()
        row["morning_line_odds_float"] = row.pop("morning_line_odds")  # rename
        row["race_id"] = race_id
        row["race_date"] = request.race_date
        row["distance_yards"] = request.distance_yards
        row["surface"] = request.surface
        row["course_desc"] = request.course_desc
        row["race_class_rating"] = request.race_class_rating
        row["purse"] = request.purse
        row["age_restriction"] = request.age_restriction
        row["sex_restriction"] = request.sex_restriction
        row["field_size"] = field_size
        rows.append(row)
    return pl.DataFrame(rows)
