"""Pydantic request/response models for the prediction API.

Runner input fields mirror the columns that `build_raw_df` + `aggregate_pp_features`
produce, so the API can assemble a DataFrame that feeds directly into
`predict_from_raw`. PP and workout rollups are pre-aggregated by the client
(one row per runner, not one row per prior start).
"""

from datetime import date
from typing import Literal, TypeAlias

from pydantic import BaseModel, Field

SurfaceType: TypeAlias = Literal["D", "T"]


class StaticRunnerInput(BaseModel):
    """
    Runner data that can be pre-loaded (everything except live odds). Should
    include all required "raw" data fields, but not the features derived in the sklearn
    model pipeline.
    """

    # identity
    horse_name: str
    post_position: int

    # entry
    morning_line_odds: float = Field(
        description="Morning line odds as X-to-1 (e.g. 5.0 for 5-1)"
    )
    weight_carried: int | None
    entry_class_rating: float | None
    career_starts: int | None = None
    career_wins: int | None = None
    career_seconds: int | None = None
    career_thirds: int | None = None
    career_earnings: float | None = None
    surface_starts: int | None = None
    surface_wins: int | None = None
    surface_seconds: int | None = None
    surface_thirds: int | None = None

    # PP rollups — null means "no prior race at this index"
    speed_fig_L1: float | None = None
    speed_fig_L2: float | None = None
    speed_fig_L3: float | None = None
    avg_speed_fig_L3: float | None = None
    max_speed_fig_L3: float | None = None
    class_rating_L1: float | None = None
    class_rating_L2: float | None = None
    class_rating_L3: float | None = None
    avg_class_rating_L3: float | None = None
    max_class_rating_L3: float | None = None
    official_finish_L1: int | None = None
    official_finish_L2: int | None = None
    official_finish_L3: int | None = None
    num_starters_L1: int | None = None
    num_starters_L2: int | None = None
    num_starters_L3: int | None = None
    pp_odds_L1: float | None = None
    pp_odds_L2: float | None = None
    pp_odds_L3: float | None = None
    distance_yards_L1: int | None = None
    distance_yards_L2: int | None = None
    distance_yards_L3: int | None = None
    pp_surface_L1: str | None = Field(
        default=None,
        description="Surface code from last PP: T/I/O (turf), D (dirt), E (all-weather)",
    )
    last_pp_date: date | None = None
    num_prior_starts: int = 0

    # workout rollups — null means "no workouts on file"
    best_workout_rank_pct: float | None = None
    best_workout_group_size: int | None = None
    last_workout_rank_pct: float | None = None
    last_workout_group_size: int | None = None
    num_workouts: int = 0
    last_workout_date: date | None = None


class RunnerInput(StaticRunnerInput):
    """Full runner input including live odds."""

    live_odds: float = Field(
        description="Current live odds from the tote board as X-to-1 (e.g. 5.0 for 5-1)"
    )


class RaceRequest(BaseModel):
    race_id: str = ""
    race_date: date
    distance_yards: int = Field(gt=0, description="Race distance in yards")
    surface: SurfaceType = Field(description="'D' for dirt, 'T' for turf")
    course_desc: str | None = Field(
        default=None,
        description="Course description from results; used to detect 'All Weather Track'",
    )
    race_class_rating: float
    purse: float
    runners: list[RunnerInput] = Field(min_length=2)


class RunnerPrediction(BaseModel):
    horse_name: str
    post_position: int
    model_prob: float
    market_prob: float
    edge: float
    ev_per_dollar: float


class PredictionResponse(BaseModel):
    race_id: str
    predictions: list[RunnerPrediction]


# --- Race pre-loading models ---


class CreateRaceRequest(BaseModel):
    track: str = Field(description="Track code, e.g. 'CD'")
    race_number: int = Field(ge=1)
    race_date: date
    distance_yards: int = Field(gt=0, description="Race distance in yards")
    surface: SurfaceType = Field(description="'D' for dirt, 'T' for turf")
    course_desc: str | None = None
    race_class_rating: float
    purse: float
    runners: list[StaticRunnerInput] = Field(min_length=2)


class CreateRaceResponse(BaseModel):
    race_id: str


class StoredRunner(StaticRunnerInput):
    """Runner stored in memory, with optional live odds added at the track."""

    live_odds: float | None = None
    scratched: bool = False


class StoredRace(BaseModel):
    race_id: str
    track: str
    race_number: int
    race_date: date
    distance_yards: int
    surface: SurfaceType
    course_desc: str | None = None
    race_class_rating: float
    purse: float
    runners: list[StoredRunner]


class RunnerOdds(BaseModel):
    post_position: int
    live_odds: float = Field(gt=0)


class OddsUpdate(BaseModel):
    odds: list[RunnerOdds]


class RaceSummary(BaseModel):
    race_id: str
    track: str
    race_number: int
    race_date: date
    distance_yards: int
    surface: SurfaceType
    num_runners: int
