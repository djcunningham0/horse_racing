"""Pydantic request/response models for the prediction API."""

from typing import Literal, TypeAlias

from pydantic import BaseModel, Field

SurfaceType: TypeAlias = Literal["D", "T"]


class StaticRunnerInput(BaseModel):
    """Runner data that can be pre-loaded (everything except tote odds)."""

    horse_name: str
    post_position: int
    morning_line_decimal: float = Field(
        description="Morning line in decimal odds, i.e. total payout per $1 (e.g. 6.0)"
    )
    weight_carried: int
    class_rating: float
    speed_fig_last1: float | None = None
    speed_fig_last2: float | None = None
    speed_fig_last3: float | None = None
    days_since_last: int | None = None
    num_prior_starts: int = 0


class RunnerInput(StaticRunnerInput):
    """Full runner input including live tote odds."""

    tote_odds: float = Field(
        description="Current tote board odds as X-to-1 (e.g. 5.0 for 5-1; decimal odds = tote_odds + 1)"
    )


class RaceRequest(BaseModel):
    race_id: str = ""
    distance: float = Field(description="Distance in furlongs")
    surface: SurfaceType = Field(description="'D' for dirt, 'T' for turf")
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
    distance: float = Field(description="Distance in furlongs")
    surface: SurfaceType = Field(description="'D' for dirt, 'T' for turf")
    runners: list[StaticRunnerInput] = Field(min_length=2)


class CreateRaceResponse(BaseModel):
    race_id: str


class StoredRunner(StaticRunnerInput):
    """Runner stored in memory, with optional tote odds added at the track."""

    tote_odds: float | None = None
    scratched: bool = False


class StoredRace(BaseModel):
    race_id: str
    track: str
    race_number: int
    distance: float
    surface: SurfaceType
    runners: list[StoredRunner]


class RunnerOdds(BaseModel):
    post_position: int
    tote_odds: float = Field(gt=0)


class OddsUpdate(BaseModel):
    odds: list[RunnerOdds]


class RaceSummary(BaseModel):
    race_id: str
    track: str
    race_number: int
    distance: float
    surface: SurfaceType
    num_runners: int
