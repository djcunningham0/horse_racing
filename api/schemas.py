"""Pydantic request/response models for the prediction API."""

from typing import Literal

from pydantic import BaseModel, Field


class RunnerInput(BaseModel):
    horse_name: str
    post_position: int
    morning_line_decimal: float = Field(
        description="Morning line in decimal odds, i.e. total payout per $1 (e.g. 6.0)"
    )
    tote_odds: float = Field(
        description="Current tote board odds as X-to-1 (e.g. 5.0 for 5-1; decimal odds = tote_odds + 1)"
    )
    weight_carried: int
    class_rating: float
    speed_fig_last1: float | None = None
    speed_fig_last2: float | None = None
    speed_fig_last3: float | None = None
    days_since_last: int | None = None
    num_prior_starts: int = 0


class RaceRequest(BaseModel):
    race_id: str = ""
    distance: float = Field(description="Distance in furlongs")
    surface: Literal["D", "T"] = Field(description="'D' for dirt, 'T' for turf")
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
