"""FastAPI app for horse racing predictions."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from api.predict import load_model, predict_race
from api.schemas import PredictionResponse, RaceRequest


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model_bundle = load_model()
    yield


app = FastAPI(title="Horse Racing Predictor", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(race_request: RaceRequest, request: Request) -> PredictionResponse:
    predictions = predict_race(race_request, request.app.state.model_bundle)
    return PredictionResponse(race_id=race_request.race_id, predictions=predictions)
