"""FastAPI app for horse racing predictions."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.persistence import get_store_path, load_races
from api.predict import load_model, predict_race
from api.races import router as races_router
from api.schemas import PredictionResponse, RaceRequest


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model_bundle = load_model()
    app.state.races_path = get_store_path()
    app.state.races = load_races(app.state.races_path)
    yield


app = FastAPI(title="Horse Racing Predictor", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(races_router)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(race_request: RaceRequest, request: Request) -> PredictionResponse:
    predictions = predict_race(race_request, request.app.state.model_bundle)
    return PredictionResponse(race_id=race_request.race_id, predictions=predictions)


# serve the frontend from the `frontend/` directory (this line must be after all routes
# have been defined))
app.mount("/app", StaticFiles(directory="frontend", html=True), name="frontend")
