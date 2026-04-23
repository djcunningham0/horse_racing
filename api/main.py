"""FastAPI app for horse racing predictions."""

import base64
import os
import secrets
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from starlette.status import HTTP_401_UNAUTHORIZED

from api.persistence import get_store_path, load_races
from api.predict import load_model, predict_race
from api.races import router as races_router
from api.schemas import PredictionResponse, RaceRequest

# paths that skip Basic Auth (Render health checks hit /health)
AUTH_EXEMPT_PATHS = {"/health"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model_bundle = load_model()
    app.state.races_path = get_store_path()
    app.state.races = load_races(app.state.races_path)
    yield


app = FastAPI(title="Horse Racing Predictor", lifespan=lifespan)

_username = os.environ.get("APP_USERNAME")
_password = os.environ.get("APP_PASSWORD")

if _username and _password:
    _expected_auth = "Basic " + base64.b64encode(
        f"{_username}:{_password}".encode()
    ).decode()

    @app.middleware("http")
    async def basic_auth(request: Request, call_next):
        if request.url.path in AUTH_EXEMPT_PATHS:
            return await call_next(request)
        header = request.headers.get("authorization", "")
        if secrets.compare_digest(header, _expected_auth):
            return await call_next(request)
        return Response(
            status_code=HTTP_401_UNAUTHORIZED,
            headers={"WWW-Authenticate": 'Basic realm="Horse Racing Predictor"'},
        )
else:
    print(
        "WARNING: APP_USERNAME/APP_PASSWORD not set; API is unauthenticated",
        file=sys.stderr,
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
