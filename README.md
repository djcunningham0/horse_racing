# Horse Racing Predictor

Proof-of-concept system to predict horse racing outcomes and identify positive expected value win bets, built for Churchill Downs opening weekend 2026.

Check out my blog post for a detailed overview of the project:
https://dannycunningham.com/posts/2026-05-08-horse-racing-model/

## Setup

```bash
python -m venv venv_horse_racing
source venv_horse_racing/bin/activate
pip install -e ".[dev]"
```

## Running

Train a model (writes a `joblib` artifact under `model/artifacts/`):

```bash
python -m model.train
```

Serve the API locally:

```bash
uvicorn api.main:app --reload
# or
docker compose up
```

Launch the frontend by opening `frontend/index.html` in a browser (point it at the local API).

## Project Structure

```
data/           Schema definitions, ingestion, raw/processed data
model/          Feature engineering, training, evaluation, serialized models
                (XGBoost classifier → softmax probabilities → EV vs. tote odds)
api/            FastAPI serving layer, Dockerized
frontend/       Mobile-friendly web app for odds entry and prediction display
scripts/        Data ingestion and race pre-loading utilities
notebooks/      EDA and prototyping
tests/          Tests
```
