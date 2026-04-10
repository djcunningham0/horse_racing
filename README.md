# Horse Racing Predictor

Proof-of-concept system to predict horse racing outcomes and identify positive expected value win bets, built for Churchill Downs opening weekend 2026.

## Setup

```bash
python -m venv venv_horse_racing
source venv_horse_racing/bin/activate
pip install -e ".[dev]"
```

## Project Structure

```
data/           Schema definitions, ingestion, raw/processed data
model/          Feature engineering, training, evaluation, serialized models
api/            FastAPI serving layer
frontend/       Mobile web app
notebooks/      EDA and prototyping
tests/          Tests
```

## Workstreams

1. **Data collection** — canonical schema, ingestion pipeline, parsers for provider formats
2. **Modeling** — XGBoost ranker → softmax probabilities → EV calculation vs. tote odds
3. **API** — FastAPI `/predict` endpoint, Dockerized
4. **Frontend** — Mobile-friendly web app for odds entry and prediction display at the track
