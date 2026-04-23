"""POST a staged race-day CSV to a running API.

Reads the flat CSV produced by `scripts/ingest_brisnet`, groups by `race_number`,
builds a `CreateRaceRequest` per race, and POSTs to `/races`. Errors out if any
race already exists on the server (the API returns 409); resolve by clearing
state on the server first.

Usage:
    python -m scripts.post_staged_races data/staging/CDX0425.csv
    python -m scripts.post_staged_races data/staging/CDX0425.csv --base-url https://api.example.com
"""

import argparse
import logging
import os
from pathlib import Path

import httpx
import polars as pl

from api.schemas import CreateRaceRequest, StaticRunnerInput

logger = logging.getLogger(__name__)

RACE_LEVEL_FIELDS = [
    "track",
    "race_number",
    "race_date",
    "distance_yards",
    "surface",
    "course_desc",
    "race_class_rating",
    "purse",
    "age_restriction",
    "sex_restriction",
]


def load_staged(csv_path: Path) -> pl.DataFrame:
    return pl.read_csv(csv_path, null_values=[""], try_parse_dates=True)


def to_create_race_request(race_df: pl.DataFrame) -> CreateRaceRequest:
    first = race_df.row(0, named=True)
    runner_fields = set(StaticRunnerInput.model_fields.keys())
    runners = [
        StaticRunnerInput(**{k: v for k, v in row.items() if k in runner_fields})
        for row in race_df.iter_rows(named=True)
    ]
    return CreateRaceRequest(
        **{f: first[f] for f in RACE_LEVEL_FIELDS},
        runners=runners,
    )


def post_staged(csv_path: Path, base_url: str):
    df = load_staged(csv_path)
    race_numbers = df["race_number"].unique(maintain_order=True).to_list()
    logger.info(
        f"loaded {df.height} runners across {len(race_numbers)} races from {csv_path}"
    )

    user, pwd = os.environ.get("APP_USERNAME"), os.environ.get("APP_PASSWORD")
    auth = (user, pwd) if user and pwd else None
    with httpx.Client(base_url=base_url, timeout=10.0, auth=auth) as client:
        for race_number in race_numbers:
            race_df = df.filter(pl.col("race_number") == race_number).sort(
                "post_position"
            )
            payload = to_create_race_request(race_df).model_dump(mode="json")
            response = client.post("/races", json=payload)
            if response.status_code == 409:
                raise RuntimeError(
                    f"race {race_number}: {response.json().get('detail', response.text)}"
                )
            response.raise_for_status()
            created_id = response.json()["race_id"]
            logger.info(f"  created {created_id} ({race_df.height} runners)")


def main():
    parser = argparse.ArgumentParser(
        description="POST a staged race-day CSV to a running API.",
    )
    parser.add_argument(
        "csv_path",
        type=Path,
        help="Path to staged CSV (e.g. data/staging/CDX0425.csv)",
    )
    parser.add_argument("--base-url", default="http://localhost:8000")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    post_staged(args.csv_path, args.base_url)


if __name__ == "__main__":
    main()
