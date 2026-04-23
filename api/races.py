"""Race pre-loading endpoints: create, list, update odds, scratch, predict."""

from fastapi import APIRouter, HTTPException, Request
from starlette.status import HTTP_201_CREATED, HTTP_204_NO_CONTENT

from api.persistence import save_races
from api.predict import predict_race
from api.schemas import (
    CreateRaceRequest,
    CreateRaceResponse,
    OddsUpdate,
    PredictionResponse,
    RaceRequest,
    RaceSummary,
    RunnerInput,
    StoredRace,
    StoredRunner,
    TwinSpiresOddsResponse,
)
from api import twinspires

router = APIRouter(prefix="/races", tags=["races"])


def _get_race(request: Request, race_id: str) -> StoredRace:
    races: dict[str, StoredRace] = request.app.state.races
    try:
        return races[race_id]
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Race '{race_id}' not found")


def _persist(request: Request):
    save_races(request.app.state.races_path, request.app.state.races)


def _find_runner(race: StoredRace, post_position: int) -> StoredRunner:
    for r in race.runners:
        if r.post_position == post_position:
            return r
    raise HTTPException(
        status_code=404, detail=f"No runner at post position {post_position}"
    )


def _active_runners(race: StoredRace) -> list[StoredRunner]:
    return [r for r in race.runners if not r.scratched]


@router.post("", response_model=CreateRaceResponse, status_code=HTTP_201_CREATED)
def create_race(body: CreateRaceRequest, request: Request) -> CreateRaceResponse:
    race_id = f"{body.track}-R{body.race_number}"
    if race_id in request.app.state.races:
        raise HTTPException(status_code=409, detail=f"Race '{race_id}' already exists")

    runners = [
        StoredRunner(**r.model_dump(), live_odds=r.morning_line_odds)
        for r in body.runners
    ]
    race = StoredRace(
        race_id=race_id,
        track=body.track,
        race_number=body.race_number,
        race_date=body.race_date,
        distance_yards=body.distance_yards,
        surface=body.surface,
        course_desc=body.course_desc,
        race_class_rating=body.race_class_rating,
        purse=body.purse,
        age_restriction=body.age_restriction,
        sex_restriction=body.sex_restriction,
        runners=runners,
    )
    request.app.state.races[race_id] = race
    _persist(request)
    return CreateRaceResponse(race_id=race_id)


@router.get("", response_model=list[RaceSummary])
def list_races(request: Request) -> list[RaceSummary]:
    races: dict[str, StoredRace] = request.app.state.races
    return [
        RaceSummary(
            race_id=race.race_id,
            track=race.track,
            race_number=race.race_number,
            race_date=race.race_date,
            distance_yards=race.distance_yards,
            surface=race.surface,
            num_runners=len(_active_runners(race)),
        )
        for race in races.values()
    ]


@router.get("/{race_id}", response_model=StoredRace)
def get_race(race_id: str, request: Request) -> StoredRace:
    return _get_race(request, race_id)


@router.delete("", status_code=HTTP_204_NO_CONTENT)
def delete_all_races(request: Request):
    request.app.state.races.clear()
    _persist(request)


@router.delete("/{race_id}", status_code=HTTP_204_NO_CONTENT)
def delete_race(race_id: str, request: Request):
    races: dict[str, StoredRace] = request.app.state.races
    if race_id not in races:
        raise HTTPException(status_code=404, detail=f"Race '{race_id}' not found")
    del races[race_id]
    _persist(request)


@router.patch("/{race_id}/odds", response_model=StoredRace)
def update_odds(race_id: str, body: OddsUpdate, request: Request) -> StoredRace:
    race = _get_race(request, race_id)
    runners_by_post = {r.post_position: r for r in race.runners}

    for entry in body.odds:
        runner = runners_by_post.get(entry.post_position)
        if runner is None:
            raise HTTPException(
                status_code=422,
                detail=f"No runner at post position {entry.post_position}",
            )
        runner.live_odds = entry.live_odds

    _persist(request)
    return race


@router.patch("/{race_id}/runners/{post_position}/scratch", response_model=StoredRace)
def scratch_runner(race_id: str, post_position: int, request: Request) -> StoredRace:
    race = _get_race(request, race_id)
    runner = _find_runner(race, post_position)

    if runner.scratched:
        raise HTTPException(
            status_code=422,
            detail=f"Runner at post position {post_position} is already scratched",
        )
    if len(_active_runners(race)) < 3:
        raise HTTPException(
            status_code=422, detail="Cannot scratch: race must have at least 2 runners"
        )

    runner.scratched = True
    _persist(request)
    return race


@router.patch("/{race_id}/runners/{post_position}/unscratch", response_model=StoredRace)
def unscratch_runner(race_id: str, post_position: int, request: Request) -> StoredRace:
    race = _get_race(request, race_id)
    runner = _find_runner(race, post_position)

    if not runner.scratched:
        raise HTTPException(
            status_code=422,
            detail=f"Runner at post position {post_position} is not scratched",
        )

    runner.scratched = False
    _persist(request)
    return race


@router.post("/{race_id}/fetch-twinspires-odds", response_model=TwinSpiresOddsResponse)
def fetch_twinspires_odds_endpoint(
    race_id: str,
    request: Request,
) -> TwinSpiresOddsResponse:
    race = _get_race(request, race_id)
    try:
        fetched_at, odds_by_post = twinspires.fetch_twinspires_odds(
            race.track, race.race_number
        )
    except twinspires.TwinSpiresRaceNotFound as e:
        raise HTTPException(status_code=404, detail=str(e))
    except twinspires.TwinSpiresError as e:
        raise HTTPException(status_code=502, detail=f"TwinSpires unavailable: {e}")

    applied: list[int] = []
    missing: list[int] = []
    for runner in race.runners:
        if runner.scratched:
            continue
        new_odds = odds_by_post.get(runner.post_position)
        if new_odds is None:
            missing.append(runner.post_position)
        else:
            runner.live_odds = new_odds
            applied.append(runner.post_position)

    if applied:
        _persist(request)

    return TwinSpiresOddsResponse(
        race=race,
        fetched_at=fetched_at,
        applied_post_positions=applied,
        missing_post_positions=missing,
    )


@router.post("/{race_id}/predict", response_model=PredictionResponse)
def predict_stored_race(race_id: str, request: Request) -> PredictionResponse:
    race = _get_race(request, race_id)

    active = _active_runners(race)
    missing = [r.post_position for r in active if r.live_odds is None]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Live odds not set for post positions: {missing}",
        )

    runner_inputs = [RunnerInput(**r.model_dump()) for r in active]
    race_request = RaceRequest(
        race_id=race.race_id,
        race_date=race.race_date,
        distance_yards=race.distance_yards,
        surface=race.surface,
        course_desc=race.course_desc,
        race_class_rating=race.race_class_rating,
        purse=race.purse,
        age_restriction=race.age_restriction,
        sex_restriction=race.sex_restriction,
        runners=runner_inputs,
    )
    predictions = predict_race(race_request, request.app.state.model_bundle)
    return PredictionResponse(race_id=race_id, predictions=predictions)
