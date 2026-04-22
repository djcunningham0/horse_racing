"""JSON-file persistence for pre-loaded races.

The API keeps pre-loaded races in `app.state.races` so they survive across
requests. This module writes that dict to disk after every mutation so a
container restart doesn't wipe the day's entries and odds edits.
"""

import json
import os
import tempfile
from pathlib import Path

from api.schemas import StoredRace

DEFAULT_STORE_PATH = Path("state/races.json")


def get_store_path() -> Path:
    """Resolve the races store path from the RACES_STORE_PATH env var."""
    return Path(os.environ.get("RACES_STORE_PATH", DEFAULT_STORE_PATH))


def load_races(path: Path) -> dict[str, StoredRace]:
    """Load races from a JSON file, or return an empty dict if missing."""
    if not path.exists():
        return {}
    with path.open() as f:
        data = json.load(f)
    return {race_id: StoredRace(**race) for race_id, race in data.items()}


def save_races(path: Path, races: dict[str, StoredRace]):
    """Persist races to a JSON file atomically (write-temp + rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {race_id: race.model_dump(mode="json") for race_id, race in races.items()}
    # write to a temp file in the same directory, then atomic rename
    fd, tmp_path = tempfile.mkstemp(prefix=path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp_path, path)
    except Exception:
        Path(tmp_path).unlink(missing_ok=True)
        raise
