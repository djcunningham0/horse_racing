"""Tests for race store persistence across app restarts."""

from contextlib import contextmanager
from unittest.mock import patch

from fastapi.testclient import TestClient

from tests.api.conftest import _fake_predict_from_raw, sample_race_body


@contextmanager
def _make_client(store_path):
    """Spin up a TestClient pointing at a specific races.json path."""
    with (
        patch.dict("os.environ", {"RACES_STORE_PATH": str(store_path)}),
        patch("api.predict.load_model", return_value={"pipeline": None}),
        patch("api.predict.predict_from_raw", side_effect=_fake_predict_from_raw),
    ):
        from api.main import app

        with TestClient(app) as c:
            yield c


class TestPersistence:
    def test_races_survive_restart(self, tmp_path):
        store = tmp_path / "races.json"

        with _make_client(store) as c1:
            c1.post("/races", json=sample_race_body())
            c1.patch(
                "/races/CD-R1/odds",
                json={"odds": [{"post_position": 1, "live_odds": 3.5}]},
            )
            c1.patch("/races/CD-R1/runners/2/scratch")

        assert store.exists()

        with _make_client(store) as c2:
            race = c2.get("/races/CD-R1").json()
            runners = {r["post_position"]: r for r in race["runners"]}
            assert runners[1]["live_odds"] == 3.5
            assert runners[2]["scratched"] is True

    def test_empty_store_file_absent_on_startup(self, tmp_path):
        store = tmp_path / "races.json"
        with _make_client(store) as c:
            assert c.get("/races").json() == []
        # no mutations -> no file written
        assert not store.exists()

    def test_create_race_writes_to_disk(self, tmp_path):
        store = tmp_path / "races.json"
        with _make_client(store) as c:
            c.post("/races", json=sample_race_body())
        assert store.exists()
        assert "CD-R1" in store.read_text()
