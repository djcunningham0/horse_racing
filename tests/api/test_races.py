"""Tests for race pre-loading endpoints."""

from datetime import datetime

from tests.api.conftest import sample_race_body


class TestCreateRace:
    def test_create_race(self, client):
        r = client.post("/races", json=sample_race_body())
        assert r.status_code == 201
        assert r.json()["race_id"] == "CD-R1"

    def test_live_odds_default_to_morning_line(self, client):
        _ = client.post("/races", json=sample_race_body())
        r = client.get("/races/CD-R1")
        race = r.json()
        for runner in race["runners"]:
            assert runner["live_odds"] == runner["morning_line_odds"]

    def test_duplicate_race(self, client):
        client.post("/races", json=sample_race_body())
        r = client.post("/races", json=sample_race_body())
        assert r.status_code == 409

    def test_post_time_roundtrips(self, client):
        body = sample_race_body()
        body["post_time"] = "5:34 PM ET"
        client.post("/races", json=body)
        r = client.get("/races/CD-R1")
        assert r.json()["post_time"] == "5:34 PM ET"

    def test_post_time_optional(self, client):
        # sample_race_body does not set post_time; should default to None
        client.post("/races", json=sample_race_body())
        r = client.get("/races/CD-R1")
        assert r.json()["post_time"] is None


class TestGetRace:
    def test_get_race(self, client):
        client.post("/races", json=sample_race_body())
        r = client.get("/races/CD-R1")
        assert r.status_code == 200
        assert r.json()["race_id"] == "CD-R1"
        assert len(r.json()["runners"]) == 3

    def test_get_missing_race(self, client):
        r = client.get("/races/NOPE-R1")
        assert r.status_code == 404


class TestListRaces:
    def test_list_empty(self, client):
        r = client.get("/races")
        assert r.status_code == 200
        assert r.json() == []

    def test_list_races(self, client):
        client.post("/races", json=sample_race_body())
        client.post("/races", json=sample_race_body(race_number=2))
        r = client.get("/races")
        assert len(r.json()) == 2

    def test_list_exposes_post_time_and_restrictions(self, client):
        body = sample_race_body()
        body["post_time"] = "5:34 PM ET"
        body["age_restriction"] = "3U"
        body["sex_restriction"] = "F"
        client.post("/races", json=body)
        summaries = client.get("/races").json()
        assert summaries[0]["post_time"] == "5:34 PM ET"
        assert summaries[0]["age_restriction"] == "3U"
        assert summaries[0]["sex_restriction"] == "F"


class TestDeleteRace:
    def test_delete_race(self, client):
        client.post("/races", json=sample_race_body())
        r = client.delete("/races/CD-R1")
        assert r.status_code == 204
        assert client.get("/races/CD-R1").status_code == 404

    def test_delete_missing_race(self, client):
        r = client.delete("/races/NOPE-R1")
        assert r.status_code == 404

    def test_delete_all_races(self, client):
        client.post("/races", json=sample_race_body())
        client.post("/races", json=sample_race_body(race_number=2))
        r = client.delete("/races")
        assert r.status_code == 204
        assert client.get("/races").json() == []

    def test_delete_all_when_empty(self, client):
        r = client.delete("/races")
        assert r.status_code == 204


class TestUpdateOdds:
    def test_update_partial_odds(self, client):
        client.post("/races", json=sample_race_body())
        r = client.patch(
            "/races/CD-R1/odds",
            json={"odds": [{"post_position": 1, "live_odds": 3.5}]},
        )
        assert r.status_code == 200
        runners = {x["post_position"]: x for x in r.json()["runners"]}
        assert runners[1]["live_odds"] == 3.5
        # runner 2 still has morning line default
        assert runners[2]["live_odds"] == 7.0

    def test_update_all_odds(self, client):
        client.post("/races", json=sample_race_body())
        odds = [{"post_position": i, "live_odds": 2.0 + i} for i in range(1, 4)]
        r = client.patch("/races/CD-R1/odds", json={"odds": odds})
        assert r.status_code == 200
        runners = {x["post_position"]: x for x in r.json()["runners"]}
        for i in range(1, 4):
            assert runners[i]["live_odds"] == 2.0 + i

    def test_update_odds_bad_post(self, client):
        client.post("/races", json=sample_race_body())
        r = client.patch(
            "/races/CD-R1/odds",
            json={"odds": [{"post_position": 99, "live_odds": 5.0}]},
        )
        assert r.status_code == 422

    def test_update_odds_race_not_found(self, client):
        r = client.patch(
            "/races/INVALID-R1/odds",
            json={"odds": [{"post_position": 1, "live_odds": 5.0}]},
        )
        assert r.status_code == 404


class TestScratchRunner:
    def test_scratch_runner(self, client):
        client.post("/races", json=sample_race_body())
        r = client.patch("/races/CD-R1/runners/3/scratch")
        assert r.status_code == 200
        # runner is still in the list but marked as scratched
        runners = {x["post_position"]: x for x in r.json()["runners"]}
        assert len(runners) == 3
        assert runners[3]["scratched"] is True
        assert runners[1]["scratched"] is False

    def test_scratch_missing_runner(self, client):
        client.post("/races", json=sample_race_body())
        r = client.patch("/races/CD-R1/runners/99/scratch")
        assert r.status_code == 404

    def test_scratch_already_scratched(self, client):
        client.post("/races", json=sample_race_body())
        client.patch("/races/CD-R1/runners/3/scratch")
        r = client.patch("/races/CD-R1/runners/3/scratch")
        assert r.status_code == 422

    def test_scratch_to_below_minimum(self, client):
        client.post("/races", json=sample_race_body(num_runners=2))
        r = client.patch("/races/CD-R1/runners/1/scratch")
        assert r.status_code == 422
        # verify runner was NOT scratched
        race = client.get("/races/CD-R1").json()
        assert all(not x["scratched"] for x in race["runners"])

    def test_scratch_race_not_found(self, client):
        r = client.patch("/races/INVALID-R1/runners/1/scratch")
        assert r.status_code == 404


class TestUnscratchRunner:
    def test_unscratch_runner(self, client):
        client.post("/races", json=sample_race_body())
        client.patch("/races/CD-R1/runners/3/scratch")
        r = client.patch("/races/CD-R1/runners/3/unscratch")
        assert r.status_code == 200
        runners = {x["post_position"]: x for x in r.json()["runners"]}
        assert runners[3]["scratched"] is False

    def test_unscratch_not_scratched(self, client):
        client.post("/races", json=sample_race_body())
        r = client.patch("/races/CD-R1/runners/1/unscratch")
        assert r.status_code == 422

    def test_unscratch_missing_runner(self, client):
        client.post("/races", json=sample_race_body())
        r = client.patch("/races/CD-R1/runners/99/unscratch")
        assert r.status_code == 404


class TestFetchTwinSpiresOdds:
    def _patch_fetch(self, monkeypatch, odds_by_post=None, raises=None):
        from api import twinspires

        fetched_at = datetime(2026, 4, 23, 15, 30, 0).astimezone()

        def _fake(track, race_number, *, breed="Thoroughbred"):
            if raises is not None:
                raise raises
            return fetched_at, dict(odds_by_post or {})

        monkeypatch.setattr(twinspires, "fetch_twinspires_odds", _fake)

    def test_happy_path(self, client, monkeypatch):
        client.post("/races", json=sample_race_body())
        self._patch_fetch(monkeypatch, odds_by_post={1: 2.5, 2: 3.0, 3: 4.0})

        r = client.post("/races/CD-R1/fetch-twinspires-odds")
        assert r.status_code == 200
        data = r.json()
        runners = {x["post_position"]: x for x in data["race"]["runners"]}
        assert runners[1]["live_odds"] == 2.5
        assert runners[2]["live_odds"] == 3.0
        assert runners[3]["live_odds"] == 4.0
        assert sorted(data["applied_post_positions"]) == [1, 2, 3]
        assert data["missing_post_positions"] == []
        assert data["fetched_at"].startswith("2026-04-23")

    def test_partial_odds_leaves_existing_untouched(self, client, monkeypatch):
        client.post("/races", json=sample_race_body())
        # ML defaults: post 1->6.0, 2->7.0, 3->8.0 (per sample_race_body)
        self._patch_fetch(monkeypatch, odds_by_post={1: 2.5})

        r = client.post("/races/CD-R1/fetch-twinspires-odds")
        assert r.status_code == 200
        data = r.json()
        runners = {x["post_position"]: x for x in data["race"]["runners"]}
        assert runners[1]["live_odds"] == 2.5
        assert runners[2]["live_odds"] == 7.0  # morning-line default preserved
        assert runners[3]["live_odds"] == 8.0
        assert data["applied_post_positions"] == [1]
        assert sorted(data["missing_post_positions"]) == [2, 3]

    def test_all_null_odds(self, client, monkeypatch):
        client.post("/races", json=sample_race_body())
        self._patch_fetch(monkeypatch, odds_by_post={})

        r = client.post("/races/CD-R1/fetch-twinspires-odds")
        assert r.status_code == 200
        data = r.json()
        assert data["applied_post_positions"] == []
        assert sorted(data["missing_post_positions"]) == [1, 2, 3]
        runners = {x["post_position"]: x for x in data["race"]["runners"]}
        # morning-line defaults preserved
        assert runners[1]["live_odds"] == 6.0

    def test_scratched_runner_skipped(self, client, monkeypatch):
        client.post("/races", json=sample_race_body())
        client.patch("/races/CD-R1/runners/3/scratch")
        self._patch_fetch(monkeypatch, odds_by_post={1: 2.5, 2: 3.0, 3: 9.9})

        r = client.post("/races/CD-R1/fetch-twinspires-odds")
        assert r.status_code == 200
        data = r.json()
        runners = {x["post_position"]: x for x in data["race"]["runners"]}
        assert runners[3]["live_odds"] == 8.0  # scratched runner untouched
        assert runners[3]["scratched"] is True
        assert 3 not in data["applied_post_positions"]
        assert 3 not in data["missing_post_positions"]

    def test_network_error_surfaces_as_502(self, client, monkeypatch):
        from api.twinspires import TwinSpiresError

        client.post("/races", json=sample_race_body())
        self._patch_fetch(monkeypatch, raises=TwinSpiresError("boom"))

        r = client.post("/races/CD-R1/fetch-twinspires-odds")
        assert r.status_code == 502
        assert "TwinSpires unavailable" in r.json()["detail"]

    def test_race_not_found_at_twinspires_surfaces_as_404(self, client, monkeypatch):
        from api.twinspires import TwinSpiresRaceNotFound

        client.post("/races", json=sample_race_body())
        self._patch_fetch(
            monkeypatch, raises=TwinSpiresRaceNotFound("no such race at TS")
        )

        r = client.post("/races/CD-R1/fetch-twinspires-odds")
        assert r.status_code == 404
        assert "no such race at TS" in r.json()["detail"]

    def test_race_not_found(self, client):
        r = client.post("/races/NOPE-R1/fetch-twinspires-odds")
        assert r.status_code == 404

    def test_updates_persist_across_get(self, client, monkeypatch):
        client.post("/races", json=sample_race_body())
        self._patch_fetch(monkeypatch, odds_by_post={1: 2.5, 2: 3.0})

        client.post("/races/CD-R1/fetch-twinspires-odds")

        race = client.get("/races/CD-R1").json()
        runners = {x["post_position"]: x for x in race["runners"]}
        assert runners[1]["live_odds"] == 2.5
        assert runners[2]["live_odds"] == 3.0


class TestPredictStoredRace:
    def _create_race_with_odds(self, client):
        client.post("/races", json=sample_race_body())
        odds = [{"post_position": i, "live_odds": 2.0 + i} for i in range(1, 4)]
        client.patch("/races/CD-R1/odds", json={"odds": odds})

    def test_predict_with_odds(self, client):
        self._create_race_with_odds(client)
        r = client.post("/races/CD-R1/predict")
        assert r.status_code == 200
        data = r.json()
        assert data["race_id"] == "CD-R1"
        assert len(data["predictions"]) == 3

        # each prediction should have the expected fields
        pred = data["predictions"][0]
        assert "model_prob" in pred
        assert "market_prob" in pred
        assert "edge" in pred
        assert "ev_per_dollar" in pred

    def test_predict_with_morning_line_defaults(self, client):
        """Predicting works immediately using morning line as default odds."""
        client.post("/races", json=sample_race_body())
        r = client.post("/races/CD-R1/predict")
        assert r.status_code == 200
        assert len(r.json()["predictions"]) == 3

    def test_predict_race_not_found(self, client):
        r = client.post("/races/NOPE-R1/predict")
        assert r.status_code == 404

    def test_predict_after_scratch(self, client):
        """Predict excludes scratched runners."""
        self._create_race_with_odds(client)
        client.patch("/races/CD-R1/runners/3/scratch")
        r = client.post("/races/CD-R1/predict")
        assert r.status_code == 200
        preds = r.json()["predictions"]
        assert len(preds) == 2
        assert all(p["post_position"] != 3 for p in preds)
