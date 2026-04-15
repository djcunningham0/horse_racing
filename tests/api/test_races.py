"""Tests for race pre-loading endpoints."""

from tests.api.conftest import sample_race_body


class TestCreateRace:
    def test_create_race(self, client):
        r = client.post("/races", json=sample_race_body())
        assert r.status_code == 201
        assert r.json()["race_id"] == "CD-R1"

    def test_tote_odds_default_to_morning_line(self, client):
        _ = client.post("/races", json=sample_race_body())
        r = client.get("/races/CD-R1")
        race = r.json()
        for runner in race["runners"]:
            # tote_odds should be morning_line_decimal - 1
            expected = runner["morning_line_decimal"] - 1
            assert runner["tote_odds"] == expected

    def test_duplicate_race(self, client):
        client.post("/races", json=sample_race_body())
        r = client.post("/races", json=sample_race_body())
        assert r.status_code == 409


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


class TestUpdateOdds:
    def test_update_partial_odds(self, client):
        client.post("/races", json=sample_race_body())
        r = client.patch(
            "/races/CD-R1/odds",
            json={"odds": [{"post_position": 1, "tote_odds": 3.5}]},
        )
        assert r.status_code == 200
        runners = {x["post_position"]: x for x in r.json()["runners"]}
        assert runners[1]["tote_odds"] == 3.5
        # runner 2 still has morning line default (morning_line_decimal - 1 = 6.0)
        assert runners[2]["tote_odds"] == 6.0

    def test_update_all_odds(self, client):
        client.post("/races", json=sample_race_body())
        odds = [{"post_position": i, "tote_odds": 2.0 + i} for i in range(1, 4)]
        r = client.patch("/races/CD-R1/odds", json={"odds": odds})
        assert r.status_code == 200
        runners = {x["post_position"]: x for x in r.json()["runners"]}
        for i in range(1, 4):
            assert runners[i]["tote_odds"] == 2.0 + i

    def test_update_odds_bad_post(self, client):
        client.post("/races", json=sample_race_body())
        r = client.patch(
            "/races/CD-R1/odds",
            json={"odds": [{"post_position": 99, "tote_odds": 5.0}]},
        )
        assert r.status_code == 422

    def test_update_odds_race_not_found(self, client):
        r = client.patch(
            "/races/INVALID-R1/odds",
            json={"odds": [{"post_position": 1, "tote_odds": 5.0}]},
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


class TestPredictStoredRace:
    def _create_race_with_odds(self, client):
        client.post("/races", json=sample_race_body())
        odds = [{"post_position": i, "tote_odds": 2.0 + i} for i in range(1, 4)]
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
