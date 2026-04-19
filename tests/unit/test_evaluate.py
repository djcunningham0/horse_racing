import numpy as np
import polars as pl
import pytest

from model.evaluate import _per_race_softmax


def test_per_race_softmax_known_values():
    df = pl.DataFrame({"race_id": [1, 1, 1], "score": [1.0, 2.0, 3.0]})
    out = _per_race_softmax(df, "score", "prob")
    expected = np.exp([1, 2, 3]) / np.exp([1, 2, 3]).sum()
    np.testing.assert_allclose(out["prob"].to_numpy(), expected)


def test_per_race_softmax_uniform_input():
    df = pl.DataFrame({"race_id": [1, 1, 1], "score": [5.0, 5.0, 5.0]})
    out = _per_race_softmax(df, "score", "prob")
    np.testing.assert_allclose(out["prob"].to_numpy(), [1 / 3, 1 / 3, 1 / 3])


def test_per_race_softmax_numerical_stability():
    df = pl.DataFrame({"race_id": [1, 1, 1], "score": [1000.0, 1001.0, 1002.0]})
    out = _per_race_softmax(df, "score", "prob")
    probs = out["prob"].to_numpy()
    assert np.all(np.isfinite(probs))
    assert pytest.approx(probs.sum()) == 1.0


def test_per_race_softmax_normalizes_per_race():
    df = pl.DataFrame({
        "race_id": [1, 1, 2, 2, 2],
        "score": [1.0, 2.0, 0.0, 0.0, 0.0],
    })
    out = _per_race_softmax(df, "score", "prob")
    sums = out.group_by("race_id").agg(pl.col("prob").sum())["prob"].to_numpy()
    np.testing.assert_allclose(sums, [1.0, 1.0])
    race2 = out.filter(pl.col("race_id") == 2)["prob"].to_numpy()
    np.testing.assert_allclose(race2, [1 / 3, 1 / 3, 1 / 3])


def test_per_race_softmax_preserves_row_order():
    df = pl.DataFrame({
        "race_id": [2, 1, 2, 1],
        "score": [0.5, 1.0, 1.5, 2.0],
    })
    out = _per_race_softmax(df, "score", "prob")
    assert out["race_id"].to_list() == [2, 1, 2, 1]


def test_per_race_softmax_temperature_sharpens():
    df = pl.DataFrame({"race_id": [1, 1, 1], "score": [1.0, 2.0, 3.0]})
    cold = _per_race_softmax(df, "score", "prob", temperature=0.5)["prob"].to_numpy()
    hot = _per_race_softmax(df, "score", "prob", temperature=2.0)["prob"].to_numpy()
    assert cold.max() > hot.max()  # lower T concentrates mass on top score
    expected_cold = np.exp(np.array([1, 2, 3]) / 0.5)
    expected_cold = expected_cold / expected_cold.sum()
    np.testing.assert_allclose(cold, expected_cold)


def test_per_race_softmax_temperature_one_matches_default():
    df = pl.DataFrame({"race_id": [1, 1, 1], "score": [1.0, 2.0, 3.0]})
    default = _per_race_softmax(df, "score", "prob")["prob"].to_numpy()
    explicit = _per_race_softmax(df, "score", "prob", temperature=1.0)["prob"].to_numpy()
    np.testing.assert_allclose(default, explicit)
