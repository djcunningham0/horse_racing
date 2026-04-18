import numpy as np
import pytest

from model.evaluate import _softmax


def test_softmax_known_values():
    result = _softmax(np.array([1.0, 2.0, 3.0]))
    expected = np.exp([1, 2, 3]) / np.exp([1, 2, 3]).sum()
    np.testing.assert_allclose(result, expected)
    assert pytest.approx(result.sum()) == 1.0


def test_softmax_uniform_input():
    result = _softmax(np.array([5.0, 5.0, 5.0]))
    np.testing.assert_allclose(result, [1 / 3, 1 / 3, 1 / 3])


def test_softmax_numerical_stability():
    # large values should not overflow to inf/nan
    result = _softmax(np.array([1000.0, 1001.0, 1002.0]))
    assert np.all(np.isfinite(result))
    assert pytest.approx(result.sum()) == 1.0
