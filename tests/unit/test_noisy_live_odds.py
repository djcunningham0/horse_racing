import numpy as np
import pytest

from model.features import NoisyOddsConfig, _noisy_live_odds_numpy


@pytest.fixture
def small_race():
    """Two races with distinct ML and final odds."""
    final = np.array([2.0, 3.0, 5.0, 9.0, 1.5, 8.0])
    ml = np.array([3.0, 2.5, 4.0, 12.0, 2.0, 6.0])
    race_codes = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    return final, ml, race_codes


def _shares_from_odds(odds: np.ndarray, race_codes: np.ndarray) -> np.ndarray:
    """Implied probs normalized per race (recovers pool shares)."""
    p = 1.0 / (odds + 1.0)
    sums = np.bincount(race_codes, weights=p)
    return p / sums[race_codes]


def test_alpha_extremes_recover_endpoints(small_race):
    final, ml, race_codes = small_race

    # alpha -> 0: noisy shares should track ML
    cfg_ml = NoisyOddsConfig(beta_a=1e-6, beta_b=1e6, kappa=1e9)
    odds_ml = _noisy_live_odds_numpy(final, ml, race_codes, cfg_ml, seed=0)
    assert np.allclose(
        _shares_from_odds(odds_ml, race_codes),
        _shares_from_odds(ml, race_codes),
        atol=1e-3,
    )

    # alpha -> 1: noisy shares should track final
    cfg_fin = NoisyOddsConfig(beta_a=1e6, beta_b=1e-6, kappa=1e9)
    odds_fin = _noisy_live_odds_numpy(final, ml, race_codes, cfg_fin, seed=0)
    assert np.allclose(
        _shares_from_odds(odds_fin, race_codes),
        _shares_from_odds(final, race_codes),
        atol=1e-3,
    )


def test_takeout_preserved(small_race):
    final, ml, race_codes = small_race

    odds = _noisy_live_odds_numpy(final, ml, race_codes, NoisyOddsConfig(), seed=0)
    implied_sum = np.bincount(race_codes, weights=1.0 / (odds + 1.0))
    final_sum = np.bincount(race_codes, weights=1.0 / (final + 1.0))
    assert np.allclose(implied_sum, final_sum, atol=1e-6)


def test_max_odds_clamp():
    # longshot with implied share well below the floor should never exceed max_odds
    final = np.array([2.0, 4.0, 8.0, 200.0])
    ml = np.array([2.5, 3.5, 10.0, 150.0])
    race_codes = np.zeros(4, dtype=np.int64)
    cfg = NoisyOddsConfig(max_odds=30.0, kappa=50.0)
    for seed in range(20):
        odds = _noisy_live_odds_numpy(final, ml, race_codes, cfg, seed=seed)
        assert odds.max() <= 30.0 + 1e-9


def test_seed_reproducible(small_race):
    final, ml, race_codes = small_race

    cfg = NoisyOddsConfig()
    odds_a = _noisy_live_odds_numpy(final, ml, race_codes, cfg, seed=42)
    odds_b = _noisy_live_odds_numpy(final, ml, race_codes, cfg, seed=42)
    odds_c = _noisy_live_odds_numpy(final, ml, race_codes, cfg, seed=43)

    assert np.array_equal(odds_a, odds_b)
    assert not np.array_equal(odds_a, odds_c)
