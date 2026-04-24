"""Train/val/test partitioning by race_id."""

import numpy as np
import polars as pl
from numpy.typing import NDArray

DEFAULT_VAL_FRAC = 0.15
DEFAULT_TEST_FRAC = 0.15
DEFAULT_RANDOM_SEED = 0
DEFAULT_SPLIT_MODE = "chronological"
SPLIT_MODES = ("random", "chronological")


def split_by_race(
    df: pl.DataFrame,
    val_frac: float = DEFAULT_VAL_FRAC,
    test_frac: float = DEFAULT_TEST_FRAC,
    seed: int = DEFAULT_RANDOM_SEED,
    mode: str = DEFAULT_SPLIT_MODE,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Split by race_id into train/val/test. All rows of a given race stay together, and
    the splits are deterministic (if you use the same dataframe and seed).

    Parameters
    ----------
    mode
        - "chronological": sort by race_date, with the earliest races as train and the
          latest as test;`seed` is ignored.
        - "random": shuffle race_ids with `seed`. Safe if none of the features are keyed
          on horse, jockey, or trainer identity, so no (or minimal) cross-split leakage.
    """
    train_ids, val_ids, test_ids = get_race_id_splits(
        df=df,
        val_frac=val_frac,
        test_frac=test_frac,
        seed=seed,
        mode=mode,
    )
    return (
        df.filter(pl.col("race_id").is_in(train_ids)),
        df.filter(pl.col("race_id").is_in(val_ids)),
        df.filter(pl.col("race_id").is_in(test_ids)),
    )


def get_race_id_splits(
    df: pl.DataFrame,
    val_frac: float = DEFAULT_VAL_FRAC,
    test_frac: float = DEFAULT_TEST_FRAC,
    seed: int = DEFAULT_RANDOM_SEED,
    mode: str = DEFAULT_SPLIT_MODE,
) -> tuple[NDArray[np.str_], NDArray[np.str_], NDArray[np.str_]]:
    """
    Return the (train_ids, val_ids, test_ids) to split races in to train, validation,
    and test set. See `split_by_race` for `mode` semantics.
    """
    if mode not in SPLIT_MODES:
        raise ValueError(f"unknown split mode: {mode!r}; expected one of {SPLIT_MODES}")

    n_unique = df["race_id"].n_unique()
    n_test = int(n_unique * test_frac)
    n_val = int(n_unique * val_frac)

    if mode == "random":
        race_ids = np.sort(df["race_id"].unique().to_numpy())
        rng = np.random.default_rng(seed)
        rng.shuffle(race_ids)
        test_ids = race_ids[:n_test]
        val_ids = race_ids[n_test : n_test + n_val]
        train_ids = race_ids[n_test + n_val :]
    else:  # chronological
        race_ids = (
            df.group_by("race_id")
            .agg(pl.col("race_date").min())
            .sort("race_date", "race_id")["race_id"]
            .to_numpy()
        )
        n_train = n_unique - n_val - n_test
        train_ids = race_ids[:n_train]
        val_ids = race_ids[n_train : n_train + n_val]
        test_ids = race_ids[n_train + n_val :]

    return train_ids, val_ids, test_ids
