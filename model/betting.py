"""Betting logic: EV calculation, bet rules, and ROI summaries.

Operates on DataFrames that already have ``model_prob`` and ``dollar_odds``
columns (i.e., the enriched frames produced by :mod:`model.evaluate`).
"""

import polars as pl

# Horses with zero/null odds can't be bet on meaningfully.
_VALID_ODDS = pl.col("dollar_odds") > 0

BET_RULES = (
    "ev_threshold",
    "top_ev_per_race",
    "top_model_per_race",
    "favorite",
    "ml_favorite",
    "top_speed_fig",
)


def add_ev_columns(
    df: pl.DataFrame,
    dollar_odds_col: str = "dollar_odds",
    prob_col: str = "model_prob",
    out_decimal_odds_col: str = "decimal_odds",
    out_ev_col: str = "ev_per_dollar",
) -> pl.DataFrame:
    """Add ``decimal_odds`` and ``ev_per_dollar`` columns.

    ``ev_per_dollar = model_prob * decimal_odds - 1``.  A positive value means
    the model thinks the horse is underpriced by the market.
    """
    return df.with_columns(
        (pl.col(dollar_odds_col) + 1).alias(out_decimal_odds_col),
        (pl.col(prob_col) * (pl.col(dollar_odds_col) + 1) - 1).alias(out_ev_col),
    )


def apply_bet_rule(
    df: pl.DataFrame,
    rule: str = "ev_threshold",
    ev_threshold: float = 0.0,
    stake: float = 2.0,
) -> pl.DataFrame:
    """Return rows where a bet is placed, with ``stake`` and ``payout`` columns.

    Rules
    -----
    ev_threshold
        Bet on every horse whose ``ev_per_dollar > ev_threshold``.
    top_ev_per_race
        Bet on the single highest-EV horse per race (if its EV > threshold).
    top_model_per_race
        Bet on each race's top-ranked horse by ``model_prob``.
    favorite
        Bet on the market favorite (highest ``market_prob``) every race.
    ml_favorite
        Bet on the morning-line favorite (lowest ``morning_line_odds_float``).
    """
    eligible = df.filter(_VALID_ODDS)

    if rule == "ev_threshold":
        bets = eligible.filter(pl.col("ev_per_dollar") > ev_threshold)
    elif rule == "top_ev_per_race":
        bets = (
            eligible.sort("ev_per_dollar", descending=True)
            .group_by("race_id", maintain_order=True)
            .first()
            .filter(pl.col("ev_per_dollar") > ev_threshold)
        )
    elif rule == "top_model_per_race":
        bets = (
            eligible.sort("model_prob", descending=True)
            .group_by("race_id", maintain_order=True)
            .first()
        )
    elif rule == "favorite":
        bets = (
            eligible.sort("market_prob", descending=True)
            .group_by("race_id", maintain_order=True)
            .first()
        )
    elif rule == "ml_favorite":
        bets = (
            eligible.filter(pl.col("morning_line_odds_float").is_not_null())
            .sort("morning_line_odds_float")
            .group_by("race_id", maintain_order=True)
            .first()
        )
    elif rule == "top_speed_fig":
        bets = (
            eligible.filter(pl.col("speed_fig_L1").is_not_null())
            .sort("speed_fig_L1", descending=True)
            .group_by("race_id", maintain_order=True)
            .first()
        )
    else:
        raise ValueError(f"Unknown rule: {rule!r}. Choose from {BET_RULES}")

    return bets.with_columns(
        pl.lit(stake).alias("stake"),
        (pl.col("won") * stake * pl.col("decimal_odds")).alias("payout"),
    )


def summarize_roi(bets: pl.DataFrame) -> dict:
    """Aggregate bet-level rows into a single ROI summary."""
    if bets.is_empty():
        return {
            "n_bets": 0,
            "total_staked": 0.0,
            "total_return": 0.0,
            "profit": 0.0,
            "roi": 0.0,
            "hit_rate": 0.0,
            "avg_ev": 0.0,
        }
    total_staked = bets["stake"].sum()
    total_return = bets["payout"].sum()
    profit = total_return - total_staked
    n_bets = len(bets)
    return {
        "n_bets": int(n_bets),
        "total_staked": float(total_staked),
        "total_return": float(total_return),
        "profit": float(profit),
        "roi": float(profit / total_staked) if total_staked else 0.0,
        "hit_rate": float(bets["won"].mean()),
        "avg_ev": float(bets["ev_per_dollar"].mean()),
    }
