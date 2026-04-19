"""
Calibration diagnostics for the XGBRanker + softmax pipeline.

Helpers for quantifying probability quality on a split: per-race entropy, reliability
binning, odds-bucket and field-size stratification, and a temperature-scaling probe.
Reuses the enrichment logic from ``model.evaluate`` so numbers match what
``python -m model.evaluate`` prints.
"""

import math

import numpy as np
import plotly.graph_objects as go
import polars as pl

from model.betting import add_ev_columns, apply_bet_rule, summarize_roi
from model.evaluate import _log_loss_winner, _market_probs, _per_race_softmax

EPS = 1e-12

# odds buckets: <2, 2-5, 5-10, 10-20, 20+
ODDS_EDGES = [0.0, 2.0, 5.0, 10.0, 20.0, math.inf]
ODDS_LABELS = ["<2", "2-5", "5-10", "10-20", "20+"]

# field-size buckets: <=6, 7-8, 9-10, 11+
FIELD_EDGES = [0, 6, 8, 10, math.inf]
FIELD_LABELS = ["<=6", "7-8", "9-10", "11+"]

# reliability bins: dense at the low end (where the problem lives)
DEFAULT_RELIABILITY_EDGES = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0]


def enrich(
    df: pl.DataFrame,
    model,
    features: list[str],
    temperature: float = 1.0,
) -> pl.DataFrame:
    """Attach model_score, model_prob, market_prob, decimal_odds, ev_per_dollar.

    Drops races with no recorded winner so per-race win metrics are well defined.
    """
    X = df.select(features).to_numpy()
    scores = model.predict(X)
    df = df.with_columns(pl.Series("model_score", scores))
    df = _per_race_softmax(df, "model_score", "model_prob", temperature)
    df = _market_probs(df)
    df = df.filter(pl.col("won").max().over("race_id") == 1)
    return add_ev_columns(df)


# ---------- distributional comparisons ----------


def per_race_entropy(df: pl.DataFrame, prob_col: str) -> pl.DataFrame:
    """Shannon entropy (nats) of prob_col within each race."""
    clipped = pl.col(prob_col).clip(EPS, 1.0)
    return (
        df.with_columns((-pl.col(prob_col) * clipped.log()).alias("_e"))
        .group_by("race_id", maintain_order=True)
        .agg(pl.col("_e").sum().alias("entropy"))
    )


def per_race_favorite_prob(df: pl.DataFrame, prob_col: str) -> pl.Series:
    """Max prob_col within each race (probability assigned to the favorite)."""
    return df.group_by("race_id", maintain_order=True).agg(
        pl.col(prob_col).max().alias("_fav")
    )["_fav"]


def entropy_summary(df: pl.DataFrame) -> pl.DataFrame:
    """One-row summary comparing model vs market per-race entropy."""
    m = per_race_entropy(df, "model_prob")["entropy"]
    k = per_race_entropy(df, "market_prob")["entropy"]
    return pl.DataFrame({
        "model_entropy_mean": [float(m.mean())],
        "market_entropy_mean": [float(k.mean())],
        "delta_mean": [float((m - k).mean())],
        "n_races": [int(len(m))],
    })


# ---------- reliability ----------


def _bin_labels_from_edges(edges: list[float]) -> list[str]:
    return [f"[{a*100:.0f}%, {b*100:.0f}%)" for a, b in zip(edges[:-1], edges[1:])]


def reliability_table(
    df: pl.DataFrame,
    prob_col: str = "model_prob",
    edges: list[float] = DEFAULT_RELIABILITY_EDGES,
) -> pl.DataFrame:
    """Reliability binning: mean predicted prob, empirical win rate, count, Wilson CI."""
    labels = _bin_labels_from_edges(edges)
    binned = df.with_columns(
        pl.col(prob_col).cut(edges[1:-1], labels=labels).alias("bin")
    )
    out = (
        binned.group_by("bin", maintain_order=False)
        .agg(
            pl.col(prob_col).mean().alias("mean_pred"),
            pl.col("won").mean().alias("empirical_rate"),
            pl.len().alias("count"),
            pl.col("won").sum().alias("wins"),
        )
        .sort("mean_pred")
    )
    # Wilson 95% CI
    z = 1.96
    n = out["count"].to_numpy().astype(float)
    w = out["wins"].to_numpy().astype(float)
    center = (w + z * z / 2) / (n + z * z)
    half = (z / (n + z * z)) * np.sqrt(w * (n - w) / n + z * z / 4)
    return out.with_columns(
        pl.Series("ci_lo", center - half),
        pl.Series("ci_hi", center + half),
    )


def brier_score(df: pl.DataFrame, prob_col: str) -> float:
    """Per-row Brier score vs won."""
    p = df[prob_col].to_numpy()
    y = df["won"].to_numpy()
    return float(np.mean((p - y) ** 2))


# ---------- odds bucket ----------


def odds_bucket_table(df: pl.DataFrame) -> pl.DataFrame:
    """Per-bucket stats over dollar_odds: means, win rate, EV, bets under EV>0 rule."""
    binned = df.with_columns(
        pl.col("dollar_odds")
        .cut(ODDS_EDGES[1:-1], labels=ODDS_LABELS)
        .alias("odds_bucket")
    )
    base = binned.group_by("odds_bucket", maintain_order=False).agg(
        pl.len().alias("n"),
        pl.col("model_prob").mean().alias("mean_model_prob"),
        pl.col("market_prob").mean().alias("mean_market_prob"),
        pl.col("won").mean().alias("win_rate"),
        pl.col("ev_per_dollar").mean().alias("mean_ev"),
        (pl.col("ev_per_dollar") > 0).sum().alias("n_positive_ev"),
    )
    # ROI among positive-EV horses in each bucket (flat $2)
    pos = binned.filter(pl.col("ev_per_dollar") > 0)
    if pos.is_empty():
        roi_tbl = pl.DataFrame({
            "odds_bucket": ODDS_LABELS,
            "pos_ev_hit_rate": [0.0] * len(ODDS_LABELS),
            "pos_ev_roi": [0.0] * len(ODDS_LABELS),
        })
    else:
        pos = pos.with_columns(
            pl.lit(2.0).alias("stake"),
            (pl.col("won") * 2.0 * pl.col("decimal_odds")).alias("payout"),
        )
        roi_tbl = pos.group_by("odds_bucket", maintain_order=False).agg(
            pl.col("won").mean().alias("pos_ev_hit_rate"),
            (
                (pl.col("payout").sum() - pl.col("stake").sum()) / pl.col("stake").sum()
            ).alias("pos_ev_roi"),
        )
    table = base.join(roi_tbl, on="odds_bucket", how="left").sort(
        by=pl.col("odds_bucket").cast(pl.Enum(ODDS_LABELS))
    )
    return table


# ---------- field-size bucket ----------


def field_size_bucket_table(df: pl.DataFrame) -> pl.DataFrame:
    """Per-bucket stats over field_size: entropy gap, favorite-prob gap, log-loss gap."""
    model_ent = per_race_entropy(df, "model_prob").rename({"entropy": "model_ent"})
    market_ent = per_race_entropy(df, "market_prob").rename({"entropy": "market_ent"})
    # one row per race with its field_size
    race_fs = df.group_by("race_id", maintain_order=True).agg(
        pl.col("field_size").first()
    )
    per_race = race_fs.join(model_ent, on="race_id").join(market_ent, on="race_id")
    per_race = per_race.with_columns(
        pl.col("field_size")
        .cut(FIELD_EDGES[1:-1], labels=FIELD_LABELS)
        .alias("field_bucket")
    )

    # favorite-prob gap per race
    fav_m = df.group_by("race_id", maintain_order=True).agg(
        pl.col("model_prob").max().alias("fav_model"),
        pl.col("market_prob").max().alias("fav_market"),
    )
    per_race = per_race.join(fav_m, on="race_id")

    # log-loss per race (one winner per race)
    winners = df.filter(pl.col("won") == 1).select(
        "race_id",
        (-pl.col("model_prob").clip(EPS, 1.0).log()).alias("ll_model"),
        (-pl.col("market_prob").clip(EPS, 1.0).log()).alias("ll_market"),
    )
    per_race = per_race.join(winners, on="race_id", how="left")

    summary = (
        per_race.group_by("field_bucket", maintain_order=False)
        .agg(
            pl.len().alias("n_races"),
            (pl.col("model_ent") - pl.col("market_ent")).mean().alias("entropy_gap"),
            (pl.col("fav_model") - pl.col("fav_market")).mean().alias("favorite_gap"),
            (pl.col("ll_model") - pl.col("ll_market")).mean().alias("log_loss_gap"),
        )
        .sort(by=pl.col("field_bucket").cast(pl.Enum(FIELD_LABELS)))
    )
    return summary


# ---------- temperature probe ----------


def temperature_sweep(
    df: pl.DataFrame,
    temps: np.ndarray,
    score_col: str = "model_score",
) -> pl.DataFrame:
    """Log-loss on winners at each temperature T (softmax(score / T) per race)."""
    rows = []
    for T in temps:
        tmp = df.with_columns((pl.col(score_col) / float(T)).alias("_s"))
        tmp = _per_race_softmax(tmp, "_s", "_p")
        rows.append({"T": float(T), "log_loss": _log_loss_winner(tmp, "_p")})
    return pl.DataFrame(rows)


# ---------- plotting ----------


def plot_reliability(
    table_model: pl.DataFrame,
    table_market: pl.DataFrame | None = None,
    title: str = "Reliability diagram",
) -> go.Figure:
    """Mean predicted vs empirical, with y=x reference and optional market overlay."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line={"dash": "dash", "color": "gray"},
            name="perfect",
            showlegend=True,
        )
    )
    _add_reliability_trace(fig, table_model, name="model", color="#636EFA")
    if table_market is not None:
        _add_reliability_trace(fig, table_market, name="market", color="#EF553B")
    fig.update_layout(
        title=title,
        xaxis_title="mean predicted prob",
        yaxis_title="empirical win rate",
        xaxis={"range": [0, 0.6]},
        yaxis={"range": [0, 0.6]},
        width=620,
        height=560,
    )
    return fig


def _add_reliability_trace(fig: go.Figure, t: pl.DataFrame, name: str, color: str):
    x = t["mean_pred"].to_numpy()
    y = t["empirical_rate"].to_numpy()
    lo = t["ci_lo"].to_numpy()
    hi = t["ci_hi"].to_numpy()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers+lines",
            name=name,
            marker={"color": color, "size": 9},
            error_y={
                "type": "data",
                "symmetric": False,
                "array": hi - y,
                "arrayminus": y - lo,
            },
            hovertext=[
                f"bin={b}<br>n={n}<br>mean_pred={mp:.3f}<br>emp={e:.3f}"
                for b, n, mp, e in zip(
                    t["bin"].to_list(),
                    t["count"].to_list(),
                    t["mean_pred"].to_list(),
                    t["empirical_rate"].to_list(),
                )
            ],
        )
    )


def plot_entropy_hist(df: pl.DataFrame) -> go.Figure:
    """Overlay per-race entropy for model vs market."""
    m = per_race_entropy(df, "model_prob")["entropy"].to_numpy()
    k = per_race_entropy(df, "market_prob")["entropy"].to_numpy()
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=m, name="model", opacity=0.6, nbinsx=40))
    fig.add_trace(go.Histogram(x=k, name="market", opacity=0.6, nbinsx=40))
    fig.update_layout(
        barmode="overlay",
        title=f"Per-race entropy (model mean={m.mean():.3f}, market mean={k.mean():.3f})",
        xaxis_title="entropy (nats)",
        yaxis_title="races",
        width=720,
        height=420,
    )
    return fig


def plot_favorite_prob_hist(df: pl.DataFrame) -> go.Figure:
    """Overlay histogram of per-race favorite probability (max prob in race)."""
    m = per_race_favorite_prob(df, "model_prob").to_numpy()
    k = per_race_favorite_prob(df, "market_prob").to_numpy()
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=m, name="model", opacity=0.6, nbinsx=40))
    fig.add_trace(go.Histogram(x=k, name="market", opacity=0.6, nbinsx=40))
    fig.update_layout(
        barmode="overlay",
        title=f"Per-race favorite probability (model mean={m.mean():.3f}, market mean={k.mean():.3f})",
        xaxis_title="max prob in race",
        yaxis_title="races",
        width=720,
        height=420,
    )
    return fig


def plot_prob_scatter(
    df: pl.DataFrame, sample: int = 10000, seed: int = 0
) -> go.Figure:
    """Scatter model_prob vs market_prob, colored by won."""
    rng = np.random.default_rng(seed)
    n = min(sample, df.shape[0])
    idx = rng.choice(df.shape[0], size=n, replace=False)
    sub = df[idx]
    fig = go.Figure()
    for label, color in [(0, "rgba(99,110,250,0.35)"), (1, "rgba(239,85,59,0.9)")]:
        s = sub.filter(pl.col("won") == label)
        fig.add_trace(
            go.Scatter(
                x=s["market_prob"].to_numpy(),
                y=s["model_prob"].to_numpy(),
                mode="markers",
                name=f"won={label}",
                marker={"color": color, "size": 5},
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line={"dash": "dash", "color": "gray"},
            name="y=x",
            showlegend=True,
        )
    )
    lim = max(sub["market_prob"].max(), sub["model_prob"].max()) * 1.05
    fig.update_layout(
        title=f"model_prob vs market_prob (n={n:,})",
        xaxis_title="market_prob",
        yaxis_title="model_prob",
        xaxis={"range": [0, lim]},
        yaxis={"range": [0, lim]},
        width=620,
        height=560,
    )
    return fig


def plot_temperature_sweep(
    table: pl.DataFrame, baseline: float | None = None
) -> go.Figure:
    """Log-loss vs temperature, with the minimum highlighted."""
    T = table["T"].to_numpy()
    ll = table["log_loss"].to_numpy()
    i_min = int(np.argmin(ll))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=T, y=ll, mode="lines+markers", name="model softmax(s/T)")
    )
    fig.add_trace(
        go.Scatter(
            x=[T[i_min]],
            y=[ll[i_min]],
            mode="markers",
            marker={"size": 14, "color": "red", "symbol": "star"},
            name=f"min @ T={T[i_min]:.3f} ({ll[i_min]:.4f})",
        )
    )
    if baseline is not None:
        fig.add_hline(
            y=baseline, line_dash="dash", annotation_text=f"market ({baseline:.4f})"
        )
    fig.update_layout(
        title="Temperature sweep (validation log-loss)",
        xaxis_title="temperature T",
        yaxis_title="log-loss on winner",
        width=720,
        height=440,
    )
    return fig


def summarize_bet_rule(df: pl.DataFrame, rule: str, **kwargs) -> dict:
    """Convenience wrapper around apply_bet_rule + summarize_roi."""
    bets = apply_bet_rule(df, rule=rule, **kwargs)
    return summarize_roi(bets)
