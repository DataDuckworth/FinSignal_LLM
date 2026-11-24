"""Metrics: pooled and per-day IC, direction accuracy, long-short quintile backtest, day-cluster bootstrap CIs."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr


def pooled_ic(df, score="score", target="abn_ret"):
    return spearmanr(df[score], df[target]).correlation


def direction_acc(df, score="score"):
    d = df[df[score] != 0]
    return (np.sign(d[score]) == np.where(d["abn_ret"] > 0, 1, -1)).mean()


def ticker_day_signal(df, score):
    """One row per (ticker, entry day): the mean score of that day's headlines for the ticker.

    Why: several headlines on the same ticker-day share one target (the same abnormal return). If each headline
    were ranked separately, a busy ticker would count several times and get extra weight in the portfolio.
    """
    return (df.groupby(["ticker", "entry_date"], as_index=False)
              .agg(signal=(score, "mean"), abn_ret=("abn_ret", "mean"), n_headlines=(score, "size")))


def long_short_daily(td, q=5, min_names=10, min_leg=2):
    """Daily long-top / short-bottom quantile portfolio on a ticker-day table (columns entry_date, signal, abn_ret).

    Per day: rank the names by signal, long the top 1/q, short the bottom 1/q, equal weights, and return the
    GROSS spread  mean(AR of longs) - mean(AR of shorts). Names in the middle, including a block of exactly tied
    scores (e.g. many lexicon zeros), are not traded. Days with fewer than min_names names, or fewer than
    min_leg names in either leg, are skipped.
    """
    rows = []
    for day, g in td.groupby("entry_date"):
        n = len(g)
        if n < min_names:
            continue
        pct = g["signal"].rank(method="average", pct=True)        # ties share an average rank
        longs = g.loc[pct > 1 - 1 / q + 1e-9, "abn_ret"]
        shorts = g.loc[pct <= 1 / q + 1e-9, "abn_ret"]
        if len(longs) < min_leg or len(shorts) < min_leg:
            continue
        rows.append((day, longs.mean() - shorts.mean(), len(longs), len(shorts), n))
    return pd.DataFrame(rows, columns=["entry_date", "gross", "n_long", "n_short", "n_names"]).set_index("entry_date")


def net_of_costs(gross, cost_bps=10, legs=2, trades_per_name=2):
    """Subtract trading costs from gross daily spreads (return per $1 long and $1 short).

    Worst case: the whole book turns over every day. Each name is bought and sold once (entry close, next close),
    that is `trades_per_name` trades per leg, and there are `legs` legs: 2 x 2 x cost_bps by default. Borrow
    fees, market impact and the bid-ask spread beyond cost_bps are NOT modelled.
    """
    return gross - cost_bps / 1e4 * legs * trades_per_name


def performance(daily, n_boot=2000, seed=0):
    """Descriptive statistics of a daily return series, with a day-bootstrap CI for the mean."""
    daily = pd.Series(daily).dropna()
    lo, hi = day_bootstrap_ci(daily.values, None, n_boot, seed)
    cum = daily.cumsum()                                         # arithmetic: fine for returns this small
    return {
        "days": len(daily), "mean": daily.mean(), "ci_lo": lo, "ci_hi": hi,
        "t": daily.mean() / (daily.std() / np.sqrt(len(daily))),
        "sharpe": daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else np.nan,
        "hit_rate": (daily > 0).mean(), "max_drawdown": (cum.cummax() - cum).max(),
    }


def daily_ic(df, score="score", target="abn_ret", min_names=5):
    """Spearman IC computed inside each entry day (the cross-section), one value per day.

    Why per day: stocks move together on a given day (market and sector noise), so rows from the same
    day are not independent. Ranking stocks against each other within a day removes that shared part,
    and the daily ICs are the natural units to average and bootstrap.
    """
    out = {}
    for day, g in df.groupby("entry_date"):
        if len(g) >= min_names and g[score].nunique() > 1 and g[target].nunique() > 1:
            out[day] = spearmanr(g[score], g[target]).correlation
    return pd.Series(out, name="ic", dtype=float)


def daily_hits(df, score="score"):
    """Per entry day: how many non-neutral predictions (n) and how many had the right sign (hits)."""
    d = df[df[score] != 0]
    hit = np.sign(d[score]) == np.where(d["abn_ret"] > 0, 1, -1)
    g = hit.groupby(d["entry_date"])
    return pd.DataFrame({"hits": g.sum(), "n": g.size()})


def day_bootstrap_ci(num, den=None, n=2000, seed=0):
    """95% CI for sum(num)/sum(den) over days, resampling WHOLE DAYS with replacement.

    Resampling days (a cluster bootstrap) keeps the within-day dependence intact, so the interval is
    wider and more honest than resampling single headlines. Accuracy: num = hits, den = n per day.
    Mean daily IC: num = IC per day, den = None (every day counts once).
    It does not model dependence BETWEEN days (e.g. volatility clustering), so it is still optimistic.
    """
    num = np.asarray(num, dtype=float)
    if len(num) == 0:                                            # nothing to resample (e.g. too few names per day)
        return np.array([np.nan, np.nan])
    den = np.ones_like(num) if den is None else np.asarray(den, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(num), size=(n, len(num)))          # n resamples of the day indices
    stats = num[idx].sum(axis=1) / den[idx].sum(axis=1)
    return np.percentile(stats, [2.5, 97.5])


def summarize_signal(df, score="score", n_boot=2000, seed=0):
    """One row of headline numbers for a signal: accuracy and IC, each with a day-cluster bootstrap CI."""
    hits = daily_hits(df, score)
    ic = daily_ic(df, score).dropna()
    acc_lo, acc_hi = day_bootstrap_ci(hits["hits"], hits["n"], n_boot, seed)
    ic_lo, ic_hi = day_bootstrap_ci(ic.values, None, n_boot, seed)
    return {
        "coverage": (df[score] != 0).mean(),          # share of rows where the signal is not neutral
        "accuracy": hits["hits"].sum() / hits["n"].sum() if hits["n"].sum() else np.nan, "acc_lo": acc_lo, "acc_hi": acc_hi,
        "pooled_ic": pooled_ic(df, score),
        "mean_daily_ic": ic.mean(), "ic_lo": ic_lo, "ic_hi": ic_hi,
        "ic_t": ic.mean() / (ic.std() / np.sqrt(len(ic))),    # t-statistic of the mean daily IC (days as units)
        "n_days": len(ic),
    }


def cross_section_ic(df, score="score", target="abn_ret", n_boot=1000, seed=0):
    """Rank correlation after removing each day's mean from BOTH columns, with a day-cluster bootstrap CI.

    Why: the target is the return minus SPY, but high-beta stocks still lose more than SPY on a crash day, and
    an LLM score is also lower on such a day. A plain pooled IC would then partly measure "bad market day", not
    "bad headline". Subtracting the day mean keeps only the comparison between stocks on the same day. This is
    meant for subsets (an event type, a sub-period) where there are too few names per day for a per-day IC;
    the day means are taken over the rows passed in, so pass the whole sample, subset afterwards if the subset
    should keep the full day's mean (see the notebook).
    """
    d = df[["entry_date", score, target]].dropna().copy()
    if len(d) < 3 or d[score].nunique() < 2:
        return {"ic": np.nan, "lo": np.nan, "hi": np.nan, "n": len(d), "n_days": d["entry_date"].nunique()}
    d["s"] = d[score] - d.groupby("entry_date")[score].transform("mean")
    d["y"] = d[target] - d.groupby("entry_date")[target].transform("mean")
    return _demeaned_ic_with_ci(d, n_boot, seed)


def _demeaned_ic_with_ci(d, n_boot, seed):
    """d has columns entry_date, s, y (already demeaned by day)."""
    ic = spearmanr(d["s"], d["y"]).correlation
    days = [g[["s", "y"]].values for _, g in d.groupby("entry_date")]
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(days), len(days))               # resample whole days
        x = np.vstack([days[i] for i in pick])
        boots.append(spearmanr(x[:, 0], x[:, 1]).correlation)
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return {"ic": ic, "lo": lo, "hi": hi, "n": len(d), "n_days": len(days)}


def subset_cross_section_ic(df, mask, score="score", target="abn_ret", n_boot=1000, seed=0):
    """cross_section_ic for df[mask], but the day means come from ALL rows of df (the full day), not just the subset."""
    d = df[["entry_date", score, target]].dropna().copy()
    keep = mask.reindex(d.index).fillna(False).astype(bool)
    d["s"] = d[score] - d.groupby("entry_date")[score].transform("mean")
    d["y"] = d[target] - d.groupby("entry_date")[target].transform("mean")
    sub = d[keep]
    if len(sub) < 3 or sub["s"].nunique() < 2:
        return {"ic": np.nan, "lo": np.nan, "hi": np.nan, "n": len(sub), "n_days": sub["entry_date"].nunique()}
    return _demeaned_ic_with_ci(sub, n_boot, seed)

