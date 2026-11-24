"""Tests for the metrics and the long-short backtest, mostly on fake data where the right answer is known."""
import numpy as np
import pandas as pd
import pytest

from src.evaluate import (cross_section_ic, day_bootstrap_ci, long_short_daily, net_of_costs, performance,
                          pooled_ic, subset_cross_section_ic, ticker_day_signal)


def fake_cross_section(n_days=60, n_names=30, seed=0):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2020-01-14", periods=n_days)
    df = pd.DataFrame({"entry_date": np.repeat(days, n_names), "ticker": np.tile([f"T{i}" for i in range(n_names)], n_days)})
    day_effect = np.repeat(rng.normal(size=n_days), n_names)
    df["abn_ret"] = day_effect * 0.01 + rng.normal(size=len(df)) * 0.01
    return df, day_effect, rng


def test_net_of_costs_is_four_times_the_cost_per_trade():
    gross = pd.Series([0.01, 0.0, -0.002])
    net = net_of_costs(gross, cost_bps=10)
    assert (gross - net).tolist() == pytest.approx([0.004] * 3)        # 2 legs x 2 trades x 10 bps


def test_ticker_day_signal_averages_headlines_of_the_same_stock_and_day():
    df = pd.DataFrame({"ticker": ["A", "A", "B"], "entry_date": pd.to_datetime(["2020-01-02"] * 3),
                       "score": [1.0, 0.0, -1.0], "abn_ret": [0.01, 0.01, -0.02]})
    td = ticker_day_signal(df, "score").set_index("ticker")
    assert td.loc["A", "signal"] == 0.5 and td.loc["A", "n_headlines"] == 2
    assert len(td) == 2


def test_long_short_oracle_is_positive_every_day_and_noise_is_about_zero():
    df, _, rng = fake_cross_section()
    oracle = long_short_daily(df.assign(signal=df["abn_ret"]))
    assert (oracle["gross"] > 0).all()
    noise = long_short_daily(df.assign(signal=rng.normal(size=len(df))))
    assert abs(noise["gross"].mean()) < 0.003                         # a few bps at most on 60 days
    assert (oracle["n_long"] == 6).all() and (oracle["n_short"] == 6).all()   # 30 names / 5 quintiles


def test_long_short_skips_days_with_too_few_names():
    df, _, _ = fake_cross_section(n_days=3, n_names=30)
    small = df[(df["entry_date"] != df["entry_date"].iloc[0]) | (df["ticker"].isin(["T0", "T1", "T2", "T3", "T4"]))]
    out = long_short_daily(small.assign(signal=small["abn_ret"]), min_names=10)
    assert len(out) == 2                                               # the day that kept only 5 names is dropped


def test_tied_scores_in_the_middle_are_not_traded():
    # 20 names: 2 clearly high, 2 clearly low, 16 tied at 0 -> only the 2 + 2 extremes are traded
    day = pd.Timestamp("2020-02-03")
    signal = [1.0, 1.0] + [0.0] * 16 + [-1.0, -1.0]
    td = pd.DataFrame({"entry_date": day, "signal": signal, "abn_ret": np.linspace(0.02, -0.02, 20)})
    out = long_short_daily(td, q=5, min_names=10, min_leg=2)
    assert out.loc[day, "n_long"] == 2 and out.loc[day, "n_short"] == 2


def test_performance_on_a_known_series():
    daily = pd.Series([0.01, -0.01, 0.01, 0.01])
    p = performance(daily, n_boot=200)
    assert p["mean"] == pytest.approx(0.005) and p["hit_rate"] == 0.75
    assert p["max_drawdown"] == pytest.approx(0.01)                    # one -1% day after +1%
    assert p["ci_lo"] <= p["mean"] <= p["ci_hi"]


def test_day_bootstrap_ci_handles_empty_input_without_crashing():
    lo, hi = day_bootstrap_ci([], None)
    assert np.isnan(lo) and np.isnan(hi)


def test_within_day_ic_ignores_a_pure_market_day_effect():
    df, day_effect, rng = fake_cross_section()
    df["day_only"] = day_effect + rng.normal(size=len(df)) * 0.3                 # follows the day, nothing stock specific
    df["real"] = (df["abn_ret"] - df.groupby("entry_date")["abn_ret"].transform("mean")) * 100 + rng.normal(size=len(df)) * 2
    day_only = cross_section_ic(df, "day_only", n_boot=100)
    real = cross_section_ic(df, "real", n_boot=100)
    assert pooled_ic(df, "day_only") > 0.4                                       # the plain IC is fooled ...
    assert abs(day_only["ic"]) < 0.1                                             # ... the within-day IC is not
    assert real["lo"] > 0.2


def test_subset_ic_uses_the_whole_days_mean():
    df, _, _ = fake_cross_section(n_days=30)
    df["real"] = df["abn_ret"] * 100
    half = pd.Series(df["ticker"].isin([f"T{i}" for i in range(15)]), index=df.index)
    out = subset_cross_section_ic(df, half, "real", n_boot=100)
    assert out["n"] == int(half.sum()) and out["ic"] > 0.5
    empty = subset_cross_section_ic(df, pd.Series(False, index=df.index), "real", n_boot=10)
    assert np.isnan(empty["ic"])
