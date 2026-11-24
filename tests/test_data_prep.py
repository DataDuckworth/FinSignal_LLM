"""Alignment tests: when can a headline be traded, and which return is attached to it.

These are the hand-made cases from notebook 01 plus a small end-to-end join on invented prices (no network).
"""
import numpy as np
import pandas as pd
import pytest

from src.data_prep import abnormal_returns_from_prices, build_dataset, entry_date


def eastern(text):
    """A timezone-aware UTC timestamp built from a US/Eastern wall-clock string."""
    return pd.Timestamp(text, tz="US/Eastern").tz_convert("UTC")


# (timestamp, time_known, expected entry day)
ENTRY_CASES = [
    (eastern("2019-06-07 15:59"), True, "2019-06-07"),     # Friday, before the close: today's close
    (eastern("2019-06-07 16:00"), True, "2019-06-10"),     # exactly at the close counts as after it
    (eastern("2019-06-07 23:30"), True, "2019-06-10"),     # Friday evening: Monday
    (eastern("2019-06-08 12:00"), True, "2019-06-10"),     # Saturday: Monday
    (eastern("2019-06-09 12:00"), True, "2019-06-10"),     # Sunday: Monday
    (eastern("2019-06-10 06:00"), True, "2019-06-10"),     # pre-market: same day's close
    (eastern("2019-01-08 15:30"), True, "2019-01-08"),     # winter (EST), before the close
    (eastern("2019-01-08 16:30"), True, "2019-01-09"),     # winter, after the close
    (eastern("2020-03-06 16:30"), True, "2020-03-09"),     # Friday after close, the weekend the clocks change
    (eastern("2020-03-09 15:30"), True, "2020-03-09"),     # first Monday of daylight saving time
    (pd.Timestamp("2019-06-10", tz="UTC"), False, "2019-06-11"),   # date only (Mon): next business day
    (pd.Timestamp("2019-06-07", tz="UTC"), False, "2019-06-10"),   # date only (Fri): Monday
    (pd.Timestamp("2019-06-08", tz="UTC"), False, "2019-06-10"),   # date only (Sat): Monday
]


@pytest.mark.parametrize("stamp, known, expected", ENTRY_CASES)
def test_entry_date(stamp, known, expected):
    assert entry_date(stamp, known).strftime("%Y-%m-%d") == expected


def test_date_only_news_never_enters_on_its_own_date():
    """The leak-free rule: with only a date we cannot know the time, so the entry must be strictly later."""
    for day in pd.date_range("2019-03-01", "2019-03-31"):
        got = entry_date(pd.Timestamp(day, tz="UTC"), False)
        assert got > day.tz_localize(None)


def make_prices():
    """Invented adjusted closes for one stock and SPY. The Monday 2020-01-20 holiday has no row."""
    days = [d for d in pd.bdate_range("2020-01-02", "2020-02-14") if d != pd.Timestamp("2020-01-20")]
    rng = np.random.default_rng(0)
    px = pd.DataFrame({"AAA": 100 * np.cumprod(1 + rng.normal(0, 0.01, len(days))),
                       "SPY": 300 * np.cumprod(1 + rng.normal(0, 0.005, len(days)))}, index=pd.DatetimeIndex(days))
    return px


def hand_abn_ret(px, day):
    """Abnormal return from the close of `day` to the next close, written out directly."""
    i = px.index.get_loc(pd.Timestamp(day))
    stock = px["AAA"].iloc[i + 1] / px["AAA"].iloc[i] - 1
    market = px["SPY"].iloc[i + 1] / px["SPY"].iloc[i] - 1
    return stock - market


def test_abnormal_returns_from_prices_matches_hand_calculation():
    px = make_prices()
    table = abnormal_returns_from_prices(px).set_index(["entry_date", "ticker"])["abn_ret"]
    assert table[(pd.Timestamp("2020-01-07"), "AAA")] == pytest.approx(hand_abn_ret(px, "2020-01-07"))
    assert "SPY" not in table.index.get_level_values("ticker")          # the benchmark has no abnormal return


def test_build_dataset_attaches_the_right_return():
    px = make_prices()
    news = pd.DataFrame({
        "published_at": [eastern("2020-01-07 10:00"),      # during market hours  -> enter Tue 01-07
                         eastern("2020-01-07 17:00"),      # after the close      -> enter Wed 01-08
                         pd.Timestamp("2020-01-07", tz="UTC"),    # date only     -> enter Wed 01-08
                         pd.Timestamp("2020-01-17", tz="UTC")],   # date only Fri -> Mon 01-20 is a holiday -> Tue 01-21
        "ticker": ["AAA"] * 4,
        "headline": ["h1", "h2", "h3", "h4"],
        "time_known": [True, True, False, False],
    })
    out = build_dataset(news, prices=px).set_index("headline")
    assert out.loc["h1", "entry_date"] == pd.Timestamp("2020-01-07")
    assert out.loc["h1", "abn_ret"] == pytest.approx(hand_abn_ret(px, "2020-01-07"))
    assert out.loc["h2", "abn_ret"] == pytest.approx(hand_abn_ret(px, "2020-01-08"))
    assert out.loc["h3", "abn_ret"] == pytest.approx(hand_abn_ret(px, "2020-01-08"))
    # holiday: the entry date stays Jan 20 but the return is the next trading day's (Jan 21 -> Jan 22)
    assert out.loc["h4", "abn_ret"] == pytest.approx(hand_abn_ret(px, "2020-01-21"))
    assert (out["label_up"] == (out["abn_ret"] > 0).astype(int)).all()


def test_build_dataset_drops_rows_without_a_return():
    px = make_prices()
    news = pd.DataFrame({"published_at": [pd.Timestamp("2020-06-01", tz="UTC")],      # far beyond the price table
                         "ticker": ["AAA"], "headline": ["late"], "time_known": [False]})
    assert len(build_dataset(news, prices=px)) == 0
