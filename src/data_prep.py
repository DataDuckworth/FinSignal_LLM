"""Load news, align each headline to a tradeable date, attach next-day abnormal return.

News columns: published_at (UTC timestamp), ticker, headline
Optional column: time_known (False = only the calendar date is known, see entry_date)
Convention (conservative): you can trade at the close of the first trading day whose close is
AFTER the news time. Target = return from that close to the next close, minus SPY's.

build_dataset() takes either a CSV path or a DataFrame that is already in memory, and optionally a
price table you have already downloaded, so a notebook can go straight from its cleaned DataFrame to
the final dataset without a CSV round trip or a second download.
"""
import pandas as pd
import yfinance as yf
from pandas.tseries.offsets import BDay


def clean_news(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True)
    df = df.dropna(subset=["ticker", "headline"]).drop_duplicates(["ticker", "headline"])
    return df.reset_index(drop=True)


def load_news(path: str) -> pd.DataFrame:
    return clean_news(pd.read_csv(path))


def entry_date(ts: pd.Timestamp, time_known: bool = True) -> pd.Timestamp:
    if not time_known:
        # Only the calendar date is known (e.g. Benzinga rows stamped 00:00:00). The news may have
        # come out after the 4pm close, so the first close we can be SURE is after it is the next
        # business day's. Costs us a day of reaction but cannot leak the answer into the features.
        stated_date = ts.tz_convert("UTC").tz_localize(None).normalize()
        return stated_date + BDay(1)    # Fri/Sat/Sun -> Monday, Mon -> Tuesday, ...
    t = ts.tz_convert("US/Eastern")
    d = t.tz_localize(None).normalize()
    if t.weekday() >= 5:          # weekend -> Monday
        return d + BDay(0)
    if t.hour >= 16:              # after close -> next business day
        return d + BDay(1)
    return d                      # during market hours -> today's close


def abnormal_returns_from_prices(px: pd.DataFrame) -> pd.DataFrame:
    """px: adjusted closes, one column per ticker plus 'SPY'. Returns long table (entry_date, ticker, abn_ret)."""
    ret = px.pct_change(fill_method=None).shift(-1)   # close(t) -> close(t+1)
    abn = ret.drop(columns="SPY").sub(ret["SPY"], axis=0)
    abn.index.name, abn.columns.name = "entry_date", "ticker"
    # dropna: pandas 3 keeps NaN rows in stack(); without this the asof join below could land on a
    # NaN row (ticker not trading that day) instead of the next valid trading day.
    return abn.stack().dropna().rename("abn_ret").reset_index()


def abnormal_returns(tickers, start, end) -> pd.DataFrame:
    px = yf.download(list(tickers) + ["SPY"], start=start, end=end, auto_adjust=True, progress=False)["Close"]
    return abnormal_returns_from_prices(px)


def build_dataset(news, prices: pd.DataFrame | None = None) -> pd.DataFrame:
    """news: CSV path or DataFrame. prices: optional adjusted-close table (with 'SPY') to avoid a download."""
    news = clean_news(news) if isinstance(news, pd.DataFrame) else load_news(news)
    if "time_known" not in news.columns:      # plain 3-column input is assumed to have real times
        news["time_known"] = True
    news["entry_date"] = [entry_date(t, known) for t, known in zip(news["published_at"], news["time_known"])]

    if prices is None:
        rets = abnormal_returns(news["ticker"].unique(),
                                news["entry_date"].min() - pd.Timedelta(days=5),
                                news["entry_date"].max() + pd.Timedelta(days=10))
    else:
        # the table must reach about a week past the last entry date, or the last returns are NaN and dropped
        wanted = [t for t in news["ticker"].unique() if t in prices.columns]
        rets = abnormal_returns_from_prices(prices[wanted + ["SPY"]])

    # merge_asof needs identical key dtypes; pandas 3 can mix datetime resolutions (s/us/ns)
    news["entry_date"] = news["entry_date"].astype("datetime64[ns]")
    rets["entry_date"] = rets["entry_date"].astype("datetime64[ns]")
    # forward asof handles holidays: take the next available trading day
    out = pd.merge_asof(news.sort_values("entry_date"), rets.sort_values("entry_date"),
                        on="entry_date", by="ticker", direction="forward",
                        tolerance=pd.Timedelta(days=5))
    out = out.dropna(subset=["abn_ret"])
    out["label_up"] = (out["abn_ret"] > 0).astype(int)
    return out.reset_index(drop=True)


if __name__ == "__main__":
    ds = build_dataset("data/raw/news.csv")
    ds.to_csv("data/processed/dataset.csv", index=False)
    print(ds.shape)
