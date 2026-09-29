import pandas as pd

NEWS_FILE = "data/news/raw_analyst_ratings.csv"


def load_headlines(ticker, trading_days):

    news = pd.read_csv(NEWS_FILE, usecols=["headline", "date", "stock"])
    news = news[news["stock"] == ticker].copy()

    raw = news["date"]
    one_day = pd.Timedelta(days=1)

    # Two formats in this file:
    #   "2020-05-22 11:38:59-04:00" -> exact time with time zone
    #   "2020-05-28 00:00:00"       -> date only, publish time unknown
    has_time = raw.str.contains(r"[+-]\d\d:\d\d$")

    day = pd.Series(pd.NaT, index=news.index, dtype="datetime64[ns]")

    # Exact times: convert to market (New York) time; at/after 4 PM -> next day
    ts = pd.to_datetime(raw[has_time], utc=True).dt.tz_convert("America/New_York")
    d = ts.dt.normalize().dt.tz_localize(None)
    d = d + (ts.dt.hour >= 16) * one_day   # adds 1 day only where the hour is 16 or later
    day[has_time] = d

    # Date only: assume it came out after the close (conservative) -> next day
    day[~has_time] = pd.to_datetime(raw[~has_time].str[:10]) + one_day

    # Weekends/holidays → roll forward to the next trading day
    trading_days = pd.DatetimeIndex(trading_days)
    idx = trading_days.searchsorted(day)
    # Keep only news inside our price range (not before the first day or after the last)
    valid = (day.values >= trading_days[0]) & (idx < len(trading_days))
    news = news[valid].copy()
    news["trade_date"] = trading_days[idx[valid]]

    return news[["trade_date", "headline"]].sort_values("trade_date")


if __name__ == "__main__":
    from data import loadprices
    prices = loadprices("NVDA", "2015-01-01", "2020-06-10")
    news = load_headlines("NVDA", prices.index)
    print(news.head(10))
    print(len(news), "headlines on", news["trade_date"].nunique(), "trading days")