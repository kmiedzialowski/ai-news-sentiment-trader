import pandas as pd


def make_label(prices: pd.DataFrame) -> pd.Series:
    close = prices["Close"]
    next_close = close.shift(-1)

    label = (next_close > close).astype(int)
    return label[next_close.notna()].rename("label")

def make_price_features(prices: pd.DataFrame) -> pd.DataFrame:
    close = prices["Close"]
    ret_1d = close.pct_change()
    ret_5d = close.pct_change(5)
    ret_20d = close.pct_change(20)
    ma20_ratio = close / close.rolling(20).mean()
    vol_20d = ret_1d.rolling(20).std()

    features = pd.DataFrame({
        "ret_1d": ret_1d,
        "ret_5d": ret_5d,
        "ret_20d": ret_20d,
        "ma20_ratio": ma20_ratio,
        "vol_20d": vol_20d,
    })
    return features

def make_sentiment_features(daily: pd.DataFrame, trading_days: pd.DatetimeIndex) -> pd.DataFrame:
    sent_1d = daily["sentiment"].reindex(trading_days).fillna(0)
    sent_5d = sent_1d.rolling(5).mean()
    n_headlines = daily["n_headlines"].reindex(trading_days).fillna(0)

    features = pd.DataFrame({
        "sent_1d": sent_1d,
        "sent_5d": sent_5d,
        "n_headlines": n_headlines,
    },
    index=trading_days
    )
    return features

def make_dataset(prices: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    price_feats = make_price_features(prices)
    sent_feats = make_sentiment_features(daily, prices.index)
    y = make_label(prices)

    data = pd.concat([price_feats, sent_feats, y], axis=1).dropna()
    data["label"] = data["label"].astype(int)
    return data

def split_by_date(data: pd.DataFrame):
    train = data.loc["2015":"2017"].iloc[:-1]
    val = data.loc["2018"].iloc[:-1]
    test = data.loc["2019":"2020"]

    return train, val, test