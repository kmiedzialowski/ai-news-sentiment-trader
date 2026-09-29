import pandas as pd

def ma_crossover(df, short=20, long=50):
    short_ma = df["Close"].rolling(short).mean()
    long_ma = df["Close"].rolling(long).mean()
    signal = (short_ma > long_ma).astype(int)
    return signal


def sentiment_signal(daily, trading_days, window=5, threshold=0.0):

    # One value per trading day; days with no news count as neutral (0)
    sent = daily["sentiment"].reindex(trading_days).fillna(0)

    # Average over the last `window` trading days
    smoothed = sent.rolling(window, min_periods=1).mean()

    return (smoothed > threshold).astype(int)
