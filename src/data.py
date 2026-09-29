import os
import pandas as pd
import yfinance as yf

DATA_DIR = "data"


def loadprices(ticker, start, end):
    path = os.path.join(DATA_DIR, f"{ticker}.csv")

    if os.path.exists(path):
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df

    df = yf.download(ticker, start=start, end=end, auto_adjust=True)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df = df.dropna()

    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(path)

    return df

