import os
import pandas as pd
import yfinance as yf

DATA_DIR = "data"


def loadprices(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Load daily prices for a ticker (cached in data/), trimmed to [start, end]."""
    path = os.path.join(DATA_DIR, f"{ticker}.csv")

    if os.path.exists(path):
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    else:
        df = yf.download(ticker, start=start, end=end, auto_adjust=True)

        # Don't save an empty file if the download failed
        if df.empty:
            raise ValueError(f"No data downloaded for {ticker}. Check the ticker, "
                             "your internet connection, or update yfinance.")

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df[["Open", "High", "Low", "Close", "Volume"]]
        df = df.dropna()

        os.makedirs(DATA_DIR, exist_ok=True)
        df.to_csv(path)

    # The cached file may cover a different range, so keep only the dates asked for
    return df.loc[start:end]

