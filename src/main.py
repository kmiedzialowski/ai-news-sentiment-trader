import pandas as pd
from data import loadprices
from news import load_headlines
from sentiment import daily_sentiment
from strategy import ma_crossover, sentiment_signal
from backtest import run_backtest, summarize, plot_results

TICKER = "NVDA"
START, END = "2015-01-01", "2020-06-10"

prices = loadprices(TICKER, START, END)
news = load_headlines(TICKER, prices.index)
daily = daily_sentiment(TICKER, news)

signals = {
    "ma_crossover": ma_crossover(prices),
    "sentiment": sentiment_signal(daily, prices.index),
}

results = pd.DataFrame()
for name, signal in signals.items():
    bt = run_backtest(prices, signal)
    results[name] = bt["strategy"]
results["buy_and_hold"] = bt["buy_and_hold"]

for col in results.columns:
    print(f"{col:15}", summarize(results[col]))

plot_results(results, f"{TICKER}: sentiment vs. MA crossover vs. buy & hold")