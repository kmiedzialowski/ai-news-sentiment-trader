import pandas as pd
from data import loadprices
from news import load_headlines
from backtest import run_backtest, plot_results, summarize
from sentiment import daily_sentiment
from features import make_dataset, split_by_date
from model import train_model, evaluate
from strategy import ma_crossover, sentiment_signal

TICKER = "NVDA"
START, END = "2015-01-01", "2020-06-10"

prices = loadprices(TICKER, START, END)
news = load_headlines(TICKER, prices.index)
daily = daily_sentiment(TICKER, news)

data = make_dataset(prices, daily)
train, val, test = split_by_date(data)

train_full = pd.concat([train, val])
model = train_model(train_full)

acc, up_share = evaluate(model, test)
baseline = test["label"].mean()

print(f"Test accuracy:    {acc:.1%}")
print(f"Always-up baseline: {baseline:.1%}")
print(f"Share predicted up: {up_share:.1%}")

X_test = test.drop(columns="label")
ml_signal = pd.Series(model.predict(X_test), index=test.index)

ma_signal = ma_crossover(prices).loc[test.index]
sent_signal = sentiment_signal(daily, prices.index).loc[test.index]

test_prices = prices.loc[test.index]

results = pd.DataFrame()
for name, signal in {"ml_model": ml_signal, "ma_crossover": ma_signal,
                     "sentiment": sent_signal}.items():
    bt = run_backtest(test_prices, signal)
    results[name] = bt["strategy"]
results["buy_and_hold"] = bt["buy_and_hold"]

for col in results.columns:
    print(f"{col:15}", summarize(results[col]))

plot_results(results, "NVDA test period (2019-2020): ML vs. baselines",
             filename="results/equity_curve_test.png")
