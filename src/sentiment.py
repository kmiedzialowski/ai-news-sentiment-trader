from transformers import pipeline
import os
import pandas as pd


#finbert
_classifier = pipeline(
    "text-classification",
    model="ProsusAI/finbert",
    top_k=None,  
)


def score_headline(text):

    results = _classifier(text[:512])[0]
    probs = {r["label"]: r["score"] for r in results}
    return probs["positive"] - probs["negative"]


def score_headlines(texts, batch_size=32):
    results = _classifier(list(texts), batch_size=batch_size, truncation=True)
    scores = []
    for r in results:
        probs = {item["label"]: item["score"] for item in r}
        scores.append(probs["positive"] - probs["negative"])
    return scores

def daily_sentiment(ticker, news):
    path = os.path.join("data", f"sentiment_{ticker}.csv")

    if os.path.exists(path):
        scored = pd.read_csv(path, parse_dates=["trade_date"])
    else:
        print(f"Scoring {len(news)} headlines with FinBERT (one-time)...")
        scored = news.copy()
        scored["score"] = score_headlines(scored["headline"])
        scored.to_csv(path, index=False)

    daily = scored.groupby("trade_date")["score"].agg(["mean", "count"])
    daily.columns = ["sentiment", "n_headlines"]
    return daily

if __name__ == "__main__":
    from data import loadprices
    from news import load_headlines

    prices = loadprices("NVDA", "2015-01-01", "2020-06-10")
    news = load_headlines("NVDA", prices.index)
    daily = daily_sentiment("NVDA", news)

    print(daily.head(10))
    print(daily["sentiment"].describe())
