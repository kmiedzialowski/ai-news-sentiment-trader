# AI News-Sentiment Trader

Can the sentiment of financial news headlines drive a better trading strategy than a simple technical rule, or than just buying and holding? This project scores thousands of headlines with **FinBERT**, a language model built for financial text, turns the scores into buy/sell signals, and backtests them on NVIDIA (NVDA) from 2015 to 2020.

![Equity curves: sentiment vs. MA crossover vs. buy & hold](results/equity_curve.png)

## Results

$10,000 starting capital, NVDA, Jan 2, 2015 – Jun 9, 2020 (1,368 trading days):

| Strategy | Total return | Max drawdown | Final value | Time invested | Position changes |
| --- | --- | --- | --- | --- | --- |
| Buy & hold | **+1,766.5%** | -56.0% | $186,647 | 100% | 0 |
| 20/50-day MA crossover | +630.7% | **-37.6%** | $73,070 | 69.5% | 25 |
| News sentiment (FinBERT) | +217.6% | -55.8% | $31,759 | 59.9% | 218 |

### What the results show

- **Buy-and-hold wins.** NVDA grew about 18x over this period. Any strategy that spends time in cash misses some of the biggest up days, and on a stock like this that is very costly.
- **The moving-average rule traded return for safety.** It cut the worst drawdown from -56% to -38%. During the late-2018 crash (Sep 2018 – Jan 2019) it lost 13%, while buy-and-hold lost 49%.
- **Sentiment underperformed on both measures.** It lost 54% in the same crash window, worse than simply holding. A likely reason: many headlines describe moves that have already happened (*"NVIDIA shares are trading down 2.9% after..."*), so sentiment turns negative *after* the drop. The strategy sells near the bottom and misses the rebound. It also switched positions 218 times, about 9x as often as the MA rule.

**Takeaway:** averaged daily headline sentiment on its own was not a useful timing signal for NVDA in this period. The next phase tests whether it adds value when *combined* with price-based features in a trained model.

## How it works

```
Prices (Yahoo Finance) ──────────────────────────┐
                                                 ├─► Strategy signals ─► Backtest ─► Results + chart
News headlines (Kaggle) ─► FinBERT sentiment ────┘
```

1. **Prices.** Daily split- and dividend-adjusted prices are downloaded with `yfinance` and cached locally.
2. **News.** 2,472 NVDA headlines covering 841 trading days are loaded from a public Kaggle dataset. Each headline is assigned to the first trading day it could have affected (see below).
3. **Sentiment.** Every headline is scored with [FinBERT](https://huggingface.co/ProsusAI/finbert) as `P(positive) - P(negative)`, a number from -1 to +1. Scores are averaged into one value per day and cached, so scoring only runs once.
4. **Strategies.**
   - *MA crossover:* hold the stock when the 20-day moving average is above the 50-day, otherwise hold cash.
   - *Sentiment:* hold the stock when the 5-day average sentiment is above 0. Days with no news count as neutral.
5. **Backtest.** Each strategy is either fully invested or fully in cash. Daily returns are compounded and compared against buy-and-hold.

## Avoiding look-ahead bias

A backtest is only meaningful if it never uses information that wasn't available yet. These are the decisions I made to prevent that:

- **Trade the next day.** A signal calculated from a day's closing price can't be acted on until the following day, so every signal is shifted forward one day before it is applied.
- **Headline timing.** Timestamps are converted to New York market time. A headline published at or after 4:00 PM (market close) counts toward the next trading day, and weekend or holiday headlines roll forward to the next day the market is open.
- **Headlines with no time.** Over 99% of the NVDA headlines in the dataset have a date but no publish time. These are conservatively treated as published *after* the close and assigned to the next trading day. In the worst case the strategy reacts a day late, but it never trades on news before it existed.
- **Model knowledge.** I used FinBERT instead of a modern general-purpose LLM. A recent LLM may have read about how NVDA actually performed from 2015 to 2020, which could leak future knowledge into the sentiment scores. FinBERT's financial training data (Reuters news from 2008–2010 and the Financial PhraseBank) gives it much less of that knowledge.
- **No tuning on the test period.** The strategy settings (20/50-day windows, 5-day sentiment average, threshold of 0) were chosen up front and were not adjusted after seeing these results.

## Limitations

- One stock over one period. NVDA's exceptional run makes buy-and-hold especially hard to beat.
- No transaction costs or slippage. Including them would hurt the sentiment strategy most, since it trades the most often.
- The news data ends in June 2020, and about 39% of trading days have no NVDA headline.
- Many headlines are generic market lists (*"Stocks That Hit 52-Week Highs On Friday"*) or report price moves that already happened, which adds noise.
- Long-or-cash only: no short selling and no partial position sizing.

## Project structure

```
.
├── src/
│   ├── data.py         download and cache daily price data
│   ├── news.py         load headlines and align them to trading days
│   ├── sentiment.py    FinBERT scoring and daily averaging (cached)
│   ├── strategy.py     MA crossover and sentiment signals
│   ├── backtest.py     backtest engine, performance metrics, chart
│   ├── main.py         runs the full pipeline
│   └── features.py     (planned) features for the ML model
├── results/
│   └── equity_curve.png
└── requirements.txt
```

## Running it

Requires Python 3.

1. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Download the [Daily Financial News for 6000+ Stocks](https://www.kaggle.com/datasets/miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests) dataset from Kaggle and place `raw_analyst_ratings.csv` in `data/news/`. The data isn't included in this repo because of its size.
3. From the project's root folder, run:
   ```
   python src/main.py
   ```

The first run downloads the price data and the FinBERT model (about 400 MB), then scores every headline, which takes a few minutes on a CPU. Later runs use the cached files in `data/` and finish quickly. If you change how headlines are loaded, delete `data/sentiment_NVDA.csv` so they get re-scored.

> **Windows note:** if `pip install torch` fails with *"The filename or extension is too long"*, enable long file paths in Windows and try again.

## Roadmap

- [ ] Train a machine-learning model (logistic regression, then gradient boosting) on price features plus sentiment. Train on 2015–2018 and test on 2019–2020, which the model never sees during training.
- [ ] Add transaction costs to the backtest.
- [ ] Filter out generic, market-wide headlines.
- [ ] Support multiple tickers and portfolio-level backtests.
- [ ] Connect a live news API for current data.

## Disclaimer

This is an educational project. All results are simulated backtests on historical data, and nothing here is financial advice.

## Author

**Konrad Miedzialowski**, Computer Science, Marquette University
[LinkedIn](https://www.linkedin.com/in/konrad-miedzialowski-a32b0b1a3) · [GitHub](https://github.com/kmiedzialowski)
