import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_results(results: pd.DataFrame, title: str, filename: str = "results/equity_curve.png") -> None:
    """Plot every strategy column vs. buy-and-hold and save as a PNG."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for col in results.columns:
        style = {"color": "gray"} if col == "buy_and_hold" else {}
        ax.plot(results.index, results[col], label=col.replace("_", " ").title(), **style)

    ax.set_title(title)
    ax.set_ylabel("Portfolio value ($)")
    ax.legend()
    ax.grid(alpha=0.3)

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Chart saved to {filename}")
def run_backtest(df: pd.DataFrame, signal: pd.Series, initial_cash: float = 10000) -> pd.DataFrame:
    # Percent change perday
    daily_return = df["Close"].pct_change().fillna(0)

    # acting on signal only until the next day
    position = signal.shift(1).fillna(0)

    strategy_return = position * daily_return 

    #Grow money day by day
    strategy_equity = initial_cash * (1 + strategy_return).cumprod()
    buyhold_equity = initial_cash * (1 + daily_return).cumprod()

    results = pd.DataFrame({"strategy": strategy_equity, "buy_and_hold": buyhold_equity})
    return results


def summarize(equity: pd.Series) -> dict[str, str]:
    total_return = equity.iloc[-1]/ equity.iloc[0] - 1
    running_peak = equity.cummax()
    drawdown = equity / running_peak - 1
    return {
        "total_return": f"{total_return:.1%}",
        "max_drawdown": f"{drawdown.min():.1%}",
        "final_value": f"${equity.iloc[-1]:,.2f}",
    }
