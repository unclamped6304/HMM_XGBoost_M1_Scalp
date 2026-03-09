"""
backtest.py — Out-of-sample backtest on the held-out test set (last 15%).

Simulates bar-by-bar execution on H1 data:
  - Signal generated at H1 bar close
  - Trade entered at same bar close (next-bar-open approximation)
  - Stop and target tracked bar-by-bar through subsequent H1 highs/lows
  - Max 1 open position per pair at a time
  - 1 pip spread already baked into entry price via predict()

Metrics per pair:
  trades, win_rate, total_R, avg_R, max_drawdown_R,
  profit_factor, avg_duration_bars, trades_per_week

Usage:
  python -m src.signal.backtest --symbol EURUSD
  python -m src.signal.backtest --symbol ALL
  python -m src.signal.backtest --symbol ALL --plot
"""

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

from src.hmm.features import load_bars, TRAIN_RATIO, VAL_RATIO
from src.signal.predict import SignalEngine
from src.signal.label import MAX_BARS

RESULTS_DIR        = Path("models/signal/backtest")
CONFIDENCE_THRESHOLD = 0.70   # minimum model confidence to take a trade

ALL_PAIRS = [
    "AUDCAD", "AUDCHF", "AUDJPY", "AUDNZD", "AUDUSD",
    "CADCHF", "CADJPY", "CHFJPY",
    "EURAUD", "EURCAD", "EURCHF", "EURGBP", "EURJPY", "EURNZD", "EURUSD",
    "GBPAUD", "GBPCAD", "GBPCHF", "GBPJPY", "GBPNZD", "GBPUSD",
    "NZDCAD", "NZDCHF", "NZDJPY", "NZDUSD",
    "USDCAD", "USDCHF", "USDJPY",
]


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class Trade:
    symbol:        str
    direction:     str        # "long" or "short"
    entry_time:    pd.Timestamp
    entry_price:   float
    stop:          float
    target:        float
    R:             float
    exit_time:     pd.Timestamp = None
    exit_price:    float       = 0.0
    outcome:       str         = "open"   # "win", "loss", "expired"
    pnl_R:         float       = 0.0
    duration_bars: int         = 0
    regime:        int         = -1
    confidence:    float       = 0.0


# ── Core simulation ───────────────────────────────────────────────────────────

def _simulate_trade(
    trade: Trade,
    bars:  pd.DataFrame,
    entry_idx: int,
) -> Trade:
    """
    Walk forward from entry_idx bar-by-bar until stop, target, or MAX_BARS.
    Mutates and returns the trade.
    """
    highs  = bars["high"].values
    lows   = bars["low"].values
    closes = bars["close"].values
    times  = bars.index
    n      = len(bars)

    for j in range(entry_idx + 1, min(entry_idx + 1 + MAX_BARS, n)):
        h = highs[j]
        l = lows[j]

        if trade.direction == "long":
            hit_target = h >= trade.target
            hit_stop   = l <= trade.stop
        else:
            hit_target = l <= trade.target
            hit_stop   = h >= trade.stop

        if hit_stop and not hit_target:
            trade.exit_time    = times[j]
            trade.exit_price   = trade.stop
            trade.outcome      = "loss"
            trade.pnl_R        = -1.0
            trade.duration_bars = j - entry_idx
            return trade

        if hit_target and not hit_stop:
            trade.exit_time    = times[j]
            trade.exit_price   = trade.target
            trade.outcome      = "win"
            trade.pnl_R        = 2.0
            trade.duration_bars = j - entry_idx
            return trade

        if hit_target and hit_stop:
            # Both hit same bar — conservative: stop wins
            trade.exit_time    = times[j]
            trade.exit_price   = trade.stop
            trade.outcome      = "loss"
            trade.pnl_R        = -1.0
            trade.duration_bars = j - entry_idx
            return trade

    # Expired without hitting either level
    trade.exit_time    = times[min(entry_idx + MAX_BARS, n - 1)]
    trade.exit_price   = closes[min(entry_idx + MAX_BARS, n - 1)]
    trade.outcome      = "expired"
    trade.pnl_R        = 0.0
    trade.duration_bars = MAX_BARS
    return trade


# ── Backtest runner ───────────────────────────────────────────────────────────

def run_backtest(symbol: str, engine: SignalEngine) -> list[Trade]:
    symbol = symbol.upper()
    bars   = load_bars(symbol, "h1")

    # Test set: last 15% of bars
    n      = len(bars)
    i_test = int(n * (TRAIN_RATIO + VAL_RATIO))
    test_bars = bars.iloc[i_test:]

    highs  = test_bars["high"].values
    lows   = test_bars["low"].values
    times  = test_bars.index

    trades: list[Trade] = []
    in_trade = False
    current_trade: Trade | None = None
    current_exit_idx = -1

    print(f"[backtest] {symbol}: {len(test_bars)} test bars "
          f"({times[0].date()} to {times[-1].date()})")

    for i, dt in enumerate(times):
        # Close out current trade if its exit bar has passed
        if in_trade and i >= current_exit_idx:
            trades.append(current_trade)
            in_trade = False
            current_trade = None

        # Skip if already in a trade
        if in_trade:
            continue

        # Get signal for this bar
        try:
            sig = engine.predict(symbol, dt)
        except Exception:
            continue

        if sig["signal"] == "none" or sig["R"] <= 0:
            continue
        if sig["confidence"] < CONFIDENCE_THRESHOLD:
            continue

        # Open new trade
        trade = Trade(
            symbol      = symbol,
            direction   = sig["signal"],
            entry_time  = dt,
            entry_price = sig["entry"],
            stop        = sig["stop"],
            target      = sig["target"],
            R           = sig["R"],
            regime      = sig["regime"],
            confidence  = sig["confidence"],
        )

        # Simulate forward from this bar
        # We need the absolute index in the full bars array
        abs_idx = bars.index.get_loc(dt)
        _simulate_trade(trade, bars, abs_idx)

        in_trade       = True
        current_trade  = trade
        # Find the index of exit_time in test_bars
        if trade.exit_time in times:
            current_exit_idx = times.get_loc(trade.exit_time)
        else:
            current_exit_idx = i + trade.duration_bars

    # Flush any open trade at end of test set
    if in_trade and current_trade:
        trades.append(current_trade)

    return trades


# ── Metrics ───────────────────────────────────────────────────────────────────

def summarise(trades: list[Trade], symbol: str) -> dict:
    if not trades:
        return {"symbol": symbol, "trades": 0}

    pnls      = np.array([t.pnl_R for t in trades])
    wins      = pnls[pnls > 0]
    losses    = pnls[pnls < 0]
    durations = np.array([t.duration_bars for t in trades])

    # Equity curve for drawdown
    equity     = np.cumsum(pnls)
    peak       = np.maximum.accumulate(equity)
    drawdown   = equity - peak
    max_dd     = float(drawdown.min())

    # Profit factor
    gross_win  = float(wins.sum())  if len(wins)   > 0 else 0.0
    gross_loss = float(abs(losses.sum())) if len(losses) > 0 else 1.0
    pf         = gross_win / gross_loss if gross_loss > 0 else float("inf")

    # Trade frequency (trades per week, assuming H1 bars, test ~15% of 10yr data)
    total_bars  = durations.sum()
    weeks_approx = len(trades) / max(total_bars / (24 * 5), 1)  # rough

    outcomes = [t.outcome for t in trades]

    return {
        "symbol":            symbol,
        "trades":            len(trades),
        "wins":              int((pnls == 2.0).sum()),
        "losses":            int((pnls == -1.0).sum()),
        "expired":           outcomes.count("expired"),
        "win_rate":          round(float((pnls > 0).mean() * 100), 1),
        "total_R":           round(float(pnls.sum()), 2),
        "avg_R":             round(float(pnls.mean()), 3),
        "max_drawdown_R":    round(max_dd, 2),
        "profit_factor":     round(pf, 2),
        "avg_duration_bars": round(float(durations.mean()), 1),
        "equity_curve":      equity.tolist(),
    }


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_report(results: list[dict]) -> None:
    tradeable = [r for r in results if r.get("trades", 0) > 0]
    tradeable.sort(key=lambda r: r.get("total_R", 0), reverse=True)

    print(f"\n{'='*95}")
    print(f"{'BACKTEST RESULTS — TEST SET (out-of-sample)'}")
    print(f"{'='*95}")
    print(
        f"{'Symbol':<10} {'Trades':>7} {'WinRate':>8} {'Total R':>9} "
        f"{'Avg R':>7} {'MaxDD R':>9} {'PF':>7} {'AvgDur':>8}"
    )
    print("-" * 95)
    for r in tradeable:
        print(
            f"{r['symbol']:<10} {r['trades']:>7} {r['win_rate']:>7.1f}% "
            f"{r['total_R']:>+9.1f} {r['avg_R']:>+7.3f} "
            f"{r['max_drawdown_R']:>+9.1f} {r['profit_factor']:>7.2f} "
            f"{r['avg_duration_bars']:>7.1f}h"
        )

    # Portfolio aggregate
    all_pnls = []
    for r in tradeable:
        all_pnls.extend([2.0] * r["wins"] + [-1.0] * r["losses"])
    if all_pnls:
        arr  = np.array(all_pnls)
        wins = arr[arr > 0].sum()
        loss = abs(arr[arr < 0].sum())
        print("-" * 95)
        print(
            f"{'TOTAL':<10} {len(all_pnls):>7} {100*float((arr>0).mean()):>7.1f}% "
            f"{arr.sum():>+9.1f} {arr.mean():>+7.3f} "
            f"{'':>9} {wins/loss if loss>0 else 0:>7.2f}"
        )


def plot_equity_curves(results: list[dict], save: bool = True) -> None:
    tradeable = [r for r in results if r.get("trades", 0) > 0 and r.get("equity_curve")]
    tradeable.sort(key=lambda r: r.get("total_R", 0), reverse=True)

    n     = len(tradeable)
    ncols = 4
    nrows = (n + ncols - 1) // ncols + 1   # +1 for aggregate row

    fig = plt.figure(figsize=(20, nrows * 3))
    fig.suptitle("Backtest Equity Curves — Test Set (Out-of-Sample)", fontsize=13)
    gs  = gridspec.GridSpec(nrows, ncols, figure=fig)

    # Aggregate equity curve (top row, full width)
    ax_agg = fig.add_subplot(gs[0, :])
    combined = np.zeros(max(len(r["equity_curve"]) for r in tradeable))
    for r in tradeable:
        eq = np.array(r["equity_curve"])
        combined[:len(eq)] += eq
    ax_agg.plot(combined, color="#2c3e50", linewidth=1.2)
    ax_agg.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax_agg.fill_between(range(len(combined)), combined, 0,
                         where=combined >= 0, color="#2ecc71", alpha=0.3)
    ax_agg.fill_between(range(len(combined)), combined, 0,
                         where=combined < 0,  color="#e74c3c", alpha=0.3)
    ax_agg.set_title(f"Portfolio ({len(tradeable)} pairs combined)", fontsize=10)
    ax_agg.set_ylabel("Cumulative R")

    # Per-pair charts
    for idx, r in enumerate(tradeable):
        row = (idx // ncols) + 1
        col = idx % ncols
        ax  = fig.add_subplot(gs[row, col])
        eq  = np.array(r["equity_curve"])
        color = "#2ecc71" if eq[-1] >= 0 else "#e74c3c"
        ax.plot(eq, color=color, linewidth=0.9)
        ax.axhline(0, color="gray", linewidth=0.4, linestyle="--")
        ax.set_title(
            f"{r['symbol']}  {r['total_R']:+.1f}R  WR={r['win_rate']}%",
            fontsize=7.5
        )
        ax.tick_params(labelsize=6)

    plt.tight_layout()
    if save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out = RESULTS_DIR / "equity_curves.png"
        plt.savefig(out, dpi=150)
        print(f"\nChart saved -> {out}")
    plt.show()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run out-of-sample backtest")
    parser.add_argument("--symbol", required=True, help="e.g. EURUSD or ALL")
    parser.add_argument("--plot",   action="store_true", help="Show equity curve charts")
    args = parser.parse_args()

    symbols = ALL_PAIRS if args.symbol.upper() == "ALL" else [args.symbol.upper()]

    print("Loading signal engine...")
    engine  = SignalEngine()
    results = []

    for symbol in symbols:
        try:
            trades = run_backtest(symbol, engine)
            summary = summarise(trades, symbol)
            results.append(summary)
        except Exception as e:
            print(f"  ERROR {symbol}: {e}")
            results.append({"symbol": symbol, "trades": 0})

    print_report(results)

    if args.plot:
        plot_equity_curves(results)


if __name__ == "__main__":
    main()
