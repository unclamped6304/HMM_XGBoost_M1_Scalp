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
import json
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

from src.hmm.features import load_bars, TRAIN_RATIO, VAL_RATIO
from src.signal.predict import SignalEngine
from src.signal.label import MAX_BARS, MAX_BARS_MAP, _atr
from src.signal.train import ALL_PAIRS
from src.config import LIVE_PAIRS, CONFIDENCE_THRESHOLD

CHANDELIER_MULT = 3.0   # ATR multiplier for chandelier trailing stop

RESULTS_DIR = Path("models/signal/backtest")


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
    trade:     Trade,
    bars:      pd.DataFrame,
    entry_idx: int,
    atr_vals:  np.ndarray,
    max_bars:  int = MAX_BARS,
) -> Trade:
    """
    Walk forward from entry_idx bar-by-bar until stop, target, or max_bars.

    Uses a chandelier trailing stop (CHANDELIER_MULT × ATR from the highest
    high / lowest low since entry) alongside the fixed SL.  The effective stop
    is the tighter of the two, so the chandelier only overrides the fixed SL
    once the trade is sufficiently in profit.

    Mutates and returns the trade.
    """
    highs  = bars["high"].values
    lows   = bars["low"].values
    closes = bars["close"].values
    times  = bars.index
    n      = len(bars)

    # Track the running extreme since entry for the chandelier calculation
    trail_high = highs[entry_idx]
    trail_low  = lows[entry_idx]

    for j in range(entry_idx + 1, min(entry_idx + 1 + max_bars, n)):
        h = highs[j]
        l = lows[j]
        atr_j = atr_vals[j] if (j < len(atr_vals) and not np.isnan(atr_vals[j])) else 0.0

        if trade.direction == "long":
            trail_high = max(trail_high, h)
            chandelier = trail_high - CHANDELIER_MULT * atr_j if atr_j > 0 else trade.stop
            # Effective stop: chandelier only tightens past the fixed SL
            eff_stop   = max(trade.stop, chandelier)
            hit_target = h >= trade.target
            hit_stop   = l <= eff_stop
        else:
            trail_low  = min(trail_low, l)
            chandelier = trail_low + CHANDELIER_MULT * atr_j if atr_j > 0 else trade.stop
            eff_stop   = min(trade.stop, chandelier)
            hit_target = l <= trade.target
            hit_stop   = h >= eff_stop

        if hit_stop and not hit_target:
            trade.exit_time     = times[j]
            trade.exit_price    = eff_stop
            trade.outcome       = "loss" if eff_stop == trade.stop else "chandelier"
            # P&L in units of R (variable when chandelier fires)
            if trade.direction == "long":
                trade.pnl_R = (eff_stop - trade.entry_price) / trade.R
            else:
                trade.pnl_R = (trade.entry_price - eff_stop) / trade.R
            trade.pnl_R         = round(trade.pnl_R, 4)
            trade.duration_bars = j - entry_idx
            return trade

        if hit_target and not hit_stop:
            trade.exit_time     = times[j]
            trade.exit_price    = trade.target
            trade.outcome       = "win"
            trade.pnl_R         = 2.0
            trade.duration_bars = j - entry_idx
            return trade

        if hit_target and hit_stop:
            # Both hit same bar — conservative: stop wins
            trade.exit_time     = times[j]
            trade.exit_price    = eff_stop
            trade.outcome       = "loss"
            trade.pnl_R         = -1.0
            trade.duration_bars = j - entry_idx
            return trade

    # Expired without hitting either level
    trade.exit_time     = times[min(entry_idx + max_bars, n - 1)]
    trade.exit_price    = closes[min(entry_idx + max_bars, n - 1)]
    trade.outcome       = "expired"
    trade.pnl_R         = 0.0
    trade.duration_bars = max_bars
    return trade


# ── Backtest runner ───────────────────────────────────────────────────────────

def run_backtest(symbol: str, engine: SignalEngine, timeframe: str = "h1") -> list[Trade]:
    symbol    = symbol.upper()
    timeframe = timeframe.lower()
    max_bars  = MAX_BARS_MAP.get(timeframe, MAX_BARS)

    bars = load_bars(symbol, timeframe)

    # Pre-compute ATR for the full bar series (needed for chandelier stop)
    atr_vals = _atr(bars).values

    # Test set: last 15% of bars
    n      = len(bars)
    i_test = int(n * (TRAIN_RATIO + VAL_RATIO))
    test_bars = bars.iloc[i_test:]

    times = test_bars.index

    trades: list[Trade] = []
    in_trade = False
    current_trade: Trade | None = None
    current_exit_idx = -1

    print(f"[backtest] {symbol} ({timeframe.upper()}): {len(test_bars)} test bars "
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

        # Simulate forward (absolute index in full bars array)
        abs_idx = bars.index.get_loc(dt)
        _simulate_trade(trade, bars, abs_idx, atr_vals, max_bars)

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
        "chandelier":        outcomes.count("chandelier"),
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
        f"{'Avg R':>7} {'MaxDD R':>9} {'PF':>7} {'AvgDur':>8} {'Chandelier':>11}"
    )
    print("-" * 107)
    for r in tradeable:
        print(
            f"{r['symbol']:<10} {r['trades']:>7} {r['win_rate']:>7.1f}% "
            f"{r['total_R']:>+9.1f} {r['avg_R']:>+7.3f} "
            f"{r['max_drawdown_R']:>+9.1f} {r['profit_factor']:>7.2f} "
            f"{r['avg_duration_bars']:>7.1f}h "
            f"{r.get('chandelier', 0):>11}"
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
    positive  = sorted([r for r in tradeable if r.get("total_R", 0) >= 0],
                       key=lambda r: r["total_R"], reverse=True)
    negative  = sorted([r for r in tradeable if r.get("total_R", 0) <  0],
                       key=lambda r: r["total_R"], reverse=True)

    ncols  = 5
    n_pos  = len(positive)
    n_neg  = len(negative)
    n_rows = 1 + (n_pos + ncols - 1) // ncols + (1 if n_neg else 0)

    fig = plt.figure(figsize=(24, n_rows * 3.5 + 1))
    fig.suptitle(
        "Backtest Equity Curves — Sep 2024 to Mar 2026 (Out-of-Sample, Confidence > 0.70)",
        fontsize=13, fontweight="bold", y=0.995,
    )
    gs = gridspec.GridSpec(n_rows, ncols, figure=fig, hspace=0.55, wspace=0.3)

    # ── Portfolio aggregate (full width, top) ─────────────────────────────
    ax_p = fig.add_subplot(gs[0, :])
    if tradeable:
        max_len  = max(len(r["equity_curve"]) for r in tradeable)
        combined = np.zeros(max_len)
        for r in tradeable:
            eq = np.array(r["equity_curve"])
            combined[:len(eq)] += eq
        ax_p.plot(combined, color="#2c3e50", linewidth=1.5)
        ax_p.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax_p.fill_between(range(max_len), combined, 0,
                          where=combined >= 0, color="#2ecc71", alpha=0.35)
        ax_p.fill_between(range(max_len), combined, 0,
                          where=combined <  0, color="#e74c3c", alpha=0.35)
        total = combined[-1]
        ax_p.set_title(
            f"All Pairs Portfolio — Total: {total:+.0f}R | "
            f"{len(positive)} profitable pairs | {len(negative)} excluded pairs",
            fontsize=10,
        )
    ax_p.set_ylabel("Cumulative R")
    ax_p.tick_params(labelsize=8)

    # ── Profitable pairs ──────────────────────────────────────────────────
    for idx, r in enumerate(positive):
        row = (idx // ncols) + 1
        col = idx % ncols
        ax  = fig.add_subplot(gs[row, col])
        eq  = np.array(r["equity_curve"])
        ax.plot(eq, color="#27ae60", linewidth=1.0)
        ax.axhline(0, color="gray", linewidth=0.4, linestyle="--")
        ax.fill_between(range(len(eq)), eq, 0, color="#27ae60", alpha=0.15)
        ax.set_title(
            f"{r['symbol']}  {r['total_R']:+.0f}R\n"
            f"WR={r['win_rate']:.0f}%  PF={r['profit_factor']:.2f}  "
            f"DD={r['max_drawdown_R']:.0f}R",
            fontsize=7.5, fontweight="bold",
        )
        ax.tick_params(labelsize=6)

    # ── Negative/excluded pairs ───────────────────────────────────────────
    if negative:
        neg_row = 1 + (n_pos + ncols - 1) // ncols
        for idx, r in enumerate(negative):
            ax = fig.add_subplot(gs[neg_row, idx])
            eq = np.array(r["equity_curve"])
            ax.plot(eq, color="#c0392b", linewidth=1.0)
            ax.axhline(0, color="gray", linewidth=0.4, linestyle="--")
            ax.fill_between(range(len(eq)), eq, 0, color="#c0392b", alpha=0.15)
            ax.set_title(
                f"[EXCL] {r['symbol']}  {r['total_R']:+.0f}R\n"
                f"WR={r['win_rate']:.0f}%  PF={r['profit_factor']:.2f}",
                fontsize=7.5, color="#c0392b",
            )
            ax.tick_params(labelsize=6)

    if save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out = RESULTS_DIR / "equity_curves.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"\nChart saved -> {out}")
    plt.show()


# ── CLI ───────────────────────────────────────────────────────────────────────

CACHE_FILE = RESULTS_DIR / "results_cache.json"


def save_results(results: list[dict]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "w") as f:
        json.dump(results, f)
    print(f"Results cached -> {CACHE_FILE}")


def load_results() -> list[dict] | None:
    if CACHE_FILE.exists():
        with open(CACHE_FILE) as f:
            return json.load(f)
    return None


def main():
    parser = argparse.ArgumentParser(description="Run out-of-sample backtest")
    parser.add_argument("--symbol",    required=False,      default="ALL", help="e.g. EURUSD or ALL")
    parser.add_argument("--timeframe", default="h1",                       help="Bar timeframe: h1, m15, m5 (default: h1)")
    parser.add_argument("--plot",      action="store_true", help="Show equity curve charts")
    parser.add_argument("--replot",    action="store_true", help="Replot from cached results (no rerun)")
    args = parser.parse_args()

    tf = args.timeframe.lower()

    # Cache file is per-timeframe
    global CACHE_FILE
    CACHE_FILE = RESULTS_DIR / f"results_cache_{tf}.json"

    # Fast replot from cache
    if args.replot:
        results = load_results()
        if results is None:
            print(f"No cached results found for {tf}. Run without --replot first.")
            return
        print_report(results)
        plot_equity_curves(results)
        return

    symbols = ALL_PAIRS if args.symbol.upper() == "ALL" else [args.symbol.upper()]

    print(f"Loading signal engine ({tf.upper()})...")
    engine  = SignalEngine(timeframe=tf)
    results = []

    for symbol in symbols:
        try:
            trades  = run_backtest(symbol, engine, timeframe=tf)
            summary = summarise(trades, symbol)
            results.append(summary)
        except Exception as e:
            print(f"  ERROR {symbol}: {e}")
            results.append({"symbol": symbol, "trades": 0})

    print_report(results)
    save_results(results)

    if args.plot:
        plot_equity_curves(results)


if __name__ == "__main__":
    main()
