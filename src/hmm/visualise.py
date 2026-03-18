"""
visualise.py — Plot HMM regime labels on price charts to validate model output.

Two plot types:
  1. pair   — H4 price chart with regime colour bands + regime stats
  2. currency — D1 currency strength line with regime colour bands

The test set is highlighted separately so you can see out-of-sample regime quality.

Usage:
  python -m src.hmm.visualise --mode pair --symbol EURUSD
  python -m src.hmm.visualise --mode currency --symbol EUR

  # Zoom into a date range
  python -m src.hmm.visualise --mode pair --symbol EURUSD --start 2022-01-01 --end 2023-01-01
"""

import argparse

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from src.hmm.features import load_bars, load_features, load_currency_features, CURRENCY_PAIRS
from src.hmm.train import load_model, MODELS_DIR

# 3 visually distinct colours for the regimes
REGIME_COLOURS = [
    "#2ecc71",  # 0 — green
    "#f39c12",  # 1 — orange
    "#e74c3c",  # 2 — red
]
REGIME_ALPHA = 0.25


SMOOTH_WINDOW = 6   # bars — rolling mode window (6 × H4 = 24h, 6 × D1 = 6 days)


def _label_data(features: pd.DataFrame, model, scaler, smooth: bool = True) -> pd.Series:
    """
    Run the model on a feature DataFrame and return a Series of regime labels.
    If smooth=True, apply a rolling mode filter to remove single-bar flips.
    """
    scaled = scaler.transform(features.values)
    labels = model.predict(scaled)
    s = pd.Series(labels, index=features.index, name="regime")
    if smooth:
        s = (
            s.rolling(SMOOTH_WINDOW, center=True, min_periods=1)
             .apply(lambda x: pd.Series(x).mode()[0], raw=True)
             .astype(int)
        )
    return s


def _add_regime_bands(ax, labels: pd.Series, dates: pd.DatetimeIndex):
    """Shade the background of ax with regime colours."""
    if len(labels) == 0:
        return
    current = labels.iloc[0]
    start   = labels.index[0]

    for dt, regime in labels.items():
        if regime != current:
            ax.axvspan(start, dt, color=REGIME_COLOURS[current], alpha=REGIME_ALPHA, linewidth=0)
            current = regime
            start   = dt
    ax.axvspan(start, labels.index[-1], color=REGIME_COLOURS[current], alpha=REGIME_ALPHA, linewidth=0)


def _regime_legend(n_states: int) -> list:
    return [
        mpatches.Patch(color=REGIME_COLOURS[i], alpha=0.6, label=f"Regime {i}")
        for i in range(n_states)
    ]


def _split_indices(n: int, train_ratio=0.70, val_ratio=0.15):
    i_val = int(n * train_ratio)
    i_tst = int(n * (train_ratio + val_ratio))
    return i_val, i_tst


# ─── PAIR CHART ───────────────────────────────────────────────────────────────

def plot_pair(symbol: str, start: str = None, end: str = None, smooth: bool = True):
    model_name = f"{symbol}_h4"
    if not (MODELS_DIR / f"{model_name}.pkl").exists():
        print(f"No model found at models/hmm/{model_name}.pkl — train it first.")
        return

    model, scaler, meta = load_model(model_name)
    bars     = load_bars(symbol, "h4")
    features = load_features(symbol, "h4", split=False)

    # Align bars to feature index (features lose some rows to rolling windows)
    bars = bars.loc[features.index]

    labels = _label_data(features, model, scaler, smooth=smooth)

    # Train/val/test boundaries
    n = len(features)
    i_val, i_tst = _split_indices(n)
    val_start  = features.index[i_val]
    test_start = features.index[i_tst]

    # Optional date zoom
    if start:
        mask = labels.index >= start
        labels = labels[mask]; bars = bars[mask]
    if end:
        mask = labels.index <= end
        labels = labels[mask]; bars = bars[mask]

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(f"{symbol} H4 — HMM Regime Labels ({meta['n_states']} states)", fontsize=14)

    ax  = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2, sharex=ax)
    ax3 = fig.add_subplot(3, 1, 3)            # no sharex — different axis (regime numbers)

    # ── Price chart ──
    ax.plot(bars.index, bars["close"], color="#2c3e50", linewidth=0.7, zorder=2)
    _add_regime_bands(ax, labels, bars.index)
    ax.axvline(val_start,   color="navy",    linewidth=1.2, linestyle="--", alpha=0.7)
    ax.axvline(test_start,  color="crimson", linewidth=1.2, linestyle="--", alpha=0.7)
    ax.set_ylabel("Close price")
    ax.legend(handles=_regime_legend(meta["n_states"]) + [
        mpatches.Patch(color="navy",    label="Val start"),
        mpatches.Patch(color="crimson", label="Test start"),
    ], loc="upper left", fontsize=7, ncol=3)

    # ── Regime label stream ──
    ax2.step(labels.index, labels.values, where="post", color="#2c3e50", linewidth=0.7)
    _add_regime_bands(ax2, labels, bars.index)
    ax2.set_ylabel("Regime")
    ax2.set_yticks(range(meta["n_states"]))
    ax2.axvline(val_start,  color="navy",    linewidth=1.2, linestyle="--", alpha=0.7)
    ax2.axvline(test_start, color="crimson", linewidth=1.2, linestyle="--", alpha=0.7)

    # ── Regime distribution bar chart ──
    counts = labels.value_counts().sort_index()
    ax3.bar(counts.index, counts.values,
            color=[REGIME_COLOURS[i] for i in counts.index], edgecolor="white")
    ax3.set_xlabel("Regime")
    ax3.set_ylabel("Bar count")
    ax3.set_xticks(range(meta["n_states"]))

    plt.tight_layout()
    out = MODELS_DIR / f"{symbol}_h4_regimes.png"
    plt.savefig(out, dpi=150)
    print(f"Saved chart -> {out}")
    plt.show()


# ─── CURRENCY CHART ───────────────────────────────────────────────────────────

def plot_currency(currency: str, start: str = None, end: str = None, smooth: bool = True):
    currency   = currency.upper()
    model_name = f"{currency}_d1"
    if not (MODELS_DIR / f"{model_name}.pkl").exists():
        print(f"No model found at models/hmm/{model_name}.pkl — train it first.")
        return

    model, scaler, meta = load_model(model_name)
    features = load_currency_features(currency, split=False)
    labels   = _label_data(features, model, scaler, smooth=smooth)

    n = len(features)
    i_val, i_tst = _split_indices(n)
    val_start  = features.index[i_val]
    test_start = features.index[i_tst]

    if start:
        mask = labels.index >= start
        labels = labels[mask]; features = features[mask]
    if end:
        mask = labels.index <= end
        labels = labels[mask]; features = features[mask]

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(f"{currency} D1 Currency Strength — HMM Regime Labels ({meta['n_states']} states)", fontsize=14)

    ax  = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2, sharex=ax)
    ax3 = fig.add_subplot(3, 1, 3)            # no sharex — different axis

    # ── Strength line ──
    cumulative = features["log_return"].cumsum()
    ax.plot(cumulative.index, cumulative.values, color="#2c3e50", linewidth=0.8, zorder=2)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    _add_regime_bands(ax, labels, features.index)
    ax.axvline(val_start,  color="navy",    linewidth=1.2, linestyle="--", alpha=0.7)
    ax.axvline(test_start, color="crimson", linewidth=1.2, linestyle="--", alpha=0.7)
    ax.set_ylabel(f"{currency} cumulative strength")
    ax.legend(handles=_regime_legend(meta["n_states"]) + [
        mpatches.Patch(color="navy",    label="Val start"),
        mpatches.Patch(color="crimson", label="Test start"),
    ], loc="upper left", fontsize=7, ncol=3)

    # ── Regime label stream ──
    ax2.step(labels.index, labels.values, where="post", color="#2c3e50", linewidth=0.7)
    _add_regime_bands(ax2, labels, features.index)
    ax2.set_ylabel("Regime")
    ax2.set_yticks(range(meta["n_states"]))
    ax2.axvline(val_start,  color="navy",    linewidth=1.2, linestyle="--", alpha=0.7)
    ax2.axvline(test_start, color="crimson", linewidth=1.2, linestyle="--", alpha=0.7)

    # ── Distribution ──
    counts = labels.value_counts().sort_index()
    ax3.bar(counts.index, counts.values,
            color=[REGIME_COLOURS[i] for i in counts.index], edgecolor="white")
    ax3.set_xlabel("Regime")
    ax3.set_ylabel("Bar count")
    ax3.set_xticks(range(meta["n_states"]))

    plt.tight_layout()
    out = MODELS_DIR / f"{currency}_d1_regimes.png"
    plt.savefig(out, dpi=150)
    print(f"Saved chart -> {out}")
    plt.show()


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Visualise HMM regime labels")
    parser.add_argument("--mode",   choices=["pair", "currency"], required=True)
    parser.add_argument("--symbol", required=True, help="e.g. EURUSD or EUR")
    parser.add_argument("--start",     default=None,        help="Zoom start date YYYY-MM-DD")
    parser.add_argument("--end",       default=None,        help="Zoom end date YYYY-MM-DD")
    parser.add_argument("--no-smooth", action="store_true", help="Disable rolling mode smoothing")
    args = parser.parse_args()

    smooth = not args.no_smooth

    if args.mode == "pair":
        plot_pair(args.symbol.upper(), args.start, args.end, smooth=smooth)
    else:
        plot_currency(args.symbol.upper(), args.start, args.end, smooth=smooth)


if __name__ == "__main__":
    main()
