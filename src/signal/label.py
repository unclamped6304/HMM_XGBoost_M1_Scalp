"""
label.py — Forward-looking trade labeller for supervised signal training.

For every H1 bar, looks forward up to MAX_BARS to determine whether a
long or short entry at that bar would have achieved a 2:1 R:R outcome.

Label values:
  0 = No trade  (target not reached within MAX_BARS, or ambiguous)
  1 = Long      (2R target hit before 1R stop, long direction)
  2 = Short     (2R target hit before 1R stop, short direction)

Stop/Target definition:
  R = ATR(14) on H1 × ATR_MULT[h4_regime]
  Long:  stop = entry - R,   target = entry + 2R
  Short: stop = entry + R,   target = entry - 2R
  Spread is deducted: entry is shifted by +spread (long) or -spread (short)

Constraints:
  - Minimum hold = 1 H1 bar (naturally enforced — no intra-bar close)
  - Maximum hold = MAX_BARS H1 bars (48h)
  - If stop and target hit on same bar → stop wins (conservative)
  - Spread = 1 pip per pair (JPY pairs: 0.01, others: 0.0001)

Usage:
  from src.signal.label import compute_labels

  df = compute_labels("EURUSD")
  # df.columns: [H1 features..., h4_regime, base_regime, quote_regime, label]
  # df.label:   0=No trade, 1=Long, 2=Short
"""

import numpy as np
import pandas as pd

from src.hmm.features import load_bars, _compute_ohlcv_features
from src.hmm.regime import RegimeLookup

# ── Config ────────────────────────────────────────────────────────────────────

MAX_BARS = 16    # max H1 bars to look forward (16h max hold)
ATR_PERIOD = 14

# ATR multiplier per H4 regime (0-6).
# We start uniform at 1.5 — tune per-regime after reviewing regime semantics.
ATR_MULT: dict[int, float] = {
    0: 1.5,
    1: 1.5,
    2: 1.5,
    3: 1.5,
    4: 1.5,
    5: 1.5,
    6: 1.5,
}
DEFAULT_MULT = 1.5   # fallback if regime is unknown

# Spread in price units per pair
def _spread(symbol: str) -> float:
    jpy_pairs = {
        "USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "CADJPY",
        "CHFJPY", "NZDJPY",
    }
    return 0.01 if symbol.upper() in jpy_pairs else 0.0001


# ── ATR ───────────────────────────────────────────────────────────────────────

def _atr(bars: pd.DataFrame) -> pd.Series:
    prev_close = bars["close"].shift(1)
    tr = pd.concat([
        bars["high"] - bars["low"],
        (bars["high"] - prev_close).abs(),
        (bars["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(ATR_PERIOD).mean()


# ── Forward labeller ──────────────────────────────────────────────────────────

def _label_bar(
    i: int,
    closes: np.ndarray,
    highs:  np.ndarray,
    lows:   np.ndarray,
    r:      float,
    spread: float,
) -> tuple[bool, bool]:
    """
    For bar i, look forward up to MAX_BARS to determine if a long or
    short entry would achieve 2:1 R:R after accounting for spread.

    Returns (long_win, short_win).
    """
    entry = closes[i]
    n     = len(closes)

    # Long: we buy at Ask = entry + spread
    long_entry  = entry + spread
    long_stop   = long_entry - r
    long_target = long_entry + 2 * r

    # Short: we sell at Bid = entry (no spread adjustment needed on entry,
    # but we pay spread on close → target must be 2R + spread away)
    short_entry  = entry - spread
    short_stop   = short_entry + r
    short_target = short_entry - 2 * r

    long_win   = False
    short_win  = False
    long_dead  = False   # True once long stop has been hit
    short_dead = False   # True once short stop has been hit

    for j in range(i + 1, min(i + 1 + MAX_BARS, n)):
        h = highs[j]
        l = lows[j]

        # Long outcome — only check if still alive
        if not long_win and not long_dead:
            if h >= long_target and l > long_stop:
                long_win = True          # target hit without stop being hit
            elif l <= long_stop:
                long_dead = True         # stopped out — never revisit

        # Short outcome — only check if still alive
        if not short_win and not short_dead:
            if l <= short_target and h < short_stop:
                short_win = True         # target hit without stop being hit
            elif h >= short_stop:
                short_dead = True        # stopped out — never revisit

        # Early exit once both directions are resolved
        if (long_win or long_dead) and (short_win or short_dead):
            break

    return long_win, short_win


# ── Main API ──────────────────────────────────────────────────────────────────

def compute_labels(symbol: str, regime_lookup: RegimeLookup = None) -> pd.DataFrame:
    """
    Compute H1 bar features + regime context + trade labels for a symbol.

    Args:
        symbol:         e.g. "EURUSD"
        regime_lookup:  shared RegimeLookup instance (created if not provided)

    Returns:
        DataFrame indexed by H1 datetime with columns:
          [H1 features, h4_regime, base_regime, quote_regime, R, label]
    """
    symbol = symbol.upper()
    if regime_lookup is None:
        regime_lookup = RegimeLookup()

    print(f"[label] {symbol}: loading H1 bars...")
    bars = load_bars(symbol, "h1")

    # H1 features (reuses the same feature set as HMM but on H1)
    features = _compute_ohlcv_features(bars, include_volume=True)

    # ATR series aligned to bar index
    atr = _atr(bars).reindex(features.index)

    # Regime labels for every H1 bar
    print(f"[label] {symbol}: loading regime context...")
    regime_hist = regime_lookup.label_history(symbol)

    # Align: keep only bars present in all three
    idx = features.index.intersection(regime_hist.index).intersection(atr.dropna().index)
    features    = features.loc[idx]
    regime_hist = regime_hist.loc[idx]
    atr         = atr.loc[idx]

    # Build R per bar based on H4 regime
    h4_regimes = regime_hist["pair_regime"].values
    atr_vals   = atr.values
    r_vals     = np.array([
        atr_vals[k] * ATR_MULT.get(int(h4_regimes[k]), DEFAULT_MULT)
        for k in range(len(idx))
    ])

    spread = _spread(symbol)

    # Pre-extract arrays for speed
    closes = bars["close"].reindex(idx).values
    highs  = bars["high"].reindex(idx).values
    lows   = bars["low"].reindex(idx).values
    n      = len(idx)

    print(f"[label] {symbol}: labelling {n} bars (max lookahead={MAX_BARS}h)...")

    labels     = np.zeros(n, dtype=np.int8)   # 0=No trade
    long_wins  = 0
    short_wins = 0
    ambiguous  = 0

    for i in range(n - MAX_BARS):
        r = r_vals[i]
        if np.isnan(r) or r <= 0:
            continue

        long_win, short_win = _label_bar(i, closes, highs, lows, r, spread)

        if long_win and not short_win:
            labels[i] = 1
            long_wins += 1
        elif short_win and not long_win:
            labels[i] = 2
            short_wins += 1
        elif long_win and short_win:
            ambiguous += 1   # both would win — skip (leave as 0)

    # Assemble output DataFrame
    df = features.copy()
    df["h4_regime"]    = regime_hist["pair_regime"].values
    df["base_regime"]  = regime_hist["base_regime"].values
    df["quote_regime"] = regime_hist["quote_regime"].values
    df["R"]            = r_vals
    df["label"]        = labels

    total    = n - MAX_BARS
    tradeable = long_wins + short_wins
    print(
        f"[label] {symbol}: {total} bars evaluated | "
        f"Long={long_wins} ({100*long_wins/total:.1f}%) | "
        f"Short={short_wins} ({100*short_wins/total:.1f}%) | "
        f"Ambiguous={ambiguous} | "
        f"No trade={total - tradeable - ambiguous} ({100*(total-tradeable-ambiguous)/total:.1f}%)"
    )

    return df


if __name__ == "__main__":
    df = compute_labels("EURUSD")
    print("\nLabel distribution:")
    print(df["label"].value_counts().sort_index().rename({0: "No trade", 1: "Long", 2: "Short"}))
    print("\nLabel counts per H4 regime:")
    print(df.groupby(["h4_regime", "label"]).size().unstack(fill_value=0))
