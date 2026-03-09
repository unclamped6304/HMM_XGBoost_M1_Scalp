"""
features.py — Load OHLCV bars from Postgres and compute HMM input features.

Two feature sets:
  1. load_features(symbol, timeframe)  — per-pair H4 features for HMM Layer 1
  2. load_currency_features(currency)  — D1 currency strength features for HMM Layer 2

Per-bar features (H4 Layer 1):
  Single-bar:
    - log_return  : log(close / prev_close)
    - hl_range    : (high - low) / prev_close
    - atr_ratio   : hl_range / ATR(14)
    - vol_ratio   : tick_volume / mean_volume(20)
    - body_ratio  : abs(close - open) / (high - low)
  Multi-bar context (gives HMM structural memory):
    - momentum_5  : rolling mean log_return over 5 bars  (~20h)
    - momentum_20 : rolling mean log_return over 20 bars (~3.5 days)
    - momentum_60 : rolling mean log_return over 60 bars (~10 days)
    - vol_regime  : rolling std of log_return over 20 bars (volatility level)
    - vol_trend   : vol_regime / vol_regime.rolling(60).mean() (vol expanding or contracting)
    - price_pos   : (close - low_20) / (high_20 - low_20) — position in 20-bar range

Currency strength (D1 layer):
  For each currency, average log-returns across all related pairs,
  flipping the sign when the currency is the quote currency.

Train/val/test split (chronological, no shuffling):
  70% train | 15% val | 15% test

Usage:
  from src.hmm.features import load_features, load_currency_features

  # H4 per-pair (Layer 1)
  train, val, test = load_features("EURUSD", "h4", split=True)
  full = load_features("EURUSD", "h4", split=False)

  # D1 currency strength (Layer 2)
  train, val, test = load_currency_features("EUR", split=True)
  full = load_currency_features("EUR", split=False)
"""

import os
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

ATR_PERIOD  = 14
VOL_PERIOD  = 20
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# TEST_RATIO  = remaining 0.15

# Maps each currency to its pairs and whether it is the BASE (+1) or QUOTE (-1).
# Sign tells us whether to flip the return when computing currency strength.
CURRENCY_PAIRS = {
    "EUR": {
        "EURUSD": +1, "EURGBP": +1, "EURJPY": +1,
        "EURAUD": +1, "EURCAD": +1, "EURCHF": +1, "EURNZD": +1,
    },
    "GBP": {
        "GBPUSD": +1, "GBPJPY": +1, "GBPAUD": +1,
        "GBPCAD": +1, "GBPCHF": +1, "GBPNZD": +1,
        "EURGBP": -1,
    },
    "USD": {
        "EURUSD": -1, "GBPUSD": -1, "AUDUSD": -1, "NZDUSD": -1,
        "USDJPY": +1, "USDCAD": +1, "USDCHF": +1,
    },
    "JPY": {
        "USDJPY": -1, "EURJPY": -1, "GBPJPY": -1, "AUDJPY": -1,
        "CADJPY": -1, "CHFJPY": -1, "NZDJPY": -1,
    },
    "AUD": {
        "AUDUSD": +1, "AUDCAD": +1, "AUDCHF": +1,
        "AUDJPY": +1, "AUDNZD": +1,
        "EURAUD": -1, "GBPAUD": -1,
    },
    "NZD": {
        "NZDUSD": +1, "NZDCAD": +1, "NZDCHF": +1, "NZDJPY": +1,
        "AUDNZD": -1, "EURNZD": -1, "GBPNZD": -1,
    },
    "CAD": {
        "USDCAD": -1, "AUDCAD": -1, "NZDCAD": -1,
        "CADCHF": +1, "CADJPY": +1,
        "EURCAD": -1, "GBPCAD": -1,
    },
    "CHF": {
        "USDCHF": -1, "AUDCHF": -1, "CADCHF": -1, "NZDCHF": -1,
        "CHFJPY": +1,
        "EURCHF": -1, "GBPCHF": -1,
    },
}


# ─── DB ───────────────────────────────────────────────────────────────────────

def _get_engine():
    host     = os.getenv("DB_HOST", "localhost")
    dbname   = os.getenv("DB_NAME", "postgres")
    user     = os.getenv("DB_USER", "postgres")
    password = os.getenv("DB_PASSWORD")
    return create_engine(f"postgresql+psycopg2://{user}:{password}@{host}/{dbname}")


def load_bars(symbol: str, timeframe: str) -> pd.DataFrame:
    """Load all OHLCV bars for a symbol/timeframe, sorted ascending."""
    table = f'"historicalData".{symbol.lower()}_{timeframe.lower()}'
    query = f"""
        SELECT date, time, open, high, low, close, tick_volume
        FROM {table}
        ORDER BY date ASC, time ASC
    """
    with _get_engine().connect() as conn:
        df = pd.read_sql(query, conn)

    df["datetime"] = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str))
    df = df.drop(columns=["date", "time"]).set_index("datetime")
    df = df.astype({
        "open": float, "high": float, "low": float,
        "close": float, "tick_volume": float,
    })
    return df


# ─── FEATURE COMPUTATION ──────────────────────────────────────────────────────

def _compute_ohlcv_features(df: pd.DataFrame, include_volume: bool = True) -> pd.DataFrame:
    """Compute HMM input features from raw OHLCV bars."""
    f = pd.DataFrame(index=df.index)

    # ── Single-bar features ──────────────────────────────────────────────────
    log_ret = np.log(df["close"] / df["close"].shift(1))
    f["log_return"] = log_ret
    f["hl_range"]   = (df["high"] - df["low"]) / df["close"].shift(1)

    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(ATR_PERIOD).mean()
    f["atr_ratio"] = f["hl_range"] / atr

    if include_volume:
        vol = df["tick_volume"].replace(0, np.nan).ffill()
        f["vol_ratio"] = vol / vol.rolling(VOL_PERIOD).mean()

        candle_range = (df["high"] - df["low"]).replace(0, np.nan)
        f["body_ratio"] = (df["close"] - df["open"]).abs() / candle_range

    # ── Multi-bar context features ───────────────────────────────────────────
    # Trend momentum at three timeframes
    f["momentum_5"]  = log_ret.rolling(5).mean()
    f["momentum_20"] = log_ret.rolling(20).mean()
    f["momentum_60"] = log_ret.rolling(60).mean()

    # Volatility level and whether it is expanding or contracting
    vol_regime       = log_ret.rolling(20).std()
    f["vol_regime"]  = vol_regime
    f["vol_trend"]   = vol_regime / vol_regime.rolling(60).mean()

    # Price position within the recent 20-bar range (0 = at lows, 1 = at highs)
    high_20 = df["high"].rolling(20).max()
    low_20  = df["low"].rolling(20).min()
    f["price_pos"] = (df["close"] - low_20) / (high_20 - low_20).replace(0, np.nan)

    return f.dropna()


def _compute_strength_features(strength: pd.Series) -> pd.DataFrame:
    """
    Compute HMM features from a currency strength series (D1 layer).
    The strength series is already a normalised return, so we treat it
    directly as log_return and derive volatility features from it.
    """
    f = pd.DataFrame(index=strength.index)

    f["log_return"]  = strength
    f["volatility"]  = strength.rolling(ATR_PERIOD).std()
    f["vol_ratio"]   = f["volatility"] / f["volatility"].rolling(VOL_PERIOD).mean()
    f["momentum_5"]  = strength.rolling(5).mean()   # short-term drift
    f["momentum_20"] = strength.rolling(20).mean()  # medium-term drift

    return f.dropna()


# ─── SPLITS ───────────────────────────────────────────────────────────────────

def _split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological 70/15/15 split. No shuffling."""
    n     = len(df)
    i_val = int(n * TRAIN_RATIO)
    i_tst = int(n * (TRAIN_RATIO + VAL_RATIO))
    return df.iloc[:i_val], df.iloc[i_val:i_tst], df.iloc[i_tst:]


# ─── PUBLIC API ───────────────────────────────────────────────────────────────

def load_features(
    symbol: str,
    timeframe: str,
    split: bool = True,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and compute H4 (or any timeframe) per-pair features.

    Args:
        symbol:    e.g. "EURUSD"
        timeframe: e.g. "h4", "d1"
        split:     if True returns (train, val, test); else returns full DataFrame

    Returns:
        DataFrame or (train_df, val_df, test_df)
    """
    bars     = load_bars(symbol, timeframe)
    features = _compute_ohlcv_features(bars, include_volume=True)
    print(f"[features] {symbol} {timeframe.upper()}: {len(features)} bars, {features.shape[1]} features")

    if split:
        train, val, test = _split(features)
        print(f"           train={len(train)}  val={len(val)}  test={len(test)}")
        return train, val, test
    return features


def load_currency_features(
    currency: str,
    split: bool = True,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute D1 currency strength and return HMM features for the D1 layer.

    Strength = average log-return across all pairs involving the currency,
               with sign flipped when the currency is the quote currency.

    Args:
        currency: e.g. "EUR", "USD", "JPY"
        split:    if True returns (train, val, test); else returns full DataFrame

    Returns:
        DataFrame or (train_df, val_df, test_df)
    """
    currency = currency.upper()
    pairs    = CURRENCY_PAIRS[currency]

    returns = {}
    for symbol, sign in pairs.items():
        try:
            bars = load_bars(symbol, "d1")
            ret  = np.log(bars["close"] / bars["close"].shift(1)) * sign
            returns[symbol] = ret
        except Exception as e:
            print(f"[currency] WARNING: could not load {symbol} d1 — {e}")

    if not returns:
        raise RuntimeError(f"No D1 data found for currency {currency}")

    # Align on common dates and average
    aligned  = pd.DataFrame(returns).dropna(how="all")
    strength = aligned.mean(axis=1).dropna()

    features = _compute_strength_features(strength)
    print(f"[currency] {currency} D1: {len(features)} bars from {len(returns)} pairs")

    if split:
        train, val, test = _split(features)
        print(f"           train={len(train)}  val={len(val)}  test={len(test)}")
        return train, val, test
    return features


if __name__ == "__main__":
    print("=== H4 Per-Pair Features (EURUSD) ===")
    train, val, test = load_features("EURUSD", "h4", split=True)
    print(train.describe())

    print("\n=== D1 Currency Strength Features (EUR) ===")
    train, val, test = load_currency_features("EUR", split=True)
    print(train.describe())
