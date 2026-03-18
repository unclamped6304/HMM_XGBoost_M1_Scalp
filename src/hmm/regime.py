"""
regime.py — Regime lookup: given a symbol and datetime, return the current
HMM regime labels from both layers.

Layer 1 (H4): What is this pair doing right now?
Layer 2 (D1): What are the two underlying currencies doing?

All models are loaded once and cached in memory on first use.

Usage:
  from src.hmm.regime import RegimeLookup

  rl = RegimeLookup()

  # Get full regime context for a pair at a specific datetime
  ctx = rl.get("EURUSD", pd.Timestamp("2024-06-15 12:00:00"))
  # ctx = {
  #   "pair_regime":  3,        # H4 EURUSD regime
  #   "base_regime":  1,        # D1 EUR regime
  #   "quote_regime": 5,        # D1 USD regime
  #   "base":  "EUR",
  #   "quote": "USD",
  # }

  # Label the full history of a pair (for backtesting)
  df = rl.label_history("EURUSD")
  # df.columns: pair_regime, base_regime, quote_regime
"""

import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from hmmlearn import _hmmc

from src.hmm.features import (
    load_features,
    load_currency_features,
    CURRENCY_PAIRS,
)
from src.hmm.train import load_model, MODELS_DIR
from src.hmm.visualise import SMOOTH_WINDOW

warnings.filterwarnings("ignore")

# Which currency is base/quote for each pair
_PAIR_CURRENCIES: dict[str, tuple[str, str]] = {
    "AUDCAD": ("AUD", "CAD"), "AUDCHF": ("AUD", "CHF"), "AUDJPY": ("AUD", "JPY"),
    "AUDNZD": ("AUD", "NZD"), "AUDUSD": ("AUD", "USD"), "CADCHF": ("CAD", "CHF"),
    "CADJPY": ("CAD", "JPY"), "CHFJPY": ("CHF", "JPY"), "EURAUD": ("EUR", "AUD"),
    "EURCAD": ("EUR", "CAD"), "EURCHF": ("EUR", "CHF"), "EURGBP": ("EUR", "GBP"),
    "EURJPY": ("EUR", "JPY"), "EURNZD": ("EUR", "NZD"), "EURUSD": ("EUR", "USD"),
    "GBPAUD": ("GBP", "AUD"), "GBPCAD": ("GBP", "CAD"), "GBPCHF": ("GBP", "CHF"),
    "GBPJPY": ("GBP", "JPY"), "GBPNZD": ("GBP", "NZD"), "GBPUSD": ("GBP", "USD"),
    "NZDCAD": ("NZD", "CAD"), "NZDCHF": ("NZD", "CHF"), "NZDJPY": ("NZD", "JPY"),
    "NZDUSD": ("NZD", "USD"), "USDCAD": ("USD", "CAD"), "USDCHF": ("USD", "CHF"),
    "USDJPY": ("USD", "JPY"),
}


def _forward_decode(model, scaled: np.ndarray) -> np.ndarray:
    """
    Causal (forward-only) HMM state decoding.

    Uses the forward algorithm (filtering) so the regime at time t is
    determined solely by observations up to and including t — no future
    data leaks in.  This matches what is achievable in live trading.

    Contrast with model.predict() which runs Viterbi on the full sequence
    and therefore uses all future observations when labelling each bar.
    """
    log_frameprob = model._compute_log_likelihood(scaled)
    _, fwdlattice  = _hmmc.forward_log(
        model.startprob_, model.transmat_, log_frameprob
    )
    # Normalise each row: log P(state | obs_1..t)  →  probabilities
    log_norm   = logsumexp(fwdlattice, axis=1, keepdims=True)
    posteriors = np.exp(fwdlattice - log_norm)
    return posteriors.argmax(axis=1)


def _smooth_labels(labels: np.ndarray) -> np.ndarray:
    """
    Apply causal rolling mode smoothing (center=False).

    Uses only the current bar and the preceding SMOOTH_WINDOW-1 bars,
    so no future data is consumed.  center=True would leak future bars.
    """
    s = pd.Series(labels)
    return (
        s.rolling(SMOOTH_WINDOW, center=False, min_periods=1)
         .apply(lambda x: pd.Series(x).mode()[0], raw=True)
         .astype(int)
         .values
    )


class RegimeLookup:
    """
    Loads all trained HMM models once, labels full histories, and provides
    fast O(1) regime lookup by datetime.
    """

    def __init__(self):
        self._pair_labels:     dict[str, pd.Series] = {}
        self._currency_labels: dict[str, pd.Series] = {}
        self._loaded_pairs:    set[str] = set()
        self._loaded_currencies: set[str] = set()

    def _load_pair(self, symbol: str) -> None:
        if symbol in self._loaded_pairs:
            return
        model_name = f"{symbol}_h4"
        if not (MODELS_DIR / f"{model_name}.pkl").exists():
            raise FileNotFoundError(
                f"No model for {symbol} H4. Run: "
                f"python -m src.hmm.train --mode pair --symbol {symbol}"
            )
        model, scaler, _ = load_model(model_name)
        features = load_features(symbol, "h4", split=False)
        scaled   = scaler.transform(features.values)
        raw      = _forward_decode(model, scaled)
        smoothed = _smooth_labels(raw)
        self._pair_labels[symbol] = pd.Series(smoothed, index=features.index, name="pair_regime")
        self._loaded_pairs.add(symbol)

    def _load_currency(self, currency: str) -> None:
        if currency in self._loaded_currencies:
            return
        model_name = f"{currency}_d1"
        if not (MODELS_DIR / f"{model_name}.pkl").exists():
            raise FileNotFoundError(
                f"No model for {currency} D1. Run: "
                f"python -m src.hmm.train --mode currency --symbol {currency}"
            )
        model, scaler, _ = load_model(model_name)
        features = load_currency_features(currency, split=False)
        scaled   = scaler.transform(features.values)
        raw      = _forward_decode(model, scaled)
        smoothed = _smooth_labels(raw)
        self._currency_labels[currency] = pd.Series(smoothed, index=features.index, name="regime")
        self._loaded_currencies.add(currency)

    def _nearest_pair_label(self, symbol: str, dt: pd.Timestamp) -> int:
        """Return the H4 regime label at or just before dt."""
        labels = self._pair_labels[symbol]
        idx    = labels.index.asof(dt)
        if pd.isna(idx):
            return -1
        return int(labels[idx])

    def _nearest_currency_label(self, currency: str, dt: pd.Timestamp) -> int:
        """Return the D1 regime label at or just before dt."""
        labels = self._currency_labels[currency]
        # Use the date only — D1 bars align to day boundaries
        day_dt = pd.Timestamp(dt.date())
        idx    = labels.index.asof(day_dt)
        if pd.isna(idx):
            return -1
        return int(labels[idx])

    def get(self, symbol: str, dt: pd.Timestamp) -> dict:
        """
        Return regime context for a symbol at a specific datetime.

        Returns dict with keys:
          pair_regime  — H4 regime (0-6) for this pair
          base_regime  — D1 regime (0-6) for the base currency
          quote_regime — D1 regime (0-6) for the quote currency
          base         — base currency string
          quote        — quote currency string
        """
        symbol = symbol.upper()
        base, quote = _PAIR_CURRENCIES[symbol]

        self._load_pair(symbol)
        self._load_currency(base)
        self._load_currency(quote)

        return {
            "pair_regime":  self._nearest_pair_label(symbol, dt),
            "base_regime":  self._nearest_currency_label(base, dt),
            "quote_regime": self._nearest_currency_label(quote, dt),
            "base":  base,
            "quote": quote,
        }

    def label_history(self, symbol: str) -> pd.DataFrame:
        """
        Return a DataFrame with regime labels for the full H4 history of a pair.
        Index is the H4 bar datetime. D1 currency regimes are forward-filled
        to align with H4 timestamps.

        Columns: pair_regime, base_regime, quote_regime
        """
        symbol = symbol.upper()
        base, quote = _PAIR_CURRENCIES[symbol]

        self._load_pair(symbol)
        self._load_currency(base)
        self._load_currency(quote)

        pair_idx = self._pair_labels[symbol].index

        # Reindex D1 labels onto H4 timestamps via forward fill
        base_aligned = (
            self._currency_labels[base]
            .reindex(pair_idx, method="ffill")
            .rename("base_regime")
        )
        quote_aligned = (
            self._currency_labels[quote]
            .reindex(pair_idx, method="ffill")
            .rename("quote_regime")
        )

        df = pd.DataFrame({
            "pair_regime":  self._pair_labels[symbol],
            "base_regime":  base_aligned,
            "quote_regime": quote_aligned,
        })
        return df.dropna()


if __name__ == "__main__":
    rl = RegimeLookup()

    # Point-in-time lookup
    dt  = pd.Timestamp("2024-06-15 12:00:00")
    ctx = rl.get("EURUSD", dt)
    print(f"\nEURUSD regime context at {dt}:")
    print(f"  H4 pair regime : {ctx['pair_regime']}")
    print(f"  D1 {ctx['base']} regime  : {ctx['base_regime']}")
    print(f"  D1 {ctx['quote']} regime  : {ctx['quote_regime']}")

    # Full history
    print("\nLabelling full EURUSD history...")
    hist = rl.label_history("EURUSD")
    print(hist.tail(10))
    print(f"\nTotal labelled bars: {len(hist)}")
    print("\nRegime combination counts (top 10):")
    print(hist.value_counts().head(10))
