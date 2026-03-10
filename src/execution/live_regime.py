"""
live_regime.py — Live HMM regime detection using MT5 bars.

Mirrors the logic in src/hmm/regime.py but sources bars from MT5
instead of the PostgreSQL database, so no DB access is needed at runtime.

Caches results with a TTL:
  - H4 pair regime:     30 minutes (H4 bar closes every 4 hours)
  - D1 currency regime: 60 minutes (D1 bar closes once per day)
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from src.hmm.features import (
    _compute_ohlcv_features,
    _compute_strength_features,
    CURRENCY_PAIRS,
)
from src.hmm.regime import _PAIR_CURRENCIES
from src.hmm.train import load_model
from src.hmm.visualise import SMOOTH_WINDOW
from src.execution.mt5_connector import get_bars

log = logging.getLogger(__name__)

N_H4_BARS = 300   # enough for all rolling features (momentum_60 needs 60 bars)
N_D1_BARS = 200

_H4_TTL = timedelta(minutes=30)
_D1_TTL = timedelta(minutes=60)


def _smooth(labels: np.ndarray) -> np.ndarray:
    s = pd.Series(labels)
    return (
        s.rolling(SMOOTH_WINDOW, center=True, min_periods=1)
         .apply(lambda x: pd.Series(x).mode()[0], raw=True)
         .astype(int)
         .values
    )


class LiveRegimeDetector:
    """
    Detects current regimes using live MT5 bars and the trained HMM models.

    Models are loaded once. Regime results are cached per symbol/currency
    with TTLs to avoid redundant computation across pairs sharing currencies.
    """

    def __init__(self):
        self._pair_models:     dict[str, tuple] = {}   # symbol   -> (model, scaler)
        self._currency_models: dict[str, tuple] = {}   # currency -> (model, scaler)

        # Cache: key -> (regime_int, computed_at)
        self._pair_cache:     dict[str, tuple[int, datetime]] = {}
        self._currency_cache: dict[str, tuple[int, datetime]] = {}

    # ── Model loaders ─────────────────────────────────────────────────────────

    def _get_pair_model(self, symbol: str):
        if symbol not in self._pair_models:
            model, scaler, _ = load_model(f"{symbol}_h4")
            self._pair_models[symbol] = (model, scaler)
        return self._pair_models[symbol]

    def _get_currency_model(self, currency: str):
        if currency not in self._currency_models:
            model, scaler, _ = load_model(f"{currency}_d1")
            self._currency_models[currency] = (model, scaler)
        return self._currency_models[currency]

    # ── Regime computation ────────────────────────────────────────────────────

    def get_pair_regime(self, symbol: str) -> int:
        """
        Return the current H4 regime for a pair.
        Result is cached for 30 minutes.
        """
        now = datetime.now(timezone.utc)
        if symbol in self._pair_cache:
            regime, ts = self._pair_cache[symbol]
            if now - ts < _H4_TTL:
                return regime

        try:
            model, scaler = self._get_pair_model(symbol)
            bars     = get_bars(symbol, "h4", N_H4_BARS)
            features = _compute_ohlcv_features(bars, include_volume=True)
            if features.empty:
                return -1

            scaled   = scaler.transform(features.values)
            raw      = model.predict(scaled)
            smoothed = _smooth(raw)
            regime   = int(smoothed[-1])

        except Exception as e:
            log.error(f"[LiveRegime] pair regime failed for {symbol}: {e}")
            return -1

        self._pair_cache[symbol] = (regime, now)
        return regime

    def get_currency_regime(self, currency: str) -> int:
        """
        Return the current D1 regime for a currency.
        Result is cached for 60 minutes.
        """
        now = datetime.now(timezone.utc)
        if currency in self._currency_cache:
            regime, ts = self._currency_cache[currency]
            if now - ts < _D1_TTL:
                return regime

        try:
            model, scaler = self._get_currency_model(currency)
            pairs   = CURRENCY_PAIRS[currency]
            returns = {}

            for pair_symbol, sign in pairs.items():
                try:
                    bars = get_bars(pair_symbol, "d1", N_D1_BARS)
                    ret  = np.log(bars["close"] / bars["close"].shift(1)) * sign
                    returns[pair_symbol] = ret
                except Exception:
                    continue

            if not returns:
                return -1

            aligned  = pd.DataFrame(returns).dropna(how="all")
            strength = aligned.mean(axis=1).dropna()
            features = _compute_strength_features(strength)
            if features.empty:
                return -1

            scaled   = scaler.transform(features.values)
            raw      = model.predict(scaled)
            smoothed = _smooth(raw)
            regime   = int(smoothed[-1])

        except Exception as e:
            log.error(f"[LiveRegime] currency regime failed for {currency}: {e}")
            return -1

        self._currency_cache[currency] = (regime, now)
        return regime

    def get(self, symbol: str) -> dict:
        """
        Return full regime context for a pair:
          pair_regime, base_regime, quote_regime, base, quote
        """
        symbol      = symbol.upper()
        base, quote = _PAIR_CURRENCIES[symbol]

        return {
            "pair_regime":  self.get_pair_regime(symbol),
            "base_regime":  self.get_currency_regime(base),
            "quote_regime": self.get_currency_regime(quote),
            "base":  base,
            "quote": quote,
        }
