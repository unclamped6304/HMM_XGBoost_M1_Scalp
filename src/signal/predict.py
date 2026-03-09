"""
predict.py — Runtime signal inference engine.

Loads all trained per-regime XGBoost models and provides two interfaces:

  1. predict(symbol, dt) — backtest mode
     Given a symbol and a historical datetime, looks up the features and
     regime for that bar and returns a signal.

  2. predict_live(symbol, bar) — live mode
     Given a symbol and a dict/Series of the current H1 bar's OHLCV values,
     computes features on-the-fly and returns a signal.

Signal output dict:
  {
    "signal":      "long" | "short" | "none",
    "regime":      int (0-6, the H4 regime that fired),
    "confidence":  float (model probability for predicted class),
    "R":           float (stop size in price units),
    "entry":       float (suggested entry price),
    "stop":        float (stop loss price),
    "target":      float (take profit price),
  }

Usage:
  from src.signal.predict import SignalEngine

  engine = SignalEngine()

  # Backtest: look up signal for a known historical bar
  sig = engine.predict("EURUSD", pd.Timestamp("2024-06-15 12:00:00"))

  # Live: pass in the just-closed H1 bar
  sig = engine.predict_live("EURUSD", {
      "open": 1.0850, "high": 1.0872, "low": 1.0841, "close": 1.0865,
      "tick_volume": 3200,
  })
"""

from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.hmm.features import load_bars, _compute_ohlcv_features  # load_bars used in predict_live
from src.hmm.regime import RegimeLookup
from src.signal.label import ATR_MULT, DEFAULT_MULT, _atr, _spread
from src.signal.train import FEATURE_COLS, MODELS_DIR

warnings.filterwarnings("ignore")

SIGNAL_MAP = {0: "none", 1: "long", 2: "short"}
NO_SIGNAL  = {
    "signal": "none", "regime": -1, "confidence": 0.0,
    "R": 0.0, "entry": 0.0, "stop": 0.0, "target": 0.0,
}


class SignalEngine:
    """
    Loads all per-regime signal models once and serves predictions.
    Safe to instantiate once and reuse across many bars.
    """

    def __init__(self):
        self._models:        dict[str, dict[int, object]] = {}  # symbol -> {regime -> model}
        self._feature_cache: dict[str, pd.DataFrame]      = {}  # symbol -> full feature df
        self._close_cache:   dict[str, pd.Series]         = {}  # symbol -> H1 close series
        self._regime_lookup: RegimeLookup | None          = None

    # ── Lazy loaders ──────────────────────────────────────────────────────────

    def _get_regime_lookup(self) -> RegimeLookup:
        if self._regime_lookup is None:
            self._regime_lookup = RegimeLookup()
        return self._regime_lookup

    def _load_models(self, symbol: str) -> dict[int, object]:
        if symbol not in self._models:
            models = {}
            for regime in range(7):
                path = MODELS_DIR / f"{symbol}_regime{regime}.pkl"
                if path.exists():
                    models[regime] = joblib.load(path)
            if not models:
                raise FileNotFoundError(
                    f"No signal models found for {symbol}. "
                    f"Run: python -m src.signal.train --symbol {symbol}"
                )
            self._models[symbol] = models
        return self._models[symbol]

    def _load_features(self, symbol: str) -> pd.DataFrame:
        """Load and cache full H1 feature history for backtest lookups."""
        if symbol not in self._feature_cache:
            bars     = load_bars(symbol, "h1")
            features = _compute_ohlcv_features(bars, include_volume=True)
            atr      = _atr(bars).reindex(features.index)

            rl          = self._get_regime_lookup()
            regime_hist = rl.label_history(symbol)

            idx = features.index.intersection(regime_hist.index).intersection(atr.dropna().index)
            features    = features.loc[idx].copy()
            regime_hist = regime_hist.loc[idx]
            atr         = atr.loc[idx]

            features["h4_regime"]    = regime_hist["pair_regime"].values
            features["base_regime"]  = regime_hist["base_regime"].values
            features["quote_regime"] = regime_hist["quote_regime"].values
            features["atr_raw"]      = atr.values
            features["close_price"]  = bars["close"].reindex(idx).values

            self._feature_cache[symbol] = features
        return self._feature_cache[symbol]

    # ── Signal computation ────────────────────────────────────────────────────

    def _run_model(
        self,
        symbol:  str,
        regime:  int,
        feature_row: np.ndarray,
        entry_price: float,
        atr_val: float,
    ) -> dict:
        """Run the regime-specific model on one feature row and build signal dict."""
        models = self._load_models(symbol)
        if regime not in models:
            return NO_SIGNAL

        model  = models[regime]
        X      = feature_row.reshape(1, -1).astype(np.float32)
        proba  = model.predict_proba(X)[0]   # [p_none, p_long, p_short]
        pred   = int(np.argmax(proba))
        conf   = float(proba[pred])

        if pred == 0:
            return NO_SIGNAL

        sp     = _spread(symbol)
        r_mult = ATR_MULT.get(regime, DEFAULT_MULT)
        R      = atr_val * r_mult

        if pred == 1:   # Long
            entry  = entry_price + sp
            stop   = entry - R
            target = entry + 2 * R
        else:           # Short
            entry  = entry_price - sp
            stop   = entry + R
            target = entry - 2 * R

        return {
            "signal":     SIGNAL_MAP[pred],
            "regime":     regime,
            "confidence": round(conf, 4),
            "R":          round(R, 6),
            "entry":      round(entry, 6),
            "stop":       round(stop, 6),
            "target":     round(target, 6),
        }

    # ── Public API ────────────────────────────────────────────────────────────

    def predict(self, symbol: str, dt: pd.Timestamp) -> dict:
        """
        Backtest mode: return signal for a historical H1 bar datetime.

        Args:
            symbol: e.g. "EURUSD"
            dt:     exact H1 bar datetime (must match a bar close timestamp)

        Returns:
            Signal dict or NO_SIGNAL if bar not found or no trade.
        """
        symbol   = symbol.upper()
        features = self._load_features(symbol)

        if dt not in features.index:
            # Try nearest bar at or before dt
            idx = features.index.asof(dt)
            if pd.isna(idx):
                return NO_SIGNAL
            dt = idx

        row    = features.loc[dt]
        regime = int(row["h4_regime"])
        X      = row[FEATURE_COLS].values.astype(np.float32)

        entry_price = float(row["close_price"]) if not np.isnan(row["close_price"]) else 0.0
        atr_val     = float(row["atr_raw"])     if not np.isnan(row["atr_raw"])     else 0.0

        return self._run_model(symbol, regime, X, entry_price, atr_val)

    def predict_live(self, symbol: str, bar: dict | pd.Series, regime_context: dict) -> dict:
        """
        Live mode: compute features from the just-closed H1 bar and predict.

        Args:
            symbol:          e.g. "EURUSD"
            bar:             dict with keys: open, high, low, close, tick_volume
                             Should be the last ~20 bars as a DataFrame for
                             rolling features to be meaningful — see note below.
            regime_context:  output of RegimeLookup.get(symbol, dt)

        Note:
            For reliable rolling features (momentum_60 etc.), pass a DataFrame
            of the last 80+ H1 bars as `bar`. Single-bar dicts are supported
            but rolling features will be NaN-filled with 0.

        Returns:
            Signal dict or NO_SIGNAL.
        """
        symbol  = symbol.upper()
        regime  = regime_context.get("pair_regime", -1)
        if regime < 0:
            return NO_SIGNAL

        # If a full DataFrame is provided, compute features properly
        if isinstance(bar, pd.DataFrame):
            features = _compute_ohlcv_features(bar, include_volume=True)
            if features.empty:
                return NO_SIGNAL
            row     = features.iloc[-1]
            atr_val = float(_atr(bar).iloc[-1])
            entry_price = float(bar["close"].iloc[-1])
        else:
            # Single bar — rolling features will be missing, fill with 0
            row_dict = {col: 0.0 for col in FEATURE_COLS}
            row_dict["log_return"] = np.log(bar["close"] / bar["open"]) if bar["open"] > 0 else 0.0
            row_dict["hl_range"]   = (bar["high"] - bar["low"]) / bar["close"] if bar["close"] > 0 else 0.0
            row = pd.Series(row_dict)
            atr_val     = bar.get("atr", 0.0)
            entry_price = bar["close"]

        row["base_regime"]  = float(regime_context.get("base_regime",  0))
        row["quote_regime"] = float(regime_context.get("quote_regime", 0))

        X = np.array([row.get(col, 0.0) for col in FEATURE_COLS], dtype=np.float32)
        return self._run_model(symbol, regime, X, entry_price, atr_val)


if __name__ == "__main__":
    engine = SignalEngine()

    # Spot-check a few historical bars
    test_cases = [
        ("EURUSD", pd.Timestamp("2024-01-15 10:00:00")),
        ("EURUSD", pd.Timestamp("2024-06-15 12:00:00")),
        ("EURUSD", pd.Timestamp("2023-03-20 08:00:00")),
        ("GBPUSD", pd.Timestamp("2024-06-15 12:00:00")),
    ]

    print(f"\n{'Symbol':<10} {'Datetime':<25} {'Signal':<8} {'Regime':<8} {'Conf':<8} {'R':>10} {'Stop':>10} {'Target':>10}")
    print("-" * 90)
    for symbol, dt in test_cases:
        sig = engine.predict(symbol, dt)
        print(
            f"{symbol:<10} {str(dt):<25} {sig['signal']:<8} {sig['regime']:<8} "
            f"{sig['confidence']:<8.3f} {sig['R']:>10.5f} {sig['stop']:>10.5f} {sig['target']:>10.5f}"
        )
