"""
train.py — Train per-regime XGBoost signal classifiers.

For each pair, trains 7 XGBoost models — one per H4 regime.
Each model learns: given current H1 market features, is this bar a
good Long entry, Short entry, or No trade within this specific regime?

Model inputs  : H1 OHLCV features + D1 base/quote currency regimes
Model output  : 0=No trade, 1=Long, 2=Short

Models saved to: models/signal/{SYMBOL}_regime{N}.pkl

Usage:
  python -m src.signal.train --symbol EURUSD
  python -m src.signal.train --symbol ALL
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.hmm.features import TRAIN_RATIO, VAL_RATIO
from src.hmm.regime import RegimeLookup
from src.signal.label import compute_labels

MODELS_DIR = Path("models/signal")
N_REGIMES  = 7
MIN_SAMPLES = 50   # minimum training samples per regime to bother training

# Feature columns used as model input (excludes h4_regime, R, label)
FEATURE_COLS = [
    "log_return", "hl_range", "atr_ratio", "vol_ratio", "body_ratio",
    "momentum_5", "momentum_20", "momentum_60",
    "vol_regime", "vol_trend", "price_pos",
    "base_regime", "quote_regime",
]

ALL_PAIRS = [
    "AUDCAD", "AUDCHF", "AUDJPY", "AUDNZD", "AUDUSD",
    "CADCHF", "CADJPY", "CHFJPY",
    "EURAUD", "EURCAD", "EURCHF", "EURGBP", "EURJPY", "EURNZD", "EURUSD",
    "GBPAUD", "GBPCAD", "GBPCHF", "GBPJPY", "GBPNZD", "GBPUSD",
    "NZDCAD", "NZDCHF", "NZDJPY", "NZDUSD",
    "USDCAD", "USDCHF", "USDJPY",
]


def _chronological_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Same 70/15/15 chronological split used across the whole project."""
    n     = len(df)
    i_val = int(n * TRAIN_RATIO)
    i_tst = int(n * (TRAIN_RATIO + VAL_RATIO))
    return df.iloc[:i_val], df.iloc[i_val:i_tst], df.iloc[i_tst:]


def _eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute per-class precision, recall, and overall accuracy."""
    acc = float((y_true == y_pred).mean())
    metrics = {"accuracy": round(acc, 4)}
    for cls, name in [(1, "long"), (2, "short")]:
        tp = int(((y_pred == cls) & (y_true == cls)).sum())
        fp = int(((y_pred == cls) & (y_true != cls)).sum())
        fn = int(((y_pred != cls) & (y_true == cls)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        metrics[f"{name}_precision"] = round(precision, 4)
        metrics[f"{name}_recall"]    = round(recall, 4)
    return metrics


def _simulated_pnl(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Rough P&L simulation in units of R assuming 2:1 R:R.
    Win = +2R, Loss = -1R, No trade predicted = 0R.
    """
    pnl = 0.0
    for pred, true in zip(y_pred, y_true):
        if pred == 0:
            continue   # model said no trade
        if pred == true:
            pnl += 2.0  # correct direction — 2R win
        elif true == 0:
            pnl -= 1.0  # model traded but label was no-trade → conservative loss
        else:
            pnl -= 1.0  # wrong direction → 1R loss
    return round(pnl, 2)


def train_pair(symbol: str, regime_lookup: RegimeLookup) -> None:
    symbol = symbol.upper()
    print(f"\n{'='*60}")
    print(f"Training signal models: {symbol}")
    print(f"{'='*60}")

    # Compute labels for full history
    df = compute_labels(symbol, regime_lookup)

    # Chronological split on the full dataset
    train_df, val_df, test_df = _chronological_split(df)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    summary = []

    for regime in range(N_REGIMES):
        # Filter each split to this regime only
        tr = train_df[train_df["h4_regime"] == regime]
        vl = val_df[val_df["h4_regime"] == regime]
        ts = test_df[test_df["h4_regime"] == regime]

        if len(tr) < MIN_SAMPLES:
            print(f"  Regime {regime}: only {len(tr)} train samples — skipping")
            continue

        X_tr = tr[FEATURE_COLS].values.astype(np.float32)
        y_tr = tr["label"].values.astype(np.int32)
        X_vl = vl[FEATURE_COLS].values.astype(np.float32)
        y_vl = vl["label"].values.astype(np.int32)

        # XGBoost mlogloss on val requires all classes present in val set
        use_eval = len(vl) >= MIN_SAMPLES and len(np.unique(y_vl)) == 3

        model = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="mlogloss",
            num_class=3,
            objective="multi:softprob",
            random_state=42,
            verbosity=0,
        )
        if use_eval:
            model.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], verbose=False)
        else:
            model.fit(X_tr, y_tr, verbose=False)

        val_pred  = model.predict(X_vl)
        val_metrics = _eval_metrics(y_vl, val_pred)
        val_pnl     = _simulated_pnl(y_vl, val_pred)

        # Save
        name       = f"{symbol}_regime{regime}"
        model_path = MODELS_DIR / f"{name}.pkl"
        meta_path  = MODELS_DIR / f"{name}.json"

        joblib.dump(model, model_path)

        label_counts = {
            "no_trade": int((y_tr == 0).sum()),
            "long":     int((y_tr == 1).sum()),
            "short":    int((y_tr == 2).sum()),
        }
        meta = {
            "symbol":        symbol,
            "regime":        regime,
            "feature_cols":  FEATURE_COLS,
            "train_samples": len(tr),
            "val_samples":   len(vl),
            "label_counts":  label_counts,
            "val_metrics":   val_metrics,
            "val_pnl_R":     val_pnl,
            "train_start":   str(tr.index[0]),
            "train_end":     str(tr.index[-1]),
            "trained_at":    datetime.now(timezone.utc).isoformat(),
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        print(
            f"  Regime {regime}: train={len(tr):>5} | val={len(vl):>4} | "
            f"acc={val_metrics['accuracy']:.3f} | "
            f"long_prec={val_metrics['long_precision']:.3f} | "
            f"short_prec={val_metrics['short_precision']:.3f} | "
            f"val_PnL={val_pnl:+.1f}R"
        )
        summary.append(meta)

    total_val_pnl = sum(m["val_pnl_R"] for m in summary)
    print(f"\n  Total val P&L across regimes: {total_val_pnl:+.1f}R")


def train_all(symbols: list[str] = None) -> None:
    if symbols is None:
        symbols = ALL_PAIRS

    print("Loading regime models (one-time)...")
    rl = RegimeLookup()

    for symbol in symbols:
        try:
            train_pair(symbol, rl)
        except Exception as e:
            print(f"  ERROR training {symbol}: {e}")

    print("\nAll done.")


def main():
    parser = argparse.ArgumentParser(description="Train per-regime signal models")
    parser.add_argument("--symbol", required=True, help="e.g. EURUSD or ALL")
    args = parser.parse_args()

    if args.symbol.upper() == "ALL":
        train_all()
    else:
        rl = RegimeLookup()
        train_pair(args.symbol.upper(), rl)


if __name__ == "__main__":
    main()
