"""
train.py — Train 7-state HMM regime models and save to disk.

Two modes:
  1. Per-pair H4 model  (Layer 1) — one model per trading pair
  2. Per-currency D1 model (Layer 2) — one model per major currency

Models are saved to models/hmm/ as joblib pickle files.
A metadata JSON is saved alongside each model with:
  - train/val log-likelihood
  - date range of training data
  - feature list
  - n_components

Usage:
  # Train a single pair
  python -m src.hmm.train --mode pair --symbol EURUSD

  # Train all pairs
  python -m src.hmm.train --mode pair --symbol ALL

  # Train a single currency
  python -m src.hmm.train --mode currency --symbol EUR

  # Train all currencies
  python -m src.hmm.train --mode currency --symbol ALL
"""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

from src.hmm.features import (
    CURRENCY_PAIRS,
    load_currency_features,
    load_features,
)

N_STATES        = 7
N_ITER          = 200
RANDOM_STATE    = 42
# Probability of staying in the current state — biases the HMM toward
# persistent regimes rather than bar-by-bar flipping.
# 0.90 means ~10% chance of switching per bar on average.
SELF_LOOP_PROB  = 0.90
MODELS_DIR    = Path("models/hmm")

ALL_PAIRS = [
    "AUDCAD", "AUDCHF", "AUDJPY", "AUDNZD", "AUDUSD",
    "CADCHF", "CADJPY", "CHFJPY",
    "EURAUD", "EURCAD", "EURCHF", "EURGBP", "EURJPY", "EURNZD", "EURUSD",
    "GBPAUD", "GBPCAD", "GBPCHF", "GBPJPY", "GBPNZD", "GBPUSD",
    "NZDCAD", "NZDCHF", "NZDJPY", "NZDUSD",
    "USDCAD", "USDCHF", "USDJPY",
]
ALL_CURRENCIES = list(CURRENCY_PAIRS.keys())


# ─── CORE ─────────────────────────────────────────────────────────────────────

def train_hmm(
    train_data: np.ndarray,
    val_data: np.ndarray,
) -> tuple[GaussianHMM, StandardScaler, float, float]:
    """
    Fit a GaussianHMM on train_data, score on val_data.
    Data is StandardScaled before fitting.

    Returns:
        model, scaler, train_score, val_score
    """
    scaler     = StandardScaler()
    train_scaled = scaler.fit_transform(train_data)
    val_scaled   = scaler.transform(val_data)

    model = GaussianHMM(
        n_components=N_STATES,
        covariance_type="full",
        n_iter=N_ITER,
        random_state=RANDOM_STATE,
        verbose=False,
        params="stmc",   # train: start, transition, means, covariances
        init_params="smc",  # initialise everything EXCEPT transition matrix
    )

    # Initialise transition matrix with high self-loop probability so the
    # model starts biased toward regime persistence and the EM algorithm
    # refines from there rather than discovering chaotic switching.
    off_diag = (1.0 - SELF_LOOP_PROB) / (N_STATES - 1)
    transmat  = np.full((N_STATES, N_STATES), off_diag)
    np.fill_diagonal(transmat, SELF_LOOP_PROB)
    model.transmat_ = transmat

    model.fit(train_scaled)

    train_score = model.score(train_scaled)
    val_score   = model.score(val_scaled)

    return model, scaler, train_score, val_score


def save_model(
    name: str,
    model: GaussianHMM,
    scaler: StandardScaler,
    feature_names: list[str],
    train_score: float,
    val_score: float,
    train_start: str,
    train_end: str,
) -> None:
    """Save model + scaler as a single joblib file, plus a JSON metadata sidecar."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / f"{name}.pkl"
    meta_path  = MODELS_DIR / f"{name}.json"

    joblib.dump({"model": model, "scaler": scaler}, model_path)

    meta = {
        "name":         name,
        "n_states":     N_STATES,
        "features":     feature_names,
        "train_score":  round(train_score, 4),
        "val_score":    round(val_score, 4),
        "train_start":  train_start,
        "train_end":    train_end,
        "trained_at":   datetime.now(timezone.utc).isoformat(),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Saved  -> {model_path}")
    print(f"  Meta   -> {meta_path}")


def load_model(name: str) -> tuple[GaussianHMM, StandardScaler, dict]:
    """Load a saved model + scaler + metadata by name."""
    model_path = MODELS_DIR / f"{name}.pkl"
    meta_path  = MODELS_DIR / f"{name}.json"

    bundle = joblib.load(model_path)
    with open(meta_path) as f:
        meta = json.load(f)

    return bundle["model"], bundle["scaler"], meta


# ─── PAIR TRAINING ────────────────────────────────────────────────────────────

def train_pair(symbol: str) -> None:
    print(f"\n[train] Pair: {symbol} H4")
    try:
        train, val, _ = load_features(symbol, "h4", split=True)
    except Exception as e:
        print(f"  ERROR loading features: {e}")
        return

    feature_names = list(train.columns)
    model, scaler, train_score, val_score = train_hmm(
        train.values, val.values
    )

    print(f"  Train log-likelihood: {train_score:.4f}")
    print(f"  Val   log-likelihood: {val_score:.4f}")

    save_model(
        name          = f"{symbol}_h4",
        model         = model,
        scaler        = scaler,
        feature_names = feature_names,
        train_score   = train_score,
        val_score     = val_score,
        train_start   = str(train.index[0]),
        train_end     = str(train.index[-1]),
    )


# ─── CURRENCY TRAINING ────────────────────────────────────────────────────────

def train_currency(currency: str) -> None:
    print(f"\n[train] Currency: {currency} D1")
    try:
        train, val, _ = load_currency_features(currency, split=True)
    except Exception as e:
        print(f"  ERROR loading features: {e}")
        return

    feature_names = list(train.columns)
    model, scaler, train_score, val_score = train_hmm(
        train.values, val.values
    )

    print(f"  Train log-likelihood: {train_score:.4f}")
    print(f"  Val   log-likelihood: {val_score:.4f}")

    save_model(
        name          = f"{currency}_d1",
        model         = model,
        scaler        = scaler,
        feature_names = feature_names,
        train_score   = train_score,
        val_score     = val_score,
        train_start   = str(train.index[0]),
        train_end     = str(train.index[-1]),
    )


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train HMM regime models")
    parser.add_argument(
        "--mode", choices=["pair", "currency"], required=True,
        help="pair = H4 per-pair model, currency = D1 currency strength model",
    )
    parser.add_argument(
        "--symbol", required=True,
        help="Symbol/currency to train (e.g. EURUSD, EUR) or ALL",
    )
    args = parser.parse_args()

    if args.mode == "pair":
        targets = ALL_PAIRS if args.symbol.upper() == "ALL" else [args.symbol.upper()]
        for symbol in targets:
            train_pair(symbol)

    elif args.mode == "currency":
        targets = ALL_CURRENCIES if args.symbol.upper() == "ALL" else [args.symbol.upper()]
        for currency in targets:
            train_currency(currency)

    print("\nDone.")


if __name__ == "__main__":
    main()
