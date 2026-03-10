# AIPropFirmScalper — System Architecture

## Overview

An AI-driven quantitative trading platform running on a single Windows VPS.
Trades 10 selected forex pairs on Darwinex via MetaTrader 5, using a two-layer
HMM regime detector feeding per-regime XGBoost signal classifiers.

## Goals

- Generate consistent long-term returns on a Darwinex funded account
- Build a strong DARWIN signal track record to attract investor allocation
- Single codebase, minimal dependencies, easy to maintain solo

---

## Infrastructure

- **Single Windows VPS** — MT5 requires Windows; Python runs on the same machine
- **MetaTrader 5** — Darwinex-Live account; Python communicates directly via the
  official `MetaTrader5` Python package (no ZeroMQ, no MQL5 EA required)
- **PostgreSQL** — historical OHLCV data used for model training only

---

## Architecture Diagram

```
Windows VPS
│
│  MetaTrader 5 Terminal (Darwinex-Live)
│  └── MT5 Python package (IPC)
│         │
│         ▼
│  Python Live Trader  (src/execution/live_trader.py)
│  ├── LiveRegimeDetector  — H4 HMM + D1 currency HMM on live MT5 bars
│  ├── SignalEngine        — per-regime XGBoost classifiers
│  ├── MT5 Connector       — lot sizing, order placement, position management
│  └── RiskGuard           — session drawdown monitor (halt at 8%)
│
│  PostgreSQL  (training only — not used at runtime)
│  └── historicalData schema — OHLCV bars for all symbols/timeframes
```

---

## Data Flow

### Training (offline, run once or when retraining)
1. Historical OHLCV bars stored in PostgreSQL (imported via `scripts/import_csv.py`)
2. H4 per-pair HMM trained to detect 7 market regimes (`src/hmm/train.py`)
3. D1 per-currency HMM trained on currency strength (`src/hmm/train.py`)
4. H1 bars forward-labelled with 2:1 R:R outcomes (`src/signal/label.py`)
5. Per-regime XGBoost classifiers trained for each of 10 live pairs (`src/signal/train.py`)
6. Out-of-sample backtest validates pair selection (`src/signal/backtest.py`)

### Live Trading (runtime)
1. `live_trader.py` fires at each H1 bar close (top of every hour)
2. `LiveRegimeDetector` fetches recent H4/D1 bars from MT5 and runs the HMM models
3. `SignalEngine.predict_live()` runs the XGBoost model for the current regime
4. If signal confidence >= 0.70 and open trades < 3: size and place order
5. `RiskGuard` monitors session drawdown — halts at 8% from session high
6. Positions exceeding 16h are force-closed

---

## Signal Logic

- **Regime detection**: 7-state GaussianHMM on H4 bars per pair + D1 currency strength
- **Signal model**: XGBoost multi-class (0=No trade, 1=Long, 2=Short), one model per regime per pair
- **Entry**: market order at current ask/bid
- **Stop**: ATR(14) × 1.5 from entry
- **Target**: 2× stop distance (2:1 R:R)
- **Max hold**: 16 H1 bars (16 hours)

---

## Live Pairs (out-of-sample test: Sep 2024 – Mar 2026)

| Pair   | Total R | Win Rate | Profit Factor |
|--------|---------|----------|---------------|
| USDCHF | +139R   | 46.8%    | 2.72          |
| EURUSD | +137R   | 44.6%    | 2.47          |
| EURAUD | +105R   | 41.3%    | 2.48          |
| NZDJPY |  +93R   | 43.3%    | 2.24          |
| AUDUSD |  +65R   | 37.4%    | 1.68          |
| EURCHF |  +63R   | 36.7%    | 1.54          |
| GBPUSD |  +60R   | 36.9%    | 1.67          |
| GBPCHF |  +56R   | 35.2%    | 1.52          |
| AUDJPY |  +52R   | 36.4%    | 1.76          |
| GBPAUD |  +50R   | 38.4%    | 2.09          |

---

## Risk Parameters

| Parameter            | Value | Rationale                                        |
|----------------------|-------|--------------------------------------------------|
| Risk per trade       | 0.25% | Conservative — Darwinex applies D-Leverage on top |
| Max open trades      | 3     | Limits correlated exposure                       |
| Max hold time        | 16h   | Force-close after 16 H1 bars                    |
| Drawdown halt        | 8%    | 2% buffer before Darwinex 10% limit              |
| Confidence threshold | 0.70  | Tuned on EURUSD validation set                   |

---

## Technology Stack

| Component        | Technology                        |
|------------------|-----------------------------------|
| Execution        | MetaTrader 5 + MT5 Python package |
| Regime detection | hmmlearn (GaussianHMM, 7 states)  |
| Signal model     | XGBoost (multi-class classifier)  |
| Feature pipeline | pandas / numpy                    |
| Database         | PostgreSQL (training only)        |
| Language         | Python 3.11+                      |

---

## Build Phases

| Phase | Status   | Description                                                         |
|-------|----------|---------------------------------------------------------------------|
| 1     | Complete | PostgreSQL schema, historical data ingestion (10 years, 32 symbols) |
| 2     | Complete | HMM regime detection, XGBoost signal models, backtesting, pair selection |
| 3     | Complete | Live execution bridge via MT5 Python package (Darwinex)             |
| 4     | Pending  | Live monitoring, performance tracking, scaling                      |

---

## Project Structure

```
src/
├── config.py                  # Central config — pairs, thresholds, risk params
├── hmm/
│   ├── features.py            # OHLCV feature engineering + DB loader
│   ├── train.py               # HMM training (H4 pair + D1 currency)
│   ├── regime.py              # RegimeLookup — historical regime labelling
│   └── visualise.py           # Regime chart visualisation
├── signal/
│   ├── label.py               # Forward-looking 2:1 R:R trade labeller
│   ├── train.py               # Per-regime XGBoost training
│   ├── predict.py             # SignalEngine — backtest + live inference
│   └── backtest.py            # Out-of-sample backtester
└── execution/
    ├── mt5_connector.py       # MT5 connection, bars, orders, lot sizing
    ├── live_regime.py         # Live HMM regime detection from MT5 bars
    ├── risk_guard.py          # Session drawdown monitor
    └── live_trader.py         # Main live trading loop

scripts/
└── import_csv.py              # Historical data import (Phase 1 / retraining)

ddl/
└── historicalData/            # PostgreSQL table definitions per symbol
```
