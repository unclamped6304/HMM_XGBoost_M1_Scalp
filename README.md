# AIPropFirmScalper

An AI-driven quantitative trading platform targeting Darwinex, running on a single Windows VPS.
Uses a two-layer HMM regime detector feeding per-regime XGBoost signal classifiers across 10 forex pairs.

See [ARCHITECTURE.md](ARCHITECTURE.md) for full system design.

---

## Prerequisites

- Windows VPS
- [PostgreSQL](https://www.postgresql.org/download/windows/) (for model training)
- [Python 3.11+](https://www.python.org/downloads/)
- [MetaTrader 5](https://www.metatrader5.com/) logged in to Darwinex-Live

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/unclamped6304/AIPropFirmScalper.git
cd AIPropFirmScalper
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
```

Edit `.env` with your PostgreSQL credentials:

```env
DB_HOST=localhost
DB_NAME=postgres
DB_USER=postgres
DB_PASSWORD=your_password_here
```

---

## Training the Models (Phase 1 & 2)

Only needed on first setup or when retraining on fresh data.

### Import historical data

Export OHLCV CSVs from MT5 (M1 timeframe) then import:

```bash
python scripts/import_csv.py "path/to/EURUSD_GMT+2_US-DST__3-7-2016_3-6-2026_M1.csv"
```

### Train HMM regime models

```bash
python -m src.hmm.train --mode all
```

### Train XGBoost signal models

```bash
python -m src.signal.train --symbol ALL
```

### Run out-of-sample backtest

```bash
python -m src.signal.backtest --symbol ALL --plot
```

---

## Live Trading (Phase 3)

Make sure MT5 is open and logged in to Darwinex-Live, then:

```bash
python -m src.execution.live_trader
```

Logs are written to `logs/live_trader.log`.

The trader fires at each H1 bar close, runs regime detection and signal models
for all 10 pairs, and places orders with SL/TP via the MT5 Python package.
Trading halts automatically if session drawdown reaches 8%.

---

## Project Structure

```
src/
├── config.py          # Central config — pairs, thresholds, risk params
├── hmm/               # HMM regime detection (H4 pair + D1 currency)
├── signal/            # XGBoost signal models + backtester
└── execution/         # Live trading bridge (MT5 Python package)

scripts/
└── import_csv.py      # Historical data import

ddl/
└── historicalData/    # PostgreSQL table definitions
```
