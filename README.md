# AIPropFirmScalper

An AI-driven quantitative trading platform for prop firm trading (FTMO and others), running on a single Windows VPS. See [ARCHITECTURE.md](ARCHITECTURE.md) for full system design.

---

## Prerequisites

- Windows VPS
- [PostgreSQL 18+](https://www.postgresql.org/download/windows/)
- [Python 3.11+](https://www.python.org/downloads/)
- [Node.js](https://nodejs.org/) (for ZeroMQ bindings)
- [MetaTrader 5](https://www.metatrader5.com/)

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/unclamped6304/AIPropFirmScalper.git
cd AIPropFirmScalper
```

### 2. Install Node dependencies (ZeroMQ)

```bash
npm install
```

### 3. Install Python dependencies

```bash
pip install psycopg2-binary
```

### 4. Configure environment

Copy `.env.example` to `.env` and fill in your database credentials:

```bash
cp .env.example .env
```

```env
DB_HOST=localhost
DB_NAME=postgres
DB_USER=postgres
DB_PASSWORD=your_password_here
```

### 5. Create the database schema

Run the DDL scripts for each instrument you need. The scripts are organised by symbol and timeframe under `ddl/historicalData/`.

Example — create all EURUSD tables:

```bash
for f in ddl/historicalData/EURUSD/*.sql; do
  psql -h localhost -U postgres -d postgres -c "SET search_path TO \"historicalData\";" -f "$f"
done
```

---

## Importing Historical Data

Historical OHLCV data is exported from MetaTrader 5 via [Tick Data Suite](https://www.tickdatasuite.com/) at 99% modelling quality, GMT+2 with US DST.

CSV files follow the naming convention:

```
{SYMBOL}_GMT+2_US-DST__{date_range}_{TIMEFRAME}.csv
```

To import a CSV file into its corresponding table:

```bash
python scripts/import_csv.py "path/to/EURUSD_GMT+2_US-DST__3-7-2016_3-6-2026_M1.csv"
```

The script automatically resolves the target table from the filename, skips the source header, and imports in batches of 10,000 rows.

---

## Project Structure

```
├── ddl/
│   └── historicalData/      # Table definitions per symbol and timeframe
│       ├── EURUSD/
│       ├── GBPUSD/
│       └── ...
├── scripts/
│   └── import_csv.py        # Historical data import tool
├── .env.example             # Environment variable template
├── ARCHITECTURE.md          # Full system architecture
└── README.md
```

---

## Build Phases

| Phase | Status | Description |
|-------|--------|-------------|
| 1 — Data Foundation | 🔄 In Progress | PostgreSQL schema, historical data ingestion, feature pipeline |
| 2 — Research | Pending | Regime labelling, model training, strategy backtesting |
| 3 — Execution Bridge | Pending | MQL5 Data EA, Python signal engine, MQL5 Execution EA |
| 4 — Production | Pending | Live FTMO challenge, scaling to multiple prop firms |
