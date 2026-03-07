# AIPropFirmScalper — System Architecture

## Overview

A quant trading platform for prop firm trading (FTMO and others), running on a single Windows VPS.
Designed for simplicity, clean separation of concerns, and easy extension to multiple prop firm accounts.

## Goals

- Generate consistent long-term income across multiple prop firms
- Single codebase, multiple funded accounts trading identical signals
- Each prop firm account enforces its own risk rules locally
- Simple enough to run and maintain solo; structured enough to extend

---

## Infrastructure

- **Single Windows VPS** — MT5 requires Windows; Python runs on the same machine
- **Multiple MT5 terminals** — one per prop firm account
- **PostgreSQL** — historical OHLCV data + trade logging
- **ZeroMQ** — low-latency socket messaging between MT5 and Python - http://wiki.zeromq.org/docs:windows-installations

---

## Architecture Diagram

```
Windows VPS
│
│  MT5 Terminal #1 (FTMO)
│  ├── Data EA        ──[ZMQ PUB]──────────────────────────┐
│  └── Execution EA   ◄─[ZMQ SUB]──────────────────┐       │
│                                                   │       │
│  MT5 Terminal #2 (Funded Next)                    │       ▼
│  └── Execution EA   ◄─[ZMQ SUB]───────────  Python Signal Engine
│                                                   │  ├── ZMQ SUB: live bars
│  MT5 Terminal #3 (Apex / other)                   │  ├── Feature pipeline
│  └── Execution EA   ◄─[ZMQ SUB]───────────────────┘  ├── ML regime classifier
│                                                       ├── Signal logic
│  PostgreSQL                                           └── ZMQ PUB: signals
│  ├── ohlcv_1m  (historical + live bars)
│  └── trades    (fills, P&L, performance)
```

---

## Data Flow

### Market Data (MT5 → Python)
1. Data EA in MT5 Terminal #1 streams live 1M bars via ZMQ PUB socket
2. Python subscribes to the feed via ZMQ SUB
3. Python stores bars to PostgreSQL and maintains a rolling in-memory feature window

### Signal Generation (Python internal)
1. Feature pipeline computes indicators on incoming bars (ATR, vol, autocorrelation, session flags, etc.)
2. ML regime classifier determines current market state (trending / mean-reverting / avoid)
3. Signal logic generates entry/exit signals gated by regime
4. Position sizing calculated per account risk rules

### Signal Delivery (Python → MT5)
1. Python publishes signals via ZMQ PUB socket
2. All Execution EAs across all MT5 terminals subscribe to the same feed
3. Each EA applies its own firm-specific risk rules before placing orders

---

## Component Responsibilities

### Data EA (MQL5)
- Streams live OHLCV bars to Python via ZMQ PUB
- No strategy logic — data pipe only
- Runs in MT5 Terminal #1

### Python Signal Engine
- Subscribes to live bar feed (ZMQ SUB)
- Runs feature pipeline and ML model
- Generates and publishes trading signals (ZMQ PUB)
- Logs all signals and decisions to PostgreSQL
- All intelligence lives here

### Execution EA (MQL5)
- Subscribes to Python signal feed (ZMQ SUB)
- Places, modifies, and closes orders via MT5
- Enforces firm-specific hard risk rules:
  - Daily loss limit
  - Max drawdown
  - Max position size
- Reports fills back to Python (ZMQ PUB) — future
- One instance per prop firm account

### PostgreSQL
- `ohlcv_1m` — historical 1M bars (seeded from external data source, appended live)
- `trades` — trade log: entry, exit, P&L, account, strategy version

---

## ZeroMQ Topology

| Socket | Pattern | Direction | Purpose |
|--------|---------|-----------|---------|
| Data feed | PUB/SUB | MT5 → Python | Live bar stream |
| Signal feed | PUB/SUB | Python → MT5 | Trading signals to all EAs |

One publisher per feed. Any number of subscribers. Adding a new prop firm account
requires no changes to Python — just deploy another MT5 terminal + Execution EA.

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Execution & data feed | MQL5 / MetaTrader 5 |
| Signal engine & ML | Python 3.11+ |
| Messaging | ZeroMQ (Darwinex connector for MQL5) |
| Database | PostgreSQL |
| ML models | scikit-learn / XGBoost |

---

## Build Phases

### Phase 1 — Data Foundation
- Set up PostgreSQL schema
- Ingest historical 1M data
- Build feature pipeline

### Phase 2 — Research
- Label historical regimes
- Train and walk-forward validate regime classifier
- Define and backtest entry/exit strategy

### Phase 3 — Execution Bridge
- Build MQL5 Data EA (ZMQ PUB)
- Build Python signal engine (ZMQ SUB/PUB)
- Build MQL5 Execution EA (ZMQ SUB + FTMO risk rules)
- Paper trade on FTMO demo

### Phase 4 — Production
- Connect to live FTMO challenge account
- Monitor P&L, drawdown, signal quality
- Add additional prop firm terminals as accounts are funded

---

## Prop Firm Scaling Model

```
Stage 1: Pass FTMO Challenge (demo → funded $10K)
Stage 2: Stack challenges — FTMO + Funded Next + Apex simultaneously
Stage 3: Multiple funded accounts, identical signals, firm-specific risk rules
Stage 4: Scale within firms (FTMO scales profitable accounts up to $2M)
```

Risk is isolated per firm — one account blowing up does not affect others.
