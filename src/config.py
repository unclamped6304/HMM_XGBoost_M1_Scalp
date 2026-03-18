"""
config.py — Central configuration for the trading system.

Single source of truth for which pairs are active, confidence thresholds,
and other system-wide settings. Import this everywhere instead of
hardcoding lists.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Database ───────────────────────────────────────────────────────────────────
DB_HOST     = os.environ["DB_HOST"]
DB_NAME     = os.environ["DB_NAME"]
DB_USER     = os.environ["DB_USER"]
DB_PASSWORD = os.environ["DB_PASSWORD"]

# ── MT5 connection ─────────────────────────────────────────────────────────────
MT5_PATH  = os.environ["MT5_PATH"]
MT5_LOGIN = int(os.environ["MT5_LOGIN"])

# ── Signal timeframe ──────────────────────────────────────────────────────────
SIGNAL_TIMEFRAME = "m15"      # bar timeframe for signal models (h1, m15, m5)

# ── Active trading pairs ───────────────────────────────────────────────────────
# Selected based on M15 out-of-sample backtest (Sep 2024 – Mar 2026):
#   Tier 1 only: PF >= 2.0
#   3-state HMM, causal forward-only regime decoding (Mar 2026)

LIVE_PAIRS = [
    "NZDCAD",   #  +885R  WR=86.2%  PF=13.72  AvgR=+1.59
    "AUDNZD",   #  +843R  WR=86.3%  PF=13.68  AvgR=+1.59
    "GBPAUD",   #  +393R  WR=84.0%  PF=12.01  AvgR=+1.53
    "EURNZD",   #  +369R  WR=77.8%  PF= 7.66  AvgR=+1.34
    "GBPNZD",   #  +376R  WR=77.7%  PF= 7.65  AvgR=+1.33
    "EURCHF",   #  +775R  WR=76.8%  PF= 7.13  AvgR=+1.28
    "GBPCHF",   #  +882R  WR=76.2%  PF= 6.74  AvgR=+1.26
    "CADCHF",   #  +352R  WR=72.6%  PF= 5.84  AvgR=+1.18
    "AUDCHF",   #  +530R  WR=71.3%  PF= 5.09  AvgR=+1.13
    "NZDCHF",   #  +640R  WR=70.1%  PF= 5.08  AvgR=+1.11
]

# ── Signal model settings ─────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.70   # minimum XGBoost confidence to take a trade

# ── Risk settings ─────────────────────────────────────────────────────────────
RISK_PER_TRADE_PCT = 0.15     # % of account balance risked per trade (1R = this)
MAX_OPEN_TRADES    = 7        # max simultaneous open positions across all pairs
MIN_HOLD_BARS      = 4        # minimum M15 bars before closing (prop firm rule, ~1h)
MAX_HOLD_BARS      = 64       # maximum M15 bars before force-close (16h)

# ── ATR multiplier per H4 regime ─────────────────────────────────────────────
# Controls stop size: R = ATR(14) × multiplier
# All set to 1.5 — tune per-regime after live observation
ATR_MULT = {0: 1.5, 1: 1.5, 2: 1.5}
