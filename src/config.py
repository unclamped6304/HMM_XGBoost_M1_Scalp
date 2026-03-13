"""
config.py — Central configuration for the trading system.

Single source of truth for which pairs are active, confidence thresholds,
and other system-wide settings. Import this everywhere instead of
hardcoding lists.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── MT5 connection ─────────────────────────────────────────────────────────────
MT5_PATH     = r"C:\Program Files\Darwinex MetaTrader 5\terminal64.exe"
MT5_LOGIN = int(os.environ["MT5_LOGIN"])

# ── Signal timeframe ──────────────────────────────────────────────────────────
SIGNAL_TIMEFRAME = "m15"      # bar timeframe for signal models (h1, m15, m5)

# ── Active trading pairs ───────────────────────────────────────────────────────
# Selected based on M15 out-of-sample backtest (Sep 2024 – Mar 2026):
#   Tier 1 only: PF >= 2.0, stable equity curves

LIVE_PAIRS = [
    "AUDNZD",   # +1187R  WR=59.3%  PF=2.99  AvgR=+0.77
    "CHFJPY",   # +1357R  WR=58.5%  PF=2.86  AvgR=+0.74
    "GBPCAD",   #  +690R  WR=59.3%  PF=2.92  AvgR=+0.75
    "GBPNZD",   # +1251R  WR=53.5%  PF=2.35  AvgR=+0.61
    "GBPCHF",   # +1378R  WR=54.6%  PF=2.39  AvgR=+0.61
    "EURNZD",   #  +830R  WR=52.2%  PF=2.24  AvgR=+0.57
    "AUDCAD",   # +1115R  WR=51.7%  PF=2.16  AvgR=+0.53
    "EURCHF",   # +1048R  WR=51.1%  PF=2.14  AvgR=+0.52
    "EURAUD",   # +1102R  WR=50.5%  PF=2.10  AvgR=+0.51
    "AUDCHF",   # +1479R  WR=50.4%  PF=2.07  AvgR=+0.51
    "EURGBP",   #  +420R  WR=49.8%  PF=2.00  AvgR=+0.48
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
ATR_MULT = {0: 1.5, 1: 1.5, 2: 1.5, 3: 1.5, 4: 1.5, 5: 1.5, 6: 1.5}
