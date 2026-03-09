"""
config.py — Central configuration for the trading system.

Single source of truth for which pairs are active, confidence thresholds,
and other system-wide settings. Import this everywhere instead of
hardcoding lists.
"""

# ── Active trading pairs ───────────────────────────────────────────────────────
# Selected based on out-of-sample backtest (Sep 2024 – Mar 2026):
#   - Positive profit factor
#   - Smooth, stable equity curves
#   - Excluded: NZDUSD, NZDCHF, GBPJPY, USDCAD (negative PF)
#   - Excluded: CADJPY, AUDNZD, GBPNZD, EURCAD, EURNZD, USDJPY, CADCHF (unstable)

LIVE_PAIRS = [
    "USDCHF",   # +139R  WR=46.8%  PF=2.72
    "EURUSD",   # +137R  WR=44.6%  PF=2.47
    "EURAUD",   # +105R  WR=41.3%  PF=2.48
    "NZDJPY",   #  +93R  WR=43.3%  PF=2.24
    "AUDUSD",   #  +65R  WR=37.4%  PF=1.68
    "EURCHF",   #  +63R  WR=36.7%  PF=1.54
    "GBPUSD",   #  +60R  WR=36.9%  PF=1.67
    "GBPCHF",   #  +56R  WR=35.2%  PF=1.52
    "AUDJPY",   #  +52R  WR=36.4%  PF=1.76
    "GBPAUD",   #  +50R  WR=38.4%  PF=2.09
    "AUDCAD",   #  +45R  WR=33.3%  PF=1.46
    "GBPCAD",   #  +45R  WR=38.8%  PF=1.65
    "AUDCHF",   #  +44R  WR=32.9%  PF=1.31
    "EURGBP",   #  +44R  WR=37.6%  PF=1.59
    "EURJPY",   #  +37R  WR=37.6%  PF=1.82
    "CHFJPY",   #  +35R  WR=35.5%  PF=1.54
]

# ── Signal model settings ─────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.70   # minimum XGBoost confidence to take a trade

# ── Risk settings ─────────────────────────────────────────────────────────────
RISK_PER_TRADE_PCT = 1.0      # % of account balance risked per trade (1R = this)
MAX_OPEN_TRADES    = 3        # max simultaneous open positions across all pairs
MIN_HOLD_BARS      = 1        # minimum H1 bars before closing (prop firm rule)
MAX_HOLD_BARS      = 16       # maximum H1 bars before force-close (16h)

# ── ATR multiplier per H4 regime ─────────────────────────────────────────────
# Controls stop size: R = ATR(14) × multiplier
# All set to 1.5 — tune per-regime after live observation
ATR_MULT = {0: 1.5, 1: 1.5, 2: 1.5, 3: 1.5, 4: 1.5, 5: 1.5, 6: 1.5}
