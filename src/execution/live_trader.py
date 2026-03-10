"""
live_trader.py — Main live trading loop.

Fires at each H1 bar close (top of every hour + 5s buffer to ensure MT5
has confirmed the bar). For each pair in LIVE_PAIRS:
  1. Skip if already in a trade for that pair
  2. Detect live regime via H4 HMM + D1 currency HMM
  3. Run XGBoost signal model
  4. If signal passes confidence threshold and MAX_OPEN_TRADES not reached:
     - Calculate lot size (0.25% account risk)
     - Place market order with SL and TP

Also checks every loop iteration for positions that have exceeded
MAX_HOLD_BARS (16h) and force-closes them.

Usage:
  python -m src.execution.live_trader
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from src.config import (
    LIVE_PAIRS,
    CONFIDENCE_THRESHOLD,
    MAX_OPEN_TRADES,
    MAX_HOLD_BARS,
)
from src.execution.mt5_connector import (
    MAGIC,
    connect,
    disconnect,
    get_bars,
    get_our_positions,
    calculate_lot_size,
    place_order,
    close_position,
)
from src.execution.live_regime import LiveRegimeDetector
from src.execution.risk_guard import RiskGuard
from src.signal.predict import SignalEngine

H1_SECONDS    = 3600
LOOP_SLEEP_S  = 30    # poll interval (seconds)
BAR_BUFFER_S  = 5     # seconds after the hour to wait for bar to confirm

Path("logs").mkdir(exist_ok=True)


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/live_trader.log", encoding="utf-8"),
        ],
    )


# ── Position lifecycle ────────────────────────────────────────────────────────

def _close_expired(positions: list[dict]) -> None:
    """Force-close any position open for >= MAX_HOLD_BARS hours."""
    now_ts = datetime.now(timezone.utc).timestamp()
    for pos in positions:
        bars_open = (now_ts - pos["time"]) / H1_SECONDS
        if bars_open >= MAX_HOLD_BARS:
            logging.info(
                f"[expire] {pos['symbol']} ticket={pos['ticket']} "
                f"open {bars_open:.1f}h >= {MAX_HOLD_BARS}h — closing"
            )
            close_position(pos["ticket"])


# ── Signal processing ─────────────────────────────────────────────────────────

def _process_signals(
    engine:   SignalEngine,
    detector: LiveRegimeDetector,
    guard:    RiskGuard,
) -> None:

    if not guard.check():
        logging.warning("[trader] Risk guard active — no new trades")
        return

    positions       = get_our_positions()
    pairs_in_trade  = {p["symbol"] for p in positions}
    open_count      = len(positions)

    if open_count >= MAX_OPEN_TRADES:
        logging.info(f"[trader] {open_count}/{MAX_OPEN_TRADES} trades open — skipping")
        return

    for symbol in LIVE_PAIRS:
        if open_count >= MAX_OPEN_TRADES:
            break

        if symbol in pairs_in_trade:
            continue

        try:
            # Live regime context
            regime_ctx = detector.get(symbol)
            if regime_ctx["pair_regime"] < 0:
                logging.debug(f"[{symbol}] No regime — skipping")
                continue

            # Last 80 H1 bars for feature computation
            bars = get_bars(symbol, "h1", 80)

            # Signal
            sig = engine.predict_live(symbol, bars, regime_ctx)

            if sig["signal"] == "none":
                continue
            if sig["confidence"] < CONFIDENCE_THRESHOLD:
                continue
            if sig["R"] <= 0:
                continue

            # Lot size
            lots = calculate_lot_size(symbol, sig["R"])

            # Place order
            place_order(
                symbol    = symbol,
                direction = sig["signal"],
                stop      = sig["stop"],
                target    = sig["target"],
                lots      = lots,
            )

            open_count      += 1
            pairs_in_trade.add(symbol)

            logging.info(
                f"[{symbol}] {sig['signal'].upper()} | "
                f"regime={sig['regime']} | conf={sig['confidence']:.3f} | "
                f"R={sig['R']:.5f} | lots={lots} | "
                f"dd={guard.drawdown_pct:.2f}%"
            )

        except Exception as e:
            logging.error(f"[{symbol}] Error: {e}")


# ── Main loop ─────────────────────────────────────────────────────────────────

def run() -> None:
    _setup_logging()
    logging.info("=" * 60)
    logging.info("AIPropFirmScalper — Live Trader starting")
    logging.info("=" * 60)

    if not connect():
        logging.error("Cannot connect to MT5 — aborting")
        return

    engine   = SignalEngine()
    detector = LiveRegimeDetector()
    guard    = RiskGuard()

    logging.info(
        f"Pairs: {len(LIVE_PAIRS)} | "
        f"Confidence threshold: {CONFIDENCE_THRESHOLD} | "
        f"Max open trades: {MAX_OPEN_TRADES} | "
        f"Risk per trade: 0.25%"
    )
    logging.info("Waiting for H1 bar close...")

    last_processed_hour = -1

    try:
        while True:
            now = datetime.now(timezone.utc)

            # ── Check for expired positions every loop ──────────────────────
            try:
                _close_expired(get_our_positions())
            except Exception as e:
                logging.error(f"[expire check] {e}")

            # ── Fire on H1 bar close (top of hour + buffer) ─────────────────
            at_bar_close = (
                now.minute == 0
                and now.second >= BAR_BUFFER_S
                and now.hour != last_processed_hour
            )

            if at_bar_close:
                last_processed_hour = now.hour
                logging.info(
                    f"[bar] H1 bar closed at {now.strftime('%Y-%m-%d %H:%M UTC')}"
                )
                try:
                    _process_signals(engine, detector, guard)
                except Exception as e:
                    logging.error(f"[process_signals] {e}")

            time.sleep(LOOP_SLEEP_S)

    except KeyboardInterrupt:
        logging.info("Shutdown requested — stopping gracefully")
    finally:
        disconnect()
        logging.info("MT5 disconnected. Bye.")


if __name__ == "__main__":
    run()
