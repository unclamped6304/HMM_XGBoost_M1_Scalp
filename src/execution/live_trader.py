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

import numpy as np

from src.config import (
    LIVE_PAIRS,
    CONFIDENCE_THRESHOLD,
    MAX_OPEN_TRADES,
    MAX_HOLD_BARS,
    SIGNAL_TIMEFRAME,
    MT5_LOGIN,
)
from src.execution.mt5_connector import (
    MAGIC,
    connect,
    disconnect,
    get_bars,
    get_our_positions,
    get_server_time,
    calculate_lot_size,
    place_order,
    close_position,
    modify_sltp,
)
from src.execution.live_regime import LiveRegimeDetector
from src.execution.risk_guard import RiskGuard
from src.execution import rollover_store
from src.signal.predict import SignalEngine
from src.signal.label import _atr, _adx, N_BARS_LIVE

# Bar timing per timeframe
_BAR_SECONDS = {"h1": 3600, "m15": 900, "m5": 300, "m1": 60}
BAR_SECONDS       = _BAR_SECONDS.get(SIGNAL_TIMEFRAME, 3600)
LOOP_SLEEP_S      = 10    # poll interval (seconds) — tighter for M15
BAR_BUFFER_S      = 3     # seconds after bar close to wait for MT5 confirmation
CHANDELIER_MULT   = 3.0   # ATR multiplier for chandelier trailing stop
ADX_TREND_DEATH   = 20.0  # exit when ADX falls below this (trend is dead)

# ── Rollover protection ────────────────────────────────────────────────────────
ROLLOVER_START = (23, 45)   # server time HH:MM to widen stops
ROLLOVER_END   = ( 0, 30)   # server time HH:MM to restore stops
SL_WIDEN_MULT  = 4.0        # SL distance multiplied by this during rollover
TP_WIDEN_MULT  = 2.0        # TP distance multiplied by this during rollover

# ── Friday close protection ────────────────────────────────────────────────────
FRIDAY_CUTOFF_HOUR = 20     # server time — no new trades on Friday at/after this hour
                            # forex closes ~00:00 server time (EET), so 20:00 = ~4h before

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
    """Force-close any position open for >= MAX_HOLD_BARS bars."""
    now_ts = datetime.now(timezone.utc).timestamp()
    for pos in positions:
        bars_open = (now_ts - pos["time"]) / BAR_SECONDS
        if bars_open >= MAX_HOLD_BARS:
            logging.info(
                f"[expire] {pos['symbol']} ticket={pos['ticket']} "
                f"open {bars_open:.1f} bars >= {MAX_HOLD_BARS} — closing"
            )
            close_position(pos["ticket"])


# ── Trend exit monitor ────────────────────────────────────────────────────────

def _check_trend_exits(positions: list[dict]) -> None:
    """
    Exit positions where the trend is falling apart.

    Two exit conditions (checked at each H1 bar close):
      1. ADX < ADX_TREND_DEATH  — trend has died (no directional force)
      2. Chandelier stop hit    — price crossed below highest_high - 3*ATR (long)
                                  or above lowest_low + 3*ATR (short)

    The chandelier stop only triggers once the trade is in profit (i.e., the
    chandelier level has moved past the entry price), to avoid premature exits
    when the trade hasn't had room to breathe.
    """
    now_ts = datetime.now(timezone.utc).timestamp()

    for pos in positions:
        symbol = pos["symbol"]
        try:
            bars_held = max(1, int((now_ts - pos["time"]) / BAR_SECONDS))
            n_fetch   = max(bars_held + 50, N_BARS_LIVE.get(SIGNAL_TIMEFRAME, 250))
            bars      = get_bars(symbol, SIGNAL_TIMEFRAME, n_fetch)
            if bars is None or len(bars) < 20:
                continue

            atr_s = _atr(bars)
            adx_s = _adx(bars)
            atr_val = float(atr_s.iloc[-1])
            adx_val = float(adx_s.iloc[-1])

            if np.isnan(atr_val) or atr_val <= 0:
                continue

            # ── ADX check: trend has died ──────────────────────────────────
            if not np.isnan(adx_val) and adx_val < ADX_TREND_DEATH:
                logging.info(
                    f"[{symbol}] Trend exit — ADX={adx_val:.1f} < {ADX_TREND_DEATH} "
                    f"| ticket={pos['ticket']}"
                )
                close_position(pos["ticket"])
                continue

            # ── Chandelier check: trailing stop ────────────────────────────
            current_close = float(bars["close"].iloc[-1])
            entry_price   = float(pos["price_open"])

            # Bars since position opened (index from end of fetched array)
            entry_bar_idx = max(0, len(bars) - bars_held - 1)
            since_entry   = bars.iloc[entry_bar_idx:]

            if pos["type"] == 0:   # Long (MT5 ORDER_TYPE_BUY = 0)
                highest_high    = float(since_entry["high"].max())
                chandelier_stop = highest_high - CHANDELIER_MULT * atr_val
                # Only trigger once chandelier has moved above entry (trade is profitable)
                if chandelier_stop > entry_price and current_close < chandelier_stop:
                    logging.info(
                        f"[{symbol}] Chandelier exit LONG — "
                        f"close={current_close:.5f} < stop={chandelier_stop:.5f} "
                        f"| ticket={pos['ticket']}"
                    )
                    close_position(pos["ticket"])
            else:                  # Short (MT5 ORDER_TYPE_SELL = 1)
                lowest_low      = float(since_entry["low"].min())
                chandelier_stop = lowest_low + CHANDELIER_MULT * atr_val
                if chandelier_stop < entry_price and current_close > chandelier_stop:
                    logging.info(
                        f"[{symbol}] Chandelier exit SHORT — "
                        f"close={current_close:.5f} > stop={chandelier_stop:.5f} "
                        f"| ticket={pos['ticket']}"
                    )
                    close_position(pos["ticket"])

        except Exception as e:
            logging.error(f"[{symbol}] Trend exit check error: {e}")


# ── Rollover protection ────────────────────────────────────────────────────────

def _in_rollover_window(server_dt: datetime) -> bool:
    """True if server time is between ROLLOVER_START and ROLLOVER_END."""
    t = (server_dt.hour, server_dt.minute)
    return t >= ROLLOVER_START or t < ROLLOVER_END


def _widen_stops(
    positions: list[dict],
    originals: dict[int, tuple[float, float]],
) -> None:
    """
    Widen SL (4×) and TP (2×) on all open positions for rollover protection.
    Saves original SL/TP in originals dict keyed by ticket so they can be restored.
    """
    for pos in positions:
        ticket     = pos["ticket"]
        entry      = float(pos["price_open"])
        orig_sl    = float(pos["sl"])
        orig_tp    = float(pos["tp"])

        if pos["type"] == 0:   # Long
            sl_dist = entry - orig_sl
            tp_dist = orig_tp - entry
            new_sl  = entry - sl_dist * SL_WIDEN_MULT
            new_tp  = entry + tp_dist * TP_WIDEN_MULT
        else:                  # Short
            sl_dist = orig_sl - entry
            tp_dist = entry - orig_tp
            new_sl  = entry + sl_dist * SL_WIDEN_MULT
            new_tp  = entry - tp_dist * TP_WIDEN_MULT

        if modify_sltp(ticket, round(new_sl, 6), round(new_tp, 6)):
            originals[ticket] = (orig_sl, orig_tp)
            rollover_store.save(MT5_LOGIN, ticket, orig_sl, orig_tp)
            logging.info(
                f"[rollover] {pos['symbol']} ticket={ticket} — "
                f"SL {orig_sl:.5f}→{new_sl:.5f}  TP {orig_tp:.5f}→{new_tp:.5f}"
            )


def _restore_stops(
    positions: list[dict],
    originals: dict[int, tuple[float, float]],
) -> None:
    """
    Restore original SL/TP after rollover window.
    Tickets that closed during rollover are silently skipped.
    """
    live_tickets = {pos["ticket"] for pos in positions}

    for ticket, (orig_sl, orig_tp) in list(originals.items()):
        if ticket not in live_tickets:
            # Trade closed during rollover (spread spike etc.) — nothing to restore
            logging.info(f"[rollover] ticket={ticket} closed during rollover — skipping restore")
            del originals[ticket]
            rollover_store.delete(MT5_LOGIN, ticket)
            continue

        if modify_sltp(ticket, orig_sl, orig_tp):
            symbol = next((p["symbol"] for p in positions if p["ticket"] == ticket), "?")
            logging.info(
                f"[rollover] {symbol} ticket={ticket} — SL/TP restored to {orig_sl:.5f} / {orig_tp:.5f}"
            )
            del originals[ticket]
            rollover_store.delete(MT5_LOGIN, ticket)


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

            # Fetch enough bars for all rolling features
            bars = get_bars(symbol, SIGNAL_TIMEFRAME, N_BARS_LIVE.get(SIGNAL_TIMEFRAME, 250))

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
    logging.info(f"HMM_XGBoost_{SIGNAL_TIMEFRAME.upper()}_TrendFollow — Live Trader starting")
    logging.info("=" * 60)

    rollover_store.ensure_table()

    if not connect():
        logging.error("Cannot connect to MT5 — aborting")
        return

    engine   = SignalEngine(timeframe=SIGNAL_TIMEFRAME)
    detector = LiveRegimeDetector()
    guard    = RiskGuard()

    logging.info(
        f"Timeframe: {SIGNAL_TIMEFRAME.upper()} | "
        f"Pairs: {len(LIVE_PAIRS)} | "
        f"Confidence threshold: {CONFIDENCE_THRESHOLD} | "
        f"Max open trades: {MAX_OPEN_TRADES} | "
        f"Risk per trade: 0.15%"
    )
    logging.info(f"Waiting for {SIGNAL_TIMEFRAME.upper()} bar close...")

    last_processed_hour  = -1
    rollover_active      = False
    rollover_originals: dict[int, tuple[float, float]] = rollover_store.load_all(MT5_LOGIN)
    if rollover_originals:
        logging.info(f"[rollover] Loaded {len(rollover_originals)} persisted originals from DB (restart recovery)")

    try:
        while True:
            now         = datetime.now(timezone.utc)
            server_now  = get_server_time()
            in_rollover = _in_rollover_window(server_now)

            # ── Rollover: widen stops on entry, restore on exit ─────────────
            # On Friday specifically: close all positions instead of widening
            try:
                open_positions = get_our_positions()
                if in_rollover and not rollover_active:
                    if server_now.weekday() == 4:   # Friday
                        logging.info(
                            f"[friday-close] Market closing — closing all "
                            f"{len(open_positions)} position(s) at server "
                            f"{server_now.strftime('%H:%M')}"
                        )
                        for pos in open_positions:
                            close_position(pos["ticket"])
                    else:
                        logging.info(
                            f"[rollover] Window opening at server "
                            f"{server_now.strftime('%H:%M')} — widening stops"
                        )
                        _widen_stops(open_positions, rollover_originals)
                    rollover_active = True

                elif not in_rollover and rollover_active:
                    logging.info(
                        f"[rollover] Window closing at server "
                        f"{server_now.strftime('%H:%M')} — restoring stops"
                    )
                    _restore_stops(open_positions, rollover_originals)
                    rollover_active = False
            except Exception as e:
                logging.error(f"[rollover] {e}")

            # ── Check for expired positions every loop ──────────────────────
            try:
                _close_expired(get_our_positions())
            except Exception as e:
                logging.error(f"[expire check] {e}")

            # ── Fire on bar close (timeframe-aware) ─────────────────────────
            bar_index    = int(now.timestamp()) // BAR_SECONDS
            at_bar_close = (
                (int(now.timestamp()) % BAR_SECONDS) >= BAR_BUFFER_S
                and bar_index != last_processed_hour
            )

            if at_bar_close:
                last_processed_hour = bar_index
                logging.info(
                    f"[bar] {SIGNAL_TIMEFRAME.upper()} bar closed at "
                    f"{now.strftime('%Y-%m-%d %H:%M UTC')} | "
                    f"server={server_now.strftime('%H:%M')} | "
                    f"rollover={'YES' if in_rollover else 'no'}"
                )
                try:
                    _check_trend_exits(get_our_positions())
                except Exception as e:
                    logging.error(f"[trend_exits] {e}")
                friday_cutoff = (
                    server_now.weekday() == 4
                    and server_now.hour >= FRIDAY_CUTOFF_HOUR
                )
                if friday_cutoff:
                    logging.info(
                        f"[friday] No new signals after {FRIDAY_CUTOFF_HOUR}:00 server time on Fridays"
                    )
                elif not in_rollover:
                    try:
                        _process_signals(engine, detector, guard)
                    except Exception as e:
                        logging.error(f"[process_signals] {e}")
                else:
                    logging.info("[rollover] Skipping new signals during rollover window")

            time.sleep(LOOP_SLEEP_S)

    except KeyboardInterrupt:
        logging.info("Shutdown requested — stopping gracefully")
    finally:
        disconnect()
        logging.info("MT5 disconnected. Bye.")


if __name__ == "__main__":
    run()
