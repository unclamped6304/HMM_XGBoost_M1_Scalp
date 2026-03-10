"""
mt5_connector.py — MetaTrader5 connection and market operations.

Wraps the MetaTrader5 Python package for bar fetching, account queries,
lot size calculation, order placement, and position management.

All monetary values are in the account's deposit currency.
"""

from __future__ import annotations

import logging

import MetaTrader5 as mt5
import pandas as pd

from src.config import RISK_PER_TRADE_PCT, MT5_PATH, MT5_LOGIN

log = logging.getLogger(__name__)

MAGIC = 20260309   # EA identifier — used to track our own positions

_TF_MAP = {
    "m1":  mt5.TIMEFRAME_M1,
    "m5":  mt5.TIMEFRAME_M5,
    "m15": mt5.TIMEFRAME_M15,
    "h1":  mt5.TIMEFRAME_H1,
    "h4":  mt5.TIMEFRAME_H4,
    "d1":  mt5.TIMEFRAME_D1,
}


# ── Connection ────────────────────────────────────────────────────────────────

def connect() -> bool:
    """Initialise MT5 and verify account connection."""
    if not mt5.initialize(path=MT5_PATH):
        log.error(f"MT5 initialize failed: {mt5.last_error()}")
        return False

    info = mt5.account_info()
    if info is None:
        log.error(f"Cannot get account info: {mt5.last_error()}")
        return False

    if info.login != MT5_LOGIN:
        log.error(f"Wrong account: connected to {info.login}, expected {MT5_LOGIN}")
        return False

    log.info(
        f"Connected: login={info.login} | server={info.server} | "
        f"balance={info.balance:.2f} {info.currency}"
    )
    return True


def disconnect() -> None:
    mt5.shutdown()


# ── Market data ───────────────────────────────────────────────────────────────

def get_bars(symbol: str, timeframe: str, n: int) -> pd.DataFrame:
    """
    Return the last n *closed* bars as a DataFrame.

    Uses pos=1 to skip the currently forming bar.
    Columns: open, high, low, close, tick_volume
    Index:   datetime (UTC)
    """
    tf    = _TF_MAP[timeframe.lower()]
    rates = mt5.copy_rates_from_pos(symbol, tf, 1, n)   # pos=1 skips forming bar

    if rates is None or len(rates) == 0:
        raise RuntimeError(
            f"No bars for {symbol} {timeframe}: {mt5.last_error()}"
        )

    df = pd.DataFrame(rates)
    df["datetime"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_localize(None)
    df = df.set_index("datetime")[["open", "high", "low", "close", "tick_volume"]]
    return df.astype(float)


# ── Account ───────────────────────────────────────────────────────────────────

def get_account_balance() -> float:
    info = mt5.account_info()
    if info is None:
        raise RuntimeError(f"Cannot get account info: {mt5.last_error()}")
    return float(info.balance)


def get_our_positions() -> list[dict]:
    """Return open positions placed by this EA (filtered by magic number)."""
    positions = mt5.positions_get()
    if positions is None:
        return []
    return [p._asdict() for p in positions if p.magic == MAGIC]


# ── Position sizing ───────────────────────────────────────────────────────────

def calculate_lot_size(symbol: str, stop_distance: float) -> float:
    """
    Calculate lot size so that the stop loss costs exactly RISK_PER_TRADE_PCT
    of current account balance.

    Args:
        symbol:        e.g. "EURUSD"
        stop_distance: stop loss in price units (the R value from the signal)

    Returns:
        Lot size, clamped to broker min/max and rounded to volume_step.
    """
    balance  = get_account_balance()
    risk_amt = balance * RISK_PER_TRADE_PCT / 100.0

    info = mt5.symbol_info(symbol)
    if info is None:
        raise RuntimeError(f"Symbol info not found: {symbol}")

    # Monetary cost of stop per 1 lot
    stop_ticks    = stop_distance / info.trade_tick_size
    risk_per_lot  = stop_ticks * info.trade_tick_value

    if risk_per_lot <= 0:
        raise ValueError(f"Cannot compute lot size for {symbol}: risk_per_lot={risk_per_lot}")

    raw_lots = risk_amt / risk_per_lot

    # Round to broker's volume step and clamp to allowed range
    step = info.volume_step
    lots = round(round(raw_lots / step) * step, 8)
    lots = max(info.volume_min, min(info.volume_max, lots))

    return round(lots, 2)


# ── Order management ──────────────────────────────────────────────────────────

def _filling_type(symbol: str) -> int:
    """Pick the first supported filling mode for the symbol."""
    info = mt5.symbol_info(symbol)
    mode = info.filling_mode if info else 0
    if mode & 1:
        return mt5.ORDER_FILLING_FOK
    if mode & 2:
        return mt5.ORDER_FILLING_IOC
    return mt5.ORDER_FILLING_RETURN


def place_order(
    symbol:    str,
    direction: str,   # "long" or "short"
    stop:      float,
    target:    float,
    lots:      float,
) -> dict:
    """
    Place a market order with SL and TP.
    Uses the current ask/bid as entry price (more accurate than signal entry).

    Returns dict with ticket, symbol, direction, lots, entry, stop, target.
    Raises RuntimeError on failure.
    """
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        raise RuntimeError(f"Cannot get tick for {symbol}: {mt5.last_error()}")

    if direction == "long":
        order_type = mt5.ORDER_TYPE_BUY
        price      = tick.ask
    else:
        order_type = mt5.ORDER_TYPE_SELL
        price      = tick.bid

    request = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       symbol,
        "volume":       lots,
        "type":         order_type,
        "price":        price,
        "sl":           stop,
        "tp":           target,
        "deviation":    20,
        "magic":        MAGIC,
        "comment":      "HMM_XGBoost_H1_Swing",
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": _filling_type(symbol),
    }

    result = mt5.order_send(request)
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        code = result.retcode if result else "None"
        raise RuntimeError(
            f"Order failed {symbol} {direction}: retcode={code} | {mt5.last_error()}"
        )

    log.info(
        f"Order placed: {symbol} {direction.upper()} | "
        f"lots={lots} | entry={result.price} | sl={stop} | tp={target} | ticket={result.order}"
    )
    return {
        "ticket":    result.order,
        "symbol":    symbol,
        "direction": direction,
        "lots":      lots,
        "entry":     result.price,
        "stop":      stop,
        "target":    target,
    }


def close_position(ticket: int) -> bool:
    """
    Close an open position by ticket number.
    Returns True on success.
    """
    positions = mt5.positions_get(ticket=ticket)
    if not positions:
        log.warning(f"Position {ticket} not found — may already be closed")
        return False

    pos  = positions[0]
    tick = mt5.symbol_info_tick(pos.symbol)
    if tick is None:
        log.error(f"Cannot get tick for {pos.symbol}")
        return False

    if pos.type == mt5.ORDER_TYPE_BUY:
        close_type = mt5.ORDER_TYPE_SELL
        price      = tick.bid
    else:
        close_type = mt5.ORDER_TYPE_BUY
        price      = tick.ask

    request = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       pos.symbol,
        "volume":       pos.volume,
        "type":         close_type,
        "position":     ticket,
        "price":        price,
        "deviation":    20,
        "magic":        MAGIC,
        "comment":      "HMM_XGBoost_H1_Swing_expire",
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": _filling_type(pos.symbol),
    }

    result = mt5.order_send(request)
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        code = result.retcode if result else "None"
        log.error(f"Close failed ticket={ticket}: retcode={code} | {mt5.last_error()}")
        return False

    log.info(f"Position closed: ticket={ticket} {pos.symbol}")
    return True
