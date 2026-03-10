"""
risk_guard.py — Session drawdown monitor.

Tracks the session equity high and halts trading if drawdown exceeds the
threshold. For Darwinex, the account-level limit is 10% max drawdown.
We halt at 8% to leave a 2% buffer.
"""

from __future__ import annotations

import logging

from src.execution.mt5_connector import get_account_balance

log = logging.getLogger(__name__)

MAX_DRAWDOWN_PCT = 8.0   # halt trading at 8% drawdown from session high


class RiskGuard:
    """
    Monitors live account balance against the session high-water mark.

    Once halted, trading stays halted for the rest of the session.
    Restart the process to reset (intentional — forces human review).
    """

    def __init__(self):
        self._session_high:  float | None = None
        self.trading_halted: bool         = False

    def check(self) -> bool:
        """
        Returns True if trading is allowed, False if halted.
        Also logs the current drawdown level.
        """
        if self.trading_halted:
            return False

        try:
            balance = get_account_balance()
        except Exception as e:
            log.error(f"[RiskGuard] Cannot get balance: {e} — allowing trade")
            return True

        if self._session_high is None:
            self._session_high = balance
            log.info(f"[RiskGuard] Session started. Balance: {balance:.2f}")

        self._session_high = max(self._session_high, balance)

        drawdown_pct = (self._session_high - balance) / self._session_high * 100.0

        if drawdown_pct >= MAX_DRAWDOWN_PCT:
            log.warning(
                f"[RiskGuard] TRADING HALTED — drawdown {drawdown_pct:.2f}% "
                f">= {MAX_DRAWDOWN_PCT}% limit | "
                f"peak={self._session_high:.2f} current={balance:.2f}"
            )
            self.trading_halted = True
            return False

        if drawdown_pct >= MAX_DRAWDOWN_PCT * 0.75:
            log.warning(
                f"[RiskGuard] Drawdown WARNING: {drawdown_pct:.2f}% "
                f"(limit={MAX_DRAWDOWN_PCT}%)"
            )

        return True

    @property
    def drawdown_pct(self) -> float:
        """Current drawdown from session high, in percent."""
        if self._session_high is None:
            return 0.0
        try:
            balance = get_account_balance()
            return (self._session_high - balance) / self._session_high * 100.0
        except Exception:
            return 0.0
