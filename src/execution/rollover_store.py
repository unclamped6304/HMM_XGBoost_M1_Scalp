"""
rollover_store.py — Persist rollover original SL/TP values in PostgreSQL.

Survives process restarts so widened stops can always be restored even
if the trader crashes or is restarted during the rollover window.
"""

from __future__ import annotations

import logging

from src.db import get_connection

log = logging.getLogger(__name__)

_CREATE = """
CREATE TABLE IF NOT EXISTS rollover_state (
    ticket   BIGINT PRIMARY KEY,
    sl       DOUBLE PRECISION NOT NULL,
    tp       DOUBLE PRECISION NOT NULL,
    saved_at TIMESTAMPTZ DEFAULT NOW()
)
"""


def ensure_table() -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(_CREATE)


def save(ticket: int, sl: float, tp: float) -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO rollover_state (ticket, sl, tp)
                VALUES (%s, %s, %s)
                ON CONFLICT (ticket) DO UPDATE SET sl = EXCLUDED.sl, tp = EXCLUDED.tp, saved_at = NOW()
                """,
                (ticket, sl, tp),
            )


def load_all() -> dict[int, tuple[float, float]]:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT ticket, sl, tp FROM rollover_state")
            return {row[0]: (row[1], row[2]) for row in cur.fetchall()}


def delete(ticket: int) -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM rollover_state WHERE ticket = %s", (ticket,))


def clear_all() -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM rollover_state")
