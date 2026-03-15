"""
rollover_store.py — Persist rollover original SL/TP values in PostgreSQL.

Survives process restarts so widened stops can always be restored even
if the trader crashes or is restarted during the rollover window.

Each row is keyed by (account, ticket) so the table can serve multiple
MT5 accounts simultaneously.
"""

from __future__ import annotations

import logging

from src.db import get_connection

log = logging.getLogger(__name__)

_CREATE = """
CREATE TABLE IF NOT EXISTS rollover_state (
    account  BIGINT NOT NULL,
    ticket   BIGINT NOT NULL,
    sl       DOUBLE PRECISION NOT NULL,
    tp       DOUBLE PRECISION NOT NULL,
    saved_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (account, ticket)
)
"""

# Migrate old schema (ticket-only PK) if the account column is missing.
_MIGRATE = """
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'rollover_state' AND column_name = 'account'
    ) THEN
        ALTER TABLE rollover_state DROP CONSTRAINT rollover_state_pkey;
        ALTER TABLE rollover_state ADD COLUMN account BIGINT NOT NULL DEFAULT 0;
        ALTER TABLE rollover_state ADD PRIMARY KEY (account, ticket);
    END IF;
END
$$;
"""


def ensure_table() -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(_CREATE)
            cur.execute(_MIGRATE)


def save(account: int, ticket: int, sl: float, tp: float) -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO rollover_state (account, ticket, sl, tp)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (account, ticket) DO UPDATE
                    SET sl = EXCLUDED.sl, tp = EXCLUDED.tp, saved_at = NOW()
                """,
                (account, ticket, sl, tp),
            )


def load_all(account: int) -> dict[int, tuple[float, float]]:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT ticket, sl, tp FROM rollover_state WHERE account = %s",
                (account,),
            )
            return {row[0]: (row[1], row[2]) for row in cur.fetchall()}


def delete(account: int, ticket: int) -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM rollover_state WHERE account = %s AND ticket = %s",
                (account, ticket),
            )


def clear_all(account: int) -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM rollover_state WHERE account = %s",
                (account,),
            )
