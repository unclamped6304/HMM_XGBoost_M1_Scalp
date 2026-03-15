"""
db.py — PostgreSQL connection factory.
"""

import psycopg2
from psycopg2.extensions import connection as PgConnection

from src.config import DB_HOST, DB_NAME, DB_USER, DB_PASSWORD


def get_connection() -> PgConnection:
    return psycopg2.connect(
        host=DB_HOST,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )
