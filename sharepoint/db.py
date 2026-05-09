"""
db.py
=====
Lightweight PostgreSQL connection wrapper.
All credentials are read from .env — nothing hardcoded.

Used by:
    sharepoint_parser.py
    load_to_postgres.py

Environment variables required in .env:
    PG_HOST       (default: localhost)
    PG_PORT       (default: 5432)
    PG_DBNAME
    PG_USER
    PG_PASSWORD
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path as _Path

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv(dotenv_path=_Path(__file__).parent.parent / ".env", override=True)

# ── Connection config ─────────────────────────────────────────────────────────

DB_CONFIG = {
    "host":     os.getenv("DB_HOST",  "localhost"),
    "port":     int(os.getenv("DB_PORT", 5432)),
    "dbname":   os.getenv("DB_NAME"),
    "user":     os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
}


# ── Core connection context manager ──────────────────────────────────────────

@contextmanager
def get_connection():
    """
    Yields a psycopg2 connection.
    Commits on clean exit, rolls back on exception, always closes.

    Usage:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(...)
    """
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── Convenience helpers ───────────────────────────────────────────────────────

def execute(sql: str, params=None) -> None:
    """Run a single statement with no return value (UPDATE, DELETE, TRUNCATE)."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)


def fetchone(sql: str, params=None) -> dict | None:
    """Run a SELECT and return the first row as a dict, or None."""
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            return cur.fetchone()


def fetchall(sql: str, params=None) -> list[dict]:
    """Run a SELECT and return all rows as a list of dicts."""
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            return cur.fetchall()


def insert_returning_id(sql: str, params=None) -> int:
    """
    Run an INSERT ... RETURNING id and return the new row's id.
    SQL must end with RETURNING id.
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchone()[0]


def bulk_insert(conn, table: str, columns: list[str], rows: list[tuple]) -> int:
    """
    Insert many rows into a table using execute_batch (500 rows per page).
    Uses the provided connection — caller controls the transaction.

    Returns number of rows inserted.

    Usage:
        with get_connection() as conn:
            n = bulk_insert(conn, 'stg_sales_reports', ['branch', ...], rows)
    """
    if not rows:
        return 0

    col_str      = ", ".join(columns)
    placeholders = ", ".join(["%s"] * len(columns))
    sql          = f"INSERT INTO {table} ({col_str}) VALUES ({placeholders})"

    with conn.cursor() as cur:
        psycopg2.extras.execute_batch(cur, sql, rows, page_size=500)

    return len(rows)


def bulk_insert_safe(conn, table: str, columns: list[str], rows: list[tuple],
                     conflict_columns: list[str]) -> tuple[int, int]:
    """
    Insert rows using ON CONFLICT DO NOTHING on the given conflict_columns.
 
    Unlike bulk_insert(), this will not raise an error if a duplicate row
    is encountered — it simply skips that row and continues.
 
    Requires a unique constraint covering conflict_columns to exist on the table.
 
    Returns:
        (attempted, skipped) — attempted = len(rows), skipped = rows not inserted.
 
    Usage:
        with get_connection() as conn:
            attempted, skipped = bulk_insert_safe(
                conn,
                'stg_sales_reports',
                ['branch', 'transaction_id', ...],
                rows,
                conflict_columns=['transaction_id', 'branch', 'date_sold',
                                  'description', 'qty_sold', 'total_tax_ex'],
            )
    """
    if not rows:
        return 0, 0
 
    col_str      = ", ".join(columns)
    placeholders = ", ".join(["%s"] * len(columns))
    conflict_str = ", ".join(conflict_columns)
 
    sql = (
        f"INSERT INTO {table} ({col_str}) "
        f"VALUES ({placeholders}) "
        f"ON CONFLICT ({conflict_str}) DO NOTHING"
    )
 
    with conn.cursor() as cur:
        psycopg2.extras.execute_batch(cur, sql, rows, page_size=500)
        # rowcount after execute_batch = total rows actually inserted
        inserted = cur.rowcount if cur.rowcount >= 0 else len(rows)
 
    skipped = len(rows) - inserted
    return len(rows), skipped
 


# ── Quick connection test ─────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        row = fetchone("SELECT current_database() AS db, NOW() AS ts")
        print(f"✅ Connected to: {row['db']} at {row['ts']}")
    except Exception as e:
        print(f"❌ Connection failed: {e}")