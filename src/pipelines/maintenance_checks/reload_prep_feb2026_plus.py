"""
reload_prep_feb2026_plus.py
============================
Prepares a clean reload of GALLERIA and PORTAL_2R from February 2026 onwards.

What this script does:
  1. Dry-run: shows rows that will be deleted, by table and month
  2. On confirmation:
     a. Deletes fact_sales_lineitems rows  (GALLERIA + PORTAL_2R, sale_date >= 2026-02-01)
     b. Deletes fact_sales_transactions rows (same scope)
     c. Resets watermarks to 2026-01-31 so load_to_postgres.py reloads from Feb 2026

After this script commits, run:
    python src/pipelines/pos_finance/load_to_postgres.py

The updated ETL writes total_sales_amount directly from staging (with total_tax_ex
fallback for pre-format rows), so the reloaded data will be clean.

Run:
    python src/pipelines/maintenance_checks/reload_prep_feb2026_plus.py
"""

import os
from pathlib import Path

from dotenv import load_dotenv
import psycopg2

load_dotenv(Path(__file__).resolve().parents[3] / ".env", override=True)

conn = psycopg2.connect(
    host=os.getenv("DB_HOST"),
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    port=int(os.getenv("DB_PORT", 5432)),
    sslmode="disable",
)
cur = conn.cursor()

LOCATIONS = ["GALLERIA", "PORTAL_2R"]
CUTOFF    = "2026-02-01"

# ── Dry run ───────────────────────────────────────────────────────────────────
print("RELOAD PREP — DRY RUN")
print(f"Scope    : GALLERIA and PORTAL_2R, sale_date >= {CUTOFF}")
print("Action   : delete from fact_sales_lineitems and fact_sales_transactions")
print("Watermark: reset to 2026-01-31 for both branches")
print()

# fact_sales_lineitems breakdown
cur.execute("""
    SELECT
        location,
        DATE_TRUNC('month', sale_date) AS month,
        COUNT(*)                        AS rows
    FROM fact_sales_lineitems
    WHERE location = ANY(%s)
      AND sale_date >= %s
    GROUP BY location, DATE_TRUNC('month', sale_date)
    ORDER BY location, month
""", [LOCATIONS, CUTOFF])
li_rows = cur.fetchall()

print("fact_sales_lineitems rows to delete:")
print(f"  {'Location':<22} {'Month':<10} {'Rows':>10}")
print("  " + "-" * 46)
li_total = 0
for loc, month, cnt in li_rows:
    print(f"  {loc:<22} {str(month)[:7]:<10} {cnt:>10,}")
    li_total += cnt
print("  " + "-" * 46)
print(f"  {'TOTAL':<22} {'':10} {li_total:>10,}")
print()

# fact_sales_transactions breakdown
cur.execute("""
    SELECT
        location,
        DATE_TRUNC('month', sale_date) AS month,
        COUNT(*)                        AS rows
    FROM fact_sales_transactions
    WHERE location = ANY(%s)
      AND sale_date >= %s
    GROUP BY location, DATE_TRUNC('month', sale_date)
    ORDER BY location, month
""", [LOCATIONS, CUTOFF])
txn_rows = cur.fetchall()

print("fact_sales_transactions rows to delete:")
print(f"  {'Location':<22} {'Month':<10} {'Rows':>10}")
print("  " + "-" * 46)
txn_total = 0
for loc, month, cnt in txn_rows:
    print(f"  {loc:<22} {str(month)[:7]:<10} {cnt:>10,}")
    txn_total += cnt
print("  " + "-" * 46)
print(f"  {'TOTAL':<22} {'':10} {txn_total:>10,}")
print()

# Watermark preview
cur.execute("""
    SELECT branch, max_date_loaded
    FROM fact_load_watermarks
    WHERE branch = ANY(%s)
    ORDER BY branch
""", [LOCATIONS])
print("Watermarks to be reset (current -> 2026-01-31):")
for branch, current in cur.fetchall():
    print(f"  {branch:<22} {str(current)} -> 2026-01-31")
print()

# ── Confirmation ──────────────────────────────────────────────────────────────
print(f"This will permanently delete {li_total:,} line items and {txn_total:,} transactions.")
answer = input("Type 'delete' to confirm: ").strip().lower()
if answer != "delete":
    print("Aborted — no changes made.")
    cur.close()
    conn.close()
    raise SystemExit(0)

# ── Execute ───────────────────────────────────────────────────────────────────
print()
print("Deleting fact_sales_lineitems...")
cur.execute("""
    DELETE FROM fact_sales_lineitems
    WHERE location = ANY(%s)
      AND sale_date >= %s
""", [LOCATIONS, CUTOFF])
deleted_li = cur.rowcount
print(f"  Deleted: {deleted_li:,} rows")

print("Deleting fact_sales_transactions...")
cur.execute("""
    DELETE FROM fact_sales_transactions
    WHERE location = ANY(%s)
      AND sale_date >= %s
""", [LOCATIONS, CUTOFF])
deleted_txn = cur.rowcount
print(f"  Deleted: {deleted_txn:,} rows")

print("Resetting watermarks to 2026-01-31...")
for branch in LOCATIONS:
    cur.execute("""
        UPDATE fact_load_watermarks
        SET max_date_loaded = '2026-01-31',
            last_updated_at = NOW()
        WHERE branch = %s
    """, [branch])
    if cur.rowcount == 0:
        cur.execute("""
            INSERT INTO fact_load_watermarks (branch, max_date_loaded, last_updated_at)
            VALUES (%s, '2026-01-31', NOW())
        """, [branch])
    print(f"  {branch} watermark set to 2026-01-31")

conn.commit()
print()
print("Committed.")
print()
print("Next step: run the ETL to reload from staging:")
print("    python src/pipelines/pos_finance/load_to_postgres.py")
print()
print("The updated ETL writes total_sales_amount from staging with a")
print("total_tax_ex fallback for pre-format rows. Feb 2026+ data will")
print("be clean after the reload.")

cur.close()
conn.close()
