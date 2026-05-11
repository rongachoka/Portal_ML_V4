"""
backfill_pass2_sales_amount.py
==============================
One-time backfill: fills total_sales_amount = total_tax_ex for pre-format
rows in fact_sales_lineitems where total_sales_amount is still NULL.

Scope: sale_date < 2026-02-01 only.
February 2026 onwards is handled by a clean reload via load_to_postgres.py
(see reload_prep_feb2026_plus.py), so this pass must not touch those rows.

Runs a dry-run SELECT first and requires explicit confirmation before
committing the UPDATE.

Run AFTER reload_prep_feb2026_plus.py + load_to_postgres.py have completed.

Run:
    python src/pipelines/maintenance_checks/backfill_pass2_sales_amount.py
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

# ── Dry run ───────────────────────────────────────────────────────────────────
print("Pass 2 — DRY RUN")
print("Scope : sale_date < 2026-02-01 (pre-format rows only)")
print("Action: set total_sales_amount = total_tax_ex where total_sales_amount IS NULL")
print()

cur.execute("""
    SELECT
        location,
        COUNT(*)          AS affected_rows,
        MIN(sale_date)    AS earliest_date,
        MAX(sale_date)    AS latest_date,
        SUM(total_tax_ex) AS sum_tax_ex
    FROM fact_sales_lineitems
    WHERE total_sales_amount IS NULL
      AND total_tax_ex       IS NOT NULL
      AND sale_date           < '2026-02-01'
    GROUP BY location
    ORDER BY location
""")
rows = cur.fetchall()

if not rows:
    print("No rows would be affected. Pass 2 is not needed.")
    cur.close()
    conn.close()
    raise SystemExit(0)

print(f"  {'Location':<25} {'Rows':>8} {'Earliest':>14} {'Latest':>14} {'Sum tax_ex':>16}")
print("  " + "-" * 81)
total_rows = 0
for loc, cnt, earliest, latest, sum_te in rows:
    print(f"  {loc:<25} {cnt:>8,} {str(earliest):>14} {str(latest):>14} {float(sum_te or 0):>16,.2f}")
    total_rows += cnt
print()
print(f"  Total rows that would be updated: {total_rows:,}")
print()

# ── Confirmation ──────────────────────────────────────────────────────────────
answer = input("Proceed with Pass 2 UPDATE? [yes/no]: ").strip().lower()
if answer != "yes":
    print("Aborted — no changes made.")
    cur.close()
    conn.close()
    raise SystemExit(0)

# ── Execute ───────────────────────────────────────────────────────────────────
cur.execute("""
    UPDATE fact_sales_lineitems
    SET total_sales_amount = total_tax_ex
    WHERE total_sales_amount IS NULL
      AND total_tax_ex       IS NOT NULL
      AND sale_date           < '2026-02-01'
""")
updated = cur.rowcount
conn.commit()

print()
print(f"Rows updated : {updated:,}")
print("Committed.")

cur.close()
conn.close()
