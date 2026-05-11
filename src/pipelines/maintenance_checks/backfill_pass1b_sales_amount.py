"""
backfill_pass1b_sales_amount.py
================================
Second attempt at copying total_sales_amount from stg_sales_reports into
fact_sales_lineitems for rows that Pass 1 missed due to a strict
total_tax_ex exact-match requirement.

Join key: transaction_id + description + qty_sold + branch mapping
Scope   : April and May 2026 only (sale_date 2026-04-01 to 2026-05-31)
Excluded: CENTURION_2R — duplicate audit not yet complete for that branch

Runs a dry-run SELECT (rows affected by branch + month) before any UPDATE.
Requires explicit confirmation to proceed.

Run:
    python src/pipelines/maintenance_checks/backfill_pass1b_sales_amount.py
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

BRANCH_MAP = """
    CASE stg.branch
        WHEN 'Galleria'     THEN 'GALLERIA'
        WHEN 'Portal 2R'    THEN 'PORTAL_2R'
        WHEN 'Portal CBD'   THEN 'PORTAL_CBD'
        WHEN 'ABC'          THEN 'PHARMART_ABC'
        WHEN 'Milele'       THEN 'NGONG_MILELE'
        ELSE UPPER(REPLACE(stg.branch, ' ', '_'))
    END
"""

# ── Dry run ───────────────────────────────────────────────────────────────────
print("Pass 1b — DRY RUN")
print("Join key : transaction_id + description + qty_sold + branch")
print("Scope    : sale_date 2026-04-01 to 2026-05-31")
print("Excluded : CENTURION_2R (all months)")
print()

cur.execute(f"""
    SELECT
        fsl.location,
        DATE_TRUNC('month', fsl.sale_date)  AS month,
        COUNT(*)                             AS rows_to_update
    FROM fact_sales_lineitems fsl
    JOIN stg_sales_reports stg
        ON  fsl.transaction_id = stg.transaction_id
        AND fsl.description    = stg.description
        AND fsl.qty_sold       = stg.qty_sold
        AND fsl.location       = {BRANCH_MAP}
    WHERE fsl.total_sales_amount IS NULL
      AND fsl.sale_date >= '2026-04-01'
      AND fsl.sale_date <  '2026-06-01'
      AND fsl.location  <> 'CENTURION_2R'
      AND stg.total_sales_amount IS NOT NULL
      AND stg.total_sales_amount  > 0
    GROUP BY fsl.location, DATE_TRUNC('month', fsl.sale_date)
    ORDER BY fsl.location, month
""")
rows = cur.fetchall()

if not rows:
    print("No rows would be updated — nothing to do.")
    cur.close()
    conn.close()
    raise SystemExit(0)

total = 0
print(f"  {'Location':<22} {'Month':<10} {'Rows to update':>16}")
print("  " + "-" * 52)
for loc, month, cnt in rows:
    print(f"  {loc:<22} {str(month)[:7]:<10} {cnt:>16,}")
    total += cnt
print("  " + "-" * 52)
print(f"  {'TOTAL':<22} {'':10} {total:>16,}")
print()

# ── Confirmation ──────────────────────────────────────────────────────────────
answer = input("Proceed with Pass 1b UPDATE? [yes/no]: ").strip().lower()
if answer != "yes":
    print("Aborted — no changes made.")
    cur.close()
    conn.close()
    raise SystemExit(0)

# ── Execute ───────────────────────────────────────────────────────────────────
cur.execute(f"""
    UPDATE fact_sales_lineitems fsl
    SET total_sales_amount = stg.total_sales_amount
    FROM stg_sales_reports stg
    WHERE fsl.transaction_id = stg.transaction_id
      AND fsl.description    = stg.description
      AND fsl.qty_sold       = stg.qty_sold
      AND fsl.location       = {BRANCH_MAP}
      AND fsl.total_sales_amount IS NULL
      AND fsl.sale_date >= '2026-04-01'
      AND fsl.sale_date <  '2026-06-01'
      AND fsl.location  <> 'CENTURION_2R'
      AND stg.total_sales_amount IS NOT NULL
      AND stg.total_sales_amount  > 0
""")
updated = cur.rowcount
conn.commit()

print()
print(f"Rows updated : {updated:,}")
print("Committed.")
print()

# ── Verification ──────────────────────────────────────────────────────────────
cur.execute("""
    SELECT
        location,
        COUNT(*)                  AS total_rows,
        COUNT(total_sales_amount) AS populated,
        COUNT(*) - COUNT(total_sales_amount) AS still_null
    FROM fact_sales_lineitems
    WHERE sale_date >= '2026-04-01'
      AND sale_date <  '2026-06-01'
      AND location  <> 'CENTURION_2R'
    GROUP BY location
    ORDER BY location
""")
print("Remaining NULLs for April–May 2026 (excl. Centurion 2R):")
print(f"  {'Location':<22} {'Total':>8} {'Populated':>10} {'Still NULL':>12}")
print("  " + "-" * 56)
for loc, total, pop, null in cur.fetchall():
    print(f"  {loc:<22} {total:>8,} {pop:>10,} {null:>12,}")

print()

cur.execute("""
    SELECT branch, SUM(revenue) AS total_revenue
    FROM vw_sales_base
    WHERE sale_date >= '2026-05-01' AND sale_date < '2026-06-01'
      AND branch IN ('PORTAL_2R', 'GALLERIA')
    GROUP BY branch
    ORDER BY branch
""")
accountant = {"GALLERIA": 2_763_867.00, "PORTAL_2R": 3_469_850.00}
print("May 2026 verification from vw_sales_base:")
print(f"  {'Branch':<20} {'DB Revenue':>16} {'Accountant':>16} {'Gap':>12}")
print("  " + "-" * 68)
for branch, rev in cur.fetchall():
    rev = float(rev or 0)
    acct = accountant.get(branch, 0)
    gap = rev - acct
    print(f"  {branch:<20} {rev:>16,.2f} {acct:>16,.2f} {gap:>12,.2f}")

cur.close()
conn.close()
