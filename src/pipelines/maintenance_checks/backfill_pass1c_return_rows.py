"""
backfill_pass1c_return_rows.py
================================
Fixes return line items (qty_sold < 0) that were skipped by Pass 1 because
the stg.total_sales_amount > 0 guard excluded negative values.

Join key  : transaction_id + description + total_tax_ex + branch  (same as Pass 1)
Guard     : stg.total_sales_amount > 0 removed — negative values copied as-is
Target    : fact_sales_lineitems rows where qty_sold < 0 AND total_sales_amount IS NULL
Scope     : April and May 2026 only (sale_date 2026-04-01 to 2026-05-31)
Excluded  : CENTURION_2R (all months — duplicate audit not yet complete)

Pass 1b rows are untouched — this pass uses the strict total_tax_ex join key
so it cannot overlap with Pass 1b's looser matches.

Dry-run SELECT runs first. Requires explicit confirmation before any UPDATE.

Run:
    python src/pipelines/maintenance_checks/backfill_pass1c_return_rows.py
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
print("Pass 1c — DRY RUN")
print("Target   : qty_sold < 0 AND total_sales_amount IS NULL (return rows skipped by Pass 1)")
print("Join key : transaction_id + description + total_tax_ex + branch")
print("Guard    : stg.total_sales_amount > 0 removed — negative values copied as-is")
print("Scope    : sale_date 2026-04-01 to 2026-05-31")
print("Excluded : CENTURION_2R (all months)")
print()

cur.execute(f"""
    SELECT
        fsl.location,
        DATE_TRUNC('month', fsl.sale_date)  AS month,
        COUNT(*)                             AS rows_to_update,
        SUM(stg.total_sales_amount)          AS sum_sales_amount_to_write
    FROM fact_sales_lineitems fsl
    JOIN stg_sales_reports stg
        ON  fsl.transaction_id = stg.transaction_id
        AND fsl.description    = stg.description
        AND fsl.total_tax_ex   = stg.total_tax_ex
        AND fsl.location       = {BRANCH_MAP}
    WHERE fsl.qty_sold            < 0
      AND fsl.total_sales_amount IS NULL
      AND fsl.sale_date          >= '2026-04-01'
      AND fsl.sale_date           < '2026-06-01'
      AND fsl.location           <> 'CENTURION_2R'
      AND stg.total_sales_amount IS NOT NULL
    GROUP BY fsl.location, DATE_TRUNC('month', fsl.sale_date)
    ORDER BY fsl.location, month
""")
rows = cur.fetchall()

if not rows:
    print("No return rows found to update — nothing to do.")
    cur.close()
    conn.close()
    raise SystemExit(0)

total_rows = 0
print(f"  {'Location':<22} {'Month':<10} {'Rows':>8} {'Sum sales_amount':>18}")
print("  " + "-" * 62)
for loc, month, cnt, total_sa in rows:
    print(f"  {loc:<22} {str(month)[:7]:<10} {cnt:>8,} {float(total_sa or 0):>18,.2f}")
    total_rows += cnt
print("  " + "-" * 62)
print(f"  {'TOTAL':<22} {'':10} {total_rows:>8,}")
print()

# ── Confirmation ──────────────────────────────────────────────────────────────
answer = input("Proceed with Pass 1c UPDATE? [yes/no]: ").strip().lower()
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
    WHERE fsl.transaction_id   = stg.transaction_id
      AND fsl.description      = stg.description
      AND fsl.total_tax_ex     = stg.total_tax_ex
      AND fsl.location         = {BRANCH_MAP}
      AND fsl.qty_sold          < 0
      AND fsl.total_sales_amount IS NULL
      AND fsl.sale_date        >= '2026-04-01'
      AND fsl.sale_date         < '2026-06-01'
      AND fsl.location         <> 'CENTURION_2R'
      AND stg.total_sales_amount IS NOT NULL
""")
updated = cur.rowcount
conn.commit()

print()
print(f"Rows updated : {updated:,}")
print("Committed.")
print()

# ── Verification ──────────────────────────────────────────────────────────────
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
