"""
backfill_pass1_sales_amount.py
==============================
One-time backfill: copies total_sales_amount from stg_sales_reports into
fact_sales_lineitems for rows where the value is currently NULL.

Only touches rows where stg_sales_reports has a non-NULL, positive
total_sales_amount — pre-format rows (column absent in original export)
are left NULL and handled separately by Pass 2.

Run:
    python src/pipelines/maintenance_checks/backfill_pass1_sales_amount.py
"""

import os
import sys
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

print("Pass 1 — copying total_sales_amount from stg_sales_reports into fact_sales_lineitems")
print("Matching on: transaction_id + description + total_tax_ex + branch mapping")
print()

cur.execute("""
    UPDATE fact_sales_lineitems fsl
    SET total_sales_amount = stg.total_sales_amount
    FROM stg_sales_reports stg
    WHERE fsl.transaction_id = stg.transaction_id
      AND fsl.description    = stg.description
      AND fsl.total_tax_ex   = stg.total_tax_ex
      AND fsl.location = CASE stg.branch
          WHEN 'Galleria'     THEN 'GALLERIA'
          WHEN 'Portal 2R'    THEN 'PORTAL_2R'
          WHEN 'Portal CBD'   THEN 'PORTAL_CBD'
          WHEN 'ABC'          THEN 'PHARMART_ABC'
          WHEN 'Milele'       THEN 'NGONG_MILELE'
          WHEN 'Centurion 2R' THEN 'CENTURION_2R'
          ELSE UPPER(REPLACE(stg.branch, ' ', '_'))
      END
      AND fsl.total_sales_amount IS NULL
      AND stg.total_sales_amount IS NOT NULL
      AND stg.total_sales_amount > 0
""")
updated = cur.rowcount
conn.commit()

print(f"Rows updated : {updated:,}")
print("Committed.")
print()

# Spot-check: how many NULLs remain?
cur.execute("""
    SELECT location, COUNT(*) AS remaining_nulls
    FROM fact_sales_lineitems
    WHERE total_sales_amount IS NULL
    GROUP BY location
    ORDER BY location
""")
rows = cur.fetchall()
if rows:
    print("Remaining NULL rows per location (candidates for Pass 2):")
    for loc, cnt in rows:
        print(f"  {loc:<25} {cnt:,}")
else:
    print("No NULL rows remain — Pass 2 is not needed.")

print()

# Verification for May 2026
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
