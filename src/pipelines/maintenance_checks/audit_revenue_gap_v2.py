"""
check_social_revenue_q1_2026.py
================================
Checks social media revenue (Ordered Via = 'respond.io') for Jan-Mar 2026
against accountant figures, broken down by branch and month.

Run from your project root:
    python check_social_revenue_q1_2026.py
"""

import os
from datetime import date
from pathlib import Path

import pandas as pd
import psycopg2
import psycopg2.extras

# ── DB connection ─────────────────────────────────────────────────────────────
DB_HOST = os.getenv("DB_HOST", "68.183.0.188")
DB_NAME = os.getenv("DB_NAME", "portal_pharmacy")
DB_USER = os.getenv("DB_USER", "portal_user")
DB_PASS = os.getenv("DB_PASS", "Ishm@el12345")
DB_PORT = int(os.getenv("DB_PORT", 5432))

# ── Paste accountant's Jan-Mar 2026 figures here once you have them ───────────
ACCOUNTANT_2026 = {
    "2026-01": None,   # e.g. 1_234_567
    "2026-02": None,
    "2026-03": None,
}

REPORT_LINES = []

def log(text=""):
    print(text)
    REPORT_LINES.append(text)

def section(title):
    log(f"\n{'═' * 68}")
    log(f"  {title}")
    log(f"{'═' * 68}")

def get_conn():
    return psycopg2.connect(
        host=DB_HOST, dbname=DB_NAME, user=DB_USER,
        password=DB_PASS, port=DB_PORT,
    )

def query_df(sql, params=None):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params or [])
            cols = [d[0] for d in cur.description]
            return pd.DataFrame(cur.fetchall(), columns=cols)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — What values actually live in Ordered Via? (sanity check)
# ══════════════════════════════════════════════════════════════════════════════
def check_ordered_via_values():
    section("STEP 1 — Distinct 'ordered_via' values in cashier (sanity check)")
    log("  Confirms the exact strings present so we match the right filter.\n")

    sql = """
        SELECT
            LOWER(TRIM(ordered_via))   AS ordered_via_clean,
            COUNT(DISTINCT receipt_txn_no) AS txn_count,
            COUNT(*)                        AS row_count
        FROM stg_cashier_reports
        WHERE transaction_date >= '2026-01-01'
          AND transaction_date <  '2026-04-01'
        GROUP BY 1
        ORDER BY 2 DESC
        LIMIT 20
    """
    df = query_df(sql)
    if df.empty:
        log("  ❌ No cashier data found for Jan-Mar 2026.")
        log("     Check that stg_cashier_reports has been populated for this period.")
        return

    log(f"  {'Ordered Via Value':<30} {'Txns':>8} {'Rows':>8}")
    log(f"  {'-'*50}")
    for _, row in df.iterrows():
        val = str(row["ordered_via_clean"] or "(null/empty)")
        log(f"  {val:<30} {int(row['txn_count']):>8,} {int(row['row_count']):>8,}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Social revenue by month (Jan-Mar 2026)
# ══════════════════════════════════════════════════════════════════════════════
def check_social_revenue_monthly():
    section("STEP 2 — Social revenue by month (respond.io, Jan-Mar 2026)")

    sql = """
        SELECT
            TO_CHAR(s.date_sold, 'YYYY-MM')         AS month,
            SUM(s.total_tax_ex)                      AS revenue_tax_ex,
            COUNT(DISTINCT s.transaction_id)         AS transactions,
            COUNT(*)                                 AS line_items,
            ROUND(AVG(s.total_tax_ex)::numeric, 0)  AS avg_line_item
        FROM stg_sales_reports s
        JOIN stg_cashier_reports c
          ON  c.receipt_txn_no  = s.transaction_id
          AND LOWER(TRIM(c.branch)) = LOWER(TRIM(s.branch))
        WHERE s.date_sold >= '2026-01-01'
          AND s.date_sold <  '2026-04-01'
          AND LOWER(TRIM(c.ordered_via)) = 'respond.io'
        GROUP BY 1
        ORDER BY 1
    """
    df = query_df(sql)

    if df.empty:
        log("  ❌ No respond.io rows found when joining sales + cashier.")
        log("     Trying cashier-only revenue (Amount column) below...\n")
        check_cashier_only_social()
        return

    log(f"\n  {'Month':<10} {'Rev (Tax Ex)':>14} {'Accountant':>14} {'Delta':>12} {'Txns':>7} {'Lines':>7}")
    log(f"  {'-'*70}")
    for _, row in df.iterrows():
        month = str(row["month"])
        rev   = float(row["revenue_tax_ex"] or 0)
        acct  = ACCOUNTANT_2026.get(month)
        if acct:
            delta = rev - acct
            flag  = "  ⚠" if abs(delta) > 1000 else "  ✅"
            acct_str = f"{acct:>14,.0f}"
            delta_str = f"{delta:>+12,.0f}"
        else:
            acct_str  = f"{'(not set)':>14}"
            delta_str = f"{'—':>12}"
            flag = ""
        log(
            f"  {month:<10} {rev:>14,.0f} {acct_str} {delta_str} "
            f"{int(row['transactions']):>7,} {int(row['line_items']):>7,}{flag}"
        )

    log("\n  Note: Revenue column = total_tax_ex (tax-exclusive).")
    log("  If accountant uses tax-inclusive, see Step 4 for Amount comparison.")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Social revenue by branch by month
# ══════════════════════════════════════════════════════════════════════════════
def check_social_revenue_by_branch():
    section("STEP 3 — Social revenue by branch and month (respond.io)")

    sql = """
        SELECT
            s.branch,
            TO_CHAR(s.date_sold, 'YYYY-MM')  AS month,
            SUM(s.total_tax_ex)               AS revenue_tax_ex,
            COUNT(DISTINCT s.transaction_id)  AS transactions
        FROM stg_sales_reports s
        JOIN stg_cashier_reports c
          ON  c.receipt_txn_no  = s.transaction_id
          AND LOWER(TRIM(c.branch)) = LOWER(TRIM(s.branch))
        WHERE s.date_sold >= '2026-01-01'
          AND s.date_sold <  '2026-04-01'
          AND LOWER(TRIM(c.ordered_via)) = 'respond.io'
        GROUP BY s.branch, TO_CHAR(s.date_sold, 'YYYY-MM')
        ORDER BY s.branch, month
    """
    df = query_df(sql)

    if df.empty:
        log("  ❌ No data — see Step 2 fallback.")
        return

    log(f"\n  {'Branch':<16} {'Month':<10} {'Revenue (Tax Ex)':>18} {'Txns':>8}")
    log(f"  {'-'*58}")

    prev_branch = None
    branch_totals = df.groupby("branch")["revenue_tax_ex"].sum()

    for _, row in df.iterrows():
        branch = str(row["branch"])
        if branch != prev_branch and prev_branch is not None:
            total = branch_totals.get(prev_branch, 0)
            log(f"  {'':16} {'SUBTOTAL':<10} {float(total):>18,.0f}")
            log(f"  {'-'*58}")
        prev_branch = branch
        log(
            f"  {branch:<16} {str(row['month']):<10} "
            f"{float(row['revenue_tax_ex'] or 0):>18,.0f} "
            f"{int(row['transactions']):>8,}"
        )

    # Last branch subtotal
    if prev_branch:
        total = branch_totals.get(prev_branch, 0)
        log(f"  {'':16} {'SUBTOTAL':<10} {float(total):>18,.0f}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Cashier Amount (tax-inclusive) for same social transactions
# ══════════════════════════════════════════════════════════════════════════════
def check_cashier_amount_social():
    section("STEP 4 — Cashier Amount (tax-inclusive) for respond.io transactions")
    log("  Compares tax-ex (sales report) vs Amount (cashier) for same txns.\n")

    sql = """
        SELECT
            TO_CHAR(c.transaction_date, 'YYYY-MM') AS month,
            COUNT(DISTINCT c.receipt_txn_no)        AS txns,
            SUM(c.amount)                           AS cashier_amount,
            SUM(s.total_tax_ex)                     AS sales_tax_ex,
            SUM(c.amount) - SUM(s.total_tax_ex)     AS tax_difference
        FROM stg_cashier_reports c
        JOIN stg_sales_reports s
          ON  s.transaction_id  = c.receipt_txn_no
          AND LOWER(TRIM(s.branch)) = LOWER(TRIM(c.branch))
        WHERE c.transaction_date >= '2026-01-01'
          AND c.transaction_date <  '2026-04-01'
          AND LOWER(TRIM(c.ordered_via)) = 'respond.io'
        GROUP BY 1
        ORDER BY 1
    """
    df = query_df(sql)

    if df.empty:
        log("  ❌ No matched rows — branch name mismatch likely.")
        check_cashier_only_social()
        return

    log(f"  {'Month':<10} {'Txns':>7} {'Cashier Amt':>14} {'Tax Ex':>14} {'Tax Diff':>12}")
    log(f"  {'-'*65}")
    for _, row in df.iterrows():
        log(
            f"  {str(row['month']):<10} {int(row['txns']):>7,} "
            f"{float(row['cashier_amount'] or 0):>14,.0f} "
            f"{float(row['sales_tax_ex'] or 0):>14,.0f} "
            f"{float(row['tax_difference'] or 0):>+12,.0f}"
        )


def check_cashier_only_social():
    """
    Fallback: pulls respond.io revenue from cashier table alone (no sales join).
    Useful when branch name mismatch prevents the join from working.
    """
    log("\n  ── Cashier-only fallback (no join to sales) ──────────────────────")
    log("  Uses stg_cashier_reports.amount directly for respond.io txns.\n")

    sql = """
        SELECT
            TO_CHAR(transaction_date, 'YYYY-MM') AS month,
            COUNT(DISTINCT receipt_txn_no)        AS txns,
            SUM(amount)                           AS cashier_amount,
            branch
        FROM stg_cashier_reports
        WHERE transaction_date >= '2026-01-01'
          AND transaction_date <  '2026-04-01'
          AND LOWER(TRIM(ordered_via)) = 'respond.io'
        GROUP BY TO_CHAR(transaction_date, 'YYYY-MM'), branch
        ORDER BY branch, month
    """
    df = query_df(sql)

    if df.empty:
        log("  ❌ Still no data. Either:")
        log("     a) ordered_via column uses a different string — check Step 1 output")
        log("     b) stg_cashier_reports has no Jan-Mar 2026 data yet")
        log("     c) The cashier files for this period haven't been uploaded to SharePoint")
        return

    log(f"  {'Branch':<16} {'Month':<10} {'Txns':>8} {'Amount':>14}")
    log(f"  {'-'*55}")
    for _, row in df.iterrows():
        log(
            f"  {str(row['branch']):<16} {str(row['month']):<10} "
            f"{int(row['txns']):>8,} "
            f"{float(row['cashier_amount'] or 0):>14,.0f}"
        )

    monthly = df.groupby("month")["cashier_amount"].sum()
    log(f"\n  Monthly totals (cashier Amount, all branches):")
    log(f"  {'Month':<10} {'Total':>14} {'Accountant':>14} {'Delta':>12}")
    log(f"  {'-'*55}")
    for month, total in monthly.items():
        acct = ACCOUNTANT_2026.get(str(month))
        if acct:
            delta = float(total) - acct
            flag  = "  ⚠" if abs(delta) > 1000 else "  ✅"
            log(f"  {month:<10} {float(total):>14,.0f} {acct:>14,.0f} {delta:>+12,.0f}{flag}")
        else:
            log(f"  {month:<10} {float(total):>14,.0f} {'(not set)':>14} {'—':>12}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Daily social revenue (spot thin days / missing uploads)
# ══════════════════════════════════════════════════════════════════════════════
def check_daily_social():
    section("STEP 5 — Daily social revenue (Jan-Mar 2026, respond.io)")
    log("  Spots specific dates with zero or unusually low revenue.\n")

    sql = """
        SELECT
            c.transaction_date                       AS sale_date,
            COUNT(DISTINCT c.receipt_txn_no)          AS txns,
            SUM(c.amount)                             AS amount
        FROM stg_cashier_reports c
        WHERE c.transaction_date >= '2026-01-01'
          AND c.transaction_date <  '2026-04-01'
          AND LOWER(TRIM(c.ordered_via)) = 'respond.io'
        GROUP BY c.transaction_date
        ORDER BY c.transaction_date
    """
    df = query_df(sql)

    if df.empty:
        log("  ❌ No daily data — check ordered_via string (Step 1).")
        return

    # Flag days with revenue more than 50% below the monthly average
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0)
    df["month"]  = pd.to_datetime(df["sale_date"]).dt.to_period("M").astype(str)
    monthly_avg  = df.groupby("month")["amount"].mean()
    df["avg"]    = df["month"].map(monthly_avg)
    df["flag"]   = df["amount"] < df["avg"] * 0.5

    log(f"  {'Date':<14} {'Txns':>6} {'Amount':>12}  Note")
    log(f"  {'-'*50}")
    for _, row in df.iterrows():
        note = "  ⚠ LOW" if row["flag"] else ""
        log(
            f"  {str(row['sale_date']):<14} {int(row['txns']):>6,} "
            f"{float(row['amount']):>12,.0f}{note}"
        )

    log(f"\n  Total days with data: {len(df)}")
    log(f"  Low-revenue days flagged: {df['flag'].sum()}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    log("=" * 68)
    log("  PORTAL ML — SOCIAL REVENUE CHECK  (respond.io, Jan-Mar 2026)")
    log(f"  Run date: {date.today()}")
    log("=" * 68)
    log(
        "\n  Paste accountant's Jan-Mar 2026 figures into ACCOUNTANT_2026 at the"
        "\n  top of this script to see delta columns populate.\n"
    )

    try:
        check_ordered_via_values()
        check_social_revenue_monthly()
        check_social_revenue_by_branch()
        check_cashier_amount_social()
        check_daily_social()

    except Exception as e:
        import traceback
        log(f"\n❌ Error: {e}")
        log(traceback.format_exc())

    report = Path("social_revenue_q1_2026_report.txt")
    report.write_text("\n".join(REPORT_LINES), encoding="utf-8")
    log(f"\n📄 Report saved: {report.resolve()}")


if __name__ == "__main__":
    main()