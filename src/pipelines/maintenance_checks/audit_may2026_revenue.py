"""
audit_may2026_revenue.py
========================
Audits the May 2026 revenue discrepancy between the database figures and the
accountants' figures for Portal 2R and Galleria.

Checks performed (read-only — no data is modified):
  1. Schema: does stg_sales_reports / fact tables have total_sales_amount?
  2. Live definition of vw_sales_base
  3. Revenue from vw_sales_base (what Power BI currently shows)
  4. Revenue from raw fact tables (unfiltered cross-check)
  5. Row exclusion audit — which filters remove how much revenue
  6. Negative qty_sold (returns reducing revenue)
  7. NULL total_tax_ex rows
  8. Watermark — are all May 2026 records actually loaded?
  9. CSV cross-check — Total (Tax Ex) vs Total Sales Amount gap
 10. Final reconciliation summary

Run:
    python src/pipelines/maintenance_checks/audit_may2026_revenue.py
"""

import io
import os
import sys
from pathlib import Path

# Force UTF-8 output on Windows consoles that default to cp1252
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd
import psycopg2

# ── Env / path setup ──────────────────────────────────────────────────────────
try:
    from Portal_ML_V4.src.config.settings import BASE_DIR
except ImportError:
    BASE_DIR = Path(__file__).resolve().parents[3]

env_path = BASE_DIR / ".env"
if env_path.exists():
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=env_path, override=True)

DB_HOST = os.getenv("DB_HOST", "68.183.0.188")
DB_NAME = os.getenv("DB_NAME", "portal_pharmacy")
DB_USER = os.getenv("DB_USER", "portal_user")
DB_PASS = os.getenv("DB_PASSWORD", "Ishm@el12345")
DB_PORT = int(os.getenv("DB_PORT", 5432))

ACCOUNTANT = {
    "PORTAL_2R": 3_469_850.00,
    "GALLERIA":  2_763_867.00,
}
BRANCHES = list(ACCOUNTANT.keys())
MAY_START = "2026-05-01"
MAY_END   = "2026-06-01"

CSV_PATH = (
    BASE_DIR / "data" / "03_processed" / "pos_data" / "all_locations_sales_NEW.csv"
)

REPORT_LINES: list[str] = []


# ── Helpers ───────────────────────────────────────────────────────────────────

def log(text: str = "") -> None:
    print(text)
    REPORT_LINES.append(text)


def section(title: str) -> None:
    log()
    log("═" * 70)
    log(f"  {title}")
    log("═" * 70)


def get_conn():
    return psycopg2.connect(
        host=DB_HOST, dbname=DB_NAME, user=DB_USER,
        password=DB_PASS, port=DB_PORT, sslmode="disable",
    )


def query_df(sql: str, params=None) -> pd.DataFrame:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params or [])
            cols = [d[0] for d in cur.description]
            return pd.DataFrame(cur.fetchall(), columns=cols)


def query_scalar(sql: str, params=None):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params or [])
            row = cur.fetchone()
            return row[0] if row else None


def fmt(value) -> str:
    if value is None:
        return "NULL"
    try:
        return f"{float(value):>15,.2f}"
    except (TypeError, ValueError):
        return str(value)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Schema checks
# ══════════════════════════════════════════════════════════════════════════════

def check_schema() -> None:
    section("1. SCHEMA — does total_sales_amount exist in staging / fact tables?")

    tables_to_check = [
        "stg_sales_reports",
        "fact_sales_lineitems",
        "fact_sales_transactions",
    ]
    col_sql = """
        SELECT table_name, column_name
        FROM information_schema.columns
        WHERE table_name = ANY(%s)
          AND column_name IN ('total_sales_amount', 'total_tax_ex')
        ORDER BY table_name, column_name
    """
    df = query_df(col_sql, [tables_to_check])

    if df.empty:
        log("  ⚠  No matching columns found at all — check connection or table names.")
        return

    for table in tables_to_check:
        cols = df[df["table_name"] == table]["column_name"].tolist()
        has_tax_ex      = "total_tax_ex" in cols
        has_sales_amt   = "total_sales_amount" in cols
        log(f"  {table}:")
        log(f"      total_tax_ex       : {'✅ present' if has_tax_ex    else '❌ MISSING'}")
        log(f"      total_sales_amount : {'✅ present' if has_sales_amt  else '❌ MISSING — not stored'}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Live definition of vw_sales_base
# ══════════════════════════════════════════════════════════════════════════════

def check_vw_sales_base_definition() -> None:
    section("2. vw_sales_base — live view definition from pg_views")

    defn = query_scalar(
        "SELECT definition FROM pg_views WHERE viewname = 'vw_sales_base'"
    )
    if defn is None:
        log("  ❌  vw_sales_base does NOT exist in the database.")
        log("      Revenue cannot be computed from it — check Power BI data source.")
    else:
        log("  ✅  vw_sales_base exists. Definition:")
        for line in defn.strip().splitlines():
            log(f"      {line}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Revenue from vw_sales_base (what Power BI shows)
# ══════════════════════════════════════════════════════════════════════════════

def check_vw_revenue() -> dict[str, float]:
    section("3. REVENUE FROM vw_sales_base (current Power BI figures)")

    # Check if the view exists first
    exists = query_scalar(
        "SELECT 1 FROM pg_views WHERE viewname = 'vw_sales_base'"
    )
    if not exists:
        log("  ⚠  vw_sales_base not found — skipping this section.")
        return {}

    # Try 'revenue' column; fall back to summing real_transaction_value if different
    try:
        df = query_df("""
            SELECT branch AS location, SUM(revenue) AS vw_revenue
            FROM vw_sales_base
            WHERE sale_date >= %s AND sale_date < %s
              AND branch = ANY(%s)
            GROUP BY branch
            ORDER BY branch
        """, [MAY_START, MAY_END, BRANCHES])
    except Exception as e:
        log(f"  ⚠  Could not query vw_sales_base.revenue: {e}")
        log("      Trying to list columns...")
        try:
            cols_df = query_df(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'vw_sales_base' ORDER BY ordinal_position"
            )
            log(f"      Columns: {cols_df['column_name'].tolist()}")
        except Exception:
            pass
        return {}

    result = {}
    for _, row in df.iterrows():
        loc = row["location"]
        rev = float(row["vw_revenue"] or 0)
        result[loc] = rev
        acct = ACCOUNTANT.get(loc, 0)
        gap  = acct - rev
        log(f"  {loc:<20} vw_revenue = {fmt(rev)}   accountant = {fmt(acct)}   gap = {fmt(gap)}")

    return result


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Revenue from raw fact tables (no view filters)
# ══════════════════════════════════════════════════════════════════════════════

def check_raw_fact_revenue() -> tuple[dict, dict]:
    section("4. REVENUE FROM RAW FACT TABLES (unfiltered cross-check)")

    # 4a — fact_sales_transactions
    df_txn = query_df("""
        SELECT location,
               COUNT(*)                        AS txn_count,
               SUM(real_transaction_value)      AS sum_real,
               SUM(pos_txn_sum)                 AS sum_pos,
               SUM(cashier_amount)              AS sum_cashier
        FROM fact_sales_transactions
        WHERE sale_date >= %s AND sale_date < %s
          AND location = ANY(%s)
        GROUP BY location
        ORDER BY location
    """, [MAY_START, MAY_END, BRANCHES])

    log("  fact_sales_transactions (ALL rows, no filter):")
    log(f"  {'Location':<20} {'Txns':>8} {'real_txn_value':>16} {'pos_txn_sum':>16} {'cashier_amt':>16}")
    log("  " + "-" * 78)
    txn_totals = {}
    for _, row in df_txn.iterrows():
        loc = row["location"]
        txn_totals[loc] = float(row["sum_real"] or 0)
        log(
            f"  {loc:<20} {int(row['txn_count'] or 0):>8,} "
            f"{fmt(row['sum_real']):>16} {fmt(row['sum_pos']):>16} {fmt(row['sum_cashier']):>16}"
        )

    log()

    # 4b — fact_sales_lineitems
    df_li = query_df("""
        SELECT location,
               COUNT(*)          AS row_count,
               SUM(total_tax_ex) AS sum_tax_ex
        FROM fact_sales_lineitems
        WHERE sale_date >= %s AND sale_date < %s
          AND location = ANY(%s)
        GROUP BY location
        ORDER BY location
    """, [MAY_START, MAY_END, BRANCHES])

    log("  fact_sales_lineitems (ALL rows, no filter):")
    log(f"  {'Location':<20} {'Rows':>10} {'sum(total_tax_ex)':>18}")
    log("  " + "-" * 52)
    li_totals = {}
    for _, row in df_li.iterrows():
        loc = row["location"]
        li_totals[loc] = float(row["sum_tax_ex"] or 0)
        log(f"  {loc:<20} {int(row['row_count'] or 0):>10,} {fmt(row['sum_tax_ex']):>18}")

    return txn_totals, li_totals


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Row exclusion audit
# ══════════════════════════════════════════════════════════════════════════════

def check_exclusions() -> dict[str, dict]:
    section("5. ROW EXCLUSION AUDIT — what each filter removes")

    exclusions: dict[str, dict] = {b: {} for b in BRANCHES}

    # 5a — real_transaction_value <= 0 (filtered by mv_transaction_master and vw_sales_base)
    df = query_df("""
        SELECT location,
               COUNT(*)                    AS txn_count,
               SUM(real_transaction_value) AS sum_value
        FROM fact_sales_transactions
        WHERE sale_date >= %s AND sale_date < %s
          AND location = ANY(%s)
          AND real_transaction_value <= 0
        GROUP BY location
    """, [MAY_START, MAY_END, BRANCHES])

    log("  A) Transactions with real_transaction_value <= 0 (filtered by mv / vw_sales_base):")
    for _, row in df.iterrows():
        loc = row["location"]
        val = float(row["sum_value"] or 0)
        exclusions[loc]["zero_or_negative_txn"] = val
        log(f"     {loc:<20} {int(row['txn_count'] or 0):>6} txns   value = {fmt(val)}")
    if df.empty:
        log("     (none found)")

    log()

    # 5b — GOODS / #NULL# / NULL descriptions (filtered by vw_sales_with_margin)
    df = query_df("""
        SELECT location,
               COUNT(*)          AS row_count,
               SUM(total_tax_ex) AS sum_value
        FROM fact_sales_lineitems
        WHERE sale_date >= %s AND sale_date < %s
          AND location = ANY(%s)
          AND (
              description ILIKE '%%GOODS%%'
              OR description ILIKE '%%#NULL#%%'
              OR description IS NULL
          )
        GROUP BY location
    """, [MAY_START, MAY_END, BRANCHES])

    log("  B) Line items with GOODS / #NULL# / NULL description (filtered by vw_sales_with_margin):")
    for _, row in df.iterrows():
        loc = row["location"]
        val = float(row["sum_value"] or 0)
        exclusions[loc]["goods_null_desc"] = val
        log(f"     {loc:<20} {int(row['row_count'] or 0):>6} rows   value = {fmt(val)}")
    if df.empty:
        log("     (none found)")

    log()

    # 5c — NULL total_tax_ex (rows that contribute 0 even if present)
    df = query_df("""
        SELECT location, COUNT(*) AS row_count
        FROM fact_sales_lineitems
        WHERE sale_date >= %s AND sale_date < %s
          AND location = ANY(%s)
          AND total_tax_ex IS NULL
        GROUP BY location
    """, [MAY_START, MAY_END, BRANCHES])

    log("  C) Line items with NULL total_tax_ex (contribute nothing to revenue):")
    for _, row in df.iterrows():
        loc = row["location"]
        cnt = int(row["row_count"] or 0)
        exclusions[loc]["null_tax_ex_rows"] = cnt
        log(f"     {loc:<20} {cnt:>6} rows")
    if df.empty:
        log("     (none found)")

    log()

    # 5d — Negative qty_sold (returns that reduce revenue totals)
    df = query_df("""
        SELECT location,
               COUNT(*)          AS row_count,
               SUM(total_tax_ex) AS sum_value
        FROM fact_sales_lineitems
        WHERE sale_date >= %s AND sale_date < %s
          AND location = ANY(%s)
          AND qty_sold < 0
        GROUP BY location
    """, [MAY_START, MAY_END, BRANCHES])

    log("  D) Line items with negative qty_sold (returns — reduce revenue totals):")
    for _, row in df.iterrows():
        loc = row["location"]
        val = float(row["sum_value"] or 0)
        exclusions[loc]["negative_qty_value"] = val
        log(f"     {loc:<20} {int(row['row_count'] or 0):>6} rows   value = {fmt(val)}")
    if df.empty:
        log("     (none found)")

    return exclusions


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Watermark check
# ══════════════════════════════════════════════════════════════════════════════

def check_watermarks() -> None:
    section("6. WATERMARK CHECK — are all May 2026 records loaded?")

    df = query_df("""
        SELECT branch, max_date_loaded, last_updated_at
        FROM fact_load_watermarks
        WHERE branch = ANY(%s)
        ORDER BY branch
    """, [BRANCHES])

    if df.empty:
        log("  ⚠  No watermarks found for PORTAL_2R or GALLERIA.")
        return

    for _, row in df.iterrows():
        mark = str(row["max_date_loaded"])
        updated = str(row["last_updated_at"])
        ok = row["max_date_loaded"] is not None and str(row["max_date_loaded"]) >= "2026-05-31"
        flag = "✅" if ok else "⚠  WATERMARK BEHIND MAY 31 — some rows may not be loaded!"
        log(f"  {row['branch']:<20} max_date_loaded = {mark}   last_updated = {updated}   {flag}")

    # Also check the max sale_date actually in fact tables
    log()
    log("  Max sale_date in fact_sales_transactions for each branch:")
    df2 = query_df("""
        SELECT location, MAX(sale_date) AS max_sale_date, COUNT(*) AS txn_count
        FROM fact_sales_transactions
        WHERE location = ANY(%s)
        GROUP BY location
        ORDER BY location
    """, [BRANCHES])
    for _, row in df2.iterrows():
        log(f"     {row['location']:<20} max_sale_date = {row['max_sale_date']}   txns = {int(row['txn_count'] or 0):,}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — CSV cross-check: Total (Tax Ex) vs Total Sales Amount
# ══════════════════════════════════════════════════════════════════════════════

def check_csv_column_gap() -> None:
    section("7. CSV CROSS-CHECK — Total (Tax Ex) vs Total Sales Amount")

    if not CSV_PATH.exists():
        log(f"  ⚠  CSV not found at {CSV_PATH}")
        log("      Run etl_local.py first to regenerate it.")
        return

    log(f"  Reading {CSV_PATH.name} ...")
    try:
        df = pd.read_csv(CSV_PATH, dtype=str, low_memory=False)
    except Exception as e:
        log(f"  ❌  Could not read CSV: {e}")
        return

    log(f"  Columns present: {list(df.columns)}")

    has_tax_ex   = "Total (Tax Ex)" in df.columns
    has_sales_amt = "Total Sales Amount" in df.columns

    log(f"  'Total (Tax Ex)'      : {'✅ present' if has_tax_ex    else '❌ MISSING'}")
    log(f"  'Total Sales Amount'  : {'✅ present' if has_sales_amt  else '❌ MISSING'}")

    if not (has_tax_ex and has_sales_amt):
        log("  Cannot compare columns — at least one is missing from the CSV.")
        return

    # Filter to May 2026 for the two branches
    if "Sale_Date_Str" in df.columns:
        date_col = "Sale_Date_Str"
    elif "Sale_Date" in df.columns:
        date_col = "Sale_Date"
    else:
        log("  ⚠  No date column found (Sale_Date_Str / Sale_Date). Cannot filter to May 2026.")
        return

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    may_mask = (df[date_col] >= "2026-05-01") & (df[date_col] < "2026-06-01")
    loc_col  = "Location" if "Location" in df.columns else None

    if loc_col:
        branch_mask = df[loc_col].isin(BRANCHES)
        df_may = df[may_mask & branch_mask].copy()
    else:
        df_may = df[may_mask].copy()

    log(f"  May 2026 rows in CSV (Portal 2R + Galleria): {len(df_may):,}")

    if df_may.empty:
        log("  No May 2026 rows — CSV may not have been regenerated yet.")
        return

    df_may["_tax_ex"]    = pd.to_numeric(df_may["Total (Tax Ex)"],     errors="coerce").fillna(0)
    df_may["_sales_amt"] = pd.to_numeric(df_may["Total Sales Amount"], errors="coerce").fillna(0)

    if loc_col:
        grp = df_may.groupby(loc_col)[["_tax_ex", "_sales_amt"]].sum()
        log()
        log(f"  {'Location':<20} {'Sum Tax Ex':>18} {'Sum Sales Amt':>18} {'Difference':>14}")
        log("  " + "-" * 74)
        for loc, row in grp.iterrows():
            diff = row["_sales_amt"] - row["_tax_ex"]
            log(f"  {loc:<20} {fmt(row['_tax_ex']):>18} {fmt(row['_sales_amt']):>18} {fmt(diff):>14}")
    else:
        total_tax_ex   = df_may["_tax_ex"].sum()
        total_sales_amt = df_may["_sales_amt"].sum()
        diff = total_sales_amt - total_tax_ex
        log(f"  All branches  Sum Tax Ex = {fmt(total_tax_ex)}  Sum Sales Amt = {fmt(total_sales_amt)}  Diff = {fmt(diff)}")

    null_sales_amt = df_may["_sales_amt"].isna().sum()
    zero_sales_amt = (df_may["Total Sales Amount"].astype(str).str.strip().isin(["", "0", "0.0", "nan", "None"])).sum()
    log()
    log(f"  Rows where Total Sales Amount is NULL/empty/zero: {null_sales_amt + zero_sales_amt:,}")
    log("  (These are likely old-format rows that predate the column)")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — Final reconciliation summary
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(vw_revenue: dict, txn_totals: dict, exclusions: dict) -> None:
    section("8. FINAL RECONCILIATION SUMMARY")

    log(f"  {'Location':<20} {'DB (vw_base)':>16} {'Accountant':>16} {'Gap':>14} {'Gap %':>8}")
    log("  " + "-" * 78)
    for loc in BRANCHES:
        db_rev = vw_revenue.get(loc) or txn_totals.get(loc, 0.0)
        acct   = ACCOUNTANT[loc]
        gap    = acct - db_rev
        gap_pct = (gap / acct * 100) if acct else 0
        log(f"  {loc:<20} {fmt(db_rev):>16} {fmt(acct):>16} {fmt(gap):>14} {gap_pct:>7.2f}%")

    log()
    log("  Likely causes of the gap (from audit):")
    log()
    log("  1. PRIMARY — Revenue computed from 'Total (Tax Ex)' not 'Total Sales Amount'")
    log("     stg_sales_reports has NO total_sales_amount column.")
    log("     All revenue in the DB is tax-exclusive. Accountants use tax-inclusive totals")
    log("     for VAT-able items. The DB needs a total_sales_amount column added to:")
    log("       • stg_sales_reports  (ALTER TABLE + re-ingest)")
    log("       • fact_sales_lineitems (add column + recompute pos_txn_sum)")
    log("       • vw_sales_base (switch revenue to SUM(total_sales_amount))")
    log()
    log("  2. SECONDARY — Rows excluded by filters (see Section 5 above)")
    log("     • real_transaction_value <= 0 (zero-value/refund transactions removed from mv)")
    log("     • GOODS / #NULL# / NULL description rows removed from vw_sales_with_margin")
    log("     Check whether the accountants include these transactions in their totals.")
    log()
    log("  3. CHECK — Negative qty_sold returns (Section 5D) reduce revenue totals.")
    log("     If accountants count gross sales before returns, this adds to the gap.")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    log("=" * 70)
    log("  PORTAL PHARMACY — MAY 2026 REVENUE DISCREPANCY AUDIT")
    log(f"  Branches : {', '.join(BRANCHES)}")
    log(f"  Period   : {MAY_START} to {MAY_END}")
    log("=" * 70)

    log(f"\n  Connecting to {DB_HOST} / {DB_NAME} as {DB_USER} ...")
    try:
        conn = get_conn()
        conn.close()
        log("  ✅  Connection OK\n")
    except Exception as e:
        log(f"  ❌  Cannot connect: {e}")
        sys.exit(1)

    check_schema()
    check_vw_sales_base_definition()
    vw_revenue = check_vw_revenue()
    txn_totals, _li_totals = check_raw_fact_revenue()
    exclusions = check_exclusions()
    check_watermarks()
    check_csv_column_gap()
    print_summary(vw_revenue, txn_totals, exclusions)

    log()
    log("=" * 70)
    log("  AUDIT COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()
