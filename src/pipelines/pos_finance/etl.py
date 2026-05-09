"""
etl.py
======
POS ETL V5 — reads directly from DB staging tables.

Source of truth is now PostgreSQL (stg_sales_reports + stg_cashier_reports),
populated by the SharePoint downloader pipeline. Physical files are no longer
read here — the downloader handles that.

Processing logic is unchanged from V4:
  - Per-branch sales + cashier load
  - Cashier aggregation and name cleaning
  - Memory-safe merge with chunked fallback
  - Strict deduplication
  - Transaction_Total calculation
  - Output to all_locations_sales.csv for attribution pipeline
"""

import gc
import os

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool
from sqlalchemy.engine import URL
from dotenv import load_dotenv

from Portal_ML_V4.src.config.settings import PROCESSED_DATA_DIR, BASE_DIR
from Portal_ML_V4.src.utils.name_cleaner import clean_name_series
from Portal_ML_V4.sharepoint.sharepoint_parser import get_branch_watermark

load_dotenv(dotenv_path=BASE_DIR / ".env", override=True)


# ==========================================
# 1. CONFIGURATION
# ==========================================

OUTPUT_DIR  = PROCESSED_DATA_DIR / "pos_data"
OUTPUT_FILE = OUTPUT_DIR / "all_locations_sales.csv"

# Maps stg_sales_reports.branch values → output Location label
# Branch values in DB match what sharepoint_parser inserts (SharePoint folder names)
BRANCH_TO_LOCATION = {
    "Centurion 2R": "CENTURION 2R",
    "Galleria":     "GALLERIA",
    "Milele":       "MILELE",
    "ABC":          "PHARMART_ABC",
    "Portal 2R":    "PORTAL 2R",
    "Portal CBD":   "PORTAL CBD",
}


# ==========================================
# 2. ENGINE
# ==========================================

def get_engine():
    return create_engine(
        URL.create(
            drivername = "postgresql+psycopg2",
            username   = os.getenv("DB_USER"),
            password   = os.getenv("DB_PASSWORD"),
            host       = os.getenv("DB_HOST", "localhost"),
            port       = int(os.getenv("DB_PORT", 5432)),
            database   = os.getenv("DB_NAME"),
        ),
        poolclass = NullPool,
    )


# ==========================================
# 3. HELPERS
# ==========================================

def clean_id_col(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r'\.0$', '', regex=True).str.strip()


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast columns to save RAM — critical for large branch tables."""
    for col in df.columns:
        col_type = df[col].dtype
        if col_type == 'object':
            if df[col].nunique() / max(len(df), 1) < 0.5:
                df[col] = df[col].astype('category')
        elif col_type == 'float64':
            df[col] = pd.to_numeric(df[col], downcast='float')
        elif col_type == 'int64':
            df[col] = pd.to_numeric(df[col], downcast='integer')
    return df


# ==========================================
# 4. DB LOAD FUNCTIONS
# ==========================================

def load_sales_from_db(engine, branch: str, after_date=None, after_datetime=None) -> pd.DataFrame:
    """
    Query stg_sales_reports for one branch.
    Filters by after_date (watermark) if provided — avoids reloading history.
    Returns DataFrame with column names matching the original POS file format.
    """
    if after_datetime:
        sql = text("""
            SELECT
                department     AS "Department",
                category       AS "Category",
                item           AS "Item",
                description    AS "Description",
                on_hand        AS "On Hand",
                last_sold      AS "Last Sold",
                qty_sold       AS "Qty Sold",
                total_tax_ex   AS "Total (Tax Ex)",
                transaction_id AS "Transaction ID",
                date_sold      AS "Date Sold",
                sale_datetime  AS "Date_Obj"
            FROM stg_sales_reports
            WHERE branch = :branch
              AND sale_datetime > :after_datetime
        """)
        params = {"branch": branch, "after_datetime": after_datetime}
    
    elif after_date:
        sql = text("""
            SELECT
                department     AS "Department",
                category       AS "Category",
                item           AS "Item",
                description    AS "Description",
                on_hand        AS "On Hand",
                last_sold      AS "Last Sold",
                qty_sold       AS "Qty Sold",
                total_tax_ex   AS "Total (Tax Ex)",
                transaction_id AS "Transaction ID",
                date_sold      AS "Date Sold",
                sale_datetime  AS "Date_Obj"
            FROM stg_sales_reports
            WHERE branch = :branch
              AND date_sold > :after_date
        """)
        params = {"branch": branch, "after_date": after_date}
    else:
        sql = text("""
            SELECT
                department     AS "Department",
                category       AS "Category",
                item           AS "Item",
                description    AS "Description",
                on_hand        AS "On Hand",
                last_sold      AS "Last Sold",
                qty_sold       AS "Qty Sold",
                total_tax_ex   AS "Total (Tax Ex)",
                transaction_id AS "Transaction ID",
                date_sold      AS "Date Sold",
                sale_datetime  AS "Date_Obj"
            FROM stg_sales_reports
            WHERE branch = :branch
        """)
        params = {"branch": branch}

    try:
        df = pd.read_sql(sql, engine, params=params)
        df['Date_Obj'] = pd.to_datetime(df['Date_Obj'], errors='coerce')
        df['Date Sold'] = df['Date_Obj'].dt.date.astype(str)
        df['Transaction ID'] = clean_id_col(df['Transaction ID'].fillna(''))
        return df
    except Exception as e:
        print(f"    ❌ DB error loading sales for {branch}: {e}")
        return pd.DataFrame()


def load_cashier_from_db(engine, branch: str, after_date=None) -> pd.DataFrame:
    """
    Query stg_cashier_reports for one branch.
    Filters by after_date (watermark) if provided.
    Returns DataFrame with column names matching original cashier file format.
    """
    if after_date:
        sql = text("""
            SELECT
                receipt_txn_no AS "Receipt Txn No",
                amount         AS "Amount",
                txn_costs      AS "Txn Costs",
                txn_time       AS "Time",
                txn_type       AS "Txn Type",
                ordered_via    AS "Ordered Via",
                client_name    AS "Client Name",
                phone_number   AS "Phone Number",
                sales_rep      AS "Sales Rep"
            FROM stg_cashier_reports
            WHERE branch = :branch
              AND transaction_date > :after_date
        """)
        params = {"branch": branch, "after_date": after_date}
    else:
        sql = text("""
            SELECT
                receipt_txn_no AS "Receipt Txn No",
                amount         AS "Amount",
                txn_costs      AS "Txn Costs",
                txn_time       AS "Time",
                txn_type       AS "Txn Type",
                ordered_via    AS "Ordered Via",
                client_name    AS "Client Name",
                phone_number   AS "Phone Number",
                sales_rep      AS "Sales Rep"
            FROM stg_cashier_reports
            WHERE branch = :branch
        """)
        params = {"branch": branch}

    try:
        df = pd.read_sql(sql, engine, params=params)
        return df
    except Exception as e:
        print(f"    ❌ DB error loading cashier for {branch}: {e}")
        return pd.DataFrame()


# ==========================================
# 4. MAIN PIPELINE
# ==========================================

def run_pos_etl_v3():
    print("🌍 STARTING POS ETL V5 (DB Source Mode)...")
    print("📅 Reading from stg_sales_reports + stg_cashier_reports")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    engine = get_engine()
    master_list = []

    for branch, location_label in BRANCH_TO_LOCATION.items():
        print(f"\n📍 PROCESSING BRANCH: {location_label}")

        # Read watermark — only load rows after last successful run
        watermark, watermark_dt = get_branch_watermark(branch)
        if watermark:
            print(f"    📅 Watermark: loading rows after {watermark} (datetime: {watermark_dt})")
        else:
            print("    📅 No watermark — loading all data")

        # A. LOAD SALES FROM DB
        df_sales = load_sales_from_db(engine, branch, after_date=watermark, after_datetime=watermark_dt)
        if df_sales.empty:
            print(f"    ⚠️ No new sales data in DB for branch '{branch}'")
            continue

        df_sales = optimize_dtypes(df_sales)
        print(f"    ✅ Sales Loaded: {len(df_sales):,} rows")

        # B. LOAD CASHIER FROM DB
        df_cashier = load_cashier_from_db(engine, branch, after_date=watermark)

        # C. PREPARE CASHIER
        if not df_cashier.empty:
            df_cashier = optimize_dtypes(df_cashier)

            if 'Client Name' in df_cashier.columns:
                df_cashier['Client Name'] = clean_name_series(df_cashier['Client Name'])

            if 'Time' in df_cashier.columns:
                df_cashier['Time'] = df_cashier['Time'].astype(str).fillna("00:00:00")

            for col in ['Amount', 'Txn Costs']:
                if col in df_cashier.columns:
                    df_cashier[col] = (
                        df_cashier[col].astype(str)
                        .str.replace(',', '', regex=False)
                        .str.replace(r'[^\d\.\-]', '', regex=True)
                    )
                    df_cashier[col] = pd.to_numeric(
                        df_cashier[col], errors='coerce'
                    ).fillna(0.0)

            # Aggregate to one row per transaction
            agg_rules = {'Amount': 'sum', 'Txn Costs': 'sum'}
            if 'Time' in df_cashier.columns:
                agg_rules['Time'] = 'min'
            meta_cols = [
                c for c in df_cashier.columns
                if c not in ['Receipt Txn No', 'Amount', 'Txn Costs', 'Time']
            ]
            for c in meta_cols:
                agg_rules[c] = 'first'

            if 'Receipt Txn No' in df_cashier.columns:
                df_cashier['Receipt Txn No'] = clean_id_col(
                    df_cashier['Receipt Txn No'].fillna('')
                )

            df_cashier = df_cashier.groupby(
                'Receipt Txn No', as_index=False
            ).agg(agg_rules)

            # D. MERGE — robust with chunked fallback for large branches
            print("    🔗 Running Safe Merge...")
            try:
                df_merged = pd.merge(
                    df_sales,
                    df_cashier,
                    left_on='Transaction ID',
                    right_on='Receipt Txn No',
                    how='left',
                    copy=False,
                )
            except (MemoryError, np.core._exceptions._ArrayMemoryError):
                print("    ⚠️ Memory limit hit — switching to chunked merge...")
                chunks = []
                chunk_size = 50_000
                for i in range(0, len(df_sales), chunk_size):
                    chunk = df_sales.iloc[i:i + chunk_size]
                    merged_chunk = pd.merge(
                        chunk,
                        df_cashier,
                        left_on='Transaction ID',
                        right_on='Receipt Txn No',
                        how='left',
                        copy=False,
                    )
                    chunks.append(merged_chunk)
                    gc.collect()
                df_merged = pd.concat(chunks, ignore_index=True)

            df_merged['Audit_Status'] = np.where(
                df_merged['Receipt Txn No'].isna(), 'No Cashier Data', 'Matched'
            )
        else:
            df_merged = df_sales.copy()
            df_merged['Audit_Status'] = 'No Cashier Data'

        df_merged['Location'] = location_label
        master_list.append(df_merged)

        del df_sales, df_cashier, df_merged
        gc.collect()

    # -------------------------------------------------------------------------
    # 5. FINAL STACK & STRICT DEDUPLICATION
    # -------------------------------------------------------------------------
    if not master_list:
        print("\n❌ No data processed — are the DB staging tables populated?")
        return

    print("\n🏗️  Stacking All Locations...")
    final_df = pd.concat(master_list, ignore_index=True)
    gc.collect()

    # Sale_Date columns from Date_Obj
    if 'Date_Obj' in final_df.columns: 
        final_df['Sale_Date']     = pd.to_datetime(
            final_df['Date_Obj'], errors='coerce'
        ).dt.normalize()
        final_df['Sale_Date_Str'] = pd.to_datetime(
            final_df['Date_Obj'], errors='coerce'
        ).dt.strftime('%Y-%m-%d')
        final_df.drop(columns=['Date_Obj'], inplace=True)
        gc.collect()

    before_len = len(final_df)

    # Strict deduplication
    dedup_cols = ['Transaction ID', 'Date Sold', 'Total (Tax Ex)', 'Description']
    actual_dedup_cols = [c for c in dedup_cols if c in final_df.columns]
    final_df.drop_duplicates(subset=actual_dedup_cols, inplace=True)

    # Transaction_Total — sum of all line items per transaction
    if 'Total (Tax Ex)' in final_df.columns and 'Transaction ID' in final_df.columns:
        final_df['Total (Tax Ex)'] = pd.to_numeric(
            final_df['Total (Tax Ex)'], errors='coerce'
        ).fillna(0)
        final_df['Transaction_Total'] = (
            final_df.groupby('Transaction ID')['Total (Tax Ex)'].transform('sum')
        )

    after_len = len(final_df)

    final_df.to_csv(OUTPUT_FILE, index=False)

    print(f"🚀 PIPELINE SUCCESS!")
    print(f"   📊 Rows Before Deduplication : {before_len:,}")
    print(f"   ✂️  Duplicates Removed        : {before_len - after_len:,}")
    print(f"   🧾 Total Valid Rows           : {after_len:,}")
    print(f"   📂 Output                    : {OUTPUT_FILE}")

    engine.dispose()


if __name__ == "__main__":
    run_pos_etl_v3()