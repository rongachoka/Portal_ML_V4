import pandas as pd
import numpy as np
import re
import os
import warnings
from pathlib import Path

from Portal_ML_V4.src.config.settings import (
    BASE_DIR,
    PROCESSED_DATA_DIR,
)
from Portal_ML_V4.src.utils.phone import normalize_phone, clean_id_excel_safe

warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

# ==========================================
# 1. CONFIGURATION
# ==========================================
WEBSITE_RAW_PATH = BASE_DIR / "data" / "01_raw" / "website" / "portal_order_with prices.csv"

OUTPUT_DIR        = PROCESSED_DATA_DIR / "website_data"
OUTPUT_FACT       = OUTPUT_DIR / "fact_website_orders.csv"
OUTPUT_CLIENTS    = OUTPUT_DIR / "website_dim_clients.csv"
OUTPUT_LINEITEMS  = OUTPUT_DIR / "website_fact_lineitems.csv"   # one row per order line — used by run_client_export

# Canonical column names we expect in the CSV
EXPECTED_COLS = [
    'order_id', 'client_name', 'phone_number', 'email',
    'product_id', 'product_bought', 'product_model',
    'quantity', 'unit_price', 'line_total',
    'purchase_date', 'purchased_at', 'order_status'
]

# Statuses that count as a real sale
COMPLETED_STATUSES = {'complete', 'completed', 'processing', 'shipped', 'delivered'}

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def parse_website_date(series):
    """
    Handles the messy DD/MM/YYYY and DD/MM/YYYY HH:MM formats
    found in purchase_date and purchased_at columns.
    """
    return pd.to_datetime(series, dayfirst=True, errors='coerce')


def validate_columns(df):
    """Warn about any expected columns that are missing."""
    missing = [c for c in EXPECTED_COLS if c not in df.columns]
    if missing:
        print(f"   ⚠️  Missing expected columns: {missing}")
        print(f"       Columns found: {list(df.columns)}")
    return missing


# ==========================================
# 3. LOAD & CLEAN
# ==========================================

def load_website_orders():
    """
    Loads the raw website orders CSV, cleans every column,
    and returns a standardised DataFrame.
    """
    if not WEBSITE_RAW_PATH.exists():
        print(f"   ❌ File not found: {WEBSITE_RAW_PATH}")
        return pd.DataFrame()

    print(f"   📂 Loading: {WEBSITE_RAW_PATH.name}")

    # low_memory=False avoids dtype inference warnings on mixed columns
    df = pd.read_csv(WEBSITE_RAW_PATH, low_memory=False)

    # Normalise column names: lowercase + strip
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

    missing = validate_columns(df)
    if len(missing) == len(EXPECTED_COLS):
        print("   ❌ File structure looks completely wrong. Aborting.")
        return pd.DataFrame()

    # ----- Dates -----
    if 'purchase_date' in df.columns:
        df['purchase_date'] = parse_website_date(df['purchase_date'])
    if 'purchased_at' in df.columns:
        df['purchased_at'] = parse_website_date(df['purchased_at'])

    # Use purchased_at (has time) as master timestamp; fall back to purchase_date
    df['order_timestamp'] = df.get('purchased_at', df.get('purchase_date'))
    df['order_date']      = df['purchase_date'].dt.normalize() if 'purchase_date' in df.columns else df['order_timestamp'].dt.normalize()
    df['order_date_str']  = df['order_date'].dt.strftime('%Y-%m-%d')

    # ----- Numeric -----
    for col in ['quantity', 'unit_price', 'line_total', 'order_id', 'product_id']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df['quantity']   = df['quantity'].fillna(0).astype(int)
    df['unit_price'] = df['unit_price'].fillna(0.0)
    df['line_total'] = df['line_total'].fillna(0.0)

    # ----- Phone -----
    if 'phone_number' in df.columns:
        df['phone_clean']      = df['phone_number'].apply(normalize_phone)
        df['phone_excel_safe'] = df['phone_number'].apply(clean_id_excel_safe)

    # ----- Status normalisation -----
    if 'order_status' in df.columns:
        df['order_status_clean'] = df['order_status'].astype(str).str.strip().str.lower()
        df['is_completed']       = df['order_status_clean'].isin(COMPLETED_STATUSES).astype(int)
    else:
        df['order_status_clean'] = 'unknown'
        df['is_completed']       = 0

    # ----- Text cleanup -----
    for col in ['client_name', 'product_bought', 'product_model', 'email']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace('nan', np.nan)

    # ----- Source tag (used later in client list) -----
    df['source'] = 'Website Order'

    # ----- Deduplication -----
    # A product line on the same order should only appear once
    dedup_cols = [c for c in ['order_id', 'product_id', 'quantity', 'line_total'] if c in df.columns]
    before = len(df)
    df = df.drop_duplicates(subset=dedup_cols)
    dupes_removed = before - len(df)
    if dupes_removed:
        print(f"   ✂️  Removed {dupes_removed:,} duplicate rows.")

    return df


# ==========================================
# 4. DIAGNOSTICS
# ==========================================

def print_diagnostics(df):
    print("\n   📊 WEBSITE ORDERS DIAGNOSTICS")
    print(f"      Total order lines       : {len(df):,}")
    print(f"      Unique orders           : {df['order_id'].nunique():,}")
    print(f"      Unique clients (phone)  : {df['phone_clean'].nunique():,}")
    print(f"      Unique clients (email)  : {df['email'].nunique():,}")
    print(f"      Date range              : {df['order_date'].min().date()} → {df['order_date'].max().date()}")
    print(f"      Total revenue (complete): {df[df['is_completed']==1]['line_total'].sum():,.0f}")

    print("\n      Order Status Breakdown:")
    status_counts = df['order_status_clean'].value_counts()
    for status, count in status_counts.items():
        print(f"         {status:<20} {count:,} rows")

    print("\n      Top 10 Products by Revenue:")
    top_prods = (
        df[df['is_completed'] == 1]
        .groupby('product_bought')['line_total']
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )
    for prod, rev in top_prods.items():
        print(f"         {prod:<50} KES {rev:,.0f}")

    print("\n      Top 10 Clients by Revenue:")
    top_clients = (
        df[df['is_completed'] == 1]
        .groupby('client_name')['line_total']
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )
    for client, rev in top_clients.items():
        print(f"         {client:<50} KES {rev:,.0f}")


# ==========================================
# 5. BUILD CLIENT SUMMARY
# ==========================================

def build_website_client_list(df):
    """
    Creates three outputs:
    - Fact lineitems : one row per order × product line (order_date intact) ← used by run_client_export
    - Item-level     : one row per client × product (aggregated with All_Order_Dates)
    - Client summary : one row per client (fully rolled-up)
    """
    completed = df[df['is_completed'] == 1].copy()

    if completed.empty:
        print("   ⚠️  No completed orders to summarise.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # --- A. Fact lineitems — one row per order line, order_date intact ---
    # This is the source of truth for Purchase History in the client export.
    # Keeps: phone, name, email, order_id, order_date, product, qty, unit_price, line_total
    fact_cols = [c for c in [
        'phone_clean', 'client_name', 'email',
        'order_id', 'order_date', 'order_date_str',
        'product_bought', 'quantity', 'unit_price', 'line_total', 'source'
    ] if c in completed.columns]
    fact_lineitems = (
        completed[fact_cols]
        .sort_values(['phone_clean', 'order_date', 'order_id'])
        .reset_index(drop=True)
    )

    # --- B. Item-level history (aggregated — kept for reference / All_Order_Dates) ---
    item_level = (
        completed
        .groupby(['phone_clean', 'client_name', 'email', 'product_bought'], dropna=False)
        .agg(
            Times_Ordered      = ('order_id',       'nunique'),
            Total_Qty          = ('quantity',        'sum'),
            Total_Spend        = ('line_total',      'sum'),
            Unit_Price         = ('unit_price',      'first'),
            First_Order_Date   = ('order_date',      'min'),
            Last_Order_Date    = ('order_date',      'max'),
            All_Order_Dates    = ('order_date_str',  lambda x: ' | '.join(sorted(x.dropna().unique())))
        )
        .reset_index()
        .sort_values(['phone_clean', 'Last_Order_Date'], ascending=[True, False])
    )
    item_level['source'] = 'Website Order'

    # --- B. Client summary ---
    client_summary = (
        completed
        .groupby(['phone_clean', 'client_name', 'email'], dropna=False)
        .agg(
            Total_Orders       = ('order_id',      'nunique'),
            Total_Items_Bought = ('quantity',       'sum'),
            Total_Spend        = ('line_total',     'sum'),
            First_Order        = ('order_date',     'min'),
            Last_Order         = ('order_date',     'max'),
            Products_Bought    = ('product_bought', lambda x: ' | '.join(sorted(x.dropna().unique())))
        )
        .reset_index()
        .sort_values('Total_Spend', ascending=False)
    )
    client_summary['source'] = 'Website Order'

    return fact_lineitems, item_level, client_summary


# ==========================================
# 6. MAIN PIPELINE
# ==========================================

def run_website_orders_etl():
    print("=" * 60)
    print("🌐 WEBSITE ORDERS ETL")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load & clean
    df = load_website_orders()
    if df.empty:
        print("❌ No data loaded. Exiting.")
        return

    print(f"   ✅ Loaded {len(df):,} rows.")

    # Diagnostics — gives you full visibility before touching the client list
    print_diagnostics(df)

    # Save cleaned fact table (all orders, all statuses)
    df.to_csv(OUTPUT_FACT, index=False)
    print(f"\n   💾 Fact table saved: {OUTPUT_FACT}")

    # Build client summaries
    df_lineitems, df_items, df_clients = build_website_client_list(df)

    if not df_lineitems.empty:
        df_lineitems.to_csv(OUTPUT_LINEITEMS, index=False)
        print(f"   💾 Fact lineitems saved  : {OUTPUT_LINEITEMS}")
        print(f"      → {len(df_lineitems):,} order line rows (one row per transaction line, order_date intact)")

    if not df_items.empty:
        df_items.to_csv(OUTPUT_DIR / "website_item_history.csv", index=False)
        print(f"   💾 Item history saved    : {OUTPUT_DIR / 'website_item_history.csv'}")
        print(f"      → {len(df_items):,} product-client rows (aggregated)")

    if not df_clients.empty:
        df_clients.to_csv(OUTPUT_CLIENTS, index=False)
        print(f"   💾 Client summary saved: {OUTPUT_CLIENTS}")
        print(f"      → {len(df_clients):,} unique website clients")

    print("\n✅ WEBSITE ORDERS ETL COMPLETE.")

    return df  # return for inspection / chaining


if __name__ == "__main__":
    run_website_orders_etl()