import pandas as pd
import numpy as np
import glob
import os
from pathlib import Path

# ✅ IMPORT PATHS
from Portal_ML_V4.src.config.settings import (
    BASE_DIR,
    PROCESSED_DATA_DIR,
)

# ==========================================
# 1. CONFIGURATION
# ==========================================
# Path to your raw POS data folder
POS_RAW_DIR = BASE_DIR / "data" / "01_raw" / "pos_data"
OUTPUT_FILE = PROCESSED_DATA_DIR / "pos_data" / "fact_pos_sales_enriched.csv"

# ==========================================
# 2. HELPER: FILE SCANNERS
# ==========================================
def get_latest_file(directory, pattern):
    """
    Scans directory for files matching a pattern (e.g., 'sales*') 
    and returns the one with the most recent modification time.
    """
    search_path = os.path.join(directory, pattern)
    files = glob.glob(search_path)
    
    if not files:
        # Debugging help: Print what it found vs what it wanted
        print(f"⚠️  No files found matching: {pattern}")
        print(f"    Searching in: {directory}")
        return None

    # Sort by modification time (Latest first)
    latest_file = max(files, key=os.path.getmtime)
    print(f"   📂 Auto-Detected File: {os.path.basename(latest_file)}")
    return Path(latest_file)

# ==========================================
# 3. MAIN ETL PIPELINE
# ==========================================
def run_pos_etl():
    print("🛒 STARTING POS DATA PIPELINE...")

    if not POS_RAW_DIR.exists():
        print(f"❌ Missing Directory: {POS_RAW_DIR}")
        return

    # --- STEP 1: LOAD CASHIER REPORT (The "Who" & "Context") ---
    # Matches: "Portal 2R  Daily Cashier report  Jan 2026(25).csv"
    cashier_path = get_latest_file(POS_RAW_DIR, "Portal 2R*Cashier*Jan 2026*.csv")
    
    if cashier_path is None:
        print("❌ CRITICAL: Could not find Cashier Report.")
        return

    print("   📥 Loading Cashier Report...")
    # 'low_memory=False' helps pandas guess types better for mixed columns
    df_cashier = pd.read_csv(cashier_path, low_memory=False)

    # Clean Cashier Data
    # 1. Ensure Join Key is String and clean (remove .0)
    if 'Receipt Txn No' in df_cashier.columns:
        df_cashier['Receipt Txn No'] = df_cashier['Receipt Txn No'].astype(str).str.replace(r'\.0$', '', regex=True)

    # 2. 🕒 TIME CONVERSION FIX (Handle 20.14 -> 20:14:00)
    if 'Time' in df_cashier.columns:
        def fix_pos_time(val):
            try:
                # Force to float to handle "10.5" string or number
                f_val = float(val)
                # Format with 2 decimal places (10.5 -> "10.50", 10.05 -> "10.05")
                # Then replace dot with colon
                return "{:.2f}".format(f_val).replace('.', ':')
            except (ValueError, TypeError):
                return None

        # Apply formatting
        df_cashier['Time_Str'] = df_cashier['Time'].apply(fix_pos_time)
        # Convert to actual Time object
        df_cashier['Time'] = pd.to_datetime(df_cashier['Time_Str'], format='%H:%M', errors='coerce').dt.time
        # Clean up temp column
        df_cashier.drop(columns=['Time_Str'], inplace=True)
    
    # 3. Select only necessary columns
    cashier_cols = [
        'Receipt Txn No', 'Sales Rep', 'Client Name', 'Phone Number', 
        'Txn Type', 'Ordered Via', 'Time'
    ]
    # Keep only columns that actually exist to prevent KeyErrors
    df_cashier = df_cashier[[c for c in cashier_cols if c in df_cashier.columns]]

    # --- STEP 2: LOAD SALES REPORT (The "What" & "Inventory") ---
    # Matches: "sales25jan2025(in).csv"
    sales_path = get_latest_file(POS_RAW_DIR, "sales*jan2025*.csv")

    if sales_path is None:
        print("❌ CRITICAL: Could not find Sales Details Report.")
        return

    print("   📥 Loading Sales Details...")
    df_sales = pd.read_csv(sales_path, low_memory=False)

    # Clean Sales Data
    # 1. Ensure Join Key is String
    if 'Transaction ID' in df_sales.columns:
        df_sales['Transaction ID'] = df_sales['Transaction ID'].astype(str).str.replace(r'\.0$', '', regex=True)
    else:
        print(f"⚠️  Warning: 'Transaction ID' column missing. Columns found: {list(df_sales.columns)}")

    # --- STEP 3: MERGE (Index Match Logic) ---
    print("   🔗 Merging Sales Items with Cashier Details...")
    
    # Left Join: We want every line item from Sales, matched with Cashier info
    if 'Transaction ID' in df_sales.columns and 'Receipt Txn No' in df_cashier.columns:
        df_merged = pd.merge(
            df_sales,
            df_cashier,
            left_on='Transaction ID',
            right_on='Receipt Txn No',
            how='left'
        )
    else:
        print("❌ Cannot merge: Missing Key Columns (Transaction ID / Receipt Txn No)")
        return

    # --- STEP 4: ENRICHMENT & CLEANING ---
    
    # 1. Date Parsing (Handling format like: #2026-01-25 17:55:08#)
    if 'Date Sold' in df_merged.columns:
        # Remove hash signs if present
        df_merged['Date_Clean'] = df_merged['Date Sold'].astype(str).str.replace('#', '')
        df_merged['Sale_DateTime'] = pd.to_datetime(df_merged['Date_Clean'], errors='coerce')
        df_merged['Sale_Date'] = df_merged['Sale_DateTime'].dt.date
    
    # 2. Fill Missing Sales Reps
    if 'Sales Rep' in df_merged.columns:
        df_merged['Sales Rep'] = df_merged['Sales Rep'].fillna("Unknown/System")
    
    # 3. Validation
    if 'Total (Tax Ex)' in df_merged.columns:
        total_rev = df_merged['Total (Tax Ex)'].sum()
        print(f"   💰 Total POS Revenue Processed: {total_rev:,.2f}")
    
    print(f"   🧾 Unique Transactions: {df_merged['Transaction ID'].nunique()}")

    # --- STEP 5: EXPORT ---
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    df_merged.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ POS Data Pipeline Complete. Saved to: {OUTPUT_FILE}")
    
    # Preview
    print("\n   📊 Preview:")
    cols_preview = ['Transaction ID', 'Item', 'Total (Tax Ex)', 'Sales Rep', 'Client Name']
    print(df_merged[[c for c in cols_preview if c in df_merged.columns]].head())

if __name__ == "__main__":
    run_pos_etl()