import pandas as pd
import numpy as np
import re
from pathlib import Path
import warnings

# ✅ IMPORT PATHS
from Portal_ML_V4.src.config.settings import (
    PROCESSED_DATA_DIR, RAW_DATA_DIR, BASE_DIR
)

# Suppress Excel warnings
warnings.simplefilter(action='ignore', category=UserWarning)

# ==========================================
# 1. CONFIGURATION
# ==========================================

POS_DATA_DIR = BASE_DIR / "data" / "01_raw" / "pos_data"
OUTPUT_FILE = POS_DATA_DIR / "fact_all_locations_sales.csv"

# Folder Mappings
HISTORICAL_DIRS = [
    POS_DATA_DIR / "Dec 2025",
    POS_DATA_DIR / "Closed"  # Recursively searches Apr, Aug, Feb, etc.
]

CURRENT_DIRS = [
    POS_DATA_DIR / "centurion_2R",
    POS_DATA_DIR / "galleria",
    POS_DATA_DIR / "ngong_milele",
    POS_DATA_DIR / "pharmart_abc",
    POS_DATA_DIR / "portal_2R",
    POS_DATA_DIR / "portal_cbd",
]

# Standard Column Mapping (Adjust based on your actual Excel headers)
# We map variations to a standard name
COL_MAP = {
    'Date': 'Sale_Date',
    'Bill Date': 'Sale_Date',
    'Time': 'Time',
    'Bill Time': 'Time',
    'Receipt No': 'Receipt Txn No',
    'Bill No': 'Receipt Txn No',
    'Doc No': 'Receipt Txn No',
    'Description': 'Description',
    'Item Name': 'Description',
    'Product': 'Description',
    'Amount': 'Amount',
    'Total': 'Amount',
    'Net Amount': 'Amount',
    'Qty': 'Qty',
    'Quantity': 'Qty',
    'Location': 'Location',
    'Branch': 'Location'
}

# Location Normalizer
LOCATION_MAP = {
    'Abc': 'Pharmart ABC',
    'Centurion 2r': 'Centurion 2 Rivers',
    'Centurion 2Rivers': 'Centurion 2 Rivers',
    'Galleria': 'Galleria',
    'Milele': 'Milele',
    'Portal 2r': 'Portal 2 Rivers',
    'Portal 2R': 'Portal 2 Rivers',
    'Portal Cbd': 'Portal CBD',
    'Portal CBD': 'Portal CBD'
}

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def infer_location_from_filename(filename):
    """ Guesses location if missing in file data """
    fname = filename.lower()
    if 'abc' in fname: return 'Pharmart ABC'
    if 'centurion' in fname: return 'Centurion 2 Rivers'
    if 'galleria' in fname: return 'Galleria'
    if 'milele' in fname: return 'Milele'
    if 'portal 2r' in fname: return 'Portal 2 Rivers'
    if 'cbd' in fname: return 'Portal CBD'
    return 'Unknown'

def read_sales_file(path):
    """ Smart reader for CSV or Excel """
    try:
        # 1. Read File
        if path.suffix in ['.xlsx', '.xlsm']:
            # Assume header is in first few rows. Safe bet is row 0.
            df = pd.read_excel(path)
        elif path.suffix == '.csv':
            df = pd.read_csv(path, low_memory=False)
        else:
            return pd.DataFrame()

        # 2. Normalize Columns
        df = df.rename(columns={c: c.strip() for c in df.columns}) # Trim spaces
        df = df.rename(columns=COL_MAP)
        
        # 3. Validation: Must have at least Date and Amount
        if 'Sale_Date' not in df.columns or 'Amount' not in df.columns:
            # Try skipping metadata rows (common in Reports)
            if path.suffix in ['.xlsx', '.xlsm']:
                 df = pd.read_excel(path, header=1) # Try row 1
                 df = df.rename(columns={c: c.strip() for c in df.columns})
                 df = df.rename(columns=COL_MAP)
        
        # 4. Standardize Location
        if 'Location' not in df.columns:
            df['Location'] = infer_location_from_filename(path.name)
        else:
            # Fill NaN locations with filename guess
            df['Location'] = df['Location'].fillna(infer_location_from_filename(path.name))

        # 5. Clean Data types
        # Force numeric Amount
        df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce').fillna(0)
        # Drop zero/empty sales
        df = df[df['Amount'] > 0]
        
        # Date parsing
        df['Sale_Date'] = pd.to_datetime(df['Sale_Date'], errors='coerce')
        df = df.dropna(subset=['Sale_Date'])
        
        return df

    except Exception as e:
        print(f"   ⚠️ Failed to read {path.name}: {e}")
        return pd.DataFrame()

# ==========================================
# 3. MAIN PIPELINE
# ==========================================
def compile_sales_history():
    print("📊 STARTING FULL SALES HISTORY COMPILATION (2025-2026)...")
    
    all_sales = []
    
    # --- STEP 1: Process Current (Jan 2026) ---
    print("   📂 Processing Current Month (Jan 2026)...")
    for folder in CURRENT_DIRS:
        if not folder.exists(): continue
        # Grab CSVs and Excel files
        files = list(folder.glob("*.csv")) + list(folder.glob("*.xlsx")) + list(folder.glob("*.xlsm"))
        
        for f in files:
            # Skip massive historical dumps if they are duplicates of daily reports
            if "Jan 2023" in f.name: continue 
            
            df = read_sales_file(f)
            if not df.empty:
                all_sales.append(df)

    # --- STEP 2: Process Historical (Dec 2025 & Closed) ---
    print("   📂 Processing Historical Archives (Dec 2025 & Closed)...")
    
    historical_files = []
    
    # Dec 2025
    if (POS_DATA_DIR / "Dec 2025").exists():
        historical_files.extend((POS_DATA_DIR / "Dec 2025").glob("*.xlsm"))
        
    # Closed Folders (Recursive)
    if (POS_DATA_DIR / "Closed").exists():
        # rglob searches all subfolders (Apr 2025, Aug 2025, etc.)
        historical_files.extend((POS_DATA_DIR / "Closed").rglob("*.xlsm"))

    print(f"      Found {len(historical_files)} historical files.")
    
    for f in historical_files:
        df = read_sales_file(f)
        if not df.empty:
            all_sales.append(df)

    # --- STEP 3: Merge & Final Clean ---
    if not all_sales:
        print("❌ No sales data found!")
        return

    print("   🔄 Merging Datasets...")
    master_df = pd.concat(all_sales, ignore_index=True)
    
    # Final cleanup
    # Map Location Names
    master_df['Location'] = master_df['Location'].map(LOCATION_MAP).fillna(master_df['Location'])
    
    # Ensure standard columns exist
    req_cols = ['Sale_Date', 'Time', 'Location', 'Receipt Txn No', 'Description', 'Amount', 'Qty']
    for c in req_cols:
        if c not in master_df.columns:
            master_df[c] = np.nan

    # Sort
    master_df = master_df.sort_values(['Sale_Date', 'Time'])
    
    # Deduplicate (Critical when combining daily + monthly reports)
    before_dedup = len(master_df)
    master_df = master_df.drop_duplicates(subset=['Sale_Date', 'Receipt Txn No', 'Description', 'Amount'])
    after_dedup = len(master_df)
    
    print(f"   🧹 Deduplication: Removed {before_dedup - after_dedup:,} duplicate rows.")

    # Save
    master_df.to_csv(OUTPUT_FILE, index=False)
    print(f"🚀 SUCCESS: Compiled {len(master_df):,} Sales Records.")
    print(f"   📂 Output: {OUTPUT_FILE}")
    
    # Quick Check
    print("\n🔎 Sales by Month:")
    print(master_df['Sale_Date'].dt.to_period('M').value_counts().sort_index())

if __name__ == "__main__":
    compile_sales_history()