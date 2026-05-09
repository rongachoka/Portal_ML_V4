import pandas as pd
import numpy as np
import glob
import os
import re
import warnings
from datetime import datetime
from pathlib import Path

# ✅ SUPPRESS WARNINGS
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

# ✅ IMPORT PATHS
from Portal_ML_V4.src.config.settings import (
    BASE_DIR,
    PROCESSED_DATA_DIR,
)

# ==========================================
# 1. CONFIGURATION
# ==========================================
RAW_DIR = BASE_DIR / "data" / "01_raw" / "pos_data"
CLOSED_DIR = RAW_DIR / "Closed"
DEC_DIR = RAW_DIR / "Dec 2025"

OUTPUT_DIR = PROCESSED_DATA_DIR / "pos_data"
OUTPUT_FILE = OUTPUT_DIR / "fact_all_locations_sales_Jan25-Jan26.csv"

# Map Folder Names (Keys) to Human Labels (Values)
# We use this Key to find matching files in the Closed folder
LOCATION_MAP = {
    "centurion_2R": "Centurion",
    "galleria": "Galleria",
    "ngong_milele": "Milele",
    "pharmart_abc": "ABC Place",
    "portal_2R": "Portal 2R",
    "portal_cbd": "CBD"
}

# 🔒 COLUMNS TO KEEP
SALES_COLS_KEEP = [
    'Department', 'Category', 'Item', 'Description', 
    'On Hand', 'Last Sold', 'Qty Sold', 'Total (Tax Ex)', 
    'Transaction ID', 'Date Sold'
]

CASHIER_COLS_KEEP = [
    'Receipt Txn No', 'Sales Rep', 'Client Name', 'Phone Number', 
    'Txn Type', 'Ordered Via', 'Time', 'Amount', 'Txn Costs'
]

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def get_branch_key_from_filename(filename):
    """
    Determines which branch folder a historical file belongs to based on its name.
    Returns the dictionary KEY from LOCATION_MAP (e.g., 'pharmart_abc').
    """
    fname = filename.lower()
    
    if 'abc' in fname: return 'pharmart_abc'
    if 'centurion' in fname: return 'centurion_2R'
    if 'galleria' in fname: return 'galleria'
    if 'milele' in fname: return 'ngong_milele'
    if 'portal 2r' in fname or 'portal_2r' in fname: return 'portal_2R'
    if 'cbd' in fname: return 'portal_cbd'
    
    return None

def clean_id_col(series):
    return series.astype(str).str.replace(r'\.0$', '', regex=True).str.strip()

def clean_time_col(val):
    if pd.isna(val) or val == '': return None
    s_val = str(val).strip()
    if ':' in s_val: return s_val
    try:
        f_val = float(val)
        return "{:.2f}".format(f_val).replace('.', ':')
    except: return None

def apply_whitelist(df, whitelist):
    existing_cols = [c for c in whitelist if c in df.columns]
    return df[existing_cols].copy()

def extract_date_from_filename(filename):
    s = filename.lower()
    # Logic to prioritize Master Files (Multi-year)
    if "2023" in s and "2026" in s: 
        return datetime(2026, 12, 31) # Give it high priority

    # Existing regex logic...
    match = re.search(r"(\d{1,2})\.(\d{1,2})\.(\d{2,4})", s)
    if match: return datetime.now() # Simplified for brevity, use your full logic if needed
    return datetime.min

# ==========================================
# 3. DATA LOADING FUNCTIONS
# ==========================================

def read_single_cashier_file(file_path):
    """Reads a single XLSM/XLSX/CSV cashier report safely."""
    try:
        if str(file_path).endswith('.csv'):
            df = pd.read_csv(file_path, low_memory=False)
        else:
            # Excel: Read all sheets, look for one with 'Receipt Txn No'
            dfs = []
            xls = pd.ExcelFile(file_path)
            for sheet in xls.sheet_names:
                df_sheet = pd.read_excel(xls, sheet_name=sheet)
                df_sheet.columns = df_sheet.columns.str.strip()
                if 'Receipt Txn No' in df_sheet.columns:
                    dfs.append(df_sheet)
            
            if dfs:
                df = pd.concat(dfs, ignore_index=True)
            else:
                return pd.DataFrame()

        df.columns = df.columns.str.strip()
        df = apply_whitelist(df, CASHIER_COLS_KEEP)
        return df
    except Exception as e:
        # print(f"    ⚠️ Error reading {os.path.basename(file_path)}: {e}")
        return pd.DataFrame()

def load_historical_cashier_data():
    """
    Crawls 'Closed' and 'Dec 2025' folders.
    Returns a Dictionary: { 'pharmart_abc': combined_df, 'galleria': combined_df ... }
    """
    print("   🕰️  Scanning Historical Cashier Reports (Closed + Dec 2025)...")
    
    historical_map = {k: [] for k in LOCATION_MAP.keys()}
    
    # 1. Collect all candidate files
    files_to_scan = []
    
    # Dec 2025
    if DEC_DIR.exists():
        files_to_scan.extend(DEC_DIR.glob("*.xlsm"))
        files_to_scan.extend(DEC_DIR.glob("*.xlsx"))
        
    # Closed (Recursive)
    if CLOSED_DIR.exists():
        files_to_scan.extend(CLOSED_DIR.rglob("*.xlsm"))
        files_to_scan.extend(CLOSED_DIR.rglob("*.xlsx"))
        
    print(f"      Found {len(files_to_scan)} historical files. Sorting by branch...")
    
    for f in files_to_scan:
        branch_key = get_branch_key_from_filename(f.name)
        if branch_key:
            df = read_single_cashier_file(f)
            if not df.empty:
                historical_map[branch_key].append(df)
                
    # Concatenate per branch
    final_map = {}
    for branch, df_list in historical_map.items():
        if df_list:
            final_map[branch] = pd.concat(df_list, ignore_index=True)
            print(f"      🔹 {branch}: Loaded {len(final_map[branch]):,} rows (History)")
        else:
            final_map[branch] = pd.DataFrame()
            
    return final_map

def load_sales_files(folder_path):
    """ Loads the BEST sales file (prioritizing Master 2023-2026 files). """
    all_files = glob.glob(os.path.join(folder_path, "*"))
    candidates = []

    for f in all_files:
        fname = os.path.basename(f)
        fname_lower = fname.lower()
        
        if "cashier" in fname_lower: continue
        if not (fname_lower.endswith(".csv") or fname_lower.endswith(".xlsx")): continue
        
        # Priority: Master File
        if "2023" in fname_lower and "2026" in fname_lower:
            candidates.append((100, f)) # High score
        # Secondary: Dated files
        else:
            candidates.append((1, f))

    if not candidates: return pd.DataFrame()

    # Pick highest priority
    candidates.sort(key=lambda x: x[0])
    best_file = candidates[-1][1]
    fname = os.path.basename(best_file)
    
    print(f"    🏆 Selected Sales Data: {fname}")

    try:
        if fname.lower().endswith(".csv"):
            try: df = pd.read_csv(best_file, low_memory=False)
            except: df = pd.read_csv(best_file, low_memory=False, encoding='cp1252')
        else:
            df = pd.read_excel(best_file)
        
        df.columns = df.columns.str.strip()
        if 'Transaction ID' in df.columns:
            df = apply_whitelist(df, SALES_COLS_KEEP)
            df['Transaction ID'] = clean_id_col(df['Transaction ID'])
            return df
    except Exception as e:
        print(f"      ❌ Error reading {fname}: {e}")

    return pd.DataFrame()

# ==========================================
# 4. MAIN PIPELINE
# ==========================================
def run_multi_location_etl():
    print("🌍 STARTING POS ETL (With Historical Cashier Merge)...")
    
    if not RAW_DIR.exists():
        print(f"❌ Error: Raw directory not found at {RAW_DIR}")
        return
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Pre-load Historical Cashier Data
    historical_cashier_data = load_historical_cashier_data()
    
    master_list = []

    # 2. Iterate Branches
    for folder_name, loc_label in LOCATION_MAP.items():
        loc_path = RAW_DIR / folder_name
        if not loc_path.exists(): continue

        print(f"\n📍 Processing: {loc_label} ({folder_name})")

        # A. Load Sales Data (Prefer Master File)
        df_sales = load_sales_files(loc_path)
        if df_sales.empty:
            print(f"    ⚠️ No Sales Data found.")
            continue

        # B. Load Current Cashier Data (Local Folder)
        current_cashier_files = glob.glob(os.path.join(loc_path, "*Cashier report*.xlsm"))
        df_current_cashier = pd.DataFrame()
        if current_cashier_files:
            df_current_cashier = read_single_cashier_file(current_cashier_files[0])
            print(f"    ✅ Loaded Current Cashier Report ({len(df_current_cashier)} rows)")

        # C. Merge Current + Historical Cashier Data
        df_hist_cashier = historical_cashier_data.get(folder_name, pd.DataFrame())
        
        df_cashier = pd.concat([df_current_cashier, df_hist_cashier], ignore_index=True)
        
        if not df_cashier.empty:
            print(f"    🔗 Combined Cashier Data: {len(df_cashier):,} rows total.")
            
            # --- AGGREGATION LOGIC (Handle Split Payments) ---
            if 'Client Name' in df_cashier.columns:
                df_cashier['Name_Len'] = df_cashier['Client Name'].astype(str).str.len()
                df_cashier.sort_values('Name_Len', ascending=False, inplace=True)
                df_cashier.drop(columns=['Name_Len'], inplace=True)

            if 'Time' in df_cashier.columns:
                df_cashier['Time'] = df_cashier['Time'].astype(str)

            money_cols = ['Amount', 'Txn Costs']
            for col in money_cols:
                if col in df_cashier.columns:
                    df_cashier[col] = pd.to_numeric(df_cashier[col], errors='coerce').fillna(0.0)

            agg_rules = {'Amount': 'sum', 'Txn Costs': 'sum', 'Time': 'min'}
            other_cols = [c for c in df_cashier.columns if c not in ['Receipt Txn No', 'Amount', 'Txn Costs', 'Time']]
            for c in other_cols: agg_rules[c] = 'first'

            # Clean Receipt IDs before Grouping
            if 'Receipt Txn No' in df_cashier.columns:
                df_cashier['Receipt Txn No'] = clean_id_col(df_cashier['Receipt Txn No'])
                
            df_cashier = df_cashier.groupby('Receipt Txn No', as_index=False).agg(agg_rules)

            # --- MERGE SALES + CASHIER ---
            df_merged = pd.merge(
                df_sales,
                df_cashier,
                left_on='Transaction ID',
                right_on='Receipt Txn No',
                how='left'
            )
            
            # --- AUDIT ---
            receipt_sums = df_sales.groupby('Transaction ID')['Total (Tax Ex)'].sum().reset_index()
            receipt_sums.rename(columns={'Total (Tax Ex)': 'Calc_Items_Total'}, inplace=True)
            df_merged = pd.merge(df_merged, receipt_sums, on='Transaction ID', how='left')
            
            df_merged['Calc_Items_Total'] = pd.to_numeric(df_merged['Calc_Items_Total'], errors='coerce').fillna(0)
            df_merged['Amount'] = pd.to_numeric(df_merged['Amount'], errors='coerce').fillna(0)
            df_merged['Audit_Diff'] = df_merged['Calc_Items_Total'] - df_merged['Amount']
            
            df_merged['Audit_Status'] = np.where(
                (df_merged['Amount'] == 0) | (df_merged['Calc_Items_Total'] == 0), 'Not Audited',
                np.where(abs(df_merged['Audit_Diff']) > 1.0, 'MISMATCH', 'Match')
            )
            
        else:
            print("    ⚠️ No Cashier Data available (Current or Historical).")
            df_merged = df_sales
            df_merged['Audit_Status'] = 'No Cashier Data'

        df_merged['Location'] = loc_label
        df_merged = df_merged[df_merged['Transaction ID'].str.len() > 0]
        master_list.append(df_merged)

    # 3. Final Concatenation
    if master_list:
        print("\n🏗️  Stacking All Locations...")
        final_df = pd.concat(master_list, ignore_index=True)
        
        # Clean Date
        if 'Date Sold' in final_df.columns:
            final_df['Date_Clean'] = final_df['Date Sold'].astype(str).str.replace('#', '')
            final_df['Sale_Date'] = pd.to_datetime(final_df['Date_Clean'], errors='coerce').dt.date

        final_df.to_csv(OUTPUT_FILE, index=False)
        print(f"🚀 PIPELINE SUCCESS!")
        print(f"   📂 Output: {OUTPUT_FILE}")
        print(f"   🧾 Total Rows: {len(final_df):,}")
    else:
        print("\n❌ No data processed.")

if __name__ == "__main__":
    run_multi_location_etl()