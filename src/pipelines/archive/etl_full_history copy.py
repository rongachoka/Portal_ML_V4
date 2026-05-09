import pandas as pd
import numpy as np
import glob
import os
import gc  # 🟢 For memory management
import re
import warnings
from datetime import datetime
from pathlib import Path

# ✅ IMPORT PATHS
# Ensure these point to your actual project structure
from Portal_ML_V4.src.config.settings import (
    BASE_DIR,
    PROCESSED_DATA_DIR,
)

# ✅ SUPPRESS WARNINGS
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

# ==========================================
# 1. CONFIGURATION
# ==========================================
RAW_DIR = BASE_DIR / "data" / "01_raw" / "pos_data"
CLOSED_DIR = RAW_DIR / "Closed"
DEC_DIR = RAW_DIR / "Dec 2025"

OUTPUT_DIR = PROCESSED_DATA_DIR / "pos_data"
# 🟢 CHANGED: Updated filename to reflect full history
OUTPUT_FILE = OUTPUT_DIR / "all_locations_sales_FULL_HISTORY.csv"

# 🟢 CHANGED: Set to a very old date to capture everything
START_DATE = pd.Timestamp("2000-01-01") 

# Branch Map
LOCATION_MAP = {
    "centurion_2R":   ["centurion", "c2r"],
    "galleria":       ["galleria"],
    "ngong_milele":   ["milele"],
    "pharmart_abc":   ["abc", "pharmart abc"],
    "portal_2R":      ["portal 2r", "portal_2r", "portal 2 rivers"],
    "portal_cbd":     ["cbd", "portal cbd"]
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

def clean_id_col(series):
    return series.astype(str).str.replace(r'\.0$', '', regex=True).str.strip()

def parse_pos_date(date_series):
    s = date_series.astype(str)
    s = s.str.replace('#', '', regex=False).str.strip()
    return pd.to_datetime(s, errors='coerce')

def is_cashier_file_for_branch(filename, branch_keywords):
    fname = filename.lower()
    return any(k in fname for k in branch_keywords)

def apply_whitelist(df, whitelist):
    existing_cols = [c for c in whitelist if c in df.columns]
    return df[existing_cols].copy()

def extract_date_from_filename(filename):
    """
    Intelligent parser for messy filenames.
    """
    fname = os.path.basename(filename).lower()
    
    # Pattern 1: DD.MM.YYYY
    match = re.search(r'(\d{2})\.(\d{2})\.(\d{4})', fname)
    if match:
        return pd.Timestamp(f"{match.group(3)}-{match.group(2)}-{match.group(1)}")

    # Pattern 2: DDMMYY or DMMYY
    match = re.search(r'(?:^|s|sales|_|\s)(\d{6})(?:\.|$)', fname)
    if match:
        d_str = match.group(1)
        return pd.to_datetime(d_str, format='%d%m%y', errors='coerce')

    return pd.NaT

def optimize_dtypes(df):
    """
    🟢 Downcasts columns to save RAM. Critical for full history.
    """
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type == 'object':
            if df[col].nunique() / len(df) < 0.5:
                df[col] = df[col].astype('category')
        elif col_type == 'float64':
            df[col] = pd.to_numeric(df[col], downcast='float')
        elif col_type == 'int64':
            df[col] = pd.to_numeric(df[col], downcast='integer')
            
    return df

# ==========================================
# 3. DATA LOADING FUNCTIONS
# ==========================================

def load_cashier_file(file_path):
    try:
        if str(file_path).endswith('.csv'):
            df = pd.read_csv(file_path, low_memory=False)
        else:
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
        return pd.DataFrame()

def load_historical_cashier_for_branch(branch_keywords):
    files_found = []
    if DEC_DIR.exists():
        files_found.extend(DEC_DIR.glob("*.xlsm"))
        files_found.extend(DEC_DIR.glob("*.xlsx"))
    if CLOSED_DIR.exists():
        files_found.extend(CLOSED_DIR.rglob("*.xlsm"))
        files_found.extend(CLOSED_DIR.rglob("*.xlsx"))

    relevant_dfs = []
    for f in files_found:
        if is_cashier_file_for_branch(f.name, branch_keywords):
            df = load_cashier_file(f)
            if not df.empty:
                relevant_dfs.append(df)
    
    if relevant_dfs:
        return pd.concat(relevant_dfs, ignore_index=True)
    return pd.DataFrame()

def load_sales_files_for_branch(folder_path):
    all_files = glob.glob(str(folder_path / "*"))
    sales_dfs = []

    def sort_key(fpath):
        d = extract_date_from_filename(fpath)
        return d if pd.notna(d) else pd.Timestamp.min
    
    # Sort files chronologically to ensure historical order
    all_files.sort(key=sort_key)

    for f in all_files:
        fname = os.path.basename(f).lower()
        if "cashier" in fname or "~$" in fname or fname.endswith(".sql"): continue
        if not (fname.endswith(".csv") or fname.endswith(".xlsx")): continue

        try:
            if fname.endswith(".csv"):
                try: df = pd.read_csv(f, low_memory=False)
                except: df = pd.read_csv(f, low_memory=False, encoding='cp1252')
            else:
                df = pd.read_excel(f)
            
            df.columns = df.columns.str.strip()
            
            if 'Transaction ID' in df.columns and 'Date Sold' in df.columns:
                df = apply_whitelist(df, SALES_COLS_KEEP)
                df['Date_Obj'] = parse_pos_date(df['Date Sold'])
                
                # 🟢 CHANGED: Removed the date filter. 
                # This will now accept any valid date found in the file.
                # df = df[df['Date_Obj'] >= START_DATE].copy() 
                
                if len(df) > 0:
                    df['Transaction ID'] = clean_id_col(df['Transaction ID'])
                    sales_dfs.append(df)
                    
        except Exception as e:
            print(f"      ❌ Error reading {fname}: {e}")

    if not sales_dfs:
        return pd.DataFrame()

    full_df = pd.concat(sales_dfs, ignore_index=True)
    return full_df

# ==========================================
# 4. MAIN PIPELINE
# ==========================================
def run_pos_etl_full_history():
    print("🌍 STARTING POS ETL - FULL HISTORY MODE...")
    
    if not RAW_DIR.exists():
        print(f"❌ Error: Raw directory not found at {RAW_DIR}")
        return
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    master_list = []

    for folder_name, branch_keywords in LOCATION_MAP.items():
        loc_path = RAW_DIR / folder_name
        if not loc_path.exists(): continue

        branch_label = folder_name.upper()
        print(f"\n📍 PROCESSING BRANCH: {branch_label}")

        # A. LOAD SALES (All History)
        df_sales = load_sales_files_for_branch(loc_path)
        if df_sales.empty:
            print(f"    ⚠️ No Sales Data found.")
            continue
        
        df_sales = optimize_dtypes(df_sales)
        print(f"    ✅ Sales Loaded & Optimized: {len(df_sales):,} rows")

        # B. LOAD CASHIER
        current_files = list(loc_path.glob("*Cashier*.xlsm")) + list(loc_path.glob("*Cashier*.xlsx"))
        
        cashier_dfs = []
        for cf in current_files:
            if "~$" not in cf.name:
                cashier_dfs.append(load_cashier_file(cf))
                
        df_hist_cashier = load_historical_cashier_for_branch(branch_keywords)
        if not df_hist_cashier.empty:
            cashier_dfs.append(df_hist_cashier)
            
        if cashier_dfs:
            df_cashier = pd.concat(cashier_dfs, ignore_index=True)
        else:
            df_cashier = pd.DataFrame()
        
        # C. PREPARE CASHIER
        if not df_cashier.empty:
            df_cashier = optimize_dtypes(df_cashier)

            if 'Time' in df_cashier.columns:
                df_cashier['Time'] = df_cashier['Time'].astype(str).fillna("00:00:00")

            for col in ['Amount', 'Txn Costs']:
                if col in df_cashier.columns:
                    df_cashier[col] = (
                        df_cashier[col].astype(str)
                        .str.replace(',', '', regex=False)
                        .str.replace(r'[^\d\.\-]', '', regex=True)
                    )
                    df_cashier[col] = pd.to_numeric(df_cashier[col], errors='coerce').fillna(0.0)

            agg_rules = {'Amount': 'sum', 'Txn Costs': 'sum'}
            if 'Time' in df_cashier.columns: agg_rules['Time'] = 'min'
            
            meta_cols = [c for c in df_cashier.columns if c not in ['Receipt Txn No', 'Amount', 'Txn Costs', 'Time']]
            for c in meta_cols: agg_rules[c] = 'first'

            if 'Receipt Txn No' in df_cashier.columns:
                df_cashier['Receipt Txn No'] = clean_id_col(df_cashier['Receipt Txn No'])
                
            df_cashier = df_cashier.groupby('Receipt Txn No', as_index=False).agg(agg_rules)

            # D. SAFE MERGE
            print("    🔗 Running Safe Merge...")
            try:
                df_merged = pd.merge(
                    df_sales,
                    df_cashier,
                    left_on='Transaction ID',
                    right_on='Receipt Txn No',
                    how='left',
                    copy=False 
                )
            except (MemoryError, np.core._exceptions._ArrayMemoryError):
                print("    ⚠️ Memory Limit Hit! Switching to Chunked Merge...")
                chunks = []
                chunk_size = 50000 
                
                for i in range(0, len(df_sales), chunk_size):
                    chunk = df_sales.iloc[i:i+chunk_size]
                    merged_chunk = pd.merge(
                        chunk,
                        df_cashier,
                        left_on='Transaction ID',
                        right_on='Receipt Txn No',
                        how='left',
                        copy=False
                    )
                    chunks.append(merged_chunk)
                    gc.collect()
                    
                df_merged = pd.concat(chunks, ignore_index=True)
            
            df_merged['Audit_Status'] = np.where(
                df_merged['Receipt Txn No'].isna(), 'No Cashier Data', 'Matched'
            )
        else:
            df_merged = df_sales
            df_merged['Audit_Status'] = 'No Cashier Data'

        df_merged['Location'] = branch_label
        master_list.append(df_merged)
        
        # Clean up
        del df_sales, df_cashier, df_merged
        gc.collect()

    # ---------------------------------------------------------
    # 5. FINAL STACK
    # ---------------------------------------------------------
    if master_list:
        print("\n🏗️  Stacking All Locations...")
        final_df = pd.concat(master_list, ignore_index=True)

        gc.collect()
        
        if 'Date_Obj' in final_df.columns:
            final_df['Sale_Date'] = final_df['Date_Obj'].dt.normalize()
            final_df['Sale_Date_Str'] = final_df['Date_Obj'].dt.strftime('%Y-%m-%d')
            final_df = final_df.drop(columns=['Date_Obj'])
            gc.collect()

        before_len = len(final_df)

        # STRICT DEDUPLICATION
        dedup_cols = ['Transaction ID', 'Date Sold', 'Total (Tax Ex)', 'Description']
        actual_dedup_cols = [c for c in dedup_cols if c in final_df.columns]
        final_df.drop_duplicates(subset=actual_dedup_cols, inplace=True)

        # Calculate Transaction Totals
        if 'Total (Tax Ex)' in final_df.columns and 'Transaction ID' in final_df.columns:
            final_df['Total (Tax Ex)'] = pd.to_numeric(final_df['Total (Tax Ex)'], errors='coerce').fillna(0)
            final_df['Transaction_Total'] = final_df.groupby('Transaction ID')['Total (Tax Ex)'].transform('sum')
        
        after_len = len(final_df)

        final_df.to_csv(OUTPUT_FILE, index=False)
        
        print(f"🚀 FULL HISTORY PIPELINE SUCCESS!")
        print(f"   📊 Rows Before Deduplication: {before_len:,}")
        print(f"   ✂️  Strict Deduplication: {before_len - after_len:,} duplicate rows removed.")
        print(f"   📂 Output: {OUTPUT_FILE}")
        print(f"   🧾 Total Valid Rows: {len(final_df):,}")
    else:
        print("\n❌ No data processed.")

if __name__ == "__main__":
    run_pos_etl_full_history()