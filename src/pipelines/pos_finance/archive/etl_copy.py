import pandas as pd
import numpy as np
import glob
import os
import gc  # 🟢 ADDED: For memory management
import re
import warnings
import openpyxl
from datetime import datetime
from pathlib import Path

from Portal_ML_V4.src.utils.name_cleaner import clean_name_series

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
OUTPUT_FILE = OUTPUT_DIR / "all_locations_sales_Jan25-Jan26.csv"

# DATE CUTOFF
START_DATE = pd.Timestamp("2025-01-01")

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
    Intelligent parser for messy filenames like:
    - 060226.csv -> 2026-02-06
    - s140126.csv -> 2026-01-14
    """
    fname = os.path.basename(filename).lower()
    
    # Pattern 1: DD.MM.YYYY (e.g., 05.02.2026)
    match = re.search(r'(\d{2})\.(\d{2})\.(\d{4})', fname)
    if match:
        return pd.Timestamp(f"{match.group(3)}-{match.group(2)}-{match.group(1)}")

    # Pattern 2: DDMMYY or DMMYY (e.g., 060226)
    match = re.search(r'(?:^|s|sales|_|\s)(\d{6})(?:\.|$)', fname)
    if match:
        d_str = match.group(1)
        return pd.to_datetime(d_str, format='%d%m%y', errors='coerce')

    return pd.NaT

# def optimize_dtypes(df):
#     """
#     🟢 NEW: Downcasts columns to save 70%+ RAM.
#     Critical for avoiding 'ArrayMemoryError'.
#     """
#     for col in df.columns:
#         col_type = df[col].dtype
        
#         # Object -> Category (Huge savings for repeated text)
#         if col_type == 'object':
#             if df[col].nunique() / len(df) < 0.5:
#                 df[col] = df[col].astype('category')
        
#         # Float64 -> Float32
#         elif col_type == 'float64':
#             df[col] = pd.to_numeric(df[col], downcast='float')
            
#         # Int64 -> Int32
#         elif col_type == 'int64':
#             df[col] = pd.to_numeric(df[col], downcast='integer')
            
#     return df
def optimize_dtypes(df):
    KEY_COLS = {'Transaction ID', 'Receipt Txn No', 'Phone Number', 'Date Sold'}
    for col in df.columns:
        col_type = df[col].dtype

        if col_type == 'object' and col not in KEY_COLS:
            try:
                if df[col].nunique() / len(df) < 0.5:
                    df[col] = df[col].astype('category')
            except (MemoryError, TypeError, ValueError):
                pass  # Leave column as-is if conversion fails

        elif col_type == 'float64':
            df[col] = pd.to_numeric(df[col], downcast='float')

        elif col_type == 'int64':
            df[col] = pd.to_numeric(df[col], downcast='integer')

        # 🟢 NEW: Sanitize unexpected numeric types (complex128, float128, etc.)
        elif col_type not in ['float32', 'float16', 'int32', 'int16', 'int8', 'bool']:
            if hasattr(col_type, 'kind') and col_type.kind == 'c':  # complex
                df[col] = df[col].apply(lambda x: x.real if isinstance(x, complex) else x)
                df[col] = pd.to_numeric(df[col], errors='coerce')

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
                try:
                    df_sheet = pd.read_excel(xls, sheet_name=sheet)
                    # 🟢 FIX: convert to string first before stripping
                    df_sheet.columns = [str(c).strip() for c in df_sheet.columns]
                    if 'Receipt Txn No' in df_sheet.columns:
                        dfs.append(df_sheet)
                except Exception as e:
                    print(f"      ⚠️ Skipping sheet '{sheet}' in {file_path.name}: {e}")
                    continue
            if dfs:
                df = pd.concat(dfs, ignore_index=True)
            else:
                return pd.DataFrame()

        df.columns = [str(c).strip() for c in df.columns]  # 🟢 FIX here too
        df = apply_whitelist(df, CASHIER_COLS_KEEP)
        return df
    except Exception as e:
        print(f"      ❌ Failed to load cashier file {file_path.name}: {e}")
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
        return d if pd.notna(d) else pd.Timestamp.max
    
    all_files.sort(key=sort_key)

    for f in all_files:
        fname = os.path.basename(f).lower()
        if "cashier" in fname or "~$" in fname or fname.endswith(".sql"): continue
        if not (fname.endswith(".csv") or fname.endswith(".xlsx")): continue

        try:
            if fname.endswith(".csv"):
                try:
                    df = pd.read_csv(f, low_memory=False)
                except UnicodeDecodeError:
                    try:
                        df = pd.read_csv(f, low_memory=False, encoding='cp1252')
                    except UnicodeDecodeError:
                        df = pd.read_csv(f, low_memory=False, encoding='latin-1')
                except MemoryError:
                    print(f"      ⚠️ Memory limit on {fname}, trying chunked CSV read...")
                    chunks = []
                    try:
                        reader = pd.read_csv(f, low_memory=False, chunksize=50000)
                    except UnicodeDecodeError:
                        try:
                            reader = pd.read_csv(f, low_memory=False, chunksize=50000, encoding='cp1252')
                        except UnicodeDecodeError:
                            reader = pd.read_csv(f, low_memory=False, chunksize=50000, encoding='latin-1')
                    for chunk in reader:
                        chunks.append(chunk)
                        gc.collect()
                    df = pd.concat(chunks, ignore_index=True)
                    del chunks
                    gc.collect()
            else:
                # Replace the single read_excel call with:
                try:
                    df = pd.read_excel(f)
                except MemoryError:
                    # Read in chunks via openpyxl directly
                    print(f"      ⚠️ Memory limit on {fname}, trying chunked read...")
                    chunks = []
                    wb = openpyxl.load_workbook(f, read_only=True, data_only=True)
                    ws = wb.active

                    it = ws.iter_rows(values_only=True)
                    headers = [str(c) if c is not None else '' for c in next(it)]

                    chunk_size = 50000
                    buffer = []
                    chunks = []

                    for i, row in enumerate(it, start=1):
                        buffer.append(row)
                        if i % chunk_size == 0:
                            chunks.append(pd.DataFrame(buffer, columns=headers))
                            buffer = []
                            gc.collect()

                    if buffer:
                        chunks.append(pd.DataFrame(buffer, columns=headers))

                    wb.close()
                    gc.collect()
                    df = pd.concat(chunks, ignore_index=True)
            
            df.columns = df.columns.str.strip()
            
            if 'Transaction ID' in df.columns and 'Date Sold' in df.columns:
                df = apply_whitelist(df, SALES_COLS_KEEP)
                df['Date_Obj'] = parse_pos_date(df['Date Sold'])
                
                # Filter by Date
                df = df[df['Date_Obj'] >= START_DATE].copy()
                
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
def run_pos_etl_v3():
    print("🌍 STARTING POS ETL V4 (Robust Memory Mode)...")
    print(f"📅 Keeping Sales From: {START_DATE.date()} onwards")
    
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

        # A. LOAD SALES
        df_sales = load_sales_files_for_branch(loc_path)
        if df_sales.empty:
            print(f"    ⚠️ No valid 2025+ Sales Data found.")
            continue
        
        # 🟢 OPTIMIZE MEMORY IMMEDIATELY
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
            if 'Receipt Txn No' in df_cashier.columns and 'Amount' in df_cashier.columns:
                before_dedup = len(df_cashier)
                df_cashier = df_cashier.drop_duplicates(
                    subset=['Receipt Txn No', 'Amount'],
                    keep='first'
                )
                dupes_removed = before_dedup - len(df_cashier)
                if dupes_removed > 0:
                    print(f"    🧹 Cashier Dedup: {dupes_removed:,} duplicate rows removed before merge.")
        else:
            df_cashier = pd.DataFrame()
        
        # C. PREPARE CASHIER
        if not df_cashier.empty:
            if 'Receipt Txn No' not in df_cashier.columns:
                print("    ⚠️ Cashier data missing 'Receipt Txn No' column — skipping cashier merge.")
                df_cashier = pd.DataFrame()
            # 🟢 OPTIMIZE CASHIER MEMORY
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
                    df_cashier[col] = pd.to_numeric(df_cashier[col], errors='coerce').fillna(0.0)

            agg_rules = {'Amount': 'sum', 'Txn Costs': 'sum'}
            if 'Time' in df_cashier.columns: agg_rules['Time'] = 'min'
            
            meta_cols = [c for c in df_cashier.columns if c not in ['Receipt Txn No', 'Amount', 'Txn Costs', 'Time']]
            for c in meta_cols: agg_rules[c] = 'first'

            if 'Receipt Txn No' in df_cashier.columns:
                df_cashier['Receipt Txn No'] = clean_id_col(df_cashier['Receipt Txn No'])
                
            df_cashier = df_cashier.groupby('Receipt Txn No', as_index=False).agg(agg_rules)

            # 🟢 D. ROBUST MERGE STRATEGY
            print("    🔗 Running Safe Merge...")
            try:
                # Try Normal Merge first
                df_merged = pd.merge(
                    df_sales,
                    df_cashier,
                    left_on='Transaction ID',
                    right_on='Receipt Txn No',
                    how='left',
                    copy=False  # 🟢 Important: Avoid data duplication
                )
            # REPLACE everything from the except down to df_merged = pd.concat(chunks...)
            except (MemoryError, np.core._exceptions._ArrayMemoryError):
                print("    ⚠️ Memory Limit Hit! Switching to Disk-Based Chunked Merge...")
                chunk_size = 50000
                tmp_path = OUTPUT_DIR / f"_tmp_merge_{branch_label}.csv"
                first_chunk = True

                for i in range(0, len(df_sales), chunk_size):
                    chunk = df_sales.iloc[i:i+chunk_size].copy()
                    merged_chunk = pd.merge(
                        chunk,
                        df_cashier,
                        left_on='Transaction ID',
                        right_on='Receipt Txn No',
                        how='left',
                        copy=False
                    )
                    merged_chunk.to_csv(
                        tmp_path,
                        mode='w' if first_chunk else 'a',
                        header=first_chunk,
                        index=False
                    )
                    first_chunk = False
                    del chunk, merged_chunk
                    gc.collect()

                df_merged = pd.read_csv(tmp_path, low_memory=False)
                os.remove(tmp_path)
                gc.collect()
            
            df_merged['Audit_Status'] = np.where(
                df_merged['Receipt Txn No'].isna(), 'No Cashier Data', 'Matched'
            )
        else:
            df_merged = df_sales
            df_merged['Audit_Status'] = 'No Cashier Data'

        df_merged['Location'] = branch_label
        # master_list.append(df_merged)
        for col in df_merged.select_dtypes(include='category').columns:
            df_merged[col] = df_merged[col].astype('object')

        master_list.append(df_merged)
        del df_merged
        gc.collect()

        # 🟢 Clean up after branch processing
        try:
            del df_sales
        except NameError:
            pass
        try:
            del df_cashier
        except NameError:
            pass
        gc.collect()

    # ---------------------------------------------------------
    # 5. FINAL STACK & STRICT DEDUPLICATION
    # ---------------------------------------------------------
    if master_list:
        print("\n🏗️  Stacking All Locations...")
        final_df = pd.concat(master_list, ignore_index=True)

        gc.collect()
        
        if 'Date_Obj' in final_df.columns:
            final_df['Sale_Date'] = final_df['Date_Obj'].dt.normalize()
            final_df['Sale_Date_Str'] = final_df['Date_Obj'].dt.strftime('%Y-%m-%d')
            # 🟢 Safe Drop
            final_df = final_df.drop(columns=['Date_Obj'])
            gc.collect()

        before_len = len(final_df)

        # STRICT DEDUPLICATION
        dedup_cols = ['Transaction ID', 'Date Sold', 'Item', 'Description', 'Qty Sold', 'Total (Tax Ex)']
        actual_dedup_cols = [c for c in dedup_cols if c in final_df.columns]
        final_df.drop_duplicates(subset=actual_dedup_cols, inplace=True)

        # 🚨 THE MAGIC CALCULATION (Transaction Total)
        if 'Total (Tax Ex)' in final_df.columns and 'Transaction ID' in final_df.columns:
            final_df['Total (Tax Ex)'] = pd.to_numeric(final_df['Total (Tax Ex)'], errors='coerce').fillna(0)
            final_df['Transaction_Total'] = final_df.groupby('Transaction ID')['Total (Tax Ex)'].transform('sum')
        
        after_len = len(final_df)

        final_df.to_csv(OUTPUT_FILE, index=False)
        
        print(f"🚀 PIPELINE SUCCESS!")
        print(f"   📊 Rows Before Deduplication: {before_len:,}")
        print(f"   ✂️  Strict Deduplication: {before_len - after_len:,} duplicate rows removed.")
        print(f"   📂 Output: {OUTPUT_FILE}")
        print(f"   🧾 Total Valid Rows: {len(final_df):,}")
    else:
        print("\n❌ No data processed.")

if __name__ == "__main__":
    run_pos_etl_v3()