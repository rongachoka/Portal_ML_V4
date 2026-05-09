import pandas as pd
import numpy as np
import glob
import os
import gc  # 🟢 For memory management
import re
import warnings
from datetime import datetime
from pathlib import Path
from Portal_ML_V4.src.utils.name_cleaner import clean_name_series
# ✅ IMPORT PATHS
try:
    from Portal_ML_V4.src.config.settings import (
        BASE_DIR,
        PROCESSED_DATA_DIR,
    )
except ImportError:
    print("⚠️ Using manual fallback paths...")
    BASE_DIR = Path(os.getcwd())
    PROCESSED_DATA_DIR = BASE_DIR / "data" / "03_processed"
# ✅ SUPPRESS WARNINGS
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')
# ==========================================
# 1. CONFIGURATION
# ==========================================
RAW_DIR = BASE_DIR / "data" / "01_raw" / "pos_data"
CLOSED_DIR = RAW_DIR / "Closed"
DEC_DIR = RAW_DIR / "Dec 2025"
OUTPUT_DIR = PROCESSED_DATA_DIR / "pos_data"
OUTPUT_FILE = OUTPUT_DIR / "all_locations_sales_FULL_HISTORY.csv"
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
    fname = os.path.basename(filename).lower()
    match = re.search(r'(\d{2})\.(\d{2})\.(\d{4})', fname)
    if match:
        return pd.Timestamp(f"{match.group(3)}-{match.group(2)}-{match.group(1)}")
    match = re.search(r'(?:^|s|sales|_|\s)(\d{6})(?:\.|$)', fname)
    if match:
        d_str = match.group(1)
        return pd.to_datetime(d_str, format='%d%m%y', errors='coerce')
    return pd.NaT

def optimize_dtypes(df):
    # Only downcast numerics. Avoid category/string dtypes entirely
    # as they cause ArrowStringArray conflicts at every pd.concat call.
    for col in df.columns:
        col_type = df[col].dtype
        if col_type == 'float64':
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
    except Exception:
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
        # Normalise all frames to plain object dtype before concat.
        # Category columns from a previous branch's optimize_dtypes call
        # cause dtype conflicts when pandas tries to concatenate them.
        normed = []
        for df in relevant_dfs:
            # Force ALL columns to plain Python object dtype.
            # astype(str) can produce ArrowStringArray which still
            # causes OOM when pandas reconciles levels across frames.
            df = df.copy()
            for c in df.columns:
                if df[c].dtype.name in ('category', 'string') or hasattr(df[c], 'cat'):
                    df[c] = df[c].astype(object)
            normed.append(df)
        return pd.concat(normed, ignore_index=True)
    return pd.DataFrame()

def load_sales_files_for_branch(folder_path):
    all_files = glob.glob(str(folder_path / "*"))
    sales_dfs = []

    def sort_key(fpath):
        d = extract_date_from_filename(fpath)
        return d if pd.notna(d) else pd.Timestamp.min
    
    all_files.sort(key=sort_key)

    for f in all_files:
        fname = os.path.basename(f).lower()
        if "cashier" in fname or "~$" in fname or fname.endswith(".sql"): continue
        if not (fname.endswith(".csv") or fname.endswith(".xlsx")): continue
        try:
            if fname.endswith(".csv"):
                chunk_frames = []
                for encoding in ['utf-8', 'cp1252']:
                    chunk_frames = []
                    try:
                        reader = pd.read_csv(f, low_memory=False, chunksize=50000, encoding=encoding)
                        for chunk in reader:
                            chunk.columns = chunk.columns.str.strip()
                            if 'Transaction ID' in chunk.columns and 'Date Sold' in chunk.columns:
                                chunk = apply_whitelist(chunk, SALES_COLS_KEEP)
                                chunk['Date_Obj'] = parse_pos_date(chunk['Date Sold'])
                                chunk['Transaction ID'] = clean_id_col(chunk['Transaction ID'])
                                chunk_frames.append(chunk)
                        break
                    except (UnicodeDecodeError, Exception):
                        chunk_frames = []
                        continue
                if chunk_frames:
                    df = pd.concat(chunk_frames, ignore_index=True)
                    del chunk_frames
                    gc.collect()
                    sales_dfs.append(df)
                continue
            else:
                # Stream xlsx row-by-row via openpyxl read_only.
                # Writes batches of 10k rows directly to a temp CSV,
                # then reads that CSV back in chunks - never holds full xlsx in RAM.
                import openpyxl, tempfile, csv as _csv
                xl_tmp = Path(f).parent / f'__xl_tmp_{fname}.csv'
                try:
                    wb = openpyxl.load_workbook(f, read_only=True, data_only=True)
                    ws = wb.active
                    row_iter = ws.iter_rows(values_only=True)
                    raw_headers = [str(c).strip() if c is not None else '' for c in next(row_iter)]
                    needed_idx = [i for i, h in enumerate(raw_headers) if h in SALES_COLS_KEEP]
                    needed_headers = [raw_headers[i] for i in needed_idx]
                    if not needed_idx:
                        wb.close()
                        continue
                    with open(xl_tmp, 'w', newline='', encoding='utf-8') as xl_out:
                        writer = _csv.writer(xl_out)
                        writer.writerow(needed_headers)
                        for row in row_iter:
                            writer.writerow([row[i] for i in needed_idx])
                    wb.close()
                except Exception as xe:
                    print(f'      ❌ Error reading {fname}: {type(xe).__name__}: {xe}')
                    if xl_tmp.exists(): xl_tmp.unlink()
                    continue
                # Now read the temp CSV in chunks
                xl_chunk_frames = []
                try:
                    for chunk in pd.read_csv(xl_tmp, low_memory=False, chunksize=50000):
                        chunk.columns = chunk.columns.str.strip()
                        if 'Transaction ID' in chunk.columns and 'Date Sold' in chunk.columns:
                            chunk = apply_whitelist(chunk, SALES_COLS_KEEP)
                            chunk['Date_Obj'] = parse_pos_date(chunk['Date Sold'])
                            chunk['Transaction ID'] = clean_id_col(chunk['Transaction ID'])
                            xl_chunk_frames.append(chunk)
                finally:
                    if xl_tmp.exists(): xl_tmp.unlink()
                if xl_chunk_frames:
                    df = pd.concat(xl_chunk_frames, ignore_index=True)
                    del xl_chunk_frames
                    gc.collect()
                    sales_dfs.append(df)
                continue

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
    # Clear previous discrepancy log so each run starts fresh
    _old_log = OUTPUT_DIR / 'discrepancy_log.csv'
    if _old_log.exists(): _old_log.unlink()
    TEMP_DIR = OUTPUT_DIR / 'temp_branches'
    os.makedirs(TEMP_DIR, exist_ok=True)
    branch_files = []

    for folder_name, branch_keywords in LOCATION_MAP.items():
        loc_path = RAW_DIR / folder_name
        if not loc_path.exists(): continue

        branch_label = folder_name.upper()
        print(f"\n📍 PROCESSING BRANCH: {branch_label}")

        # A. LOAD SALES
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
            if 'Receipt Txn No' in df_cashier.columns and 'Amount' in df_cashier.columns:
                before_dedup = len(df_cashier)
                df_cashier = df_cashier.drop_duplicates(
                    subset=['Receipt Txn No', 'Amount', 'Phone Number'],
                    keep='first'
                )
                dupes_removed = before_dedup - len(df_cashier)
                if dupes_removed > 0:
                    print(f"    🧹 Cashier Dedup: {dupes_removed:,} duplicate rows removed before merge.")
        else:
            df_cashier = pd.DataFrame()
        
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
                    df_cashier[col] = pd.to_numeric(df_cashier[col], errors='coerce').fillna(0.0)

            agg_rules = {'Amount': 'sum', 'Txn Costs': 'sum'}
            if 'Time' in df_cashier.columns: agg_rules['Time'] = 'min'
            
            meta_cols = [c for c in df_cashier.columns if c not in ['Receipt Txn No', 'Amount', 'Txn Costs', 'Time']]
            for c in meta_cols: agg_rules[c] = 'first'

            if 'Receipt Txn No' in df_cashier.columns:
                df_cashier['Receipt Txn No'] = clean_id_col(df_cashier['Receipt Txn No'])
                
            df_cashier = df_cashier.groupby('Receipt Txn No', as_index=False).agg(agg_rules)

            # D. CHUNKED MERGE (always used for all branches)
            # Prevents memory errors on large branches like Galleria/Portal 2R.
            # Time column is cast to string here to prevent datetime64 dtype
            # conflicts when concatenating chunks.
            if 'Time' in df_cashier.columns:
                df_cashier['Time'] = df_cashier['Time'].astype(str)
            print("    🔗 Running Chunked Merge...")
            chunks = []
            chunk_size = 50000
            for i in range(0, len(df_sales), chunk_size):
                chunk = df_sales.iloc[i:i+chunk_size].copy()
                merged_chunk = pd.merge(
                    chunk, df_cashier,
                    left_on='Transaction ID',
                    right_on='Receipt Txn No',
                    how='left'
                )
                # Normalize all non-numeric columns to plain object dtype
                # before stacking chunks to prevent ArrowString/category OOM
                for _c in merged_chunk.columns:
                    if merged_chunk[_c].dtype.name in ('category', 'string') or hasattr(merged_chunk[_c], 'cat'):
                        merged_chunk[_c] = merged_chunk[_c].astype(object)
                chunks.append(merged_chunk)
                del chunk, merged_chunk
                gc.collect()
            df_merged = pd.concat(chunks, ignore_index=True)
            del chunks
            gc.collect()
            
            df_merged['Audit_Status'] = np.where(
                df_merged['Receipt Txn No'].isna(), 'No Cashier Data', 'Matched'
            )
        else:
            df_merged = df_sales
            df_merged['Audit_Status'] = 'No Cashier Data'

        df_merged['Location'] = branch_label

        # 🟢 FIX: Convert Date_Obj and deduplicate PER BRANCH before appending.
        # Previously this happened after stacking all 1.2M rows, causing an
        # ArrayMemoryError on the datetime64[us] column during drop_duplicates.
        # Doing it here keeps each branch's frame small and avoids the crash.
        if 'Date_Obj' in df_merged.columns:
            df_merged['Sale_Date']     = df_merged['Date_Obj'].dt.normalize().dt.strftime('%Y-%m-%d')
            df_merged['Sale_Date_Str'] = df_merged['Sale_Date']
            df_merged = df_merged.drop(columns=['Date_Obj'])

        dedup_cols_branch = ['Transaction ID', 'Date Sold', 'Total (Tax Ex)', 'Description']
        actual_dedup_branch = [c for c in dedup_cols_branch if c in df_merged.columns]
        before_branch = len(df_merged)
        df_merged.drop_duplicates(subset=actual_dedup_branch, inplace=True)
        print(f"    🧹 Branch dedup: {before_branch - len(df_merged):,} duplicates removed")

        # Calculate Transaction_Total per branch (fits in RAM; avoids doing it on 900k rows)
        if 'Total (Tax Ex)' in df_merged.columns and 'Transaction ID' in df_merged.columns:
            df_merged['Total (Tax Ex)'] = pd.to_numeric(df_merged['Total (Tax Ex)'], errors='coerce').fillna(0)
            if 'Amount' not in df_merged.columns:
                df_merged['Amount'] = 0
            else:
                df_merged['Amount'] = pd.to_numeric(df_merged['Amount'], errors='coerce').fillna(0)
            df_merged['POS_Txn_Sum'] = df_merged.groupby('Transaction ID')['Total (Tax Ex)'].transform('sum')
            # Always use POS line-item sum as the transaction value.
            # Cashier Amount is unreliable: part-payments and misattribution
            # mean it understates or misallocates revenue.
            df_merged['Transaction_Total'] = df_merged['POS_Txn_Sum']

            # --- DISCREPANCY AUDIT: save per-branch rows where cashier != POS ---
            # Runs here while POS_Txn_Sum is still available.
            # Written to a CSV that accumulates across all branches each run.
            audit_cols = [c for c in ['Transaction ID', 'Sale_Date', 'Description',
                                       'Amount', 'POS_Txn_Sum', 'Phone Number', 'Client Name']
                          if c in df_merged.columns]
            audit_src = df_merged[audit_cols].drop_duplicates(subset=['Transaction ID'])
            discrepancy_mask = (
                (audit_src['Amount'] != 0) &
                (abs(audit_src['POS_Txn_Sum'] - audit_src['Amount']) > 1)
            )
            discrepant = audit_src[discrepancy_mask].copy()
            if not discrepant.empty:
                discrepant['Location'] = folder_name.upper()
                discrepant['Discrepancy'] = discrepant['POS_Txn_Sum'] - discrepant['Amount']
                audit_file = OUTPUT_DIR / 'discrepancy_log.csv'
                write_header = not audit_file.exists()
                discrepant.to_csv(audit_file, mode='a', index=False, header=write_header)
                print(f'    ⚠️  Discrepancies logged: {len(discrepant):,} transactions')
            else:
                print(f'    ✅ No discrepancies found')

            df_merged = df_merged.drop(columns=['POS_Txn_Sum'])

        # Write branch to temp CSV and free RAM immediately
        temp_path = TEMP_DIR / f'{folder_name}_temp.csv'
        df_merged.to_csv(temp_path, index=False)
        branch_files.append(temp_path)
        print(f'    💾 Written to temp: {temp_path.name}')
        
        del df_sales, df_cashier, df_merged
        gc.collect()

    # ---------------------------------------------------------
    # 5. FINAL STACK & DISCREPANCY LOGIC
    # ---------------------------------------------------------
    if branch_files:
        print("\n Stacking All Locations (zero-RAM file merge)...")
        # Concatenate branch CSVs at the file level - no pandas, no RAM spike
        total_rows = 0
        header_written = False
        with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as out_f:
            for branch_path in branch_files:
                with open(branch_path, 'r', encoding='utf-8') as in_f:
                    header_line = in_f.readline()
                    if not header_written:
                        out_f.write(header_line)
                        header_written = True
                    for line in in_f:
                        out_f.write(line)
                        total_rows += 1
        import shutil
        shutil.rmtree(TEMP_DIR, ignore_errors=True)
        print(f" FULL HISTORY PIPELINE SUCCESS!")
        print(f"   Output: {OUTPUT_FILE}")
        print(f"   Total Rows Written: {total_rows:,}")
    else:
        print("\n No data processed.")

if __name__ == "__main__":
    run_pos_etl_full_history()