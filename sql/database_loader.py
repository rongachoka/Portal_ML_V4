## ============================================================
## REFRESH — Run these two lines after every loader run
## (Need to add them to the end of thenightly batch script when created)
## ============================================================
## REFRESH MATERIALIZED VIEW CONCURRENTLY mv_transaction_master;
## REFRESH MATERIALIZED VIEW CONCURRENTLY mv_client_list;
# =====================================================================

import pandas as pd
import glob
import os
import re
import warnings
from pathlib import Path
from sqlalchemy import create_engine, text
# ✅ SUPPRESS WARNINGS
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')
# ==========================================
# 1. CONFIGURATION & DATABASE CONNECTION
# ==========================================
try:
    from Portal_ML_V4.src.config.settings import BASE_DIR, DB_CONNECTION_STRING
except ImportError:
    print("❌ Error: Could not load secure settings. Check your settings.py and .env files.")
    exit()
RAW_DIR = BASE_DIR / "data" / "01_raw" / "pos_data"
CLOSED_DIR = RAW_DIR / "Closed"
DEC_DIR = RAW_DIR / "Dec 2025"
LOCATION_MAP = {
    "centurion_2R":   ["centurion", "c2r"],
    "galleria":       ["galleria"],
    "ngong_milele":   ["milele"],
    "pharmart_abc":   ["abc", "pharmart abc"],
    "portal_2R":      ["portal 2r", "portal_2r", "portal 2 rivers"],
    "portal_cbd":     ["cbd", "portal cbd"]
}
# 🔒 COLUMNS TO KEEP (Mapped exactly to our SQL Tables)
SALES_COLS_MAP = {
    'Department': 'department', 
    'Category': 'category', 
    'Item': 'item_barcode', 
    'Description': 'description', 
    'On Hand': 'on_hand', 
    'Last Sold': 'last_sold', 
    'Qty Sold': 'qty_sold', 
    'Total (Tax Ex)': 'total_tax_ex', 
    'Transaction ID': 'transaction_id', 
    'Date Sold': 'date_sold'
}
CASHIER_COLS_MAP = {
    'Receipt Txn No': 'receipt_txn_no', 
    'Sales Rep': 'sales_rep', 
    'Client Name': 'client_name', 
    'Phone Number': 'phone_number', 
    'Txn Type': 'txn_type', 
    'Ordered Via': 'ordered_via', 
    'Time': 'time', 
    'Amount': 'amount', 
    'Txn Costs': 'txn_costs'
}
# Month name → abbreviation for sheet_date extraction
MONTH_MAP = {
    'jan': 'Jan', 'feb': 'Feb', 'mar': 'Mar', 'apr': 'Apr',
    'may': 'May', 'jun': 'Jun', 'jul': 'Jul', 'aug': 'Aug',
    'sep': 'Sep', 'oct': 'Oct', 'nov': 'Nov', 'dec': 'Dec',
    'january': 'Jan', 'february': 'Feb', 'march': 'Mar', 'april': 'Apr',
    'june': 'Jun', 'july': 'Jul', 'august': 'Aug', 'september': 'Sep',
    'october': 'Oct', 'november': 'Nov', 'december': 'Dec',
}

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def clean_id_col(series):
    return series.astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
def extract_month_from_filename(filename):
    """Extracts month abbreviation from cashier filename e.g. 'Feb' from
    'Centurion 2r Daily Cashier report Feb 2025.xlsm'"""
    import re as _re
    fname = os.path.basename(filename).lower()
    for month_str, month_abbr in MONTH_MAP.items():
        if _re.search(r'\b' + month_str + r'\b', fname):
            return month_abbr
    return None

def is_numbered_sheet(sheet_name):
    """Returns True only for sheets named 01-31."""
    import re as _re
    return bool(_re.fullmatch(r'0[1-9]|[12][0-9]|3[01]', str(sheet_name).strip()))

def is_cashier_file_for_branch(filename, branch_keywords):
    fname = filename.lower()
    return any(k in fname for k in branch_keywords)
def extract_date_from_filename(filename):
    fname = os.path.basename(filename).lower()
    match = re.search(r'(\d{2})\.(\d{2})\.(\d{4})', fname)
    if match: return pd.Timestamp(f"{match.group(3)}-{match.group(2)}-{match.group(1)}")
    match = re.search(r'(?:^|s|sales|_|\s)(\d{6})(?:\.|$)', fname)
    if match: return pd.to_datetime(match.group(1), format='%d%m%y', errors='coerce')
    return pd.NaT
# ==========================================
# 3. DATA LOADING FUNCTIONS
# ==========================================
def load_cashier_file(file_path):
    try:
        month = extract_month_from_filename(str(file_path))
        if month is None:
            print(f"    ⚠️  Could not extract month from '{os.path.basename(str(file_path))}' — sheet_date will be UNKNOWN-xx")

        if str(file_path).endswith('.csv'):
            df = pd.read_csv(file_path, low_memory=False)
            df.columns = df.columns.astype(str).str.strip()
            existing_cols = [c for c in CASHIER_COLS_MAP.keys() if c in df.columns]
            df_cleaned = df[existing_cols].rename(columns=CASHIER_COLS_MAP).copy()
            df_cleaned['sheet_date'] = f"{month}-CSV" if month else "UNKNOWN-CSV"
            df_cleaned['source_sheet'] = os.path.basename(str(file_path))
            return df_cleaned
        else:
            dfs = []
            xls = pd.ExcelFile(file_path)

            for sheet_name in xls.sheet_names:
                # Only load sheets named 01-31 — skip Summary, Sheet1 etc.
                if not is_numbered_sheet(sheet_name):
                    continue

                df_sheet = pd.read_excel(xls, sheet_name=sheet_name)
                df_sheet.columns = df_sheet.columns.astype(str).str.strip()

                if 'Receipt Txn No' not in df_sheet.columns:
                    continue

                existing_cols = [c for c in CASHIER_COLS_MAP.keys() if c in df_sheet.columns]
                if not existing_cols:
                    continue

                df_clean = df_sheet[existing_cols].rename(columns=CASHIER_COLS_MAP).copy()

                sheet_num  = str(sheet_name).strip()
                sheet_date = f"{month}-{sheet_num}" if month else f"UNKNOWN-{sheet_num}"
                df_clean['sheet_date']   = sheet_date
                df_clean['source_sheet'] = f"{os.path.basename(str(file_path))} → Sheet {sheet_name}"
                dfs.append(df_clean)

            if dfs:
                return pd.concat(dfs, ignore_index=True)
            return pd.DataFrame()

    except Exception as e:
        print(f"    ⚠️ Error loading {file_path}: {e}")
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
                df['source_sheet'] = f.name + " -> " + df['source_sheet'].astype(str)
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
                existing_cols = [c for c in SALES_COLS_MAP.keys() if c in df.columns]
                df = df[existing_cols].rename(columns=SALES_COLS_MAP)
                
                if len(df) > 0:
                    df['transaction_id'] = clean_id_col(df['transaction_id'])
                    sales_dfs.append(df)
                    
        except Exception as e:
            print(f"      ❌ Error reading {fname}: {e}")
    if not sales_dfs:
        return pd.DataFrame()
    full_df = pd.concat(sales_dfs, ignore_index=True)
    return full_df
# ==========================================
# 4. MAIN ELT PIPELINE
# ==========================================
def run_postgres_loader():
    print("🌍 STARTING ELT DATABASE LOADER V1...")
    
    if not RAW_DIR.exists():
        print(f"❌ Error: Raw directory not found at {RAW_DIR}")
        return
        
    print("🔌 Connecting to PostgreSQL...")
    engine = create_engine(DB_CONNECTION_STRING)
    
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE raw_sales;"))
        conn.execute(text("TRUNCATE TABLE raw_cashier;"))
    print("🗑️  Emptied old data from staging tables.")

    try:
        for folder_name, branch_keywords in LOCATION_MAP.items():
            loc_path = RAW_DIR / folder_name
            if not loc_path.exists(): continue
            branch_label = folder_name.upper()
            print(f"\n📍 PROCESSING & UPLOADING: {branch_label}")
            # A. LOAD & PUSH SALES
            df_sales = load_sales_files_for_branch(loc_path)
            if not df_sales.empty:
                df_sales['location'] = branch_label
                df_sales = df_sales.replace('#NULL#', None)
                df_sales['item_barcode'] = df_sales['item_barcode'].fillna('UNKNOWN_BARCODE')
                df_sales['total_tax_ex'] = pd.to_numeric(df_sales['total_tax_ex'], errors='coerce')
                df_sales['qty_sold'] = pd.to_numeric(df_sales['qty_sold'], errors='coerce')
                df_sales['on_hand'] = pd.to_numeric(df_sales['on_hand'], errors='coerce')
                df_sales.to_sql('raw_sales', engine, if_exists='append', index=False, chunksize=50000)
                print(f"    ✅ Uploaded Sales: {len(df_sales):,} rows")
            # B. LOAD & PUSH CASHIER
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
                df_cashier['location'] = branch_label
                df_cashier = df_cashier.replace('#NULL#', None)
                df_cashier['amount'] = df_cashier['amount'].astype(str).str.replace(',', '').str.replace(r'[^\d\.\-]', '', regex=True)
                df_cashier['amount'] = pd.to_numeric(df_cashier['amount'], errors='coerce').fillna(0)
                df_cashier['txn_costs'] = df_cashier['txn_costs'].astype(str).str.replace(',', '').str.replace(r'[^\d\.\-]', '', regex=True)
                df_cashier['txn_costs'] = pd.to_numeric(df_cashier['txn_costs'], errors='coerce').fillna(0)
                df_cashier['receipt_txn_no'] = clean_id_col(df_cashier['receipt_txn_no'])
                df_cashier = df_cashier.dropna(subset=['receipt_txn_no'])
                df_cashier.to_sql('raw_cashier', engine, if_exists='append', index=False, chunksize=50000)
                print(f"    ✅ Uploaded Cashier Reports: {len(df_cashier):,} rows")
            else:
                print(f"    ⚠️ No Cashier Data found.")

        print("\n🎉 ELT LOAD COMPLETE! All data is now in PostgreSQL.")

    finally:
        engine.dispose()
        print("🔌 Database connections closed.")

if __name__ == "__main__":
    run_postgres_loader()