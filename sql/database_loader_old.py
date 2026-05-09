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

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def clean_id_col(series):
    return series.astype(str).str.replace(r'\.0$', '', regex=True).str.strip()

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
        if str(file_path).endswith('.csv'):
            df = pd.read_csv(file_path, low_memory=False)
            df['source_sheet'] = 'CSV_File'
        else:
            dfs = []
            xls = pd.ExcelFile(file_path)
            
            for sheet_number, sheet_name in enumerate(xls.sheet_names, start=1):
                df_sheet = pd.read_excel(xls, sheet_name=sheet_name)
                df_sheet.columns = df_sheet.columns.str.strip()
                
                if 'Receipt Txn No' in df_sheet.columns:
                    # 🏷️ Create our custom metadata column
                    df_sheet['source_sheet'] = f"Sheet_{sheet_number}: {sheet_name}"
                    dfs.append(df_sheet)
                    
            if dfs:
                df = pd.concat(dfs, ignore_index=True)
            else:
                return pd.DataFrame()

        df.columns = df.columns.str.strip()
        
        # 🧹 This step deletes everything not in the map (including our custom column!)
        existing_cols = [c for c in CASHIER_COLS_MAP.keys() if c in df.columns]
        df_cleaned = df[existing_cols].rename(columns=CASHIER_COLS_MAP)
        
        # 🚨 THE FIX: Safely glue our metadata back onto the clean dataset
        df_cleaned['source_sheet'] = df.get('source_sheet', 'Unknown Sheet')
        
        return df_cleaned
        
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
                # 🚨 THE UPGRADE: Stamp the file name in front of the sheet name
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
    print("🌍 STARTING ELT DATABASE LOADER...")
    
    if not RAW_DIR.exists():
        print(f"❌ Error: Raw directory not found at {RAW_DIR}")
        return
        
    # Connect to PostgreSQL
    print("🔌 Connecting to PostgreSQL...")
    engine = create_engine(DB_CONNECTION_STRING)
    
    # Empty the tables before loading to prevent duplicate rows if you run this multiple times
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE raw_sales;"))
        conn.execute(text("TRUNCATE TABLE raw_cashier;"))
    print("🗑️  Emptied old data from staging tables.")

    for folder_name, branch_keywords in LOCATION_MAP.items():
        loc_path = RAW_DIR / folder_name
        if not loc_path.exists(): continue

        branch_label = folder_name.upper()
        print(f"\n📍 PROCESSING & UPLOADING: {branch_label}")

        # A. LOAD & PUSH SALES
        df_sales = load_sales_files_for_branch(loc_path)
        if not df_sales.empty:
            df_sales['location'] = branch_label # 🏷️ Add Location Label
            
            # 🧹 NEW: Destroy the literal text "#NULL#" across all columns
            df_sales = df_sales.replace('#NULL#', None)
            
            # Clean numeric types before SQL (Forcing errors='coerce' turns bad text into blanks)
            df_sales['total_tax_ex'] = pd.to_numeric(df_sales['total_tax_ex'], errors='coerce')
            df_sales['qty_sold'] = pd.to_numeric(df_sales['qty_sold'], errors='coerce')
            df_sales['on_hand'] = pd.to_numeric(df_sales['on_hand'], errors='coerce')  # 👈 Added this!
            
            # 🚀 PUSH TO POSTGRES
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
            df_cashier['location'] = branch_label # 🏷️ Add Location Label
            
            # 🧹 Clean literal text "#NULL#" across all columns
            df_cashier = df_cashier.replace('#NULL#', None)
            
            # Clean numeric types before SQL (Strips out letters like "2K DEPO" and forces numbers)
            df_cashier['amount'] = df_cashier['amount'].astype(str).str.replace(',', '').str.replace(r'[^\d\.\-]', '', regex=True)
            df_cashier['amount'] = pd.to_numeric(df_cashier['amount'], errors='coerce').fillna(0)
            
            # 🚨 NEW: Scrub the txn_costs column!
            df_cashier['txn_costs'] = df_cashier['txn_costs'].astype(str).str.replace(',', '').str.replace(r'[^\d\.\-]', '', regex=True)
            df_cashier['txn_costs'] = pd.to_numeric(df_cashier['txn_costs'], errors='coerce').fillna(0)
            
            df_cashier['receipt_txn_no'] = clean_id_col(df_cashier['receipt_txn_no'])
            
            # 🚀 PUSH TO POSTGRES
            df_cashier.to_sql('raw_cashier', engine, if_exists='append', index=False, chunksize=50000)
            print(f"    ✅ Uploaded Cashier Reports: {len(df_cashier):,} rows")
        else:
            print(f"    ⚠️ No Cashier Data found.")

    print("\n🎉 ELT LOAD COMPLETE! All data is now in PostgreSQL.")

if __name__ == "__main__":
    run_postgres_loader()