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
OUTPUT_DIR = PROCESSED_DATA_DIR / "pos_data"
OUTPUT_FILE = OUTPUT_DIR / "fact_all_locations_sales.csv"

LOCATION_MAP = {
    "centurion_2R": "Centurion",
    "galleria": "Galleria",
    "ngong_milele": "Milele",
    "pharmart_abc": "ABC Place",
    "portal_2R": "Portal 2R",
    "portal_cbd": "CBD"
}

# 🔒 FIREWALLS
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

def clean_time_col(val):
    """
    Robust Time Cleaner:
    - Handles floats: 12.30 -> "12:30"
    - Handles strings: "12:30" -> "12:30"
    - Handles NaNs -> None
    """
    if pd.isna(val) or val == '':
        return None
        
    # Convert to string to check format
    s_val = str(val).strip()
    
    # If it already looks like a time (has a colon), keep it
    if ':' in s_val:
        return s_val

    # If it's a dot-separated number (12.30)
    try:
        f_val = float(val)
        return "{:.2f}".format(f_val).replace('.', ':')
    except (ValueError, TypeError):
        return None

def apply_whitelist(df, whitelist):
    existing_cols = [c for c in whitelist if c in df.columns]
    return df[existing_cols].copy()

def extract_date_from_filename(filename):
    s = filename.lower().replace('.csv', '').replace('.xlsx', '').replace('.xlsm', '')
    s = s.strip()
    
    DEFAULT_YEAR = 2026
    DEFAULT_MONTH = 1

    # PATTERN 0: Typo 7-Digit (2701026 -> 27/01/26)
    match = re.search(r"^(\d{2})(\d{2})0(2[0-9])$", s)
    if match:
        d, m, y_short = map(int, match.groups())
        try: return datetime(2000 + y_short, m, d)
        except ValueError: pass

    # PATTERN 1: DD.MM.YY
    match = re.search(r"(\d{1,2})\.(\d{1,2})\.(\d{2,4})", s)
    if match:
        d, m, y = map(int, match.groups())
        if y < 100: y += 2000
        try: return datetime(y, m, d)
        except ValueError: pass

    # PATTERN 2: Six Digit DDMMYY
    match = re.search(r"(?<!\d)(\d{2})(\d{2})(2[3-6])(?!\d)", s)
    if match:
        d, m, y_short = map(int, match.groups())
        try: return datetime(2000 + y_short, m, d)
        except ValueError: pass

    # PATTERN 3: Day + Month Name
    match = re.search(r"(\d{1,2})\s*(?:st|nd|rd|th)?\s*([a-z]{3})", s)
    if match:
        day, month_str = match.groups()
        try:
            m_obj = datetime.strptime(month_str, "%b")
            y_match = re.search(r"202[3-6]", s)
            y = int(y_match.group()) if y_match else DEFAULT_YEAR
            return datetime(y, m_obj.month, int(day))
        except ValueError: pass

    # PATTERN 4: Compact Day+Month
    match = re.search(r"^(\d{2})1sales", s)
    if match:
        try: return datetime(DEFAULT_YEAR, 1, int(match.group(1)))
        except ValueError: pass
        
    # PATTERN 5: Sales + Day
    match = re.search(r"sales(\d{1,2})$", s)
    if match:
        try: return datetime(DEFAULT_YEAR, DEFAULT_MONTH, int(match.group(1)))
        except ValueError: pass
    
    match = re.search(r"^(\d{1,2})(?:st|nd|rd|th)?\s*sales", s)
    if match:
        try: return datetime(DEFAULT_YEAR, DEFAULT_MONTH, int(match.group(1)))
        except ValueError: pass

    return datetime.min

def load_cashier_report(folder_path):
    files = glob.glob(os.path.join(folder_path, "*Cashier report*.xlsm")) + \
            glob.glob(os.path.join(folder_path, "*Cashier report*.xlsx")) + \
            glob.glob(os.path.join(folder_path, "*Cashier report*.csv"))
    
    if not files:
        print(f"    ⚠️ No Cashier Report found.")
        return pd.DataFrame()
    
    file_path = files[0]

    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path, low_memory=False)
        df.columns = df.columns.str.strip()
        df = apply_whitelist(df, CASHIER_COLS_KEEP)
    else:
        all_sheets = pd.read_excel(file_path, sheet_name=None)
        valid_dfs = []
        for sheet_name, df_sheet in all_sheets.items():
            df_sheet.columns = df_sheet.columns.str.strip()
            if 'Receipt Txn No' in df_sheet.columns:
                df_clean = apply_whitelist(df_sheet, CASHIER_COLS_KEEP)
                valid_dfs.append(df_clean)
        
        df = pd.concat(valid_dfs, ignore_index=True) if valid_dfs else pd.DataFrame()

    if df.empty: return df

    if 'Receipt Txn No' in df.columns:
        df['Receipt Txn No'] = clean_id_col(df['Receipt Txn No'])
    
    if 'Time' in df.columns:
        df['Time_Str'] = df['Time'].apply(clean_time_col)
        df['Time'] = df['Time_Str']
        df.drop(columns=['Time_Str'], inplace=True)

    return df

def load_sales_files(folder_path):
    all_files = glob.glob(os.path.join(folder_path, "*"))
    candidates = []

    for f in all_files:
        fname = os.path.basename(f)
        fname_lower = fname.lower()
        
        if "cashier" in fname_lower: continue
        if fname_lower.endswith(".sql"): continue
        if not (fname_lower.endswith(".csv") or fname_lower.endswith(".xlsx") or fname_lower.endswith(".xlsm")): continue
        if "2023" in fname_lower: continue 

        file_date = extract_date_from_filename(fname)
        
        if file_date != datetime.min:
            candidates.append((file_date, f))

    if not candidates:
        return pd.DataFrame()

    candidates.sort(key=lambda x: x[0])
    best_date, latest_file = candidates[-1]
    fname = os.path.basename(latest_file)
    print(f"    🏆 Selected: {fname} (Date: {best_date.strftime('%Y-%m-%d')})")

    try:
        if fname.lower().endswith(".csv"):
            try:
                df = pd.read_csv(latest_file, low_memory=False)
            except UnicodeDecodeError:
                df = pd.read_csv(latest_file, low_memory=False, encoding='cp1252')
        else:
            df = pd.read_excel(latest_file)
        
        df.columns = df.columns.str.strip()

        if 'Transaction ID' in df.columns:
            df = apply_whitelist(df, SALES_COLS_KEEP)
            df['Source_File'] = fname
            df['Transaction ID'] = clean_id_col(df['Transaction ID'])
            return df
            
    except Exception as e:
        print(f"      ❌ Error reading {fname}: {e}")

    return pd.DataFrame()

# ==========================================
# 3. MAIN PIPELINE
# ==========================================
def run_multi_location_etl():
    print("🌍 STARTING MULTI-LOCATION POS PIPELINE (Type-Safe & Sanitized)...")
    
    if not RAW_DIR.exists():
        print(f"❌ Error: Raw directory not found at {RAW_DIR}")
        return
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    master_list = []

    for folder_name, loc_label in LOCATION_MAP.items():
        loc_path = RAW_DIR / folder_name
        
        if not loc_path.exists(): continue

        print(f"\n📍 Processing: {loc_label}")

        df_cashier = load_cashier_report(loc_path)
        df_sales = load_sales_files(loc_path)

        if df_sales.empty:
            print(f"    ⚠️ No valid dated Sales Data found.")
            continue

        print(f"    ✅ Loaded: {len(df_sales):,} Rows")

        if not df_cashier.empty:
            # --- 🔧 FIX: AGGREGATE CASHIER DATA (Handle Split Payments) ---
            
            # 1. Sort for Text Quality (Longest name first)
            if 'Client Name' in df_cashier.columns:
                df_cashier['Name_Len'] = df_cashier['Client Name'].astype(str).str.len()
                df_cashier.sort_values('Name_Len', ascending=False, inplace=True)
                df_cashier.drop(columns=['Name_Len'], inplace=True)

            # 🚨 FIX 1: Convert Time to String (Prevents min() crash)
            if 'Time' in df_cashier.columns:
                df_cashier['Time'] = df_cashier['Time'].astype(str)

            # 🚨 FIX 2: Force Money Columns to Float (Prevents sum() crash)
            money_cols = ['Amount', 'Txn Costs']
            for col in money_cols:
                if col in df_cashier.columns:
                    # Coerce errors turns "1,000" or " " into NaN, then we fill with 0.0
                    df_cashier[col] = pd.to_numeric(df_cashier[col], errors='coerce').fillna(0.0)

            # 2. Define Aggregation Logic
            agg_rules = {
                'Amount': 'sum',      
                'Txn Costs': 'sum',
                'Time': 'min',        
            }
            
            # Add dynamic columns to agg rules (Client Name, Sales Rep, etc)
            other_cols = [c for c in df_cashier.columns if c not in ['Receipt Txn No', 'Amount', 'Txn Costs', 'Time']]
            for c in other_cols:
                agg_rules[c] = 'first'

            # 3. Group By Receipt ID
            df_cashier = df_cashier.groupby('Receipt Txn No', as_index=False).agg(agg_rules)
            # -------------------------------------------------------------

            df_merged = pd.merge(
                df_sales,
                df_cashier,
                left_on='Transaction ID',
                right_on='Receipt Txn No',
                how='left'
            )
            
            # --- 🛡️ AUDIT LOGIC 🛡️ ---
            receipt_sums = df_sales.groupby('Transaction ID')['Total (Tax Ex)'].sum().reset_index()
            receipt_sums.rename(columns={'Total (Tax Ex)': 'Calc_Items_Total'}, inplace=True)
            
            df_merged = pd.merge(df_merged, receipt_sums, on='Transaction ID', how='left')
            
            df_merged['Calc_Items_Total'] = pd.to_numeric(df_merged['Calc_Items_Total'], errors='coerce').fillna(0)
            df_merged['Amount'] = pd.to_numeric(df_merged['Amount'], errors='coerce').fillna(0)
            
            df_merged['Audit_Diff'] = df_merged['Calc_Items_Total'] - df_merged['Amount']
            
            conditions = [
                (df_merged['Amount'] == 0) | (df_merged['Calc_Items_Total'] == 0),
                (abs(df_merged['Audit_Diff']) > 1.0)
            ]
            choices = ['Not Audited', 'MISMATCH']
            
            df_merged['Audit_Status'] = np.select(conditions, choices, default='Match')

        else:
            df_merged = df_sales
            print("    ⚠️ Warning: Merging without Cashier data (Skipping Audit)")
            df_merged['Audit_Status'] = 'No Cashier Data'

        df_merged['Location'] = loc_label
        df_merged = df_merged[df_merged['Transaction ID'].str.len() > 0]
        
        master_list.append(df_merged)

    if master_list:
        print("\n🏗️  Stacking All Locations...")
        final_df = pd.concat(master_list, ignore_index=True)
        
        if 'Date Sold' in final_df.columns:
            final_df['Date_Clean'] = final_df['Date Sold'].astype(str).str.replace('#', '')
            final_df['Sale_Date'] = pd.to_datetime(final_df['Date_Clean'], errors='coerce').dt.date

        final_df.to_csv(OUTPUT_FILE, index=False)
        print(f"🚀 PIPELINE SUCCESS!")
        print(f"   📂 Output Saved: {OUTPUT_FILE}")
        print(f"   💰 Total Revenue: {final_df['Total (Tax Ex)'].sum():,.2f}")
        print(f"   🧾 Total Rows: {len(final_df):,}")
        
        if 'Audit_Status' in final_df.columns:
            mismatches = final_df[final_df['Audit_Status'] == 'MISMATCH']
            print(f"\n🔎 Audit Summary:")
            print(f"   ⚠️ Price Mismatches: {len(mismatches):,}")
            
    else:
        print("\n❌ No data processed.")

if __name__ == "__main__":
    run_multi_location_etl()