import pandas as pd
import numpy as np
import re
from datetime import timedelta

# ✅ IMPORT PATHS
from Portal_ML_V4.src.config.settings import (
    BASE_DIR,
    PROCESSED_DATA_DIR,
)

# ==========================================
# 1. CONFIGURATION
# ==========================================
CHAT_DATA_PATH = PROCESSED_DATA_DIR / "fact_sessions_enriched.csv"
POS_DATA_PATH = PROCESSED_DATA_DIR / "pos_data" / "fact_all_locations_sales_Jan25-Jan26.csv"

OUTPUT_DIR = PROCESSED_DATA_DIR / "sales_attribution"
OUTPUT_FILE = OUTPUT_DIR / "jan_25_26_fact_social_sales_attribution.csv"

# ⏱️ TIME WINDOW: 24 Hours
TIME_WINDOW_MINUTES = 1440 

# Regex for money
MONEY_PATTERN = r'(?:Ksh\.?|Kes\.?)?\s*(\d{1,3}(?:,\d{3})*|\d{3,})'

# Anchor Keywords
CONFIRMATION_KEYWORDS = [
    'payment received', 'payment well received', 'received with thanks', 
    'received payment', 'confirmed', 'payment confirmed', 'received'
]

# 🚫 BLACKLIST NUMBERS
BLACKLIST_NUMBERS = [
    247247, 666226, 552800, 222666, 217004, 542542, 666222, 894353, 633450
]
PAYBILL_KEYWORDS = ['paybill', 'till', 'account', 'pay to', 'no.', 'number']

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def clean_pos_datetime(row):
    """ Priority 1: Date Sold (Exact). Priority 2: Sale_Date + Time """
    date_sold = str(row.get('Date Sold', ''))
    if date_sold and date_sold.lower() != 'nan':
        clean_str = date_sold.replace('#', '').strip()
        try: return pd.to_datetime(clean_str)
        except: pass

    try:
        s_date = pd.to_datetime(row['Sale_Date'])
        t_val = row.get('Time')
        if pd.isna(t_val): return s_date 
        t_str = str(t_val).strip()
        if ':' in t_str:
            t = pd.to_datetime(t_str, format='%H:%M').time()
        else: return s_date 
        return pd.Timestamp.combine(s_date.date(), t)
    except: return pd.NaT

def get_smart_amount(row):
    if pd.notna(row.get('mpesa_amount')) and float(row['mpesa_amount']) > 0:
        return float(row['mpesa_amount'])

    text = str(row.get('full_context', ''))
    if not text: return None

    clean_text = re.sub(r'(?:07|\+254)\d{8,}', '', text)
    text_lower = clean_text.lower()
    
    confirm_indices = []
    for keyword in CONFIRMATION_KEYWORDS:
        matches = [m.start() for m in re.finditer(re.escape(keyword), text_lower)]
        confirm_indices.extend(matches)
    
    has_confirmation = len(confirm_indices) > 0
    is_crm_converted = row.get('is_converted') == 1

    if not (has_confirmation or is_crm_converted): return None 

    candidates = []
    scan_window = text_lower
    
    if has_confirmation:
        last_confirm_idx = max(confirm_indices)
        start_scan = max(0, last_confirm_idx - 2000)
        end_scan = min(len(clean_text), last_confirm_idx + 100)
        scan_window = clean_text[start_scan:end_scan]

    for m in re.finditer(MONEY_PATTERN, scan_window, re.IGNORECASE):
        val_str = m.group(1).replace(',', '')
        try:
            val = float(val_str)
            if 50 <= val < 500000:
                if int(val) not in BLACKLIST_NUMBERS:
                    candidates.append(val)
        except: continue
        
    if candidates: return max(candidates)
    return None

def check_staff_confirmation(text):
    if pd.isna(text): return False
    text_lower = str(text).lower()
    return any(k in text_lower for k in CONFIRMATION_KEYWORDS)

# ==========================================
# 3. MAIN PIPELINE
# ==========================================
def run_attribution_pipeline():
    print("🔗 STARTING ATTRIBUTION (Item-Level Detail)...")

    if not CHAT_DATA_PATH.exists() or not POS_DATA_PATH.exists():
        print("❌ Error: Missing Input Files.")
        return

    # --- 1. LOAD ---
    print("   📥 Loading Datasets...")
    df_chats = pd.read_csv(CHAT_DATA_PATH, parse_dates=['session_start'])
    df_pos = pd.read_csv(POS_DATA_PATH, low_memory=False)

    # --- 2. PREP POS ---
    print("   ⚙️  Cleaning POS Times...")
    df_pos['POS_Full_DateTime'] = df_pos.apply(clean_pos_datetime, axis=1)
    df_pos = df_pos.dropna(subset=['POS_Full_DateTime'])
    
    # We DO NOT aggregate POS items here. We keep them separate.
    # But we need integer keys for matching.
    
    # --- 3. EXTRACT FROM CHATS ---
    print("   🕵️ Scanning Chats...")
    df_chats['Staff_Confirmed'] = df_chats['full_context'].apply(check_staff_confirmation)
    mask_relevant = (df_chats['is_converted'] == 1) | (df_chats['Staff_Confirmed'] == True)
    potential_sales = df_chats[mask_relevant].copy()
    
    potential_sales['Extracted_Amount'] = potential_sales.apply(get_smart_amount, axis=1)
    potential_sales = potential_sales.dropna(subset=['Extracted_Amount'])
    
    potential_sales['Amount_Key'] = potential_sales['Extracted_Amount'].astype(int)
    df_pos['Amount_Key'] = pd.to_numeric(df_pos['Amount'], errors='coerce').fillna(0).astype(int)
    
    print(f"   ✅ Candidates with Prices: {len(potential_sales)}")

    # --- 4. MATCHING (Logic Updated to Keep All Items) ---
    print("   🤝 Matching Chats to Receipts...")
    
    # Step A: Perform the Merge (This creates multiple rows per chat if receipt has multiple items)
    candidates = pd.merge(
        potential_sales,
        df_pos,
        on='Amount_Key',
        how='inner',
        suffixes=('_Chat', '_POS')
    )
    
    # Step B: Filter Time
    candidates['Time_Diff_Minutes'] = (
        candidates['POS_Full_DateTime'] - candidates['session_start']
    ).abs().dt.total_seconds() / 60.0
    
    matches = candidates[candidates['Time_Diff_Minutes'] <= TIME_WINDOW_MINUTES].copy()
    
    # Step C: Select BEST Receipt per Chat (Crucial Step!)
    # We want all items from ONE receipt, not items from 5 different receipts that happened to have the same price.
    
    # 1. Rank matches by Time Difference (Smallest diff is best)
    matches['Rank'] = matches.groupby('session_id')['Time_Diff_Minutes'].rank(method='min')
    
    # 2. Identify the "Winning" Transaction ID for each session (The one with Rank 1)
    best_receipts = matches[matches['Rank'] == 1][['session_id', 'Transaction ID']].drop_duplicates()
    
    # 3. Filter our matches to ONLY include rows belonging to the Winning Receipt
    final_df = pd.merge(
        matches,
        best_receipts,
        on=['session_id', 'Transaction ID'],
        how='inner'
    )

    # --- 5. OUTPUT ---
    print("   📝 Formatting Final Report...")
    cols_to_keep = [
        'Transaction ID', 'Receipt Txn No', 'Location', 'Sale_Date', 'Time', 'Date Sold',
        'Department', 'Category', 'Item', 'Description', 'Total (Tax Ex)', 'Amount',
        'Client Name', 'Phone Number',
        'Contact ID', 'session_id', 'contact_name', 'session_start', 'Extracted_Amount',
        'Staff_Confirmed', 'is_converted', 'full_context', 'channel_name', 'active_staff',
        'Time_Diff_Minutes', 'POS_Full_DateTime'
    ]
    
    available = [c for c in cols_to_keep if c in final_df.columns]
    final_df = final_df[available]
    
    final_df = final_df.sort_values(['Time_Diff_Minutes', 'Transaction ID'])

    final_df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"🚀 SUCCESS: Linked {final_df['session_id'].nunique()} Chat Sessions to {len(final_df)} Line Items.")
    print(f"   📂 Output: {OUTPUT_FILE}")
    
    if not final_df.empty:
        print("\n🔎 Match Preview (Multi-Item Receipts):")
        preview_cols = ['Location', 'Amount', 'Description', 'Total (Tax Ex)']
        print(final_df[[c for c in preview_cols if c in final_df.columns]].head(10))

if __name__ == "__main__":
    run_attribution_pipeline()