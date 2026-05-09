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
POS_DATA_PATH = PROCESSED_DATA_DIR / "pos_data" / "fact_all_locations_sales.csv"

OUTPUT_DIR = PROCESSED_DATA_DIR / "sales_attribution"
OUTPUT_FILE = OUTPUT_DIR / "fact_social_sales_attribution.csv"

# Time window: 60 mins
TIME_WINDOW_MINUTES = 60 

# Regex for money (captures 2200, 2,200)
MONEY_PATTERN = r'(?:Ksh\.?|Kes\.?)?\s*(\d{1,3}(?:,\d{3})*|\d{3,})'

# Anchor Keywords
CONFIRMATION_KEYWORDS = [
    'payment received', 'payment well received', 'received with thanks', 
    'received payment', 'confirmed', 'payment confirmed', 'received'
]

# 🚫 BLACKLIST NUMBERS
BLACKLIST_NUMBERS = [
    247247, 666226,  # Paybill / Acc
    552800, 222666,  # Paybill / Acc
    217004,          # Acc
    542542, 666222,  # Paybill / Acc
    894353,          # Till
    633450           # Till
]

PAYBILL_KEYWORDS = ['paybill', 'till', 'account', 'pay to', 'no.', 'number']

# ==========================================
# 2. SMART EXTRACTION FUNCTIONS
# ==========================================

def extract_valid_numbers_from_text(text_block):
    """
    Helper: Returns a list of valid price candidates from a text block.
    Applies filters (Range, Blacklist, Context).
    """
    candidates = []
    
    for m in re.finditer(MONEY_PATTERN, text_block, re.IGNORECASE):
        val_str = m.group(1).replace(',', '')
        try:
            val = float(val_str)
        except: continue
        
        # Filter 1: Range
        if not (50 <= val < 500000): continue
        
        # Filter 2: Blacklist (Integer Check)
        try:
            if int(val) in BLACKLIST_NUMBERS: continue
        except: pass
        
        # Filter 3: Context Check (Look behind 25 chars)
        start_check = max(0, m.start() - 25)
        context_before = text_block[start_check:m.start()].lower()
        if any(bad in context_before for bad in PAYBILL_KEYWORDS): continue
            
        candidates.append(val)
        
    return candidates

def get_smart_amount(row):
    """
    Extracts 'Sale Amount' with a Safety Net.
    """
    # --- PRIORITY 1: Existing MPESA Data ---
    if pd.notna(row.get('mpesa_amount')) and float(row['mpesa_amount']) > 0:
        return float(row['mpesa_amount'])

    text = str(row.get('full_context', ''))
    if not text: return None

    # Clean text (Remove Phone Numbers)
    clean_text = re.sub(r'(?:07|\+254)\d{8,}', '', text)
    text_lower = clean_text.lower()
    
    # Check for Confirmation
    confirm_indices = []
    for keyword in CONFIRMATION_KEYWORDS:
        matches = [m.start() for m in re.finditer(re.escape(keyword), text_lower)]
        confirm_indices.extend(matches)
    
    # Determine Status
    has_confirmation = len(confirm_indices) > 0
    is_crm_converted = row.get('is_converted') == 1

    if not (has_confirmation or is_crm_converted):
        return None # Not a sale candidate

    # --- PRIORITY 2: PRECISION SCAN (Window around 'Received') ---
    if has_confirmation:
        last_confirm_idx = max(confirm_indices)
        
        # Look back 2000 chars AND forward 100 chars (To catch "Received 2200")
        start_scan = max(0, last_confirm_idx - 2000)
        end_scan = min(len(clean_text), last_confirm_idx + 100)
        
        scan_window = clean_text[start_scan:end_scan]
        
        candidates = extract_valid_numbers_from_text(scan_window)
        if candidates:
            return max(candidates) # Found valid price near confirmation

    # --- PRIORITY 3: GLOBAL FALLBACK ---
    # If the precision scan failed (e.g. price was mentioned 3 days ago), 
    # BUT we know it's a confirmed sale, scan the WHOLE chat.
    
    global_candidates = extract_valid_numbers_from_text(clean_text)
    
    if global_candidates:
        return max(global_candidates)

    return None

def check_staff_confirmation(text):
    if pd.isna(text): return False
    text_lower = str(text).lower()
    return any(k in text_lower for k in CONFIRMATION_KEYWORDS)

# ==========================================
# 3. MAIN PIPELINE
# ==========================================
def run_attribution_pipeline():
    print("🔗 STARTING SOCIAL-TO-POS ATTRIBUTION (With Global Fallback)...")

    if not CHAT_DATA_PATH.exists() or not POS_DATA_PATH.exists():
        print("❌ Error: Missing Input Files.")
        return

    # --- 1. LOAD ---
    print("   📥 Loading Datasets...")
    df_chats = pd.read_csv(CHAT_DATA_PATH, parse_dates=['session_start'])
    df_pos = pd.read_csv(POS_DATA_PATH)

    # --- 2. PREP POS ---
    print("   ⚙️  Preparing POS Dates...")
    df_pos['Sale_Date'] = pd.to_datetime(df_pos['Sale_Date'], errors='coerce')
    def combine_pos_datetime(row):
        try:
            t = pd.to_datetime(str(row['Time']), format='%H:%M').time()
            return pd.Timestamp.combine(row['Sale_Date'].date(), t)
        except: return pd.NaT
    df_pos['POS_Full_DateTime'] = df_pos.apply(combine_pos_datetime, axis=1)
    df_pos = df_pos.dropna(subset=['POS_Full_DateTime'])

    # --- 3. EXTRACT FROM CHATS ---
    print("   🕵️ Scanning Chats...")
    df_chats['Staff_Confirmed'] = df_chats['full_context'].apply(check_staff_confirmation)
    
    mask_relevant = (df_chats['is_converted'] == 1) | (df_chats['Staff_Confirmed'] == True)
    potential_sales = df_chats[mask_relevant].copy()
    
    print(f"   🔹 Filtering: {len(df_chats)} Total -> {len(potential_sales)} Sales Candidates.")

    potential_sales['Extracted_Amount'] = potential_sales.apply(get_smart_amount, axis=1)
    potential_sales = potential_sales.dropna(subset=['Extracted_Amount'])
    print(f"   ✅ Candidates with Prices: {len(potential_sales)}")

    # --- 4. MATCHING ---
    print("   🤝 Matching Chats to Receipts...")
    potential_sales['Join_Date'] = potential_sales['session_start'].dt.date
    df_pos['Join_Date'] = df_pos['Sale_Date'].dt.date
    
    candidates = pd.merge(
        potential_sales,
        df_pos,
        left_on=['Join_Date', 'Extracted_Amount'],
        right_on=['Join_Date', 'Amount'],
        how='inner',
        suffixes=('_Chat', '_POS')
    )

    # --- 5. FILTERS ---
    # Time Filter
    candidates['Time_Diff_Minutes'] = (
        candidates['POS_Full_DateTime'] - candidates['session_start']
    ).abs().dt.total_seconds() / 60.0
    
    matches = candidates[candidates['Time_Diff_Minutes'] <= TIME_WINDOW_MINUTES].copy()

    # --- 6. OUTPUT ---
    print("   📝 Formatting Final Report...")
    cols_to_keep = [
        'Transaction ID', 'Receipt Txn No', 'Location', 'Sale_Date', 'Time',
        'Department', 'Category', 'Item', 'Description', 'Total (Tax Ex)', 'Amount',
        'Client Name', 'Phone Number',
        'Contact ID', 'session_id', 'contact_name', 'session_start', 'Extracted_Amount',
        'Staff_Confirmed', 'is_converted', 'full_context', 'channel_name', 'active_staff',
        'Time_Diff_Minutes'
    ]
    
    available = [c for c in cols_to_keep if c in matches.columns]
    final_df = matches[available]
    final_df = final_df.sort_values('Time_Diff_Minutes')

    final_df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"🚀 SUCCESS: Linked {len(final_df)} Chat Sessions to Sales.")
    print(f"   📂 Output: {OUTPUT_FILE}")
    
    if not final_df.empty:
        print("\n🔎 Match Preview:")
        preview_cols = ['Location', 'Amount', 'Description', 'contact_name', 'Time_Diff_Minutes']
        print(final_df[[c for c in preview_cols if c in final_df.columns]].head())

if __name__ == "__main__":
    run_attribution_pipeline()