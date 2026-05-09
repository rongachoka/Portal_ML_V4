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

# Time window: 90 mins
TIME_WINDOW_MINUTES = 90

# Regex for money (captures 2200, 2,200)
MONEY_PATTERN = r'(?:Ksh\.?|Kes\.?)?\s*(\d{1,3}(?:,\d{3})*|\d{3,})'

# Anchor Keywords (To find the end of the transaction)
CONFIRMATION_KEYWORDS = [
    'payment received', 'payment well received', 'received with thanks', 
    'received payment', 'confirmed', 'payment confirmed', 'received'
]

# 🚫 BLACKLIST NUMBERS (Paybills, Tills, Accounts)
# These numbers will NEVER be treated as a Price, even if they look like one.
BLACKLIST_NUMBERS = [
    247247, 666226,  # Paybill / Acc
    552800, 222666,  # Paybill / Acc
    217004,          # Acc
    542542, 666222,  # Paybill / Acc
    894353,          # Till
    633450           # Till
]

# Blacklist Context (Backup for unlisted Tills)
PAYBILL_KEYWORDS = ['paybill', 'till', 'account', 'pay to', 'no.', 'number']

# ==========================================
# 2. SMART EXTRACTION FUNCTIONS
# ==========================================

def get_smart_amount(row):
    """
    Extracts the most likely 'Sale Amount' by looking back from the 'Received' message.
    Explicitly ignores KNOWN Paybill/Till numbers.
    """
    # --- PRIORITY 1: Existing MPESA Data ---
    if pd.notna(row.get('mpesa_amount')) and float(row['mpesa_amount']) > 0:
        return float(row['mpesa_amount'])

    text = str(row.get('full_context', ''))
    if not text: return None

    # Clean text (Remove Phone Numbers 07xx/254xx to prevent false millions)
    clean_text = re.sub(r'(?:07|\+254)\d{8,}', '', text)
    text_lower = clean_text.lower()
    
    # Find where staff said "Received"
    confirm_indices = []
    for keyword in CONFIRMATION_KEYWORDS:
        matches = [m.start() for m in re.finditer(re.escape(keyword), text_lower)]
        confirm_indices.extend(matches)
    
    # If no "Received", check if 'is_converted' is 1. 
    if not confirm_indices:
        if row.get('is_converted') == 1:
            last_confirm_idx = len(clean_text) # Scan whole chat
        else:
            return None # Not a sale
    else:
        last_confirm_idx = max(confirm_indices)

    # --- THE SMART WINDOW ---
    # Look back 3000 chars from the confirmation point
    start_scan = max(0, last_confirm_idx - 3000)
    scan_window = clean_text[start_scan:last_confirm_idx]
    
    # Find all number candidates in this window
    candidates = []
    
    for m in re.finditer(MONEY_PATTERN, scan_window, re.IGNORECASE):
        val_str = m.group(1).replace(',', '')
        try:
            val = float(val_str)
        except: continue
        
        # Filter 1: Rational Price Range (50 to 500k)
        if not (50 <= val < 500000): continue
        
        # 🛡️ FILTER 2: BLACKLIST CHECK (The Titanium Shield)
        # If the number matches one of your Tills/Paybills, DROP IT immediately.
        if val in BLACKLIST_NUMBERS:
            continue
        
        # Filter 3: CONTEXT CHECK (Backup for unlisted numbers)
        # Look at the 25 characters BEFORE the number
        start_check = max(0, m.start() - 25)
        context_before = scan_window[start_check:m.start()].lower()
        
        if any(bad in context_before for bad in PAYBILL_KEYWORDS):
            continue
            
        candidates.append(val)
    
    if candidates:
        # Return the largest remaining number (The Bill Total)
        return max(candidates)

    return None

def check_staff_confirmation(text):
    if pd.isna(text): return False
    text_lower = str(text).lower()
    return any(k in text_lower for k in CONFIRMATION_KEYWORDS)

# ==========================================
# 3. MAIN PIPELINE
# ==========================================
def run_attribution_pipeline():
    print("🔗 STARTING SOCIAL-TO-POS ATTRIBUTION (Blacklist Enforced)...")

    if not CHAT_DATA_PATH.exists() or not POS_DATA_PATH.exists():
        print("❌ Error: Missing Input Files.")
        return

    # --- STEP 1: LOAD DATA ---
    print("   📥 Loading Datasets...")
    df_chats = pd.read_csv(CHAT_DATA_PATH, parse_dates=['session_start'])
    df_pos = pd.read_csv(POS_DATA_PATH)

    # --- STEP 2: PREPARE POS DATA ---
    print("   ⚙️  Preparing POS Dates...")
    df_pos['Sale_Date'] = pd.to_datetime(df_pos['Sale_Date'], errors='coerce')
    
    def combine_pos_datetime(row):
        try:
            t = pd.to_datetime(str(row['Time']), format='%H:%M').time()
            return pd.Timestamp.combine(row['Sale_Date'].date(), t)
        except: return pd.NaT

    df_pos['POS_Full_DateTime'] = df_pos.apply(combine_pos_datetime, axis=1)
    df_pos = df_pos.dropna(subset=['POS_Full_DateTime'])


    # --- STEP 3: FILTER & EXTRACT FROM CHATS ---
    print("   🕵️ Scanning Chats...")
    
    df_chats['Staff_Confirmed'] = df_chats['full_context'].apply(check_staff_confirmation)
    
    # Filter: Must be Converted OR Have "Received" keyword
    mask_relevant = (df_chats['is_converted'] == 1) | (df_chats['Staff_Confirmed'] == True)
    potential_sales = df_chats[mask_relevant].copy()
    
    print(f"   🔹 Filtering: {len(df_chats)} Total -> {len(potential_sales)} Sales Candidates.")

    # Apply Smart Extraction (With Blacklist)
    potential_sales['Extracted_Amount'] = potential_sales.apply(get_smart_amount, axis=1)
    
    # Drop where no price was found
    potential_sales = potential_sales.dropna(subset=['Extracted_Amount'])
    
    print(f"   ✅ Candidates with Detectable Prices: {len(potential_sales)}")


    # --- STEP 4: MATCHING ---
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

    # --- STEP 5: TIME FILTER ---
    candidates['Time_Diff_Minutes'] = (
        candidates['POS_Full_DateTime'] - candidates['session_start']
    ).abs().dt.total_seconds() / 60.0
    
    matches = candidates[candidates['Time_Diff_Minutes'] <= TIME_WINDOW_MINUTES].copy()


    # --- STEP 6: OUTPUT ---
    print("   📝 Formatting Final Report...")
    
    cols_to_keep = [
        # POS
        'Transaction ID', 'Receipt Txn No', 'Location', 'Sale_Date', 'Time',
        'Department', 'Category', 'Item', 'Description', 'Total (Tax Ex)', 'Amount',
        'Client Name', 'Phone Number',
        # CHAT
        'Contact ID', 'session_id', 'contact_name', 'session_start', 'Extracted_Amount',
        'Staff_Confirmed', 'is_converted', 'full_context', 'channel_name', 'active_staff',
        # METRICS
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