import pandas as pd
import numpy as np
import re
from datetime import timedelta
from pathlib import Path

# ✅ IMPORT PATHS (Keep your V3 settings)
from Portal_ML_V4.src.config.settings import (
    BASE_DIR,
    PROCESSED_DATA_DIR,
)

# ==========================================
# 1. CONFIGURATION
# ==========================================
CHAT_DATA_PATH = PROCESSED_DATA_DIR / "fact_sessions_enriched.csv"
POS_DATA_PATH = PROCESSED_DATA_DIR / "pos_data" / "all_locations_sales_Jan25-Jan26.csv"
OUTPUT_DIR = PROCESSED_DATA_DIR / "sales_attribution"
OUTPUT_FILE = OUTPUT_DIR / "updated_fact_social_sales_attribution.csv"

# ⏱️ TIME WINDOWS
PHONE_WINDOW_HOURS = 96   # 4 Days for Phone matches (High confidence)
PRICE_WINDOW_MINUTES = 2880 # 48 Hours for Price matches (Lower confidence)

# Regex
MONEY_PATTERN = r'(?:Ksh\.?|Kes\.?)?\s*(\d{1,3}(?:,\d{3})*|\d{3,})'
BLACKLIST_NUMBERS = [247247, 666226, 552800, 222666, 217004, 542542, 666222, 894353, 633450]
PAYBILL_KEYWORDS = ['paybill', 'till', 'account', 'pay to', 'no.', 'number', 'acc']

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def normalize_phone_9_digits(val):
    """Extracts last 9 digits for matching (e.g. 712345678)"""
    s = str(val).replace('.0', '')
    digits = re.sub(r'[^\d]', '', s)
    if len(digits) >= 9:
        return digits[-9:]
    return None

def clean_pos_datetime(row):
    # Try Date Sold first
    date_sold = str(row.get('Date Sold', ''))
    if date_sold and date_sold.lower() != 'nan':
        clean_str = date_sold.replace('#', '').strip()
        try: return pd.to_datetime(clean_str)
        except: pass
    # Fallback to Sale_Date + Time
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
    # 1. Prefer MPESA Engine if available
    if pd.notna(row.get('mpesa_amount')) and float(row['mpesa_amount']) > 0:
        return float(row['mpesa_amount'])
    
    # 2. Fallback to Text Extraction
    text = str(row.get('full_context', ''))
    clean_text = re.sub(r'(?:07|\+254)\d{8,}', '', text) # Remove phone numbers from text
    
    candidates = []
    for m in re.finditer(MONEY_PATTERN, clean_text, re.IGNORECASE):
        try:
            val = float(m.group(1).replace(',', ''))
            if 50 <= val < 500000 and int(val) not in BLACKLIST_NUMBERS:
                candidates.append(val)
        except: continue
        
    return max(candidates) if candidates else None

def calculate_content_score(row):
    chat_text = str(row.get('full_context', '')).lower()
    pos_desc = str(row.get('Description', '')).lower()
    junk = ['150ml', '200ml', '50ml', 'pcs', 'tabs', 'caps', 'g', 'ml', 'no', 'x', 'mg']
    pos_tokens = {t for t in re.findall(r'\w+', pos_desc) if len(t) > 2 and t not in junk}
    
    if not pos_tokens: return 0.1 
    hits = sum(1 for t in pos_tokens if t in chat_text)
    score = (hits / len(pos_tokens)) * 100
    
    if str(row.get('Category', '')).lower() in str(row.get('final_tags', '')).lower(): 
        score += 30 
    return score

# ==========================================
# 3. MAIN PIPELINE
# ==========================================
def run_hybrid_attribution():
    print("🔗 STARTING HYBRID ATTRIBUTION (Phone First -> Price Fallback)...")

    if not CHAT_DATA_PATH.exists() or not POS_DATA_PATH.exists():
        print("❌ Error: Missing Input Files.")
        return

    # 1. LOAD
    print("   📥 Loading Datasets...")
    df_chats = pd.read_csv(CHAT_DATA_PATH, parse_dates=['session_start'])
    df_pos = pd.read_csv(POS_DATA_PATH, low_memory=False)

    # 2. PREP DATA
    print("   ⚙️  Preparing Keys...")
    df_pos['POS_Full_DateTime'] = df_pos.apply(clean_pos_datetime, axis=1)
    df_pos = df_pos.dropna(subset=['POS_Full_DateTime'])
    
    # Calculate Transaction Total for Price Matching
    df_pos['Total (Tax Ex)'] = pd.to_numeric(df_pos['Total (Tax Ex)'], errors='coerce').fillna(0)
    if 'Transaction_Total' not in df_pos.columns:
        df_pos['Transaction_Total'] = df_pos.groupby('Transaction ID')['Total (Tax Ex)'].transform('sum')
    
    # Normalize Phones
    df_chats['match_phone'] = df_chats['phone_number'].apply(normalize_phone_9_digits)
    
    # Check if POS has Clean_Phone or Phone Number
    pos_phone_col = 'Clean_Phone' if 'Clean_Phone' in df_pos.columns else 'Phone Number'
    df_pos['match_phone'] = df_pos[pos_phone_col].apply(normalize_phone_9_digits)

    # Filter Chats to Converted Only
    potential_sales = df_chats[(df_chats['is_converted'] == 1) | (df_chats['mpesa_amount'] > 0)].copy()
    potential_sales['Extracted_Amount'] = potential_sales.apply(get_smart_amount, axis=1)
    
    print(f"   pool: {len(potential_sales)} converted sessions.")

    # ==========================================
    # STRATEGY A: PHONE MATCHING (Gold Standard)
    # ==========================================
    print("   📲 Running Strategy A: Phone Number Matching...")
    
    # Get unique transactions from POS to avoid duplicating rows yet
    pos_unique = df_pos.drop_duplicates(subset=['Transaction ID'])
    
    phone_merge = pd.merge(
        potential_sales.dropna(subset=['match_phone']),
        pos_unique.dropna(subset=['match_phone']),
        on='match_phone',
        how='inner',
        suffixes=('_Chat', '_POS')
    )
    
    # Time Check
    phone_merge['Time_Diff_Hours'] = (phone_merge['POS_Full_DateTime'] - phone_merge['session_start']).abs().dt.total_seconds() / 3600.0
    valid_phone = phone_merge[phone_merge['Time_Diff_Hours'] <= PHONE_WINDOW_HOURS].copy()
    
    # Deduplicate (Best time match per session)
    valid_phone.sort_values('Time_Diff_Hours', inplace=True)
    best_phone = valid_phone.drop_duplicates(subset=['session_id'])
    best_phone['Match_Type'] = 'Phone_Verified'
    
    print(f"      ✅ Found {len(best_phone)} verified phone matches.")

    # ==========================================
    # STRATEGY B: PRICE MATCHING (Fallback)
    # ==========================================
    print("   🕵️ Running Strategy B: Price + Time Fallback...")
    
    # Remove already matched sessions
    matched_ids = set(best_phone['session_id'])
    remaining_chats = potential_sales[~potential_sales['session_id'].isin(matched_ids)].copy()
    remaining_chats = remaining_chats.dropna(subset=['Extracted_Amount'])
    
    # Remove already matched transactions
    matched_txns = set(best_phone['Transaction ID'])
    remaining_pos = df_pos[~df_pos['Transaction ID'].isin(matched_txns)].drop_duplicates(subset=['Transaction ID'])
    
    # Create Match Keys
    remaining_chats['Amount_Key'] = remaining_chats['Extracted_Amount'].astype(int)
    remaining_pos['Amount_Key'] = remaining_pos['Transaction_Total'].astype(int)
    
    price_merge = pd.merge(
        remaining_chats,
        remaining_pos,
        on='Amount_Key',
        how='inner',
        suffixes=('_Chat', '_POS')
    )
    
    # Time Check
    price_merge['Time_Diff_Minutes'] = (price_merge['POS_Full_DateTime'] - price_merge['session_start']).abs().dt.total_seconds() / 60.0
    valid_price = price_merge[price_merge['Time_Diff_Minutes'] <= PRICE_WINDOW_MINUTES].copy()
    
    # Tie Breaker: Content Score
    valid_price['Content_Score'] = valid_price.apply(calculate_content_score, axis=1)
    valid_price.sort_values(['Time_Diff_Minutes', 'Content_Score'], ascending=[True, False], inplace=True)
    
    best_price = valid_price.drop_duplicates(subset=['session_id'])
    best_price['Match_Type'] = 'Price_Inferred'
    
    print(f"      ✅ Found {len(best_price)} price matches.")

    # ==========================================
    # 4. CONSOLIDATE & EXPAND
    # ==========================================
    print("   📦 Consolidating Results...")
    
    # Combine just the ID links
    cols_link = ['session_id', 'Transaction ID', 'Match_Type', 'Time_Diff_Minutes']
    
    # Calculate minutes for phone matches for consistency
    best_phone['Time_Diff_Minutes'] = best_phone['Time_Diff_Hours'] * 60
    
    all_links = pd.concat([best_phone[cols_link], best_price[cols_link]])
    
    # 🚨 EXPAND: Merge back to full POS data to get line items
    final_df = pd.merge(all_links, df_pos, on='Transaction ID', how='inner')
    
    # Merge back Chat Details
    chat_cols = ['session_id', 'Contact ID', 'contact_name', 'session_start', 
                 'Extracted_Amount', 'is_converted', 'full_context', 
                 'channel_name', 'active_staff', 'final_tags']
    
    # Get original chat data (no dupes)
    chat_source = df_chats[chat_cols].drop_duplicates('session_id')
    final_df = pd.merge(final_df, chat_source, on='session_id', how='left')

    # Formatting
    final_df.sort_values(['Time_Diff_Minutes', 'Transaction ID'], inplace=True)
    final_df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"🚀 SUCCESS: Linked {final_df['session_id'].nunique()} Chat Sessions.")
    print(f"   📂 Output: {OUTPUT_FILE}")

if __name__ == "__main__":
    run_hybrid_attribution()