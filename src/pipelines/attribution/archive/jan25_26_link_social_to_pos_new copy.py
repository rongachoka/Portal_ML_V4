import re
from difflib import SequenceMatcher

import numpy as np
import pandas as pd

# ✅ IMPORT PATHS & SETTINGS
from Portal_ML_V4.src.config.settings import PROCESSED_DATA_DIR, SESSION_GAP_HOURS
from Portal_ML_V4.src.utils.phone import normalize_phone


# ==========================================
# 1. CONFIGURATION
# ==========================================
CHAT_DATA_PATH = PROCESSED_DATA_DIR / "fact_sessions_enriched.csv"
POS_DATA_PATH = PROCESSED_DATA_DIR / "pos_data" / "all_locations_sales_Jan25-Jan26.csv"
OUTPUT_FILE = PROCESSED_DATA_DIR / "sales_attribution" / "attributed_sales_waterfall_v7.csv"

# Time boundaries based on settings.py
MAX_HOURS = SESSION_GAP_HOURS
MIN_HOURS = -1  # 1-hour buffer for POS/Chat clock sync discrepancies

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
# Moved the logic to utils.phone for single source of truth
# def normalize_phone(val):
#     """Strict 9-digit normalization with enhanced NaN handling."""
#     if pd.isna(val) or str(val).strip().lower() == 'nan':
#         return None
        
#     s = str(val).replace('.0', '').strip().lstrip("'")  # 🟢 strip leading apostrophe
#     if 'E' in s.upper() or 'e' in s:
#         try:
#             s = "{:.0f}".format(float(s))
#         except ValueError:
#             pass
            
#     digits = re.sub(r'[^\d]', '', s)
#     if len(digits) >= 9:
#         return digits[-9:]
#     return None


def normalize_name(text):
    """Cleans names for string similarity comparisons."""
    if pd.isna(text) or str(text).strip().lower() == 'nan':
        return ""
    return re.sub(r'[^a-zA-Z\s]', '', str(text).lower()).strip()


def check_name_similarity(name1, name2):
    """Compares names using both fuzzy matching and token intersection."""
    n1 = normalize_name(name1)
    n2 = normalize_name(name2)
    
    if len(n1) < 3 or len(n2) < 3:
        return False
        
    blacklist = ["unknown", "customer", "guest", "cash sale", "walking client"]
    if any(b in n1 for b in blacklist) or any(b in n2 for b in blacklist):
        return False
        
    if n1 in n2 or n2 in n1:
        return True
        
    # Sequence matching
    if SequenceMatcher(None, n1, n2).ratio() > 0.85:
        return True
        
    # Token overlap fallback (e.g., "Mary Akinyi" vs "Mary A")
    tokens1 = set(n1.split())
    tokens2 = set(n2.split())
    inter = tokens1.intersection(tokens2)
    if len(inter) >= 2:
        return True
    if len(inter) == 1 and SequenceMatcher(None, n1, n2).ratio() > 0.75:
        return True
    return False


def parse_hash_date(val):
    ""
    s = str(val).strip().replace('#', '')
    return pd.to_datetime(s, errors='coerce')


def check_content_match(chat_row, pos_desc_string):
    """Checks if chat brand/product exists in POS text."""
    # Robust fallback for column names
    brand = str(
        chat_row.get('matched_brand', chat_row.get('Matched_Brand', ''))
    ).lower()
    product = str(
        chat_row.get('matched_product', chat_row.get('Matched_Product', ''))
    ).lower()
    
    # Strip punctuation and collapse spaces for exact matching
    pos_text = re.sub(r'[^\w\s]', '', str(pos_desc_string).lower())
    pos_text = re.sub(r'\s+', ' ', pos_text).strip()
    
    if brand not in ['unknown', 'nan', ''] and len(brand) > 3:
        if re.sub(r'[^\w\s]', '', brand) in pos_text:
            return True
            
    if product not in ['unknown', 'nan', '']:
        ignore = ['general', 'inquiry', 'product', 'cream', 'gel', 'wash', 'unknown']
        clean_prod = re.sub(r'[^\w\s]', '', product)
        tokens = [t for t in clean_prod.split() if len(t) > 3 and t not in ignore]
        for t in tokens:
            if t in pos_text:
                return True
                
    return False


# ==========================================
# 3. WATERFALL ENGINE
# ==========================================
def run_attribution_v7():
    print(f"🌊 STARTING WATERFALL V7 ({MAX_HOURS}-Hour Attribution)...")

    # 1. LOAD DATA
    print("   📥 Loading Data...")
    df_chat = pd.read_csv(CHAT_DATA_PATH)
    df_pos_raw = pd.read_csv(POS_DATA_PATH, low_memory=False)

    # 2. PREP CHAT
    df_chat = df_chat[
        (df_chat['is_converted'] == 1) | (df_chat['mpesa_amount'] > 0)
    ].copy()
    
    df_chat['session_start'] = pd.to_datetime(
        df_chat['session_start'], errors='coerce')
    df_chat['norm_phone'] = df_chat['phone_number'].apply(normalize_phone)
    df_chat['mpesa_amount'] = pd.to_numeric(
        df_chat['mpesa_amount'], errors='coerce'
    ).fillna(0)
    
    print(f"   🔹 Chat Pool: {len(df_chat)} converted sessions.")

    # 3. PREP POS
    print("   📅 Parsing POS Dates...")
    df_pos_raw['POS_DateTime'] = df_pos_raw['Date Sold'].apply(parse_hash_date)
    
    if df_pos_raw['POS_DateTime'].notna().sum() == 0:
        print("   ❌ CRITICAL ERROR: Dates failed to parse. Check format.")
        return

    df_pos_raw['norm_phone'] = df_pos_raw['Phone Number'].apply(normalize_phone)
    
    for col in ['Total (Tax Ex)', 'Amount']:
        if col in df_pos_raw.columns:
            df_pos_raw[col] = pd.to_numeric(
                df_pos_raw[col], errors='coerce'
            ).fillna(0)
        else:
            df_pos_raw[col] = 0.0

    df_pos_text = df_pos_raw.groupby('Transaction ID')['Description'].apply(
        lambda x: " | ".join([str(s) for s in x])
    ).reset_index()
    df_pos_text.rename(columns={'Description': 'Full_Basket_Desc'}, inplace=True)

    df_pos_headers = df_pos_raw.groupby('Transaction ID', as_index=False).agg({
        'POS_DateTime': 'min',
        'norm_phone': 'first',
        'Client Name': 'first',
        'Total (Tax Ex)': 'sum',   
        'Amount': 'max'            
    })
    
    df_pos_headers['Final_Amount_Inc'] = np.where(
        df_pos_headers['Amount'] > 0, 
        df_pos_headers['Amount'], 
        df_pos_headers['Total (Tax Ex)'] 
    )

    df_pos_headers = pd.merge(
        df_pos_headers, df_pos_text, on='Transaction ID', how='left'
    )
    print(f"   🔹 POS Pool: {len(df_pos_headers)} unique transactions.")

    # --- POOLS ---
    unmatched_chat = df_chat.copy()
    unmatched_pos = df_pos_headers.copy()
    results = []

    def log_match(layer, matched_df):
        nonlocal unmatched_chat, unmatched_pos
        print(f"      ✅ {layer}: {len(matched_df)} matches.")
        if matched_df.empty:
            return
            
        results.append(matched_df)
        unmatched_chat = unmatched_chat[
            ~unmatched_chat['session_id'].isin(matched_df['session_id'])
        ]
        unmatched_pos = unmatched_pos[
            ~unmatched_pos['Transaction ID'].isin(matched_df['Transaction ID'])
        ]

    # =========================================================
    # 🥇 LAYER 1: GOLD (Phone)
    # =========================================================
    print(f"   🥇 Layer 1: Gold (Phone Match within {MAX_HOURS}h)...")
    m1 = pd.merge(
        unmatched_chat.dropna(subset=['norm_phone']),
        unmatched_pos.dropna(subset=['norm_phone']),
        on='norm_phone',
        how='inner'
    )
    
    if not m1.empty:
        m1['diff_hours'] = (
            m1['POS_DateTime'] - m1['session_start']
        ).dt.total_seconds() / 3600
        m1 = m1[(m1['diff_hours'] > MIN_HOURS) & (m1['diff_hours'] <= MAX_HOURS)] 
        
        m1 = m1.sort_values('diff_hours')
        m1 = m1.drop_duplicates('Transaction ID', keep='first')
        m1 = m1.drop_duplicates('session_id', keep='first')
        
        if not m1.empty:
            m1['match_type'] = 'Gold (Phone)'
            m1['confidence'] = 95
            log_match("Gold", m1)

    # =========================================================
    # 🥈 LAYER 2: SILVER (Name + Price)
    # =========================================================
    print(f"   🥈 Layer 2: Silver (Identity Match within {MAX_HOURS}h)...")
    # unmatched_chat['amt_key'] = unmatched_chat['mpesa_amount'].round(0).astype(int)
    # unmatched_pos['amt_key'] = unmatched_pos['Final_Amount_Inc'].round(0).astype(int)
    BAND = 50  # or 100
    unmatched_chat['amt_key'] = (unmatched_chat['mpesa_amount'] / BAND).round().astype(int)
    unmatched_pos['amt_key']  = (unmatched_pos['Final_Amount_Inc'] / BAND).round().astype(int)
    
    # subset_chat = unmatched_chat[unmatched_chat['amt_key'] > 500]
    subset_chat = unmatched_chat[unmatched_chat['mpesa_amount'] > 500]
    
    m2 = pd.merge(
        subset_chat, unmatched_pos, on='amt_key', 
        how='inner', suffixes=('_chat', '_pos')
    )
    
    if not m2.empty:
        m2['diff_hours'] = (
            m2['POS_DateTime'] - m2['session_start']
        ).dt.total_seconds() / 3600
        m2 = m2[(m2['diff_hours'] > MIN_HOURS) & (m2['diff_hours'] <= MAX_HOURS)]
        
        m2['name_match'] = m2.apply(
            lambda x: check_name_similarity(x['contact_name'], x['Client Name']), 
            axis=1
        )
        m2 = m2[m2['name_match']]
        
        m2 = m2.sort_values('diff_hours')
        m2 = m2.drop_duplicates('Transaction ID', keep='first')
        m2 = m2.drop_duplicates('session_id', keep='first')
        
        if not m2.empty:
            m2['match_type'] = 'Silver (Name+Price)'
            m2['confidence'] = 85
            log_match("Silver", m2)

    # =========================================================
    # 🥉 LAYER 3: BRONZE (Price + Context)
    # =========================================================
    print(f"   🥉 Layer 3: Bronze (Context Match within {MAX_HOURS}h)...")
    subset_chat = unmatched_chat[unmatched_chat['amt_key'] > 500]
    
    m3 = pd.merge(
        subset_chat, unmatched_pos, on='amt_key', 
        how='inner', suffixes=('_chat', '_pos')
    )
    
    if not m3.empty:
        m3['diff_hours'] = (
            m3['POS_DateTime'] - m3['session_start']
        ).dt.total_seconds() / 3600
        m3 = m3[(m3['diff_hours'] > MIN_HOURS) & (m3['diff_hours'] <= MAX_HOURS)]
        
        m3['content_match'] = m3.apply(
            lambda x: check_content_match(x, x['Full_Basket_Desc']), axis=1
        )
        m3 = m3[m3['content_match']]
        
        m3 = m3.sort_values('diff_hours')
        m3 = m3.drop_duplicates('Transaction ID', keep='first')
        m3 = m3.drop_duplicates('session_id', keep='first')
        
        if not m3.empty:
            m3['match_type'] = 'Bronze (Price+Content)'
            m3['confidence'] = 75
            log_match("Bronze", m3)

    # ==========================================
    # 4. FINAL EXPORT
    # ==========================================
    if results:
        final_links = pd.concat(results, ignore_index=True)
        link_cols = [
            'session_id', 'Transaction ID', 'match_type', 'confidence', 'diff_hours'
        ]
        df_links = final_links[link_cols]
        
        final_df = pd.merge(df_links, df_pos_raw, on='Transaction ID', how='left')
        
        chat_cols = [
            'session_id', 'contact_name', 'session_start', 'mpesa_amount', 
            'active_staff', 'primary_category', 'final_tags'
        ]
        chat_ref = df_chat[chat_cols].drop_duplicates('session_id')
        final_df = pd.merge(final_df, chat_ref, on='session_id', how='left')
        
        final_df.to_csv(OUTPUT_FILE, index=False)
        print("\n" + "=" * 50)
        print(f"🚀 DONE. Total Linked: {len(df_links)}")
        print(f"📂 Output: {OUTPUT_FILE}")
        print(df_links['match_type'].value_counts())
    else:
        print("\n❌ NO MATCHES.")


if __name__ == "__main__":
    run_attribution_v7()