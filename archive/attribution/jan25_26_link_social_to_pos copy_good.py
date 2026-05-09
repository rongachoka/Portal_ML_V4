import pandas as pd
import numpy as np
import re
from difflib import SequenceMatcher
from pathlib import Path

# ✅ IMPORT PATHS
from Portal_ML_V4.src.config.settings import PROCESSED_DATA_DIR

# ==========================================
# 1. CONFIGURATION
# ==========================================
CHAT_DATA_PATH = PROCESSED_DATA_DIR / "fact_sessions_enriched.csv"
POS_DATA_PATH = PROCESSED_DATA_DIR / "pos_data" / "all_locations_sales_Jan25-Jan26.csv"
OUTPUT_FILE = PROCESSED_DATA_DIR / "sales_attribution" / "attributed_sales_waterfall_v7.csv"

# ==========================================
# 2. HELPERS
# ==========================================

def normalize_phone(val):
    """Strict 9-digit normalization."""
    if pd.isna(val) or val == 'nan': return None
    s = str(val).replace('.0', '').strip()
    if 'E' in s or 'e' in s:
        try: s = "{:.0f}".format(float(s))
        except: pass
    digits = re.sub(r'[^\d]', '', s)
    if len(digits) >= 9: return digits[-9:]
    return None

def normalize_name(text):
    if pd.isna(text): return ""
    return re.sub(r'[^a-zA-Z\s]', '', str(text).lower()).strip()

def check_name_similarity(name1, name2):
    n1 = normalize_name(name1)
    n2 = normalize_name(name2)
    if len(n1) < 3 or len(n2) < 3: return False
    blacklist = ["unknown", "customer", "guest", "cash sale", "walking client"]
    if any(b in n1 for b in blacklist) or any(b in n2 for b in blacklist): return False
    if n1 in n2 or n2 in n1: return True
    return SequenceMatcher(None, n1, n2).ratio() > 0.85

def parse_hash_date(val):
    """
    Parses format: #2025-11-01 09:28:42#
    """
    s = str(val).strip()
    # Remove hashes
    clean = s.replace('#', '')
    try:
        return pd.to_datetime(clean)
    except:
        return pd.NaT

def check_content_match(chat_row, pos_desc_string):
    brand = str(chat_row.get('matched_brand', '')).lower()
    product = str(chat_row.get('matched_product', '')).lower()
    pos_text = str(pos_desc_string).lower()
    
    if brand not in ['unknown', 'nan', ''] and len(brand) > 3:
        if brand in pos_text: return True
        
    if product not in ['unknown', 'nan', '']:
        ignore = ['general', 'inquiry', 'product', 'cream', 'gel', 'wash', 'lotion', 'unknown']
        tokens = [t for t in product.split() if len(t) > 3 and t not in ignore]
        for t in tokens:
            if t in pos_text: return True
    return False

# ==========================================
# 3. WATERFALL ENGINE
# ==========================================
def run_attribution_v7():
    print("🌊 STARTING WATERFALL V7 (Hash Date Fix)...")

    # 1. LOAD DATA
    print("   📥 Loading Data...")
    df_chat = pd.read_csv(CHAT_DATA_PATH)
    df_pos_raw = pd.read_csv(POS_DATA_PATH, low_memory=False)

    # 2. PREP CHAT
    df_chat = df_chat[(df_chat['is_converted'] == 1) | (df_chat['mpesa_amount'] > 0)].copy()
    df_chat['session_start'] = pd.to_datetime(df_chat['session_start'])
    df_chat['norm_phone'] = df_chat['phone_number'].apply(normalize_phone)
    df_chat['mpesa_amount'] = pd.to_numeric(df_chat['mpesa_amount'], errors='coerce').fillna(0)
    
    print(f"   🔹 Chat Pool: {len(df_chat)} converted sessions.")

    # 3. PREP POS (Using Custom Hash Parser)
    print("   📅 Parsing POS Dates (Format: #YYYY-MM-DD HH:MM:SS#)...")
    df_pos_raw['POS_DateTime'] = df_pos_raw['Date Sold'].apply(parse_hash_date)
    
    # Validation Print
    print(f"   🔍 Check First 5 Dates:")
    print(df_pos_raw[['Date Sold', 'POS_DateTime']].head())
    
    valid_dates = df_pos_raw['POS_DateTime'].notna().sum()
    if valid_dates == 0:
        print("   ❌ CRITICAL ERROR: Dates failed to parse. Check format.")
        return

    df_pos_raw['norm_phone'] = df_pos_raw['Phone Number'].apply(normalize_phone)
    
    # Calc Totals
    cols_to_num = ['Total (Tax Ex)', 'Amount']
    for c in cols_to_num:
        if c in df_pos_raw.columns:
            df_pos_raw[c] = pd.to_numeric(df_pos_raw[c], errors='coerce').fillna(0)
        else:
            df_pos_raw[c] = 0.0

    # Aggregate Text for Context Matching
    df_pos_text = df_pos_raw.groupby('Transaction ID')['Description'].apply(lambda x: " | ".join([str(s) for s in x])).reset_index()
    df_pos_text.rename(columns={'Description': 'Full_Basket_Desc'}, inplace=True)

    # Create Header
    df_pos_headers = df_pos_raw.groupby('Transaction ID', as_index=False).agg({
        'POS_DateTime': 'min',
        'norm_phone': 'first',
        'Client Name': 'first',
        'Total (Tax Ex)': 'sum',   
        'Amount': 'max'            
    })
    
    # Smart Amount Logic
    df_pos_headers['Final_Amount_Inc'] = np.where(
        df_pos_headers['Amount'] > 0, 
        df_pos_headers['Amount'], 
        df_pos_headers['Total (Tax Ex)'] 
    )

    df_pos_headers = pd.merge(df_pos_headers, df_pos_text, on='Transaction ID', how='left')
    print(f"   🔹 POS Pool: {len(df_pos_headers)} unique transactions.")

    # --- POOLS ---
    unmatched_chat = df_chat.copy()
    unmatched_pos = df_pos_headers.copy()
    results = []

    def log_match(layer, df):
        print(f"      ✅ {layer}: {len(df)} matches.")
        if not df.empty:
            results.append(df)
            nonlocal unmatched_chat, unmatched_pos
            unmatched_chat = unmatched_chat[~unmatched_chat['session_id'].isin(df['session_id'])]
            unmatched_pos = unmatched_pos[~unmatched_pos['Transaction ID'].isin(df['Transaction ID'])]

    # =========================================================
    # 🥇 LAYER 1: GOLD (Phone) - 24 HOURS
    # =========================================================
    print("   🥇 Layer 1: Gold (Phone)...")
    m1 = pd.merge(
        unmatched_chat.dropna(subset=['norm_phone']),
        unmatched_pos.dropna(subset=['norm_phone']),
        on='norm_phone',
        how='inner'
    )
    
    if not m1.empty:
        # 1. Calc Time Diff (Hours)
        m1['diff_hours'] = (m1['POS_DateTime'] - m1['session_start']).dt.total_seconds() / 3600
        
        # 2. Constraint: Sale must happen AFTER session (allowing small 1h buffer for clock sync issues)
        m1 = m1[m1['diff_hours'] > -1]
        
        # 3. Constraint: Within 24 Hours
        m1 = m1[m1['diff_hours'] <= 24] 
        
        m1 = m1.sort_values('diff_hours').drop_duplicates('session_id')
        
        if not m1.empty:
            m1['match_type'] = 'Gold (Phone)'
            m1['confidence'] = 95
            log_match("Gold", m1)

    # =========================================================
    # 🥈 LAYER 2: SILVER (Name + Price)
    # =========================================================
    print("   🥈 Layer 2: Silver (Identity Match)...")
    unmatched_chat['amt_key'] = unmatched_chat['mpesa_amount'].astype(int)
    unmatched_pos['amt_key'] = unmatched_pos['Final_Amount_Inc'].astype(int)
    
    subset_chat = unmatched_chat[unmatched_chat['amt_key'] > 500]
    
    m2 = pd.merge(subset_chat, unmatched_pos, on='amt_key', how='inner', suffixes=('_chat', '_pos'))
    
    if not m2.empty:
        m2['diff_hours'] = (m2['POS_DateTime'] - m2['session_start']).dt.total_seconds() / 3600
        m2 = m2[(m2['diff_hours'] > -1) & (m2['diff_hours'] <= 48)]
        
        m2['name_match'] = m2.apply(lambda x: check_name_similarity(x['contact_name'], x['Client Name']), axis=1)
        m2 = m2[m2['name_match'] == True]
        
        m2 = m2.sort_values('diff_hours').drop_duplicates('session_id')
        
        if not m2.empty:
            m2['match_type'] = 'Silver (Name+Price)'
            m2['confidence'] = 85
            log_match("Silver", m2)

    # =========================================================
    # 🥉 LAYER 3: BRONZE (Price + Context)
    # =========================================================
    print("   🥉 Layer 3: Bronze (Context Match)...")
    subset_chat = unmatched_chat[unmatched_chat['amt_key'] > 500]
    
    m3 = pd.merge(subset_chat, unmatched_pos, on='amt_key', how='inner', suffixes=('_chat', '_pos'))
    
    if not m3.empty:
        m3['diff_hours'] = (m3['POS_DateTime'] - m3['session_start']).dt.total_seconds() / 3600
        m3 = m3[(m3['diff_hours'] > -1) & (m3['diff_hours'] <= 24)]
        
        m3['content_match'] = m3.apply(lambda x: check_content_match(x, x['Full_Basket_Desc']), axis=1)
        m3 = m3[m3['content_match'] == True]
        
        m3 = m3.sort_values('diff_hours').drop_duplicates('session_id')
        
        if not m3.empty:
            m3['match_type'] = 'Bronze (Price+Content)'
            m3['confidence'] = 75
            log_match("Bronze", m3)


    # ==========================================
    # 4. FINAL EXPORT
    # ==========================================
    if results:
        final_links = pd.concat(results, ignore_index=True)
        link_cols = ['session_id', 'Transaction ID', 'match_type', 'confidence', 'diff_hours']
        df_links = final_links[link_cols]
        
        final_df = pd.merge(df_links, df_pos_raw, on='Transaction ID', how='left')
        
        chat_cols = ['session_id', 'contact_name', 'session_start', 'mpesa_amount', 'active_staff', 'primary_category']
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