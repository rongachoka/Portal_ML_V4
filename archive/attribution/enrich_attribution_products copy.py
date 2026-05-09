import pandas as pd
import numpy as np
import re
from difflib import SequenceMatcher
from pathlib import Path
import os

# ✅ IMPORT PATHS (V3 Settings)
from Portal_ML_V4.src.config.settings import (
    BASE_DIR,
    PROCESSED_DATA_DIR,
)

# ✅ IMPORT CONFIGS
try:
    from Portal_ML_V4.src.config.brands import BRAND_LIST, BRAND_ALIASES
except ImportError:
    BRAND_LIST = []

# ==========================================
# 1. CONFIGURATION
# ==========================================
ATTRIBUTION_PATH = PROCESSED_DATA_DIR / "sales_attribution" / "attributed_sales_waterfall_v7.csv"
SESSIONS_PATH = PROCESSED_DATA_DIR / "fact_sessions_enriched.csv"
KB_PATH = BASE_DIR / "data" / "01_raw" / "Final_Knowledge_Base_PowerBI.csv"
OUTPUT_FILE = PROCESSED_DATA_DIR / "sales_attribution" / "social_sales_Jan25_Jan26.csv"

# 🎛️ TUNING
PRICE_TOLERANCE = 0.20
MATCH_THRESHOLD = 0.55
STOP_WORDS = {'FOR', 'WITH', 'OF', 'TO', 'IN', 'ON', 'AT', 'ML', 'GM', 'PCS', 'TUBE', 'BOTTLE', 'CAPS', 'TABS', 'SYR'}

# 🚨 MANUAL POS TRANSLATIONS
POS_ALIASES = {
    "LRP": "LA ROCHE POSAY",
    "LA ROC": "LA ROCHE POSAY",
    "LA ROCHE": "LA ROCHE POSAY",
    "EFFACLAR": "LA ROCHE POSAY",
    "BOJ": "BEAUTY OF JOSEON",
    "THE ORDI": "THE ORDINARY",
    "ORDINARY": "THE ORDINARY",
    "VB": "VITABIOTICS",
    "MEDIX": "MEDIX",
    "SS": "SEVEN SEAS",
    "PALMERS": "PALMER'S"
}

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def expand_aliases(text):
    """Replaces POS abbreviations with full brand names."""
    if pd.isna(text): return ""
    text = str(text).upper()
    for alias, full in POS_ALIASES.items():
        pattern = r'\b' + re.escape(alias) + r'\b'
        text = re.sub(pattern, full, text)
    return text

def clean_text_for_matching(text):
    if pd.isna(text): return ""
    clean = str(text).upper()
    
    # 1. EXPAND ALIASES (Fixes LRP -> La Roche Posay)
    clean = expand_aliases(clean)
    
    # 2. Standard Clean
    clean = re.sub(r'[^A-Z0-9\s]', ' ', clean)
    tokens = [w for w in clean.split() if w not in STOP_WORDS and len(w) > 2]
    return " ".join(tokens)

def fuzzy_similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

def extract_brands_from_tags(tag_string):
    """Extracts Brand Names from the 'final_tags' column."""
    if pd.isna(tag_string): return []
    tags = str(tag_string).split(' | ')
    found_brands = []
    
    valid_brands_upper = {str(b).upper() for b in BRAND_LIST}
    
    for t in tags:
        clean_t = t.strip()
        if clean_t.upper() in valid_brands_upper:
            found_brands.append(clean_t)
        elif clean_t.upper() in BRAND_ALIASES:
            found_brands.append(BRAND_ALIASES[clean_t.upper()])
            
    return list(set(found_brands))

def detect_brand_from_description(desc):
    """
    Scans the POS Description for known brands if tags are missing.
    """
    if pd.isna(desc): return []
    
    # Expand aliases first so we can find the real brand name
    desc_expanded = expand_aliases(str(desc).upper())
    found = []
    
    # Check strict Brand List
    for brand in BRAND_LIST:
        b_upper = str(brand).upper()
        if re.search(r'\b' + re.escape(b_upper) + r'\b', desc_expanded):
            found.append(brand)
            
    # Check Aliases
    for alias, true_name in BRAND_ALIASES.items():
        if re.search(r'\b' + re.escape(alias) + r'\b', desc_expanded):
            found.append(true_name)
            
    return list(set(found))

def is_valid_phone(val):
    """Checks if the value is a pure integer string."""
    if pd.isna(val): return False
    # Remove .0 for floats
    s = str(val).strip().replace('.0', '')
    # Check if purely digits and reasonable length
    return s.isdigit() and len(s) >= 9

def optimize_memory(df):
    for col in df.columns:
        col_type = df[col].dtype
        if col_type == 'object':
            if df[col].nunique() / len(df) < 0.5:
                df[col] = df[col].astype('category')
        elif col_type == 'float64':
            df[col] = pd.to_numeric(df[col], downcast='float')
        elif col_type == 'int64':
            df[col] = pd.to_numeric(df[col], downcast='integer')
    return df

# ==========================================
# 3. MAIN LOGIC
# ==========================================
def run_smart_enrichment():
    print("🏷️  STARTING SMART ENRICHMENT (V6 - PowerBI Ready)...")
    
    if not ATTRIBUTION_PATH.exists():
        print("❌ Attribution file missing.")
        return

    # 1. LOAD DATA
    df_attr = pd.read_csv(ATTRIBUTION_PATH)
    df_attr = optimize_memory(df_attr)
    
    # 🧹 HEADER CLEANING
    df_attr.columns = df_attr.columns.str.strip()
    
    # 🚨 CHECK FOR ACTIVE_STAFF
    if 'active_staff' not in df_attr.columns:
        print("   ⚠️ Column 'active_staff' missing. Fetching from Sessions...")
        if SESSIONS_PATH.exists():
            df_sess = pd.read_csv(SESSIONS_PATH, usecols=['session_id', 'active_staff'])
            staff_map = df_sess.set_index('session_id')['active_staff'].to_dict()
            df_attr['active_staff'] = df_attr['session_id'].map(staff_map).fillna('Unassigned')
        else:
            df_attr['active_staff'] = "Unassigned"

    # ==================================================
    # 🧹 NUCLEAR PHONE CLEANING (FIX FOR POWERBI)
    # ==================================================
    print("   🧹 Cleaning Invalid Phone Numbers...")
    
    # Check BOTH potential column names
    target_cols = [c for c in ['norm_phone', 'Phone Number'] if c in df_attr.columns]
    
    for p_col in target_cols:
        # 1. Force to String (Object) to avoid Category errors
        df_attr[p_col] = df_attr[p_col].astype(str)
        
        # 2. Identify Invalid Rows (Includes 'LOOP', 'nan', short numbers)
        valid_mask = df_attr[p_col].apply(is_valid_phone)
        
        # 3. Wipe invalid data
        invalid_count = (~valid_mask).sum()
        if invalid_count > 0:
            print(f"      - {p_col}: Wiping {invalid_count} invalid entries (LOOP/Text)")
            df_attr.loc[~valid_mask, p_col] = None 

    # 2. LOAD KNOWLEDGE BASE
    if KB_PATH.exists():
        df_kb = pd.read_csv(KB_PATH)
        df_kb.columns = df_kb.columns.str.strip()
        kb_cols = {c.lower(): c for c in df_kb.columns}
        
        col_map = {
            'name': kb_cols.get('name', kb_cols.get('item description', 'Name')),
            'brand': kb_cols.get('brand', 'Brand'),
            'price': kb_cols.get('price', 'Price'),
            'category': kb_cols.get('canonical_category', kb_cols.get('canonical category', 'Canonical_Category')),
            'subcategory': kb_cols.get('sub_category', kb_cols.get('sub category', 'Sub_Category')),
            'concerns': kb_cols.get('concerns', 'Concerns'),
            'audience': kb_cols.get('target_audience', kb_cols.get('target audience', 'Target_Audience'))
        }
        
        df_kb['Clean_Desc'] = df_kb[col_map['name']].apply(clean_text_for_matching)
        df_kb['Price'] = pd.to_numeric(df_kb[col_map['price']], errors='coerce').fillna(0)
        
        kb_by_brand = {}
        if col_map['brand'] in df_kb.columns:
            for brand, grp in df_kb.groupby(col_map['brand']):
                kb_by_brand[str(brand).upper()] = grp.to_dict('records')
        else:
            print("❌ KB missing 'Brand' column.")
            return
    else:
        print("❌ KB Not Found.")
        return

    # Results Containers
    matched_data = {
        'Matched_Product': [],
        'Matched_Brand': [],
        'Matched_Category': [],
        'Matched_Sub_Category': [],
        'Matched_Concern': [],
        'Matched_Audience': []
    }

    print("   🔍 Matching POS Descriptions to KB...")

    for idx, row in df_attr.iterrows():
        pos_desc = str(row.get('Description', ''))
        pos_price = float(row.get('Total (Tax Ex)', 0))
        qty = float(row.get('Qty Sold', 1))
        unit_price = pos_price / qty if qty > 0 else pos_price
        
        # 1. TRY CHAT TAGS FIRST
        detected_brands = extract_brands_from_tags(row.get('final_tags', ''))
        
        # 2. BRAND RESCUE
        if not detected_brands:
            detected_brands = detect_brand_from_description(pos_desc)

        # Default Logic
        res = {
            'name': f"{pos_desc}",
            'brand': "Unknown",
            'cat': "General",
            'sub': "General",
            'concern': "General",
            'audience': "General"
        }
        
        if detected_brands:
            best_score = 0
            best_candidate = None
            pos_clean = clean_text_for_matching(pos_desc)
            
            for brand in detected_brands:
                candidates = kb_by_brand.get(str(brand).upper(), [])
                
                for cand in candidates:
                    kb_clean = cand['Clean_Desc']
                    score = fuzzy_similarity(pos_clean, kb_clean)
                    
                    kb_price = cand['Price']
                    if kb_price > 0:
                        diff = abs(unit_price - kb_price) / kb_price
                        if diff > PRICE_TOLERANCE:
                            score -= 0.2
                            
                    if score > best_score:
                        best_score = score
                        best_candidate = cand
                        best_brand = brand
            
            if best_candidate and best_score > MATCH_THRESHOLD:
                # Exact Match
                res['name'] = best_candidate.get(col_map['name'], "Unknown Product")
                res['brand'] = best_candidate.get(col_map['brand'], best_brand)
                res['cat'] = best_candidate.get(col_map['category'], "General")
                res['sub'] = best_candidate.get(col_map['subcategory'], "General")
                res['concern'] = best_candidate.get(col_map['concerns'], "General")
                res['audience'] = best_candidate.get(col_map['audience'], "General")
            else:
                # Brand Known, Product Unknown
                final_brand = POS_ALIASES.get(detected_brands[0], detected_brands[0])
                res['brand'] = final_brand
                res['name'] = f"General {final_brand} Product"
        
        # Delivery Fee Override
        if "DELIVERY" in pos_desc.upper():
            res['brand'] = "Logistics"
            res['name'] = "Delivery Fee"
            res['cat'] = "Service"

        matched_data['Matched_Product'].append(res['name'])
        matched_data['Matched_Brand'].append(res['brand'])
        matched_data['Matched_Category'].append(res['cat'])
        matched_data['Matched_Sub_Category'].append(res['sub'])
        matched_data['Matched_Concern'].append(res['concern'])
        matched_data['Matched_Audience'].append(res['audience'])

    # 3. APPEND RESULTS
    for col, values in matched_data.items():
        df_attr[col] = values

    # 4. SAVE FINAL FILE
    print(f"   💾 Saving Enriched Data to: {OUTPUT_FILE}")
    os.makedirs(OUTPUT_FILE.parent, exist_ok=True)
    df_attr.to_csv(OUTPUT_FILE, index=False)
    print("   ✅ Done!")

if __name__ == "__main__":
    run_smart_enrichment()