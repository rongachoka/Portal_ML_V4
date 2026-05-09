"""
enrich_attribution_products.py
===============================
Labels brands and categories on attributed social media sales by fuzzy-matching
POS descriptions against the Knowledge Base.

Reads the attribution waterfall output and, for each line item, attempts to
match the POS description to a KB product using brand detection + fuzzy
token similarity (SequenceMatcher). Outputs an enriched file with
Matched_Brand, Matched_Product, and Canonical_Category columns.

Inputs:
    data/03_processed/sales_attribution/attributed_sales_waterfall_v7.csv
    data/03_processed/fact_sessions_enriched.csv
    data/01_raw/Final_Knowledge_Base_PowerBI.csv

Output:
    data/03_processed/sales_attribution/social_sales_Jan25_Jan26.csv

Entry point: run_smart_enrichment()
"""

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
    KB_PATH
)

# ✅ IMPORT CONFIGS
# We only import aliases now, NOT the static brand list
try:
    from Portal_ML_V4.src.config.brands import BRAND_ALIASES
except ImportError:
    BRAND_ALIASES = {}
    

# Phone Cleaning Logic
from Portal_ML_V4.src.utils.phone import is_valid_phone


# ==========================================
# 0. DYNAMIC KNOWLEDGE BASE & CONFIG
# ==========================================
ATTRIBUTION_PATH = PROCESSED_DATA_DIR / "sales_attribution" / "attributed_sales_waterfall_v7.csv"
SESSIONS_PATH = PROCESSED_DATA_DIR / "fact_sessions_enriched.csv"
OUTPUT_FILE = PROCESSED_DATA_DIR / "sales_attribution" / "social_sales_Jan25_Jan26.csv"

# 🎛️ TUNING
PRICE_TOLERANCE = 0.20
MATCH_THRESHOLD = 0.50
STOP_WORDS = {
    'FOR', 'WITH', 'OF', 'TO', 'IN', 'ON', 'AT', 'ML', 'GM', 'PCS', 
    'TUBE', 'BOTTLE', 'CAPS', 'TABS', 'SYR'
}


# 🚨 DYNAMIC BRAND LOADER
# This replaces the old static BRAND_LIST
try:
    kb_temp_df = pd.read_csv(KB_PATH)
    kb_temp_df = kb_temp_df.dropna(subset=['Brand'])
    DYNAMIC_BRAND_LIST = sorted(
        kb_temp_df['Brand'].astype(str).str.strip().str.title().unique().tolist(),
        key=len, 
        reverse=True
    )
    print(f"✅ Loaded {len(DYNAMIC_BRAND_LIST)} brands dynamically from KB.")
except Exception as e:
    print(f"⚠️ Warning: Could not load dynamic brands: {e}")
    DYNAMIC_BRAND_LIST = []


try: 
    from Portal_ML_V4.src.config.pos_aliases import TERM_ALIASES
except ImportError:
    TERM_ALIASES = {}

# 🚨 ALIAS SWAP
# We use the config aliases to avoid the "SS" -> "Seven Seas" bug
POS_ALIASES = BRAND_ALIASES

# TERM_ALIASES = {
#     "CRM": "CREAM",
#     "SUSP": "SUSPENSION",
#     "CAPS": "CAPSULES",
#     "CAP": "CAPSULE",
#     "TABS": "TABLETS",
#     "TAB": "TABLET",
#     "SYR": "SYRUP",
#     "SOLN": "SOLUTION",
#     "SOL": "SOLUTION",
#     "SQUAL" : "SQUALANE",
#     "CLEANS" : "CLEANSER",
#     "INJ": "INJECTION",
#     "OINT": "OINTMENT",
#     "LOT": "LOTION",
#     "QTY": "QUANTITY",
#     "X1": "1PC",
#     "X2": "2PCS",
#     "MOIST": "MOISTURIZING",
#     "V.DRY": "VERY DRY",
#     "MOISTUR" : "MOISTURIZING",
#     "HYDRAT"  : "HYDRATING",
#     "CLEANSR" : "CLEANSER",
#     "EFFACL"  : "EFFACLAR",
#     "ANTIHLS" : "ANTHELIOS",
#     "CICAPLS" : "CICAPLAST",
#     "NIACIN"  : "NIACINAMIDE",
#     "RETINL"  : "RETINOL",
    
# }

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

# def expand_aliases(text):
#     """Replaces POS abbreviations with full brand names."""
#     if pd.isna(text): return ""
#     text = str(text).upper()
    
#     # We use Regex Word Boundaries (\b) to prevent partial matches
#     for alias, full in POS_ALIASES.items():
#         pattern = r'\b' + re.escape(alias.upper()) + r'\b'
#         text = re.sub(pattern, full.upper(), text)
#     return text

def expand_aliases(text):
    """Replaces POS abbreviations (Brands AND Terms) with full names."""
    if pd.isna(text): return ""
    text = str(text).upper()
    
    # 1. Merge both dictionaries
    # We put TERM_ALIASES first, but order doesn't matter much with Word Boundaries
    all_aliases = {**TERM_ALIASES, **POS_ALIASES}
    
    # 2. Run Replacements using Word Boundaries (\b)
    for alias, full in all_aliases.items():
        # \b matches "CRM" but ignores "MCCRM" or "CRMS"
        pattern = r'\b' + re.escape(alias.upper()) + r'\b'
        text = re.sub(pattern, full.upper(), text)
        
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


def fuzzy_similarity(a, b, brand=None):
    """Hybrid: token overlap + sequence match. Better for abbreviated POS descriptions."""
    if not a or not b:
        return 0.0

    # 🟢 Strip brand name from POS text before comparing
    # Prevents "SPORT SUPPLIES MAGNESIUM GLYCINATE" scoring low against "MAGNESIUM GLYCINATE"
    if brand:
        brand_clean = re.sub(r'[^A-Z0-9\s]', ' ', str(brand).upper()).strip()
        a = re.sub(r'\b' + re.escape(brand_clean) + r'\b', '', a).strip()
        a = re.sub(r'\s+', ' ', a).strip()

    # 1. Sequence match
    seq_score = SequenceMatcher(None, a, b).ratio()
    
    # 2. Token overlap score
    tokens_a = set(a.split())
    tokens_b = set(b.split())
    
    size_tokens = {'8OZ', '16OZ', '250ML', '500ML', '100ML', '200ML', '30ML', 
                   '50ML', '150ML', '400ML', '1L', '2L', '300ML', '10OZ'}
    tokens_a -= size_tokens
    tokens_b -= size_tokens
    
    if not tokens_a or not tokens_b:
        return seq_score
    
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    jaccard = len(intersection) / len(union)
    
    shorter = tokens_a if len(tokens_a) <= len(tokens_b) else tokens_b
    longer = tokens_b if len(tokens_a) <= len(tokens_b) else tokens_a
    coverage = len(shorter & longer) / len(shorter) if shorter else 0
    
    token_score = (jaccard * 0.4) + (coverage * 0.6)
    final_score = (seq_score * 0.35) + (token_score * 0.65)
    
    return final_score

def extract_brands_from_tags(tag_string):
    """Extracts Brand Names ignoring punctuation and spaces."""
    if pd.isna(tag_string): return []
    tags = str(tag_string).split('|')
    found_brands = []
    
    # Map purely alphanumeric names to their official KB names (LaRochePosay -> La Roche-Posay)
    safe_brand_map = {re.sub(r'[^A-Z0-9]', '', str(b).upper()): b for b in DYNAMIC_BRAND_LIST}
    
    for t in tags:
        clean_t = re.sub(r'[^A-Z0-9]', '', str(t).upper())
        matched = False
        for safe_key, brand_name in safe_brand_map.items():
            if safe_key in clean_t:  # ← substring check, finds "LAROCHEPOSAY" inside "LAROCHEPOSAYSKINCARE"
                found_brands.append(brand_name)
                matched = True
                break
        if not matched:
            for alias, true_name in BRAND_ALIASES.items():
                if clean_t == re.sub(r'[^A-Z0-9]', '', str(alias).upper()):
                    found_brands.append(true_name)
                    break
                    
    return list(set(found_brands))
            

def detect_brand_from_description(desc):
    """
    Scans the POS Description for known brands.
    Now bulletproof against missing hyphens and "THE " prefixes!
    """
    if pd.isna(desc): return []
    
    # Strip ALL punctuation from POS description so "LA ROCHE POSAY" matches easily
    safe_desc = re.sub(r'[^A-Z0-9\s]', ' ', str(desc).upper())
    safe_desc = re.sub(r'\s+', ' ', safe_desc).strip()
    
    found = []
    
    # 1. Check strict Dynamic Brand List
    for brand in DYNAMIC_BRAND_LIST:
        # Strip punctuation from the KB brand name too
        b_clean = re.sub(r'[^A-Z0-9\s]', ' ', str(brand).upper())
        b_clean = re.sub(r'\s+', ' ', b_clean).strip()
        
        # Strip "THE " prefix (e.g. "THE ORDINARY" -> "ORDINARY")
        b_no_the = re.sub(r'^THE\s+', '', b_clean)
        
        # Word boundary check on the safe description
        if re.search(r'\b' + re.escape(b_clean) + r'\b', safe_desc):
            found.append(brand)
        elif b_no_the != b_clean and re.search(r'\b' + re.escape(b_no_the) + r'\b', safe_desc):
            found.append(brand)
            
    # 2. Check Aliases (LRP, SS, etc.)
    for alias, true_name in BRAND_ALIASES.items():
        alias_clean = re.sub(r'[^A-Z0-9\s]', ' ', str(alias).upper())
        alias_clean = re.sub(r'\s+', ' ', alias_clean).strip()
        if re.search(r'\b' + re.escape(alias_clean) + r'\b', safe_desc):
            found.append(true_name)
            
    return list(set(found))

# def is_valid_phone(val):
#     """Checks if the value is a pure integer string."""
#     if pd.isna(val): return False
#     # Remove .0 for floats
#     s = str(val).strip().replace('.0', '')
#     # Check if purely digits and reasonable length
#     return s.isdigit() and len(s) >= 9

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
    print("🏷️  STARTING SMART ENRICHMENT (V6 - Dynamic KB + PowerBI Fixes)...")
    
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
            # Create a dictionary mapping session_id -> active_staff
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
        'target_audience': [] # Renamed from 'Matched_Audience' to match V3 standard
    }

    print("   🔍 Matching POS Descriptions to KB...")

    for idx, row in df_attr.iterrows():
        pos_desc = str(row.get('Description', ''))
        pos_price = pd.to_numeric(row.get('Total (Tax Ex)', 0), errors='coerce')
        qty = pd.to_numeric(row.get('Qty Sold', 1), errors='coerce')

        pos_price = 0 if pd.isna(pos_price) else float(pos_price)
        qty = 1 if (pd.isna(qty) or qty == 0) else float(qty)
        unit_price = pos_price / qty if qty > 0 else pos_price
        
        # 1. TRY CHAT TAGS FIRST
        detected_brands = extract_brands_from_tags(row.get('final_tags', ''))

        if detected_brands:
            normalized = []
            for b in detected_brands:
                b_lower = str(b).lower().strip()
                # Check if this brand has a canonical form in BRAND_ALIASES
                canonical = BRAND_ALIASES.get(b_lower, b)
                normalized.append(canonical)
            detected_brands = list(set(normalized))
        
        # 2. BRAND RESCUE
        if not detected_brands:
            detected_brands = detect_brand_from_description(pos_desc)

            if detected_brands:
                normalized = []
                for b in detected_brands:
                    b_lower = str(b).lower().strip()
                    # Check if this brand has a canonical form in BRAND_ALIASES
                    canonical = BRAND_ALIASES.get(b_lower, b)
                    normalized.append(canonical)
                detected_brands = list(set(normalized))

        # Default Logic
        res = {
            'name': f"{pos_desc}",
            'brand': "Unknown",
            'cat': "General",
            'sub': "General",
            'concern': "General",
            'audience': "General"
        }
        
        best_brand = "Unknown"
        
        if detected_brands:
            best_score = 0
            best_candidate = None
            pos_clean = clean_text_for_matching(pos_desc)
            
            for brand in detected_brands:
                candidates = kb_by_brand.get(str(brand).upper(), [])
                
                for cand in candidates:
                    kb_clean = cand['Clean_Desc']
                    score = fuzzy_similarity(pos_clean, kb_clean, brand=brand)
                    
                    kb_price = cand['Price']
                    if kb_price > 0:
                        diff = abs(unit_price - kb_price) / kb_price
                        if diff > PRICE_TOLERANCE:
                            score *= 0.85
                            
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
                # We use the detected brand, cleaning up any aliases
                # final_brand = POS_ALIASES.get(detected_brands[0].upper(), detected_brands[0])
                # res['brand'] = final_brand
                # res['name'] = f"General {final_brand} Product"
                res['brand'] = str(detected_brands[0]).title()
                res['name'] = f"General {res['brand']} Product"
        
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
        matched_data['target_audience'].append(res['audience'])

    # 3. APPEND RESULTS
    for col, values in matched_data.items():
        df_attr[col] = values

    # 3.5 FIX - Dedup & Fitering to only relevant brands
    # A. Dedup exact duplicate line items
    # dedup_cols = ['Transaction ID', 'Description', 'Qty Sold', 'Total (Tax Ex)',
    #               'On Hand']
    # before_dedup = len(df_attr)
    # df_attr = df_attr.drop_duplicates(subset=dedup_cols)
    dedup_cols = ['Transaction ID', 'Description', 'Qty Sold', 'Total (Tax Ex)', 'On Hand']
    actual_dedup_cols = [c for c in dedup_cols if c in df_attr.columns]
    if len(actual_dedup_cols) < len(dedup_cols):
        missing = [c for c in dedup_cols if c not in df_attr.columns]
        print(f"   ⚠️ Dedup cols missing, skipping: {missing}")

    before_dedup = len(df_attr)
    df_attr = df_attr.drop_duplicates(subset=actual_dedup_cols)

    print(f"   ✂️  Dedup: {before_dedup - len(df_attr):,} duplicate line items removed")

    # B. Filter to brand-relevant line items only
    print("   🔍 Filtering to brand-relevant line items only...")

    def is_line_item_relevant(row):
        brand = str(row.get('Matched_Brand', '')).strip()
        desc = str(row.get('Description', '')).upper()
        matched_product = str(row.get('Matched_Product', '')).strip()

        # Always keep delivery fees
        if brand == 'Logistics':
            return True

        # Drop unknown brands
        if brand in ['Unknown', 'General', '']:
            return False

        # 🟢 KEY CHANGE: If we got a real KB match (not a fallback),
        # keep it regardless of whether brand appears in description
        # This preserves supplements, vitamins etc. where brand isn't in POS desc
        if not matched_product.startswith('General ') and matched_product not in ['', 'Unknown']:
            return True

        # For "General X Product" fallbacks, require brand in description
        # to avoid wrongly attributing unrelated basket items
        brand_clean = re.sub(r'[^A-Z0-9\s]', ' ', brand.upper()).strip()
        brand_no_the = re.sub(r'^THE\s+', '', brand_clean)

        if re.search(r'\b' + re.escape(brand_clean) + r'\b', desc):
            return True
        if brand_no_the != brand_clean and re.search(r'\b' + re.escape(brand_no_the) + r'\b', desc):
            return True

        # Check aliases
        for alias, canonical in BRAND_ALIASES.items():
            if canonical == brand:
                alias_clean = re.sub(r'[^A-Z0-9\s]', ' ', alias.upper()).strip()
                if re.search(r'\b' + re.escape(alias_clean) + r'\b', desc):
                    return True

        return False

    # def is_line_item_relevant(row):
    #     brand = str(row.get('Matched_Brand', '')).strip()
    #     desc = str(row.get('Description', '')).upper()

    #     # Always keep delivery fees
    #     if brand == 'Logistics':
    #         return True

    #     # Drop unknown brands
    #     if brand in ['Unknown', 'General', '']:
    #         return False

    #     # Drop "General X Product" fallbacks — no confirmed KB match
    #     if str(row.get('Matched_Product', '')).startswith('General '):
    #         return False

    #     # Check if brand name appears in POS description
    #     brand_clean = re.sub(r'[^A-Z0-9\s]', ' ', brand.upper()).strip()
    #     brand_no_the = re.sub(r'^THE\s+', '', brand_clean)

    #     if re.search(r'\b' + re.escape(brand_clean) + r'\b', desc):
    #         return True
    #     if brand_no_the != brand_clean and re.search(r'\b' + re.escape(brand_no_the) + r'\b', desc):
    #         return True

    #     # Check aliases
    #     for alias, canonical in BRAND_ALIASES.items():
    #         if canonical == brand:
    #             alias_clean = re.sub(r'[^A-Z0-9\s]', ' ', alias.upper()).strip()
    #             if re.search(r'\b' + re.escape(alias_clean) + r'\b', desc):
    #                 return True

    #     return False

    before_filter = len(df_attr)
    df_attr = df_attr[df_attr.apply(is_line_item_relevant, axis=1)].copy()
    print(f"   ✂️  Filter: {before_filter - len(df_attr):,} non-brand line items removed")
    print(f"   ✅ Remaining: {len(df_attr):,} brand-matched line items")


    # 4. SAVE FINAL FILE
    print(f"   💾 Saving Enriched Data to: {OUTPUT_FILE}")
    os.makedirs(OUTPUT_FILE.parent, exist_ok=True)
    df_attr.to_csv(OUTPUT_FILE, index=False)
    print("   ✅ Done!")

if __name__ == "__main__":
    run_smart_enrichment()