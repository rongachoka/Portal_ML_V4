import pandas as pd
import numpy as np
import re
from pathlib import Path

# ✅ IMPORT PATHS
from Portal_ML_V4.src.config.settings import (
    BASE_DIR,
    PROCESSED_DATA_DIR,
)

# ✅ IMPORT CONFIGS
try:
    from Portal_ML_V4.src.config.brands import BRAND_LIST, BRAND_ALIASES
    from Portal_ML_V4.src.config.department_map import DEPARTMENT_TO_CANONICAL
except ImportError:
    BRAND_LIST = []
    BRAND_ALIASES = {}
    DEPARTMENT_TO_CANONICAL = {}

# ==========================================
# 1. CONFIGURATION
# ==========================================
# 🚨 INPUT: The file from the Linker (which you currently have)
ATTRIBUTION_PATH = PROCESSED_DATA_DIR / "sales_attribution" / "updated_fact_social_sales_attribution.csv"
KB_PATH = BASE_DIR / "data" / "01_raw" / "Final_Knowledge_Base_PowerBI.csv"

# 🚨 OUTPUT: The Final File with Matched_Brand
OUTPUT_FILE = PROCESSED_DATA_DIR / "sales_attribution" / "social_sales_Jan25_Jan26.csv"

# Thresholds
MIN_TOKEN_MATCH_RATIO = 0.60
PRICE_TOLERANCE = 0.25 

STOP_WORDS = {
    'AND', 'FOR', 'WITH', 'OF', 'TO', 'IN', 'ON', 'AT', 'MY', 'A', 'IS',
    'ML', 'GM', 'KG', 'PCS', 'OZ', 'G', 'PACK', 'BOX', 'BOTTLE', 'TUBE',
    'S', 'X', 'NO'
}

ABBREVIATION_MAP = {
    "ONT": "OINTMENT", "SOLN": "SOLUTION", "TAB": "TABLETS", "TABS": "TABLETS",
    "CAPS": "CAPSULES", "FL": "FLUID", "MOIST": "MOISTURIZING", "CRM": "CREAM",
    "LOT": "LOTION", "EZCEMA": "ECZEMA"
}

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def clean_id(val):
    if pd.isna(val): return None
    s = str(val).strip().replace('.0', '')
    s = ''.join(filter(str.isdigit, s)) 
    if len(s) == 0: return None
    if s.startswith('254'): return f"+{s}"
    return s

def clean_id_excel_safe(val):
    if pd.isna(val): return None
    s = str(val).strip().replace('.0', '')
    s = ''.join(filter(str.isdigit, s))
    if len(s) == 0: return None
    if s.startswith('254'): return f"'+{s}"
    return f"'{s}"

def clean_and_tokenize(text):
    if pd.isna(text): return set()
    clean = str(text).upper()
    for abbr, full in ABBREVIATION_MAP.items():
        clean = re.sub(r'\b' + re.escape(abbr) + r'\b', full, clean)
    clean = re.sub(r'[^A-Z0-9\s]', ' ', clean)
    tokens = set(clean.split())
    return tokens - STOP_WORDS

def build_master_brand_map(df_kb):
    brand_map = {}
    for b in BRAND_LIST: brand_map[b.upper()] = b
    for alias, canonical in BRAND_ALIASES.items(): brand_map[alias.upper()] = canonical
    
    if 'Brand' in df_kb.columns:
        kb_brands = df_kb['Brand'].dropna().astype(str).str.strip()
        for b in kb_brands:
            if b.upper() not in brand_map: brand_map[b.upper()] = b

    emergency_aliases = {
        "ANTHELIOS": "La Roche Posay", "LIPIKAR": "La Roche Posay",
        "EFFACLAR": "La Roche Posay", "CICAPLAST": "La Roche Posay",
        "AQUAPHOR": "Aquaphor", "AVEENO": "Aveeno", "MEDICUBE": "Medicube",
        "SS": "Seven Seas", "SUDOCREM": "Sudocrem", "BIO OIL": "Bio-Oil"
    }
    for k, v in emergency_aliases.items(): brand_map[k] = v
    return brand_map

def load_knowledge_base(path):
    if not path.exists(): return {}, {}, []
    try:
        df_kb = pd.read_csv(path)
        brand_map_case = build_master_brand_map(df_kb)
        products_by_brand = {}
        all_products = []

        if 'Name' in df_kb.columns:
            df_kb['Price'] = pd.to_numeric(df_kb['Price'], errors='coerce').fillna(0)
            for _, row in df_kb.iterrows():
                raw_brand = str(row.get('Brand', '')).strip().upper()
                b_name = brand_map_case.get(raw_brand, raw_brand) 
                p_name = str(row['Name']).strip()
                item = {
                    'name_orig': p_name,
                    'concern': str(row.get('Concerns', 'Unknown')).strip(),
                    'tokens': clean_and_tokenize(p_name),
                    'price': row['Price'],
                    'brand_upper': b_name.upper()
                }
                if b_name and b_name != 'NAN':
                    products_by_brand.setdefault(b_name.upper(), []).append(item)
                all_products.append(item)
        return brand_map_case, products_by_brand, all_products
    except: return {}, {}, []

def match_brand_smart(text, brand_map_case):
    if pd.isna(text): return None, "Unknown"
    text_upper = str(text).upper()
    for b_upper in brand_map_case:
        if text_upper.startswith(b_upper + " ") or text_upper == b_upper:
            return b_upper, brand_map_case[b_upper]
    sorted_brands = sorted(brand_map_case.keys(), key=len, reverse=True)
    for b_upper in sorted_brands:
        if re.search(r'\b' + re.escape(b_upper) + r'\b', text_upper):
            return b_upper, brand_map_case[b_upper]
    return None, "Unknown"

def get_canonical_category(row):
    sources = [str(row.get('Department', '')), str(row.get('Category', '')), str(row.get('Description', ''))]
    for text in sources:
        if not text or text.lower() == 'nan': continue
        for pattern, canonical in DEPARTMENT_TO_CANONICAL.items():
            if re.search(pattern, text, re.IGNORECASE): return canonical
    return "Unknown"

def match_product_smart(row, products_by_brand, all_products):
    pos_desc = str(row.get('Description', '')).upper()
    if any(x in pos_desc for x in ['DELIVERY', 'CHARGE', 'FEE']): return "Delivery Fee", "Logistics"

    pos_tokens = clean_and_tokenize(pos_desc)
    if not pos_tokens: return "Unknown", "Unknown"
    pos_price = float(row.get('Total (Tax Ex)', row.get('Amount', 0)))
    brand_upper = str(row.get('Matched_Brand_Upper', ''))

    if brand_upper and brand_upper in products_by_brand:
        candidates = products_by_brand[brand_upper]
    else:
        candidates = all_products

    scored_candidates = []
    for cand in candidates:
        intersection = pos_tokens.intersection(cand['tokens'])
        if not intersection: continue
        score_val = len(intersection) / len(pos_tokens)
        
        if score_val >= MIN_TOKEN_MATCH_RATIO:
            price_match = True
            if cand['price'] > 0 and pos_price > 0:
                diff = abs(cand['price'] - pos_price)
                ratio = diff / cand['price']
                if ratio > PRICE_TOLERANCE: price_match = False
            
            if price_match:
                scored_candidates.append({
                    'match': cand, 'score': score_val, 'price_diff': abs(cand['price'] - pos_price)
                })
    
    if not scored_candidates: return "Unknown", "Unknown"
    scored_candidates.sort(key=lambda x: (-x['score'], x['price_diff']))
    best = scored_candidates[0]
    
    if best['score'] >= 0.75:
        return best['match']['name_orig'], best['match']['concern']
    return "Unknown", "Unknown"

# ==========================================
# 3. MAIN PIPELINE
# ==========================================
def run_enrichment():
    print("🏷️  STARTING ENRICHMENT (V4.0 Final)...")
    if not ATTRIBUTION_PATH.exists():
        print(f"❌ Error: Input file missing at {ATTRIBUTION_PATH}")
        return

    df_attr = pd.read_csv(ATTRIBUTION_PATH)
    print("   🧹 Standardizing IDs...")
    id_cols = [c for c in df_attr.columns if 'contact' in c.lower() or 'id' in c.lower() or 'phone' in c.lower()]
    for col in id_cols:
        if 'contact' in col.lower(): df_attr[col] = df_attr[col].apply(clean_id)
        elif 'phone' in col.lower() or 'mobile' in col.lower(): df_attr[col] = df_attr[col].apply(clean_id_excel_safe)

    brand_map_case, products_by_brand, all_products = load_knowledge_base(KB_PATH)
    if not brand_map_case: 
        print("❌ Knowledge Base failed to load.")
        return

    print("   🔍 Matching Brands...")
    brand_matches = df_attr.apply(lambda row: match_brand_smart(row.get('Description', ''), brand_map_case), axis=1)
    df_attr['Matched_Brand_Upper'] = [x[0] for x in brand_matches]
    df_attr['Matched_Brand'] = [x[1] for x in brand_matches]

    df_attr['Canonical_Category'] = df_attr.apply(get_canonical_category, axis=1)

    print("   🔍 Matching Products (With Strict Guards)...")
    product_matches = df_attr.apply(lambda row: match_product_smart(row, products_by_brand, all_products), axis=1)
    df_attr['Matched_Product'] = [x[0] for x in product_matches]
    df_attr['Matched_Concern'] = [x[1] for x in product_matches]

    print("   📝 Saving...")
    cols = list(df_attr.columns)
    if 'Matched_Brand_Upper' in cols: cols.remove('Matched_Brand_Upper')
    
    target_cols = ['Matched_Brand', 'Canonical_Category', 'Matched_Product', 'Matched_Concern']
    for c in target_cols: 
        if c in cols: cols.remove(c)
    if 'Description' in cols:
        idx = cols.index('Description')
        for i, c in enumerate(target_cols):
            cols.insert(idx + 1 + i, c)
            
    df_attr = df_attr[cols]
    df_attr.to_csv(OUTPUT_FILE, index=False)

    total = len(df_attr)
    prod_found = len(df_attr[df_attr['Matched_Product'] != 'Unknown'])
    print(f"🚀 SUCCESS! Processed {total:,} rows.")
    print(f"   ✅ Products Confidently Matched: {prod_found:,} ({prod_found/total:.1%})")
    print(f"   📂 Output: {OUTPUT_FILE}")

if __name__ == "__main__":
    run_enrichment()