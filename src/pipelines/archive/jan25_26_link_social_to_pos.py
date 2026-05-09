import pandas as pd
import numpy as np
import re
from difflib import SequenceMatcher
from pathlib import Path

# ✅ IMPORT PATHS (V3 Settings)
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
    DEPARTMENT_TO_CANONICAL = {}

# ==========================================
# 1. CONFIGURATION
# ==========================================
# 🚨 INPUT: The file created by Waterfall V7
ATTRIBUTION_PATH = PROCESSED_DATA_DIR / "sales_attribution" / "attributed_sales_waterfall_v7.csv"
# 🚨 INPUT: The Source of Truth for Tags & Context
SESSIONS_PATH = PROCESSED_DATA_DIR / "fact_sessions_enriched.csv"
# 🚨 KB: Knowledge Base
KB_PATH = BASE_DIR / "data" / "01_raw" / "Final_Knowledge_Base_PowerBI.csv"

# 🚨 OUTPUT: The Final Report
OUTPUT_FILE = PROCESSED_DATA_DIR / "sales_attribution" / "social_sales_final_enriched_audit.csv"

# 🎛️ TUNING
PRICE_TOLERANCE = 0.20
MATCH_THRESHOLD = 0.60
STOP_WORDS = {'AND', 'FOR', 'WITH', 'OF', 'TO', 'IN', 'ON', 'AT', 'ML', 'GM', 'PCS', 'TUBE', 'BOTTLE'}

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def clean_text_for_matching(text):
    if pd.isna(text): return ""
    clean = str(text).upper()
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

def get_canonical_category(row):
    # Prefer the matched category if available, else fall back to POS Department
    if pd.notna(row.get('Matched_Category')):
        return row['Matched_Category']
        
    dept = str(row.get('Department', '')).upper()
    cat = str(row.get('Category', '')).upper()
    for pattern, canonical in DEPARTMENT_TO_CANONICAL.items():
        if re.search(pattern, dept, re.IGNORECASE) or re.search(pattern, cat, re.IGNORECASE):
            return canonical.replace("Product Inquiry - ", "").strip()
    return cat.title() if cat != 'NAN' else "General"

# ==========================================
# 3. MAIN LOGIC
# ==========================================
def run_final_enrichment_audit():
    print("🏷️  STARTING FINAL PRODUCT ENRICHMENT (WITH AUDIT CONTEXT)...")
    
    if not ATTRIBUTION_PATH.exists():
        print(f"❌ Error: Missing {ATTRIBUTION_PATH}")
        return

    # 1. LOAD ATTRIBUTION DATA
    print("   📥 Loading Waterfall Results...")
    df_attr = pd.read_csv(ATTRIBUTION_PATH)
    
    # 2. LOAD SESSIONS (To recover 'final_tags' AND 'full_context')
    if SESSIONS_PATH.exists():
        print("   📥 Loading Session Context & Tags...")
        df_sess = pd.read_csv(SESSIONS_PATH)
        
        # Create lookup dictionaries
        # We need session_id -> final_tags AND session_id -> full_context
        tags_map = df_sess.set_index('session_id')['final_tags'].to_dict()
        context_map = df_sess.set_index('session_id')['full_context'].to_dict()
        
        # Map them back to the attribution file
        df_attr['final_tags'] = df_attr['session_id'].map(tags_map)
        df_attr['full_context'] = df_attr['session_id'].map(context_map)
    else:
        print("   ⚠️ Warning: Session file missing. Context will be empty.")
        df_attr['final_tags'] = ""
        df_attr['full_context'] = ""

    # 3. LOAD KNOWLEDGE BASE
    if KB_PATH.exists():
        df_kb = pd.read_csv(KB_PATH)
        name_col = 'Name' if 'Name' in df_kb.columns else 'Item Description'
        df_kb['Clean_Desc'] = df_kb[name_col].apply(clean_text_for_matching)
        df_kb['Price'] = pd.to_numeric(df_kb['Price'], errors='coerce').fillna(0)
        
        # Index by Brand for speed
        kb_by_brand = {}
        if 'Brand' in df_kb.columns:
            for brand, grp in df_kb.groupby('Brand'):
                kb_by_brand[str(brand).upper()] = grp.to_dict('records')
    else:
        print("❌ KB Not Found. Skipping product drill-down.")
        return

    matched_products = []
    matched_brands = []
    matched_concerns = []
    matched_cats = []

    print("   🔍 Matching POS Descriptions to Knowledge Base...")

    for idx, row in df_attr.iterrows():
        pos_desc = str(row.get('Description', ''))
        # Handle 'Total (Tax Ex)' vs 'Amount' ambiguity from previous merges
        pos_price = float(row.get('Total (Tax Ex)', 0))
        if pos_price == 0: pos_price = float(row.get('Amount', 0))
        
        qty = float(row.get('Qty Sold', 1))
        unit_price = pos_price / qty if qty > 0 else pos_price
        
        # 1. Get Brand from CHAT TAGS (High Confidence)
        detected_brands = extract_brands_from_tags(row.get('final_tags', ''))
        
        best_match_name = f"POS: {pos_desc}"
        best_brand = "Unknown"
        best_concern = "General"
        best_cat = "General"
        
        # If we have a brand from the chat, restrict search to that brand
        if detected_brands:
            best_score = 0
            best_candidate = None
            pos_clean = clean_text_for_matching(pos_desc)
            
            for brand in detected_brands:
                candidates = kb_by_brand.get(str(brand).upper(), [])
                
                for cand in candidates:
                    kb_clean = cand['Clean_Desc']
                    score = fuzzy_similarity(pos_clean, kb_clean)
                    
                    # Price Check
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
                best_match_name = best_candidate[name_col]
                best_concern = str(best_candidate.get('Concerns', 'General'))
                best_cat = str(best_candidate.get('Canonical_Category', 'General'))
            else:
                # Brand is known, product is not
                best_brand = detected_brands[0]
                best_match_name = f"General {best_brand} Product"
                # Infer category from brand
                best_cat = get_canonical_category({'Department': best_brand}) 

        matched_products.append(best_match_name)
        matched_brands.append(best_brand)
        matched_concerns.append(best_concern)
        matched_cats.append(best_cat)

    # 4. ASSIGN & SAVE
    df_attr['Matched_Brand'] = matched_brands
    df_attr['Matched_Product'] = matched_products
    df_attr['Matched_Concern'] = matched_concerns
    df_attr['Matched_Category'] = matched_cats
    
    # Final Canonical Category Cleanup
    df_attr['Canonical_Category'] = df_attr.apply(get_canonical_category, axis=1)

    print("   📝 Saving Final Audit Report...")
    
    # Reorder for Audit Readability
    cols = list(df_attr.columns)
    prio = [
        'session_id', 
        'Transaction ID', 
        'match_type', 
        'confidence', 
        'full_context',          # <--- CONTEXT COLUMN FOR AUDIT
        'Sale_Date', 
        'Description', 
        'Matched_Brand', 
        'Matched_Product', 
        'Total (Tax Ex)', 
        'contact_name', 
        'primary_category'
    ]
    final_cols = [c for c in prio if c in cols] + [c for c in cols if c not in prio]
    
    df_attr[final_cols].to_csv(OUTPUT_FILE, index=False)
    
    kb_matches = len(df_attr[~df_attr['Matched_Product'].str.startswith("POS:", na=False)])
    print(f"🚀 SUCCESS! Processed {len(df_attr)} linked rows.")
    print(f"   ✅ KB Product Matches: {kb_matches} ({(kb_matches/len(df_attr))*100:.1f}%)")
    print(f"   📂 Final Audit Report: {OUTPUT_FILE}")

if __name__ == "__main__":
    run_final_enrichment_audit()