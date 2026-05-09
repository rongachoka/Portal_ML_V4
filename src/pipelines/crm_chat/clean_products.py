"""
clean_products.py
=================
Cleans and fuzzy-matches POS product descriptions against the Knowledge Base.

For each product description in the attributed sales output, applies token
matching and fuzzy similarity to assign Matched_Brand, Matched_Product,
and Canonical_Category from the KB.

Inputs:
    data/03_processed/sales_attribution/final_enriched_social_sales_Jan25_Jan26.csv
    data/01_raw/Final_Knowledge_Base_PowerBI.csv

Output:
    data/03_processed/fact_social_sales_attribution_enriched.csv
    — attribution file with brand/product/category columns filled in

Entry point: run_clean_products() (called manually or from reporting scripts)
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path

# ✅ IMPORT PATHS
from Portal_ML_V4.src.config.settings import (
    BASE_DIR,
    PROCESSED_DATA_DIR,
)

# ==========================================
# 1. CONFIGURATION
# ==========================================
KB_PATH = BASE_DIR / "data" / "01_raw" / "Final_Knowledge_Base_PowerBI.csv"
ATTRIBUTION_PATH = PROCESSED_DATA_DIR / "sales_attribution" / "final_enriched_social_sales_Jan25_Jan26.csv"
OUTPUT_FILE = PROCESSED_DATA_DIR / "fact_social_sales_attribution_enriched.csv"

# Thresholds
MIN_TOKEN_MATCH_RATIO = 0.6  
PRICE_TOLERANCE = 0.20       

# Stop Words
STOP_WORDS = {
    'AND', 'FOR', 'WITH', 'OF', 'TO', 'IN', 'ON', 'AT', 'MY', 'A', 'IS',
    'ML', 'GM', 'KG', 'PCS', 'OZ', 'G', 'PACK', 'BOX', 'BOTTLE', 'TUBE'
}

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def clean_and_tokenize(text):
    if pd.isna(text): return set()
    clean = re.sub(r'[^A-Z0-9\s]', ' ', str(text).upper())
    tokens = set(clean.split())
    return tokens - STOP_WORDS

def load_knowledge_base(path):
    """
    Loads KB and builds lookup dicts.
    Now captures 'Concern' as well.
    """
    if not path.exists():
        print(f"❌ Error: Knowledge Base not found at {path}")
        return {}, {}, []

    try:
        df_kb = pd.read_csv(path)
        
        brand_map_case = {}
        products_by_brand = {}
        all_products = []

        if 'Brand' in df_kb.columns:
            brands = df_kb['Brand'].dropna().astype(str).str.strip()
            for b in brands:
                brand_map_case[b.upper()] = b

        if 'Name' in df_kb.columns:
            df_kb['Price'] = pd.to_numeric(df_kb['Price'], errors='coerce').fillna(0)
            
            for _, row in df_kb.iterrows():
                b_name = str(row.get('Brand', '')).strip().upper()
                p_name = str(row['Name']).strip()
                p_tokens = clean_and_tokenize(p_name)
                
                # ✅ Capture Concern
                p_concern = str(row.get('Concern', 'Unknown')).strip()
                
                item = {
                    'name_orig': p_name,
                    'concern': p_concern,  # Store Concern
                    'tokens': p_tokens,
                    'price': row['Price'],
                    'brand_upper': b_name
                }
                
                if b_name and b_name != 'NAN':
                    if b_name not in products_by_brand:
                        products_by_brand[b_name] = []
                    products_by_brand[b_name].append(item)
                
                all_products.append(item)
            
        return brand_map_case, products_by_brand, all_products
        
    except Exception as e:
        print(f"❌ Error reading Knowledge Base: {e}")
        return {}, {}, []

def match_brand_first(text, brand_map_case):
    if pd.isna(text): return None, "Unknown"
    text_upper = str(text).upper()
    
    sorted_brands = sorted(brand_map_case.keys(), key=len, reverse=True)
    
    for b_upper in sorted_brands:
        if re.search(r'\b' + re.escape(b_upper) + r'\b', text_upper):
            return b_upper, brand_map_case[b_upper]
            
    return None, "Unknown"

def calculate_overlap_score(input_tokens, candidate_tokens):
    if not input_tokens or not candidate_tokens: return 0.0
    intersection = input_tokens.intersection(candidate_tokens)
    score = len(intersection) / len(input_tokens)
    return score

def match_product_smart(row, products_by_brand, all_products):
    """
    Matches product and returns (Product Name, Concern).
    """
    pos_desc = str(row.get('Description', '')).upper()
    
    if 'DELIVERY' in pos_desc or 'CHARGE' in pos_desc:
        return "Unknown", "Unknown"

    pos_tokens = clean_and_tokenize(pos_desc)
    if not pos_tokens: return "Unknown", "Unknown"
    
    pos_price = float(row.get('Total (Tax Ex)', 0))

    brand_upper = str(row.get('Matched_Brand_Upper', ''))
    
    if brand_upper and brand_upper in products_by_brand:
        candidates = products_by_brand[brand_upper]
    else:
        candidates = all_products

    scored_candidates = []
    
    for cand in candidates:
        score = calculate_overlap_score(pos_tokens, cand['tokens'])
        
        if score >= MIN_TOKEN_MATCH_RATIO:
            scored_candidates.append({
                'match': cand,
                'score': score,
                'price_diff': abs(cand['price'] - pos_price)
            })
    
    if not scored_candidates:
        return "Unknown", "Unknown"
        
    scored_candidates.sort(key=lambda x: (-x['score'], x['price_diff']))
    
    best = scored_candidates[0]
    
    # Return Tuple: (Name, Concern)
    if best['score'] >= 0.9:
        return best['match']['name_orig'], best['match']['concern']
        
    if best['match']['price'] > 0 and pos_price > 0:
        price_ratio = abs(best['match']['price'] - pos_price) / best['match']['price']
        if price_ratio <= PRICE_TOLERANCE:
            return best['match']['name_orig'], best['match']['concern']
    
    if best['score'] > 0.8:
        return best['match']['name_orig'], best['match']['concern']

    return "Unknown", "Unknown"

# ==========================================
# 3. MAIN PIPELINE
# ==========================================
def run_enrichment():
    print("🏷️  STARTING BRAND-FIRST MATCHING (With Concern)...")

    # 1. Load Data
    if not ATTRIBUTION_PATH.exists():
        print("❌ Error: Attribution file missing.")
        return

    df_attr = pd.read_csv(ATTRIBUTION_PATH)
    brand_map_case, products_by_brand, all_products = load_knowledge_base(KB_PATH)

    if not brand_map_case:
        print("❌ Error: Knowledge Base empty.")
        return

    print(f"   📚 Knowledge Base: {len(brand_map_case):,} Brands Loaded.")

    # 2. MATCH BRAND
    print("   🔍 Matching Brands...")
    brand_matches = df_attr.apply(
        lambda row: match_brand_first(
            row.get('Category') if pd.notna(row.get('Category')) and row.get('Category') != '#NULL#' else row.get('Description'),
            brand_map_case
        ), axis=1
    )
    
    df_attr['Matched_Brand_Upper'] = [x[0] for x in brand_matches]
    df_attr['Matched_Brand'] = [x[1] for x in brand_matches]

    # 3. MATCH PRODUCT & CONCERN
    print("   🔍 Matching Products & Concerns...")
    
    # Apply returns a tuple (Product, Concern)
    product_matches = df_attr.apply(
        lambda row: match_product_smart(row, products_by_brand, all_products), axis=1
    )
    
    # Split tuple into two columns
    df_attr['Matched_Product'] = [x[0] for x in product_matches]
    df_attr['Matched_Concern'] = [x[1] for x in product_matches]

    # 4. Save
    print("   📝 Saving Results...")
    
    cols = list(df_attr.columns)
    if 'Matched_Brand_Upper' in cols: cols.remove('Matched_Brand_Upper')
    
    # Clean up column order
    target_cols = ['Matched_Brand', 'Matched_Product', 'Matched_Concern']
    for c in target_cols: 
        if c in cols: cols.remove(c)
        
    if 'Description' in cols:
        idx = cols.index('Description')
        cols.insert(idx + 1, 'Matched_Brand')
        cols.insert(idx + 2, 'Matched_Product')
        cols.insert(idx + 3, 'Matched_Concern')
        df_attr = df_attr[cols]

    df_attr.to_csv(OUTPUT_FILE, index=False)

    # 5. Stats
    total = len(df_attr)
    prod_unk = len(df_attr[df_attr['Matched_Product'] == 'Unknown'])

    print(f"🚀 SUCCESS! Processed {total:,} rows.")
    print(f"   ✅ Products Matched: {total - prod_unk:,} ({1 - prod_unk/total:.1%})")
    print(f"   📂 Output: {OUTPUT_FILE}")

    if prod_unk > 0:
        print("\n🔎 Top Unknown Items:")
        print(df_attr[df_attr['Matched_Product'] == 'Unknown']['Description'].value_counts().head(5))

if __name__ == "__main__":
    run_enrichment()