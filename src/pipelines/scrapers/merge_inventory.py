import pandas as pd
import numpy as np
import re
from pathlib import Path
from difflib import SequenceMatcher

# --- CONFIGURATION ---
BASE_DIR = Path("C:\\Users\\Portal Pharmacy\\Documents\\Portal ML Analys\\Ron's Work (2)\\Portal_ML\\Portal_ML_V4\\data\\01_raw")
INVENTORY_FILE = BASE_DIR / "Products 31 Jan.csv" 
SCRAPED_FILE = BASE_DIR / "scraped_prices_jan2026.csv"
OUTPUT_FILE = BASE_DIR / "Golden_Product_Catalog_Verified.csv"


# 🛑 STRICTER THRESHOLD
# 0.80 ensures we only match if we are VERY sure. 
# Bad matches in your example were ~0.70-0.77, so this cuts them off.
NAME_THRESHOLD = 0.80
PRICE_TOLERANCE = 10.0

# 🧪 VARIANT GUARD (The "Flavor/Ingredient" Check)
# If A has one of these and B has a DIFFERENT one from the same group -> Penalty.
# OR if A has "Mint" and B does not -> Penalty (optional, but strict).
# Simple approach: If A has "Glycolic" and B doesn't, that's suspicious.
VARIANT_KEYWORDS = {
    'acids': ['glycolic', 'salicylic', 'lactic', 'azelaic', 'hyaluronic', 'retinol', 'benzoyl'],
    'scents': ['mint', 'berry', 'berries', 'vanilla', 'cocoa', 'shea', 'argan', 'coconut', 'lavender', 'rose', 'lemon'],
    'actives': ['caffeine', 'niacinamide', 'vitamin c', 'zinc', 'magnesium', 'charcoal'],
    'hair_types': ['leave-in', 'rinse-out', 'shampoo', 'conditioner', 'mask'],
}

def normalize_units(text):
    """Cleans 20mls -> 20ml, 50gms -> 50g to help good matches score higher."""
    if not isinstance(text, str): return ""
    text = text.lower().strip()
    text = re.sub(r'(\d+)\s*mls?\b', r'\1ml', text) # 20mls -> 20ml
    text = re.sub(r'(\d+)\s*gms?\b', r'\1g', text)   # 50gms -> 50g
    text = re.sub(r'(\d+)\s*oz\b', r'\1oz', text)
    text = re.sub(r'\s+', ' ', text) # Remove double spaces
    return text

def get_first_word(text):
    if not text: return ""
    return text.split()[0]

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def clean_price(price_val):
    if pd.isna(price_val): return 0.0
    s = str(price_val).replace(',', '').replace('KSh', '').replace('ksh', '').strip()
    try:
        return float(s)
    except:
        return 0.0

# ✅ SANDWICH FIX + VARIANT CHECK
def check_word_overlap(name_a, name_b):
    set_a = set(name_a.split())
    set_b = set(name_b.split())
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    if union == 0: return 0.0
    return intersection / union

def check_variant_conflict(name_a, name_b):
    """
    Returns TRUE if there is a specific keyword conflict.
    e.g. 'Glycolic' in A, 'Salicylic' in B -> Conflict.
    """
    for category, words in VARIANT_KEYWORDS.items():
        # Check if A has a word from this category
        found_a = [w for w in words if w in name_a]
        found_b = [w for w in words if w in name_b]
        
        # If both have words from this category, but they are DIFFERENT
        # e.g. A has 'mint', B has 'berry'
        if found_a and found_b:
            if set(found_a) != set(found_b):
                return True # Conflict!
                
        # Stricter: If A has 'Caffeine' and B doesn't have it at all -> Conflict
        # (Only applies if B doesn't have a *different* active, which we caught above)
        if found_a and not found_b:
             # Check if B is missing a KEY defining word
             return True
             
    return False

def run_golden_merge_v8():
    print("✨ Starting Golden Merge V8 (Strict Variant Guard)...")
    
    try:
        df_inv = pd.read_csv(INVENTORY_FILE, encoding='utf-8', on_bad_lines='skip')
        df_web = pd.read_csv(SCRAPED_FILE)
        
        df_inv['clean_price'] = df_inv['Price'].apply(clean_price)
        df_web['clean_price'] = df_web['Website_Price'].apply(clean_price)
        
        # Apply Unit Normalization
        df_inv['match_name'] = df_inv['Name'].apply(normalize_units)
        df_web['match_name'] = df_web['Website_Name'].apply(normalize_units)
        
        print(f"   📂 Inventory Rows: {len(df_inv)} | Web Rows: {len(df_web)}")

    except Exception as e:
        print(f"   ❌ Data Load Error: {e}")
        return

    web_records = df_web.to_dict('records')
    final_rows = []
    
    print("   🔍 Matching... (High Threshold 0.80)")
    
    matches_found = 0
    
    for idx, row in df_inv.iterrows():
        
        # Default Row (No Match)
        final_row = {
            'ItemCode': row.get('ItemCode', ''),
            'Name': row.get('Name', ''),
            'Price': row.get('Price', 0),
            'Quantity': row.get('Quantity', 0),
            'Category 1': row.get('Category 1', ''),
            'Category 2': row.get('Category 2', ''),
            'Web_Name_Used': '',      # LEFT BLANK if no match
            'Concerns': '',           # LEFT BLANK
            'Product_Link': '',       # LEFT BLANK
            'Detailed_Desc': '',      # LEFT BLANK
            'Match_Score': 0.0
        }

        inv_name = row['match_name']
        inv_price = row['clean_price']
        inv_first_word = get_first_word(inv_name)
        
        if len(inv_name) < 3:
            final_rows.append(final_row)
            continue

        candidates = [w for w in web_records if inv_first_word in w['match_name']]
        
        if candidates:
            best_score = 0
            best_match = None

            for web_item in candidates:
                web_name = web_item['match_name']
                web_price = web_item['clean_price']
                
                # 🛑 VARIANT GUARD (New)
                if check_variant_conflict(inv_name, web_name): 
                    continue # Skip this candidate immediately
                
                # Scores
                seq_score = similar(inv_name, web_name)
                token_score = check_word_overlap(inv_name, web_name)
                text_score = (seq_score + token_score) / 2

                price_match = False
                if inv_price > 0 and web_price > 0:
                    if abs(inv_price - web_price) <= PRICE_TOLERANCE:
                        price_match = True
                
                final_score = text_score
                
                if price_match:
                    # Only boost if text is already decent (>0.5)
                    if text_score > 0.5: final_score += 0.25
                elif inv_price > 0 and web_price > 0 and abs(inv_price - web_price) > 500:
                    final_score -= 0.20

                if final_score > best_score:
                    best_score = final_score
                    best_match = web_item

            # CHECK THRESHOLD (0.80)
            if best_score >= NAME_THRESHOLD and best_match:
                final_row['Web_Name_Used'] = best_match['Website_Name']
                final_row['Concerns'] = best_match.get('Concerns_Identified', 'General Inquiry')
                final_row['Product_Link'] = best_match.get('Product_URL', '')
                final_row['Detailed_Desc'] = best_match.get('Raw_Description', '')
                final_row['Match_Score'] = round(best_score, 2)
                matches_found += 1
        
        final_rows.append(final_row)

    df_final = pd.DataFrame(final_rows)
    print(f"   ✅ Processed {len(df_final)} products.")
    print(f"   🔗 Matches Found: {matches_found} / {len(df_final)}")
    
    df_final.to_csv(OUTPUT_FILE, index=False)
    print(f"   🎉 Saved: {OUTPUT_FILE}")
    
    # AUDIT BAD MERGES
    print("\n   🔎 Audit of previously 'Bad' Merges (Should be Empty or Fixed):")
    # Check for the Gaviscon mismatch
    audit = df_final[df_final['Name'].str.contains("Gaviscon", na=False)]
    for _, row in audit.iterrows():
        print(f"      {row['Name']} -> {row['Web_Name_Used']} (Score: {row['Match_Score']})")

if __name__ == "__main__":
    run_golden_merge_v8()