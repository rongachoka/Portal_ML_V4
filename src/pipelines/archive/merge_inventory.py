import pandas as pd
import numpy as np
from pathlib import Path
from difflib import SequenceMatcher

# --- CONFIGURATION ---
BASE_DIR = Path("C:\\Users\\Portal Pharmacy\\Documents\\Portal ML Analys\\Ron's Work (2)\\Portal_ML\\Portal_ML_V4\\data\\01_raw")
INVENTORY_FILE = BASE_DIR / "Products 31 Jan.csv" 
SCRAPED_FILE = BASE_DIR / "scraped_prices_jan2026.csv"
OUTPUT_FILE = BASE_DIR / "Golden_Product_Catalog_Verified.csv"

NAME_THRESHOLD = 0.65
PRICE_TOLERANCE = 10.0

# 🛑 CRITICAL KEYWORDS (Type Guard)
CRITICAL_TYPES = [
    {'deodorant', 'roll-on', 'stick', 'spray', 'mist'},
    {'oil', 'serum', 'drops'},
    {'shampoo', 'conditioner', 'mask'},
    {'tablet', 'tab', 'capsule', 'cap', 'pill', 'softgel'},
    {'syrup', 'suspension', 'liquid', 'solution'},
    {'diaper', 'pant', 'nappy'},
    {'wipe', 'tissue'},
    {'gel', 'cream', 'lotion', 'moisturizer', 'balm'},
    {'scissors', 'clipper', 'cutter'},
    {'floss', 'brush', 'paste'},
]

def normalize(text):
    if not isinstance(text, str): return ""
    return text.lower().strip()

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

def has_critical_mismatch(name_a, name_b):
    set_a = set(name_a.split())
    set_b = set(name_b.split())
    type_a, type_b = None, None
    for i, group in enumerate(CRITICAL_TYPES):
        if not set_a.isdisjoint(group): type_a = i
        if not set_b.isdisjoint(group): type_b = i
    if type_a is not None and type_b is not None:
        if type_a != type_b: return True 
    return False

# ✅ SANDWICH FIX
def check_word_overlap(name_a, name_b):
    set_a = set(name_a.split())
    set_b = set(name_b.split())
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    if union == 0: return 0.0
    return intersection / union

def run_golden_merge_v7():
    print("✨ Starting Golden Merge V7 (No IDs - Direct Row Build)...")
    
    try:
        # Load Inventory (ignoring ItemCode issues)
        df_inv = pd.read_csv(INVENTORY_FILE, encoding='utf-8', on_bad_lines='skip')
        df_web = pd.read_csv(SCRAPED_FILE)
        
        # Helper columns for matching
        df_inv['clean_price'] = df_inv['Price'].apply(clean_price)
        df_web['clean_price'] = df_web['Website_Price'].apply(clean_price)
        
        df_inv['match_name'] = df_inv['Name'].apply(normalize)
        df_web['match_name'] = df_web['Website_Name'].apply(normalize)
        
        print(f"   📂 Inventory Rows: {len(df_inv)} | Web Rows: {len(df_web)}")

    except Exception as e:
        print(f"   ❌ Data Load Error: {e}")
        return

    web_records = df_web.to_dict('records')
    final_rows = []
    
    print("   🔍 Building Final Catalog Row-by-Row...")
    
    # Iterate Inventory ONCE
    for idx, row in df_inv.iterrows():
        
        # 1. Start with the Inventory Data
        # We assume 'ItemCode' might be junk, but we keep it just in case you want to see it
        # You can drop it later if you want.
        final_row = {
            'ItemCode': row.get('ItemCode', ''),
            'Name': row.get('Name', ''),
            'Price': row.get('Price', 0),
            'Quantity': row.get('Quantity', 0),
            'Category 1': row.get('Category 1', ''),
            'Category 2': row.get('Category 2', ''),
            # Defaults for Website Data (will be overwritten if match found)
            'Web_Name_Used': '',
            'Concerns': 'General Inquiry',
            'Product_Link': '',
            'Detailed_Desc': '',
            'Match_Score': 0.0
        }

        inv_name = row['match_name']
        inv_price = row['clean_price']
        inv_first_word = get_first_word(inv_name)
        
        if len(inv_name) < 3:
            final_rows.append(final_row)
            continue

        # 2. Find Match
        candidates = [w for w in web_records if inv_first_word in w['match_name']]
        
        if candidates:
            best_score = 0
            best_match = None

            for web_item in candidates:
                web_name = web_item['match_name']
                web_price = web_item['clean_price']
                
                # Type Guard
                if has_critical_mismatch(inv_name, web_name): continue
                
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
                    if text_score > 0.4: final_score += 0.25
                elif inv_price > 0 and web_price > 0 and abs(inv_price - web_price) > 500:
                    final_score -= 0.20

                if final_score > best_score:
                    best_score = final_score
                    best_match = web_item

            # 3. If Match Found, Enrich the Row
            if best_score >= NAME_THRESHOLD and best_match:
                final_row['Web_Name_Used'] = best_match['Website_Name']
                final_row['Concerns'] = best_match.get('Concerns_Identified', 'General Inquiry')
                final_row['Product_Link'] = best_match.get('Product_URL', '')
                final_row['Detailed_Desc'] = best_match.get('Raw_Description', '')
                final_row['Match_Score'] = round(best_score, 2)
        
        # 4. Add to Final List
        final_rows.append(final_row)

    # 5. Save
    df_final = pd.DataFrame(final_rows)
    print(f"   ✅ Processed {len(df_final)} products.")
    
    # Final check on duplicates (should be impossible now, but good practice)
    # df_final = df_final.drop_duplicates(subset=['Name', 'Price']) 
    
    df_final.to_csv(OUTPUT_FILE, index=False)
    print(f"   🎉 Saved Clean File: {OUTPUT_FILE}")

if __name__ == "__main__":
    run_golden_merge_v7()