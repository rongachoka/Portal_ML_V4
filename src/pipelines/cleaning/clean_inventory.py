"""
clean_inventory.py
==================
Cleans a raw POS item quantity CSV and maps departments to canonical categories.

Reads the Item Quantity List export from the POS system, applies
DEPARTMENT_TO_CANONICAL regex mapping to normalise department names,
and outputs a clean master inventory file.

Input:  data/01_raw/Item Quantity List_18jan2026.csv  (or equivalent)
Output: data/01_raw/clean_master_inventory.csv

Run manually after each POS inventory export.
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
import sys

try:
    # ✅ IMPORT THE SOURCE OF TRUTH
    from Portal_ML_V4.src.config.department_map import DEPARTMENT_TO_CANONICAL
    print("✅ Successfully imported DEPARTMENT_TO_CANONICAL from config.")
except ImportError as e:
    print(f"❌ Error importing config: {e}")
    print(f"   Current sys.path: {sys.path}")
    sys.exit(1)

# --- CONFIGURATION ---
BASE_DIR = Path("C:\\Users\\Portal Pharmacy\\Documents\\Portal ML Analys\\Ron's Work (2)\\Portal_ML\\Portal_ML_V4")
RAW_FILE = BASE_DIR / "data/01_raw/Item Quantity List_18jan2026.csv"
OUTPUT_FILE = BASE_DIR / "data/01_raw/clean_master_inventory.csv"


# 2. SUB-CATEGORY KEYWORDS (To fill the second column)
SUB_CAT_RULES = {
    'Antibiotics': ['antibiotic', 'augmentin', 'amoxyl', 'cipro', 'doxy'],
    'Pain Relief': ['pain', 'panadol', 'paracetamol', 'ibuprofen', 'diclofenac', 'aspirin'],
    'Cold & Flu': ['cold', 'flu', 'cough', 'syrup', 'sinus', 'throat'],
    'Acne': ['acne', 'pimple', 'effaclar', 'salicylic', 'benzoyl'],
    'Sunscreen': ['sunscreen', 'spf', 'uv', 'sun'],
    'Moisturizer': ['lotion', 'cream', 'moisturizer', 'hydrate'],
    'Cleanser': ['wash', 'cleanser', 'soap', 'scrub'],
    'Diapers': ['diaper', 'pampers', 'huggies', 'pants'],
    'Formula': ['milk', 'formula', 'nan', 'sma', 'lactogen'],
    'Vitamins': ['vitamin', 'multivitamin', 'zinc', 'iron', 'calcium'],
    'Essential Oils': ['essential oil', 'tea tree', 'lavender', 'eucalyptus', 'peppermint', 'rosemary', 'scented oil'],
    'Natural Oils': ['castor oil', 'magnesium oil', 'neem', 'bio-oil'],
    'Stress & Sleep': ['rescue remedy', 'bach', 'sleep', 'calm'],
    'Homeopathic Remedies': ['homeopathic', 'dilution', 'globules'],
}

def clean_text(text):
    return str(text).strip().title().replace("`", "").replace("'", "")

def categorize_product(row):
    """
    Uses DEPARTMENT_TO_CANONICAL to find the Main Category.
    Uses SUB_CAT_RULES to find the Sub Category.
    """
    # Create a full text string to search against (Dept + Category + Desc)
    full_text = f"{row.get('Department','')} {row.get('Category','')} {row.get('Description','')}".upper()
    
    # 1. DETERMINE MAIN CATEGORY (From Config Map)
    final_cat = "Other"
    
    # Iterate through your regex keys
    for pattern, intent_name in DEPARTMENT_TO_CANONICAL.items():
        if re.search(pattern, full_text):
            # Strip "Product Inquiry - " to get clean category name
            final_cat = intent_name.replace("Product Inquiry - ", "")
            break
            
    # 2. DETERMINE SUB-CATEGORY (Keyword Search)
    final_sub = "General"
    desc_lower = str(row.get('Description', '')).lower()
    
    for sub_name, keywords in SUB_CAT_RULES.items():
        if any(k in desc_lower for k in keywords):
            final_sub = sub_name
            break

    # 3. BRAND EXTRACTION (Updated with Full Website List)
    
    # MASTER LIST (Copied from your website data)
    ALL_WEBSITE_BRANDS = [
        "La Roche-Posay", "Advanced Clinicals", "Bath & Bodyworks", "Beauty Of Joseon", 
        "Clean And Clear", "Dark And Lovely", "Oxygen Botanicals", "Shea Moisture", 
        "Summer Fridays", "Victoria Secret", "African Pride", "Beauty Formula", 
        "Johnson's Baby", "Animal Parade", "Baby Dove", "Dr Organic", "Haliborange", 
        "Seven Seas", "The Ordinary", "Paulas Choice", "Reedle Shot", "Neutrogena", 
        "Heliocare", "Bioderma", "Aquaphor", "Vaseline", "Cetaphil", "Oatveen", 
        "Oilatum", "Sebamed", "Eucerin", "Palmers", "Regaine", "Garnier", "Blistex", 
        "Body Shop", "Burts Bees", "Laneige", "Sephora", "Topicals", "Bennets", 
        "Panoxyl", "St Ives", "Uncover", "Berocca", "Centrum", "Forever", "CeraVe", 
        "Aveeno", "Cantu", "Miadi", "Vichy", "Nivea", "Byoma", "Fenty", "Gisou", 
        "Anua", "Dove", "Fino", "Pixi", "Nyx", "E 45", "Olay", "Sinoz", "Acnes", 
        "Avent", "Radox", "Mielle", "Mizani", "Carmex", "Simple", "Cosrx", 
        "Avene", "Loreal", "Epimax", "Motions", "Biretix", "Medicube", "Naturium", 
        "Johnson", "Eos"
    ]

    found_brand = "General"
    desc_lower = str(row.get('Description', '')).lower()
    clean_name_lower = str(row.get('Clean_Name', '')).lower()
    
    # Combine text for maximum search area
    search_text = f" {desc_lower} {clean_name_lower} " 

    # ⚡ CRITICAL: Sort brands by length (Longest first)
    # This ensures "Baby Dove" matches before "Dove"
    # and "La Roche-Posay" matches before "La Roche"
    sorted_brands = sorted(ALL_WEBSITE_BRANDS, key=len, reverse=True)

    for brand in sorted_brands:
        # Check if brand is in text
        # We use simple inclusion, but you could add spaces f" {brand.lower()} " 
        # to prevent "NK" matching inside "PINK" if that becomes an issue.
        if brand.lower() in search_text:
            found_brand = brand
            break

    return pd.Series([found_brand, final_cat, final_sub])

def run_cleaning():
    print("⏳ Loading Raw Inventory...")
    df = pd.read_csv(RAW_FILE, on_bad_lines='skip')
    
    # Filter Junk (Same as before)
    df = df.dropna(subset=['Department', 'Description'])
    df = df[~df['Department'].str.contains("BRANCHES", na=False)]
    
    print("🧹 Categorizing based on Config Map...")
    df['Clean_Name'] = df['Description'].apply(clean_text)
    
    # Apply the new logic
    df[['Brand', 'Canonical_Category', 'Sub_Category']] = df.apply(categorize_product, axis=1)

    # Export
    cols_to_keep = ['Item Lookup Code', 'Clean_Name', 'Brand', 'Canonical_Category', 'Sub_Category', 'On-Hand']
    df_final = df[cols_to_keep]
    
    df_final.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Success! Standardized Inventory Saved: {OUTPUT_FILE}")
    print(df_final.head(10))

if __name__ == "__main__":
    run_cleaning()