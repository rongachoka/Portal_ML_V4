"""
analyze_brand_performance.py
============================
Brand-level performance analysis over attributed social media sessions.

Aggregates session and conversion data from the enriched attribution output
grouped by Matched_Brand, computing inquiry count, conversion rate, and
average revenue per brand. Cross-references the Knowledge Base for category
grouping.

Inputs:
    data/03_processed/sales_attribution/final_enriched_social_sales_Jan25_Jan26.csv
    data/01_raw/Final_Knowledge_Base_PowerBI.csv
Output: console report / CSV (path set inside script)

Run manually for brand performance deep-dives.
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path

# ✅ IMPORT PATHS
from Portal_ML_V4.src.config.settings import PROCESSED_DATA_DIR

# ✅ IMPORT BRANDS CONFIGURATION
# We try to import from your config file. If that fails, we use the fallback list you provided.
try:
    from Portal_ML_V4.src.config.brands import BRAND_LIST, BRAND_ALIASES
except ImportError:
    print("⚠️ Warning: Could not import config.brands.")

    BRAND_ALIASES = {
        "lrp": "La Roche-Posay", "la roche": "La Roche-Posay", "effaclar": "La Roche-Posay",
        "anthelios": "La Roche-Posay", "cicaplast": "La Roche-Posay", "lipikar": "La Roche-Posay",
        "palmers": "Palmer's", "dr organic": "Dr Organics", "dr. organic": "Dr Organics",
        "johnsons": "Johnson's", "body shop": "The Body Shop", "loreal": "L'Oreal",
        "sheamoisture": "Shea Moisture", "shea/m": "Shea Moisture", "mizan": "Mizani", "oxygen botanicals": "Oxygen",
        "ss": "Seven Seas"
    }

# ==========================================
# 1. CONFIGURATION
# ==========================================
# We use the ENRICHED attribution file which has Matched_Brand
INPUT_FILE = PROCESSED_DATA_DIR / "sales_attribution" / "social_sales_Jan25_Jan26.csv"
OUTPUT_FILE = PROCESSED_DATA_DIR / "report_top_products_by_brand.csv"

# Brands to Exclude from "Top Brands" analysis
EXCLUDE_BRANDS = ['Delivery', 'Delivery Fee', 'Logistics', 'Unknown', 'Generic', 'Other', 'Nan']

# ==========================================
# 2. PREPARE BRAND LOOKUP
# ==========================================
# We build a dictionary that maps UPPERCASE versions of brands/aliases to the Title Case Canonical Name
BRAND_LOOKUP = {}

# 1. Add Main Brands
for b in BRAND_LIST:
    BRAND_LOOKUP[b.upper()] = b

# 2. Add Aliases (Aliases overwrite generics if conflict exists)
for alias, canonical in BRAND_ALIASES.items():
    BRAND_LOOKUP[alias.upper()] = canonical

# 3. Sort Keys by Length (DESCENDING)
# This is crucial. We must match "LA ROCHE-POSAY" (14 chars) before "LA ROCHE" (8 chars).
SORTED_BRAND_KEYS = sorted(BRAND_LOOKUP.keys(), key=len, reverse=True)

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def extract_brand_smart(description):
    """
    Extracts Brand using the Config List.
    Strategy: Greedy Match at the Start of the string.
    """
    if pd.isna(description): return "Unknown"
    
    # Clean up description
    desc_clean = str(description).upper().strip()
    
    # Iterate through sorted brands (longest first)
    for key in SORTED_BRAND_KEYS:
        # Check if description STARTS with this brand
        if desc_clean.startswith(key):
            # Boundary Check: Ensure we didn't match half a word.
            # e.g. "CERAV" should not match "CERAVE" (handled by startswith), 
            # but "CERA" should not match "CERAMIC"
            
            # If exact match or next char is a space/punctuation
            if len(desc_clean) == len(key) or not desc_clean[len(key)].isalnum():
                return BRAND_LOOKUP[key]
    
    # Fallback: If no config brand found, return "Other"
    # This keeps the "Top Brands" list clean of random first words like "Pack" or "Set"
    return "Other"

# ==========================================
# 4. MAIN ANALYSIS
# ==========================================
def run_brand_analysis():
    print("📊 STARTING BRAND & PRODUCT ANALYSIS (Config-Based Matching)...")

    if not INPUT_FILE.exists():
        print(f"❌ Error: Input file not found at {INPUT_FILE}")
        return

    # 1. Load Data
    df = pd.read_csv(INPUT_FILE)
    print(f"   📥 Loaded {len(df):,} attributed sales transactions.")

    # 🚨 FIX: Create 'Qty Sold' if missing (1 Row = 1 Item)
    if 'Qty Sold' not in df.columns:
        df['Qty Sold'] = 1

    # 2. Extract Brand using New Logic
    print("   🏷️  Extracting Brands using Config List...")
    # We prefer 'Description' from POS as the source of truth for "What was sold"
    df['Brand_Final'] = df['Description'].apply(extract_brand_smart)

    # 3. IDENTIFY TOP 5 BRANDS (By Revenue)
    valid_brands = df[~df['Brand_Final'].isin(EXCLUDE_BRANDS)]
    
    # Determine revenue column
    amount_col = 'Total (Tax Ex)' if 'Total (Tax Ex)' in df.columns else 'Amount'
    
    # Group By Brand
    brand_revenue = valid_brands.groupby('Brand_Final')[amount_col].sum().reset_index()
    top_brands_list = brand_revenue.sort_values(amount_col, ascending=False).head(5)['Brand_Final'].tolist()
    
    print(f"   🏆 Top 5 Brands: {', '.join(top_brands_list)}")

    # 4. FILTER & RANK PRODUCTS (Top 10 per Brand)
    final_rows = []
    
    for brand in top_brands_list:
        # Get data for this brand
        brand_df = df[df['Brand_Final'] == brand].copy()
        
        # Determine Product Name Column
        # Prefer 'Matched_Product' if enriched, else 'Description'
        prod_col = 'Matched_Product' if 'Matched_Product' in brand_df.columns and brand_df['Matched_Product'].nunique() > 1 else 'Description'
        
        # Aggregate
        product_stats = brand_df.groupby(prod_col).agg({
            'Qty Sold': 'sum',
            amount_col: 'sum'
        }).reset_index()
        
        product_stats.rename(columns={
            'Qty Sold': 'Units_Sold', 
            amount_col: 'Total_Revenue'
        }, inplace=True)
        
        # Sort by Units Sold
        product_stats = product_stats.sort_values('Units_Sold', ascending=False).head(10)
        
        product_stats['Brand'] = brand
        product_stats['Rank'] = range(1, len(product_stats) + 1)
        product_stats['Product_Name'] = product_stats[prod_col]
        
        final_rows.append(product_stats)

    # 5. EXPORT
    if final_rows:
        final_report = pd.concat(final_rows)
        # Select clean columns
        final_report = final_report[['Brand', 'Rank', 'Product_Name', 'Units_Sold', 'Total_Revenue']]
        
        final_report.to_csv(OUTPUT_FILE, index=False)
        print(f"\n🚀 REPORT GENERATED: {OUTPUT_FILE}")
    else:
        print("⚠️ No data found for top brands.")

if __name__ == "__main__":
    run_brand_analysis()