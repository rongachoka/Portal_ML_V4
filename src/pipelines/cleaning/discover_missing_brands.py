"""
discover_missing_brands.py
==========================
Diagnostic utility: surfaces unmatched product descriptions in attribution output.

Scans the enriched social sales CSV for rows where Matched_Brand == "Unknown",
tokenises the description, and reports the most common leading tokens — these
are candidates to add to BRAND_LIST or BRAND_ALIASES in config/brands.py.

Input:  data/03_processed/sales_attribution/final_enriched_social_sales_Jan25_Jan26.csv
Output: console report of unmatched token frequencies

Run manually after a KB update or when Unknown Brand counts are high.
"""

import pandas as pd
import re
from collections import Counter
from Portal_ML_V4.src.config.settings import PROCESSED_DATA_DIR

# ==========================================
# 1. CONFIGURATION
# ==========================================
INPUT_FILE = PROCESSED_DATA_DIR / "sales_attribution" / "final_enriched_social_sales_Jan25_Jan26.csv"

# Words to ignore if they appear at the start of a description
IGNORE_WORDS = {
    'DELIVERY', 'CHARGE', 'CONSULTATION', 'PHARMACY', 'DISPENSING', 
    'GENERAL', 'A', 'AN', 'FOR', 'WITH', 'BY', 'X', 'PACK'
}

# ==========================================
# 2. DISCOVERY LOGIC
# ==========================================
def discover_brands():
    print("🕵️  MINING DATA FOR MISSING BRANDS...")

    if not INPUT_FILE.exists():
        print("❌ Input file not found.")
        return

    df = pd.read_csv(INPUT_FILE)
    
    # 1. Filter for the "Unknowns"
    # We only care about rows where we failed to find a brand
    unknowns = df[
        (df['Matched_Brand'] == 'Unknown') | 
        (df['Matched_Brand'] == 'General') | 
        (df['Matched_Brand'].isna())
    ].copy()

    if unknowns.empty:
        print("✅ No unknown brands found! Your data is 100% mapped.")
        return

    print(f"   📉 Analyzing {len(unknowns):,} unmatched sales records...")

    # 2. Extract Candidates (First Word & First 2 Words)
    candidates = []
    
    for _, row in unknowns.iterrows():
        desc = str(row.get('Description', '')).upper().strip()
        amount = float(row.get('Amount', 0))
        
        # Remove numbers and special chars
        clean_desc = re.sub(r'[^A-Z\s]', '', desc)
        tokens = clean_desc.split()
        
        if not tokens: continue

        # Candidate A: First Word (e.g., "HIMALAYA")
        word_1 = tokens[0]
        if len(word_1) > 2 and word_1 not in IGNORE_WORDS:
            candidates.append({'Term': word_1, 'Revenue': amount, 'Type': '1-Word'})

        # Candidate B: First 2 Words (e.g., "DR ORGANIC")
        if len(tokens) >= 2:
            word_2 = f"{tokens[0]} {tokens[1]}"
            if word_1 not in IGNORE_WORDS and tokens[1] not in IGNORE_WORDS:
                candidates.append({'Term': word_2, 'Revenue': amount, 'Type': '2-Word'})

    # 3. Aggregate & Rank
    df_cand = pd.DataFrame(candidates)
    
    # Group by Term and sum Revenue
    leaderboard = df_cand.groupby('Term').agg(
        Total_Revenue=('Revenue', 'sum'),
        Count=('Revenue', 'count')
    ).reset_index()

    # Sort by Money (That's what we care about)
    leaderboard = leaderboard.sort_values('Total_Revenue', ascending=False).head(30)

    # 4. Output
    print("\n💰 TOP 30 MISSING BRANDS (By Revenue Impact):")
    print(f"{'CANDIDATE':<25} | {'REVENUE':<10} | {'COUNT':<5}")
    print("-" * 45)
    
    for _, row in leaderboard.iterrows():
        print(f"{row['Term']:<25} | {int(row['Total_Revenue']):<10,} | {row['Count']:<5}")

    print("\n💡 ACTION: Copy the valid brands above into your src/config/brands.py list!")

if __name__ == "__main__":
    discover_brands()