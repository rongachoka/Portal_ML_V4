import pandas as pd
import numpy as np
import re
from pathlib import Path

# ✅ IMPORT PATHS
from Portal_ML_V4.src.config.settings import (
    PROCESSED_DATA_DIR,
    BASE_DIR
)

# ==========================================
# 1. CONFIGURATION
# ==========================================
# Sources
CHATS_FILE = PROCESSED_DATA_DIR / "fact_sessions_enriched.csv"
KB_FILE = BASE_DIR / "data" / "01_raw" / "Final_Knowledge_Base_PowerBI.csv"

# Output
OUTPUT_FILE = PROCESSED_DATA_DIR / "respond_io" / "ad_hoc_skin_care_segment.csv"

# ==========================================
# 2. FAST GENERATOR
# ==========================================
def run_fast_segmentation():
    print("🚀 STARTING AD-HOC SEGMENTATION (Chat Context Scan)...")

    # --- 1. LOAD KNOWLEDGE BASE (Source of Truth) ---
    print("   📚 Indexing Brands from KB...")
    if not KB_FILE.exists():
        print("❌ CRITICAL: KB File missing.")
        return

    df_kb = pd.read_csv(KB_FILE)
    
    # Dynamic Column Finder for Brand
    brand_col = 'Brand' if 'Brand' in df_kb.columns else None
    if not brand_col:
        print("❌ CRITICAL: No 'Brand' column in KB.")
        return

    # Create a set of valid brands (Lowercase for matching, Original for tagging)
    # We filter for Skincare if possible, otherwise grab all
    if 'Canonical_Category' in df_kb.columns:
        # Optional: Prioritize Skincare, but for safety grab all brands mentioned
        # skin_brands = df_kb[df_kb['Canonical_Category'].str.contains('Skin', case=False, na=False)]
        pass 

    kb_brands = df_kb[brand_col].dropna().unique()
    # Dictionary: "cerave" -> "CeraVe"
    brand_lookup = {str(b).lower().strip(): str(b).strip() for b in kb_brands if len(str(b)) > 2}
    
    print(f"   ✅ Indexed {len(brand_lookup)} unique brands.")

    # --- 2. LOAD CHATS ---
    print("   📥 Loading Chat Sessions...")
    if not CHATS_FILE.exists():
        print("❌ CRITICAL: Enriched Sessions file missing.")
        return

    df_chats = pd.read_csv(CHATS_FILE)
    
    # Ensure ID exists
    if 'Contact ID' in df_chats.columns:
        df_chats['contact_id'] = df_chats['Contact ID']
    elif 'ContactID' in df_chats.columns:
        df_chats['contact_id'] = df_chats['ContactID']
    
    df_chats = df_chats.dropna(subset=['contact_id', 'full_context'])
    df_chats['contact_id'] = df_chats['contact_id'].astype(str).str.replace(r'\.0$', '', regex=True)

    print(f"   🔍 Scanning {len(df_chats):,} conversations...")

    # --- 3. SCANNING LOGIC ---
    # We combine context + tags to be sure
    df_chats['search_text'] = (
        df_chats['full_context'].astype(str) + " " + 
        df_chats['final_tags'].astype(str)
    ).str.lower()

    contact_tags = {}

    # Iterate (It's an ad-hoc script, iteration is fine for <50k rows)
    for _, row in df_chats.iterrows():
        cid = row['contact_id']
        text = row['search_text']
        
        if cid not in contact_tags:
            contact_tags[cid] = set()

        # Check every brand against the text
        # Fast substring check first, then strict regex if needed
        # For ad-hoc speed, substring with spaces is usually sufficient
        for b_lower, b_real in brand_lookup.items():
            if b_lower in text:
                contact_tags[cid].add(b_real)

    # --- 4. FORMATTING FOR RESPOND.IO ---
    print("   🔗 Consolidating Tags...")
    
    final_rows = []
    for cid, tags in contact_tags.items():
        if tags:
            # Join tags with comma
            tag_string = ",".join(list(tags))
            final_rows.append({'contact_id': cid, 'Tags': tag_string})

    df_final = pd.DataFrame(final_rows)

    if df_final.empty:
        print("⚠️ No brands found in conversations.")
        return

    # --- 5. EXPORT ---
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(OUTPUT_FILE, index=False)

    print(f"🚀 SUCCESS! Segment created for {len(df_final):,} contacts.")
    print(f"   📂 File: {OUTPUT_FILE}")
    print("\n🔎 Sample:")
    print(df_final.head())

if __name__ == "__main__":
    run_fast_segmentation()