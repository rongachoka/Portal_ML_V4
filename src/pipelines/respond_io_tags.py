import pandas as pd
import numpy as np
from pathlib import Path

# ✅ IMPORT PATHS
from Portal_ML_V4.src.config.settings import (
    PROCESSED_DATA_DIR,
)

# ==========================================
# 1. CONFIGURATION
# ==========================================
INPUT_FILE = PROCESSED_DATA_DIR / "sales_attribution" / "final_enriched_social_sales_Jan25_Jan26.csv"
OUTPUT_FILE = PROCESSED_DATA_DIR / "respond_io" / "contact_tags_for_broadcast_upload.csv"

TAG_COLUMNS = {
    'Matched_Concern': 'Interest',   
    'Matched_Brand': 'Brand',        
    'Canonical_Category': 'Category' 
}

# 🚫 BLACKLIST (Tags to Exclude)
BLACKLIST_TAGS = {
    "LOGISTICS",
    "DELIVERY FEE",
    "DELIVERY CHARGE",
    "STRETCH MARKS",
    "GENERAL CARE",
    "UNKNOWN",
    "NAN",
    "NONE"
}

# ==========================================
# 2. PROCESSING FUNCTION
# ==========================================
def generate_tags():
    print("🏷️  GENERATING RESPOND.IO UPLOAD FILE (Consolidated Tags)...")

    if not INPUT_FILE.exists():
        print(f"❌ Error: Input file not found at {INPUT_FILE}")
        return

    # 1. Load Data
    df = pd.read_csv(INPUT_FILE)
    
    # Normalize Column Names
    if 'Contact ID' in df.columns:
        df.rename(columns={'Contact ID': 'contact_id'}, inplace=True)
    
    if 'contact_id' not in df.columns:
        print(f"❌ Error: 'contact_id' column missing. Found: {list(df.columns)}")
        return
    
    # Filter valid contacts
    df = df.dropna(subset=['contact_id'])
    df['contact_id'] = df['contact_id'].astype(str).str.replace(r'\.0$', '', regex=True)

    print(f"   📥 Loaded {len(df):,} sales records.")

    # 2. Melt (Unpivot)
    tag_frames = []

    for col, tag_type in TAG_COLUMNS.items():
        if col in df.columns:
            temp_df = df[['contact_id', col]].copy()
            temp_df.columns = ['contact_id', 'Tag']
            temp_df = temp_df.dropna()
            temp_df['Tag'] = temp_df['Tag'].astype(str)
            tag_frames.append(temp_df)

    if not tag_frames:
        print("⚠️ No valid tag columns found.")
        return

    df_tags = pd.concat(tag_frames, ignore_index=True)

    # 3. CLEAN & EXPLODE (Process individual tags first)
    df_tags['Tag'] = df_tags['Tag'].str.split(',')
    df_tags = df_tags.explode('Tag')
    df_tags['Tag'] = df_tags['Tag'].str.strip()

    # Blacklist Filter
    mask_valid = (df_tags['Tag'] != '') & (df_tags['Tag'] != 'nan')
    df_tags = df_tags[mask_valid]
    
    mask_blacklist = df_tags['Tag'].str.upper().isin(BLACKLIST_TAGS)
    df_tags = df_tags[~mask_blacklist]
    
    # Deduplicate (Remove duplicate tags per user)
    df_tags = df_tags.drop_duplicates()

    # 4. 🚨 THE FIX: CONSOLIDATE PER USER
    # Combine tags into "Tag1, Tag2, Tag3"
    print("   🔗 Consolidating tags per contact...")
    
    df_final = df_tags.groupby('contact_id')['Tag'].apply(lambda x: ','.join(x)).reset_index()
    
    # Rename for Respond.IO clarity
    df_final.rename(columns={'Tag': 'Tags'}, inplace=True)

    # 5. SAVE
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(OUTPUT_FILE, index=False)

    print(f"🚀 SUCCESS! Generated upload file for {len(df_final):,} contacts.")
    print(f"   📂 Output: {OUTPUT_FILE}")
    
    # Preview
    print("\n🔎 PREVIEW (Top 5):")
    print(df_final.head(5))

if __name__ == "__main__":
    generate_tags()