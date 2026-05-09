import pandas as pd
import os
import re
import numpy as np
from tqdm.auto import tqdm
from pathlib import Path

# V3 PRODUCTION IMPORTS
from Portal_ML_V4.src.config.settings import (
    FINAL_TAGGED_DATA, PROCESSED_DATA_DIR,
    CLEANED_DATA_DIR, 
    HIGH_LTV_SUM_THRESHOLD, LOYAL_PAYMENTS_THRESHOLD
)
from Portal_ML_V4.src.config.brands import BRAND_LIST
from Portal_ML_V4.src.config.tag_rules import CONCERN_RULES

# --- CONFIGURATIONS ---
CHANNEL_MAP = {
    389017: 'WhatsApp',
    387986: 'Instagram',
    388255: 'Facebook',
    388267: 'TikTok',
    389086: 'Web Chat'
}

ALLOWED_CONCERNS = [
    'weight management', 'hair loss', 'sleep', 'sensitive skin', 
    'dry skin', 'oily skin', 'acne', 'hyperpigmentation'
]

# --- FIX: ROBUST PATH FINDER ---
def get_messages_parquet_path():
    """
    Finds the cleaned_messages.parquet file handling folder nesting.
    """
    # 1. Try relative path from where this script is running
    current_file = Path(__file__).resolve()
    # Go up 3 levels to reach 'Portal_ML_V4' root
    project_root = current_file.parent.parent.parent
    
    # Path 1: Standard
    path_1 = project_root / "data" / "02_interim" / "cleaned_messages.parquet"
    if path_1.exists(): return path_1
    
    # Path 2: Nested (Your specific structure)
    path_2 = project_root / "Portal_ML_V4" / "data" / "02_interim" / "cleaned_messages.parquet"
    if path_2.exists(): return path_2
    
    # Path 3: Absolute Hardcoded Fallback (Based on your diagnostic)
    hardcoded = Path(r"C:\\Users\\Portal Pharmacy\\Documents\\Portal ML Analys\\Ron's Work (2)\\Portal_ML\\Portal_ML_V4\\data\\02_interim\\cleaned_messages.parquet")
    if hardcoded.exists(): return hardcoded

    return None

def extract_image_url_from_json_string(text):
    """
    Extracts URL specifically from rows that have {"type":"attachment" ...}
    """
    if not isinstance(text, str): return None
    
    # Fast Filter: Must look like an attachment JSON
    if '"attachment"' not in text and '"type":"attachment"' not in text:
        return None

    # Regex: Look for "url":"..." ignoring whitespace
    match = re.search(r'"url"\s*:\s*"(https?://[^"]+)"', text)
    
    if match:
        url = match.group(1)
        # Filter stickers if explicitly marked
        if "/stickers/" in url.lower(): return None
        # Safety check for image extensions
        if any(ext in url.lower() for ext in ['.jpg', '.png', '.jpeg', '.webp']):
            return url
    return None

def estimate_message_count(text):
    if not isinstance(text, str) or not text.strip(): return 0
    return text.count('\n') + 1

def run_analytics_pipeline():
    print("📊 V4 ANALYTICS: FACT TABLE GENERATION")
    if not os.path.exists(FINAL_TAGGED_DATA):
        print(f"❌ Error: {FINAL_TAGGED_DATA} not found.")
        return

    df_sess = pd.read_parquet(FINAL_TAGGED_DATA)
    
    # ==========================================
    # 1. FETCH IMAGES FROM RAW PARQUET (FIXED)
    # ==========================================
    msg_path = get_messages_parquet_path()
    
    if msg_path:
        print(f"🖼️  Loading messages from: {msg_path}")
        df_msgs = pd.read_parquet(msg_path)
        
        # [CRITICAL FIX] Target 'Content' (Title Case)
        content_col = 'Content' 
        
        if content_col in df_msgs.columns:
            print(f"   ℹ️  Scanning column '{content_col}' for attachments...")
            
            # Ensure string format
            df_msgs['temp_content_str'] = df_msgs[content_col].astype(str)
            
            # Apply Extraction
            df_msgs['extracted_url'] = df_msgs['temp_content_str'].apply(extract_image_url_from_json_string)
            
            # Filter valid images
            df_imgs = df_msgs[df_msgs['extracted_url'].notna()].copy()
            print(f"   ℹ️  Found {len(df_imgs)} valid image attachments.")
            
            # [CRITICAL FIX] Join on 'Contact ID' because 'session_id' doesn't exist in messages
            # Note: The raw file has 'Contact ID' (Title Case) per your diagnostic
            join_col = 'Contact ID'
            
            if join_col in df_imgs.columns and join_col in df_sess.columns:
                # Group multiple images per Contact
                img_map = df_imgs.groupby(join_col)['extracted_url'].apply(lambda x: " | ".join(set(x))).to_dict()
                
                # Merge into main dataframe
                df_sess['image_attachments'] = df_sess[join_col].map(img_map).fillna("")
                
                count = df_sess['image_attachments'].replace("", np.nan).notna().sum()
                print(f"   ✅ Mapped images to {count} sessions.")
            else:
                print(f"   ⚠️ Join failed. Msg Cols: {list(df_imgs.columns)}")
                df_sess['image_attachments'] = ""
        else:
            print(f"   ⚠️ Could not find 'Content' column. Available: {list(df_msgs.columns)}")
            df_sess['image_attachments'] = ""
    else:
        print("   ❌ CRITICAL: Could not find cleaned_messages.parquet.")
        print("   Please check that the file exists in data/02_interim/")
        df_sess['image_attachments'] = ""

    # ==========================================
    # 2. RECOVER SOURCE
    # ==========================================
    source_col_found = next((c for c in df_sess.columns if 'opened' in c.lower() and 'source' in c.lower()), None)
    if not source_col_found:
        cleaned_path = CLEANED_DATA_DIR / "cleaned_conversations.csv"
        if os.path.exists(cleaned_path):
            df_clean = pd.read_csv(cleaned_path)
            csv_source_col = next((c for c in df_clean.columns if 'opened' in c.lower() and 'source' in c.lower()), None)
            if csv_source_col:
                merge_key = 'session_id' if 'session_id' in df_clean.columns else 'Contact ID'
                source_map = df_clean.set_index(merge_key)[csv_source_col].to_dict()
                df_sess['recovered_source'] = df_sess[merge_key].map(source_map)

    # ==========================================
    # 3. CLEANING & SORTING
    # ==========================================
    print("🧹 Cleaning & Sorting Timeline...")
    df_sess['mpesa_amount'] = pd.to_numeric(df_sess['mpesa_amount'], errors='coerce').fillna(0)
    
    if 'session_start' in df_sess.columns:
        df_sess['session_start'] = pd.to_datetime(df_sess['session_start'])
        df_sess = df_sess.sort_values(by=['Contact ID', 'session_start'], ascending=[True, True])
    
    df_sess['visit_rank'] = df_sess.groupby('Contact ID').cumcount() + 1
    df_sess['traffic_type'] = np.where(df_sess['visit_rank'] == 1, 'New Customer', 'Returning Customer')

    # ==========================================
    # 4. CHANNEL & SOURCE
    # ==========================================
    channel_col = 'channel_id' if 'channel_id' in df_sess.columns else 'Channel ID'
    if channel_col in df_sess.columns:
        df_sess['channel_id_clean'] = pd.to_numeric(df_sess[channel_col], errors='coerce')
        df_sess['channel_name'] = df_sess['channel_id_clean'].map(CHANNEL_MAP).fillna('Other')
    else:
        df_sess['channel_name'] = 'Unknown'

    def classify_source(row):
        obs = str(row.get('recovered_source', row.get('Opened By Source', ''))).lower()
        if any(x in obs for x in ['ctc_ads', 'ads', 'paid', 'facebook_ads']): return 'Paid Ads'
        elif any(x in obs for x in ['contact', 'user', 'organic']): return 'Organic'
        
        utm = str(row.get('utm_source', '')).lower()
        if 'ad' in utm or 'cpc' in utm: return 'Paid Ads'
        return 'Organic'

    df_sess['acquisition_source'] = df_sess.apply(classify_source, axis=1)

    # ==========================================
    # 5. TAG EXTRACTION
    # ==========================================
    print("🏷️ Extracting Tags...")

    def extract_tags(tag_string):
        if not isinstance(tag_string, str): 
            return "Unknown", "Inquiry", "None", "General", ["General"]
        
        tags_lower = tag_string.lower()
        
        zone = "Unknown"
        if "zone:" in tags_lower:
            match = re.search(r'zone:\s*([^|]+)', tag_string, re.IGNORECASE)
            if match: zone = match.group(1).strip()
        
        funnel = "Inquiry"
        if "converted" in tags_lower: funnel = "Converted"
        elif "payment instruction" in tags_lower: funnel = "Payment Instruction Sent"
        elif "price quoted" in tags_lower: funnel = "Price Quoted"
        
        concern = "General"
        for c in ALLOWED_CONCERNS:
            if c in tags_lower:
                concern = c.title()
                break
        
        found_brands = []
        for b in BRAND_LIST:
            if b.lower() in tags_lower:
                found_brands.append(b)
        if not found_brands:
            found_brands = ["General"]
        primary_brand = found_brands[0]
        
        return zone, funnel, concern, primary_brand, found_brands

    extracted = df_sess['final_tags'].apply(extract_tags)
    
    df_sess['zone_name'] = extracted.apply(lambda x: x[0])
    df_sess['funnel_stage'] = extracted.apply(lambda x: x[1])
    df_sess['skin_concern'] = extracted.apply(lambda x: x[2])
    df_sess['primary_brand'] = extracted.apply(lambda x: x[3])
    df_sess['all_brands_list'] = extracted.apply(lambda x: ", ".join(x[4]))
    df_sess['temp_brand_list'] = extracted.apply(lambda x: x[4])

    patterns_to_remove = [
        r'\|?\s*Zone:[^|]+', 
        r'\|?\s*Funnel:[^|]+',
        r'\|?\s*Payment Instruction[^|]*',
        r'\|?\s*Concern: Price Objection'
    ]
    for pat in patterns_to_remove:
        df_sess['final_tags'] = df_sess['final_tags'].str.replace(pat, '', regex=True)
    
    df_sess['final_tags'] = df_sess['final_tags'].str.strip(' |')

    # ==========================================
    # 6. METRICS & EXPORT
    # ==========================================
    df_sess['is_converted'] = df_sess['final_tags'].str.contains("Converted", na=False)
    df_sess['messages_to_conversion'] = df_sess['full_context'].apply(estimate_message_count)

    stats = df_sess.groupby('Contact ID').agg({'session_id':'count','is_converted':'sum','mpesa_amount':'sum'})
    stats.columns = ['num_sessions', 'num_payments', 'total_spend']
    df_sess = df_sess.merge(stats, on='Contact ID', how='left')

    def get_tier(row):
        if row['visit_rank'] == 1: return "New Customer"
        if row['total_spend'] >= HIGH_LTV_SUM_THRESHOLD: return "High-LTV Customer"
        if row['num_payments'] >= LOYAL_PAYMENTS_THRESHOLD: return "Loyal Customer"
        return "Returning Customer"
    
    df_sess['customer_tier'] = df_sess.apply(get_tier, axis=1)

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    # A. Clean Session Export
    cols_to_drop = ['path_taken', 'reason_for_purchase', 'channel_id_clean', 'recovered_source', 
                    'num_sessions', 'num_payments', 'total_spend', 'temp_brand_list', 'temp_content_str']
    df_export = df_sess.drop(columns=[c for c in cols_to_drop if c in df_sess.columns])
    df_export.to_csv(PROCESSED_DATA_DIR / "fact_sessions_enriched.csv", index=False)
    
    # B. Funnel & C. Brand Exports
    funnel_rows = []
    brand_rows = []

    for _, row in tqdm(df_sess.iterrows(), total=len(df_sess), desc="Exploding Data"):
        brands_in_row = row['temp_brand_list']
        
        for brand in brands_in_row:
            brand_rows.append({
                "contact_id": row['Contact ID'],
                "date": row['session_start'],
                "brand": brand,
                "is_converted": row['is_converted'],
                "traffic_type": row['traffic_type'],
                "channel": row['channel_name'],
                "source": row['acquisition_source']
            })

        for brand in brands_in_row:
            base = {
                "contact_id": row['Contact ID'], 
                "date": row['session_start'], 
                "brand": brand,
                "concern": row['skin_concern'],
                "tier": row['customer_tier'], 
                "traffic_type": row['traffic_type'],
                "zone": row['zone_name'],
                "channel": row['channel_name'],
                "source": row['acquisition_source'],
                "messages_to_conversion": row['messages_to_conversion']
            }
            funnel_rows.append({**base, "stage": "Inquiry", "val": 1})
            if row['funnel_stage'] == "Price Quoted": funnel_rows.append({**base, "stage": "Price Given", "val": 1})
            if row['is_converted']: funnel_rows.append({**base, "stage": "Purchased", "val": 1})

    pd.DataFrame(funnel_rows).to_csv(PROCESSED_DATA_DIR / "fact_funnel_analytics.csv", index=False)
    pd.DataFrame(brand_rows).to_csv(PROCESSED_DATA_DIR / "fact_brand_mentions.csv", index=False)
    
    print(f"✅ PIPELINE SUCCESS.")
    print(f"📍 Files Created: fact_sessions_enriched.csv, fact_funnel_analytics.csv, fact_brand_mentions.csv")

if __name__ == "__main__":
    run_analytics_pipeline()