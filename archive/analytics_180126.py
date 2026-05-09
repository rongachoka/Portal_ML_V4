import pandas as pd
import os
import re
import numpy as np
from tqdm.auto import tqdm
from pathlib import Path

# V3 PRODUCTION IMPORTS
from Portal_ML_V4.src.config.settings import (
    FINAL_TAGGED_DATA, PROCESSED_DATA_DIR,
    CLEANED_DATA_DIR, BASE_DIR,
    HIGH_LTV_SUM_THRESHOLD, LOYAL_PAYMENTS_THRESHOLD
)
from Portal_ML_V4.src.config.brands import BRAND_LIST
from Portal_ML_V4.src.config.tag_rules import (
    enrich_canonical_categories_from_text
)
from Portal_ML_V4.src.config.department_map import (
    DEPARTMENT_TO_CANONICAL
)

# --- CONFIGURATIONS ---
CHANNEL_MAP = {
    389017: 'WhatsApp',
    387986: 'Instagram',
    388255: 'Facebook',
    388267: 'TikTok',
    389086: 'Web Chat'
}

MACRO_GROUP_MAP = {
    "Product Inquiry - Skincare": "Skincare",
    "Product Inquiry - Baby Care": "Baby Care",
    "Product Inquiry - Haircare": "Hair Care",
    "Product Inquiry - Medicine": "Medicine & Health",
    "Product Inquiry - Medical Devices and Kits": "Medicine & Health",
    "Product Inquiry - Homeopathy": "Medicine & Health",
    "Product Inquiry - Supplements": "Supplements & Nutrition",
    "Product Inquiry - Oral Care": "Oral Care",
    "Product Inquiry - Men Care": "Personal Care",
    "Product Inquiry - Women's Health": "Personal Care",
    "Product Inquiry - Perfumes": "Personal Care",
    "Stanley Cups": "Personal Care"
}


def load_product_category_map():
    """Maps brands to Macro/Sub categories using product CSV."""
    path = BASE_DIR / "data" / "01_raw" / "Products 31 Jan.csv"
    brand_map = {}
    if os.path.exists(path):
        try:
            df = pd.read_csv(path, encoding='utf-8', on_bad_lines='skip')
            cols = ['Category 1', 'Category 2']
            if all(c in df.columns for c in cols):
                df = df[cols].dropna().drop_duplicates()
                for _, row in df.iterrows():
                    bk = str(row['Category 1']).strip().lower()
                    rc = str(row['Category 2']).strip().upper()
                    tag = next((v for k, v in DEPARTMENT_TO_CANONICAL.items()
                                if k in rc), None)
                    if not tag:
                        t = enrich_canonical_categories_from_text(
                            str(row['Category 2']), None, source="category"
                        )
                        tag = list(t)[0] if t else None
                    if tag in MACRO_GROUP_MAP:
                        brand_map[bk] = {
                            'macro': MACRO_GROUP_MAP[tag],
                            'sub': tag.replace("Product Inquiry - ", "")
                        }
                    else:
                        brand_map[bk] = {'macro': "Other", 'sub': "General"}
        except Exception:
            pass
    return brand_map


def clean_id(val):
    """Forces ID to string and removes .0 suffix."""
    if pd.isna(val):
        return None
    s = str(val).strip()
    return s[:-2] if s.endswith('.0') else s


def find_id_col(df):
    """Identifies the ID column across Respond.io variants."""
    cands = ['Contact ID', 'ContactID', 'ID', 'id', 'contact_id']
    return next((c for c in cands if c in df.columns), None)


def run_analytics_pipeline():
    print("📊 V4 ANALYTICS: PEP 8 MASTER AUDIT")
    if not os.path.exists(FINAL_TAGGED_DATA):
        print(f"❌ Error: {FINAL_TAGGED_DATA} not found.")
        return

    df_sess = pd.read_parquet(FINAL_TAGGED_DATA)
    BRAND_MAP = load_product_category_map()
    df_sess['Contact ID'] = df_sess['Contact ID'].apply(clean_id)

    # 1. RECOVER SOURCE & NAMES
    cleaned_conv_path = CLEANED_DATA_DIR / "cleaned_conversations.csv"
    cleaned_cont_path = CLEANED_DATA_DIR / "cleaned_contacts.csv"
    source_map, name_map = {}, {}

    if os.path.exists(cleaned_conv_path):
        df_c = pd.read_csv(cleaned_conv_path)
        id_c = find_id_col(df_c)
        src_c = next((c for c in df_c.columns if 'source' in c.lower()), None)
        if id_c and src_c:
            df_c[id_c] = df_c[id_c].apply(clean_id)
            source_map = df_c.set_index(id_c)[src_c].to_dict()

    if os.path.exists(cleaned_cont_path):
        df_n = pd.read_csv(cleaned_cont_path)
        id_n = find_id_col(df_n)
        name_c = next((c for c in df_n.columns if 'name' in c.lower()), None)
        if id_n and name_c:
            df_n[id_n] = df_n[id_n].apply(clean_id)
            name_map = df_n.set_index(id_n)[name_c].to_dict()

    df_sess['recovered_source'] = df_sess['Contact ID'].map(source_map)
    df_sess['contact_name'] = df_sess['Contact ID'].map(
        name_map
    ).fillna("Unknown")

    # 2. CLEANING & CONVERSION
    df_sess['mpesa_amount'] = pd.to_numeric(
        df_sess['mpesa_amount'], errors='coerce'
    ).fillna(0)
    is_conv_str = df_sess['final_tags'].str.contains("Converted", na=False)
    df_sess['is_converted'] = (
        is_conv_str | (df_sess['mpesa_amount'] > 0)
    ).astype(int)

    if 'session_start' in df_sess.columns:
        df_sess['session_start'] = pd.to_datetime(df_sess['session_start'])
        df_sess = df_sess.sort_values(by=['Contact ID', 'session_start'])
# Maching session id in both the enriched and brand_mentions table
        df_sess['session_id'] = (
        df_sess['Contact ID'].astype(str) + "_" + 
        df_sess['session_start'].dt.strftime('%Y-%m-%d %H:%M:%S')
    )

    df_sess['visit_rank'] = df_sess.groupby('Contact ID').cumcount() + 1
    df_sess['channel_name'] = pd.to_numeric(
        df_sess.get('Channel ID'), errors='coerce'
    ).map(CHANNEL_MAP).fillna("Unknown")

    def classify_source(row):
        s = str(row.get('recovered_source', '')).lower()
        if any(x in s for x in ['ctc', 'ads', 'paid']):
            return 'Paid Ads'
        if any(x in s for x in ['user', 'contact']):
            return 'Organic / Direct'
        return 'Inbound / Unknown'

    df_sess['acquisition_source'] = df_sess.apply(classify_source, axis=1)

    # 3. TAG SPLITTING (ZONES / SECONDARY / FUNNELS / CATEGORIES)
    def split_tags(ts):
        if not isinstance(ts, str):
            return "", "", "", ""
        
        # Split the pipe-separated string into a list
        raw = [t.strip() for t in ts.split("|") if t.strip()]
        
        primary_z, secondary_z, funnel, other = [], [], [], []
        
        for t in raw:
            tl = t.lower()
            
            # 1. Catch Secondary Zones: Remove the prefix and keep the value
            if "secondary zone:" in tl:
                clean_val = re.sub(r"secondary zone:\s*", "", t, flags=re.I).strip()
                secondary_z.append(clean_val)
            
            # 2. Catch Primary Zones: Remove the prefix and keep the value
            elif "zone:" in tl:
                clean_val = re.sub(r"zone:\s*", "", t, flags=re.I).strip()
                primary_z.append(clean_val)
                
            # 3. Catch Funnel & Concerns
            elif any(x in tl for x in ["price", "payment", "converted", "objection"]):
                clean_val = re.sub(r"(funnel|concern):\s*", "", t, flags=re.I).strip()
                funnel.append(clean_val)
            
            # 4. Everything else (Brand Names / Categories)
            else:
                other.append(t.strip())
        
        # Join the cleaned lists back into strings
        return " | ".join(primary_z), " | ".join(secondary_z), " | ".join(funnel), " | ".join(other)

    # Apply the split to create the clean columns
    res = df_sess['final_tags'].apply(split_tags)
    
    df_sess['zone_name'] = res.apply(lambda x: x[0])       # e.g. "Runda"
    df_sess['secondary_zones'] = res.apply(lambda x: x[1]) # e.g. "Kiambu Road"
    df_sess['funnel_history'] = res.apply(lambda x: x[2])  # e.g. "Price Quoted"
    df_sess['final_tags'] = res.apply(lambda x: x[3])      # e.g. "CeraVe"

    # 4. METADATA (STOCK / SKIN / RECS)
    def get_meta(ts):
        tl = str(ts).lower()
        # Count flags for Dashboard Page 2, Part 3
        rf = [tl.count("stock"), tl.count("skin"), tl.count("recommendation")]
        
        # Identify brands from the master list
        brands = [b for b in BRAND_LIST if b.lower() in tl]
        
        # THE FIX: Convert list to a clean string "Brand A | Brand B"
        # This removes the Python brackets [] and quotes ''
        brand_string = " | ".join(brands) if brands else ""
        
        b_data = BRAND_MAP.get(brands[0].lower()) if brands else None
        macro = b_data['macro'] if b_data else "Other"
        sub = b_data['sub'] if b_data else "General"
        
        return (*rf, brand_string, macro, sub)

    # Applying the metadata extraction
    meta = df_sess['final_tags'].apply(get_meta)
    (df_sess['is_stock_inquiry'], df_sess['is_consultation'],
     df_sess['is_recommendation'], df_sess['temp_brands'],
     df_sess['primary_category'], df_sess['sub_category']) = (
        meta.apply(lambda x: x[0]), meta.apply(lambda x: x[1]),
        meta.apply(lambda x: x[2]), meta.apply(lambda x: x[3]),
        meta.apply(lambda x: x[4]), meta.apply(lambda x: x[5]))

    # 5. RFV, STATUS & BRACKETS
    stats = df_sess.groupby('Contact ID').agg({
        'session_id': 'count', 'mpesa_amount': 'sum', 'is_converted': 'sum',
        'session_start': 'max', 'contact_name': 'first'
    }).rename(columns={
        'session_id': 'frequency', 'mpesa_amount': 'monetary_value',
        'is_converted': 'total_orders', 'session_start': 'last_seen'
    })

    # Calculate the very first time we ever saw this customer
    first_seen = df_sess.groupby('Contact ID')['session_start'].min()
    stats['first_seen'] = stats.index.map(first_seen)

    # Calculate days from first message to the last purchase seen in this period
    stats['days_to_convert'] = (stats['last_seen'] - stats['first_seen']).dt.days

    # Handle the "Impulse Buy" logic
    stats['conversion_speed'] = np.where(stats['days_to_convert'] <= 0, 
                                         "Impulse (Same Day)",
                                np.where(stats['days_to_convert'] <= 7, 
                                         "Research (1 Week)", 
                                         "Consultative (8+ Days)"))

    stats['aov'] = np.where(
        stats['total_orders'] > 0,
        stats['monetary_value'] / stats['total_orders'], 0
    )
    stats['customer_status'] = np.where(
        stats['frequency'] > 1, "Returning", "New"
    )

    def get_bracket(v):
        if v <= 0: return "No Spend"
        if v <= 7000: return "0-7k"
        if v <= 20000: return "7k-20k"
        return "20k+"
    stats['payment_bracket'] = stats['monetary_value'].apply(get_bracket)

    def get_tier(row):
        val = row['monetary_value']
        
        # 20k+ -> Platinum (V.I.P / High-LTV)
        if val > 20000:
            return "Platinum"
        # 7k - 20k -> Gold (Loyal / Core)
        if val > 7000:
            return "Gold"
        # 0 - 7k -> Silver (Standard / Entry)
        return "Silver"

    stats['customer_tier'] = stats.apply(get_tier, axis=1)

    df_sess = df_sess.merge(
        stats[['customer_tier', 'customer_status', 'payment_bracket', 
               'days_to_convert', 'conversion_speed']], 
        on='Contact ID', how='left'
    )

    # 6. EXPORTS
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    df_sess.to_csv(
        PROCESSED_DATA_DIR / "fact_sessions_enriched.csv", index=False
    )
    stats.reset_index().to_csv(
        PROCESSED_DATA_DIR / "dim_customers_rfv.csv", index=False
    )

    f_rows = []
    for _, r in df_sess.iterrows():
        base = {
            "id": r['Contact ID'], "name": r['contact_name'],
            "date": r['session_start'], "cat": r['primary_category'],
            "src": r['acquisition_source']
        }
        f_rows.append({**base, "stage": "Inquiry", "val": 1})
        hist = str(r['funnel_history'])
        if "Price Quoted" in hist:
            f_rows.append({**base, "stage": "Price Quoted", "val": 1})
        if "Price Objection" in hist:
            f_rows.append({**base, "stage": "Price Objection", "val": 1})
        if r['is_converted']:
            f_rows.append({**base, "stage": "Converted", "val": 1})
        if r['customer_tier'] == "High-LTV Customer":
            f_rows.append({**base, "stage": "High Val Cust", "val": 1})

    pd.DataFrame(f_rows).to_csv(
        PROCESSED_DATA_DIR / "fact_funnel_analytics.csv", index=False
    )

    # ==========================================
    # 7. BRAND EXPLOSION (For Accurate Looker Charts)
    # ==========================================
    b_rows = []
    for _, r in df_sess.iterrows():
        raw_brands = re.split(r'\s*[,|]\s*', str(r['temp_brands']))
        cleaned_brands = [b.strip() for b in raw_brands if b.strip()]
        
        full_revenue = r['mpesa_amount']
        
        if not cleaned_brands:
            b_rows.append({
                "session_id": f"{r['Contact ID']}_{r['session_start']}",
                "Contact ID": r['Contact ID'],
                "brand_name": "General Inquiry", # Better than blank for charts
                "revenue" : r['mpesa_amount'],
                "is_converted": r['is_converted'],
                "customer_tier": r['customer_tier'],
                "acquisition_source": r['acquisition_source'],
                "date": r['session_start']
            })
        else:
            for brand in cleaned_brands:
                b_rows.append({
                    "session_id": f"{r['Contact ID']}_{r['session_start']}",
                    "Contact ID": r['Contact ID'],
                    "brand_name": brand,
                    "revenue" : r["mpesa_amount"],
                    "is_converted": r['is_converted'],
                    "customer_tier": r['customer_tier'],
                    "acquisition_source": r['acquisition_source'],
                    "date": r['session_start']
                })

    if b_rows:
        pd.DataFrame(b_rows).to_csv(PROCESSED_DATA_DIR / 
                                    "fact_brand_mentions.csv", index=False)
        print("✅ Brand Analytics exported (fact_brand_mentions.csv)")


    print("✅ ANALYTICS COMPLETE...")


if __name__ == "__main__":
    run_analytics_pipeline()