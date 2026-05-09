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
    DEPARTMENT_TO_CANONICAL, CATEGORY_TO_CANONICAL
)

# --- CONFIGURATIONS ---
CHANNEL_MAP = {
    389017: 'WhatsApp',
    387986: 'Instagram',
    388255: 'Facebook',
    388267: 'TikTok',
    389086: 'Web Chat'
}

# --- MACRO GROUP DEFINITION ---
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
    """
    Classifies CSV products into Macro/Sub groups using configs.
    """
    products_path = BASE_DIR / "data" / "01_raw" / "Products 31 Jan.csv"
    brand_cat_map = {}

    if os.path.exists(products_path):
        try:
            df_prod = pd.read_csv(
                products_path, encoding='utf-8', on_bad_lines='skip'
            )
            cols = ['Category 1', 'Category 2']
            if all(c in df_prod.columns for c in cols):
                mapping = df_prod[cols].dropna().drop_duplicates()
                for _, row in mapping.iterrows():
                    brand_key = str(row['Category 1']).strip().lower()
                    raw_cat_2 = str(row['Category 2']).strip().upper()

                    canonical_tag = next(
                        (v for k, v in DEPARTMENT_TO_CANONICAL.items()
                         if k in raw_cat_2), None
                    )

                    if not canonical_tag:
                        tags = enrich_canonical_categories_from_text(
                            str(row['Category 2']), None, source="category"
                        )
                        if tags:
                            canonical_tag = list(tags)[0]

                    if canonical_tag in MACRO_GROUP_MAP:
                        brand_cat_map[brand_key] = {
                            'macro': MACRO_GROUP_MAP[canonical_tag],
                            'sub': canonical_tag.replace(
                                "Product Inquiry - ", ""
                            )
                        }
                    else:
                        brand_cat_map[brand_key] = {
                            'macro': "General / Other",
                            'sub': "General"
                        }
        except Exception:
            return {}
    return brand_cat_map


def find_id_column(df):
    """Robustly identifies ID column across Respond.io exports."""
    candidates = ['Contact ID', 'ContactID', 'ID', 'id', 'contact_id']
    return next((c for c in candidates if c in df.columns), None)


def clean_id_format(val):
    """
    Forces ID to string and removes .0 usually found in float-to-string exports.
    Example: 12345.0 -> '12345'
    """
    if pd.isna(val): return None
    s = str(val).strip()
    if s.endswith('.0'):
        s = s[:-2]
    return s


def run_analytics_pipeline():
    print("📊 V4 ANALYTICS")
    if not os.path.exists(FINAL_TAGGED_DATA):
        print(f"❌ Error: {FINAL_TAGGED_DATA} not found.")
        return

    df_sess = pd.read_parquet(FINAL_TAGGED_DATA)
    BRAND_MAP = load_product_category_map()

    # ==========================================
    # 1. RECOVER SOURCE & NAMES
    # ==========================================
    cleaned_conv_path = CLEANED_DATA_DIR / "cleaned_conversations.csv"
    cleaned_cont_path = CLEANED_DATA_DIR / "cleaned_contacts.csv"
    
    source_map, name_map = {}, {}
    
    # Force Session IDs to string for perfect matching
    
    df_sess['Contact ID'] = df_sess['Contact ID'].astype(str).str.strip()

    if os.path.exists(cleaned_conv_path):
        df_c = pd.read_csv(cleaned_conv_path)
        id_col = find_id_column(df_c)
        src_col = next((c for c in df_c.columns if 'opened' in c.lower() 
                        and 'source' in c.lower()), None)
        if id_col and src_col:
            # FORCE TO STRING AND STRIP WHITESPACE
            df_c[id_col] = df_c[id_col].astype(str).str.strip()
            source_map = df_c.set_index(id_col)[src_col].to_dict()

    if os.path.exists(cleaned_cont_path):
        df_n = pd.read_csv(cleaned_cont_path)
        id_col = find_id_column(df_n)
        name_col = next((c for c in df_n.columns if 'name' in c.lower()), None)
        if id_col and name_col:
            # FORCE TO STRING AND STRIP WHITESPACE
            df_n[id_col] = df_n[id_col].astype(str).str.strip()
            name_map = df_n.set_index(id_col)[name_col].to_dict()

    df_sess['recovered_source'] = df_sess['Contact ID'].map(source_map)
    df_sess['contact_name'] = df_sess['Contact ID'].map(name_map).fillna("Unknown")

    # DEBUG PRINT: Check for failures
    unmapped_count = df_sess['recovered_source'].isna().sum()
    print(f"🔎 Source Mapping Report: {unmapped_count} sessions could not find a matching Conversation ID.")

    # ==========================================
    # 2. CLEANING & CONVERSION
    # ==========================================
    df_sess['mpesa_amount'] = pd.to_numeric(
        df_sess['mpesa_amount'], errors='coerce'
    ).fillna(0)

    is_conv_str = df_sess['final_tags'].str.contains("Converted", na=False)
    has_money = df_sess['mpesa_amount'] > 0
    df_sess['is_converted'] = (is_conv_str | has_money).astype(int)

    if 'session_start' in df_sess.columns:
        df_sess['session_start'] = pd.to_datetime(df_sess['session_start'])
        df_sess = df_sess.sort_values(by=['Contact ID', 'session_start'])

    df_sess['visit_rank'] = df_sess.groupby('Contact ID').cumcount() + 1

    # Attribution Logic
    def classify_source(row):
        src = str(row.get('recovered_source', '')).lower()
        if any(x in src for x in ['ctc', 'ads', 'paid']):
            return 'Paid Ads'
        if any(x in src for x in ['user', 'contact']):
            return 'Organic / Direct'
        return 'Inbound / Unknown'

    df_sess['acquisition_source'] = df_sess.apply(classify_source, axis=1)

    if 'Channel ID' in df_sess.columns:
        df_sess['channel_name'] = pd.to_numeric(
            df_sess['Channel ID'], errors='coerce'
        ).map(CHANNEL_MAP).fillna("Unknown")

    # ==========================================
    # 3. TAG SPLITTING
    # ==========================================
    def split_tags(tag_string):
        if not isinstance(tag_string, str):
            return "", "", ""
        raw = [t.strip() for t in tag_string.split("|") if t.strip()]
        z, f, p = [], [], []
        funnel_keywords = ["price", "payment", "converted"]
        for t in raw:
            t_low = t.lower()
            if "zone:" in t_low:
                z.append(re.sub(r"zone:\s*", "", t, flags=re.I).strip())
            elif any(x in t_low for x in funnel_keywords):
                f.append(re.sub(r"(funnel|concern):\s*", "", t, flags=re.I))
            else:
                p.append(t)
        return " | ".join(z), " | ".join(f), " | ".join(p)

    processed = df_sess['final_tags'].apply(split_tags)
    df_sess['zone_name'] = processed.apply(lambda x: x[0])
    df_sess['funnel_history'] = processed.apply(lambda x: x[1])
    df_sess['final_tags'] = processed.apply(lambda x: x[2])

    # ==========================================
    # 4. METADATA EXTRACTION
    # ==========================================
    def extract_meta(tag_string):
        t_low = str(tag_string).lower()
        res = [
            "stock" in t_low,
            "skin" in t_low,
            "recommendation" in t_low
        ]
        found_brands = [b for b in BRAND_LIST if b.lower() in t_low]

        macro, sub = "General", "General"
        if found_brands:
            brand_data = BRAND_MAP.get(found_brands[0].lower())
            if brand_data:
                macro = brand_data['macro']
                sub = brand_data['sub']
        return (*res, found_brands, macro, sub)

    meta = df_sess['final_tags'].apply(extract_meta)
    df_sess['is_stock_inquiry'] = meta.apply(lambda x: x[0])
    df_sess['is_consultation'] = meta.apply(lambda x: x[1])
    df_sess['is_recommendation'] = meta.apply(lambda x: x[2])
    df_sess['temp_brands'] = meta.apply(lambda x: x[3])
    df_sess['primary_category'] = meta.apply(lambda x: x[4])
    df_sess['sub_category'] = meta.apply(lambda x: x[5])

    # ==========================================
    # 5. RFV & TIERING
    # ==========================================
    cust_stats = df_sess.groupby('Contact ID').agg({
        'contact_name': 'first',
        'session_start': 'max',
        'session_id': 'count',
        'mpesa_amount': 'sum',
        'is_converted': 'sum'
    }).rename(columns={
        'session_start': 'last_seen',
        'session_id': 'frequency',
        'mpesa_amount': 'monetary_value',
        'is_converted': 'total_orders'
    })

    now = pd.Timestamp.now()
    cust_stats['recency_days'] = (now - cust_stats['last_seen']).dt.days
    cust_stats['aov'] = np.where(
        cust_stats['total_orders'] > 0,
        cust_stats['monetary_value'] / cust_stats['total_orders'],
        0
    )

    def get_tier(row):
        if row['monetary_value'] >= HIGH_LTV_SUM_THRESHOLD:
            return "High-LTV Customer"
        if row['total_orders'] >= LOYAL_PAYMENTS_THRESHOLD:
            return "Loyal Customer"
        return "Returning Customer" if row['frequency'] > 1 else "New Customer"

    cust_stats['customer_tier'] = cust_stats.apply(get_tier, axis=1)
    df_sess = df_sess.merge(
        cust_stats[['customer_tier']], on='Contact ID', how='left'
    )

    # ==========================================
    # 6. FINAL EXPORTS
    # ==========================================
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    df_sess.to_csv(
        PROCESSED_DATA_DIR / "fact_sessions_enriched.csv", index=False
    )
    cust_stats.reset_index().to_csv(
        PROCESSED_DATA_DIR / "dim_customers_rfv.csv", index=False
    )

    funnel_rows = []
    for _, row in df_sess.iterrows():
        base = {
            "contact_id": row['Contact ID'],
            "name": row['contact_name'],
            "date": row['session_start'],
            "brand": row['temp_brands'][0] if row['temp_brands'] else "General",
            "category": row['primary_category'],
            "sub_category": row['sub_category'],
            "source": row['acquisition_source']
        }
        funnel_rows.append({**base, "stage": "Inquiry", "val": 1})
        hist = str(row['funnel_history'])
        if "Price Quoted" in hist:
            funnel_rows.append({**base, "stage": "Price Quoted", "val": 1})
        if "Price Objection" in hist:
            funnel_rows.append({**base, "stage": "Price Objection", "val": 1})
        if row['is_converted']:
            funnel_rows.append({**base, "stage": "Converted", "val": 1})
        if row['customer_tier'] == "High-LTV Customer":
            funnel_rows.append({**base, "stage": "High Val Cust", "val": 1})

    pd.DataFrame(funnel_rows).to_csv(
        PROCESSED_DATA_DIR / "fact_funnel_analytics.csv", index=False
    )
    print("✅ ANALYTICS COMPLETE.")


if __name__ == "__main__":
    run_analytics_pipeline()