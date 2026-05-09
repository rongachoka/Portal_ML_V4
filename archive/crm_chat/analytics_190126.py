import pandas as pd
import numpy as np
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from pathlib import Path

# V3 PRODUCTION IMPORTS
from Portal_ML_V4.src.config.settings import (
    FINAL_TAGGED_DATA,
    PROCESSED_DATA_DIR,
    CLEANED_DATA_DIR,
    BASE_DIR
)
from Portal_ML_V4.src.config.brands import BRAND_LIST
from Portal_ML_V4.src.config.tag_rules import (
    CANONICAL_CATEGORY_RULES,
    CONCERN_RULES
)
from Portal_ML_V4.src.config.department_map import DEPARTMENT_TO_CANONICAL

# --- CONFIGURATIONS ---
CHANNEL_MAP = {
    389017: 'WhatsApp',
    387986: 'Instagram',
    388255: 'Facebook',
    388267: 'TikTok',
    389086: 'Web Chat'
}

# 1. MACRO MAP
MACRO_GROUP_MAP = {
    "Skincare": "Skincare",
    "Baby Care": "Baby Care",
    "Hair Care": "Hair Care",
    "Haircare": "Hair Care",
    "Medicine": "Medicine & Health",
    "Medical Devices and Kits": "Medicine & Health",
    "Homeopathy": "Medicine & Health",
    "Supplements": "Supplements & Nutrition",
    "Oral Care": "Oral Care",
    "Men Care": "Personal Care",
    "Menscare": "Personal Care",
    "Women's Health": "Personal Care",
    "Womens Health": "Personal Care",
    "Perfumes": "Personal Care",
    "Stanley Cups": "Personal Care"
}

# 2. STRICT CATEGORIES
STRICT_CATEGORIES = {
    'skincare': 'Skincare', 'skin': 'Skincare',
    'haircare': 'Haircare', 'hair': 'Haircare',
    'supplements': 'Supplements', 'vitamin': 'Supplements',
    'medicine': 'Medicine', 'medication': 'Medicine',
    'homeopathy': 'Homeopathy', 'menscare': 'Men Care',
    'womens health': "Women's Health", 'babycare': 'Baby Care',
    'perfumes': 'Perfumes', 'oral': 'Oral Care', 'dental': 'Oral Care'
}


# ==========================================
# 1. KNN PRODUCT MATCHER
# ==========================================
class ProductMatcher:
    def __init__(self):
        path = BASE_DIR / "data" / "01_raw" / "Products 31 Jan.csv"
        if not os.path.exists(path):
            self.active = False
            return

        self.df = pd.read_csv(
            path, encoding='utf-8', on_bad_lines='skip'
        ).fillna("")
        self.df.columns = self.df.columns.str.strip()

        rename_map = {
            'Item Name': 'Product Name', 'Description': 'Product Name',
            'Product': 'Product Name', 'Item Description': 'Product Name',
            'Name': 'Product Name', 'Item': 'Product Name',
            'Category 1': 'Raw_A', 'Category 2': 'Raw_B'
        }
        self.df.rename(columns=rename_map, inplace=True)

        if 'Product Name' not in self.df.columns:
            candidates = [
                c for c in self.df.columns
                if any(x in c.lower() for x in ['desc', 'name', 'product'])
            ]
            if candidates:
                self.df.rename(
                    columns={candidates[0]: 'Product Name'}, inplace=True
                )
            else:
                self.active = False
                return

        clean_data = self.df.apply(self._smart_clean_row, axis=1)
        self.df['Brand'] = clean_data.apply(lambda x: x[0])
        self.df['Category'] = clean_data.apply(lambda x: x[1])

        self.df['search_text'] = (
            self.df['Brand'] + " " + self.df['Product Name']
        ).str.lower()

        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        self.tfidf_matrix = self.vectorizer.fit_transform(
            self.df['search_text']
        )
        self.knn = NearestNeighbors(
            n_neighbors=1, metric='cosine'
        ).fit(self.tfidf_matrix)
        self.active = True

    def _detect_category_from_rules(self, text):
        if not isinstance(text, str) or not text.strip():
            return None
        
        t_lower = text.lower()
        for cat_key, rules in CANONICAL_CATEGORY_RULES.items():
            patterns = (
                rules.get('category', []) +
                rules.get('product', []) +
                rules.get('all', [])
            )
            for pat in patterns:
                if re.search(pat, t_lower):
                    return cat_key.replace("Product Inquiry - ", "").strip()
        return None

    def _smart_clean_row(self, row):
        val_a = str(row.get('Raw_A', '')).strip()
        val_b = str(row.get('Raw_B', '')).strip()

        cat_a = self._detect_category_from_rules(val_a)
        cat_b = self._detect_category_from_rules(val_b)

        if cat_a and not cat_b:
            return val_b, cat_a
        if cat_b and not cat_a:
            return val_a, cat_b
        
        # Brand Fallback
        def is_brand(txt):
            t_low = txt.lower()
            if any(b.lower() in t_low for b in BRAND_LIST):
                return True
            for pattern in DEPARTMENT_TO_CANONICAL.keys():
                if re.search(pattern, txt, re.IGNORECASE):
                    return True
            return False

        if is_brand(val_a) and not is_brand(val_b):
            return val_a, "Other"
        if is_brand(val_b) and not is_brand(val_a):
            return val_b, "Other"
            
        return val_a, "Other"

    def get_best_match(self, context, price=0):
        if not self.active:
            return "Unknown", "Unknown", 0, "Other"
        try:
            query_vec = self.vectorizer.transform([str(context).lower()])
            dist, idx = self.knn.kneighbors(query_vec)
            match = self.df.iloc[idx[0][0]]

            similarity = 1 - dist[0][0]
            cat_price = float(match.get('Price', 0)) \
                if match.get('Price') else 0
            cat_category = match.get('Category', 'Other')

            if similarity > 0.85:
                return (
                    match['Product Name'], match['Brand'],
                    cat_price, cat_category
                )
            elif price > 0 and cat_price > 0:
                min_allowed = cat_price * 0.90
                max_allowed = (cat_price * 1.10) + 600
                if min_allowed <= price <= max_allowed:
                    return (
                        match['Product Name'], match['Brand'],
                        cat_price, cat_category
                    )
        except Exception:
            pass
        return "Unknown", "Unknown", 0, "Other"


# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def clean_id(val):
    if pd.isna(val): return None
    s = str(val).strip()
    return s[:-2] if s.endswith('.0') else s

def find_id_col(df):
    cands = ['Contact ID', 'ContactID', 'ID', 'id', 'contact_id']
    return next((c for c in cands if c in df.columns), None)

def split_tags_logic(ts):
    if not isinstance(ts, str): return "", "", "", ""
    raw = [t.strip() for t in ts.split("|") if t.strip()]
    primary_z, secondary_z, funnel, other = [], [], [], []
    for t in raw:
        tl = t.lower()
        if "secondary zone:" in tl:
            secondary_z.append(re.sub(r"secondary zone:\s*", "", t, flags=re.I).strip())
        elif "zone:" in tl:
            primary_z.append(re.sub(r"zone:\s*", "", t, flags=re.I).strip())
        elif any(x in tl for x in ["price", "payment", "converted"]):
            funnel.append(re.sub(r"(funnel|concern):\s*", "", t, flags=re.I).strip())
        else:
            other.append(t.strip())
    return " | ".join(primary_z), " | ".join(secondary_z), " | ".join(funnel), " | ".join(other)

def get_category_from_brand_map(brand_name):
    if not brand_name or brand_name == "Unknown": return None
    for pattern, category in DEPARTMENT_TO_CANONICAL.items():
        if re.search(pattern, brand_name, re.IGNORECASE):
            return category.replace("Product Inquiry - ", "").strip()
    return None


# ==========================================
# 3. MAIN ANALYTICS PIPELINE
# ==========================================
def run_analytics_pipeline():
    print("📊 V6.2 ANALYTICS: FUNNEL SORT & SESSION LINKING")
    if not os.path.exists(FINAL_TAGGED_DATA): return

    df_sess = pd.read_parquet(FINAL_TAGGED_DATA)
    matcher = ProductMatcher()

    # --- A. CLEANING ---
    df_sess['Contact ID'] = df_sess['Contact ID'].apply(clean_id)
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
    df_sess['contact_name'] = df_sess['Contact ID'].map(name_map).fillna("Unknown")

    # --- B. CONVERSION ---
    df_sess['mpesa_amount'] = pd.to_numeric(df_sess['mpesa_amount'], 
                                            errors='coerce').fillna(0)
    df_sess['is_converted'] = (df_sess['final_tags'].str.contains("Converted", 
                                                                  na=False) | (df_sess['mpesa_amount'] > 0)).astype(int)
    
    if 'session_start' in df_sess.columns:
        df_sess['session_start'] = pd.to_datetime(df_sess['session_start'])
        df_sess = df_sess.sort_values(by=['Contact ID', 'session_start'])
        df_sess['session_id'] = (df_sess['Contact ID'].astype(str) + "_" + df_sess['session_start'].dt.strftime('%Y-%m-%d %H:%M:%S'))

    df_sess['visit_rank'] = df_sess.groupby('Contact ID').cumcount() + 1
    df_sess['channel_name'] = pd.to_numeric(df_sess.get('Channel ID'), errors='coerce').map(CHANNEL_MAP).fillna("Unknown")

    def classify_source(row):
        s = str(row.get('recovered_source', '')).lower()
        if any(x in s for x in ['ctc', 'ads', 'paid']): return 'Paid Ads'
        if any(x in s for x in ['user', 'contact']): return 'Organic / Direct'
        return 'Inbound / Unknown'
    df_sess['acquisition_source'] = df_sess.apply(classify_source, axis=1)

    # --- C. ENRICHMENT ---
    print("🧠 Running AI Enrichment...")
    res_split = df_sess['final_tags'].apply(split_tags_logic)
    df_sess['zone_name'], df_sess['secondary_zones'] = res_split.apply(lambda x: x[0]), res_split.apply(lambda x: x[1])
    df_sess['funnel_history'] = res_split.apply(lambda x: x[2])

    def extract_meta(row):
        tags = str(row['final_tags']).lower()
        stock, skin, rec = tags.count("stock"), tags.count("skin"), tags.count("recommendation")
        found = [b for b in BRAND_LIST if b.lower() in tags]
        return stock, skin, rec, " | ".join(found) if found else ""

    meta_res = df_sess.apply(extract_meta, axis=1)
    df_sess['is_stock_inquiry'] = meta_res.apply(lambda x: x[0])
    df_sess['is_consultation'] = meta_res.apply(lambda x: x[1])
    df_sess['is_recommendation'] = meta_res.apply(lambda x: x[2])
    df_sess['temp_brands'] = meta_res.apply(lambda x: x[3])

    def process_ai(row):
        p_name, p_brand, p_price, p_cat = matcher.get_best_match(row['full_context'], row['mpesa_amount'])
        
        sub_cat = "General"
        brand_cat = get_category_from_brand_map(p_brand)
        
        if brand_cat: sub_cat = brand_cat
        elif p_cat != "Other": sub_cat = p_cat
        else:
            tags = str(row['final_tags'])
            found_cats = [c.replace("Product Inquiry - ", "").strip() for c in CANONICAL_CATEGORY_RULES.keys() if c in tags]
            if found_cats: sub_cat = found_cats[0]

        primary_cat = MACRO_GROUP_MAP.get(sub_cat, "Other")
        if sub_cat == "General" and primary_cat == "Other": primary_cat = "General Inquiry"

        return primary_cat, sub_cat, p_name, p_brand

    res_ai = df_sess.apply(process_ai, axis=1)
    df_sess['primary_category'] = res_ai.apply(lambda x: x[0])
    df_sess['sub_category'] = res_ai.apply(lambda x: x[1])
    df_sess['matched_product'] = res_ai.apply(lambda x: x[2])
    df_sess['matched_brand'] = res_ai.apply(lambda x: x[3])

    # --- D. RFV ---
    stats = df_sess.groupby('Contact ID').agg({
        'session_id': 'count', 'mpesa_amount': 'sum', 'is_converted': 'sum', 
        'session_start': 'max'
    }).rename(columns={'session_id': 'frequency', 'mpesa_amount': 'monetary_value', 
                       'is_converted': 'total_orders', 'session_start': 'last_seen'})
    
    first_seen = df_sess.groupby('Contact ID')['session_start'].min()
    stats['first_seen'] = stats.index.map(first_seen)
    stats['days_to_convert'] = (stats['last_seen'] - stats['first_seen']).dt.days
    stats['conversion_speed'] = np.where(stats['days_to_convert'] <= 0, "Impulse (Same Day)", 
                                np.where(stats['days_to_convert'] <= 7, "Research (1 Week)", "Consultative (8+ Days)"))
    stats['customer_status'] = np.where(stats['frequency'] > 1, "Returning", "New")
    
    def get_tier(row):
        val = row['monetary_value']
        if val > 20000: return "Platinum"
        if val > 7000: return "Gold"
        return "Silver"
    stats['customer_tier'] = stats.apply(get_tier, axis=1)

    def get_bracket(v):
        if v <= 0: return "No Spend"
        if v <= 7000: return "0-7k"
        if v <= 20000: return "7k-20k"
        return "20k+"
    stats['payment_bracket'] = stats['monetary_value'].apply(get_bracket)

    df_sess = df_sess.merge(stats[['customer_tier', 'customer_status', 'payment_bracket', 'days_to_convert', 'conversion_speed']], on='Contact ID', how='left')

    # --- E. EXPORTS ---
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    df_sess.to_csv(PROCESSED_DATA_DIR / "fact_sessions_enriched.csv", index=False)
    stats.reset_index().to_csv(PROCESSED_DATA_DIR / "dim_customers_rfv.csv", index=False)

    # 1. FUNNEL ANALYTICS (FIXED: Added session_id and sort_order)
    f_rows = []
    for _, r in df_sess.iterrows():
        base = {
            "session_id": r['session_id'], # CRITICAL FIX
            "id": r['Contact ID'], "name": r['contact_name'],
            "date": r['session_start'], "cat": r['primary_category'],
            "src": r['acquisition_source']
        }
        hist = str(r['funnel_history'])
        
        # Order 1: Inquiry (Everyone)
        f_rows.append({**base, "stage": "Inquiry", "sort_order": 1, "val": 1})
        
        # Order 2: Price Quoted
        if "Price Quoted" in hist:
            f_rows.append({**base, "stage": "Price Quoted", "sort_order": 2, "val": 1})
            
        # Order 3: Price Objection
        if "Price Objection" in hist:
            f_rows.append({**base, "stage": "Price Objection", "sort_order": 3, "val": 1})
            
        # Order 4: Converted
        if r['is_converted']:
            f_rows.append({**base, "stage": "Converted", "sort_order": 4, "val": 1})
            
        # Order 5: High Value
        if r['customer_tier'] == "Platinum":
            f_rows.append({**base, "stage": "High Val Cust", "sort_order": 5, "val": 1})

    pd.DataFrame(f_rows).to_csv(PROCESSED_DATA_DIR / "fact_funnel_analytics.csv", index=False)

    b_rows = []
    for _, r in df_sess.iterrows():
        if r['matched_brand'] != "Unknown": brands_to_use = [r['matched_brand']]
        else:
            raw_brands = re.split(r'\s*[,|]\s*', str(r.get('temp_brands', '')))
            brands_to_use = [b.strip() for b in raw_brands if b.strip()]
        
        if not brands_to_use:
            b_rows.append({"session_id": r['session_id'], "Contact ID": r['Contact ID'], "brand_name": "General Inquiry", "revenue": r['mpesa_amount'], "is_converted": r['is_converted'], "date": r['session_start']})
        else:
            for brand in brands_to_use:
                b_rows.append({"session_id": r['session_id'], "Contact ID": r['Contact ID'], "brand_name": brand, "revenue": r['mpesa_amount'], "is_converted": r['is_converted'], "date": r['session_start']})
    if b_rows: pd.DataFrame(b_rows).to_csv(PROCESSED_DATA_DIR / "fact_brand_mentions.csv", index=False)

    zone_rows = []
    for _, r in df_sess.iterrows():
        if r.get('zone_name'): zone_rows.append({"session_id": r['session_id'], "type": "Primary", "loc": r['zone_name']})
        if r.get('secondary_zones'):
            for sz in str(r['secondary_zones']).split("|"):
                if sz.strip(): zone_rows.append({"session_id": r['session_id'], "type": "Secondary", "loc": sz.strip()})
    if zone_rows: pd.DataFrame(zone_rows).to_csv(PROCESSED_DATA_DIR / "fact_session_zones.csv", index=False)

    concern_rows = []
    for _, r in df_sess.iterrows():
        text_context = str(r['full_context']).lower()
        for concern, patterns in CONCERN_RULES.items():
            if any(re.search(pat, text_context) for pat in patterns.get('all', []) + patterns.get('chat', [])):
                concern_rows.append({"session_id": r['session_id'], "concern": concern, "date": r['session_start']})
    if concern_rows: pd.DataFrame(concern_rows).to_csv(PROCESSED_DATA_DIR / "fact_session_concerns.csv", index=False)

    print("✅ ANALYTICS COMPLETE: Funnel Sort + Session Links.")

if __name__ == "__main__":
    run_analytics_pipeline()