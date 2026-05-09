import pandas as pd
import numpy as np
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from pathlib import Path

# V3 PRODUCTION IMPORTS
from Portal_ML_V4.src.config.settings import (
    FINAL_TAGGED_DATA, PROCESSED_DATA_DIR, CLEANED_DATA_DIR, BASE_DIR
)
from Portal_ML_V4.src.config.brands import BRAND_LIST
from Portal_ML_V4.src.config.tag_rules import CANONICAL_CATEGORY_RULES, CONCERN_RULES
from Portal_ML_V4.src.config.department_map import DEPARTMENT_TO_CANONICAL

# --- CONFIGURATIONS ---
CHANNEL_MAP = {
    389017: 'WhatsApp', 387986: 'Instagram', 388255: 'Facebook', 
    388267: 'TikTok', 389086: 'Web Chat'
}

# 1. MACRO MAP (1-to-1 Mapping)
MACRO_GROUP_MAP = {
    "Skincare": "Skincare",
    "Baby Care": "Baby Care",
    "Haircare": "Hair Care",
    "Oral Care": "Oral Care",
    "Supplements": "Supplements",
    "Medicine": "Medicine", 
    "Medical Devices": "Medical Devices & Kits",
    "First Aid": "Medical Devices & Kits",      
    "Homeopathy": "Homeopathy",
    "Men's Care": "Men Care",
    "Women's Health": "Women's Health",
    "Fragrance": "Perfumes",
    "Lip Care": "Lip Care",
    "Sexual Health": "Sexual Health",
    "Hair Care": "Hair Care", "Menscare": "Men Care", "Perfumes": "Perfumes"
}

# ✅ FIX: REORDERED RULES (Skincare BEFORE Supplements)
SUB_CAT_RULES_ORDERED = {
    'Serum & Treatment': ['serum', 'concentrate', 'ampoule', 'spot treatment', 'peel', 'acid', 'retinol', 'niacinamide'],
    'Sunscreen': ['sunscreen', 'spf', 'sunblock', 'uv'],
    'Moisturizer': ['moisturizer', 'cream', 'lotion', 'balm', 'gel'],
    'Cleanser': ['cleanser', 'face wash', 'facewash', 'soap', 'scrub'],
    'Toner': ['toner', 'astringent', 'essence', 'mist'],
    'Vitamins': ['vitamin', 'multivitamin', 'zinc', 'iron', 'calcium', 'magnesium', 'biotin'],
    'Shampoo': ['shampoo', 'wash'],
    'Conditioner': ['conditioner', 'leave-in'],
    'Pain Relief': ['pain', 'panadol', 'paracetamol'],
}

# ==========================================
# 1. KNN PRODUCT MATCHER (BUNDLE-AWARE)
# ==========================================
class ProductMatcher:
    def __init__(self):
        path = BASE_DIR / "data" / "01_raw" / "Final_Knowledge_Base_PowerBI.csv"
        if not os.path.exists(path):
            self.active = False
            return

        self.df = pd.read_csv(path).fillna("")
        
        # ✅ UPGRADE: Boost 'Name' weight
        # We repeat 'Name' 3 times to make "Blemish Control Gel" much stronger than "Cleanser"
        self.df['search_text'] = (
            self.df['Brand'] + " " + 
            self.df['Name'] + " " + self.df['Name'] + " " + self.df['Name'] + " " + 
            self.df['Sub_Category'] + " " + 
            self.df['Concerns'] + " " + 
            self.df['Detailed_Desc']
        ).str.lower()

        self.vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 4), min_df=1, strip_accents='unicode')
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['search_text'])
        self.knn = NearestNeighbors(n_neighbors=15, metric='cosine').fit(self.tfidf_matrix) # Expanded to 15 to find specific items
        self.active = True
        print(f"   🧠 Knowledge Base Loaded: {len(self.df)} products")

    def _extract_potential_price(self, text):
        matches = re.findall(r'(?:ksh|kes)?\s?(\d{1,3}(?:,\d{3})*)\s?(?:/-)?', str(text), re.IGNORECASE)
        valid_prices = []
        for m in matches:
            try:
                val = float(m.replace(',', ''))
                if 500 < val < 150000: valid_prices.append(val)
            except: continue
        return valid_prices

    def get_best_match(self, context, mpesa_amount=0, brand_hint=None):
        if not self.active: return None
        try:
            # 1. PREPARE QUERY
            query_text = str(context).lower()
            # Inject brand hint to guide the vector
            if brand_hint and brand_hint != "Unknown":
                query_text = f"{brand_hint} {brand_hint} {query_text}" 

            query_vec = self.vectorizer.transform([query_text])
            dist, idx = self.knn.kneighbors(query_vec)
            
            # Holders for strategies
            price_match_candidate = None
            price_match_score = 0
            
            text_match_candidate = None
            text_match_score = 0
            
            target_prices = [mpesa_amount] if mpesa_amount > 0 else self._extract_potential_price(context)

            for i in range(len(idx[0])):
                match_idx = idx[0][i]
                similarity = 1 - dist[0][i]
                match_row = self.df.iloc[match_idx]
                brand_name = str(match_row['Brand']).strip()
                
                # --- FILTER: Strict Brand Check ---
                # If we know the brand is CeraVe, ignore Anua matches completely
                if brand_hint and brand_hint != "Unknown":
                    b_hint_clean = brand_hint.lower().replace("the ", "").strip()
                    b_row_clean = brand_name.lower().replace("the ", "").strip()
                    if b_hint_clean not in b_row_clean and b_row_clean not in b_hint_clean:
                        continue # Skip wrong brands entirely

                # --- STRATEGY A: Price Verification (The Safe Match) ---
                cat_price = float(match_row.get('Price', 0))
                is_price_valid = False
                if cat_price > 0:
                    for p in target_prices:
                        if p == 0: continue
                        # Range: 90% of price to 110% + 600 (delivery)
                        if (cat_price * 0.90) <= p <= (cat_price * 1.10 + 600):
                            is_price_valid = True
                            break
                
                if is_price_valid:
                    if similarity > price_match_score:
                        price_match_score = similarity
                        price_match_candidate = match_row

                # --- STRATEGY B: Pure Text Match (The Bundle/Override Match) ---
                # We track the best text match regardless of price
                if similarity > text_match_score:
                    text_match_score = similarity
                    text_match_candidate = match_row

            # 🏆 DECISION TIME 🏆
            
            # 1. Prefer Price Match if score is decent (> 0.35)
            if price_match_candidate is not None and price_match_score > 0.35:
                return price_match_candidate
            
            # 2. If Price Match Failed (e.g. Bundle), use High-Confidence Text Match
            # We require a HIGHER threshold (0.60) to trust text alone without price
            if text_match_candidate is not None and text_match_score > 0.60:
                # "CeraVe Blemish Control Gel" is a specific name, so it usually scores > 0.65
                return text_match_candidate

        except Exception: pass
        return None

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

def extract_meta(row):
    tags = str(row['final_tags']).lower()
    stock, skin, rec = tags.count("stock"), tags.count("skin"), tags.count("recommendation")
    found = [b for b in BRAND_LIST if b.lower() in tags]
    return stock, skin, rec, " | ".join(found) if found else ""

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
    print("📊 V7.5 ANALYTICS: ROBUST FALLBACK LOGIC")
    if not os.path.exists(FINAL_TAGGED_DATA): return

    df_sess = pd.read_parquet(FINAL_TAGGED_DATA)
    matcher = ProductMatcher()

    # --- A. CLEANING & PREP ---
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
    
    # Calculate is_converted immediately
    df_sess['mpesa_amount'] = pd.to_numeric(df_sess['mpesa_amount'], errors='coerce').fillna(0)
    df_sess['is_converted'] = (df_sess['final_tags'].str.contains("Converted", na=False) | (df_sess['mpesa_amount'] > 0)).astype(int)

    if 'session_start' in df_sess.columns:
        df_sess['session_start'] = pd.to_datetime(df_sess['session_start'])
        df_sess = df_sess.sort_values(by=['Contact ID', 'session_start'])
        df_sess['session_id'] = (df_sess['Contact ID'].astype(str) + "_" + df_sess['session_start'].dt.strftime('%Y-%m-%d %H:%M:%S'))

    df_sess = df_sess.sort_values(by=['Contact ID', 'session_start'])
    df_sess['visit_rank'] = df_sess.groupby('Contact ID').cumcount() + 1
    df_sess['prior_orders'] = df_sess.groupby('Contact ID')['is_converted'].cumsum().shift(1).fillna(0)
    df_sess['customer_status'] = np.where(df_sess['prior_orders'] == 0, "New", "Returning")
    df_sess['channel_name'] = pd.to_numeric(df_sess.get('Channel ID'), errors='coerce').map(CHANNEL_MAP).fillna("Unknown")

    def classify_source(row):
        s = str(row.get('recovered_source', '')).lower()
        if any(x in s for x in ['ctc', 'ads', 'paid']): return 'Paid Ads'
        if any(x in s for x in ['user', 'contact']): return 'Organic / Direct'
        return 'Inbound / Unknown'
    df_sess['acquisition_source'] = df_sess.apply(classify_source, axis=1)

    # --- B. ENRICHMENT ---
    print("🧠 Running AI Enrichment...")
    res_split = df_sess['final_tags'].apply(split_tags_logic)
    df_sess['zone_name'], df_sess['secondary_zones'] = res_split.apply(lambda x: x[0]), res_split.apply(lambda x: x[1])
    df_sess['funnel_history'] = res_split.apply(lambda x: x[2])

    meta_res = df_sess.apply(extract_meta, axis=1)
    df_sess['is_stock_inquiry'] = meta_res.apply(lambda x: x[0])
    df_sess['is_consultation'] = meta_res.apply(lambda x: x[1])
    df_sess['is_recommendation'] = meta_res.apply(lambda x: x[2])
    df_sess['temp_brands'] = meta_res.apply(lambda x: x[3])

    def process_ai(row):
        raw_brands = str(row['temp_brands']).split("|")
        brand_candidates = [b.strip() for b in raw_brands if b.strip() and b.strip() != "Unknown"]
        
        match_row = None
        # Strategy 1: Brand Hint
        if brand_candidates:
            for brand in brand_candidates:
                match_row = matcher.get_best_match(row['full_context'], row['mpesa_amount'], brand_hint=brand)
                if match_row is not None: break
        
        # Strategy 2: Global Match
        if match_row is None:
             match_row = matcher.get_best_match(row['full_context'], row['mpesa_amount'], brand_hint=None)

        if match_row is not None:
            # ✅ SUCCESS: Found specific product
            p_name = match_row['Name']
            p_brand = match_row['Brand']
            p_cat = match_row['Canonical_Category']
            p_sub = match_row['Sub_Category']
            p_concern = match_row['Concerns']
            p_audience = match_row['Target_Audience']
        else:
            # ❌ MATCH FAILED -> SAFETY NET RECOVERY
            # If we know the brand from temp_brands, FORCE it.
            if brand_candidates:
                p_brand = brand_candidates[0] # Take the first detected brand
                p_name = f"General {p_brand} Inquiry"
                
                # Recover Category from Brand Map
                cat_from_brand = get_category_from_brand_map(p_brand)
                p_cat = cat_from_brand if cat_from_brand else "General Inquiry"
            else:
                p_name, p_brand = "Unknown", "Unknown"
                p_cat = "General Inquiry"

            p_concern, p_audience = "General", "General"
            p_sub = "General"
            
            # Recover Sub-Cat from Text Rules
            text_lower = row['full_context'].lower()
            for sub, keywords in SUB_CAT_RULES_ORDERED.items():
                if any(k in text_lower for k in keywords):
                    p_sub = sub
                    # If we still don't have a category, infer it from sub-cat
                    if p_cat == "General Inquiry":
                        if sub in ['Serum & Treatment', 'Moisturizer', 'Cleanser']: p_cat = "Skincare"
                        elif sub in ['Vitamins']: p_cat = "Supplements"
                    break

        primary_cat = MACRO_GROUP_MAP.get(p_cat, "General Inquiry") 
        return primary_cat, p_sub, p_name, p_brand, p_concern, p_audience

    print("🧠 Running Brand-Guided AI Enrichment...")
    res_ai = df_sess.apply(process_ai, axis=1)
    df_sess['primary_category'] = res_ai.apply(lambda x: x[0])
    df_sess['sub_category'] = res_ai.apply(lambda x: x[1])
    df_sess['matched_product'] = res_ai.apply(lambda x: x[2])
    df_sess['matched_brand'] = res_ai.apply(lambda x: x[3])
    df_sess['matched_concern'] = res_ai.apply(lambda x: x[4]) 
    df_sess['target_audience'] = res_ai.apply(lambda x: x[5]) 

    # --- C. CONVERSION SPEED ---
    print("⏳ Calculating Session-Based Conversion Speed...")
    df_sess = df_sess.sort_values(['Contact ID', 'session_start'])
    df_sess['prev_session'] = df_sess.groupby('Contact ID')['session_start'].shift(1)
    df_sess['days_since_last'] = (df_sess['session_start'] - df_sess['prev_session']).dt.days.fillna(9999)
    df_sess['is_new_journey'] = (df_sess['days_since_last'] > 30).astype(int)
    df_sess['journey_id'] = df_sess.groupby('Contact ID')['is_new_journey'].cumsum()
    journey_starts = df_sess.groupby(['Contact ID', 'journey_id'])['session_start'].transform('min')
    df_sess['session_days_to_convert'] = (df_sess['session_start'] - journey_starts).dt.days
    
    def categorize_speed(row):
        if row.get('is_converted', 0) == 0: return "Not Converted"
        days = row.get('session_days_to_convert', 99)
        if days <= 0: return "Impulse (Same Day)"
        if days <= 7: return "Research (1 Week)"
        return "Consultative (8+ Days)"
    
    df_sess['conversion_speed'] = df_sess.apply(categorize_speed, axis=1)

    # --- D. EXPORTS ---
    stats = df_sess.groupby('Contact ID').agg({
        'session_id': 'count', 'mpesa_amount': 'sum', 'is_converted': 'sum', 'session_start': 'max'
    }).rename(columns={'session_id': 'frequency', 'mpesa_amount': 'monetary_value', 'is_converted': 'total_orders', 'session_start': 'last_seen'})
    
    first_seen = df_sess.groupby('Contact ID')['session_start'].min()
    stats['first_seen'] = stats.index.map(first_seen)
    stats['days_to_convert'] = (stats['last_seen'] - stats['first_seen']).dt.days 
    stats['customer_status'] = np.where(stats['frequency'] > 1, "Returning", "New")
    

    def get_tier(row):
        val = row['monetary_value']
        if val > 20000: return "Platinum"
        if val > 7000: return "Gold"
        return "Silver"
    stats['customer_tier'] = stats.apply(get_tier, axis=1)

    # 2. BROAD BRACKETS (Executive View)
    # Use this for Pie Charts or High-Level Filters
    def get_broad_bracket(v):
        if v <= 0: return "No Spend"
        if v <= 7000: return "0 - 7k"
        if v <= 20000: return "7k - 20k"
        return "20k+"
    stats['bracket_broad'] = stats['monetary_value'].apply(get_broad_bracket)

    # 3. GRANULAR BRACKETS (Distribution View)
    # Use this for Bar Charts / Histograms
    def get_granular_bracket(v):
        if v <= 0: 
            return "0. No Spend", 0
        if v <= 2000: 
            return "1. 0 - 2k", 1
        if v <= 4000: 
            return "2. 2k - 4k", 2
        if v <= 6000: 
            return "3. 4k - 6k", 3
        if v <= 8000: 
            return "4. 6k - 8k", 4
        if v <= 10000: 
            return "5. 8k - 10k", 5
        if v <= 12000: 
            return "6. 10k - 12k", 6
        if v <= 14000: 
            return "7. 12k - 14k", 7
        if v <= 16000: 
            return "8. 14k - 16k", 8
        if v <= 18000: 
            return "9. 16k - 18k", 9
        if v <= 20000: 
            return "10. 18k - 20k", 10
        return "11. 20k+", 11

    # Apply Logic (Returns a Tuple)
    bracket_res = stats['monetary_value'].apply(get_granular_bracket)
    
    # Split into two columns: The Label and the Sort Order
    stats['bracket_granular'] = bracket_res.apply(lambda x: x[0])
    stats['bracket_sort'] = bracket_res.apply(lambda x: x[1])

    # ✅ MERGE ALL COLUMNS (Tier, Broad, Granular)
    cols_to_merge = ['customer_tier', 'bracket_broad', 'bracket_granular', 'bracket_sort']
    df_sess = df_sess.merge(stats[cols_to_merge], on='Contact ID', how='left')

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    df_sess.drop(columns=['prev_session', 'days_since_last', 'is_new_journey', 'journey_id'], inplace=True, errors='ignore')
    df_sess.to_csv(PROCESSED_DATA_DIR / "fact_sessions_enriched.csv", index=False)
    stats.reset_index().to_csv(PROCESSED_DATA_DIR / "dim_customers_rfv.csv", index=False)

    # BRAND MENTIONS EXPORT
    b_rows = []
    for _, r in df_sess.iterrows():
        # ✅ FIX: Priority to matched_brand, fallback to temp_brands
        if r['matched_brand'] != "Unknown": 
            brands_to_use = [r['matched_brand']]
        else:
            raw_brands = re.split(r'\s*[,|]\s*', str(r.get('temp_brands', '')))
            brands_to_use = [b.strip() for b in raw_brands if b.strip()]
        
        if not brands_to_use: 
            b_rows.append({"session_id": r['session_id'], "Contact ID": r['Contact ID'], "brand_name": "General Inquiry", "revenue": r['mpesa_amount'], "is_converted": r['is_converted'], "date": r['session_start']})
        else:
            for brand in brands_to_use: 
                b_rows.append({"session_id": r['session_id'], "Contact ID": r['Contact ID'], "brand_name": brand, "revenue": r['mpesa_amount'], "is_converted": r['is_converted'], "date": r['session_start']})
    if b_rows: pd.DataFrame(b_rows).to_csv(PROCESSED_DATA_DIR / "fact_brand_mentions.csv", index=False)
    
    # ZONES / CONCERNS / FUNNEL (Standard exports)
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

    f_rows = []
    for _, r in df_sess.iterrows():
        base = {"session_id": r['session_id'], "id": r['Contact ID'], "name": r['contact_name'], "date": r['session_start'], "cat": r['primary_category'], "src": r['acquisition_source']}
        hist = str(r['funnel_history'])
        f_rows.append({**base, "stage": "Inquiry", "sort_order": 1, "val": 1})
        if "Price Quoted" in hist: f_rows.append({**base, "stage": "Price Quoted", "sort_order": 2, "val": 1})
        if "Price Objection" in hist: f_rows.append({**base, "stage": "Price Objection", "sort_order": 3, "val": 1})
        if r['is_converted']: f_rows.append({**base, "stage": "Converted", "sort_order": 4, "val": 1})
        if r['customer_tier'] == "Platinum": f_rows.append({**base, "stage": "High Val Cust", "sort_order": 5, "val": 1})
    pd.DataFrame(f_rows).to_csv(PROCESSED_DATA_DIR / "fact_funnel_analytics.csv", index=False)

    print("✅ ANALYTICS COMPLETE.")

if __name__ == "__main__":
    run_analytics_pipeline()