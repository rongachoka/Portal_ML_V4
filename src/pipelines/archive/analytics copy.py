import os
import re
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# V3 PRODUCTION IMPORTS
from Portal_ML_V4.src.config.settings import (
    FINAL_TAGGED_DATA, PROCESSED_DATA_DIR, CLEANED_DATA_DIR, BASE_DIR,
    MSG_HISTORY_RAW
)
# Phone number cleaning logi import
from Portal_ML_V4.src.utils.phone import clean_id, normalize_phone, clean_id_excel_safe

# Note: BRAND_LIST is completely removed here
from Portal_ML_V4.src.config.brands import BRAND_ALIASES
from Portal_ML_V4.src.config.tag_rules import (
    CANONICAL_CATEGORY_RULES, CONCERN_RULES
)

KB_PATH = BASE_DIR / "data" / "01_raw" / "Final_Knowledge_Base_PowerBI.csv"

try:
    kb_df = pd.read_csv(KB_PATH)
    kb_df = kb_df.dropna(subset=['Brand'])
    
    # 1. Generate the master brand list dynamically
    DYNAMIC_BRAND_LIST = (
        kb_df['Brand'].astype(str).str.strip().str.title().unique().tolist()
    )
    
    # 2. Build a fallback mapping (Brand -> Most Common Category)
    # If a customer asks about CeraVe but no product matches, 
    # the script uses this to confidently log "Skincare" instead of "Unknown"
    BRAND_TO_CATEGORY_MAP = (
        kb_df.groupby('Brand')['Canonical_Category']
        .agg(lambda x: x.mode()[0] if not x.mode().empty else "General Inquiry")
        .to_dict()
    )
except Exception as e:
    print(f"⚠️ Warning: Could not load KB for dynamic brands: {e}")
    DYNAMIC_BRAND_LIST = []
    BRAND_TO_CATEGORY_MAP = {}

# --- CONFIGURATIONS ---
CHANNEL_MAP = {
    389017: 'WhatsApp', 387986: 'Instagram', 388255: 'Facebook', 
    388267: 'TikTok', 389086: 'Web Chat'
}

MACRO_GROUP_MAP = {
    "Skincare": "Skincare", "Baby Care": "Baby Care", "Haircare": "Hair Care",
    "Oral Care": "Oral Care", "Supplements": "Supplements", "Medicine": "Medicine", 
    "Medical Devices": "Medical Devices & Kits", "First Aid": "Medical Devices & Kits",      
    "Homeopathy": "Homeopathy", "Men's Care": "Men Care", "Women's Health": "Women's Health",
    "Fragrance": "Perfumes", "Lip Care": "Lip Care", "Sexual Health": "Sexual Health",
    "Hair Care": "Hair Care", "Menscare": "Men Care", "Perfumes": "Perfumes"
}

# 🚨 1. CONTEXT VALIDATION RULES
BRAND_CONTEXT_RULES = {
    "APTAMIL": ["APTAMIL", "BABY", "MILK", "FORMULA", "INFANT", "LACTOGEN"],
    "LA ROCHE-POSAY": ["LA ROCHE", "LRP", "EFFACLAR", "CICAPLAST", "ANTHELIOS", "LIPIKAR", "POSAY"],
    "CERAVE": ["CERAVE", "CERA VE", "SA CLEANSER", "HYDRATING", "FOAMING"],
    "THE ORDINARY": ["ORDINARY", "NIACINAMIDE", "RETINOL", "PEELING", "ACID", "TO"],
    "PANADOL": ["PANADOL", "PAIN", "HEADACHE", "FEVER"],
    "PAMPERS": ["PAMPERS", "DIAPER", "BABY", "NAPPY"],
    "HUGGIES": ["HUGGIES", "DIAPER", "BABY", "NAPPY"]
}


PSEUDO_BRANDS = ["effaclar", "zelaton", "acnes", "panadol"]


# ✅ FIXED VARIABLE NAME
FORM_FACTORS = {
    "serum": ["cream", "wash", "cleanser", "oil", "spray", "tablet", "capsule"],
    "cream": ["serum", "wash", "cleanser", "oil", "spray", "tablet", "capsule"],
    "wash": ["serum", "cream", "oil", "spray", "tablet", "capsule"],
    "cleanser": ["serum", "cream", "oil", "spray", "tablet", "capsule"],
    "oil": ["serum", "cream", "wash", "cleanser", "tablet", "capsule"],
    "tablet": ["serum", "cream", "wash", "cleanser", "oil", "spray"],
    "capsule": ["serum", "cream", "wash", "cleanser", "oil", "spray"],
    "glycinate": ["oil", "spray", "cream", "gel"]
}

BLACKLIST_PHRASES = [
    "click on this", "on this link", "would like order", "would like purchase",
    "can get more", "more info", "how can assist", "where would like",
    "like us deliver", "get more info", "make purchase", "checking availability",
    "http", "www.", ".com", ".ke", "image attachment", "view and order"
]

JUNK_WORDS = [
    "hello", "hi", "hey", "how are you", "good morning", "good evening", "good afternoon",
    "morning", "evening", "afternoon", "tomorrow", "today", "tonight", "day", "night",
    "location", "located", "where", "branch", "shop", "visit",
    "delivery", "deliver", "shipping", "cost", "charge", "fee", "send",
    "pay", "payment", "mpesa", "till", "number", "code", "total",
    "available", "stock", "have", "do you have", "selling",
    "business", "learn more", "tell me", "ad", "advert", "info", "information",
    "thank you", "thanks", "welcome", "ok", "okay", "sawa", "fine", "yeah", "yes",
    "click", "link", "view", "order", "purchase", "buying",
    "help", "assist", "question", "inquiry", "product", "products", "item", "items",
    "naona", "hiyo", "ni", "kuna", "nataka", "please", "kindly", "which", "one",
    "seen", "saw", "from", "advert", "plus", "and", "with"
]

GENERIC_TERMS = [
    "sunscreen", "sun screen", "sunblock", "moisturizer", "cleanser", "wash",
    "soap", "lotion", "cream", "gel", "serum", "toner", "oil", "spray",
    "pills", "tablets", "medicine", "drugs", "capsules", "vitamins",
    "shampoo", "conditioner", "hair food", "treatment", "scrub", "mask",
    "spf", "spf50", "spf30"
]


AD_NAME_MAP = {
    "6823164424240": "La Roche BOGOF",
    "6817550498240": "CosRx Centella",
    "6941964715840": "Ashwaghanda",
    "6941951457840": "Ashwaghanda",
    "6937045949440": "Magnesium Glycinate + Ashwaghanda",
    "6937053336840": "Magnesium + Ashwaghanda",
    "6941964716840": "Magnesium + Ashwaghanda",
    "6948909029040": "Bundles",
    "6921500406040": "LRP Bundles / Products",
    "6943721123640" : "LRP Anthelios",
    "6935232756840": "LRP Anthelios Jan",
    "6826292826840": "10% Off Categories",
    "6955912833840" : "10% Off Skincare Products",
    "120221326858920683": "Clearance Sale",
    "6949654216840": "Vichy Dandruff" ,
    "6955681219240": "Vichy Dandruff Sales",
    "6915084614640": "The Vichy Aqualia Thermal Night Spa",
    "120239692754220696": "For The Girlies",
    "6936990612240" : "3 Supplements Every Woman Should Take",
    "120239692341380696" : "This is your Sign to Stock Up",
    "6942365246240" : "This is your Sign to Stock Up",
    "6941964718040" : "Hormonal Care",
    "6941964715640" : "Supplements Reel",
    "6945406805440" : "Skincare Reel",
    "6941964717640" : "CeraVe Acne Routine",
    "6943706631440" : "Lip Products",
    "6943627224640" : "EOS",
    "6943617269240" : "BBW",
    "6945414530240" : "Skincare Offers",
    "6945471200240" : "Unsure what to gift this Valentine's?",
    "6948905752240" : "Moisturizers for Oily Skin",
    "6949654359440" : "Reedle Shot",
    "6948905615040" : "LRP Giveaway",
    "6948903906440" : "Anua + Mandelic Acid",
    "6948906394440" : "Supplements for Glowing Skin",
    "6955913814640" : "Male Adult Acne"


}

#  COUNTRY CODE LOOKUP
def fmt_us(d):
    # +1 212 555 0123  (1 + 3 + 3 + 4)
    return f"+1 {d[1:4]} {d[4:7]} {d[7:]}"

def fmt_uk(d):
    # +44 7700 900 982  (44 + 4 + 3 + 3)
    return f"+44 {d[2:6]} {d[6:9]} {d[9:]}"

COUNTRY_CODES = {
    "44": ("UK",        12, fmt_uk),
    "1":  ("US/Canada", 11, fmt_us),
}
# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

# ✅ NEW: Extract Staff Name from Text
def extract_assigned_staff(text):
    if pd.isna(text): return None
    # Pattern: "assigned to [Name]" or "assigned to Name"
    # We grab the word immediately after "assigned to"
    matches = re.findall(r"assigned to\s+([a-zA-Z0-9_]+)", str(text), re.IGNORECASE)
    
    if matches:
        # Take the LAST assignment in the conversation (most recent)
        found = matches[-1].title()
        
        # Filter out system keywords if they accidentally got picked up
        if found.lower() in ['bot', 'system', 'me', 'you', 'undefined']: 
            return None
            
        return found
    return None

# Moving this logic to utils.phone

# # ✅ FOR CONTACT ID (Clean)
# def clean_id(val):
#     if val is None: return None
#     s = str(val).strip().replace('.0', '')
#     s = ''.join(filter(str.isdigit, s))
#     if len(s) == 0: return None

#     # ── KENYAN — normalise to 9 digits ────────────────────────────────────────
#     if s.startswith('254') and len(s) == 12:
#         return s[-9:]
#     if s.startswith('0') and len(s) == 10:
#         return s[-9:]
#     if len(s) == 9 and s.startswith(('7', '1')):
#         return s

#     # ── NON-KENYAN — check country code lookup ────────────────────────────────
#     if len(s) >= 11 and not s.startswith('254'):
#         # Try 2-digit codes before 1-digit to avoid '1' matching '44' numbers
#         for prefix in ["44", "1"]:
#             if s.startswith(prefix):
#                 name, expected_len, fmt_fn = COUNTRY_CODES[prefix]
#                 if len(s) == expected_len:
#                     return fmt_fn(s)        # well-formed → format nicely
#                 elif len(s) > expected_len:
#                     return f"INVALID:{s}"   # too long → flag it
#                 # too short → fall through

#         # Generic foreign number — no specific formatter
#         if len(s) > 13:
#             return f"INVALID:{s}"
#         return f"+{s}"

#     return s

# # ✅ FOR EXCEL PHONE EXPORT (With Apostrophe)
# def clean_id_excel_safe(val):
#     if pd.isna(val): return None
#     s = str(val).strip().replace('.0', '')
#     s = ''.join(filter(str.isdigit, s))
#     if len(s) == 0: return None
#     if s.startswith('254'): return f"'+{s}"
#     return f"'{s}"

# 🚨 REVENUE LOGIC V54.0: Strict Adjacency (The Anti-Ghost Code Fix)
def recalculate_mpesa_smart(row):
    # 1. TEXT NORMALIZATION
    raw_text = str(row.get('full_context', '')).lower()
    clean_text = re.sub(r'[\xc2\xa0\t\n\r]+', ' ', raw_text) # Fix "17Â 800"
    clean_text = re.sub(r'\s+', ' ', clean_text)
    
    current_amt = float(row.get('mpesa_amount', 0))
    
    # 2. NOISE MASKING (Aggressive)
    noise_patterns = [
        r"(?:account|acc)\s*(?:no|number|num)?[:\.]?\s*\d+", 
        r"paybill\s*(?:no|number|num)?[:\.]?\s*\d+", 
        r"transaction\s*cost\s*[:.,-]?\s*(?:ksh|kes)?\.?\s*[\d,]+(?:\.\d{2})?", 
        r"(?:new)?\s*m-?pesa\s*balance\s*(?:is|:)?\s*(?:ksh|kes)?\.?\s*[\d,]+(?:\.\d{2})?",
        r"amount\s*you\s*can\s*transact.*?(?:ksh|kes)?\.?\s*[\d,]+(?:\.\d{2})?" 
    ]
    for p in noise_patterns:
        clean_text = re.sub(p, " [NOISE] ", clean_text, flags=re.I)

    # 3. STRICT SYSTEM MESSAGE EXTRACTION (Dictionary for Aggregation)
    found_txns = {}
    
    # Pattern A (Code First): STRICT ADJACENCY (\s+)
    # Code must be IMMEDIATELY before "Confirmed". No gaps allowed.
    matches_std = re.finditer(r'\b([A-Z0-9]{10})\s+(?:then\s+)?Confirmed.*?(?:ksh|kes)\.?\s*([\d,\.\s]+)', clean_text, re.IGNORECASE)
    for m in matches_std:
        try:
            code = m.group(1).upper()
            amt_str = re.sub(r'[^\d.]', '', m.group(2))
            found_txns[code] = float(amt_str)
        except: pass
        
    # Pattern B (Ref Last): "Confirmed... Bill Payment... Ref [Code]"
    # Matches: "Confirmed. Bill Payment to... of KES 50.00... Ref THV..."
    matches_bill = re.finditer(r'Confirmed.*?Bill\s*Payment.*?(?:ksh|kes)\.?\s*([\d,\.\s]+).*?Ref\.?\s*[:\.]?\s*([A-Z0-9]{10})', clean_text, re.IGNORECASE)
    for m in matches_bill:
        try:
            code = m.group(2).upper()
            amt_str = re.sub(r'[^\d.]', '', m.group(1))
            found_txns[code] = float(amt_str)
        except: pass

    # 4. DECISION LOGIC
    if found_txns:
        # If M-Pesa found, IGNORE other text like "Total is 3500".
        return sum(found_txns.values())
        
    # Fallback: Manual entries
    return current_amt

# ==========================================
# 3. KNN PRODUCT MATCHER
# ==========================================
class ProductMatcher:
    def __init__(self):
        path = BASE_DIR / "data" / "01_raw" / "Final_Knowledge_Base_PowerBI.csv"
        if not os.path.exists(path):
            self.active = False
            return

        self.df = pd.read_csv(path).fillna("")
        self.df['search_text'] = (
            self.df['Brand'] + " " + 
            self.df['Name'] + " " + self.df['Name'] + " " + self.df['Name'] + " " + 
            self.df['Sub_Category'] + " " + self.df['Sub_Category'] + " " +
            self.df['Concerns']
        ).str.lower()

        self.vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 4), min_df=1, strip_accents='unicode')
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['search_text'])
        self.knn = NearestNeighbors(n_neighbors=50, metric='cosine').fit(self.tfidf_matrix) 
        self.active = True
        print(f"   🧠 Knowledge Base Loaded: {len(self.df)} products")

    def _extract_url_slugs(self, text):
        slugs = []
        matches = re.findall(r'portalpharmacy\.ke/([\w-]+)', str(text), re.IGNORECASE)
        for m in matches:
            if "index.php" not in m and "search" not in m: slugs.append(m.replace("-", " "))
        return " ".join(slugs)

    def _clean_context_smart(self, text):
        text = str(text).lower()
        url_context = self._extract_url_slugs(text)
        text = re.sub(r'http\S+', '', text)
        for phrase in BLACKLIST_PHRASES: text = text.replace(phrase, "")
        text = text + " " + url_context
        for junk in JUNK_WORDS: text = re.sub(rf"\b{junk}\b", "", text)
        for bad, good in BRAND_ALIASES.items():
            if bad in text: text = text.replace(bad, good.lower())
        text = re.sub(r'[^a-z0-9\s]', '', text)
        return text.strip()

    def get_best_match(self, context, mpesa_amount=0, brand_hint=None):
        if not self.active: return None
        try:
            cleaned_text = self._clean_context_smart(context)
            has_official_brand = (brand_hint and brand_hint != "Unknown")
            has_pseudo_brand = any(pb in cleaned_text for pb in PSEUDO_BRANDS)
            
            # --- GENERIC QUERY TRAP ---
            if not has_official_brand and not has_pseudo_brand:
                check_text = re.sub(r'\b(price|cost|how much|is|the|a|an|need|want|looking for)\b', '', cleaned_text).strip()
                words = [w for w in check_text.split() if len(w) > 2]
                if len(words) > 0 and all(w in GENERIC_TERMS for w in words): return None 
                if len(check_text) < 4: return None

            if not has_official_brand and not has_pseudo_brand: return None 

            query_text = cleaned_text
            if has_official_brand: query_text = f"{brand_hint} {brand_hint} {cleaned_text}" 
            elif has_pseudo_brand:
                for pb in PSEUDO_BRANDS:
                    if pb in cleaned_text: query_text = f"{pb} {pb} {cleaned_text}"; break

            if len(query_text) < 3: return None

            query_vec = self.vectorizer.transform([query_text])
            dist, idx = self.knn.kneighbors(query_vec)
            
            best_match = None
            highest_score = 0
            
            # 🚨 DYNAMIC CATEGORY LOOKUP (Replaces DEPARTMENT_TO_CANONICAL loop)
            allowed_macro = None
            if has_official_brand:
                # We simply ask the dynamic map: "What category is this brand usually?"
                raw_cat = BRAND_TO_CATEGORY_MAP.get(brand_hint)
                if raw_cat:
                    allowed_macro = MACRO_GROUP_MAP.get(raw_cat, raw_cat)

            for i in range(len(idx[0])):
                match_idx = idx[0][i]
                similarity = 1 - dist[0][i]
                match_row = self.df.iloc[match_idx]
                brand_name = str(match_row['Brand']).strip()
                match_sub = str(match_row['Sub_Category']).lower()
                prod_name = str(match_row['Name']).lower()
                
                if has_official_brand:
                    b_hint_clean = brand_hint.lower().replace("the ", "").strip()
                    b_row_clean = brand_name.lower().replace("the ", "").strip()
                    # Strict Brand Matching: If the hint is "CeraVe", reject "Cetaphil" products
                    if b_hint_clean not in b_row_clean and b_row_clean not in b_hint_clean: continue 

                # Safety Check: Don't match Baby products to adults unless explicitly asked
                if "baby" in prod_name or "kid" in prod_name or "pediatric" in prod_name:
                    if not any(x in cleaned_text for x in ["baby", "kid", "child", "pediatric", "born"]): continue

                # 🚨 DELETED: The 'MACRO_TO_SUB_CATS' check is gone. 
                # We no longer penalize sub-categories because the KB is the source of truth.

                # Form Factor logic (Still useful to prevent matching 'Cream' to 'Serum')
                for anchor, bad_forms in FORM_FACTORS.items():
                    if anchor in cleaned_text:
                        if any(bf in prod_name or bf in match_sub for bf in bad_forms): similarity -= 0.35 

                # Boost score if they share exact words
                common_words = set(cleaned_text.split()) & set(prod_name.split())
                if len(common_words) > 0: similarity += 0.10

                if similarity > highest_score:
                    highest_score = similarity
                    best_match = match_row
            
            threshold = 0.55 if has_official_brand else 0.65
            if best_match is not None and highest_score > threshold:
                pred_brand_upper = str(best_match['Brand']).upper()
                
                # Context Rules (Optional: You can keep or delete BRAND_CONTEXT_RULES depending on preference)
                if pred_brand_upper in BRAND_CONTEXT_RULES:
                    required_kws = BRAND_CONTEXT_RULES[pred_brand_upper]
                    context_upper = cleaned_text.upper()
                    if not any(kw in context_upper for kw in required_kws): return None 
                
                return best_match
            return None
        except Exception: pass
        return None

# ==========================================
# 4. OTHER HELPERS
# ==========================================
def find_id_col(df):
    cands = ['Contact ID', 'ContactID', 'ID', 'id', 'contact_id']
    return next((c for c in cands if c in df.columns), None)

def split_tags_logic(ts):
    if not isinstance(ts, str): return "", "", "", ""
    raw = [t.strip() for t in ts.split("|") if t.strip()]
    primary_z, secondary_z, funnel, other = [], [], [], []
    for t in raw:
        tl = t.lower()
        if "secondary zone:" in tl: secondary_z.append(re.sub(r"secondary zone:\s*", "", t, flags=re.I).strip())
        elif "zone:" in tl: primary_z.append(re.sub(r"zone:\s*", "", t, flags=re.I).strip())
        elif any(x in tl for x in ["price", "payment", "converted"]): funnel.append(re.sub(r"(funnel|concern):\s*", "", t, flags=re.I).strip())
        else: other.append(t.strip())
    return " | ".join(primary_z), " | ".join(secondary_z), " | ".join(funnel), " | ".join(other)

def extract_meta(row):
    tags = str(row.get('final_tags', '')).lower()
    context = str(row.get('full_context', '')).lower()
    
    stock = tags.count("stock")
    rec = tags.count("recommendation")
    
    is_consult = 0
    if (
        "dermatologist" in tags or 
        "skin consultation" in tags or
        "dermatologist" in context or
        "skin consultation" in context or
        "skin test" in context
    ):
        is_consult = 1
        
    # 🚨 UPDATED: Using the dynamic KB list
    found = [b for b in DYNAMIC_BRAND_LIST if b.lower() in tags]
    
    return stock, is_consult, rec, " | ".join(found) if found else ""


def normalize_brands_with_intent(brand_str, full_text):
    found_brands = set()
    
    # 1. Process tags from the previous step
    if brand_str and brand_str != "Unknown":
        raw_list = [b.strip().lower() for b in str(brand_str).split("|") if b.strip()]
        for raw in raw_list:
            matched = False
            for typo, clean in BRAND_ALIASES.items():
                if typo in raw: found_brands.add(clean); matched = True; break
            if not matched: found_brands.add(raw.title())

    # 2. 🚨 THE SEVEN SEAS FIX: Scan full text using Regex Word Boundaries
    # Old Code: if typo in text_lower: ... (This caused the bug)
    text_lower = str(full_text).lower()
    for typo, clean in BRAND_ALIASES.items():
        # \b matches the start/end of a word. 
        # It matches "seven" but IGNORES "seventy", "season", or "seven-hundred"
        if re.search(r'\b' + re.escape(typo) + r'\b', text_lower): 
            found_brands.add(clean)
        
    scored_brands = []
    buy_words = ["buy", "order", "price", "cost", "much", "link", "recommend", "need"]
    context_words = ["using", "use", "used", "currently", "have", "routine"]
    
    for brand in found_brands:
        score = 0
        b_clean = brand.lower()
        if b_clean.replace(" ", "-") in text_lower: score += 5
        matches = [m.start() for m in re.finditer(re.escape(b_clean), text_lower)]
        for m in matches:
            start = max(0, m - 50); end = min(len(text_lower), m + 50)
            window = text_lower[start:end]
            if any(w in window for w in buy_words): score += 3
            if any(w in window for w in context_words): score -= 2
        scored_brands.append((brand, score))
    
    scored_brands.sort(key=lambda x: x[1], reverse=True)
    return [b[0] for b in scored_brands]


def load_ads_for_analytics():
    ADS_DIR = Path(MSG_HISTORY_RAW).parent / "ads"
    all_files = glob.glob(str(ADS_DIR / "contacts-*.csv"))
    if not all_files: 
        return pd.DataFrame()
    
    dfs = []
    cols = ['Timestamp', 'Contact ID', 'Ad campaign ID', 'Ad group ID', 'Ad ID']
    for f in all_files:
        try:
            t = pd.read_csv(f)
            t['Timestamp'] = pd.to_datetime(t['Timestamp'], errors='coerce')
            t['Contact ID'] = pd.to_numeric(t['Contact ID'], errors='coerce').astype('Int64')
            
            # Clean Dashes immediately
            for c in ['Ad campaign ID', 'Ad group ID', 'Ad ID']:
                if c in t.columns: 
                    t[c] = t[c].replace(['-', ' -'], pd.NA)
                
            avail = [c for c in cols if c in t.columns]
            dfs.append(t.dropna(subset=['Contact ID', 'Timestamp'])[avail])
        except: 
            pass
        
    if not dfs: 
        return pd.DataFrame()
    
    df = pd.concat(dfs, ignore_index=True).sort_values('Timestamp')

    return df.drop_duplicates(subset=['Contact ID', 'Timestamp', 'Ad campaign ID'])

# ==========================================
# 5. MAIN ANALYTICS PIPELINE
# ==========================================
def run_analytics_pipeline():
    print("📊 V55.0 MASTER ANALYTICS: Intelligent Revenue + Character Fix + Aggregation")
    if not os.path.exists(FINAL_TAGGED_DATA): return

    df_sess = pd.read_parquet(FINAL_TAGGED_DATA)
    matcher = ProductMatcher()
    
    # 🚨 REVENUE LOGIC: Using Intelligent Calculation (Noise Masking + Priority)
    print("💰 Deduplicating Revenue (Intelligent Mode)...")
    df_sess['mpesa_amount'] = df_sess.apply(recalculate_mpesa_smart, axis=1)

    # 🚨 STAFF EXTRACTION LOGIC (New)
    print("🕵️ Extracting Staff from Assignment Messages...")
    df_sess['extracted_staff_name'] = df_sess['full_context'].apply(extract_assigned_staff)

    ## ADDED NEW SECTION
    ## Linking ad data to sessions before product matching, so we have campaign info available for smarter matching and better analytics.
    print("🔗 Linking Ad Campaigns to Sessions...")
    df_ads = load_ads_for_analytics()
    
    if not df_ads.empty:
        # Ensure session start is datetime
        df_sess['session_start'] = pd.to_datetime(df_sess['session_start'])
        df_sess = df_sess.sort_values('session_start')
        
        # Merge Asof: Match Session Start to closest Ad Click (within 1h lookback)
        # Using 1h here because a session might start hours after the click
        df_sess = pd.merge_asof(
            df_sess,
            df_ads,
            left_on='session_start',
            right_on='Timestamp',
            by='Contact ID',
            tolerance=pd.Timedelta('1h'), #  window for attribution
            direction='backward', # Ad click must happen BEFORE session starts (usually)
            suffixes=('', '_new')
        )
        
        # Coalesce columns (If tagging kept them, keep them. If not, use new ones)
        for col in ['Ad campaign ID', 'Ad group ID', 'Ad ID']:
            if f'{col}_new' in df_sess.columns:
                if col in df_sess.columns:
                    df_sess[col] = df_sess[col].fillna(df_sess[f'{col}_new'])
                else:
                    df_sess[col] = df_sess[f'{col}_new']
                df_sess.drop(columns=[f'{col}_new'], inplace=True)
                
        print(f"   ℹ️  Ads Linked: {df_sess['Ad campaign ID'].notna().sum():,} sessions matched.")
        df_sess.drop(columns=['Timestamp'], errors='ignore', inplace=True)

    # A. CLEANING (USE CLEAN ID HERE - NO APOSTROPHE)
    print("🧹 Cleaning IDs...")
    df_sess['Contact ID'] = df_sess['Contact ID'].apply(clean_id) 
    
    cleaned_conv_path = CLEANED_DATA_DIR / "cleaned_conversations.csv"
    cleaned_cont_path = CLEANED_DATA_DIR / "cleaned_contacts.csv"
    
    # 1. LOAD CONVERSATIONS
    if os.path.exists(cleaned_conv_path):
        df_c = pd.read_csv(cleaned_conv_path)
        conv_id_col = find_id_col(df_c)
        if conv_id_col: df_c['Contact ID'] = df_c[conv_id_col].apply(clean_id) # USE CLEAN ID
        df_c['conv_start'] = pd.to_datetime(df_c['DateTime Conversation Started'], errors='coerce')
        df_c = df_c.sort_values('conv_start')

        for col in ['Average Response Time', 'Number of Responses', 'Number of Outgoing Messages', 'Number of Incoming Messages', 'Resolution Time', 'First Response Time']:
            if col in df_c.columns: df_c[col] = pd.to_numeric(df_c[col], errors='coerce').fillna(0)

        potential_cols = [
            'Contact ID', 'conv_start', 'Conversation ID', 'Opened By Source', 
            'Assignee', 'First Assignee', 'Last Assignee', 'Closed By', 
            'First Response By', 'First Response Time', 'Average Response Time', 'Resolution Time',
            'Number of Responses', 'Number of Outgoing Messages', 'Number of Incoming Messages',
            'Conversation Category', 'Closing Note Summary'
        ]
        cols_to_merge = [c for c in potential_cols if c in df_c.columns]
        df_c_subset = df_c[cols_to_merge].copy()
    else:
        df_c_subset = pd.DataFrame()

    # 2. LOAD NAMES & PHONES
    name_map = {}
    phone_map = {}
    if os.path.exists(cleaned_cont_path):
        df_n = pd.read_csv(cleaned_cont_path)
        id_n = find_id_col(df_n)
        
        name_c = next((c for c in df_n.columns if 'name' in c.lower() and 'phone' not in c.lower()), None)
        phone_c = next((c for c in df_n.columns if 'phone' in c.lower() or 'number' in c.lower()), None)

        if id_n:
            df_n[id_n] = df_n[id_n].apply(clean_id) 
            if name_c: name_map = df_n.set_index(id_n)[name_c].to_dict()
            if phone_c:
                df_n[phone_c] = df_n[phone_c].apply(normalize_phone)
                phone_map = df_n.set_index(id_n)[phone_c].to_dict()

    # B. SMART LINKING
    if 'session_start' in df_sess.columns: df_sess['session_start'] = pd.to_datetime(df_sess['session_start'])
    df_sess = df_sess.sort_values('session_start')
    df_sess['Conversation ID'] = None
    df_sess['active_staff'] = None  
    df_sess['recovered_source'] = None 

    if not df_c_subset.empty:
        print("   🤝 Linking Sessions...")
        df_c_subset = df_c_subset.sort_values('conv_start')
        GRACE_PERIOD = pd.Timedelta(hours=24)
        
        # ✅ UPDATED TEAM MAP (Includes Jess & Jeff)
        TEAM_MAP_INTERNAL = {
            "847526": "Ishmael", "860475": "Faith", "879396": "Nimmoh",
            "879430": "Rahab", "879438": "Brenda", "962460": "Katie",
            "1000558": "Sharon", "845968": "Joy",
            "1006108": "Jess", "971945": "Jeff"
        }
        
        def resolve_staff_temp(row):
            candidates = [str(row.get('Last Assignee')), str(row.get('Assignee')), str(row.get('First Response By')), str(row.get('First Assignee'))]
            for c in candidates:
                clean = str(c).replace('.0', '').strip()
                if clean in TEAM_MAP_INTERNAL: return TEAM_MAP_INTERNAL[clean]
                if clean not in ['nan', 'None', 'System', 'Bot', 'Auto Assign'] and len(clean)>2: return clean
            return "Unassigned"

        df_c_subset['Temp_Staff_Name'] = df_c_subset.apply(resolve_staff_temp, axis=1)

        match_count = 0
        for idx, row in df_sess.iterrows():
            s_id = row['Contact ID']; s_time = row['session_start']
            
            # 🚨 THE FIX: Bound the lookback! Don't grab a conversation from 8 months ago.
            candidates = df_c_subset[
                (df_c_subset['Contact ID'] == s_id) & 
                (df_c_subset['conv_start'] <= s_time + GRACE_PERIOD) &
                (df_c_subset['conv_start'] >= s_time - pd.Timedelta(days=14)) # <-- NEW: 14 day cutoff
            ]
            
            if not candidates.empty:
                winner = candidates.iloc[-1]
                df_sess.at[idx, 'Conversation ID'] = winner['Conversation ID']
                df_sess.at[idx, 'active_staff'] = winner['Temp_Staff_Name']
                match_count += 1
        print(f"      ✅ Linked {match_count} sessions.")

        cols_needed = ['Conversation ID', 'First Response By', 'Assignee', 'Last Assignee', 'Closed By', 'First Assignee', 'Opened By Source']
        cols_present = [c for c in cols_needed if c in df_c_subset.columns]
        meta_df = df_c_subset[cols_present].drop_duplicates(subset=['Conversation ID'])
        cols_to_drop = [c for c in cols_present if c in df_sess.columns and c != 'Conversation ID']
        if cols_to_drop: df_sess.drop(columns=cols_to_drop, inplace=True)

        # 🟢 Align types before merge — None assignments make df_sess['Conversation ID'] object
        df_sess['Conversation ID'] = pd.to_numeric(df_sess['Conversation ID'], errors='coerce').astype('Int64')
        meta_df['Conversation ID'] = pd.to_numeric(meta_df['Conversation ID'], errors='coerce').astype('Int64')

        df_sess = pd.merge(df_sess, meta_df, on='Conversation ID', how='left')
        
        # cols_to_drop = [c for c in cols_present if c in df_sess.columns and c != 'Conversation ID']
        # if cols_to_drop: df_sess.drop(columns=cols_to_drop, inplace=True)
        
        # df_sess = pd.merge(df_sess, meta_df, on='Conversation ID', how='left')
    else:
        print("      ⚠️ No Conversation Data found.")

    df_sess = df_sess.sort_values(by=['Contact ID', 'session_start'])
    df_sess['session_id'] = (df_sess['Contact ID'].astype(str) + "_" + df_sess['session_start'].dt.strftime('%Y-%m-%d %H:%M:%S'))
    df_sess['contact_name'] = df_sess['Contact ID'].map(name_map).fillna("Unknown")
    df_sess['mpesa_amount'] = pd.to_numeric(df_sess['mpesa_amount'], errors='coerce').fillna(0)
    df_sess['is_converted'] = (df_sess['final_tags'].str.contains("Converted", na=False) | (df_sess['mpesa_amount'] > 0)).astype(int)

    # C. CALCULATED COLUMNS
    if 'Opened By Source' in df_sess.columns: df_sess.rename(columns={'Opened By Source': 'recovered_source'}, inplace=True)
    df_sess['channel_name'] = pd.to_numeric(df_sess.get('Channel ID'), errors='coerce').map(CHANNEL_MAP).fillna("Unknown")

    def classify_source(row):

        if pd.notna(row.get('Ad campaign ID')) and str(row.get('Ad campaign ID')) != '':
            return 'Paid Ads'

        src = str(row.get('recovered_source', '')).lower()
        chan = str(row.get('channel_name', '')).lower()
        if any(k in src for k in ['ctc', 'ads', 'paid', 'ctc_ads']): 
            return 'Paid Ads'
        if any(k in src for k in ['contact', 'user']): 
            return 'Organic / Direct'
        if any(c in chan for c in ['WhatsApp', 'Instagram', 'Facebook', 'TikTok']): 
            return 'Organic / Direct'
        return 'Inbound / Unknown'
    df_sess['acquisition_source'] = df_sess.apply(classify_source, axis=1)

    # Ad Mapping
    df_sess['clean_ad_id'] = df_sess.get('Ad ID', pd.NA).astype(str).str.replace(r'\.0$', '', regex=True)
    df_sess['Ad Name'] = df_sess['clean_ad_id'].map(AD_NAME_MAP)

    df_sess['visit_rank'] = df_sess.groupby('Contact ID').cumcount() + 1
    df_sess['prior_orders'] = df_sess.groupby('Contact ID')['is_converted'].cumsum().shift(1).fillna(0)
    df_sess['customer_status'] = np.where(df_sess['prior_orders'] == 0, "New", "Returning")

    # D. AI ENRICHMENT
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
        # 1. Detect brands using your updated regex logic
        detected_brands = normalize_brands_with_intent(row.get('temp_brands'), row['full_context'])
        
        # 2. Try to find an exact product match in the KB
        match_row = None
        if detected_brands:
            for brand in detected_brands:
                match_row = matcher.get_best_match(row['full_context'], row['mpesa_amount'], brand_hint=brand)
                if match_row is not None: break
        
        # Fallback: Try matching without a specific brand hint
        if match_row is None: 
            match_row = matcher.get_best_match(row['full_context'], row['mpesa_amount'], brand_hint=None)

        # 3. Decision Logic (The New Dynamic Block)
        if match_row is not None:
            p_name = match_row['Name']
            p_brand = match_row['Brand']
            p_cat = match_row['Canonical_Category']
            p_sub = match_row['Sub_Category']
            p_concern = match_row['Concerns']
            p_audience = match_row['Target_Audience']
        else:
            if detected_brands:
                p_brand = detected_brands[0]
                
                # 🚨 THE REVENUE PROTECTION BLOCK
                # If money was paid (>0) but we couldn't match a specific product, 
                # mark it as "Unknown" so it doesn't skew your analytics.
                if row['mpesa_amount'] > 0:
                    p_name = "Unmatched Paid Product - Manual Review"
                    p_brand = "Unknown"
                    p_cat = "General Inquiry"
                else:
                    p_name = f"General {p_brand} Inquiry"
                    # 🚨 DYNAMIC FALLBACK: Uses the KB map instead of hardcoded rules
                    p_cat = BRAND_TO_CATEGORY_MAP.get(p_brand, "General Inquiry")
            else: 
                p_name, p_brand, p_cat = "Unknown", "Unknown", "General Inquiry"
            
            p_concern, p_audience, p_sub = "General", "General", "General"

        return (
            MACRO_GROUP_MAP.get(p_cat, "General Inquiry"), 
            p_sub, p_name, p_brand, p_concern, p_audience
        )
    # Run the AI logic once
    res_ai = df_sess.apply(process_ai, axis=1)

    # Unpack all 6 values instantly into new columns (Vectorized)
    # This replaces the 6 separate .apply(lambda...) lines
    df_sess[[
        'primary_category', 
        'sub_category', 
        'matched_product', 
        'matched_brand', 
        'matched_concern', 
        'target_audience'
    ]] = pd.DataFrame(res_ai.tolist(), index=df_sess.index)

    # E. SMART STAFF MAPPING
    print("👥 Mapping Staff...")
    # ✅ UPDATED TEAM MAP (Includes Jess & Jeff)
    TEAM_MAP = {
        "847526": "Ishmael", "860475": "Faith", "879396": "Nimmoh",
        "879430": "Rahab", "879438": "Brenda", "962460": "Katie",
        "1000558": "Sharon", "845968": "Joy",
        "1006108": "Jess", "971945": "Jeff"
    }

    def map_staff(val):
        if pd.isna(val) or str(val).strip() == "": return None
        clean_val = str(val).replace(".0", "").strip()
        return TEAM_MAP.get(clean_val, f"Other ({clean_val})")

    df_sess['first_response_name'] = df_sess['First Response By'].apply(map_staff)
    df_sess['assignee_name'] = df_sess['Assignee'].apply(map_staff)
    df_sess['closed_by_name'] = df_sess['Closed By'].apply(map_staff)
    df_sess['last_assignee_name'] = df_sess['Last Assignee'].apply(map_staff)
    
    if 'active_staff' not in df_sess.columns: df_sess['active_staff'] = None
    
    # 🚨 UPDATED FALLBACK CHAIN (Added extracted_staff_name at the end)
    df_sess['active_staff'] = df_sess['active_staff'].fillna(
        df_sess['closed_by_name'].fillna(
            df_sess['last_assignee_name'].fillna(
                df_sess['assignee_name'].fillna(
                    df_sess['first_response_name'].fillna(
                        df_sess['extracted_staff_name'].fillna("Unmapped") # <--- NEW FALLBACK
                    )
                )
            )
        )
    )
    
    # Apply same logic to sales_owner for consistency
    df_sess['sales_owner'] = df_sess['closed_by_name'].fillna(
        df_sess['last_assignee_name'].fillna(
            df_sess['assignee_name'].fillna(
                df_sess['first_response_name'].fillna(
                    df_sess['extracted_staff_name'].fillna("Unmapped")
                )
            )
        )
    )

    df_sess['activity_date'] = df_sess['session_start'].dt.date

    # F. EXPORTS & METRICS
    # 1. Lifetime Stats
    stats = df_sess.groupby('Contact ID').agg({'session_id': 'count', 'mpesa_amount': 'sum', 'is_converted': 'sum', 'session_start': 'max'}).rename(columns={'session_id': 'frequency', 'mpesa_amount': 'monetary_value', 'is_converted': 'total_orders', 'session_start': 'last_seen'})
    first_seen = df_sess.groupby('Contact ID')['session_start'].min()
    stats['first_seen'] = stats.index.map(first_seen.to_dict())

    stats['first_seen'] = pd.to_datetime(stats['first_seen'], errors='coerce')
    stats['last_seen']  = pd.to_datetime(stats['last_seen'],  errors='coerce')

    stats['days_to_convert'] = (stats['last_seen'] - stats['first_seen']).dt.days 
    
    def get_lifetime_tier(row):
        val = row['monetary_value']
        if val > 20000: return "Platinum"
        if val > 13000: return "Gold"
        if val > 7000: return "Silver"
        return "Bronze"
    stats['lifetime_tier_history'] = stats.apply(get_lifetime_tier, axis=1)
    
    def get_broad_bracket(v):
        if v <= 0: return "No Spend"
        if v <= 7000: return "0 - 7k"       # Bronze
        if v <= 13000: return "7k - 13k"    # Silver
        if v <= 20000: return "13k - 20k"   # Gold
        return "20k+"                        # Platinum
    stats['lifetime_bracket'] = stats['monetary_value'].apply(get_broad_bracket)

    # 🚨 CLEAN MERGE: Avoid Duplicating Columns
    df_sess = df_sess.merge(stats[['lifetime_tier_history', 'lifetime_bracket', 'days_to_convert']], on='Contact ID', how='left')

    # Session Tier
    def get_session_tier(row):
        val = row['mpesa_amount']
        if val > 20000: return "Platinum"
        if val > 13000: return "Gold"
        if val > 7000: return "Silver"
        if val > 0: return "Bronze"
        return "No Spend"
    
    df_sess['customer_tier'] = df_sess.apply(get_session_tier, axis=1)
    df_sess['session_tier'] = df_sess['customer_tier']
    df_sess['session_bracket_broad'] = df_sess['mpesa_amount'].apply(get_broad_bracket)

    # Granular Brackets
    def get_granular_bracket(v):
        if v <= 0: return "0. No Spend", 0
        if v <= 1000: return "0 - 1", 1
        if v <= 2000: return "1-2k", 2
        if v <= 3000: return "2-3k", 3
        if v <= 4000: return "3-4k", 4
        if v <= 5000: return "4-5k", 5
        if v <= 6000: return "5-6k", 6
        if v <= 7000: return "6-7k", 7
        if v <= 8000: return "7-8k", 8
        if v <= 9000:return "8-9k", 9
        if v <= 10000: return "9-10k", 10
        return "10k+", 11
    bracket_res = df_sess['mpesa_amount'].apply(get_granular_bracket)
    df_sess['bracket_granular'] = bracket_res.apply(lambda x: x[0])
    df_sess['bracket_sort'] = bracket_res.apply(lambda x: x[1])

    # 10-20k Brackets
    def get_10_20_bracket(v):
        if v <= 10000: return "0-10k", 0
        if v <= 11000: return "10-11k", 1
        if v <= 12000: return "11-12k", 2
        if v <= 13000: return "12-13k", 3
        if v <= 14000: return "13-14k", 4
        if v <= 15000: return "14-15k", 5
        if v <= 16000: return "15-16k", 6
        if v <= 17000: return "16-17k", 7
        if v <= 18000: return "17-18k", 8
        if v <= 19000: return "18-19k", 9
        if v <= 20000: return "19-20k", 10
        return "20k+", 11
    
    bracket_10_20_res = df_sess['mpesa_amount'].apply(get_10_20_bracket)
    df_sess['bracket_10_20k'] = bracket_10_20_res.apply(lambda x: x[0])
    df_sess['bracket_10_20k_sort'] = bracket_10_20_res.apply(lambda x: x[1])

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
        if days <= 3: return "Within 3 Days"
        if days <= 7: return "Research (1 Week)"
        return "Consultative (8+ Days)"
    df_sess['conversion_speed'] = df_sess.apply(categorize_speed, axis=1)

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    df_sess.drop(columns=['prev_session', 'days_since_last', 'is_new_journey', 'journey_id'], inplace=True, errors='ignore')
    if 'acquisition_source' not in df_sess.columns: df_sess['acquisition_source'] = df_sess.apply(classify_source, axis=1)
    
    # 🚨 FINAL EXPORT CLEANUP
    df_sess['phone_number'] = df_sess['Contact ID'].map(phone_map)
    
    # Remove duplicate columns if they exist (can happen due to merges and me not being keen lol)
    if list(df_sess.columns).count('recovered_source') > 1:
        df_sess = df_sess.loc[:, ~df_sess.columns.duplicated()]

    # 2.Ensure Ad columns exist (even if no matches were found)
    ad_cols = ['Ad campaign ID', 'Ad group ID', 'Ad ID']
    for col in ad_cols:
        if col not in df_sess.columns:
            df_sess[col] = None

    # 3. REORDERING: Place Ad Columns right after 'acquisition_source'
    print("✨ Finalizing Column Order...")
    cols = list(df_sess.columns)
    
    # Target order: Source info -> Ad Info -> Everything else
    target_block = ['acquisition_source', 'recovered_source', 'channel_name',
                    'clean_ad_id', 'Ad Name'] + ad_cols
    
    # Remove target columns from the list to avoid duplication
    for c in target_block:
        if c in cols: cols.remove(c)

    # Insert them back in specific position (e.g., after session_id or Contact ID)
    # Finding a good anchor point (usually index 2 or 3)
    anchor_idx = cols.index('session_id') + 1 if 'session_id' in cols else 2
    
    for i, col_name in enumerate(target_block):
        if col_name in df_sess.columns: # Safety check
            cols.insert(anchor_idx + i, col_name)
            
    # Apply the new order
    df_sess = df_sess[cols]
    

    df_sess.to_csv(PROCESSED_DATA_DIR / "fact_sessions_enriched.csv", index=False)
    stats.reset_index().to_csv(PROCESSED_DATA_DIR / "dim_customers_rfv.csv", index=False)

    # PRODUCT EXPORT
    p_rows = []
    for _, r in df_sess.iterrows():
        session_brands = normalize_brands_with_intent(r.get('temp_brands'), r['full_context'])
        if not session_brands: session_brands = [None] 
        found_products_in_session = set()
        for brand_hint in session_brands:
            match_row = matcher.get_best_match(r['full_context'], r['mpesa_amount'], brand_hint=brand_hint)
            if match_row is not None:
                prod_name = str(match_row['Name']).strip()
                brand_name = str(match_row['Brand']).strip()
                unique_key = f"{prod_name}_{brand_name}"
                if unique_key in found_products_in_session: continue
                found_products_in_session.add(unique_key)
                clean_product_name = prod_name
                if len(brand_name) > 2 and prod_name.lower().startswith(brand_name.lower()):
                    clean_product_name = prod_name[len(brand_name):].strip(" -:")
                p_rows.append({
                    "session_id": r['session_id'], "Contact ID": r['Contact ID'], "brand_name": brand_name,
                    "product_name": clean_product_name, "original_product_name": prod_name,
                    "category": r['primary_category'], "sub_category": match_row['Sub_Category'], 
                    "concern": match_row['Concerns'], "revenue": r['mpesa_amount'],
                    "is_converted": r['is_converted'], "date": r['session_start']
                })
    if p_rows: 
        pd.DataFrame(p_rows).to_csv(PROCESSED_DATA_DIR / "fact_product_mentions.csv", index=False)


    # BRAND MENTIONS EXPORT
    b_rows = []
    for _, r in df_sess.iterrows():
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
    if b_rows: 
        pd.DataFrame(b_rows).to_csv(PROCESSED_DATA_DIR / "fact_brand_mentions.csv", index=False)

    # FUNNEL EXPORT
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
    
    # ZONE EXPORT
    zone_rows = []
    for _, r in df_sess.iterrows():
        if r.get('zone_name'): zone_rows.append({"session_id": r['session_id'], "type": "Primary", "loc": r['zone_name']})
        if r.get('secondary_zones'):
            for sz in str(r['secondary_zones']).split("|"):
                if sz.strip(): zone_rows.append({"session_id": r['session_id'], "type": "Secondary", "loc": sz.strip()})
    if zone_rows: pd.DataFrame(zone_rows).to_csv(PROCESSED_DATA_DIR / "fact_session_zones.csv", index=False)

    # CONCERN EXPORT
    concern_rows = []
    for _, r in df_sess.iterrows():
        text_context = str(r['full_context']).lower()
        for concern, patterns in CONCERN_RULES.items():
            if any(re.search(pat, text_context) for pat in patterns.get('all', []) + patterns.get('chat', [])):
                concern_rows.append({"session_id": r['session_id'], "concern": concern, "date": r['session_start']})
    if concern_rows: pd.DataFrame(concern_rows).to_csv(PROCESSED_DATA_DIR / "fact_session_concerns.csv", index=False)


    print("✅ ANALYTICS COMPLETE.") 

if __name__ == "__main__":
    run_analytics_pipeline()