import pandas as pd
import numpy as np
import re
from pathlib import Path
from difflib import SequenceMatcher
from urllib.parse import unquote

# ==========================================
# 1. CONFIGURATION
# ==========================================
BASE_DIR = Path(__file__).resolve().parent.parent.parent
SESSIONS_FILE = BASE_DIR / "data" / "03_processed" / "fact_sessions_enriched.csv"
CATALOG_FILE = BASE_DIR / "data" / "01_raw" / "Final_Knowledge_Base_PowerBI.csv"
OUTPUT_FILE = BASE_DIR / "data" / "03_processed" / "report_staff_recommendations.csv"

# 🎚️ SETTINGS
FUZZY_THRESHOLD = 0.75  # High confidence only. Filters out noise.
CONTEXT_WINDOW_CHARS = 120 # Tight window (~15 words) to stop cross-brand reading.

# ✅ CHAT CONSTRAINTS (Strict inclusions)
MUST_HAVE_KEYWORDS = {
    'vitamin c', 'vit c', 'glycolic', 'salicylic', 'retinol', 'niacinamide', 'benzoyl', 
    'hyaluronic', 'azelaic', 'lactic', 'peptide', 'snail', 'collagen', 'arbutin',
    'eczema', 'acne', 'rosacea', 'psoriasis', 'baby', 'kid', 'infant'
}

# ✅ PRODUCT CONSTRAINTS
PRODUCT_RESTRICTIVE_VARIANTS = {
    'eye': ['eye', 'circle', 'puffiness'],
    'baby': ['baby', 'kid', 'child', 'infant', 'toddler'],
    'body': ['body', 'leg', 'hand', 'arm'], 
    'lip': ['lip', 'balm'],
    'spf': ['spf', 'sun', 'uv', 'protect'],
    'psoriasis': ['psoriasis', 'scale', 'scaly'],
    'eczema': ['eczema', 'itch', 'dermatitis'],
    'men': ['men', 'male', 'man', 'him'],
}

# ✅ FORM FACTOR GROUPS (Added Deodorant)
FORM_FACTOR_MAP = {
    'cleanser': {'cleanser', 'wash', 'scrub', 'foam', 'soap', 'bar', 'gel'},
    'wash': {'cleanser', 'wash', 'scrub', 'foam', 'soap', 'bar', 'gel'},
    'balm': {'balm', 'ointment', 'butter'}, 
    'lotion': {'lotion', 'hydrator', 'milk', 'moisturizer'}, 
    'cream': {'cream', 'balm', 'moisturizer'},       
    'moisturizer': {'moisturizer', 'cream', 'lotion', 'hydrator', 'balm', 'gel'}, 
    'serum': {'serum', 'drops', 'ampoule', 'oil'},
    'toner': {'toner', 'mist', 'essence', 'spray', 'liquid'},
    'sunscreen': {'sunscreen', 'sunblock', 'spf', 'uv', 'sun'},
    'deodorant': {'deodorant', 'roll on', 'roll-on', 'antiperspirant', 'spray', 'stick'},
}

GENERIC_STOP_WORDS = {
    'ml', 'g', 'oz', 'kg', 'mg', 'pack', 'pcs', 'size',
    'product', 'skin', 'care', 'face', 'routine', 'daily', 'day', 
    'formula', 'control', 'advanced', 'intensive', 'ultra', 'active', 'complex',
    'solution', 'suspension', 'treatment', 'water', 'acid', 'therapy', 'gentle', 'natural',
    'moisturizer', 'moisturising', 'hydrating', 'hydration', 'cleanser', 'wash', 'lotion', 
    'cream', 'serum', 'toner', 'head', 'toe'
}

SHORT_KEYWORD_WHITELIST = {'sa', 'ha', 'b5', 'b3', 'c', 'e', 'ph', 'spf', 'uv', 'aha', 'bha'}

CONCERN_RULES = {
    "Acne": { "all": [r"\bacne\b", r"\bacnes\b", r"\bpimple(s)?\b", r"\bbreakout(s)?\b", r"\bwhitehead(s)?\b", r"\bblackhead(s)?\b", r"\bbenzoyl peroxide?\b", r"\bsalicylic?\b", r"\beffaclar?\b", r"\bcomedonal\b", r"\bspot corrector\b",] },
    "Hyperpigmentation": { "all": [r"\bhyperpigmentation\b", r"\bdark spot(s)?\b", r"\bdark mark(s)?\b", r"\buneven (skin )?tone\b", r"\bbrightening\b", r"\bascorbic acid\b",  r"\bpigment\b", r"\bmela b3\b", r"\bmelasma\b"]},
    "Oily Skin": { "all": [r"\boily skin\b", r"\bfor oily\b", r"\bcontrols oil\b", r"\bsebum\b", r"\blarge pores\b", r"\bpores\b", r"\boily\b", r"\bcombination to oily\b"] },
    "Dry Skin": { "all": [r"\bdry skin\b", r"\bfor dry\b", r"\bextra dry\b", r"\bvery dry\b", r"\bintense hydration\b", r"\bmy skin is dry\b"] },
    "Sensitive Skin": { "all": [r"\bsensitive skin\b", r"\bfor sensitive\b", r"\banti-irritation\b", r"\bsoothing\b"] },
    "Sleep": { "all": [r"\bsleep\b", r"\bfor sleep\b", r"\binsomnia\b", r"\brestless night(s)?\b", r"\bmagnesium glycinate\b"] },
    "Hair Loss": { "all": [r"\bhair loss\b", r"\banti-hairloss\b", r"\bthinning hair\b", r"\bminoxidil\b", r"\balopecia\b"] },
    "Weight Management": { "all": [r"\bweight loss\b", r"\bweight gain\b", r"\blose weight\b", r"\bgain weight\b", r"\bslimming\b", r"\bcut belly\b", r"\btummy trimmer\b", r"\bappetite\b", r"\badd weight\b", r"\breduce weight\b", r"\bfat burner\b"] },
    "Eczema": { "all": [r"\beczema\b", r"\batopic dermatitis\b", r"\bdermatitis\b", r"\bitchy skin\b", r"\bred patches\b"] },
}

NEGATION_TERMS = {
    "don't have", "dont have", "out of stock", "finished", "unavailable", 
    "no we don't", "not available", "do not have", "sold out"
}

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def clean_chat_text(text):
    if not isinstance(text, str): return ""
    text = unquote(text.lower())
    text = re.sub(r'https?://\S+', lambda m: m.group(0).replace('-', ' ').replace('/', ' ').replace('?', ' '), text)
    text = re.sub(r'\b(hello|hi|hey|please|kindly|thanks|thank you|price|cost|how much|available)\b', '', text)
    return text

def get_product_anchors(product_name, brand_name):
    p_parts = str(product_name).lower().split()
    brand_parts = set(str(brand_name).lower().split())
    anchors = []
    for w in p_parts:
        if w in brand_parts: continue
        if w in GENERIC_STOP_WORDS: continue
        if w in FORM_FACTOR_MAP: continue 
        if len(w) < 3 and w not in SHORT_KEYWORD_WHITELIST: continue
        clean_w = re.sub(r'[^a-z0-9]', '', w)
        if clean_w: anchors.append(clean_w)
    return anchors

def check_negation_in_window(text, keyword):
    idx = text.find(keyword)
    if idx == -1: return False
    start = max(0, idx - 40) 
    context = text[start:idx]
    for neg in NEGATION_TERMS:
        if neg in context: return True
    return False

def check_strict_constraints(product_name, chat_text):
    p_name = str(product_name).lower()
    for keyword in MUST_HAVE_KEYWORDS:
        if re.search(r'\b' + re.escape(keyword) + r'\b', chat_text):
            if keyword not in p_name:
                if keyword == 'vit c' and 'vitamin c' in p_name: continue
                return False 
    for variant, synonyms in PRODUCT_RESTRICTIVE_VARIANTS.items():
        if re.search(r'\b' + re.escape(variant) + r'\b', p_name):
            if not any(syn in chat_text for syn in synonyms):
                return False 
    return True 

def check_form_factor_compliance(product_name, chat_text):
    p_words = set(str(product_name).lower().split())
    prod_forms = set()
    for w in p_words:
        if w in FORM_FACTOR_MAP: prod_forms.add(w)
    if not prod_forms: return True 
    
    # Conflict Logic
    wash_syns = FORM_FACTOR_MAP['wash']
    moist_syns = FORM_FACTOR_MAP['cream'].union(FORM_FACTOR_MAP['lotion']).union(FORM_FACTOR_MAP['balm'])
    deo_syns = FORM_FACTOR_MAP['deodorant']

    for p_form in prod_forms:
        if p_form in ['balm', 'cream', 'lotion', 'moisturizer']:
            if any(w in chat_text for w in wash_syns) and not any(w in chat_text for w in moist_syns): return False
        if p_form in ['wash', 'cleanser']:
            if any(w in chat_text for w in moist_syns) and not any(w in chat_text for w in wash_syns): return False
        # Deodorant Protection
        if p_form == 'deodorant':
            if not any(w in chat_text for w in deo_syns): return False
    return True

def identify_form(text):
    text = text.lower()
    for form, syns in FORM_FACTOR_MAP.items():
        if any(s in text for s in syns): return form
    return "other"

def get_brand_context_window(text, brand_name):
    text = text.lower()
    brand_name = brand_name.lower()
    starts = [m.start() for m in re.finditer(re.escape(brand_name), text)]
    if not starts: return "" 
    context_parts = []
    for start_idx in starts:
        window_start = max(0, start_idx - CONTEXT_WINDOW_CHARS)
        window_end = min(len(text), start_idx + len(brand_name) + CONTEXT_WINDOW_CHARS)
        context_parts.append(text[window_start:window_end])
    return " ... ".join(context_parts)

def get_local_concern_from_window(context_window):
    for concern, rules in CONCERN_RULES.items():
        for pattern in rules.get("all", []):
            if re.search(pattern, context_window, re.IGNORECASE):
                return concern
    return "General"

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

# ==========================================
# 3. MAIN EXTRACTION LOGIC
# ==========================================
def run_recommendation_extraction():
    print("🕵️  STARTING FINAL PRODUCT MINING (V53)...")

    if not SESSIONS_FILE.exists() or not CATALOG_FILE.exists():
        print("❌ Missing input files.")
        return

    df_sess = pd.read_csv(SESSIONS_FILE)
    df_catalog = pd.read_csv(CATALOG_FILE)
    
    if 'Brand' not in df_catalog.columns:
        print("❌ CRITICAL ERROR: 'Brand' column missing in catalog.")
        return

    print("   📚 Indexing Catalog...")
    catalog_map = {}
    df_catalog['Brand'] = df_catalog['Brand'].fillna('General').astype(str).str.strip().str.title()
    df_catalog['Concerns'] = df_catalog['Concerns'].fillna('General Care').astype(str).str.lower()
    
    for _, row in df_catalog.iterrows():
        b = row['Brand']
        if b not in catalog_map: catalog_map[b] = []
        catalog_map[b].append({
            'Name': row['Name'],
            'Concerns': str(row['Concerns']).lower(),
            'Price': row['Price']
        })

    df_target = df_sess[
        (df_sess['temp_brands'].notna()) | (df_sess['matched_brand'] != "Unknown")
    ].copy()
    
    print(f"   🔍 Analyzing {len(df_target)} sessions...")

    recommendations = []

    for _, row in df_target.iterrows():
        raw_text = str(row.get('full_context', ''))
        chat_text = clean_chat_text(raw_text) 
        
        brands_in_session = set()
        if row.get('matched_brand') and str(row['matched_brand']) != "Unknown":
            brands_in_session.add(str(row['matched_brand']).title())
        if row.get('temp_brands'):
            for b in str(row['temp_brands']).split('|'):
                clean_b = b.strip().title()
                if len(clean_b) > 2: brands_in_session.add(clean_b)

        session_matches = []
        
        for brand in brands_in_session:
            if brand not in catalog_map: continue
            raw_candidates = catalog_map[brand]
            
            # Context Isolation
            brand_context_text = get_brand_context_window(chat_text, brand)
            if not brand_context_text: continue 

            local_concern = get_local_concern_from_window(brand_context_text)
            effective_concern = local_concern if local_concern != "General" else str(row.get('matched_concern', 'General')).title()
            
            filtered_candidates = []
            if effective_concern in ["General", "Nan", "", "General Care"]:
                filtered_candidates = raw_candidates
            else:
                for prod in raw_candidates:
                    prod_concerns = prod['Concerns']
                    if (effective_concern.lower() in prod_concerns) or \
                       ('general' in prod_concerns) or \
                       ('moisturizer' in prod_concerns):
                        filtered_candidates.append(prod)
                if not filtered_candidates: filtered_candidates = raw_candidates

            for prod in filtered_candidates:
                p_name = prod['Name']
                anchors = get_product_anchors(p_name, brand)
                if not anchors: continue 
                
                hits = sum(1 for a in anchors if re.search(r'\b' + re.escape(a) + r'\b', brand_context_text))
                if hits == 0: continue
                
                # Check Constraints
                if not check_strict_constraints(p_name, brand_context_text): continue
                if any(check_negation_in_window(brand_context_text, a) for a in anchors): continue
                if not check_form_factor_compliance(p_name, brand_context_text): continue

                match_ratio = hits / len(anchors)
                score = 0.65 
                if match_ratio >= 0.8: score = 0.90
                elif match_ratio >= 0.5 and hits >= 2: score = 0.80
                elif hits == 1 and len(anchors) == 1: score = 0.75
                
                p_form = identify_form(p_name)
                c_form = identify_form(brand_context_text)
                
                if p_form != "other" and c_form != "other":
                    if p_form == c_form: score += 0.15 
                    elif p_form == 'moisturizer' and c_form in ['cream', 'lotion']: score += 0.10
                    else: score -= 0.25 

                if score >= FUZZY_THRESHOLD:
                    session_matches.append({
                        "Product": p_name,
                        "Price": prod['Price'],
                        "Brand": brand,
                        "Concern": effective_concern,
                        "Form": p_form,
                        "Score": score
                    })

        # --- DEDUPLICATION ---
        session_matches.sort(key=lambda x: x['Score'], reverse=True)
        final_picks = []
        seen_keys = set()
        
        best_brand_score = 0
        if session_matches: best_brand_score = session_matches[0]['Score']

        for m in session_matches:
            if best_brand_score >= 0.95 and m['Score'] < 0.75:
                continue

            key = f"{m['Brand']}_{m['Form']}"
            if key in seen_keys and m['Form'] != 'Other' and m['Score'] < 0.90:
                continue
                
            final_picks.append(m)
            seen_keys.add(key)
            if len(final_picks) >= 6: break 

        for pick in final_picks:
            recommendations.append({
                "Date": row['activity_date'],
                "Session ID": row['session_id'],
                "Customer Name": row['contact_name'],
                "Staff Name": row.get('sales_owner', 'Unassigned'),
                "Concern Identified": pick['Concern'], 
                "Brand Filter": pick['Brand'],
                "Recommended Product": pick['Product'],
                "Product Price": pick['Price'],
                "Chat Revenue": row['mpesa_amount'],
                "Is Converted": row['is_converted'],
                "Match Score": round(pick['Score'], 2),
                "Chat Context": raw_text[:1500] 
            })

    if recommendations:
        df_rec = pd.DataFrame(recommendations)
        df_rec.to_csv(OUTPUT_FILE, index=False)
        print(f"✅ Success! Extracted {len(df_rec)} recommendations.")
        print(f"   📄 Report saved to: {OUTPUT_FILE}")
    else:
        print("⚠️ No recommendations found matching the criteria.")

if __name__ == "__main__":
    run_recommendation_extraction()