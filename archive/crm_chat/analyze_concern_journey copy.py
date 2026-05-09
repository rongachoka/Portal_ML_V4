import pandas as pd
import numpy as np
import re
import json
from pathlib import Path
from urllib.parse import unquote
from difflib import SequenceMatcher

# ✅ IMPORT PATHS
from Portal_ML_V4.src.config.settings import (
    RAW_DATA_DIR,
    MSG_INTERIM_PARQUET,
    PROCESSED_DATA_DIR,
)

# ==========================================
# 1. CONFIGURATION
# ==========================================
MESSAGES_FILE = MSG_INTERIM_PARQUET
SESSIONS_FILE = PROCESSED_DATA_DIR / "fact_sessions_enriched.csv"
CATALOG_FILE = RAW_DATA_DIR / "Final_Knowledge_Base_PowerBI.csv"
OUTPUT_FILE = PROCESSED_DATA_DIR / "report_concern_journey_analysis.csv"

CONCERN_KEYWORDS = {
    "Acne": [r"\bacne\b", r"\bpimple", r"\bbreakout", r"\bwhitehead", r"\bblackhead", r"\bzits?\b", r"\bcyst(ic)?\b"],
    "Hyperpigmentation": [r"\bhyper[- ]?pigmentation\b", r"\bdark\s*(spot|mark|patch|area)s?\b", r"\b(un)?even\s*skin(\s*tone)?\b", r"\bpigmentation\b", r"\bmelasma\b", r"\bdiscoloration\b", r"\bpost[- ]?acne\s*marks?\b", r"\bbrown\s*spots?\b"],
    "Dry Skin": [r"\bdry\s*skin\b", r"\bflaky\b", r"\bdryness\b", r"\bdehydrated\b", r"\bpeeling\b"],
    "Oily Skin": [r"\boily\b", r"\bgreasy\b", r"\bshine\b", r"\bsebum\b", r"\blarge\s*pores\b"]
}   

OBJECTION_TERMS = ["expensive", "too much", "high price", "budget", "discount", 
                   "less", "lower", "reduce", "any offer", "best price", "broke", 
                   "costly"]

BRAND_ALIASES = {
    "lrp": "La Roche-Posay", "la roche": "La Roche-Posay", "la roche posay": "La Roche-Posay", "posay": "La Roche-Posay",
    "clean & clear": "Clean & Clear", "clean and clear": "Clean & Clear",
    "paulas choice": "Paulas Choice", "bgs": "Black Girl Sunscreen",
    "the ordinary": "The Ordinary", "ordinary": "The Ordinary"
}

# ✅ STRICT IDENTITY MAP (The Hallucination Killer)
STRICT_IDENTITY_MAP = {
    'baby': 'baby', 'kid': 'kid', 'unifiant': 'unifiant', 'tinted': 'tinted',
    'matte': 'matte', 'kit': 'kit', 'set': 'set', 'bundle': 'bundle',
    'rice': 'rice', 'snail': 'snail', 'patches': 'patches', 'medicated': 'medicated',
    'arbutin': 'arbutin', 'niacinamide': 'niacinamide', 'retinol': 'retinol',
    'retinoid': 'retinoid', 'matrixyl': 'matrixyl', 'peptide': 'peptide',
    'b5': 'b5', 'salicylic': 'salicylic', 'glycolic': 'glycolic',
    'azelaic': 'azelaic', 'benzoyl': 'benzoyl', 'vitamin c': 'vitamin c',
    'spf': 'spf', 'serum': 'serum', 'cleanser': 'cleanser', 'wash': 'wash',
    'toner': 'toner', 'moisturizer': 'moisturizer', 'sunscreen': 'sunscreen'
}

UNIQUE_PRODUCT_TOKENS = {
    'effaclar', 'panoxyl', 'toleriane', 'lipikar', 'anthelios', 
    'cicaplast', 'hydrabio', 'sebium', 'pigmentbio', 'atoderm', 'duo'
}

GENERIC_STOP_WORDS = {
    'ml', 'g', 'oz', 'kg', 'pack', 'pcs', 'size', 'product', 'skin', 'care', 'face', 
    'routine', 'formula', 'solution', 'treatment', 'therapy', 
    'bottle', 'tube', 'daily', 'gentle', 'repair',
    'clear', 'control', 'acne', 'oily', 'dry', 'ageing', 'anti', 
    'protection', 'hydrating', 'moisturising', 'soothing',
    'black', 'white', 'invisible', 'comfort', 'roll', 'on', 'stick', 'water', 'thermal', 'spring'
}

FORM_FACTOR_MAP = {
    'cleanser': {'cleanser', 'wash', 'scrub', 'foam', 'soap', 'bar', 'gel'},
    'moisturizer': {'moisturizer', 'cream', 'lotion', 'hydrator', 'balm', 'gel', 'milk', 'emollient'}, 
    'serum': {'serum', 'drops', 'ampoule', 'oil', 'concentrate'},
    'toner': {'toner', 'mist', 'essence', 'spray', 'liquid'},
    'sunscreen': {'sunscreen', 'sunblock', 'spf', 'uv', 'sun', 'fluid', 'invisible'},
    'lip_care': {'lip', 'balm', 'stick', 'chapstick'},
    'deodorant': {'deodorant', 'roll on', 'roll-on', 'antiperspirant', 'spray', 'stick'}, 
    'supplement': {'supplement', 'capsule', 'tablet', 'pill', 'gummy', 'vitamin'}
}

CONTEXT_REJECTIONS = {
    'face': ['deodorant', 'roll on', 'antiperspirant', 'body wash'],
    'acne': ['deodorant', 'roll on', 'antiperspirant', 'lip_care'],
    'lip': ['cleanser', 'toner', 'serum'], 
}

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def normalize_join_id(val):
    """
    Strips Excel formatting (' and +) to create a clean join key
    Example: "'+2547..." -> "2547..."
    """
    s = str(val).replace("'", "").replace("+", "").replace(".0", "").strip()
    return s

def normalize_brand_names(text):
    text_lower = text.lower()
    for alias, canonical in BRAND_ALIASES.items():
        if alias in text_lower:
            text_lower = text_lower.replace(alias, canonical.lower())
    return text_lower

def parse_and_clean_content(raw_content):
    if not isinstance(raw_content, str): return ""
    text_content = raw_content
    try:
        if raw_content.strip().startswith('{') and 'type' in raw_content:
            data = json.loads(raw_content)
            if data.get('type') == 'text': text_content = data.get('text', '')
            elif data.get('type') == 'attachment': text_content = "[IMAGE/FILE]"
    except: pass
    return unquote(text_content).replace('\n', ' ').strip()

def extract_text_from_urls(text):
    slugs = []
    urls = re.findall(r'https?://[^\s<>"]+|www\.[^\s<>"]+', text)
    for url in urls:
        clean = unquote(url)
        parts = re.split(r'[/?=&]', clean)
        for p in parts:
            if '-' in p and '.' not in p: slugs.append(p.replace('-', ' '))
    return " ".join(slugs)

def clean_text_for_matching(text):
    text = parse_and_clean_content(text)
    text = normalize_brand_names(text)
    url_text = extract_text_from_urls(text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'\b(hello|hi|hey|please|kindly|thanks|thank you)\b', '', text)
    return (text + " " + url_text).strip()

def get_product_anchors(product_name, brand_name):
    p_parts = str(product_name).lower().split()
    brand_parts = set(str(brand_name).lower().split())
    anchors = []
    for w in p_parts:
        if w in brand_parts: continue
        if w in GENERIC_STOP_WORDS: continue
        if len(w) < 3: continue
        anchors.append(re.sub(r'[^a-z0-9]', '', w))
    return anchors

def get_concern_type(text):
    for concern, patterns in CONCERN_KEYWORDS.items():
        for pat in patterns:
            if re.search(pat, text): return concern
    return None

def extract_brands_in_text(text_normalized, known_brands_set):
    found = set()
    for brand in known_brands_set:
        brand_clean = brand.lower()
        if brand_clean in text_normalized:
            if re.search(r'\b' + re.escape(brand_clean) + r'\b', text_normalized):
                found.add(brand)
    return list(found)

def identify_form(text):
    text = text.lower()
    for form, syns in FORM_FACTOR_MAP.items():
        if any(s in text for s in syns): return form
    return "other"

def check_context_rejection(product_name, chat_text):
    p_form = identify_form(product_name)
    c_lower = chat_text.lower()
    for context_trigger, forbidden_forms in CONTEXT_REJECTIONS.items():
        if context_trigger in c_lower:
            if p_form in forbidden_forms: return False 
    if p_form == 'deodorant' and not any(x in c_lower for x in ['roll', 'deodorant', 'sweat', 'pit', 'arm']):
        return False
    return True

def check_form_compliance(product_name, chat_text):
    p_form = identify_form(product_name)
    c_form = identify_form(chat_text)
    if c_form == "other" or p_form == "other": return True
    if p_form == c_form: return True
    if p_form == 'moisturizer' and c_form in ['lotion', 'cream', 'balm']: return True
    if c_form == 'moisturizer' and p_form in ['lotion', 'cream', 'balm']: return True
    return False

def check_identity_mismatch(product_name, text):
    p_name_lower = product_name.lower()
    text_lower = text.lower()
    for token, required_text in STRICT_IDENTITY_MAP.items():
        if token in p_name_lower:
            if required_text not in text_lower:
                return True
    return False

def extract_prices_from_text(text):
    matches = re.findall(r'\b\d{1,2},?\d{3}\b', text)
    prices = []
    for m in matches:
        try:
            val = float(m.replace(',', ''))
            if 500 <= val <= 100000: prices.append(val)
        except: pass
    return prices

def mask_product_in_text(original_text, product, brand, matched_price=None):
    masked_text = original_text
    if matched_price:
        price_str = str(int(matched_price))
        masked_text = re.sub(r'\b' + price_str + r'\b', 'xxxx', masked_text)
    brand_lower = brand.lower()
    masked_text = re.sub(re.escape(brand_lower), 'xxxx', masked_text, count=1, flags=re.IGNORECASE)
    anchors = get_product_anchors(product['name'], brand)
    for anchor in anchors:
        masked_text = re.sub(re.escape(anchor), 'xxxx', masked_text, count=1, flags=re.IGNORECASE)
    return masked_text

def find_products_iterative_v21(text, product_lookup, known_brands_set, context_brands, cust_query=""):
    working_text = clean_text_for_matching(text)
    full_context_text = working_text
    if cust_query: full_context_text += " " + clean_text_for_matching(cust_query)
        
    found_products = []
    matched_names = set()
    
    for _ in range(5):
        explicit_brands = extract_brands_in_text(working_text, known_brands_set)
        mentioned_prices = extract_prices_from_text(working_text)
        active_brands = explicit_brands if explicit_brands else list(context_brands)
        if not active_brands: break

        candidates = []
        for brand in active_brands:
            brand_candidates = [p for p in product_lookup if p['brand'] == brand]
            for prod in brand_candidates:
                if prod['name'] in matched_names: continue
                score = 0
                anchors = prod['anchors']
                if not anchors: continue
                
                if not check_form_compliance(prod['name'], full_context_text): continue
                if not check_context_rejection(prod['name'], full_context_text): continue
                if check_identity_mismatch(prod['name'], full_context_text): continue 
                
                price_hit = None
                if mentioned_prices:
                    for price in mentioned_prices:
                        tolerance = 0.02 if price > 2000 else 0.05
                        if (1-tolerance) * prod['price'] <= price <= (1+tolerance) * prod['price']:
                            score += 200; price_hit = price; break
                
                hits = sum(1 for a in anchors if a in working_text)
                if hits > 0: score += (hits * 20)
                sim = SequenceMatcher(None, prod['name'].lower(), full_context_text).ratio()
                score += (sim * 20)

                if price_hit:
                    brand_lower = brand.lower()
                    has_brand = brand_lower in full_context_text
                    has_unique_token = any(t in prod['name'].lower() and t in full_context_text for t in UNIQUE_PRODUCT_TOKENS)
                    if not has_brand and not has_unique_token:
                        if score < 250: continue 

                min_score = 150 if price_hit else 50
                if score >= min_score:
                    candidates.append((score, prod, brand, price_hit))

        if not candidates: break
        candidates.sort(key=lambda x: x[0], reverse=True)
        winner_score, winner_prod, winner_brand, winner_price = candidates[0]
        found_products.append(winner_prod)
        matched_names.add(winner_prod['name'])
        working_text = mask_product_in_text(working_text, winner_prod, winner_brand, winner_price)
    
    return found_products

def generate_full_session_transcript(all_messages, start_time, end_time):
    mask = (all_messages['Date & Time'] >= start_time) & (all_messages['Date & Time'] <= end_time)
    session_msgs = all_messages[mask].sort_values('Date & Time')
    lines = []
    for _, msg in session_msgs.iterrows():
        ts = msg['Date & Time'].strftime('%d/%m %H:%M')
        sender = "Staff" if str(msg['Sender Type']).lower() in ['user', 'staff', 'agent'] else "Cust"
        content = parse_and_clean_content(msg['Content'])
        lines.append(f"[{ts}] {sender}: {content}")
    return "\n".join(lines)

# ==========================================
# 3. MAIN LOGIC
# ==========================================
def run_journey_analysis():
    print("🕵️  STARTING CONCERN JOURNEY ANALYSIS (V21 - Strict Isolation)...")

    if not MESSAGES_FILE.exists() or not SESSIONS_FILE.exists() or not CATALOG_FILE.exists():
        print("❌ Missing input files.")
        return

    print("   📚 Indexing Catalog...")
    df_catalog = pd.read_csv(CATALOG_FILE)
    df_catalog['Brand'] = df_catalog['Brand'].fillna('General').astype(str).str.strip().str.title()
    df_catalog['Price'] = pd.to_numeric(df_catalog['Price'], errors='coerce').fillna(0)
    
    known_brands_set = set(df_catalog['Brand'].unique())
    known_brands_set.update(BRAND_ALIASES.values())
    
    product_lookup = []
    for _, row in df_catalog.iterrows():
        brand = row['Brand']
        anchors = get_product_anchors(row['Name'], brand)
        if anchors:
            product_lookup.append({
                'name': row['Name'],
                'brand': brand,
                'anchors': anchors,
                'price': float(row['Price'])
            })

    print("   📥 Loading Data...")
    df_sess = pd.read_csv(SESSIONS_FILE)
    df_sess['session_start'] = pd.to_datetime(df_sess['session_start'])
    
    # 🚨 FIX START: NORMALIZE ID FOR MATCHING
    # The session file has formatted IDs ("'+254..."), but raw messages have raw IDs ("254...")
    # We create a temporary '_join_id' column to bridge the gap.
    df_sess['_join_id'] = df_sess['Contact ID'].apply(normalize_join_id)
    
    df_msg = pd.read_parquet(MESSAGES_FILE)
    df_msg['Date & Time'] = pd.to_datetime(df_msg['Date & Time'])
    df_msg['_join_id'] = df_msg['Contact ID'].apply(normalize_join_id)
    
    # Filter messages using the clean key
    valid_ids = set(df_sess['_join_id'].unique())
    df_msg = df_msg[df_msg['_join_id'].isin(valid_ids)]
    df_msg = df_msg.sort_values(['_join_id', 'Date & Time'])

    # Build Session Map using the clean key
    session_map = {}
    for _, sess in df_sess.iterrows():
        cid = sess['_join_id']
        if cid not in session_map: session_map[cid] = []
        session_map[cid].append({
            'id': sess['session_id'],
            'start': sess['session_start'],
            'converted': sess['is_converted'],
            'revenue': sess['mpesa_amount']
        })
    # 🚨 FIX END

    journey_rows = []
    print(f"   🔍 Analyzing conversations...")

    # Group by the normalized ID
    for contact_id, group in df_msg.groupby('_join_id'):
        if contact_id not in session_map: continue
        msgs_to_scan = group.reset_index(drop=True)
        active_session_brands = set()
        
        i = 0
        while i < len(msgs_to_scan):
            msg = msgs_to_scan.iloc[i]
            text_clean = clean_text_for_matching(msg['Content'])
            
            brands_here = extract_brands_in_text(text_clean, known_brands_set)
            if brands_here: active_session_brands.update(brands_here)
                
            sender = str(msg['Sender Type']).lower()
            msg_time = msg['Date & Time']
            
            current_session = None
            for s in session_map[contact_id]:
                if s['start'] - pd.Timedelta(hours=1) <= msg_time <= s['start'] + pd.Timedelta(hours=48):
                    current_session = s; break
            
            if not current_session:
                active_session_brands.clear(); i += 1; continue

            if sender == 'contact':
                concern_found = get_concern_type(text_clean)
                if concern_found:
                    j = i + 1
                    staff_reply_parts = []
                    found_recs = []
                    last_cust_query = text_clean
                    
                    while j < len(msgs_to_scan) and j < i + 5:
                        next_msg = msgs_to_scan.iloc[j]
                        next_text = clean_text_for_matching(next_msg['Content'])
                        next_sender = str(next_msg['Sender Type']).lower()
                        
                        st_brands = extract_brands_in_text(next_text, known_brands_set)
                        if st_brands: active_session_brands.update(st_brands)

                        if next_sender == 'contact':
                            if get_concern_type(next_text): break
                            last_cust_query = next_text
                        
                        if next_sender in ['user', 'staff', 'agent']:
                            staff_reply_parts.append(parse_and_clean_content(next_msg['Content']))
                            
                            recs = find_products_iterative_v21(
                                next_msg['Content'], 
                                product_lookup, 
                                known_brands_set,
                                active_session_brands,
                                cust_query=last_cust_query
                            )
                            if recs: found_recs.extend(recs)
                        j += 1
                    
                    if found_recs:
                        seen = set()
                        unique_recs = []
                        for p in found_recs:
                            if p['name'] not in seen:
                                unique_recs.append(p)
                                seen.add(p['name'])
                        
                        objection = "No"
                        k = j
                        while k < len(msgs_to_scan) and k < j + 3:
                            react = msgs_to_scan.iloc[k]
                            if str(react['Sender Type']).lower() == 'contact':
                                react_clean = clean_text_for_matching(react['Content'])
                                if any(o in react_clean for o in OBJECTION_TERMS):
                                    objection = "Yes"; break
                            k += 1

                        full_transcript = generate_full_session_transcript(group, current_session['start'] - pd.Timedelta(hours=1), current_session['start'] + pd.Timedelta(hours=48))
                        readable_q = parse_and_clean_content(msg['Content'])

                        for prod in unique_recs:
                            journey_rows.append({
                                "Date": current_session['start'].strftime('%Y-%m-%d'),
                                "Session ID": current_session['id'],
                                "Contact ID": contact_id, # This is the clean ID used for processing
                                "Concern Category": concern_found,
                                "Customer Question": readable_q[:300],
                                "Staff Reply": " | ".join(staff_reply_parts)[:500],
                                "Chat Context (Full)": full_transcript, 
                                "Recommended Brand": prod['brand'],
                                "Recommended Product": prod['name'],
                                "Product Price": prod['price'],
                                "Price Objection": objection,
                                "Is Converted": current_session['converted'],
                                "Revenue Captured": current_session['revenue'] if current_session['converted'] else 0
                            })
            i += 1

    if journey_rows:
        df_out = pd.DataFrame(journey_rows)
        df_out.to_csv(OUTPUT_FILE, index=False)
        print(f"✅ Success! Extracted {len(df_out)} strict recommendations.")
    else:
        print("⚠️ No journey data found.")

if __name__ == "__main__":
    run_journey_analysis()