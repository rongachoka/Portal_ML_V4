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
SESSIONS_FILE    = BASE_DIR / "data" / "03_processed" / "fact_sessions_enriched.csv"
CATALOG_FILE     = BASE_DIR / "data" / "01_raw"       / "Final_Knowledge_Base_PowerBI.csv"
AI_CONCERNS_FILE = BASE_DIR / "data" / "03_processed" / "ai_daily_session_concerns.csv"
SOCIAL_SALES_FILE= BASE_DIR / "data" / "03_processed" / "sales_attribution" / "social_sales_direct.csv"
OUTPUT_FILE      = BASE_DIR / "data" / "03_processed" / "report_staff_recommendations.csv"

# 🎚️ SETTINGS
FUZZY_THRESHOLD      = 0.75   # High confidence only
CONTEXT_WINDOW_CHARS = 120    # ~15 words around brand mention
SALE_MATCH_DAYS      = 2      # POS sale within N days of session counts as converted

# ✅ CHAT CONSTRAINTS
MUST_HAVE_KEYWORDS = {
    'vitamin c', 'vit c', 'glycolic', 'salicylic', 'retinol', 'niacinamide', 'benzoyl',
    'hyaluronic', 'azelaic', 'lactic', 'peptide', 'snail', 'collagen', 'arbutin',
    'eczema', 'acne', 'rosacea', 'psoriasis', 'baby', 'kid', 'infant'
}

PRODUCT_RESTRICTIVE_VARIANTS = {
    'eye':      ['eye', 'circle', 'puffiness'],
    'baby':     ['baby', 'kid', 'child', 'infant', 'toddler'],
    'body':     ['body', 'leg', 'hand', 'arm'],
    'lip':      ['lip', 'balm'],
    'spf':      ['spf', 'sun', 'uv', 'protect'],
    'psoriasis':['psoriasis', 'scale', 'scaly'],
    'eczema':   ['eczema', 'itch', 'dermatitis'],
    'men':      ['men', 'male', 'man', 'him'],
}

FORM_FACTOR_MAP = {
    'cleanser':    {'cleanser', 'wash', 'scrub', 'foam', 'soap', 'bar', 'gel'},
    'wash':        {'cleanser', 'wash', 'scrub', 'foam', 'soap', 'bar', 'gel'},
    'balm':        {'balm', 'ointment', 'butter'},
    'lotion':      {'lotion', 'hydrator', 'milk', 'moisturizer'},
    'cream':       {'cream', 'balm', 'moisturizer'},
    'moisturizer': {'moisturizer', 'cream', 'lotion', 'hydrator', 'balm', 'gel'},
    'serum':       {'serum', 'drops', 'ampoule', 'oil'},
    'toner':       {'toner', 'mist', 'essence', 'spray', 'liquid'},
    'sunscreen':   {'sunscreen', 'sunblock', 'spf', 'uv', 'sun'},
    'deodorant':   {'deodorant', 'roll on', 'roll-on', 'antiperspirant', 'spray', 'stick'},
    'supplement':  {'supplement', 'capsule', 'tablet', 'pill', 'gummy', 'vitamin'},
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

# Fallback regex concerns — used only when AI concern is unavailable
CONCERN_RULES_FALLBACK = {
    "Acne":              {"all": [r"\bacne\b", r"\bacnes\b", r"\bpimple(s)?\b", r"\bbreakout(s)?\b", r"\bbenzoyl peroxide?\b", r"\bsalicylic?\b"]},
    "Hyperpigmentation": {"all": [r"\bhyperpigmentation\b", r"\bdark spot(s)?\b", r"\bdark mark(s)?\b", r"\bbrightening\b", r"\bmelasma\b"]},
    "Oily Skin":         {"all": [r"\boily skin\b", r"\bsebum\b", r"\blarge pores\b", r"\boily\b"]},
    "Dry Skin":          {"all": [r"\bdry skin\b", r"\bintense hydration\b", r"\bmy skin is dry\b"]},
    "Sensitive Skin":    {"all": [r"\bsensitive skin\b", r"\bsoothing\b"]},
    "Sleep":             {"all": [r"\binsomnia\b", r"\brestless night(s)?\b", r"\bmagnesium glycinate\b"]},
    "Hair Loss":         {"all": [r"\bhair loss\b", r"\bthinning hair\b", r"\balopecia\b"]},
    "Weight Management": {"all": [r"\bweight loss\b", r"\blose weight\b", r"\bfat burner\b"]},
    "Eczema":            {"all": [r"\beczema\b", r"\batopic dermatitis\b", r"\bitchy skin\b"]},
}

NEGATION_TERMS = {
    "don't have", "dont have", "out of stock", "finished", "unavailable",
    "no we don't", "not available", "do not have", "sold out"
}

# ==========================================
# 2. LOAD ENRICHMENT DATA
# ==========================================

def load_ai_concerns() -> dict:
    """Returns {session_id: ai_inferred_concern}."""
    if not AI_CONCERNS_FILE.exists():
        print("   ⚠️  AI concerns file not found — falling back to regex only.")
        return {}
    df = pd.read_csv(AI_CONCERNS_FILE)
    df = df.dropna(subset=["session_id", "ai_inferred_concern"])
    df["session_id"] = df["session_id"].astype(str).str.strip()
    # Exclude "Not Analyzed" and "General Care" — these add no signal
    df = df[~df["ai_inferred_concern"].isin(["Not Analyzed", "General Care"])]
    print(f"   ✅ AI concerns loaded: {len(df):,} sessions with specific concerns")
    return df.set_index("session_id")["ai_inferred_concern"].to_dict()


def load_social_sales() -> pd.DataFrame:
    """
    Loads social_sales_direct.csv and returns a clean lookup table.
    Keyed on normalized phone + date for joining to recommendations.
    """
    if not SOCIAL_SALES_FILE.exists():
        print("   ⚠️  social_sales_direct.csv not found — sale linkage disabled.")
        return pd.DataFrame()

    df = pd.read_csv(SOCIAL_SALES_FILE, low_memory=False)
    df["Sale_Date"] = pd.to_datetime(df["Sale_Date"], errors="coerce")

    # Normalize phone to last 9 digits — same logic as the rest of the pipeline
    def norm_phone(val):
        if pd.isna(val): return None
        s = re.sub(r"[^\d]", "", str(val))
        return s[-9:] if len(s) >= 9 else None

    df["norm_phone"] = df["Phone Number"].apply(norm_phone)
    df = df.dropna(subset=["Sale_Date"])

    keep = ["Sale_Date", "norm_phone", "Transaction ID",
            "Matched_Brand", "Matched_Product", "Matched_Category",
            "Total (Tax Ex)", "Location"]
    available = [c for c in keep if c in df.columns]
    print(f"   ✅ Social sales loaded: {len(df):,} line items, "
          f"{df['Transaction ID'].nunique():,} transactions")
    return df[available].copy()


# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================

def clean_chat_text(text):
    if not isinstance(text, str): return ""
    text = unquote(text.lower())
    text = re.sub(r'https?://\S+', lambda m: m.group(0).replace('-', ' ').replace('/', ' '), text)
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
    context = text[max(0, idx - 40):idx]
    return any(neg in context for neg in NEGATION_TERMS)


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
    prod_forms = {w for w in p_words if w in FORM_FACTOR_MAP}
    if not prod_forms: return True
    wash_syns  = FORM_FACTOR_MAP['wash']
    moist_syns = FORM_FACTOR_MAP['cream'] | FORM_FACTOR_MAP['lotion'] | FORM_FACTOR_MAP['balm']
    deo_syns   = FORM_FACTOR_MAP['deodorant']
    for p_form in prod_forms:
        if p_form in ['balm', 'cream', 'lotion', 'moisturizer']:
            if any(w in chat_text for w in wash_syns) and not any(w in chat_text for w in moist_syns):
                return False
        if p_form in ['wash', 'cleanser']:
            if any(w in chat_text for w in moist_syns) and not any(w in chat_text for w in wash_syns):
                return False
        if p_form == 'deodorant':
            if not any(w in chat_text for w in deo_syns): return False
    return True


def identify_form(text):
    text = str(text).lower()
    for form, syns in FORM_FACTOR_MAP.items():
        if any(s in text for s in syns): return form
    return "other"


def get_brand_context_window(text, brand_name):
    text = text.lower()
    brand_name = brand_name.lower()
    starts = [m.start() for m in re.finditer(re.escape(brand_name), text)]
    if not starts: return ""
    parts = []
    for s in starts:
        parts.append(text[max(0, s - CONTEXT_WINDOW_CHARS):min(len(text), s + len(brand_name) + CONTEXT_WINDOW_CHARS)])
    return " ... ".join(parts)


def get_regex_concern(context_window: str) -> str:
    """Fallback: regex-based concern detection from the context window."""
    for concern, rules in CONCERN_RULES_FALLBACK.items():
        for pattern in rules.get("all", []):
            if re.search(pattern, context_window, re.IGNORECASE):
                return concern
    return "General"


def resolve_concern(session_id: str, context_window: str, ai_concern_map: dict) -> tuple[str, str]:
    """
    Returns (concern_label, concern_source).
    Priority:
      1. AI inferred concern (specific, from Gemma analysis of full session)
      2. Regex fallback from context window
      3. "General"
    """
    ai = ai_concern_map.get(str(session_id).strip())
    if ai and ai not in ("General Care", "Not Analyzed", ""):
        primary = ai.split("|")[0].strip()
        return primary, "AI"

    regex = get_regex_concern(context_window)
    if regex != "General":
        return regex, "Regex"

    return "General", "None"


def resolve_final_concern(
    concern_identified: str,
    concern_source: str,
    purchased_concerns: str
) -> tuple[str, str]:
    """
    After POS linkage, enriches the concern using KB concerns from the
    purchased product. What the customer actually bought is ground truth.

    Priority:
      1. KB concern from purchased product — if it confirms what was already
         identified, label accordingly. If different, KB wins.
      2. AI / Regex concern already identified
      3. "General"

    Returns (final_concern, final_source).
    """
    if purchased_concerns:
        kb_concerns = [
            c.strip() for c in purchased_concerns.split("|")
            if c.strip().lower() not in ("general care", "general", "nan", "")
        ]
        if kb_concerns:
            # Check if KB concern confirms what was already identified
            for kbc in kb_concerns:
                if (concern_identified.lower() in kbc.lower()
                        or kbc.lower() in concern_identified.lower()):
                    return kbc, f"KB confirms {concern_source}"
            # Different concern in KB — KB wins as ground truth
            return kb_concerns[0], "KB (Purchased Product)"

    # No KB enrichment available — keep existing concern
    return concern_identified, concern_source


def resolve_final_concern(
    concern_identified: str,
    concern_source: str,
    purchased_concerns: str
) -> tuple[str, str]:
    """
    After POS linkage, enriches the concern using KB concerns from the
    purchased product. What the customer actually bought is ground truth.

    Priority:
      1. KB concern from purchased product — if it confirms what was identified, keep it
         and label source accordingly. If different, KB wins (purchase is the proof).
      2. AI / Regex concern already identified
      3. "General"

    Returns (final_concern, final_source).
    """
    if purchased_concerns:
        kb_concerns = [
            c.strip() for c in purchased_concerns.split("|")
            if c.strip().lower() not in ("general care", "general", "nan", "")
        ]
        if kb_concerns:
            # Check if KB concern confirms what was already identified
            for kbc in kb_concerns:
                if (concern_identified.lower() in kbc.lower()
                        or kbc.lower() in concern_identified.lower()):
                    return kbc, f"KB confirms {concern_source}"
            # Different concern in KB — KB wins as ground truth
            return kb_concerns[0], "KB (Purchased Product)"

    # No KB enrichment — keep existing concern
    return concern_identified, concern_source


def find_pos_sale(session_row, social_sales: pd.DataFrame) -> dict:
    """
    Matches a session to a POS transaction using THREE signals:
      1. Phone number (last 9 digits)
      2. Date proximity (within SALE_MATCH_DAYS)
      3. Amount — transaction total must be within 2% of mpesa_amount

    All three must pass. This prevents cross-linking different customers
    who happen to share a phone suffix or bought on the same day.
    """
    if social_sales.empty: return {}

    # 1. Normalize phone
    raw_phone = str(session_row.get("phone_number", "")).strip().lstrip("'").replace("+", "")
    norm = re.sub(r"[^\d]", "", raw_phone)
    norm = norm[-9:] if len(norm) >= 9 else None
    if not norm: return {}

    # 2. Date window
    session_date = pd.to_datetime(session_row.get("session_start"), errors="coerce")
    if pd.isna(session_date): return {}

    window_start = session_date - pd.Timedelta(hours=1)
    window_end   = session_date + pd.Timedelta(days=SALE_MATCH_DAYS)

    phone_date_matches = social_sales[
        (social_sales["norm_phone"] == norm) &
        (social_sales["Sale_Date"] >= window_start) &
        (social_sales["Sale_Date"] <= window_end)
    ]
    if phone_date_matches.empty: return {}

    # 3. Amount validation — group to transaction level first, then compare
    # mpesa_amount is what the customer sent via WhatsApp
    mpesa_amt = pd.to_numeric(session_row.get("mpesa_amount", 0), errors="coerce")
    if pd.isna(mpesa_amt) or mpesa_amt <= 0:
        # No payment recorded on chat side — can't validate amount, skip
        return {}

    txn_totals = (
        phone_date_matches
        .groupby("Transaction ID")["Total (Tax Ex)"]
        .sum()
        .reset_index()
        .rename(columns={"Total (Tax Ex)": "txn_total"})
    )

    # Require amount within 2% tolerance
    txn_totals["within_tolerance"] = (
        (txn_totals["txn_total"] - mpesa_amt).abs() / mpesa_amt <= 0.02
    )
    confirmed_txns = txn_totals[txn_totals["within_tolerance"]]["Transaction ID"].tolist()

    if not confirmed_txns: return {}

    # Pull line items only for confirmed transactions
    confirmed = phone_date_matches[
        phone_date_matches["Transaction ID"].isin(confirmed_txns)
    ]

    return {
        "pos_confirmed_sale":   True,
        "pos_brands_purchased": " | ".join(confirmed["Matched_Brand"].dropna().unique()),
        "pos_products_purchased": " | ".join(confirmed["Matched_Product"].dropna().unique()) if "Matched_Product" in confirmed.columns else "",
        "pos_revenue":          confirmed["Total (Tax Ex)"].sum(),
        "pos_location":         confirmed["Location"].iloc[0] if "Location" in confirmed.columns else "",
        "pos_transaction_ids":  " | ".join(confirmed["Transaction ID"].astype(str).unique()),
    }


def get_purchased_product_concerns(pos_data: dict, catalog_map: dict) -> str:
    """
    Looks up the KB concerns for each product confirmed in the POS sale.
    Returns a pipe-separated string of unique concerns from purchased products.
    """
    products_str = pos_data.get("pos_products_purchased", "")
    if not products_str: return ""

    product_names = [p.strip() for p in products_str.split("|") if p.strip()]
    all_concerns = []

    for product_name in product_names:
        for brand_products in catalog_map.values():
            for prod in brand_products:
                if str(prod['Name']).strip().lower() == product_name.lower():
                    concerns = str(prod['Concerns']).strip()
                    if concerns and concerns.lower() not in ("general care", "general", "nan", ""):
                        all_concerns.append(concerns)
                    break

    if not all_concerns: return ""

    # Flatten and deduplicate individual concerns
    unique = []
    seen = set()
    for block in all_concerns:
        for c in block.split(","):
            c = c.strip()
            if c and c.lower() not in seen:
                seen.add(c.lower())
                unique.append(c)

    return " | ".join(unique)


# ==========================================
# 4. MAIN EXTRACTION LOGIC
# ==========================================

def run_recommendation_extraction():
    print("🕵️  STARTING STAFF RECOMMENDATIONS (V2 - AI Concerns + POS Sales)...")

    if not SESSIONS_FILE.exists() or not CATALOG_FILE.exists():
        print("❌ Missing input files.")
        return

    # Load enrichment data
    print("\n📂 Loading enrichment data...")
    ai_concern_map = load_ai_concerns()
    social_sales   = load_social_sales()

    print("\n📂 Loading sessions and catalog...")
    df_sess    = pd.read_csv(SESSIONS_FILE)
    df_catalog = pd.read_csv(CATALOG_FILE)

    if 'Brand' not in df_catalog.columns:
        print("❌ CRITICAL ERROR: 'Brand' column missing in catalog.")
        return

    print("   📚 Indexing Catalog...")
    catalog_map = {}
    df_catalog['Brand']    = df_catalog['Brand'].fillna('General').astype(str).str.strip().str.title()
    df_catalog['Concerns'] = df_catalog['Concerns'].fillna('General Care').astype(str).str.lower()

    for _, row in df_catalog.iterrows():
        b = row['Brand']
        if b not in catalog_map: catalog_map[b] = []
        catalog_map[b].append({
            'Name':     row['Name'],
            'Concerns': str(row['Concerns']).lower(),
            'Price':    row['Price']
        })

    # Filter to sessions that have brand context
    df_target = df_sess[
        df_sess['temp_brands'].notna() | (df_sess['matched_brand'] != "Unknown")
    ].copy()

    print(f"   🔍 Analyzing {len(df_target):,} sessions...")

    recommendations = []
    ai_used  = 0
    reg_used = 0

    for _, row in df_target.iterrows():
        raw_text  = str(row.get('full_context', ''))
        chat_text = clean_chat_text(raw_text)

        # Collect brands for this session
        brands_in_session = set()
        if row.get('matched_brand') and str(row['matched_brand']) != "Unknown":
            brands_in_session.add(str(row['matched_brand']).title())
        if row.get('temp_brands'):
            for b in str(row['temp_brands']).split('|'):
                cb = b.strip().title()
                if len(cb) > 2: brands_in_session.add(cb)

        session_matches = []

        for brand in brands_in_session:
            if brand not in catalog_map: continue

            brand_context = get_brand_context_window(chat_text, brand)
            if not brand_context: continue

            # ── CONCERN RESOLUTION ──────────────────────────────────────────
            concern, concern_source = resolve_concern(
                row['session_id'], brand_context, ai_concern_map
            )
            if concern_source == "AI":
                ai_used += 1
            elif concern_source == "Regex":
                reg_used += 1

            # Filter catalog by concern where possible
            raw_candidates = catalog_map[brand]
            if concern not in ("General", "General Care"):
                filtered = [
                    p for p in raw_candidates
                    if concern.lower() in p['Concerns'] or 'general' in p['Concerns']
                ]
                candidates = filtered if filtered else raw_candidates
            else:
                candidates = raw_candidates

            for prod in candidates:
                p_name  = prod['Name']
                anchors = get_product_anchors(p_name, brand)
                if not anchors: continue

                hits = sum(1 for a in anchors if re.search(r'\b' + re.escape(a) + r'\b', brand_context))
                if hits == 0: continue

                if not check_strict_constraints(p_name, brand_context): continue
                if any(check_negation_in_window(brand_context, a) for a in anchors): continue
                if not check_form_factor_compliance(p_name, brand_context): continue

                match_ratio = hits / len(anchors)
                score = 0.65
                if match_ratio >= 0.8:              score = 0.90
                elif match_ratio >= 0.5 and hits >= 2: score = 0.80
                elif hits == 1 and len(anchors) == 1:  score = 0.75

                p_form = identify_form(p_name)
                c_form = identify_form(brand_context)
                if p_form != "other" and c_form != "other":
                    if p_form == c_form:                                    score += 0.15
                    elif p_form == 'moisturizer' and c_form in ['cream', 'lotion']: score += 0.10
                    else:                                                   score -= 0.25

                if score >= FUZZY_THRESHOLD:
                    session_matches.append({
                        "Product":        p_name,
                        "Price":          prod['Price'],
                        "Brand":          brand,
                        "Concern":        concern,
                        "ConcernSource":  concern_source,
                        "Form":           p_form,
                        "Score":          score,
                    })

        # Deduplicate — best score per brand/form
        session_matches.sort(key=lambda x: x['Score'], reverse=True)
        final_picks = []
        seen_keys   = set()
        best_score  = session_matches[0]['Score'] if session_matches else 0

        for m in session_matches:
            if best_score >= 0.95 and m['Score'] < 0.75: continue
            key = f"{m['Brand']}_{m['Form']}"
            if key in seen_keys and m['Form'] != 'other' and m['Score'] < 0.90: continue
            final_picks.append(m)
            seen_keys.add(key)
            if len(final_picks) >= 6: break

        # ── POS SALE LINKAGE ─────────────────────────────────────────────────
        pos_data = find_pos_sale(row, social_sales)

        # Purchased product concerns — looked up once per session, reused for all picks
        purchased_concerns = ""
        if pos_data.get("pos_confirmed_sale"):
            purchased_concerns = get_purchased_product_concerns(pos_data, catalog_map)

        for pick in final_picks:
            # Did the POS sale include this recommendation's brand?
            brand_in_pos = False
            if pos_data.get("pos_confirmed_sale"):
                purchased_brands = str(pos_data.get("pos_brands_purchased", "")).lower()
                brand_in_pos = pick["Brand"].lower() in purchased_brands

            # Enrich concern using KB concern of purchased product
            final_concern, final_source = resolve_final_concern(
                pick['Concern'], pick['ConcernSource'], purchased_concerns
            )

            recommendations.append({
                "Date":                           row.get('activity_date'),
                "Session ID":                     row['session_id'],
                "Customer Name":                  row.get('contact_name'),
                "Staff Name":                     row.get('sales_owner', 'Unassigned'),
                # ── Concern columns ──
                "Concern Identified":             pick['Concern'],       # from AI/Regex on chat
                "Concern Source":                 pick['ConcernSource'], # AI / Regex / None
                "Concern (Final)":                final_concern,         # enriched by KB if POS matched
                "Concern (Final Source)":         final_source,          # full resolution path
                # ── Recommendation columns ──
                "Brand Filter":                   pick['Brand'],
                "Recommended Product":            pick['Product'],
                "Product Price":                  pick['Price'],
                # ── Outcome columns ──
                "Chat Revenue (M-Pesa)":          row.get('mpesa_amount', 0),
                "Is Converted (Chat)":            row.get('is_converted', 0),
                "POS Sale Confirmed":             pos_data.get("pos_confirmed_sale", False),
                "POS Brand Matches Rec":          brand_in_pos,
                "POS Products Purchased":         pos_data.get("pos_products_purchased", ""),
                "POS Purchased Product Concerns": purchased_concerns,
                "POS Revenue":                    pos_data.get("pos_revenue", 0),
                "POS Location":                   pos_data.get("pos_location", ""),
                "POS Transaction IDs":            pos_data.get("pos_transaction_ids", ""),
                # ── Quality ──
                "Match Score":                    round(pick['Score'], 2),
                "Chat Context":                   raw_text[:1500],
            })

    if recommendations:
        df_rec = pd.DataFrame(recommendations)
        df_rec.to_csv(OUTPUT_FILE, index=False)

        total      = len(df_rec)
        pos_conf   = df_rec["POS Sale Confirmed"].sum()
        brand_hit  = df_rec["POS Brand Matches Rec"].sum()

        print(f"\n✅ Success!")
        print(f"   📊 Recommendations extracted:    {total:,}")
        print(f"   🧠 Concerns from AI:             {ai_used:,}")
        print(f"   🔍 Concerns from Regex:          {reg_used:,}")
        print(f"   🛒 With confirmed POS sale:      {pos_conf:,}")
        print(f"   🎯 Brand match in POS sale:      {brand_hit:,}")
        print(f"   📄 Saved to: {OUTPUT_FILE}")
    else:
        print("⚠️ No recommendations found matching the criteria.")


if __name__ == "__main__":
    run_recommendation_extraction()