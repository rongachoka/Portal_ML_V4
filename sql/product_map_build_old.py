"""
build_product_map.py
====================
Pulls every unique POS description from fact_sales_transactions,
runs a 3-stage classification pipeline, and writes results to dim_product_map.

Stage 1 — Pre-classify:  Interbranch/Service entries flagged immediately
Stage 2 — Auto-classify: Generic medicines given category without KB match
Stage 3 — Fuzzy match:   Branded items matched against dim_knowledge_base

Run standalone once, or after a KB update:
    python -m Portal_ML_V4.src.pipelines.pos_finance.build_product_map

Incremental — only processes descriptions not yet in dim_product_map.
"""

import os
import re
import gc
import time
import psycopg2
import psycopg2.extras
import pandas as pd
from difflib import SequenceMatcher
from dotenv import load_dotenv
from tqdm import tqdm

try:
    from Portal_ML_V4.src.config.brands import BRAND_ALIASES
except ImportError:
    BRAND_ALIASES = {}

# ── Config ────────────────────────────────────────────────────────────────────
load_dotenv()

MATCH_THRESHOLD = 0.45
BATCH_SIZE      = 500

STOP_WORDS = {
    'FOR', 'WITH', 'OF', 'TO', 'IN', 'ON', 'AT', 'ML', 'GM', 'PCS',
    'TUBE', 'BOTTLE', 'SYR', 'THE', 'AND', 'BY', 'A', 'AN',
}

# From enrich_attribution_products.py — kept in sync
TERM_ALIASES = {
    "TABS":    "TABLETS",
    "TAB":     "TABLET",
    "CAPS":    "CAPSULES",
    "CAP":     "CAPSULE",
    "SYR":     "SYRUP",
    "SOLN":    "SOLUTION",
    "SOL":     "SOLUTION",
    "CRM":     "CREAM",
    "SQUAL":   "SQUALANE",
    "CLEANS":  "CLEANSER",
    "INJ":     "INJECTION",
    "OINT":    "OINTMENT",
    "LOT":     "LOTION",
    "QTY":     "QUANTITY",
    "X1":      "1PC",
    "X2":      "2PCS",
    "MOIST":   "MOISTURIZING",
    "MOISTUR": "MOISTURIZING",
    "HYDRAT":  "HYDRATING",
    "CLEANSR": "CLEANSER",
    "EFFACL":  "EFFACLAR",
    "ANTIHLS": "ANTHELIOS",
    "CICAPLS": "CICAPLAST",
    "NIACIN":  "NIACINAMIDE",
    "RETINL":  "RETINOL",
    "SUSP":    "SUSPENSION",
    "V.DRY":   "VERY DRY",
}

SIZE_TOKENS = {
    '8OZ', '16OZ', '250ML', '500ML', '100ML', '200ML', '30ML',
    '50ML', '150ML', '400ML', '1L', '2L', '300ML', '10OZ',
}

# ── Stage 1: Interbranch / Service patterns ───────────────────────────────────
# These are NOT retail sales — exclude from all revenue charts in Power BI
INTERBRANCH_PATTERNS = [
    r'^GOODS\s*[-\s]?V',          # GOODS -V, GOODS-V, GOODS V
    r'^GOODS\s*[-\s]?Z',          # GOODS -Z, GOODS-Z, GOODS ZERO
    r'^GOODS\s*(VAT|ZERO)',       # GOODS VAT 19188, GOODS ZERO 21210
    r'^Goods\s*(VAT|Zero|Zero)',  # lowercase variants
    r'^Items?\s*VAT',             # Items VAT 28228
    r'^\#NULL\#$',                # #NULL#
]

SERVICE_PATTERNS = [
    r'^DELIVERY\s*\d*$',          # DELIVERY 250, DELIVERY 300, DELIVERY
    r'^Delivery\s*(Charge|Fee)',  # Delivery Charge, Delivery Fees
    r'^DELIVERY\s*(FEES?|CHARGE)',
    r'^COURIER',
    r'^BLOOD\s*(PRESSURE|SUGAR)\s*TEST',
    r'^SKIN\s*ANALYSIS',
    r'^CONSULTATION',
]


# ── DB ────────────────────────────────────────────────────────────────────────
def get_conn():
    return psycopg2.connect(
        host=os.getenv("DB_HOST",  "localhost"),
        port=os.getenv("DB_PORT",  5432),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
    )


# ── Text helpers ──────────────────────────────────────────────────────────────
def expand_aliases(text: str) -> str:
    if not text:
        return ""
    text = str(text).upper()
    all_aliases = {
        **TERM_ALIASES,
        **{k.upper(): v.upper() for k, v in BRAND_ALIASES.items()}
    }
    for alias, full in all_aliases.items():
        text = re.sub(r'\b' + re.escape(alias) + r'\b', full, text)
    return text


def clean_for_match(text: str) -> str:
    clean = expand_aliases(text)
    clean = re.sub(r'[^A-Z0-9\s]', ' ', clean)
    tokens = [w for w in clean.split() if w not in STOP_WORDS and len(w) > 1]
    return " ".join(tokens)


def fuzzy_score(a: str, b: str, brand: str = None) -> float:
    if not a or not b:
        return 0.0
    if brand:
        b_clean = re.sub(r'[^A-Z0-9\s]', ' ', brand.upper()).strip()
        a = re.sub(r'\b' + re.escape(b_clean) + r'\b', '', a).strip()
        a = re.sub(r'\s+', ' ', a).strip()

    seq = SequenceMatcher(None, a, b).ratio()
    ta  = set(a.split()) - SIZE_TOKENS
    tb  = set(b.split()) - SIZE_TOKENS

    if not ta or not tb:
        return seq

    inter    = ta & tb
    union    = ta | tb
    jaccard  = len(inter) / len(union)
    shorter  = ta if len(ta) <= len(tb) else tb
    longer   = tb if len(ta) <= len(tb) else ta
    coverage = len(shorter & longer) / len(shorter) if shorter else 0
    token    = jaccard * 0.4 + coverage * 0.6

    return seq * 0.35 + token * 0.65


# ── Stage 1: Interbranch / Service check ─────────────────────────────────────
def pre_classify(desc: str) -> dict | None:
    """
    Returns a result dict if this description should be pre-classified,
    or None if it should proceed to fuzzy matching.
    """
    d = str(desc).strip()

    for pattern in INTERBRANCH_PATTERNS:
        if re.search(pattern, d, re.IGNORECASE):
            return {
                'matched_name':       d,
                'brand':              'Interbranch',
                'canonical_category': 'Interbranch Transfer',
                'sub_category':       'Stock Movement',
                'concerns':           None,
                'target_audience':    None,
                'match_score':        1.0,
                'match_status':       'Interbranch',
            }

    for pattern in SERVICE_PATTERNS:
        if re.search(pattern, d, re.IGNORECASE):
            return {
                'matched_name':       d,
                'brand':              'Service',
                'canonical_category': 'Service',
                'sub_category':       'Service',
                'concerns':           None,
                'target_audience':    None,
                'match_score':        1.0,
                'match_status':       'Service',
            }

    return None



# ── Stage 3: Brand detection + fuzzy match ───────────────────────────────────
def detect_brands(desc: str, brand_list: list) -> list:
    """Return KB brands found in this POS description."""
    safe_desc = re.sub(r'[^A-Z0-9\s]', ' ', desc.upper())
    safe_desc = re.sub(r'\s+', ' ', safe_desc).strip()
    found = []

    for brand in brand_list:
        b_clean  = re.sub(r'[^A-Z0-9\s]', ' ', brand.upper())
        b_clean  = re.sub(r'\s+', ' ', b_clean).strip()
        b_no_the = re.sub(r'^THE\s+', '', b_clean)

        if re.search(r'\b' + re.escape(b_clean) + r'\b', safe_desc):
            found.append(brand)
        elif b_no_the != b_clean and \
                re.search(r'\b' + re.escape(b_no_the) + r'\b', safe_desc):
            found.append(brand)

    # Check multi-word aliases (e.g. "LA ROCHE" → "La Roche Posay")
    # We check against the expanded/cleaned description to catch abbreviations
    expanded_desc = re.sub(r'[^A-Z0-9\s]', ' ', expand_aliases(desc).upper())
    for alias, canonical in BRAND_ALIASES.items():
        if canonical in found:
            continue
        alias_clean = re.sub(r'[^A-Z0-9\s]', ' ', alias.upper()).strip()
        if len(alias_clean) < 3:
            continue
        # Use plain substring for multi-word aliases like "LA ROCHE"
        if ' ' in alias_clean:
            if alias_clean in expanded_desc:
                found.append(canonical)
        else:
            if re.search(r'\b' + re.escape(alias_clean) + r'\b', expanded_desc):
                found.append(canonical)

    return list(dict.fromkeys(found))  # dedupe preserving order


def match_description(pos_desc: str, detected_brands: list,
                       kb_by_brand: dict) -> dict:
    pos_clean = clean_for_match(pos_desc)
    result = {
        'matched_name':       None,
        'brand':              None,
        'canonical_category': None,
        'sub_category':       None,
        'concerns':           None,
        'target_audience':    None,
        'match_score':        0.0,
        'match_status':       'Unmatched',
    }

    if not detected_brands:
        return result

    best_score = 0.0
    best_row   = None

    for brand in detected_brands:
        for row in kb_by_brand.get(brand.upper(), []):
            score = fuzzy_score(pos_clean, row['_clean'], brand=brand)
            if score > best_score:
                best_score = score
                best_row   = row

    if best_row and best_score >= MATCH_THRESHOLD:
        result.update({
            'matched_name':       best_row['name'],
            'brand':              best_row['brand'],
            'canonical_category': best_row['canonical_category'],
            'sub_category':       best_row['sub_category'],
            'concerns':           best_row['concerns'],
            'target_audience':    best_row['target_audience'],
            'match_score':        round(best_score, 4),
            'match_status':       'Matched',
        })

    return result


# ── DDL ───────────────────────────────────────────────────────────────────────
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS dim_product_map (
    pos_description     TEXT PRIMARY KEY,
    matched_name        TEXT,
    brand               TEXT,
    canonical_category  TEXT,
    sub_category        TEXT,
    concerns            TEXT,
    target_audience     TEXT,
    match_score         NUMERIC(5,4),
    match_status        TEXT,
    mapped_at           TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_product_map_brand
    ON dim_product_map(brand);
CREATE INDEX IF NOT EXISTS idx_product_map_category
    ON dim_product_map(canonical_category);
CREATE INDEX IF NOT EXISTS idx_product_map_status
    ON dim_product_map(match_status);
"""

UPSERT_SQL = """
INSERT INTO dim_product_map (
    pos_description, matched_name, brand, canonical_category,
    sub_category, concerns, target_audience, match_score, match_status
) VALUES %s
ON CONFLICT (pos_description) DO UPDATE SET
    matched_name       = EXCLUDED.matched_name,
    brand              = EXCLUDED.brand,
    canonical_category = EXCLUDED.canonical_category,
    sub_category       = EXCLUDED.sub_category,
    concerns           = EXCLUDED.concerns,
    target_audience    = EXCLUDED.target_audience,
    match_score        = EXCLUDED.match_score,
    match_status       = EXCLUDED.match_status,
    mapped_at          = NOW();
"""


# ── Main ──────────────────────────────────────────────────────────────────────
def run_build_product_map(force_remap: bool = False):
    """
    force_remap=True  → reprocesses ALL descriptions (use after KB update)
    force_remap=False → incremental, only new descriptions (default)
    """
    print("=" * 65)
    print("🗺️  BUILD PRODUCT MAP — POS Descriptions → Knowledge Base")
    print(f"    Mode: {'FULL REMAP' if force_remap else 'Incremental (new only)'}")
    print("=" * 65)
    t0 = time.time()

    conn = get_conn()
    cur  = conn.cursor()

    cur.execute(CREATE_TABLE_SQL)
    conn.commit()

    # ── 1. Load KB from PostgreSQL ────────────────────────────────────────────
    print("\n📖 Loading Knowledge Base from dim_knowledge_base...")
    cur.execute("""
        SELECT name, brand, canonical_category, sub_category,
               concerns, target_audience, price
        FROM dim_knowledge_base
        WHERE name IS NOT NULL AND brand IS NOT NULL
          AND TRIM(name) != '' AND TRIM(brand) != '';
    """)
    kb_rows = cur.fetchall()
    kb_cols = ['name', 'brand', 'canonical_category', 'sub_category',
               'concerns', 'target_audience', 'price']
    df_kb = pd.DataFrame(kb_rows, columns=kb_cols)
    df_kb['_clean'] = df_kb['name'].apply(clean_for_match)

    brand_list = sorted(
        df_kb['brand'].astype(str).str.strip().str.title().unique().tolist(),
        key=len, reverse=True  # longest first — "La Roche Posay" before "La"
    )
    kb_by_brand: dict = {}
    for row in df_kb.to_dict('records'):
        key = str(row['brand']).upper().strip()
        kb_by_brand.setdefault(key, []).append(row)

    print(f"   ✅ {len(df_kb):,} products · {len(brand_list)} brands loaded")

    # ── 2. Fetch descriptions to process ─────────────────────────────────────
    print("\n🔌 Fetching POS descriptions...")
    if force_remap:
        cur.execute("""
            SELECT DISTINCT products_in_txn
            FROM fact_sales_transactions
            WHERE products_in_txn IS NOT NULL
              AND TRIM(products_in_txn) != ''
            ORDER BY products_in_txn;
        """)
    else:
        cur.execute("""
            SELECT DISTINCT products_in_txn
            FROM fact_sales_transactions
            WHERE products_in_txn IS NOT NULL
              AND TRIM(products_in_txn) != ''
              AND products_in_txn NOT IN (
                  SELECT pos_description FROM dim_product_map
              )
            ORDER BY products_in_txn;
        """)

    descriptions = [r[0] for r in cur.fetchall()]
    total = len(descriptions)
    print(f"   ✅ {total:,} descriptions to process")

    if not descriptions:
        print("\n   ℹ️  Nothing new to map. dim_product_map is up to date.")
        cur.close()
        conn.close()
        return

    # ── 3. Classify ───────────────────────────────────────────────────────────
    print("\n🔍 Running 3-stage classification...")
    counts = {'Interbranch': 0, 'Service': 0,
              'Matched': 0, 'Unmatched': 0}
    batch = []

    for desc in tqdm(descriptions, desc="Classifying", unit="desc"):

        # Stage 1 — Interbranch / Service
        result = pre_classify(desc)

        # Stage 3 — KB fuzzy match (only if stages 1 & 2 didn't fire)
        if result is None:
            brands = detect_brands(desc, brand_list)
            result = match_description(desc, brands, kb_by_brand)

        counts[result['match_status']] = counts.get(result['match_status'], 0) + 1

        batch.append((
            desc,
            result['matched_name'],
            result['brand'],
            result['canonical_category'],
            result['sub_category'],
            result['concerns'],
            result['target_audience'],
            result['match_score'],
            result['match_status'],
        ))

        if len(batch) >= BATCH_SIZE:
            psycopg2.extras.execute_values(cur, UPSERT_SQL, batch, page_size=BATCH_SIZE)
            conn.commit()
            batch = []
            gc.collect()

    if batch:
        psycopg2.extras.execute_values(cur, UPSERT_SQL, batch, page_size=BATCH_SIZE)
        conn.commit()

    # ── 4. Summary ─────────────────────────────────────────────────────────────
    elapsed  = time.time() - t0
    branded  = counts.get('Matched', 0)
    inter    = counts.get('Interbranch', 0)
    service  = counts.get('Service', 0)
    unmatched = counts.get('Unmatched', 0)
    classified = branded + inter + service

    print(f"\n{'='*65}")
    print(f"✅  PRODUCT MAP COMPLETE in {elapsed:.1f}s")
    print(f"    KB Match (branded):   {branded:>6,}  ({branded/total*100:.1f}%)")
    print(f"    Interbranch:          {inter:>6,}  ({inter/total*100:.1f}%)")
    print(f"    Service:              {service:>6,}  ({service/total*100:.1f}%)")
    print(f"    Unmatched:            {unmatched:>6,}  ({unmatched/total*100:.1f}%)")
    print(f"    ─────────────────────────────────────────────")
    print(f"    Classified total:     {classified:>6,}  ({classified/total*100:.1f}%)")
    print(f"{'='*65}")

    # ── 5. Category breakdown ──────────────────────────────────────────────────
    print("\n📊 dim_product_map — category breakdown (all time):")
    cur.execute("""
        SELECT
            COALESCE(canonical_category, 'Unmatched') AS category,
            match_status,
            COUNT(*)                                  AS descriptions
        FROM dim_product_map
        GROUP BY canonical_category, match_status
        ORDER BY
            CASE match_status
                WHEN 'Matched'          THEN 1
                WHEN 'Auto-Classified'  THEN 2
                WHEN 'Interbranch'      THEN 3
                WHEN 'Service'          THEN 4
                ELSE 5
            END,
            descriptions DESC
        LIMIT 30;
    """)
    print(f"   {'Category':<30} {'Status':<16} {'Descriptions':>12}")
    print(f"   {'-'*60}")
    for row in cur.fetchall():
        print(f"   {str(row[0]):<30} {str(row[1]):<16} {row[2]:>12,}")

    cur.close()
    conn.close()


if __name__ == "__main__":
    import sys
    force = "--force" in sys.argv
    run_build_product_map(force_remap=force)