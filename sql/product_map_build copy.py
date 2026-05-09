"""
build_product_map.py  —  V2  (Barcode + Size-Aware Matching)
=============================================================
Pipeline stages per POS description:

  Stage 1  Pre-classify   Interbranch / Service entries → flagged, skip matching
  Stage 2  Barcode match  item_lookup_code from fact_sales_lineitems matched
                          against code_1 / code_2 / item_code in dim_knowledge_base
                          Most reliable — uses actual barcodes from POS scanners
  Stage 3  Size-aware     Brand detection + fuzzy match with unit normalisation
           fuzzy          16oz ↔ 473ml, 8oz ↔ 237ml, 1L ↔ 1000ml etc.
                          Size mismatch penalises score; size match boosts it

Run:
    python -m Portal_ML_V4.src.pipelines.pos_finance.build_product_map
    python -m Portal_ML_V4.src.pipelines.pos_finance.build_product_map --force
"""

from __future__ import annotations

import gc
import math
import os
import re
import sys
import time
from difflib import SequenceMatcher

import pandas as pd
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from tqdm import tqdm

try:
    from Portal_ML_V4.src.config.brands import BRAND_ALIASES
except ImportError:
    BRAND_ALIASES = {}

# ── Config ─────────────────────────────────────────────────────────────────────
load_dotenv()

MATCH_THRESHOLD       = 0.45   # minimum fuzzy score to accept
SIZE_PENALTY          = 0.25   # deducted when sizes are detected but incompatible
SIZE_BONUS            = 0.12   # added when sizes match
SIZE_TOLERANCE        = 0.07   # 7% tolerance on unit-normalised sizes
BATCH_SIZE            = 500

STOP_WORDS = {
    'FOR','WITH','OF','TO','IN','ON','AT','THE','AND','BY','A','AN',
    'TUBE','BOTTLE','PCS','SYR',
}

TERM_ALIASES = {
    "TABS":"TABLETS","TAB":"TABLET","CAPS":"CAPSULES","CAP":"CAPSULE",
    "SYR":"SYRUP","SOLN":"SOLUTION","SOL":"SOLUTION","CRM":"CREAM",
    "SQUAL":"SQUALANE","CLEANS":"CLEANSER","INJ":"INJECTION",
    "OINT":"OINTMENT","LOT":"LOTION","QTY":"QUANTITY",
    "X1":"1PC","X2":"2PCS","MOIST":"MOISTURIZING","MOISTUR":"MOISTURIZING",
    "HYDRAT":"HYDRATING","CLEANSR":"CLEANSER","EFFACL":"EFFACLAR",
    "ANTIHLS":"ANTHELIOS","CICAPLS":"CICAPLAST","NIACIN":"NIACINAMIDE",
    "RETINL":"RETINOL","SUSP":"SUSPENSION","V.DRY":"VERY DRY",
}

# ── Unit conversions → normalise everything to ml (liquids) or g (solids) ──────
# All POS/KB sizes are converted to a canonical float in ml or g before comparing.
# This lets "16OZ" match "473ML", "1L" match "1000ML", "0.5OZ" match "14G" etc.

ML_CONVERSIONS = {
    'L':    1000.0,
    'ML':   1.0,
    'FLOZ': 29.5735,
    'OZ':   29.5735,   # assume fl oz for cosmetics/liquids
}

G_CONVERSIONS = {
    'KG':   1000.0,
    'G':    1.0,
    'MG':   0.001,
    'OZ':   28.3495,   # weight oz — only used when context is clearly solid
}

# Regex to pull a size token: optional decimal + unit
_SIZE_RE = re.compile(
    r'(\d+(?:\.\d+)?)\s*'
    r'(FL\s*OZ|FLOZ|OZ|ML|CL|DL|L|KG|MG|G)\b',
    re.IGNORECASE,
)

# Count-based units (tablets, capsules, pieces) — treated separately
_COUNT_RE = re.compile(
    r"(\d+)\s*(?:S|'S|PCS?|TABS?|CAPSULES?|CAPS?|PIECES?|SACHETS?)\b",
    re.IGNORECASE,
)

# ── Interbranch / Service ──────────────────────────────────────────────────────
INTERBRANCH_PATTERNS = [
    r'^GOODS\s*[-\s]?[VZ]',
    r'^GOODS\s*(VAT|ZERO)',
    r'^Goods\s*(VAT|Zero)',
    r'^Items?\s*VAT',
    r'^\#NULL\#$',
]
SERVICE_PATTERNS = [
    r'^DELIVERY\s*\d*$',
    r'^Delivery\s*(Charge|Fee)',
    r'^DELIVERY\s*(FEES?|CHARGE)',
    r'^COURIER',
    r'^BLOOD\s*(PRESSURE|SUGAR)\s*TEST',
    r'^SKIN\s*ANALYSIS',
    r'^CONSULTATION',
]

# ── DB ─────────────────────────────────────────────────────────────────────────
def get_conn():
    return psycopg2.connect(
        host=os.getenv("DB_HOST",  "localhost"),
        port=int(os.getenv("DB_PORT", 5432)),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
    )


# ══════════════════════════════════════════════════════════════════════════════
# SIZE NORMALISATION
# ══════════════════════════════════════════════════════════════════════════════

def normalise_unit(raw_unit: str) -> str:
    """Normalise unit string to uppercase without whitespace."""
    return re.sub(r'\s+', '', raw_unit.upper())


def extract_sizes(text: str) -> list[float]:
    """
    Return a list of sizes extracted from text, all normalised to ml (for
    liquids) or g (for solids).  OZ is treated as fl oz (liquids) unless
    the description contains explicit weight keywords.

    e.g.  "16OZ"   → [473.2]
          "250ML"  → [250.0]
          "1L"     → [1000.0]
          "100G"   → [100.0]     (grams kept as-is in g domain)
    """
    text_upper = text.upper()
    is_solid = bool(re.search(r'\b(TABLET|CAPS|TABS|POWDER|CREAM|OINT|GEL|SACHET)\b', text_upper))

    sizes = []
    for m in _SIZE_RE.finditer(text_upper):
        value = float(m.group(1))
        unit  = normalise_unit(m.group(2))

        if unit in ('ML', 'L', 'CL', 'DL', 'FLOZ', 'FL OZ'):
            conv = {'ML':1.0,'L':1000.0,'CL':10.0,'DL':100.0,'FLOZ':29.5735,'FL OZ':29.5735}
            sizes.append(value * conv.get(unit, 1.0))
        elif unit == 'OZ':
            if is_solid:
                sizes.append(value * 28.3495)   # weight oz → g
            else:
                sizes.append(value * 29.5735)   # fl oz → ml
        elif unit in ('G', 'KG', 'MG'):
            conv = {'G':1.0,'KG':1000.0,'MG':0.001}
            sizes.append(value * conv[unit])

    return sizes


def extract_count(text: str) -> int | None:
    """Extract tablet / capsule / piece count if present."""
    m = _COUNT_RE.search(text.upper())
    return int(m.group(1)) if m else None


def sizes_compatible(pos_sizes: list[float], kb_sizes: list[float]) -> str:
    """
    Compare extracted size lists.
    Returns:
        'match'    — at least one pair within SIZE_TOLERANCE
        'mismatch' — sizes detected in both but none match
        'unknown'  — insufficient size data to decide
    """
    if not pos_sizes or not kb_sizes:
        return 'unknown'

    for a in pos_sizes:
        for b in kb_sizes:
            if b == 0:
                continue
            if abs(a - b) / b <= SIZE_TOLERANCE:
                return 'match'

    return 'mismatch'


def count_compatible(pos_count: int | None, kb_count: int | None) -> str:
    if pos_count is None or kb_count is None:
        return 'unknown'
    return 'match' if pos_count == kb_count else 'mismatch'


# ══════════════════════════════════════════════════════════════════════════════
# TEXT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def expand_aliases(text: str) -> str:
    if not text:
        return ""
    text = str(text).upper()
    all_aliases = {
        **TERM_ALIASES,
        **{k.upper(): v.upper() for k, v in BRAND_ALIASES.items()},
    }
    for alias, full in all_aliases.items():
        text = re.sub(r'\b' + re.escape(alias) + r'\b', full, text)
    return text


def clean_for_match(text: str) -> str:
    """Expand aliases, strip punctuation, remove stop words."""
    clean = expand_aliases(text)
    clean = re.sub(r'[^A-Z0-9\s]', ' ', clean)
    tokens = [w for w in clean.split() if w not in STOP_WORDS and len(w) > 1]
    return " ".join(tokens)


def fuzzy_score(a: str, b: str, brand: str | None = None) -> float:
    """
    Hybrid: sequence ratio + jaccard token overlap + coverage.
    Brand name stripped from POS text before comparison to avoid dilution.
    """
    if not a or not b:
        return 0.0

    if brand:
        b_clean = re.sub(r'[^A-Z0-9\s]', ' ', brand.upper()).strip()
        a = re.sub(r'\b' + re.escape(b_clean) + r'\b', '', a).strip()
        a = re.sub(r'\s+', ' ', a).strip()

    seq = SequenceMatcher(None, a, b).ratio()
    ta  = set(a.split())
    tb  = set(b.split())

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


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1: PRE-CLASSIFY
# ══════════════════════════════════════════════════════════════════════════════

def pre_classify(desc: str) -> dict | None:
    d = str(desc).strip()

    for p in INTERBRANCH_PATTERNS:
        if re.search(p, d, re.IGNORECASE):
            return _result(d, 'Interbranch', 'Interbranch Transfer', 'Stock Movement',
                           score=1.0, status='Interbranch')

    for p in SERVICE_PATTERNS:
        if re.search(p, d, re.IGNORECASE):
            return _result(d, 'Service', 'Service', 'Service',
                           score=1.0, status='Service')

    return None


def _result(name, brand, category, sub, score=0.0, status='Unmatched',
            concerns=None, audience=None, method='pre-classify') -> dict:
    return {
        'matched_name':       name,
        'brand':              brand,
        'canonical_category': category,
        'sub_category':       sub,
        'concerns':           concerns,
        'target_audience':    audience,
        'match_score':        score,
        'match_status':       status,
        'match_method':       method,
    }


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2: BARCODE MATCH
# ══════════════════════════════════════════════════════════════════════════════

def barcode_match(desc: str, barcode_to_kb: dict) -> dict | None:
    """
    Look up the description in the barcode→KB map.
    barcode_to_kb keys are normalised barcode strings.
    Returns a result dict on hit, None on miss.
    """
    kb_row = barcode_to_kb.get(desc)
    if kb_row is None:
        return None

    return _result(
        name     = kb_row['name'],
        brand    = kb_row['brand'],
        category = kb_row['canonical_category'],
        sub      = kb_row['sub_category'],
        score    = 1.0,
        status   = 'Matched',
        concerns = kb_row.get('concerns'),
        audience = kb_row.get('target_audience'),
        method   = 'barcode',
    )


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3: SIZE-AWARE FUZZY MATCH
# ══════════════════════════════════════════════════════════════════════════════

def detect_brands(desc: str, brand_list: list) -> list:
    safe = re.sub(r'[^A-Z0-9\s]', ' ', desc.upper())
    safe = re.sub(r'\s+', ' ', safe).strip()
    found = []

    for brand in brand_list:
        bc  = re.sub(r'[^A-Z0-9\s]', ' ', brand.upper())
        bc  = re.sub(r'\s+', ' ', bc).strip()
        bnt = re.sub(r'^THE\s+', '', bc)

        if re.search(r'\b' + re.escape(bc) + r'\b', safe):
            found.append(brand)
        elif bnt != bc and re.search(r'\b' + re.escape(bnt) + r'\b', safe):
            found.append(brand)

    expanded = re.sub(r'[^A-Z0-9\s]', ' ', expand_aliases(desc).upper())
    for alias, canonical in BRAND_ALIASES.items():
        if canonical in found:
            continue
        ac = re.sub(r'[^A-Z0-9\s]', ' ', alias.upper()).strip()
        if len(ac) < 3:
            continue
        if ' ' in ac:
            if ac in expanded:
                found.append(canonical)
        else:
            if re.search(r'\b' + re.escape(ac) + r'\b', expanded):
                found.append(canonical)

    return list(dict.fromkeys(found))   # dedupe, preserve order


def size_aware_match(pos_desc: str, detected_brands: list,
                     kb_by_brand: dict) -> dict:
    """
    Fuzzy-match pos_desc against KB candidates with size compatibility scoring.

    Scoring:
      base score  = fuzzy_score(cleaned_pos, cleaned_kb_name)
      size bonus  = +SIZE_BONUS  if sizes detected and compatible
      size penalty= -SIZE_PENALTY if sizes detected in both but incompatible
      count check = same logic for tablet/capsule counts
    """
    pos_clean  = clean_for_match(pos_desc)
    pos_sizes  = extract_sizes(pos_desc)
    pos_count  = extract_count(pos_desc)

    unmatched_result = _result(
        name=None, brand=None, category=None, sub=None,
        status='Unmatched', method='fuzzy',
    )

    if not detected_brands:
        return unmatched_result

    best_score = 0.0
    best_row   = None

    for brand in detected_brands:
        candidates = kb_by_brand.get(brand.upper(), [])
        for row in candidates:
            base = fuzzy_score(pos_clean, row['_clean'], brand=brand)

            # Size compatibility adjustment
            kb_sizes = row.get('_sizes', [])
            kb_count = row.get('_count')

            size_compat  = sizes_compatible(pos_sizes, kb_sizes)
            count_compat = count_compatible(pos_count, kb_count)

            score = base
            if size_compat == 'match':
                score += SIZE_BONUS
            elif size_compat == 'mismatch':
                score -= SIZE_PENALTY

            if count_compat == 'match':
                score += SIZE_BONUS * 0.5
            elif count_compat == 'mismatch':
                score -= SIZE_PENALTY * 0.5

            score = max(0.0, min(1.0, score))

            if score > best_score:
                best_score = score
                best_row   = row

    if best_row and best_score >= MATCH_THRESHOLD:
        return _result(
            name     = best_row['name'],
            brand    = best_row['brand'],
            category = best_row['canonical_category'],
            sub      = best_row['sub_category'],
            score    = round(best_score, 4),
            status   = 'Matched',
            concerns = best_row.get('concerns'),
            audience = best_row.get('target_audience'),
            method   = 'size_fuzzy',
        )

    return unmatched_result


# ══════════════════════════════════════════════════════════════════════════════
# DDL
# ══════════════════════════════════════════════════════════════════════════════

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
    match_method        TEXT,
    mapped_at           TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_product_map_brand    ON dim_product_map(brand);
CREATE INDEX IF NOT EXISTS idx_product_map_category ON dim_product_map(canonical_category);
CREATE INDEX IF NOT EXISTS idx_product_map_status   ON dim_product_map(match_status);
"""

# Add match_method column if upgrading from V1
ALTER_TABLE_SQL = """
ALTER TABLE dim_product_map
    ADD COLUMN IF NOT EXISTS match_method TEXT;
"""

ADD_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_product_map_method ON dim_product_map(match_method);
"""

UPSERT_SQL = """
INSERT INTO dim_product_map (
    pos_description, matched_name, brand, canonical_category,
    sub_category, concerns, target_audience,
    match_score, match_status, match_method
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
    match_method       = EXCLUDED.match_method,
    mapped_at          = NOW();
"""


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run_build_product_map(force_remap: bool = False):
    """
    force_remap=True  → reprocesses ALL descriptions (use after KB update)
    force_remap=False → incremental, only new/unmatched descriptions (default)
    """
    print("=" * 65)
    print("🗺️  BUILD PRODUCT MAP  V2  (Barcode + Size-Aware Matching)")
    print(f"    Mode: {'FULL REMAP' if force_remap else 'Incremental'}")
    print("=" * 65)
    t0 = time.time()

    conn = get_conn()
    cur  = conn.cursor()

    cur.execute(CREATE_TABLE_SQL)
    cur.execute(ALTER_TABLE_SQL)
    conn.commit()

    # ── 1. Load KB ─────────────────────────────────────────────────────────────
    print("\n📖 Loading Knowledge Base from dim_knowledge_base...")
    cur.execute("""
        SELECT name, brand, canonical_category, sub_category,
               concerns, target_audience, price,
               code_1, code_2, item_code
        FROM dim_knowledge_base
        WHERE name IS NOT NULL AND brand IS NOT NULL
          AND TRIM(name) != '' AND TRIM(brand) != '';
    """)
    kb_rows = cur.fetchall()
    kb_cols = ['name','brand','canonical_category','sub_category',
               'concerns','target_audience','price',
               'code_1','code_2','item_code']
    df_kb = pd.DataFrame(kb_rows, columns=kb_cols)

    # Pre-compute clean text and sizes for every KB row
    df_kb['_clean'] = df_kb['name'].apply(clean_for_match)
    df_kb['_sizes'] = df_kb['name'].apply(extract_sizes)
    df_kb['_count'] = df_kb['name'].apply(extract_count)

    brand_list = sorted(
        df_kb['brand'].astype(str).str.strip().str.title().unique().tolist(),
        key=len, reverse=True,   # longest first: "La Roche Posay" before "La"
    )

    # Brand lookup dict (key = UPPERCASE brand)
    kb_by_brand: dict = {}
    for row in df_kb.to_dict('records'):
        key = str(row['brand']).upper().strip()
        kb_by_brand.setdefault(key, []).append(row)

    print(f"   ✅ {len(df_kb):,} products · {len(brand_list)} brands")

    # ── 2. Build barcode → KB map ───────────────────────────────────────────────
    # Maps every normalised barcode to its KB row.
    # Barcodes are stored in code_1 / code_2 / item_code in dim_knowledge_base.
    # The POS scanner stores item_lookup_code in fact_sales_lineitems.
    # We join via fact_sales_lineitems to get the barcode for each description.
    print("\n🔍 Building barcode → KB lookup...")

    def norm_barcode(v) -> str | None:
        if v is None or str(v).strip() in ('', 'nan', 'None'):
            return None
        s = re.sub(r'[^0-9]', '', str(v))   # digits only
        return s if s else None

    barcode_to_kb: dict = {}   # barcode_str → kb row dict
    for row in df_kb.to_dict('records'):
        for col in ('code_1', 'code_2', 'item_code'):
            bc = norm_barcode(row.get(col))
            if bc and bc not in barcode_to_kb:
                barcode_to_kb[bc] = row

    print(f"   ✅ {len(barcode_to_kb):,} unique barcodes indexed from KB")

    # Pull (description, item_lookup_code) pairs from fact_sales_lineitems
    cur.execute("""
        SELECT DISTINCT
            l.description,
            l.item
        FROM fact_sales_lineitems l
        JOIN fact_sales_transactions f
             ON l.transaction_id = f.transaction_id
            AND l.location       = f.location
        WHERE l.item IS NOT NULL
          AND TRIM(l.item) != ''
          AND f.products_in_txn IS NOT NULL;
    """)
    barcode_rows = cur.fetchall()

    # desc → list of barcodes (a POS description might have been sold
    # under slightly different barcodes across branches)
    desc_to_barcodes: dict = {}
    for desc, bc in barcode_rows:
        nbc = norm_barcode(bc)
        if nbc:
            desc_to_barcodes.setdefault(desc, set()).add(nbc)

    print(f"   ✅ {len(desc_to_barcodes):,} descriptions have POS barcodes")

    # ── 3. Fetch descriptions to process ───────────────────────────────────────
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
                  WHERE match_status != 'Unmatched'
              )
            ORDER BY products_in_txn;
        """)

    descriptions = [r[0] for r in cur.fetchall()]
    total = len(descriptions)
    print(f"   ✅ {total:,} descriptions to process")

    if not descriptions:
        print("\n   ℹ️  Nothing to process.")
        cur.close()
        conn.close()
        return

    # ── 4. Classify ─────────────────────────────────────────────────────────────
    print("\n🔍 Running classification pipeline...")
    counts = {k: 0 for k in ('Interbranch','Service','Matched_barcode',
                               'Matched_fuzzy','Unmatched')}
    batch: list = []

    for desc in tqdm(descriptions, desc="Classifying", unit="desc"):

        # Stage 1 — interbranch / service
        result = pre_classify(desc)
        if result:
            counts['Interbranch' if result['match_status']=='Interbranch' else 'Service'] += 1

        # Stage 2 — barcode match
        if result is None:
            barcodes = desc_to_barcodes.get(desc, set())
            for bc in barcodes:
                kb_row = barcode_to_kb.get(bc)
                if kb_row:
                    result = _result(
                        name     = kb_row['name'],
                        brand    = kb_row['brand'],
                        category = kb_row['canonical_category'],
                        sub      = kb_row['sub_category'],
                        score    = 1.0,
                        status   = 'Matched',
                        concerns = kb_row.get('concerns'),
                        audience = kb_row.get('target_audience'),
                        method   = 'barcode',
                    )
                    counts['Matched_barcode'] += 1
                    break

        # Stage 3 — size-aware fuzzy
        if result is None:
            brands = detect_brands(desc, brand_list)
            result = size_aware_match(desc, brands, kb_by_brand)
            if result['match_status'] == 'Matched':
                counts['Matched_fuzzy'] += 1
            else:
                counts['Unmatched'] += 1

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
            result['match_method'],
        ))

        if len(batch) >= BATCH_SIZE:
            psycopg2.extras.execute_values(cur, UPSERT_SQL, batch, page_size=BATCH_SIZE)
            conn.commit()
            batch = []
            gc.collect()

    if batch:
        psycopg2.extras.execute_values(cur, UPSERT_SQL, batch, page_size=BATCH_SIZE)
        conn.commit()

    # ── 5. Summary ──────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    matched = counts['Matched_barcode'] + counts['Matched_fuzzy']

    print(f"\n{'='*65}")
    print(f"✅  PRODUCT MAP COMPLETE  in {elapsed:.1f}s")
    print(f"    Barcode match:    {counts['Matched_barcode']:>6,}  "
          f"({counts['Matched_barcode']/total*100:.1f}%)")
    print(f"    Size-fuzzy match: {counts['Matched_fuzzy']:>6,}  "
          f"({counts['Matched_fuzzy']/total*100:.1f}%)")
    print(f"    Interbranch:      {counts['Interbranch']:>6,}  "
          f"({counts['Interbranch']/total*100:.1f}%)")
    print(f"    Service:          {counts['Service']:>6,}  "
          f"({counts['Service']/total*100:.1f}%)")
    print(f"    Unmatched:        {counts['Unmatched']:>6,}  "
          f"({counts['Unmatched']/total*100:.1f}%)")
    print(f"    ──────────────────────────────────────────────")
    print(f"    Total matched:    {matched:>6,}  ({matched/total*100:.1f}%)")
    print(f"{'='*65}")

    # ── 6. Category breakdown ───────────────────────────────────────────────────
    print("\n📊 dim_product_map — breakdown by method & category:")
    cur.execute("""
        SELECT
            COALESCE(match_method, '—')          AS method,
            COALESCE(canonical_category, 'NULL') AS category,
            COUNT(*)                              AS descriptions
        FROM dim_product_map
        GROUP BY match_method, canonical_category
        ORDER BY
            CASE match_method
                WHEN 'barcode'     THEN 1
                WHEN 'size_fuzzy'  THEN 2
                WHEN 'pre-classify' THEN 3
                ELSE 4
            END,
            descriptions DESC
        LIMIT 35;
    """)
    print(f"   {'Method':<14} {'Category':<28} {'Descriptions':>12}")
    print(f"   {'-'*56}")
    for row in cur.fetchall():
        print(f"   {str(row[0]):<14} {str(row[1]):<28} {row[2]:>12,}")

    cur.close()
    conn.close()


if __name__ == "__main__":
    force = "--force" in sys.argv
    run_build_product_map(force_remap=force)