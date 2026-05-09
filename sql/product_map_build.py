"""
build_product_map.py  —  V3  (Catalogue-First Matching)
=========================================================
Pipeline stages per POS description:

  Stage 1  Pre-classify     Interbranch / Service entries flagged, skip matching
  Stage 2  Barcode match    l.item from fact_sales_lineitems matched against
                            dim_product_catalogue.item_lookup_code
                            Most reliable — uses actual POS scanner barcodes
  Stage 3  Catalogue fuzzy  Size-aware fuzzy match against dim_product_catalogue
                            (22,757 products — full product range)
  Stage 4  KB enrichment    For matched rows, enrich with brand/category/concerns
                            from dim_knowledge_base via barcode or name match
                            (website KB — brand/KB data only, not identity)

Run:
    python -m Portal_ML_V4.sql.product_map_build
    python -m Portal_ML_V4.sql.product_map_build --force
"""

from __future__ import annotations

import gc
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
    from Portal_ML_V4.src.config.pos_aliases import TERM_ALIASES, BRAND_ALIASES
except ImportError:
    try:
        from Portal_ML_V4.src.config.brands import BRAND_ALIASES
        TERM_ALIASES = {}
    except ImportError:
        TERM_ALIASES  = {}
        BRAND_ALIASES = {}

# ── Config ─────────────────────────────────────────────────────────────────────
load_dotenv()

MATCH_THRESHOLD = 0.45
SIZE_PENALTY    = 0.25
SIZE_BONUS      = 0.12
SIZE_TOLERANCE  = 0.07
BATCH_SIZE      = 500

STOP_WORDS = {
    'FOR', 'WITH', 'OF', 'TO', 'IN', 'ON', 'AT', 'THE', 'AND', 'BY', 'A', 'AN',
    'TUBE', 'BOTTLE', 'PCS', 'SYR',
}

# ── Unit conversions ───────────────────────────────────────────────────────────
_SIZE_RE = re.compile(
    r'(\d+(?:\.\d+)?)\s*(FL\s*OZ|FLOZ|OZ|ML|CL|DL|L|KG|MG|G)\b',
    re.IGNORECASE,
)
_COUNT_RE = re.compile(
    r"(\d+)\s*(?:S|'S|PCS?|TABS?|CAPSULES?|CAPS?|PIECES?|SACHETS?)\b",
    re.IGNORECASE,
)

# ── Interbranch / Service patterns ────────────────────────────────────────────
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
        host     = os.getenv("DB_HOST", "localhost"),
        port     = int(os.getenv("DB_PORT", 5432)),
        dbname   = os.getenv("DB_NAME"),
        user     = os.getenv("DB_USER"),
        password = os.getenv("DB_PASSWORD"),
    )


# ══════════════════════════════════════════════════════════════════════════════
# SIZE NORMALISATION
# ══════════════════════════════════════════════════════════════════════════════

def extract_sizes(text: str) -> list[float]:
    text_upper = text.upper()
    is_solid = bool(re.search(
        r'\b(TABLET|CAPS|TABS|POWDER|CREAM|OINT|GEL|SACHET)\b', text_upper
    ))
    sizes = []
    for m in _SIZE_RE.finditer(text_upper):
        value = float(m.group(1))
        unit  = re.sub(r'\s+', '', m.group(2).upper())
        if unit in ('ML', 'L', 'CL', 'DL', 'FLOZ'):
            conv = {'ML': 1.0, 'L': 1000.0, 'CL': 10.0, 'DL': 100.0, 'FLOZ': 29.5735}
            sizes.append(value * conv.get(unit, 1.0))
        elif unit == 'OZ':
            sizes.append(value * (28.3495 if is_solid else 29.5735))
        elif unit in ('G', 'KG', 'MG'):
            conv = {'G': 1.0, 'KG': 1000.0, 'MG': 0.001}
            sizes.append(value * conv[unit])
    return sizes


def extract_count(text: str) -> int | None:
    m = _COUNT_RE.search(text.upper())
    return int(m.group(1)) if m else None


def sizes_compatible(pos: list, kb: list) -> str:
    if not pos or not kb:
        return 'unknown'
    for a in pos:
        for b in kb:
            if b == 0:
                continue
            if abs(a - b) / b <= SIZE_TOLERANCE:
                return 'match'
    return 'mismatch'


def count_compatible(a, b) -> str:
    if a is None or b is None:
        return 'unknown'
    return 'match' if a == b else 'mismatch'


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
        text = re.sub(r'\b' + re.escape(alias.upper()) + r'\b', full.upper(), text)
    return text


def clean_for_match(text: str) -> str:
    clean = expand_aliases(text)
    clean = re.sub(r'[^A-Z0-9\s]', ' ', clean)
    tokens = [w for w in clean.split() if w not in STOP_WORDS and len(w) > 1]
    return " ".join(tokens)


def fuzzy_score(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    seq      = SequenceMatcher(None, a, b).ratio()
    ta       = set(a.split())
    tb       = set(b.split())
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
# RESULT HELPER
# ══════════════════════════════════════════════════════════════════════════════

def _result(name, brand, category, sub, department=None, supplier=None,
            score=0.0, status='Unmatched', concerns=None,
            audience=None, method='fuzzy') -> dict:
    return {
        'matched_name':       name,
        'brand':              brand,
        'canonical_category': category,
        'sub_category':       sub,
        'department':         department,
        'supplier':           supplier,
        'concerns':           concerns,
        'target_audience':    audience,
        'match_score':        score,
        'match_status':       status,
        'match_method':       method,
    }


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — PRE-CLASSIFY
# ══════════════════════════════════════════════════════════════════════════════

def pre_classify(desc: str) -> dict | None:
    d = str(desc).strip()
    for p in INTERBRANCH_PATTERNS:
        if re.search(p, d, re.IGNORECASE):
            return _result(d, 'Interbranch', 'Interbranch Transfer',
                           'Stock Movement', score=1.0,
                           status='Interbranch', method='pre-classify')
    for p in SERVICE_PATTERNS:
        if re.search(p, d, re.IGNORECASE):
            return _result(d, 'Service', 'Service', 'Service',
                           score=1.0, status='Service', method='pre-classify')
    return None


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — CATALOGUE FUZZY MATCH
# ══════════════════════════════════════════════════════════════════════════════

def catalogue_fuzzy_match(pos_desc: str, cat_rows: list) -> dict:
    """
    Size-aware fuzzy match against dim_product_catalogue rows.
    cat_rows is the full catalogue list (pre-loaded).
    """
    pos_clean  = clean_for_match(pos_desc)
    pos_sizes  = extract_sizes(pos_desc)
    pos_count  = extract_count(pos_desc)

    best_score = 0.0
    best_row   = None

    for row in cat_rows:
        base = fuzzy_score(pos_clean, row['_clean'])

        size_compat  = sizes_compatible(pos_sizes, row.get('_sizes', []))
        count_compat = count_compatible(pos_count, row.get('_count'))

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
            name       = best_row['canonical_name'],
            brand      = None,               # filled by KB enrichment
            category   = None,               # filled by KB enrichment
            sub        = None,
            department = best_row.get('department'),
            supplier   = best_row.get('supplier'),
            score      = round(best_score, 4),
            status     = 'Matched',
            method     = 'catalogue_fuzzy',
        )

    return _result(None, None, None, None, status='Unmatched', method='fuzzy')


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4 — KB ENRICHMENT
# ══════════════════════════════════════════════════════════════════════════════

def kb_enrich(matched_name: str, item_barcode: str | None,
              kb_by_barcode: dict, kb_by_name: dict) -> dict:
    """
    Try to find brand/category/concerns from dim_knowledge_base.
    First tries barcode, then fuzzy name match.
    Returns enrichment dict — all fields can be None if no KB match.
    """
    empty = {'brand': None, 'canonical_category': None,
             'sub_category': None, 'concerns': None, 'target_audience': None}

    # Try barcode first
    if item_barcode:
        bc = re.sub(r'[^0-9]', '', str(item_barcode)).lstrip('0')
        row = kb_by_barcode.get(bc)
        if row:
            return {
                'brand':              row.get('brand'),
                'canonical_category': row.get('canonical_category'),
                'sub_category':       row.get('sub_category'),
                'concerns':           row.get('concerns'),
                'target_audience':    row.get('target_audience'),
            }

    # Try name fuzzy match against KB
    if matched_name:
        clean_name = clean_for_match(matched_name)
        best_score = 0.0
        best_row   = None
        for row in kb_by_name.get(clean_name[:3], []):   # bucket by first 3 chars
            score = fuzzy_score(clean_name, row['_clean'])
            if score > best_score:
                best_score = score
                best_row   = row
        if best_row and best_score >= 0.65:
            return {
                'brand':              best_row.get('brand'),
                'canonical_category': best_row.get('canonical_category'),
                'sub_category':       best_row.get('sub_category'),
                'concerns':           best_row.get('concerns'),
                'target_audience':    best_row.get('target_audience'),
            }

    return empty


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
    department          TEXT,
    supplier            TEXT,
    concerns            TEXT,
    target_audience     TEXT,
    match_score         NUMERIC(5,4),
    match_status        TEXT,
    match_method        TEXT,
    mapped_at           TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_product_map_brand
    ON dim_product_map(brand);
CREATE INDEX IF NOT EXISTS idx_product_map_category
    ON dim_product_map(canonical_category);
CREATE INDEX IF NOT EXISTS idx_product_map_status
    ON dim_product_map(match_status);
"""

ALTER_TABLE_SQL = """
ALTER TABLE dim_product_map
    ADD COLUMN IF NOT EXISTS match_method  TEXT,
    ADD COLUMN IF NOT EXISTS department    TEXT,
    ADD COLUMN IF NOT EXISTS supplier      TEXT;
"""

ADD_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_product_map_method
    ON dim_product_map(match_method);
CREATE INDEX IF NOT EXISTS idx_product_map_dept
    ON dim_product_map(department);
"""

UPSERT_SQL = """
INSERT INTO dim_product_map (
    pos_description, matched_name, brand, canonical_category,
    sub_category, department, supplier, concerns, target_audience,
    match_score, match_status, match_method
) VALUES %s
ON CONFLICT (pos_description) DO UPDATE SET
    matched_name       = EXCLUDED.matched_name,
    brand              = EXCLUDED.brand,
    canonical_category = EXCLUDED.canonical_category,
    sub_category       = EXCLUDED.sub_category,
    department         = EXCLUDED.department,
    supplier           = EXCLUDED.supplier,
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
    print("=" * 65)
    print("🗺️  BUILD PRODUCT MAP  V3  (Catalogue-First Matching)")
    print(f"    Mode: {'FULL REMAP' if force_remap else 'Incremental'}")
    print("=" * 65)
    t0 = time.time()

    conn = get_conn()
    cur  = conn.cursor()

    cur.execute(CREATE_TABLE_SQL)
    conn.commit()
    cur.execute(ALTER_TABLE_SQL)
    conn.commit()
    cur.execute(ADD_INDEX_SQL)
    conn.commit()

    # ── 1. Load dim_product_catalogue ──────────────────────────────────────────
    print("\n📖 Loading product catalogue from dim_product_catalogue...")
    cur.execute("""
        SELECT item_lookup_code, canonical_name, department, category, supplier
        FROM dim_product_catalogue
        WHERE item_lookup_code IS NOT NULL
          AND canonical_name IS NOT NULL;
    """)
    cat_rows_raw = cur.fetchall()
    cat_cols = ['item_lookup_code', 'canonical_name', 'department', 'category', 'supplier']
    df_cat = pd.DataFrame(cat_rows_raw, columns=cat_cols)

    # Pre-compute clean text, sizes, counts
    df_cat['_clean'] = df_cat['canonical_name'].apply(clean_for_match)
    df_cat['_sizes'] = df_cat['canonical_name'].apply(extract_sizes)
    df_cat['_count'] = df_cat['canonical_name'].apply(extract_count)

    # Barcode → catalogue row lookup
    cat_by_barcode: dict = {}
    for row in df_cat.to_dict('records'):
        bc = str(row['item_lookup_code']).lstrip('0').strip()
        if bc:
            cat_by_barcode[bc] = row

    cat_rows_list = df_cat.to_dict('records')

    print(f"   ✅ {len(df_cat):,} catalogue products loaded")
    print(f"   ✅ {len(cat_by_barcode):,} barcodes indexed")

    # ── 2. Load dim_knowledge_base for enrichment only ─────────────────────────
    print("\n📖 Loading Knowledge Base for brand/category enrichment...")
    cur.execute("""
        SELECT name, brand, canonical_category, sub_category,
               concerns, target_audience,
               code_1, code_2, item_code
        FROM dim_knowledge_base
        WHERE name IS NOT NULL AND brand IS NOT NULL
          AND TRIM(name) != '' AND TRIM(brand) != '';
    """)
    kb_rows_raw = cur.fetchall()
    kb_cols = ['name', 'brand', 'canonical_category', 'sub_category',
               'concerns', 'target_audience', 'code_1', 'code_2', 'item_code']
    df_kb = pd.DataFrame(kb_rows_raw, columns=kb_cols)
    df_kb['_clean'] = df_kb['name'].apply(clean_for_match)

    # KB barcode → row lookup
    kb_by_barcode: dict = {}
    for row in df_kb.to_dict('records'):
        for col in ('code_1', 'code_2', 'item_code'):
            raw = row.get(col)
            if raw and str(raw).strip() not in ('', 'nan', 'None'):
                bc = re.sub(r'[^0-9]', '', str(raw)).lstrip('0')
                if bc and bc not in kb_by_barcode:
                    kb_by_barcode[bc] = row

    # KB name bucket lookup (first 3 chars of clean name → list of rows)
    kb_by_name: dict = {}
    for row in df_kb.to_dict('records'):
        key = row['_clean'][:3] if len(row['_clean']) >= 3 else row['_clean']
        kb_by_name.setdefault(key, []).append(row)

    print(f"   ✅ {len(df_kb):,} KB products loaded for enrichment")

    # ── 3. Build desc → barcodes map from fact_sales_lineitems ────────────────
    print("\n🔍 Building barcode lookup from fact_sales_lineitems...")
    cur.execute("""
        SELECT DISTINCT
            l.description,
            l.item
        FROM fact_sales_lineitems l
        WHERE l.item IS NOT NULL
          AND TRIM(l.item) != ''
          AND l.item != '#NULL#'
          AND l.description IS NOT NULL
          AND TRIM(l.description) != '';
    """)
    barcode_rows = cur.fetchall()

    desc_to_barcodes: dict = {}
    for desc, bc in barcode_rows:
        nbc = re.sub(r'[^0-9]', '', str(bc)).lstrip('0') if bc else None
        if nbc:
            desc_to_barcodes.setdefault(desc, set()).add(nbc)

    print(f"   ✅ {len(desc_to_barcodes):,} descriptions have POS barcodes")

    # ── 4. Fetch descriptions to process ──────────────────────────────────────
    print("\n🔌 Fetching POS descriptions...")
    if force_remap:
        cur.execute("""
            SELECT DISTINCT description
            FROM fact_sales_lineitems
            WHERE description IS NOT NULL
              AND TRIM(description) != ''
              AND description != '#NULL#'
            ORDER BY description;
        """)
    else:
        cur.execute("""
            SELECT DISTINCT description
            FROM fact_sales_lineitems
            WHERE description IS NOT NULL
              AND TRIM(description) != ''
              AND description != '#NULL#'
              AND description NOT IN (
                  SELECT pos_description FROM dim_product_map
                  WHERE match_status NOT IN ('Unmatched')
              )
            ORDER BY description;
        """)

    descriptions = [r[0] for r in cur.fetchall()]
    total = len(descriptions)
    print(f"   ✅ {total:,} descriptions to process")

    if not descriptions:
        print("\n   ℹ️  Nothing to process.")
        cur.close()
        conn.close()
        return

    # ── 5. Classify ────────────────────────────────────────────────────────────
    print("\n🔍 Running 4-stage classification pipeline...")
    counts = {k: 0 for k in
              ('Interbranch', 'Service', 'Matched_barcode',
               'Matched_catalogue_fuzzy', 'Unmatched')}
    batch: list = []

    for desc in tqdm(descriptions, desc="Classifying", unit="desc"):

        # Stage 1 — interbranch / service
        result = pre_classify(desc)
        if result:
            key = 'Interbranch' if result['match_status'] == 'Interbranch' else 'Service'
            counts[key] += 1

        # Stage 2 — barcode → catalogue
        if result is None:
            barcodes = desc_to_barcodes.get(desc, set())
            for bc in barcodes:
                cat_row = cat_by_barcode.get(bc)
                if cat_row:
                    # Found in catalogue via barcode — enrich from KB
                    enrichment = kb_enrich(
                        cat_row['canonical_name'], bc,
                        kb_by_barcode, kb_by_name
                    )
                    result = _result(
                        name       = cat_row['canonical_name'],
                        brand      = enrichment['brand'],
                        category   = enrichment['canonical_category'],
                        sub        = enrichment['sub_category'],
                        department = cat_row.get('department'),
                        supplier   = cat_row.get('supplier'),
                        score      = 1.0,
                        status     = 'Matched',
                        concerns   = enrichment['concerns'],
                        audience   = enrichment['target_audience'],
                        method     = 'barcode_catalogue',
                    )
                    counts['Matched_barcode'] += 1
                    break

        # Stage 3 — catalogue fuzzy
        if result is None:
            result = catalogue_fuzzy_match(desc, cat_rows_list)
            if result['match_status'] == 'Matched':
                # Enrich matched catalogue row from KB
                enrichment = kb_enrich(
                    result['matched_name'], None,
                    kb_by_barcode, kb_by_name
                )
                result['brand']              = enrichment['brand']
                result['canonical_category'] = enrichment['canonical_category']
                result['sub_category']       = enrichment['sub_category']
                result['concerns']           = enrichment['concerns']
                result['target_audience']    = enrichment['target_audience']
                counts['Matched_catalogue_fuzzy'] += 1
            else:
                counts['Unmatched'] += 1

        batch.append((
            desc,
            result['matched_name'],
            result['brand'],
            result['canonical_category'],
            result['sub_category'],
            result['department'],
            result['supplier'],
            result['concerns'],
            result['target_audience'],
            result['match_score'],
            result['match_status'],
            result['match_method'],
        ))

        if len(batch) >= BATCH_SIZE:
            psycopg2.extras.execute_values(
                cur, UPSERT_SQL, batch, page_size=BATCH_SIZE
            )
            conn.commit()
            batch = []
            gc.collect()

    if batch:
        psycopg2.extras.execute_values(
            cur, UPSERT_SQL, batch, page_size=BATCH_SIZE
        )
        conn.commit()

    # ── 6. Summary ─────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    matched = counts['Matched_barcode'] + counts['Matched_catalogue_fuzzy']

    print(f"\n{'='*65}")
    print(f"✅  PRODUCT MAP COMPLETE  in {elapsed:.1f}s")
    print(f"    Barcode → Catalogue:  {counts['Matched_barcode']:>6,}  "
          f"({counts['Matched_barcode']/total*100:.1f}%)")
    print(f"    Catalogue fuzzy:      {counts['Matched_catalogue_fuzzy']:>6,}  "
          f"({counts['Matched_catalogue_fuzzy']/total*100:.1f}%)")
    print(f"    Interbranch:          {counts['Interbranch']:>6,}  "
          f"({counts['Interbranch']/total*100:.1f}%)")
    print(f"    Service:              {counts['Service']:>6,}  "
          f"({counts['Service']/total*100:.1f}%)")
    print(f"    Unmatched:            {counts['Unmatched']:>6,}  "
          f"({counts['Unmatched']/total*100:.1f}%)")
    print(f"    ──────────────────────────────────────────────")
    print(f"    Total matched:        {matched:>6,}  ({matched/total*100:.1f}%)")
    print(f"{'='*65}")

    # ── 7. Breakdown ───────────────────────────────────────────────────────────
    print("\n📊 dim_product_map — breakdown by method & department:")
    cur.execute("""
        SELECT
            COALESCE(match_method, '—')         AS method,
            COALESCE(department, 'No Dept')     AS department,
            COUNT(*)                            AS descriptions
        FROM dim_product_map
        GROUP BY match_method, department
        ORDER BY
            CASE match_method
                WHEN 'barcode_catalogue'    THEN 1
                WHEN 'catalogue_fuzzy'      THEN 2
                WHEN 'pre-classify'         THEN 3
                ELSE 4
            END,
            descriptions DESC
        LIMIT 40;
    """)
    print(f"   {'Method':<22} {'Department':<26} {'Descriptions':>12}")
    print(f"   {'-'*62}")
    for row in cur.fetchall():
        print(f"   {str(row[0]):<22} {str(row[1]):<26} {row[2]:>12,}")

    cur.close()
    conn.close()


if __name__ == "__main__":
    force = "--force" in sys.argv
    run_build_product_map(force_remap=force)