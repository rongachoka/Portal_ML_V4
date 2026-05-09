"""
clean_and_merge_products.py
===========================
Produces a combined product reference from dim_products (POS) + KB.

Phase 1 — Barcode join
  Joins dim_products.item_barcode to KB.ItemCode (exact match).
  Matched rows are fully enriched immediately. No cleaning needed.

Phase 2 — Clean + fuzzy match unmatched rows
  For rows that didn't match on barcode:
    1. Expand POS abbreviations (LRP → LA ROCHE-POSAY, 200M → 200ML, etc.)
    2. Extract brand from cleaned description using KB brand list
    3. Fuzzy match cleaned description against KB names (brand-filtered)
    4. Flag by confidence for review

Output
  data/01_raw/product_review.csv     — full merged file with review flags
  data/01_raw/needs_review.csv       — only rows needing manual inspection
  (After review, run finalise_product_merge.py to produce the final KB)

USAGE:
  python clean_and_merge_products.py
  python clean_and_merge_products.py --show-stats   (print coverage breakdown)
"""

import re
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from difflib import SequenceMatcher
from tqdm import tqdm
from Portal_ML_V4.src.config.settings import RAW_DATA_DIR
from Portal_ML_V4.src.config.pos_aliases import BRAND_ALIASES, TERM_ALIASES

# ── PATHS ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent
KB_PATH     = RAW_DATA_DIR  / "Final_Knowledge_Base_PowerBI.csv"
POS_PATH    = RAW_DATA_DIR / "dim_products_raw.csv"   # export from DO
OUT_REVIEW  = RAW_DATA_DIR/ "product_review.csv"
OUT_FLAGGED = RAW_DATA_DIR / "needs_review.csv"


# ── POS EXPANSION MAP ─────────────────────────────────────────────────────────
# Imported from pos_aliases — single source of truth for all POS text normalisation.
# To add new abbreviations: edit pos_aliases.py, not here.
# TERM_ALIASES  : form factors, sizes, units  (CRM→CREAM, 200M→200ML, etc.)
# BRAND_ALIASES : brand shorthand             (LRP→La Roche-Posay, SS→Seven Seas, etc.)


# Merge into one map, both uppercased. BRAND_ALIASES wins on any key collision.
POS_EXPANSION_MAP = {
    **{k.upper(): v.upper() for k, v in TERM_ALIASES.items()},
    **{k.upper(): v.upper() for k, v in BRAND_ALIASES.items()},
}


# Stop words to strip before matching (units, filler words)
STOP_WORDS = {
    'FOR', 'WITH', 'OF', 'TO', 'IN', 'ON', 'AT', 'A', 'AN', 
    'OR', 'BY',
}


# ═══════════════════════════════════════════════════════════════════════════════
# TEXT CLEANING
# ═══════════════════════════════════════════════════════════════════════════════

def expand_pos_abbreviations(text: str) -> str:
    """Expands abbreviations using word-boundary-safe regex."""
    text = str(text).upper().strip()
    # Sort by length descending so longer matches take priority
    for abbr, full in sorted(POS_EXPANSION_MAP.items(), key=lambda x: len(x[0]), reverse=True):
        pattern = r'\b' + re.escape(abbr) + r'\b'
        text = re.sub(pattern, full, text)
    return text


def clean_for_matching(text: str) -> str:
    """Full normalisation pipeline for fuzzy matching."""
    text = expand_pos_abbreviations(text)
    text = re.sub(r'[^A-Z0-9\s]', ' ', text)          # strip punctuation
    tokens = [t for t in text.split()
              if t not in STOP_WORDS and len(t) > 1]
    return ' '.join(tokens)


def normalise_barcode(val) -> str | None:
    """Strips .0 suffixes and whitespace from barcode values."""
    if pd.isna(val):
        return None
    s = str(val).strip().replace('.0', '')
    return s if s else None


# ═══════════════════════════════════════════════════════════════════════════════
# BRAND EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

# Minimum character length for a brand name to be used in detection.
# Brands shorter than this match too broadly against common words.
MIN_BRAND_LENGTH = 4

# For multi-word brands, every token longer than this must be present
# in the description. Prevents "LA ROCHE-POSAY" matching if only "ROCHE" appears.
SIGNIFICANT_TOKEN_MIN_LEN = 3

# Single-word brands must appear within the first N tokens of the description.
# POS descriptions almost always lead with the brand name.
# e.g. "NIVEA VISAGE REFRESH" → NIVEA is token 0 ✅
# e.g. "THE GARDEN HAND CARE KIT" → CARE is token 3, outside limit ✅ (blocked)
BRAND_POSITION_LIMIT = 3

# Single-word brand names that are common English words — excluded entirely.
# These generate false positives when they appear mid-description.
# Add to this list as you find new false positives in the review file.
BRAND_EXCLUSION_LIST = {
    "CARE", "STEP", "REFRESH", "PURE", "SOFT", "FRESH", "CLEAN",
    "GLOW", "BRIGHT", "SMOOTH", "PLUS", "MAX", "PRO", "ACT",
    "TOTAL", "BASIC", "PRIME", "EXTRA", "ACTIVE", "NATURAL",
    "GENTLE", "DAILY", "NIGHT", "DAY", "ULTRA", "HYDRO",
    "ADVANCED", "ORIGINAL", "CLASSIC", "SENSITIVE", "NORMAL",
}


def build_brand_patterns(brands: list) -> list[tuple]:
    """
    Pre-compiles brand patterns for two-pass matching against the KB Brand column.

    Returns list of (pattern, brand_name, significant_tokens, is_multiword) where:
      pattern           : compiled full-name regex (word-boundary safe)
      brand_name        : original KB Brand value (used in output)
      significant_tokens: tokens len >= SIGNIFICANT_TOKEN_MIN_LEN
                          ALL must be present in description for a match to count
      is_multiword      : True if brand has more than one word

    Filters out:
      - Brands shorter than MIN_BRAND_LENGTH characters
      - Single-word brands in BRAND_EXCLUSION_LIST
    """
    patterns = []
    skipped  = []

    for brand in sorted(brands, key=len, reverse=True):
        b_clean   = re.sub(r'[^A-Z0-9\s]', ' ', str(brand).upper())
        b_clean   = re.sub(r'\s+', ' ', b_clean).strip()
        b_no_the  = re.sub(r'^THE\s+', '', b_clean)

        if len(b_clean) < MIN_BRAND_LENGTH:
            skipped.append(brand)
            continue

        tokens = b_clean.split()
        if len(tokens) == 1 and tokens[0] in BRAND_EXCLUSION_LIST:
            skipped.append(brand)
            continue

        is_multiword   = len(tokens) > 1
        sig_tokens     = [t for t in tokens if len(t) >= SIGNIFICANT_TOKEN_MIN_LEN]

        pat = re.compile(r'\b' + re.escape(b_clean) + r'\b')
        patterns.append((pat, brand, sig_tokens, is_multiword))

        if b_no_the != b_clean:
            no_the_tokens = [t for t in b_no_the.split()
                             if len(t) >= SIGNIFICANT_TOKEN_MIN_LEN]
            pat2 = re.compile(r'\b' + re.escape(b_no_the) + r'\b')
            patterns.append((pat2, brand, no_the_tokens, is_multiword))

    if skipped:
        print(f"   ℹ️  {len(skipped)} brands excluded from detection (too short or generic):")
        print(f"      {sorted(skipped)}")

    return patterns


def extract_brand(cleaned_desc: str, brand_patterns: list) -> str | None:
    """
    Two-pass brand extraction. Only matches brands from the KB Brand column.

    Pass 1 — multi-word brands only
      Full brand name regex must match AND every significant token (len >= 3)
      must be present in the description.
      e.g. "LA ROCHE-POSAY" requires ROCHE + POSAY both in the description.
      This prevents partial matches like "ROCHE" alone triggering "LA ROCHE-POSAY".

    Pass 2 — single-word brands only
      Brand token must appear within the first BRAND_POSITION_LIMIT tokens.
      Brands in POS data almost always lead the description.
      e.g. "NIVEA VISAGE REFRESH 2IN1" → NIVEA is token 0 ✅
      e.g. "NIVEA" appearing at token 5 in a long description → blocked ✅

    Returns the canonical KB brand name or None.
    """
    desc_tokens = cleaned_desc.split()
    desc_set    = set(desc_tokens)
    first_n     = set(desc_tokens[:BRAND_POSITION_LIMIT])

    # Pass 1: multi-word brands — strictest check first
    for pat, brand, sig_tokens, is_multiword in brand_patterns:
        if not is_multiword:
            continue
        if not pat.search(cleaned_desc):
            continue
        # Every significant token must appear in the description
        if sig_tokens and not all(t in desc_set for t in sig_tokens):
            continue
        return brand

    # Pass 2: single-word brands — position-gated
    for pat, brand, sig_tokens, is_multiword in brand_patterns:
        if is_multiword:
            continue
        if not pat.search(cleaned_desc):
            continue
        brand_token = sig_tokens[0] if sig_tokens else None
        if brand_token and brand_token not in first_n:
            continue
        return brand

    return None


# ═══════════════════════════════════════════════════════════════════════════════
# FUZZY MATCHING
# ═══════════════════════════════════════════════════════════════════════════════

def fuzzy_score(a: str, b: str) -> float:
    """Hybrid token + sequence score."""
    if not a or not b:
        return 0.0

    seq = SequenceMatcher(None, a, b).ratio()

    ta = set(a.split()) - STOP_WORDS
    tb = set(b.split()) - STOP_WORDS

    if not ta or not tb:
        return seq

    inter    = ta & tb
    jaccard  = len(inter) / len(ta | tb)
    coverage = len(inter) / min(len(ta), len(tb))
    token    = jaccard * 0.4 + coverage * 0.6

    return seq * 0.35 + token * 0.65


def find_best_kb_match(
    cleaned_desc: str,
    brand: str | None,
    kb_by_brand: dict,
    kb_all: list
) -> tuple[dict | None, float]:
    """
    Returns (best_kb_row, score).
    If brand is known, searches only within that brand's KB products.
    If brand is unknown, searches all KB products (slower, lower threshold).
    """
    candidates = kb_by_brand.get(str(brand).upper(), []) if brand else kb_all

    best_score = 0.0
    best_match = None

    for row in candidates:
        score = fuzzy_score(cleaned_desc, row['_clean_name'])
        if score > best_score:
            best_score = score
            best_match = row

    return best_match, best_score


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--show-stats', action='store_true',
                        help='Print detailed coverage stats')
    args = parser.parse_args()

    # ── 1. LOAD ───────────────────────────────────────────────────────────────
    print("📥 Loading files...")

    if not KB_PATH.exists():
        print(f"❌ KB not found: {KB_PATH}"); sys.exit(1)
    if not POS_PATH.exists():
        print(f"❌ dim_products export not found: {POS_PATH}")
        print(f"   Run pull_and_merge_products.py first to export from DO.")
        sys.exit(1)

    df_kb  = pd.read_csv(KB_PATH)
    df_pos = pd.read_csv(POS_PATH, low_memory=False)

    df_kb.columns  = df_kb.columns.str.strip()
    df_pos.columns = df_pos.columns.str.strip()

    # Drop columns we don't want in the final file
    df_kb  = df_kb.drop(columns=['Detailed_Desc'],          errors='ignore')
    df_pos = df_pos.drop(columns=['selling_price_incl_vat'], errors='ignore')

    print(f"   KB:          {len(df_kb):,} products")
    print(f"   dim_products: {len(df_pos):,} rows")

    # ── 2. NORMALISE BARCODES ─────────────────────────────────────────────────
    print("\n🔑 Normalising barcodes...")

    df_kb['_barcode']  = df_kb['ItemCode'].apply(normalise_barcode)
    df_pos['_barcode'] = df_pos['item_barcode'].apply(normalise_barcode)

    kb_null_barcodes  = df_kb['_barcode'].isna().sum()
    pos_null_barcodes = df_pos['_barcode'].isna().sum()

    print(f"   KB  missing barcodes: {kb_null_barcodes:,} / {len(df_kb):,}")
    print(f"   POS missing barcodes: {pos_null_barcodes:,} / {len(df_pos):,}")

    # ── 3. PHASE 1: BARCODE JOIN ──────────────────────────────────────────────
    print("\n🔗 Phase 1: Barcode join...")

    # Build KB lookup keyed by barcode
    # Deduplicate first — duplicate ItemCodes cause set_index to fail.
    # Keep first occurrence; duplicates are logged so you can review them.
    df_kb_barcoded = df_kb.dropna(subset=['_barcode'])
    dupes = df_kb_barcoded[df_kb_barcoded.duplicated(subset=['_barcode'], keep=False)]
    if not dupes.empty:
        print(f"   ⚠️  {dupes['_barcode'].nunique()} duplicate barcodes in KB — keeping first occurrence.")
        print(f"      Affected barcodes: {dupes['_barcode'].unique()[:10].tolist()} ...")
    df_kb_barcoded = df_kb_barcoded.drop_duplicates(subset=['_barcode'], keep='first')
    kb_barcode_lookup = df_kb_barcoded.set_index('_barcode').to_dict('index')

    # KB enrichment columns (everything except ItemCode which becomes item_barcode)
    kb_enrich_cols = [c for c in df_kb.columns
                      if c not in ['ItemCode', '_barcode']]

    # Initialise result columns on df_pos
    for col in kb_enrich_cols:
        df_pos[f'kb_{col}'] = None

    df_pos['match_method'] = 'unmatched'
    df_pos['confidence']   = 0
    df_pos['needs_review'] = True

    barcode_matched = 0
    for idx, row in tqdm(df_pos.iterrows(), total=len(df_pos), desc='Phase 1: Barcode join'):
        bc = row['_barcode']
        if bc and bc in kb_barcode_lookup:
            kb_row = kb_barcode_lookup[bc]
            for col in kb_enrich_cols:
                df_pos.at[idx, f'kb_{col}'] = kb_row.get(col)
            df_pos.at[idx, 'match_method'] = 'barcode'
            df_pos.at[idx, 'confidence']   = 100
            df_pos.at[idx, 'needs_review'] = False
            barcode_matched += 1

    print(f"   ✅ Barcode matched: {barcode_matched:,} / {len(df_pos):,} "
          f"({barcode_matched/len(df_pos)*100:.1f}%)")

    # ── 4. PHASE 2: CLEAN + FUZZY MATCH UNMATCHED ROWS ───────────────────────
    unmatched_mask = df_pos['match_method'] == 'unmatched'
    n_unmatched    = unmatched_mask.sum()
    print(f"\n🧹 Phase 2: Cleaning + fuzzy matching {n_unmatched:,} unmatched rows...")

    # Pre-process KB for fuzzy matching
    brands = df_kb['Brand'].dropna().astype(str).str.strip().unique().tolist()
    brand_patterns = build_brand_patterns(brands)

    # Build KB lookup by brand for fast candidate filtering
    df_kb['_clean_name'] = df_kb['Name'].apply(clean_for_matching)
    kb_by_brand = {}
    for _, kb_row in df_kb.iterrows():
        brand_key = str(kb_row.get('Brand', '')).upper().strip()
        if brand_key not in kb_by_brand:
            kb_by_brand[brand_key] = []
        kb_by_brand[brand_key].append(kb_row.to_dict())
    kb_all = df_kb.to_dict('records')

    # Threshold — only accept fuzzy matches at 75% and above.
    # Anything below goes to needs_review with no KB columns populated.
    THRESHOLD = 0.75

    brand_detected_count  = 0
    fuzzy_matched_count   = 0
    brand_only_count      = 0
    still_unmatched_count = 0

    # Apply cleaning and matching to unmatched rows only
    df_pos['description_cleaned'] = df_pos['description'].astype(str)  # initialise

    for idx, row in tqdm(df_pos[unmatched_mask].iterrows(), total=n_unmatched, desc='Phase 2: Fuzzy match'):
        raw_desc     = str(row['description'])
        cleaned_desc = clean_for_matching(raw_desc)
        df_pos.at[idx, 'description_cleaned'] = cleaned_desc

        # Step 1: detect brand
        brand = extract_brand(cleaned_desc, brand_patterns)
        if brand:
            df_pos.at[idx, 'kb_Brand'] = brand
            brand_detected_count += 1

        # Step 2: fuzzy match
        best_match, score = find_best_kb_match(
            cleaned_desc, brand, kb_by_brand, kb_all
        )

        if best_match and score >= THRESHOLD:
            for col in kb_enrich_cols:
                df_pos.at[idx, f'kb_{col}'] = best_match.get(col)
            df_pos.at[idx, 'match_method'] = 'fuzzy_name'
            df_pos.at[idx, 'confidence']   = round(score * 100)
            df_pos.at[idx, 'needs_review'] = True   # spot check even high confidence
            fuzzy_matched_count += 1

        elif brand:
            # Brand detected but score below 75 — don't assign a product match.
            # KB Brand column is still populated from the brand detection step above.
            df_pos.at[idx, 'match_method'] = 'brand_only'
            df_pos.at[idx, 'confidence']   = round(score * 100) if best_match else 0
            df_pos.at[idx, 'needs_review'] = True
            brand_only_count += 1

        else:
            df_pos.at[idx, 'match_method'] = 'unmatched'
            df_pos.at[idx, 'confidence']   = 0
            df_pos.at[idx, 'needs_review'] = True
            still_unmatched_count += 1

    print(f"   Brand detected:            {brand_detected_count:,}")
    print(f"   Fuzzy matched (≥75):       {fuzzy_matched_count:,}")
    print(f"   Brand only (<75 or no hit): {brand_only_count:,}")
    print(f"   Still unmatched:           {still_unmatched_count:,}")

    # ── 5. BUILD FINAL SCHEMA ─────────────────────────────────────────────────
    print("\n📋 Building final schema...")

    # Rename kb_ prefixed columns to clean names
    rename_map = {f'kb_{col}': col for col in kb_enrich_cols}
    # Avoid collision with existing POS columns
    for kb_col, clean_name in list(rename_map.items()):
        if clean_name in df_pos.columns and not clean_name.startswith('kb_'):
            rename_map[kb_col] = f'{clean_name}_kb'
    df_pos = df_pos.rename(columns=rename_map)

    # Rename POS category to avoid collision with KB Canonical_Category
    if 'category' in df_pos.columns:
        df_pos = df_pos.rename(columns={'category': 'category_pos'})

    # Final column order — review columns first for easy scanning
    review_cols = [
        'item_barcode', 'description', 'description_cleaned',
        'match_method', 'confidence', 'needs_review',
    ]
    kb_cols_final = [
        'Name', 'Brand', 'Canonical_Category', 'Sub_Category',
        'Concerns', 'Target_Audience', 'Price_kb',
        'Quantity', 'Product_Link',
    ]
    # Handle the Price collision — KB Price becomes Price_kb
    if 'Price' in df_pos.columns and 'Price_kb' not in df_pos.columns:
        df_pos = df_pos.rename(columns={'Price': 'Price_kb'})

    pos_cols_remaining = [
        'department', 'category_pos', 'supplier',
        'cost_price', 'selling_price', 'margin_amount', 'margin_pct',
        'location', 'loaded_at', 'department_mapped',
    ]

    all_cols = (
        review_cols +
        [c for c in kb_cols_final if c in df_pos.columns] +
        [c for c in pos_cols_remaining if c in df_pos.columns]
    )
    # Add any remaining columns not explicitly ordered
    leftover = [c for c in df_pos.columns
                if c not in all_cols and not c.startswith('_')]
    final_cols = all_cols + leftover

    df_out = df_pos[[c for c in final_cols if c in df_pos.columns]].copy()

    # ── 6. SAVE ───────────────────────────────────────────────────────────────
    OUT_REVIEW.parent.mkdir(parents=True, exist_ok=True)

    df_out.to_csv(OUT_REVIEW, index=False)
    print(f"\n✅ Full review file: {OUT_REVIEW}")
    print(f"   {len(df_out):,} rows")

    # Flagged-only file for focused review
    df_flagged = df_out[df_out['needs_review'] == True].copy()
    df_flagged.to_csv(OUT_FLAGGED, index=False)
    print(f"✅ Needs review:     {OUT_FLAGGED}")
    print(f"   {len(df_flagged):,} rows flagged for review")

    # ── 7. COVERAGE SUMMARY ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("📊 COVERAGE SUMMARY")
    print("=" * 60)
    method_counts = df_out['match_method'].value_counts()
    total = len(df_out)
    for method, count in method_counts.items():
        bar = "█" * int(count / total * 40)
        print(f"   {method:<20} {count:>7,}  {count/total*100:>5.1f}%  {bar}")
    print(f"   {'─'*60}")
    print(f"   {'TOTAL':<20} {total:>7,}  100.0%")

    if args.show_stats:
        print("\n📊 CONFIDENCE DISTRIBUTION (fuzzy_name rows only)")
        fuzzy_rows = df_out[df_out['match_method'] == 'fuzzy_name']
        if not fuzzy_rows.empty:
            conf_bins = pd.cut(
                fuzzy_rows['confidence'],
                bins=[75, 80, 85, 90, 95, 100],
                labels=['75-80', '81-85', '86-90', '91-95', '96-100']
            )
            for label, count in conf_bins.value_counts().sort_index().items():
                print(f"   {label:<10} {count:>7,}")

        print("\n📊 UNMATCHED ROWS — TOP DEPARTMENTS")
        unmatched = df_out[df_out['match_method'] == 'unmatched']
        if 'department' in unmatched.columns:
            dept_counts = unmatched['department'].value_counts().head(15)
            for dept, count in dept_counts.items():
                print(f"   {str(dept):<35} {count:>6,}")

    print("\n📌 NEXT STEPS:")
    print("   1. Open needs_review.csv and inspect fuzzy matches")
    print("   2. For incorrect matches: clear the Name/Brand/Category columns")
    print("      and fill in correct values manually")
    print("   3. For unmatched rows you want to enrich: fill in Name, Brand,")
    print("      Canonical_Category columns directly")
    print("   4. Save the corrected file as needs_review_corrected.csv")
    print("   5. Run finalise_product_merge.py to produce the new KB")


if __name__ == "__main__":
    main()