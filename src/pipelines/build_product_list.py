"""
build_product_catalog.py
========================
Builds a clean product catalog from dim_products_raw using a staged approach.

Stage 1 — Barcode match:
    Join item_barcode → KB (Item Code Final, then ItemCode).
    Pulls Name, Brand, Canonical_Category, Sub_Category directly.

Stage 2 — Term aliases (on Stage 1 unmatched):
    Expands TERM_ALIASES on description (CRM→CREAM, SQUAL→SQUALANE, etc.).
    Cleans the text. Does not assign brand/category on its own.

Stage 3 — Brand aliases (on Stage 2 unmatched):
    Detects brand from expanded description using BRAND_ALIASES.
    Category comes from KB brand-to-category map.
    If brand not in KB map → category left blank.

Output: one row per unique item_barcode with Match_Stage column.
"""

import re
import os
import sys
import pandas as pd
from pathlib import Path
from collections import Counter

# ── IMPORTS ───────────────────────────────────────────────────────────────────
try:
    from Portal_ML_V4.src.config.pos_aliases import TERM_ALIASES, BRAND_ALIASES
    print(f"✅ pos_aliases: {len(TERM_ALIASES)} term aliases · {len(BRAND_ALIASES)} brand aliases")
except ImportError:
    print("❌ Could not import pos_aliases.")
    print("   Run from: D:\\Documents\\Portal ML Analys\\Portal_ML")
    sys.exit(1)

# ── PATHS ─────────────────────────────────────────────────────────────────────
BASE_DIR       = Path(r"D:\Documents\Portal ML Analys\Portal_ML\Portal_ML_V4")
DIM_PROD_RAW   = BASE_DIR / "data" / "01_raw" / "dim_products_raw.csv"
KB_PATH        = BASE_DIR / "data" / "01_raw" / "Final_Knowledge_Base_PowerBI.csv"
KB_COPY_PATH   = BASE_DIR / "data" / "01_raw" / "Final_Knowledge_Base_PowerBI - Copy.csv"
OUTPUT_DIR     = BASE_DIR / "data" / "03_processed" / "pos_catalog"
OUTPUT_FILE    = OUTPUT_DIR / "product_catalog_v1.csv"
UNMATCHED_FILE = OUTPUT_DIR / "unmatched_review.csv"


# =============================================================================
# HELPERS
# =============================================================================

def clean_barcode(val) -> str | None:
    """
    Normalise a barcode to a plain digit string.
    Handles floats (6281006123.0), leading apostrophes, whitespace, NaN.
    Returns None if the result is fewer than 4 digits (junk).
    """
    if pd.isna(val):
        return None
    s = str(val).strip().lstrip("'")
    # Remove trailing .0 from floats
    s = re.sub(r'\.0+$', '', s).strip()
    # Keep only digits
    digits = re.sub(r'[^\d]', '', s)
    return digits if len(digits) >= 4 else None


def expand_term_aliases(text: str) -> str:
    """
    Replace TERM_ALIASES tokens using word boundaries (case-insensitive).
    Returns the expanded, upper-cased string.
    """
    text = str(text).upper()
    for alias, full in TERM_ALIASES.items():
        pattern = r'\b' + re.escape(alias.upper()) + r'\b'
        text = re.sub(pattern, full.upper(), text)
    return text


def detect_brand(desc: str, kb_brands: list | None = None) -> str | None:
    """
    Two-pass brand detection:
      Pass 1 — BRAND_ALIASES (handles typos/shorthands: LRP, S/SEAS, BBW, etc.)
      Pass 2 — KB brand list (catches clean spellings already in KB: NIVEA,
               AVEENO, CERAVE, OLAY — no manual alias entry needed)
    Longest match checked first in both passes.
    Returns canonical brand name (title-cased), or None.
    """
    safe = re.sub(r'[^A-Z0-9\s]', ' ', str(desc).upper())
    safe = re.sub(r'\s+', ' ', safe).strip()

    # Pass 1: BRAND_ALIASES
    sorted_aliases = sorted(BRAND_ALIASES.items(), key=lambda x: len(x[0]), reverse=True)
    for alias, canonical in sorted_aliases:
        alias_clean = re.sub(r'[^A-Z0-9\s]', ' ', str(alias).upper()).strip()
        if not alias_clean:
            continue
        if re.search(r'\b' + re.escape(alias_clean) + r'\b', safe):
            return str(canonical).title()

    # Pass 2: KB brand list (sorted longest-first, passed in from load_kb)
    if kb_brands:
        for brand in kb_brands:
            b = re.sub(r'[^A-Z0-9\s]', ' ', str(brand).upper())
            b = re.sub(r'\s+', ' ', b).strip()
            if len(b) < 2:
                continue
            b_no_the = re.sub(r'^THE\s+', '', b)
            if re.search(r'\b' + re.escape(b) + r'\b', safe):
                return str(brand).title()
            if b_no_the != b and re.search(r'\b' + re.escape(b_no_the) + r'\b', safe):
                return str(brand).title()

    return None


def detect_brand_from_aliases(desc: str) -> str | None:
    """
    Scan desc for BRAND_ALIASES keys (longest alias checked first).
    Returns canonical brand name (title-cased), or None if no match.
    """
    safe = re.sub(r'[^A-Z0-9\s]', ' ', str(desc).upper())
    safe = re.sub(r'\s+', ' ', safe).strip()

    # Longest alias first — prevents 'roche' matching before 'la roche posay'
    sorted_aliases = sorted(BRAND_ALIASES.items(), key=lambda x: len(x[0]), reverse=True)
    for alias, canonical in sorted_aliases:
        alias_clean = re.sub(r'[^A-Z0-9\s]', ' ', str(alias).upper()).strip()
        if not alias_clean:
            continue
        if re.search(r'\b' + re.escape(alias_clean) + r'\b', safe):
            return str(canonical).title()
    return None


# =============================================================================
# 1. LOAD KNOWLEDGE BASE
# =============================================================================

def _parse_one_kb(path: Path) -> tuple[dict, dict]:
    """
    Parse a single KB CSV into:
      barcode_map   : {barcode_str → {Matched_Product, Matched_Brand,
                                       Matched_Category, Matched_Sub_Category}}
      brand_cat_map : {brand_title → Canonical_Category}
    """
    df = pd.read_csv(path, dtype=str).fillna("")
    df.columns = df.columns.str.strip()

    # ── Resolve barcode columns ───────────────────────────────────────────────
    col_final = next((c for c in df.columns if c.strip() == "Item Code Final"), None)
    col_item  = next((c for c in df.columns if c.strip() == "ItemCode"),        None)

    if col_final:
        df["_bc"] = df[col_final].str.strip()
    else:
        df["_bc"] = ""

    if col_item:
        mask_blank = df["_bc"].eq("")
        df.loc[mask_blank, "_bc"] = df.loc[mask_blank, col_item].str.strip()

    df["_bc_clean"] = df["_bc"].apply(clean_barcode)

    # Canonical_Category column (handle pandas duplicate-column suffix _1)
    cat_col = next(
        (c for c in df.columns if c.strip().lower() == "canonical_category"),
        None
    )
    brand_col = next((c for c in df.columns if c.strip() == "Brand"), None)

    # ── Barcode map ───────────────────────────────────────────────────────────
    df_with_bc = df[df["_bc_clean"].notna()].copy()
    barcode_map = {}
    for _, row in df_with_bc.iterrows():
        bc = row["_bc_clean"]
        if bc not in barcode_map:
            barcode_map[bc] = {
                "Matched_Product":      row.get("Name", "").strip(),
                "Matched_Brand":        str(row.get("Brand", "")).strip().title(),
                "Matched_Category":     row.get(cat_col, "").strip() if cat_col else "",
                "Matched_Sub_Category": row.get("Sub_Category", "").strip(),
            }

    # ── Brand → Category map ──────────────────────────────────────────────────
    brand_cat_map = {}
    if brand_col and cat_col:
        tmp = df[(df[brand_col].str.strip() != "") & (df[cat_col].str.strip() != "")]
        brand_cat_map = (
            tmp.groupby(tmp[brand_col].str.strip().str.title())[cat_col]
            .agg(lambda x: x.mode()[0] if not x.mode().empty else "")
            .to_dict()
        )
        brand_cat_map = {k: v for k, v in brand_cat_map.items() if v}

    # Sorted longest-first so "La Roche Posay" beats "Posay" in word scan
    brands = sorted(
        [b for b in df[brand_col].str.strip().str.title().unique() if b]
        if brand_col else [],
        key=len, reverse=True
    )

    return barcode_map, brand_cat_map, brands, len(df), len(df_with_bc)


def load_kb():
    """
    Loads the primary KB and the copy, merging them with clear priority:

      Barcodes      — primary KB wins; copy fills gaps only
      Brand→Category — primary KB wins; copy adds brands not present in primary

    Returns:
      barcode_map   : merged {barcode_str → product info dict}
      brand_cat_map : merged {brand_title → Canonical_Category}
    """
    if not KB_PATH.exists():
        raise FileNotFoundError(f"Primary KB not found: {KB_PATH}")

    print(f"   📄 Primary KB:  {KB_PATH.name}")
    bc_primary, bcat_primary, brands_primary, rows_p, bc_rows_p = _parse_one_kb(KB_PATH)
    print(f"      {rows_p:,} rows  │  {bc_rows_p:,} with barcode  │  {len(bc_primary):,} unique barcodes  │  {len(bcat_primary):,} brands")

    bc_copy, bcat_copy, brands_copy = {}, {}, []
    if KB_COPY_PATH.exists():
        print(f"   📄 Copy KB:     {KB_COPY_PATH.name}")
        bc_copy, bcat_copy, brands_copy, rows_c, bc_rows_c = _parse_one_kb(KB_COPY_PATH)
        print(f"      {rows_c:,} rows  │  {bc_rows_c:,} with barcode  │  {len(bc_copy):,} unique barcodes  │  {len(bcat_copy):,} brands")
    else:
        print(f"   ⚠️  Copy KB not found — using primary only.")

    # Merge: primary wins, copy fills gaps
    barcode_map   = {**bc_copy,   **bc_primary}
    brand_cat_map = {**bcat_copy, **bcat_primary}

    # Merged brand list — unique, sorted longest-first
    all_brands = list({b.title() for b in brands_primary + brands_copy if b})
    all_brands.sort(key=len, reverse=True)

    new_barcodes = len(barcode_map) - len(bc_primary)
    new_brands   = len(brand_cat_map) - len(bcat_primary)
    print(f"\n   ✅ Merged KB:")
    print(f"      {len(barcode_map):,} unique barcodes  (+{new_barcodes:,} from copy)")
    print(f"      {len(brand_cat_map):,} brand→category entries  (+{new_brands:,} from copy)")
    print(f"      {len(all_brands):,} brands available for Stage 3 detection")

    return barcode_map, brand_cat_map, all_brands


# =============================================================================
# 2. LOAD POS PRODUCT LIST
# =============================================================================

def load_pos() -> pd.DataFrame:
    if not DIM_PROD_RAW.exists():
        raise FileNotFoundError(f"dim_products_raw not found: {DIM_PROD_RAW}")

    df = pd.read_csv(DIM_PROD_RAW, dtype=str).fillna("")
    df.columns = df.columns.str.strip()
    print(f"\n📦 dim_products_raw: {len(df):,} rows")

    df["_bc_clean"] = df["item_barcode"].apply(clean_barcode)
    return df


# =============================================================================
# 3. STAGE 1 — BARCODE MATCH
# =============================================================================

def stage1_barcode(df: pd.DataFrame, barcode_map: dict):
    df = df.copy()
    df["_kb_hit"] = df["_bc_clean"].map(barcode_map)

    matched   = df[df["_kb_hit"].notna()].copy()
    unmatched = df[df["_kb_hit"].isna()].copy()

    for col in ["Matched_Product", "Matched_Brand", "Matched_Category", "Matched_Sub_Category"]:
        matched[col] = matched["_kb_hit"].apply(
            lambda x: x.get(col, "") if isinstance(x, dict) else ""
        )

    matched["Match_Stage"] = "Stage 1 – Barcode"
    matched = matched.drop(columns=["_kb_hit"], errors="ignore")

    print(f"\n   Stage 1 (Barcode):     {len(matched):>6,} matched  │  {len(unmatched):>6,} remaining")
    return matched, unmatched


# =============================================================================
# 4. STAGE 2 — TERM ALIAS EXPANSION
# =============================================================================

def stage2_expand_terms(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expands TERM_ALIASES on description → new column 'expanded_description'.
    No brand or category assigned here.
    """
    df = df.copy()
    df["expanded_description"] = df["description"].apply(expand_term_aliases)
    print(f"   Stage 2 (Term Expand): {len(df):>6,} descriptions expanded")
    return df


# =============================================================================
# 5. STAGE 3 — BRAND ALIAS DETECTION
# =============================================================================

def stage3_brand_aliases(df: pd.DataFrame, brand_cat_map: dict, kb_brands: list):
    df = df.copy()
    df["Matched_Brand"] = df["expanded_description"].apply(
        lambda d: detect_brand(d, kb_brands)
    )

    matched   = df[df["Matched_Brand"].notna()].copy()
    unmatched = df[df["Matched_Brand"].isna()].copy()

    # Category from KB brand map — blank if not in map (intentional)
    matched["Matched_Category"]     = matched["Matched_Brand"].map(brand_cat_map).fillna("")
    matched["Matched_Sub_Category"] = ""
    matched["Matched_Product"]      = ""
    matched["Match_Stage"]          = "Stage 3 – Brand Alias"

    # Consistent schema for unmatched
    for col in ["Matched_Brand", "Matched_Category", "Matched_Sub_Category", "Matched_Product"]:
        unmatched[col] = ""
    unmatched["Match_Stage"] = "Unmatched"

    print(f"   Stage 3 (Brand Alias): {len(matched):>6,} matched  │  {len(unmatched):>6,} remaining")
    return matched, unmatched


# =============================================================================
# 6. ASSEMBLE + SAVE
# =============================================================================

OUTPUT_COLS = [
    "item_barcode",
    "description",
    "expanded_description",
    "Matched_Brand",
    "Matched_Product",
    "Matched_Category",
    "Matched_Sub_Category",
    "Match_Stage",
    # Original POS reference columns
    "department",
    "category",
    "supplier",
    "selling_price",
    "cost_price",
    "location",
]


def run():
    print("=" * 65)
    print("🏗️  BUILD PRODUCT CATALOG  —  Staged Matching")
    print("    Stage 1: Barcode  |  Stage 2: Term Expand  |  Stage 3: Brand")
    print("=" * 65)

    # Load
    print("\n📚 Loading Knowledge Base...")
    barcode_map, brand_cat_map, kb_brands = load_kb()

    df_pos = load_pos()
    total_pos = len(df_pos)

    # Stages
    print("\n🔍 Running stages...")
    s1_matched, s1_remaining = stage1_barcode(df_pos, barcode_map)
    s2_expanded              = stage2_expand_terms(s1_remaining)
    s3_matched, s3_unmatched = stage3_brand_aliases(s2_expanded, brand_cat_map, kb_brands)

    # Stage 1 rows didn't go through term expansion — do it now for consistency
    s1_matched["expanded_description"] = s1_matched["description"].apply(expand_term_aliases)

    # Stack all
    df_all = pd.concat([s1_matched, s3_matched, s3_unmatched], ignore_index=True)

    # Keep only columns that exist
    present = [c for c in OUTPUT_COLS if c in df_all.columns]
    df_final = df_all[present].copy()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("📊 FINAL SUMMARY")
    print(f"   {'─'*55}")
    print(f"   {'Stage':<38} {'Count':>7}  {'%':>6}")
    print(f"   {'─'*55}")
    for stage in ["Stage 1 – Barcode", "Stage 3 – Brand Alias", "Unmatched"]:
        n   = (df_final["Match_Stage"] == stage).sum()
        pct = n / total_pos * 100 if total_pos else 0
        print(f"   {stage:<38} {n:>7,}  {pct:>5.1f}%")
    print(f"   {'─'*55}")
    print(f"   {'TOTAL':<38} {total_pos:>7,}  100.0%")
    print("=" * 65)

    # ── Save full catalog ─────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df_final.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ Catalog saved:    {OUTPUT_FILE}")

    # ── Save unmatched separately for review ──────────────────────────────────
    df_unmatched = df_final[df_final["Match_Stage"] == "Unmatched"].copy()
    if not df_unmatched.empty:
        df_unmatched.to_csv(UNMATCHED_FILE, index=False)
        print(f"📋 Unmatched saved: {UNMATCHED_FILE}  ({len(df_unmatched):,} rows)")

        # Token frequency on unmatched — reveals what to add to pos_aliases next
        SKIP = {
            'FOR','WITH','AND','OF','TO','IN','ON','AT','BY','ML','MG',
            'GM','PCS','TAB','CAP','SYR','TUBE','EACH','UNIT','PKT','NEW',
            'PLUS','PACK','SET','KIT','SIZE','REG','STD','PER','USE',
        }
        token_counts = Counter()
        for desc in df_unmatched["expanded_description"].dropna():
            tokens = [
                t for t in str(desc).upper().split()
                if len(t) >= 3 and t not in SKIP and not t.isdigit()
            ]
            token_counts.update(tokens)

        print(f"\n🔬 TOP 40 TOKENS IN UNMATCHED DESCRIPTIONS")
        print(f"   Review these — likely candidates for pos_aliases.py")
        print(f"   {'─'*40}")
        for token, count in token_counts.most_common(40):
            print(f"   {token:<28} {count:>5}x")
    else:
        print("🎉 No unmatched rows — full coverage!")


if __name__ == "__main__":
    run()