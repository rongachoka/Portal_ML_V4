"""
export_respondio_tags.py
─────────────────────────────────────────────────────────────────────
Builds a Respond.io-ready contact upload CSV.

Run any time after analytics.py has completed:
    python export_respondio_tags.py

Output: processed_data / "respondio_contact_tags_YYYY-MM-DD.csv"

Columns:
    contact_id      → Respond.io Contact ID (matches their system)
    name            → Best known name for the contact
    tags            → All unique session tags, comma-separated (one cell)
                      e.g. "Skincare, CeraVe, Converted, Gold, Paid Ads"
    
Also includes enrichment columns that are useful for Respond.io
contact attributes / segments (you can drop any you don't need):
    lifetime_tier         → Platinum / Gold / Silver
    lifetime_bracket      → Spend bucket (0-7k, 7k-12k, 12k+)
    acquisition_source    → Paid Ads / Organic / Inbound
    top_category          → Most-mentioned product category
    top_brand             → Most-mentioned brand
    total_sessions        → How many times they've contacted
    total_spend_kes       → Cumulative M-Pesa spend
    last_seen             → Date of most recent session
    customer_status       → New / Returning
─────────────────────────────────────────────────────────────────────
"""

import re
from datetime import date
from pathlib import Path

import pandas as pd

from Portal_ML_V4.src.config.settings import PROCESSED_DATA_DIR, BASE_DIR

# ── Paths ────────────────────────────────────────────────────────────
SESSIONS_PATH = PROCESSED_DATA_DIR / "fact_sessions_enriched.csv"
KB_PATH       = BASE_DIR / "data" / "01_raw" / "Final_Knowledge_Base_PowerBI.csv"
OUTPUT_DIR    = PROCESSED_DATA_DIR / "respondio_exports"

# ── Load valid brand names from KB ───────────────────────────────────
try:
    _kb = pd.read_csv(KB_PATH)
    _kb = _kb.dropna(subset=['Brand'])
    # Normalised lookup set (lowercase, no punctuation) → canonical casing
    VALID_BRANDS = {
        re.sub(r'[^a-z0-9\s]', '', str(b).lower().strip()): str(b).strip()
        for b in _kb['Brand'].unique()
    }
    print(f"   ✅ Loaded {len(VALID_BRANDS):,} brands from Knowledge Base.")
except Exception as e:
    print(f"   ⚠️  Could not load KB brands: {e}. All tags will be empty.")
    VALID_BRANDS = {}


def extract_brands_from_session(row) -> list[str]:
    """
    Pulls only KB-validated brand names for a single session row.
    Sources checked (in order):
      1. matched_brand  — the AI-matched brand from analytics.py
      2. final_tags     — the raw pipe-separated tag string (catches
                          brands detected by detect_brands() that didn't
                          get a full product match)
    """
    found = {}  # normalised_key → canonical name, preserves first-seen order

    def add(val):
        key = re.sub(r'[^a-z0-9\s]', '', str(val).lower().strip())
        if key in VALID_BRANDS and key not in found:
            found[key] = VALID_BRANDS[key]

    # 1. matched_brand column
    add(row.get('matched_brand', ''))

    # 2. Scan each pipe-separated tag against the brand list
    for part in str(row.get('final_tags', '')).split('|'):
        add(part.strip())

    return list(found.values())


def aggregate_brands(tag_series: pd.Series, brand_series: pd.Series) -> str:
    """
    Collects all brand mentions across every session for one contact,
    deduplicates, and returns a comma-separated string.
    """
    seen = {}
    for tags, brand in zip(tag_series, brand_series):
        # matched_brand column
        key = re.sub(r'[^a-z0-9\s]', '', str(brand).lower().strip())
        if key in VALID_BRANDS and key not in seen:
            seen[key] = VALID_BRANDS[key]
        # final_tags column
        for part in str(tags).split('|'):
            k = re.sub(r'[^a-z0-9\s]', '', part.lower().strip())
            if k in VALID_BRANDS and k not in seen:
                seen[k] = VALID_BRANDS[k]

    return ", ".join(seen.values()) if seen else None


# ── Main ─────────────────────────────────────────────────────────────
def build_respondio_export():
    print("📤 BUILDING RESPOND.IO CONTACT TAG EXPORT (Brands Only)...")

    if not SESSIONS_PATH.exists():
        print(f"❌ Sessions file not found: {SESSIONS_PATH}")
        print("   Run analytics.py first.")
        return

    df = pd.read_csv(SESSIONS_PATH, low_memory=False)
    print(f"   ✅ Loaded {len(df):,} sessions for {df['Contact ID'].nunique():,} contacts.")

    # ── 1. AGGREGATE BRAND TAGS per contact ──────────────────────────
    tag_agg = (
        df.groupby('Contact ID')
        .apply(lambda g: aggregate_brands(g['final_tags'], g['matched_brand']))
        .reset_index()
        .rename(columns={0: 'tags'})
    )

    # ── 2. BEST NAME per contact ──────────────────────────────────────
    def best_name(series):
        for v in series:
            s = str(v).strip()
            if s.lower() not in {'unknown', 'nan', 'none', ''}:
                return s
        return "Unknown"

    name_agg = (
        df.groupby('Contact ID')['contact_name']
        .apply(best_name)
        .reset_index()
        .rename(columns={'contact_name': 'name'})
    )

    # ── 3. JOIN & EXPORT ─────────────────────────────────────────────
    result = name_agg.merge(tag_agg, on='Contact ID', how='left')

    # Drop contacts where no brand was ever detected
    has_tags   = result['tags'].notna() & (result['tags'].str.strip() != '')
    no_tag_ct  = (~has_tags).sum()
    result     = result[has_tags].copy()

    result = result[['Contact ID', 'name', 'tags']]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filename    = f"respondio_contact_tags_{date.today().isoformat()}.csv"
    output_path = OUTPUT_DIR / filename
    result.to_csv(output_path, index=False)

    avg_brands = result['tags'].apply(lambda x: len(x.split(','))).mean()

    print(f"\n   📊 EXPORT SUMMARY")
    print(f"   {'─'*45}")
    print(f"   Contacts with ≥1 brand tag  : {len(result):,}")
    print(f"   Contacts with no brand match : {no_tag_ct:,}  (excluded)")
    print(f"   Avg brands per contact       : {avg_brands:.1f}")
    print(f"\n   💾 Saved → {output_path}")
    print(f"   ✅ DONE\n")


if __name__ == "__main__":
    build_respondio_export()