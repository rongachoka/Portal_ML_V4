"""
social_sales_direct.py
======================
Produces an accurate social media sales report by:
  1. Reading all_locations_sales_Jan25-Jan26.csv directly
  2. Filtering to rows where Ordered Via = respond.io / whatsapp
  3. Matching each POS description to the Knowledge Base for brand/product/category
  4. Outputting social_sales_direct.csv for Power BI

This bypasses the attribution waterfall entirely — no phone matching needed.
The Ordered Via field in the cashier report is the source of truth.
"""

import re
import os
import pandas as pd
import numpy as np
from pathlib import Path
from difflib import SequenceMatcher

try:
    from Portal_ML_V4.src.config.pos_aliases import TERM_ALIASES, BRAND_ALIASES
    print(f"   ✅ Loaded {len(TERM_ALIASES)} term aliases · {len(BRAND_ALIASES)} brand aliases")
except ImportError as e:
    print(f"   ⚠️ Could not load pos_aliases: {e} — using empty aliases")
    TERM_ALIASES  = {}
    BRAND_ALIASES = {}

try:
    from Portal_ML_V4.src.utils.phone import normalize_phone
except ImportError:
    def normalize_phone(val):
        if val is None: return None
        s = str(val).strip().replace('.0', '')
        s = ''.join(filter(str.isdigit, s))
        return s[-9:] if len(s) >= 9 else None

# ── CONFIG ────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(r"D:\Documents\Portal ML Analys\Portal_ML\Portal_ML_V4")
SALES_FILE   = BASE_DIR / "data" / "03_processed" / "pos_data" / "all_locations_sales_Jan25-Jan26.csv"
KB_PATH      = BASE_DIR / "data" / "01_raw" / "Final_Knowledge_Base_PowerBI.csv"
OUTPUT_FILE  = BASE_DIR / "data" / "03_processed" / "sales_attribution" / "social_sales_direct.csv"

SOCIAL_TAGS  = ['respond.io', 'respond', 'whatsapp', 'whats app']

PRICE_TOLERANCE = 0.20
MATCH_THRESHOLD = 0.50

STOP_WORDS = {
    'FOR', 'WITH', 'OF', 'TO', 'IN', 'ON', 'AT', 'ML', 'GM', 'PCS',
    'TUBE', 'BOTTLE', 'CAPS', 'TABS', 'SYR'
}

# ── DEBUG ────────────────────────────────────────────────────────────────────
def run_social_sales_direct():
    print("VERSION: No brand filter — all respond.io rows kept")

# ── LOAD KNOWLEDGE BASE ───────────────────────────────────────────────────────

def load_kb():
    if not KB_PATH.exists():
        raise FileNotFoundError(f"Knowledge Base not found: {KB_PATH}")

    df = pd.read_csv(KB_PATH).fillna("")
    df.columns = df.columns.str.strip()
    df["Brand"] = df["Brand"].str.strip()        # 🟢 strip leading/trailing spaces
    df["Name"]  = df["Name"].str.strip()         # 🟢 strip product names too

    brands = sorted(
        df["Brand"].astype(str).str.strip().str.title().unique().tolist(),
        key=len, reverse=True
    )

    kb_by_brand = {}
    for brand, grp in df.groupby("Brand"):
        kb_by_brand[str(brand).upper()] = grp.to_dict("records")

    print(f"   ✅ KB loaded: {len(df):,} products · {len(brands):,} brands")
    return df, brands, kb_by_brand


# ── TEXT HELPERS ──────────────────────────────────────────────────────────────

def expand_aliases(text: str) -> str:
    text = str(text).upper()
    for alias, full in TERM_ALIASES.items():
        text = re.sub(r'\b' + re.escape(alias) + r'\b', full, text)
    return text


def clean_for_match(text: str) -> str:
    clean = expand_aliases(str(text).upper())
    clean = re.sub(r'[^A-Z0-9\s]', ' ', clean)
    tokens = [w for w in clean.split() if w not in STOP_WORDS and len(w) > 2]
    return " ".join(tokens)


def fuzzy_score(a: str, b: str, brand: str = None) -> float:
    if not a or not b:
        return 0.0
    if brand:
        brand_clean = re.sub(r'[^A-Z0-9\s]', ' ', str(brand).upper()).strip()
        a = re.sub(r'\b' + re.escape(brand_clean) + r'\b', '', a).strip()
        a = re.sub(r'\s+', ' ', a).strip()

    seq  = SequenceMatcher(None, a, b).ratio()
    ta, tb = set(a.split()), set(b.split())
    size_tokens = {'8OZ','16OZ','250ML','500ML','100ML','200ML','30ML','50ML',
                   '150ML','400ML','1L','2L','300ML','10OZ'}
    ta -= size_tokens; tb -= size_tokens
    if not ta or not tb:
        return seq
    inter   = ta & tb
    jaccard  = len(inter) / len(ta | tb)
    coverage = len(inter) / min(len(ta), len(tb))
    token    = jaccard * 0.4 + coverage * 0.6
    return seq * 0.35 + token * 0.65


def detect_brand(desc: str, brands: list) -> list:
    safe = re.sub(r'[^A-Z0-9\s]', ' ', str(desc).upper())
    safe = re.sub(r'\s+', ' ', safe).strip()
    found = []
    for brand in brands:
        b = re.sub(r'[^A-Z0-9\s]', ' ', str(brand).upper())
        b = re.sub(r'\s+', ' ', b).strip()
        b_no_the = re.sub(r'^THE\s+', '', b)
        if re.search(r'\b' + re.escape(b) + r'\b', safe):
            found.append(brand)
        elif b_no_the != b and re.search(r'\b' + re.escape(b_no_the) + r'\b', safe):
            found.append(brand)
    return list(set(found))


# ── PRODUCT MATCHER ───────────────────────────────────────────────────────────

def match_product(pos_desc: str, unit_price: float, brands: list,
                  kb_by_brand: dict, kb_df: pd.DataFrame) -> dict:
    """Match a POS description to the KB. Returns dict with brand/product/category."""

    result = {
        "Matched_Brand":        "Unknown",
        "Matched_Product":      pos_desc,
        "Matched_Category":     "General",
        "Matched_Sub_Category": "General",
        "Matched_Concern":      "General",
        "cost_status":          "No Cost Data",
    }

    # Delivery override
    if "DELIVERY" in str(pos_desc).upper():
        result.update({"Matched_Brand": "Logistics", "Matched_Product": "Delivery Fee",
                        "Matched_Category": "Service"})
        return result

    detected = detect_brand(pos_desc, brands)
    if not detected:
        return result

    pos_clean  = clean_for_match(pos_desc)
    best_score = 0.0
    best_cand  = None
    best_brand = detected[0]

    for brand in detected:
        candidates = kb_by_brand.get(str(brand).upper(), [])
        for cand in candidates:
            kb_clean = clean_for_match(cand.get("Name", ""))
            score    = fuzzy_score(pos_clean, kb_clean, brand=brand)

            kb_price = float(cand.get("Price", 0) or 0)
            if kb_price > 0 and unit_price > 0:
                diff = abs(unit_price - kb_price) / kb_price
                if diff > PRICE_TOLERANCE:
                    score *= 0.85

            if score > best_score:
                best_score = score
                best_cand  = cand
                best_brand = brand

    if best_cand and best_score >= MATCH_THRESHOLD:
        result.update({
            "Matched_Brand":        best_cand.get("Brand", best_brand),
            "Matched_Product":      best_cand.get("Name",  pos_desc),
            "Matched_Category":     best_cand.get("Canonical_Category", "General"),
            "Matched_Sub_Category": best_cand.get("Sub_Category",       "General"),
            "Matched_Concern":      best_cand.get("Concerns",           "General"),
            "cost_status":          "Costed",
        })
    else:
        result["Matched_Brand"] = str(detected[0]).title()
        result["Matched_Product"] = f"General {result['Matched_Brand']} Product"

    return result


# ── MAIN ──────────────────────────────────────────────────────────────────────

def run_social_sales_direct():
    print("=" * 60)
    print("VERSION: No brand filter — all respond.io rows kept")
    print("=" * 60)

    # 1. Load sales file
    print(f"\n📥 Loading sales file...")
    df = pd.read_csv(SALES_FILE, low_memory=False, dtype={"Ordered Via": str})
    print(f"   Total rows: {len(df):,}")

    # 2. Filter to social media sales
    # mask = df["Ordered Via"].fillna("").str.lower().apply(
    #     lambda v: any(tag in v for tag in SOCIAL_TAGS)
    # )
    # 2. Filter to social media sales — exact match on Respond.io only
    mask = df["Ordered Via"].fillna("").str.lower().str.strip() == "respond.io"
    df_social = df[mask].copy()
    df_social = df[mask].copy()

    print(f"   Respond.io / WhatsApp rows: {len(df_social):,}")
    print(f"   Revenue (Total Tax Ex): KES {df_social['Total (Tax Ex)'].apply(pd.to_numeric, errors='coerce').sum():,.0f}")
    print(f"   Unique transactions:    {df_social['Transaction ID'].nunique():,}")

    if df_social.empty:
        print("\n❌ No social media rows found. Check 'Ordered Via' column values:")
        print(df["Ordered Via"].value_counts().head(10))
        return

    # 3. Load KB
    print(f"\n📚 Loading Knowledge Base...")
    kb_df, brands, kb_by_brand = load_kb()

    # 4. Match each UNIQUE description to KB (not each row — much faster)
    print(f"\n🔍 Matching unique descriptions to KB...")
    df_social["Total (Tax Ex)"] = pd.to_numeric(df_social["Total (Tax Ex)"], errors="coerce").fillna(0)
    df_social["Qty Sold"]       = pd.to_numeric(df_social["Qty Sold"],       errors="coerce").fillna(1)
    df_social["unit_price"]     = df_social["Total (Tax Ex)"] / df_social["Qty Sold"].replace(0, 1)

    # Clean phone numbers
    if "Phone Number" in df_social.columns:
        df_social["Phone Number"] = df_social["Phone Number"].apply(normalize_phone)

    # Get unique descriptions with their median unit price
    unique_descs = (
        df_social.groupby("Description")["unit_price"]
        .median()
        .reset_index()
    )
    print(f"   {len(unique_descs):,} unique descriptions (vs {len(df_social):,} rows) — {len(df_social)//max(len(unique_descs),1)}x speedup")

    # Match each unique description once
    desc_matches = {}
    for _, row in unique_descs.iterrows():
        desc = str(row["Description"])
        desc_matches[desc] = match_product(
            pos_desc    = desc,
            unit_price  = float(row["unit_price"]),
            brands      = brands,
            kb_by_brand = kb_by_brand,
            kb_df       = kb_df,
        )

    # Map results back column by column — avoids misalignment from pd.Series expansion
    MATCH_COLS = ["Matched_Brand", "Matched_Product", "Matched_Category",
                  "Matched_Sub_Category", "Matched_Concern", "cost_status"]
    empty_match = {c: "General" for c in MATCH_COLS}
    empty_match["Matched_Brand"]  = "Unknown"
    empty_match["cost_status"]    = "No Cost Data"

    df_social = df_social.reset_index(drop=True)
    for col in MATCH_COLS:
        df_social[col] = df_social["Description"].apply(
            lambda d: desc_matches.get(str(d), empty_match).get(col, empty_match[col])
        )
    df_out = df_social.copy()

    # 5. Keep ALL respond.io rows — label unmatched as "Other Social Sale"
    # We never drop respond.io sales — Ordered Via is the source of truth
    before = len(df_out)
    df_out["Matched_Brand"] = df_out["Matched_Brand"].apply(
        lambda b: b if b not in ["Unknown", "General", ""] else "Other Social Sale"
    )
    df_out["Matched_Category"] = df_out.apply(
        lambda r: r["Matched_Category"] if r["Matched_Brand"] != "Other Social Sale" else "Unmatched",
        axis=1
    )
    print(f"   Total rows kept: {len(df_out):,} (no rows dropped — all respond.io sales included)")
    matched = (df_out["Matched_Brand"] != "Other Social Sale").sum()
    print(f"   Matched to KB:   {matched:,} rows ({matched/len(df_out)*100:.1f}%)")
    print(f"   Unmatched:       {len(df_out)-matched:,} rows — labelled 'Other Social Sale'")

    # 6. Build clean output
    keep_cols = [
        "Sale_Date", "Location", "Transaction ID",
        "Description", "Qty Sold", "Total (Tax Ex)",
        "Ordered Via", "Client Name", "Phone Number", "Sales Rep",
        "Matched_Brand", "Matched_Product", "Matched_Category",
        "Matched_Sub_Category", "Matched_Concern", "cost_status",
    ]
    available = [c for c in keep_cols if c in df_out.columns]
    df_final = df_out[available].copy()

    # Current month summary
    from datetime import datetime
    now = datetime.now()
    month_start = pd.Timestamp(now.year, now.month, 1)

    df_final["Sale_Date"] = pd.to_datetime(df_final["Sale_Date"], errors="coerce")
    df_month = df_final[df_final["Sale_Date"] >= month_start]
    print(f"   Sale_Date range in file: {df_final['Sale_Date'].min()} to {df_final['Sale_Date'].max()}")
    print(f"   month_start = {month_start}")
    print(f"   Rows matching March: {len(df_month)}")

    month_rev  = df_month["Total (Tax Ex)"].sum()
    month_txns = df_month["Transaction ID"].nunique()
    month_label = now.strftime("%B %Y")
    total_rev  = df_final["Total (Tax Ex)"].sum()
    total_txns = df_final["Transaction ID"].nunique()

    print("\n📊 Summary:")
    print("   ── All Time ──────────────────────────")
    print(f"   Revenue:      KES {total_rev:,.0f}")
    print(f"   Transactions: {total_txns:,}")
    print(f"   Line items:   {len(df_final):,}")
    print(f"\n   ── {month_label} ──────────────────────────")
    print(f"   Revenue:      KES {month_rev:,.0f}")
    print(f"   Transactions: {month_txns:,}")
    print(f"   Line items:   {len(df_month):,}")

    print(f"\n   By brand (top 10) — {month_label}:")
    brand_rev = (
        df_month.groupby("Matched_Brand")["Total (Tax Ex)"]
        .sum().sort_values(ascending=False).head(10)
    )
    for brand, rev in brand_rev.items():
        print(f"      {brand:<30} KES {rev:>10,.0f}")

    # 7. Save
    os.makedirs(OUTPUT_FILE.parent, exist_ok=True)
    df_final.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    run_social_sales_direct()