"""
social_sales_direct.py
======================
Produces an accurate social media sales report by:
  1. Reading all_locations_sales_Jan25-Jan26.csv directly
  2. Filtering to rows where Ordered Via = respond.io
  3. Matching each POS description to the Knowledge Base for brand/product/category
  4. Outputting social_sales_direct.csv for Power BI

This bypasses the attribution waterfall entirely — no phone matching needed.
The Ordered Via field in the cashier report is the source of truth.
"""

import re
import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path
from difflib import SequenceMatcher
from Portal_ML_V4.src.config.settings import (
    BASE_DIR, MSG_HISTORY_RAW, CONTACTS_HISTORY_RAW
)
from Portal_ML_V4.src.config.ad_name_map import AD_NAME_MAP
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
KB_PATH      = BASE_DIR / "data" / "01_raw" / "Final_Knowledge_Base_PowerBI_New.csv"
OUTPUT_FILE  = BASE_DIR / "data" / "03_processed" / "sales_attribution" / "social_sales_direct.csv"
CONCERN_OUTPUT_FILE = OUTPUT_FILE.parent / "social_sales_by_concern.csv"

SOCIAL_TAGS  = ['respond.io', 'respond']

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
    df["Brand"] = df["Brand"].str.strip()
    df["Name"]  = df["Name"].str.strip()

    brands = sorted(
        df["Brand"].astype(str).str.strip().str.title().unique().tolist(),
        key=len, reverse=True
    )

    kb_by_brand = {}
    for brand, grp in df.groupby("Brand"):
        kb_by_brand[str(brand).upper()] = grp.to_dict("records")

    # ── Stage 1 lookup: Item Code Final → KB row ──────────────────────────────
    # Strip whitespace and upper-case for reliable exact matching.
    # Only include rows where the item code is actually populated.
    kb_by_item_code = {}
    if "Item Code Final" in df.columns:
        for _, row in df.iterrows():
            code = str(row["Item Code Final"]).strip().upper()
            if code and code not in ("", "NAN"):
                kb_by_item_code[code] = row.to_dict()
        print(f"   ✅ KB loaded: {len(df):,} products · {len(brands):,} brands · "
              f"{len(kb_by_item_code):,} item codes indexed")
    else:
        print(f"   ✅ KB loaded: {len(df):,} products · {len(brands):,} brands")
        print("   ⚠️  'Item Code Final' column not found — barcode stage will be skipped")

    return df, brands, kb_by_brand, kb_by_item_code


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
    desc_expanded = expand_aliases(str(desc))
    safe = re.sub(r'[^A-Z0-9\s]', ' ', desc_expanded.upper())
    safe = re.sub(r'\s+', ' ', safe).strip()
    safe_orig = re.sub(r'[^A-Z0-9\s]', ' ', str(desc).upper())
    safe_orig = re.sub(r'\s+', ' ', safe_orig).strip()
    found = []

    for brand in brands:
        b = re.sub(r'[^A-Z0-9\s]', ' ', str(brand).upper())
        b = re.sub(r'\s+', ' ', b).strip()
        b_no_the = re.sub(r'^THE\s+', '', b)

        # Standard word boundary match
        if re.search(r'\b' + re.escape(b) + r'\b', safe):
            found.append(brand)
        elif b_no_the != b and re.search(r'\b' + re.escape(b_no_the) + r'\b', safe):
            found.append(brand)
        else:
            # Multi-word brand fallback — match on significant words only
            # Catches "SOL DE JANEIRO" where "DE" is too short for boundaries
            words = [w for w in b.split() if len(w) >= 3]
            if len(words) >= 2 and len(words) < len(b.split()):
                pattern = r'\b' + r'\b.*?\b'.join(re.escape(w) for w in words) + r'\b'
                if re.search(pattern, safe):
                    found.append(brand)

    for alias, canonical in BRAND_ALIASES.items():
        alias_clean = re.sub(r'[^A-Z0-9\s]', ' ', str(alias).upper()).strip()
        if re.search(r'\b' + re.escape(alias_clean) + r'\b', safe_orig):
            found.append(str(canonical).title())

    return list(set(found))


def match_product(pos_desc: str, brands: list,
                  kb_by_brand: dict, kb_df: pd.DataFrame,
                  item_code: str = None,
                  kb_by_item_code: dict = None) -> dict:
    result = {
        "Matched_Brand":        "Unknown",
        "Matched_Product":      pos_desc,
        "Matched_Category":     "General",
        "Matched_Sub_Category": "General",
        "Matched_Concern":      "General",
        "cost_status":          "No Cost Data",
        "match_stage":          "Unmatched",
    }

    if "DELIVERY" in str(pos_desc).upper():
        result.update({"Matched_Brand": "Logistics", "Matched_Product": "Delivery Fee",
                        "Matched_Category": "Service", "match_stage": "Delivery"})
        return result
    
    # ── Prefix-based brand override ───────────────────────────────────────────
    if item_code:
        code_upper = str(item_code).strip().upper()
        if code_upper.startswith("PRE"):
            result.update({
                "Matched_Brand":    "Prescription",
                "Matched_Product":  pos_desc,
                "Matched_Category": "Medicine",
                "match_stage":      "Stage 0 - Prefix",
                "cost_status":      "Prefix Match",
            })
            return result
        if code_upper.startswith("ANT"):
            result.update({
                "Matched_Brand":    "Antibiotics",
                "Matched_Product":  pos_desc,
                "Matched_Category": "Medicine",
                "match_stage":      "Stage 0 - Prefix",
                "cost_status":      "Prefix Match",
            })
            return result

    # ── Stage 1: Exact barcode / item code join ────────────────────────────────
    if item_code and kb_by_item_code:
        code_key = str(item_code).strip().upper()
        if code_key and code_key not in ("", "NAN"):
            kb_row = kb_by_item_code.get(code_key)
            if kb_row is not None:
                result.update({
                    "Matched_Brand":        kb_row.get("Brand",              "Unknown"),
                    "Matched_Product":      kb_row.get("Name",               pos_desc),
                    "Matched_Category":     kb_row.get("Canonical_Category", "General"),
                    "Matched_Sub_Category": kb_row.get("Sub_Category",       "General"),
                    "Matched_Concern":      kb_row.get("Concerns",           "General"),
                    "cost_status":          "Barcode Match",
                    "match_stage":          "Stage 1 - Barcode",
                })
                return result

    # ── Stage 2 & 3: Brand detection → fuzzy name matching ────────────────────
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
            # kb_price = float(cand.get("Price", 0) or 0)
            # if kb_price > 0 and unit_price > 0:
            #     diff = abs(unit_price - kb_price) / kb_price
            #     if diff > PRICE_TOLERANCE:
            #         score *= 0.85
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
            "cost_status":          "Fuzzy Match",
            "match_stage":          "Stage 3 - Fuzzy",
        })
    else:
        # Brand detected but no KB product match — keep brand, use raw POS description as product name
        result["Matched_Brand"]    = str(best_brand).title()
        result["Matched_Product"]  = pos_desc  # raw description, not a synthetic "General X Sale"
        result["Matched_Category"] = "General"
        result["match_stage"]      = "Stage 2 - Brand Only"

    return result

# def detect_brand(desc: str, brands: list) -> list:
#     # Expand aliases FIRST so LRP → La Roche Posay before scanning
#     desc_expanded = expand_aliases(str(desc))
#     safe = re.sub(r'[^A-Z0-9\s]', ' ', desc_expanded.upper())
#     safe = re.sub(r'\s+', ' ', safe).strip()

#     # Also keep original for alias key matching
#     safe_orig = re.sub(r'[^A-Z0-9\s]', ' ', str(desc).upper())
#     safe_orig = re.sub(r'\s+', ' ', safe_orig).strip()

#     found = []

#     # 1. Check full KB brand names against expanded description
#     for brand in brands:
#         b = re.sub(r'[^A-Z0-9\s]', ' ', str(brand).upper())
#         b = re.sub(r'\s+', ' ', b).strip()
#         b_no_the = re.sub(r'^THE\s+', '', b)
#         if re.search(r'\b' + re.escape(b) + r'\b', safe):
#             found.append(brand)
#         elif b_no_the != b and re.search(r'\b' + re.escape(b_no_the) + r'\b', safe):
#             found.append(brand)

#     # 2. Check BRAND_ALIASES against ORIGINAL description
#     # e.g. "LRP" in original → "La Roche Posay"
#     for alias, canonical in BRAND_ALIASES.items():
#         alias_clean = re.sub(r'[^A-Z0-9\s]', ' ', str(alias).upper()).strip()
#         if re.search(r'\b' + re.escape(alias_clean) + r'\b', safe_orig):
#             found.append(str(canonical).title())

#     return list(set(found))


# # ── PRODUCT MATCHER ───────────────────────────────────────────────────────────

# def match_product(pos_desc: str, unit_price: float, brands: list,
#                   kb_by_brand: dict, kb_df: pd.DataFrame) -> dict:
#     """Match a POS description to the KB. Returns dict with brand/product/category."""

#     result = {
#         "Matched_Brand":        "Unknown",
#         "Matched_Product":      pos_desc,
#         "Matched_Category":     "General",
#         "Matched_Sub_Category": "General",
#         "Matched_Concern":      "General",
#         "cost_status":          "No Cost Data",
#     }

#     # Delivery override
#     if "DELIVERY" in str(pos_desc).upper():
#         result.update({"Matched_Brand": "Logistics", "Matched_Product": "Delivery Fee",
#                         "Matched_Category": "Service"})
#         return result

#     detected = detect_brand(pos_desc, brands)
#     if not detected:
#         return result

#     pos_clean  = clean_for_match(pos_desc)
#     best_score = 0.0
#     best_cand  = None
#     best_brand = detected[0]

#     for brand in detected:
#         candidates = kb_by_brand.get(str(brand).upper(), [])
#         for cand in candidates:
#             kb_clean = clean_for_match(cand.get("Name", ""))
#             score    = fuzzy_score(pos_clean, kb_clean, brand=brand)

#             kb_price = float(cand.get("Price", 0) or 0)
#             if kb_price > 0 and unit_price > 0:
#                 diff = abs(unit_price - kb_price) / kb_price
#                 if diff > PRICE_TOLERANCE:
#                     score *= 0.85

#             if score > best_score:
#                 best_score = score
#                 best_cand  = cand
#                 best_brand = brand

#     if best_cand and best_score >= MATCH_THRESHOLD:
#         result.update({
#             "Matched_Brand":        best_cand.get("Brand", best_brand),
#             "Matched_Product":      best_cand.get("Name",  pos_desc),
#             "Matched_Category":     best_cand.get("Canonical_Category", "General"),
#             "Matched_Sub_Category": best_cand.get("Sub_Category",       "General"),
#             "Matched_Concern":      best_cand.get("Concerns",           "General"),
#             "cost_status":          "Costed",
#         })
#     else:
#         result["Matched_Brand"] = str(detected[0]).title()
#         result["Matched_Product"] = f"General {result['Matched_Brand']} Product"

#     return result


def load_ads_for_pos() -> pd.DataFrame:
    """
    Loads the ads folder and joins to the contacts file to produce a
    norm_phone → ad info lookup table.

    Chain: ads file (Contact ID) → contacts file (Contact ID → PhoneNumber) → norm_phone
    Returns DataFrame with: norm_phone, Timestamp, Ad ID, Ad campaign ID, Ad Name
    """
    ADS_DIR = Path(MSG_HISTORY_RAW).parent / "ads"
    all_files = (
        glob.glob(str(ADS_DIR / "contacts-added*.csv")) +
        glob.glob(str(ADS_DIR / "contacts-connected*.csv"))
    )

    if not all_files:
        print("   ⚠️ No ad files found — acquisition_source will be 'Organic / Direct' for all rows.")
        return pd.DataFrame()

    # 1. Load and normalise ad files
    dfs = []
    for f in all_files:
        try:
            temp = pd.read_csv(f, dtype=str, keep_default_na=False)
            if 'Timestamp' not in temp.columns or 'Contact ID' not in temp.columns:
                continue
            for col in ['Ad ID', 'Ad campaign ID', 'Ad group ID']:
                if col in temp.columns:
                    temp[col] = temp[col].replace(['-', ' -', ''], pd.NA)
            s = temp['Timestamp'].astype(str).str.strip().str.replace('T', ' ', regex=False)
            temp['Timestamp'] = pd.to_datetime(s, errors='coerce', format='mixed', cache=True)
            cid = temp['Contact ID'].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
            temp['Contact ID'] = pd.to_numeric(cid, errors='coerce').astype('Int64')
            temp = temp.dropna(subset=['Contact ID', 'Timestamp'])
            keep = [c for c in ['Timestamp', 'Contact ID', 'Ad ID', 'Ad campaign ID', 'Ad group ID'] if c in temp.columns]
            dfs.append(temp[keep])
        except Exception as e:
            print(f"   ⚠️ Error reading {Path(f).name}: {e}")

    if not dfs:
        return pd.DataFrame()

    df_ads = pd.concat(dfs, ignore_index=True).drop_duplicates().sort_values('Timestamp')

    # 2. Load contacts to bridge Contact ID → phone
    try:
        df_contacts = pd.read_csv(
            CONTACTS_HISTORY_RAW, dtype=str,
            usecols=['ContactID', 'PhoneNumber']
        )
        df_contacts['ContactID'] = pd.to_numeric(
            df_contacts['ContactID'].str.strip().str.replace(r'\.0$', '', regex=True),
            errors='coerce'
        ).astype('Int64')
        df_contacts['norm_phone'] = df_contacts['PhoneNumber'].apply(normalize_phone)
        df_contacts = df_contacts.dropna(subset=['ContactID', 'norm_phone'])
    except Exception as e:
        print(f"   ⚠️ Could not load contacts for phone mapping: {e}")
        return pd.DataFrame()

    # 3. Join: ads → contacts → norm_phone
    df_lookup = pd.merge(
        df_ads,
        df_contacts[['ContactID', 'norm_phone']],
        left_on='Contact ID',
        right_on='ContactID',
        how='inner'
    ).drop(columns=['ContactID', 'Contact ID'])

    # 4. Map Ad Name
    df_lookup['clean_ad_id'] = (
        df_lookup['Ad ID'].astype(str).str.replace(r'\.0$', '', regex=True)
    )
    df_lookup['Ad Name'] = df_lookup['clean_ad_id'].map(AD_NAME_MAP)

    print(f"   ✅ Ad lookup built: {df_lookup['norm_phone'].nunique():,} unique phones with ad history")
    return df_lookup


def enrich_with_acquisition_source(df: pd.DataFrame, df_ads: pd.DataFrame,
                                    lookback_days: int = 7) -> pd.DataFrame:
    """
    Matches each unique POS transaction to an ad click via phone number.
    An ad click must have occurred within `lookback_days` before the sale date.

    Adds columns: acquisition_source, Ad Name, Ad campaign ID, Ad ID
    """
    # Safe defaults — always add the columns even if matching fails
    target_ad_cols = ['Ad campaign ID', 'Ad ID', 'Ad Name', 'Campaign Name']    

    for col in ['acquisition_source'] + target_ad_cols:
        if col not in df.columns:
            df[col] = None
    df['acquisition_source'] = 'Organic / Direct'

    if df_ads.empty:
        return df

    # Work at transaction level — one row per unique (Transaction ID, phone, date)
    # avoids merge_asof issues with many line items sharing the same timestamp
    txn_keys = (
        df[['Transaction ID', 'norm_phone', 'Sale_Date']]
        .drop_duplicates('Transaction ID')
        .copy()
    )
    txn_keys['Sale_Date'] = pd.to_datetime(txn_keys['Sale_Date'], errors='coerce')
    txn_keys = txn_keys.dropna(subset=['norm_phone', 'Sale_Date']).sort_values('Sale_Date')

    df_ads_sorted = df_ads.sort_values('Timestamp')

    # merge_asof: for each transaction, find the most recent ad click before
    # the sale within the lookback window, matched on phone number
    matched = pd.merge_asof(
        txn_keys,
        df_ads_sorted[['norm_phone', 'Timestamp', 'Ad ID', 'Ad campaign ID', 'Ad Name']],
        left_on='Sale_Date',
        right_on='Timestamp',
        by='norm_phone',
        tolerance=pd.Timedelta(days=lookback_days),
        direction='backward'
    )

    # Build Transaction ID maps for each ad column
    matched = matched.set_index('Transaction ID')
    for col in ['Ad campaign ID', 'Ad ID', 'Ad Name']:
        col_map = matched[col].dropna().to_dict()
        df[col] = df['Transaction ID'].map(col_map).where(
            df['Transaction ID'].map(col_map).notna(), df[col]
        )

    df['acquisition_source'] = df['Ad campaign ID'].notna().map({
        True: 'Paid Ads', False: 'Organic / Direct'
    })

    return df


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
    df_social["Sale_Date"] = pd.to_datetime(df_social["Sale_Date"], errors="coerce")
    df_social = df_social[df_social["Sale_Date"] >= pd.Timestamp("2026-01-01")].copy() 

    print(f"   Respond.io / WhatsApp rows: {len(df_social):,}")
    print(f"   Revenue (Total Tax Ex): KES {df_social['Total (Tax Ex)'].apply(pd.to_numeric, errors='coerce').sum():,.0f}")
    print(f"   Unique transactions:    {df_social['Transaction ID'].nunique():,}")

    if df_social.empty:
        print("\n❌ No social media rows found. Check 'Ordered Via' column values:")
        print(df["Ordered Via"].value_counts().head(10))
        return

    # 3. Load KB
    print(f"\n📚 Loading Knowledge Base...")
    kb_df, brands, kb_by_brand, kb_by_item_code = load_kb()

    # 4. Match each UNIQUE (Item, Description) pair to KB
    # Item code is tried first (Stage 1 barcode), description text as fallback.
    print(f"\n🔍 Matching unique items to KB...")
    df_social["Total (Tax Ex)"] = pd.to_numeric(df_social["Total (Tax Ex)"], errors="coerce").fillna(0)
    df_social["Qty Sold"]       = pd.to_numeric(df_social["Qty Sold"],       errors="coerce").fillna(1)

    # Clean phone numbers
    if "Phone Number" in df_social.columns:
        df_social["Phone Number"] = df_social["Phone Number"].apply(normalize_phone)

    # Deduplicate on (Item, Description) — item code is the primary key,
    # description is the fallback for rows where Item is null.
    df_social["_item_key"] = df_social["Item"].astype(str).str.strip().str.upper()
    unique_pairs = (
        df_social[["_item_key", "Description"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    print(f"   {len(unique_pairs):,} unique (Item, Description) pairs "
          f"(vs {len(df_social):,} rows) — "
          f"{len(df_social)//max(len(unique_pairs),1)}x speedup")

    # Match each unique pair once
    pair_matches = {}
    for _, row in unique_pairs.iterrows():
        item_key = row["_item_key"]
        desc     = str(row["Description"])
        # Skip null item codes — they'll fall straight to brand/fuzzy stages
        item_code = item_key if item_key not in ("", "NAN", "NONE") else None
        pair_matches[(item_key, desc)] = match_product(
            pos_desc        = desc,
            brands          = brands,
            kb_by_brand     = kb_by_brand,
            kb_df           = kb_df,
            item_code       = item_code,
            kb_by_item_code = kb_by_item_code,
        )

    # Map results back column by column
    MATCH_COLS = ["Matched_Brand", "Matched_Product", "Matched_Category",
                  "Matched_Sub_Category", "Matched_Concern", "cost_status", "match_stage"]
    empty_match = {c: "General" for c in MATCH_COLS}
    empty_match["Matched_Brand"]  = "Unknown"
    empty_match["cost_status"]    = "No Cost Data"
    empty_match["match_stage"]    = "Unmatched"

    df_social = df_social.reset_index(drop=True)
    for col in MATCH_COLS:
        df_social[col] = df_social.apply(
            lambda r: pair_matches.get(
                (r["_item_key"], str(r["Description"])), empty_match
            ).get(col, empty_match[col]),
            axis=1
        )
    df_social.drop(columns=["_item_key"], inplace=True)
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
    print(f"\n   Match stage breakdown:")
    for stage, count in df_out["match_stage"].value_counts().items():
        print(f"      {stage:<30} {count:>6,} rows ({count/len(df_out)*100:.1f}%)")

    # 5b. Acquisition Source Enrichment
    print("\n🎯 Enriching with Acquisition Source...")
    df_ads_lookup = load_ads_for_pos()
    
    # norm_phone column needed for ad matching — derive it from the already-cleaned Phone Number
    df_out['norm_phone'] = df_out['Phone Number'].apply(normalize_phone)  # ← add this
    
    df_out = enrich_with_acquisition_source(df_out, df_ads_lookup)
    paid    = (df_out['acquisition_source'] == 'Paid Ads').sum()
    organic = (df_out['acquisition_source'] == 'Organic / Direct').sum()
    print(f"   Paid Ads:        {paid:,} rows")
    print(f"   Organic/Direct:  {organic:,} rows")

    # 6. Build clean output
    # keep_cols = [
    #     "Sale_Date", "Location", "Transaction ID",
    #     "Description", "Qty Sold", "Total (Tax Ex)",
    #     "Ordered Via", "Client Name", "Phone Number", "Sales Rep",
    #     "Matched_Brand", "Matched_Product", "Matched_Category",
    #     "Matched_Sub_Category", "Matched_Concern", "cost_status",
    # ]
    keep_cols = [
        "Sale_Date", "Location", "Transaction ID", 
        "Description", "Qty Sold", "Total (Tax Ex)",
        "Ordered Via", "Respond Customer ID", "Client Name", 
        "Phone Number", "Sales Rep", "acquisition_source", 
        "Ad Name", "Ad campaign ID", "Ad ID",
        "Matched_Brand", "Matched_Product", "Matched_Category",
        "Matched_Sub_Category", "Matched_Concern", "cost_status", 
        "match_stage", "Item"
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


    # New Customers between 28th and 29th
    QUERY_START = pd.Timestamp("2026-03-28")
    QUERY_END   = pd.Timestamp("2026-03-29")

    df_day = df_final[(df_final["Sale_Date"] >= QUERY_START) & (df_final["Sale_Date"] < QUERY_END)]
    day_txns = df_day["Transaction ID"].nunique()
    day_rev  = df_day["Total (Tax Ex)"].sum()

    print("\n📊Daily Summary:")
    print("   ── Daily ──────────────────────────")
    print(f"   Revenue:      KES {day_rev:,.0f}")
    print(f"   Transactions: {day_txns:,}")
    print(f"   Line items:   {len(df_day):,}")


    # 7. Save
    os.makedirs(OUTPUT_FILE.parent, exist_ok=True)
    df_final.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ Saved to: {OUTPUT_FILE}")



    # -------------------------------------------------------------------------
    # Concerns Exploded Report
    # -------------------------------------------------------------------------

    df_concern = df_final.copy()

    # Split comma-separated concerns into lists, strip whitespace
    df_concern['Matched_Concern'] = df_concern['Matched_Concern'].fillna('General').str.split(',')

    # Explode — one row per concern
    df_concern = df_concern.explode('Matched_Concern')
    df_concern['Matched_Concern'] = df_concern['Matched_Concern'].str.strip()

    # Drop rows where concern is blank or General (optional — remove if boss wants to keep them)
    df_concern = df_concern[df_concern['Matched_Concern'].notna()]
    df_concern = df_concern[df_concern['Matched_Concern'] != '']

    df_concern.to_csv(CONCERN_OUTPUT_FILE, index=False)
    print(f"\n✅ Concern-level export saved: {len(df_concern):,} rows → {CONCERN_OUTPUT_FILE}")
    print(f"   Unique concerns: {df_concern['Matched_Concern'].nunique():,}")
    print(f"   Top concerns:\n{df_concern['Matched_Concern'].value_counts().head(10).to_string()}")


if __name__ == "__main__":
    run_social_sales_direct()