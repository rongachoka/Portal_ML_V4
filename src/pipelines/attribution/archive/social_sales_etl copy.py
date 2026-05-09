"""
social_sales_etl.py
===================
Social media sales report — reads from etl_local's output file.

Pipeline dependency:
    etl_local.py → all_locations_sales_NEW.csv → this script → social_sales_direct.csv

Matching waterfall per line item:
    Stage 0  — Prefix override  (PRE* → Prescription, ANT* → Antibiotics)
    Stage 1  — Exact barcode    (Item column → KB Item Code Final)
    Stage 2  — Brand Only       (brand detected, fuzzy below threshold)
    Stage 3  — Fuzzy            (brand + fuzzy name similarity above threshold)
    Unmatched — no brand found

REVENUE NOTE:
    Revenue = Total (Tax Ex) from sales line items — not cashier Amount.
    Cashier-only rows (no matching sales row) are KEPT in the output with null
    Description / zero revenue and match_stage = "Gap - No Sales Row" so the gap
    is visible in Power BI rather than silently dropped.

Run:
    python -m Portal_ML_V4.src.pipelines.social_sales_etl
    — or —
    python social_sales_etl.py
"""

import glob
import os
import re
import warnings
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ══════════════════════════════════════════════════════════════════════════════
# IMPORTS — graceful fallbacks so the file runs standalone too
# ══════════════════════════════════════════════════════════════════════════════
try:
    from Portal_ML_V4.src.config.settings import (
        BASE_DIR, PROCESSED_DATA_DIR,
        MSG_HISTORY_RAW, CONTACTS_HISTORY_RAW,
    )
except ImportError:
    BASE_DIR              = Path(r"D:\Documents\Portal ML Analys\Portal_ML\Portal_ML_V4")
    PROCESSED_DATA_DIR    = BASE_DIR / "data" / "03_processed"
    MSG_HISTORY_RAW       = BASE_DIR / "data" / "01_raw" / "respond_io" / "messages.csv"
    CONTACTS_HISTORY_RAW  = BASE_DIR / "data" / "01_raw" / "respond_io" / "contacts.csv"

try:
    from Portal_ML_V4.src.config.pos_aliases import TERM_ALIASES, BRAND_ALIASES
    print(f"   ✅ Loaded {len(TERM_ALIASES)} term aliases · {len(BRAND_ALIASES)} brand aliases")
except ImportError as e:
    print(f"   ⚠ Could not load pos_aliases ({e}) — using empty aliases")
    TERM_ALIASES  = {}
    BRAND_ALIASES = {}

try:
    from Portal_ML_V4.src.utils.phone import normalize_phone
except ImportError:
    def normalize_phone(val):
        if val is None:
            return None
        s = str(val).strip().replace(".0", "")
        s = "".join(filter(str.isdigit, s))
        return s[-9:] if len(s) >= 9 else None

try:
    from Portal_ML_V4.src.config.ad_name_map import AD_NAME_MAP
except ImportError:
    AD_NAME_MAP = {}


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

SALES_FILE  = PROCESSED_DATA_DIR / "pos_data" / "all_locations_sales_NEW.csv"
KB_PATH     = BASE_DIR / "data" / "01_raw" / "Final_Knowledge_Base_PowerBI_New.csv"
OUTPUT_DIR  = PROCESSED_DATA_DIR / "sales_attribution"
OUTPUT_FILE         = OUTPUT_DIR / "social_sales_direct.csv"
CONCERN_OUTPUT_FILE = OUTPUT_DIR / "social_sales_by_concern.csv"

SOCIAL_TAG      = "respond.io"
MATCH_THRESHOLD = 0.50

STOP_WORDS = {
    "FOR", "WITH", "OF", "TO", "IN", "ON", "AT", "ML", "GM", "PCS",
    "TUBE", "BOTTLE", "CAPS", "TABS", "SYR",
}

# Output column order — exact match to social_sales_direct.py + Respond Customer ID
KEEP_COLS = [
    "Sale_Date", "Location", "Transaction ID",
    "Description", "Qty Sold", "Total (Tax Ex)",
    "Ordered Via", "Respond Customer ID", "Client Name",
    "Phone Number", "Sales Rep", "acquisition_source",
    "Ad Name", "Ad campaign ID", "Ad ID",
    "Matched_Brand", "Matched_Product", "Matched_Category",
    "Matched_Sub_Category", "Matched_Concern", "cost_status",
    "match_stage", "Item",
]

MATCH_COLS = [
    "Matched_Brand", "Matched_Product", "Matched_Category",
    "Matched_Sub_Category", "Matched_Concern", "cost_status", "match_stage",
]
_EMPTY_MATCH = {
    "Matched_Brand":        "Unknown",
    "Matched_Product":      "",
    "Matched_Category":     "General",
    "Matched_Sub_Category": "General",
    "Matched_Concern":      "General",
    "cost_status":          "No Cost Data",
    "match_stage":          "Unmatched",
}
_GAP_MATCH = {
    "Matched_Brand":        "Other Social Sale",
    "Matched_Product":      "",
    "Matched_Category":     "Unmatched",
    "Matched_Sub_Category": "General",
    "Matched_Concern":      "General",
    "cost_status":          "No Cost Data",
    "match_stage":          "Gap - No Sales Row",
}


# ══════════════════════════════════════════════════════════════════════════════
# KNOWLEDGE BASE
# ══════════════════════════════════════════════════════════════════════════════

def load_kb():
    """
    Returns (kb_df, brands, kb_by_brand, kb_by_item_code).
    kb_by_item_code maps Item Code Final → KB row dict for exact barcode joins.
    """
    if not KB_PATH.exists():
        raise FileNotFoundError(f"Knowledge Base not found: {KB_PATH}")

    df = pd.read_csv(KB_PATH).fillna("")
    df.columns = df.columns.str.strip()
    df["Brand"] = df["Brand"].str.strip()
    df["Name"]  = df["Name"].str.strip()

    brands = sorted(
        df["Brand"].astype(str).str.strip().str.title().unique().tolist(),
        key=len, reverse=True,
    )

    kb_by_brand = {}
    for brand, grp in df.groupby("Brand"):
        kb_by_brand[str(brand).upper()] = grp.to_dict("records")

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
        print("   ⚠  'Item Code Final' column not found — barcode stage will be skipped")

    return df, brands, kb_by_brand, kb_by_item_code


# ══════════════════════════════════════════════════════════════════════════════
# TEXT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def expand_aliases(text: str) -> str:
    text = str(text).upper()
    for alias, full in TERM_ALIASES.items():
        text = re.sub(r"\b" + re.escape(alias) + r"\b", full, text)
    return text


def clean_for_match(text: str) -> str:
    clean = expand_aliases(str(text).upper())
    clean = re.sub(r"[^A-Z0-9\s]", " ", clean)
    tokens = [w for w in clean.split() if w not in STOP_WORDS and len(w) > 2]
    return " ".join(tokens)


def fuzzy_score(a: str, b: str, brand: str = None) -> float:
    if not a or not b:
        return 0.0
    if brand:
        brand_clean = re.sub(r"[^A-Z0-9\s]", " ", str(brand).upper()).strip()
        a = re.sub(r"\b" + re.escape(brand_clean) + r"\b", "", a).strip()
        a = re.sub(r"\s+", " ", a).strip()

    seq = SequenceMatcher(None, a, b).ratio()
    ta, tb = set(a.split()), set(b.split())
    size_tokens = {
        "8OZ", "16OZ", "250ML", "500ML", "100ML", "200ML", "30ML",
        "50ML", "150ML", "400ML", "1L", "2L", "300ML", "10OZ",
    }
    ta -= size_tokens
    tb -= size_tokens

    if not ta or not tb:
        return seq

    inter    = ta & tb
    jaccard  = len(inter) / len(ta | tb)
    coverage = len(inter) / min(len(ta), len(tb))
    token    = jaccard * 0.4 + coverage * 0.6
    return seq * 0.35 + token * 0.65


def detect_brand(desc: str, brands: list) -> list:
    desc_expanded = expand_aliases(str(desc))
    safe = re.sub(r"[^A-Z0-9\s]", " ", desc_expanded.upper())
    safe = re.sub(r"\s+", " ", safe).strip()

    safe_orig = re.sub(r"[^A-Z0-9\s]", " ", str(desc).upper())
    safe_orig = re.sub(r"\s+", " ", safe_orig).strip()

    found = []
    for brand in brands:
        b        = re.sub(r"[^A-Z0-9\s]", " ", str(brand).upper())
        b        = re.sub(r"\s+", " ", b).strip()
        b_no_the = re.sub(r"^THE\s+", "", b)

        if re.search(r"\b" + re.escape(b) + r"\b", safe):
            found.append(brand)
        elif b_no_the != b and re.search(r"\b" + re.escape(b_no_the) + r"\b", safe):
            found.append(brand)
        else:
            words = [w for w in b.split() if len(w) >= 3]
            if len(words) >= 2 and len(words) < len(b.split()):
                pattern = (
                    r"\b"
                    + r"\b.*?\b".join(re.escape(w) for w in words)
                    + r"\b"
                )
                if re.search(pattern, safe):
                    found.append(brand)

    for alias, canonical in BRAND_ALIASES.items():
        alias_clean = re.sub(r"[^A-Z0-9\s]", " ", str(alias).upper()).strip()
        if re.search(r"\b" + re.escape(alias_clean) + r"\b", safe_orig):
            found.append(str(canonical).title())

    return list(set(found))


# ══════════════════════════════════════════════════════════════════════════════
# KB MATCHING — 4-stage waterfall
# ══════════════════════════════════════════════════════════════════════════════

def match_product(
    pos_desc: str,
    brands: list,
    kb_by_brand: dict,
    kb_df: pd.DataFrame,
    item_code: str = None,
    kb_by_item_code: dict = None,
) -> dict:
    """
    Stage 0  — Prefix override  (PRE* → Prescription, ANT* → Antibiotics)
    Stage 1  — Exact barcode    (Item column → KB Item Code Final)
    Stage 2  — Brand Only       (brand detected, fuzzy below threshold)
    Stage 3  — Fuzzy            (brand + fuzzy name similarity above threshold)
    Unmatched — no brand found
    """
    result = dict(_EMPTY_MATCH)
    result["Matched_Product"] = pos_desc

    if "DELIVERY" in str(pos_desc).upper():
        result.update({
            "Matched_Brand":    "Logistics",
            "Matched_Product":  "Delivery Fee",
            "Matched_Category": "Service",
            "cost_status":      "Delivery",
            "match_stage":      "Delivery",
        })
        return result

    # ── Stage 0: Prefix-based medicine override ────────────────────────────
    if item_code:
        code_upper = str(item_code).strip().upper()
        if code_upper.startswith("PRE"):
            result.update({
                "Matched_Brand":    "Prescription",
                "Matched_Product":  pos_desc,
                "Matched_Category": "Medicine",
                "cost_status":      "Prefix Match",
                "match_stage":      "Stage 0 - Prefix",
            })
            return result
        if code_upper.startswith("ANT"):
            result.update({
                "Matched_Brand":    "Antibiotics",
                "Matched_Product":  pos_desc,
                "Matched_Category": "Medicine",
                "cost_status":      "Prefix Match",
                "match_stage":      "Stage 0 - Prefix",
            })
            return result

    # ── Stage 1: Exact barcode / item code join ────────────────────────────
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

    # ── Stage 2 & 3: Brand detection → fuzzy name matching ────────────────
    detected = detect_brand(pos_desc, brands)
    if not detected:
        return result   # Unmatched

    pos_clean  = clean_for_match(pos_desc)
    best_score = 0.0
    best_cand  = None
    best_brand = detected[0]

    for brand in detected:
        for cand in kb_by_brand.get(str(brand).upper(), []):
            score = fuzzy_score(pos_clean, clean_for_match(cand.get("Name", "")), brand=brand)
            if score > best_score:
                best_score = score
                best_cand  = cand
                best_brand = brand

    if best_cand and best_score >= MATCH_THRESHOLD:
        result.update({
            "Matched_Brand":        best_cand.get("Brand",              best_brand),
            "Matched_Product":      best_cand.get("Name",               pos_desc),
            "Matched_Category":     best_cand.get("Canonical_Category", "General"),
            "Matched_Sub_Category": best_cand.get("Sub_Category",       "General"),
            "Matched_Concern":      best_cand.get("Concerns",           "General"),
            "cost_status":          "Fuzzy Match",
            "match_stage":          "Stage 3 - Fuzzy",
        })
    else:
        result["Matched_Brand"]    = str(best_brand).title()
        result["Matched_Product"]  = pos_desc   # raw desc, not synthetic "General X Sale"
        result["Matched_Category"] = "General"
        result["match_stage"]      = "Stage 2 - Brand Only"

    return result


# ══════════════════════════════════════════════════════════════════════════════
# AD ATTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════

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
            if "Timestamp" not in temp.columns or "Contact ID" not in temp.columns:
                continue
            for col in ["Ad ID", "Ad campaign ID", "Ad group ID"]:
                if col in temp.columns:
                    temp[col] = temp[col].replace(["-", " -", ""], pd.NA)
            s = temp["Timestamp"].astype(str).str.strip().str.replace("T", " ", regex=False)
            temp["Timestamp"] = pd.to_datetime(s, errors="coerce", format="mixed", cache=True)
            cid = temp["Contact ID"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
            temp["Contact ID"] = pd.to_numeric(cid, errors="coerce").astype("Int64")
            temp = temp.dropna(subset=["Contact ID", "Timestamp"])
            keep = [
                c for c in ["Timestamp", "Contact ID", "Ad ID", "Ad campaign ID", "Ad group ID"]
                if c in temp.columns
            ]
            dfs.append(temp[keep])
        except Exception as e:
            print(f"   ⚠️ Error reading {Path(f).name}: {e}")

    if not dfs:
        return pd.DataFrame()

    df_ads = (
        pd.concat(dfs, ignore_index=True)
        .drop_duplicates()
        .sort_values("Timestamp")
    )

    # 2. Load contacts to bridge Contact ID → phone
    try:
        df_contacts = pd.read_csv(
            CONTACTS_HISTORY_RAW, dtype=str,
            usecols=["ContactID", "PhoneNumber"],
        )
        df_contacts["ContactID"] = pd.to_numeric(
            df_contacts["ContactID"].str.strip().str.replace(r"\.0$", "", regex=True),
            errors="coerce",
        ).astype("Int64")
        df_contacts["norm_phone"] = df_contacts["PhoneNumber"].apply(normalize_phone)
        df_contacts = df_contacts.dropna(subset=["ContactID", "norm_phone"])
    except Exception as e:
        print(f"   ⚠️ Could not load contacts for phone mapping: {e}")
        return pd.DataFrame()

    # 3. Join: ads → contacts → norm_phone
    df_lookup = pd.merge(
        df_ads,
        df_contacts[["ContactID", "norm_phone"]],
        left_on="Contact ID",
        right_on="ContactID",
        how="inner",
    ).drop(columns=["ContactID", "Contact ID"])

    # 4. Map Ad Name
    df_lookup["clean_ad_id"] = (
        df_lookup["Ad ID"].astype(str).str.replace(r"\.0$", "", regex=True)
    )
    df_lookup["Ad Name"] = df_lookup["clean_ad_id"].map(AD_NAME_MAP)

    print(
        f"   ✅ Ad lookup built: {df_lookup['norm_phone'].nunique():,} unique phones "
        f"with ad history"
    )
    return df_lookup


def enrich_with_acquisition_source(
    df: pd.DataFrame,
    df_ads: pd.DataFrame,
    lookback_days: int = 7,
) -> pd.DataFrame:
    """
    Matches each unique POS transaction to an ad click via phone number.
    An ad click must have occurred within `lookback_days` before the sale date.
    Adds columns: acquisition_source, Ad Name, Ad campaign ID, Ad ID
    """
    # Always add columns even if matching fails
    target_ad_cols = ["Ad campaign ID", "Ad ID", "Ad Name", "Campaign Name"]
    for col in ["acquisition_source"] + target_ad_cols:
        if col not in df.columns:
            df[col] = None
    df["acquisition_source"] = "Organic / Direct"

    if df_ads.empty:
        return df

    # Work at transaction level to avoid merge_asof issues with many line items
    txn_keys = (
        df[["Transaction ID", "norm_phone", "Sale_Date"]]
        .drop_duplicates("Transaction ID")
        .copy()
    )
    txn_keys["Sale_Date"] = pd.to_datetime(txn_keys["Sale_Date"], errors="coerce")
    txn_keys = txn_keys.dropna(subset=["norm_phone", "Sale_Date"]).sort_values("Sale_Date")

    df_ads_sorted = df_ads.sort_values("Timestamp")

    # For each transaction, find the most recent ad click before the sale within lookback
    matched = pd.merge_asof(
        txn_keys,
        df_ads_sorted[["norm_phone", "Timestamp", "Ad ID", "Ad campaign ID", "Ad Name"]],
        left_on="Sale_Date",
        right_on="Timestamp",
        by="norm_phone",
        tolerance=pd.Timedelta(days=lookback_days),
        direction="backward",
    )

    matched = matched.set_index("Transaction ID")
    for col in ["Ad campaign ID", "Ad ID", "Ad Name"]:
        col_map = matched[col].dropna().to_dict()
        df[col] = df["Transaction ID"].map(col_map).where(
            df["Transaction ID"].map(col_map).notna(), df[col]
        )

    df["acquisition_source"] = df["Ad campaign ID"].notna().map(
        {True: "Paid Ads", False: "Organic / Direct"}
    )
    return df


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run_social_sales_direct():
    """
    Entry point — function name kept identical to social_sales_direct.py so
    any pipeline runner calling run_social_sales_direct() needs no changes.
    """
    print("=" * 65)
    print("  SOCIAL SALES ETL  (respond.io · reads from etl_local output)")
    print(f"  Input  : {SALES_FILE.name}")
    print(f"  Output : {OUTPUT_FILE.name}")
    print("=" * 65)

    if not SALES_FILE.exists():
        print(f"\n❌  Input file not found: {SALES_FILE}")
        print("    Run etl_local.py first to generate all_locations_sales_NEW.csv")
        return

    # ── 1. Load ETL output ─────────────────────────────────────────────────
    print(f"\n📥 Loading {SALES_FILE.name}...")
    df = pd.read_csv(SALES_FILE, low_memory=False, dtype={"Ordered Via": str})
    print(f"   Total rows: {len(df):,}")

    # ── 2. Filter to respond.io ────────────────────────────────────────────
    mask = (
        df["Ordered Via"]
        .fillna("").astype(str).str.strip().str.lower() == SOCIAL_TAG
    )
    df_social = df[mask].copy()
    total_txns_cashier = df_social["Transaction ID"].nunique()

    print(f"   respond.io rows  : {len(df_social):,}")
    print(f"   Unique txn IDs   : {total_txns_cashier:,}")

    if df_social.empty:
        print("\n❌  No respond.io rows found. Check 'Ordered Via' column values:")
        print(df["Ordered Via"].value_counts().head(10).to_string())
        return

    # ── 3. Gap report — gap rows KEPT, not dropped ─────────────────────────
    # etl_local uses an outer merge so cashier-only rows land here with null
    # Description. We label them and keep them so Power BI shows the gap.
    has_sales = (
        df_social["Description"].notna()
        & (df_social["Description"].astype(str).str.strip() != "")
    )
    gap_rows = (~has_sales).sum()
    gap_txns = df_social.loc[~has_sales, "Transaction ID"].nunique()

    print(f"\n📊  REVENUE GAP REPORT")
    print(f"    respond.io txns in cashier : {total_txns_cashier:>6,}")
    print(f"    Txns with sales line items : {df_social.loc[has_sales, 'Transaction ID'].nunique():>6,}")
    if gap_txns:
        print(f"    ⚠ MISSING from sales      : {gap_txns:>6,} txns / {gap_rows:>6,} rows "
              f"← follow up with branch managers")
    else:
        print(f"    ✅ MISSING from sales      :      0  ← fully matched")

    # ── 4. Numeric prep ────────────────────────────────────────────────────
    df_social["Total (Tax Ex)"] = pd.to_numeric(
        df_social["Total (Tax Ex)"], errors="coerce"
    ).fillna(0)
    df_social["Qty Sold"] = pd.to_numeric(
        df_social["Qty Sold"], errors="coerce"
    ).fillna(1)

    # ── 5. Phone normalisation (needed for ad attribution join) ────────────
    if "Phone Number" in df_social.columns:
        df_social["norm_phone"] = df_social["Phone Number"].apply(normalize_phone)
        df_social["Phone Number"] = df_social["norm_phone"]   # clean in-place too

    # ── 6. Apply gap labels before KB matching loop ────────────────────────
    for col, val in _GAP_MATCH.items():
        df_social.loc[~has_sales, col] = val

    # ── 7. Load KB ─────────────────────────────────────────────────────────
    print(f"\n📚 Loading Knowledge Base...")
    kb_df, brands, kb_by_brand, kb_by_item_code = load_kb()

    # ── 8. KB matching — unique (Item, Description) pairs (fast path) ────────
    # Item code is tried first (Stage 0 prefix + Stage 1 barcode),
    # description text is the fallback for brand/fuzzy stages.
    print(f"\n🔍 Matching unique items to KB...")
    df_social["_item_key"] = df_social["Item"].astype(str).str.strip().str.upper()

    unique_pairs = (
        df_social.loc[has_sales, ["_item_key", "Description"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    print(
        f"   {len(unique_pairs):,} unique (Item, Description) pairs "
        f"(vs {has_sales.sum():,} rows) — "
        f"{has_sales.sum() // max(len(unique_pairs), 1)}x speedup"
    )

    pair_matches = {}
    for _, row in unique_pairs.iterrows():
        item_key  = row["_item_key"]
        desc      = str(row["Description"])
        item_code = item_key if item_key not in ("", "NAN", "NONE") else None
        pair_matches[(item_key, desc)] = match_product(
            pos_desc        = desc,
            brands          = brands,
            kb_by_brand     = kb_by_brand,
            kb_df           = kb_df,
            item_code       = item_code,
            kb_by_item_code = kb_by_item_code,
        )

    rows_to_match = df_social[has_sales].copy()
    for col in MATCH_COLS:
        df_social.loc[has_sales, col] = rows_to_match.apply(
            lambda r, c=col: pair_matches.get(
                (r["_item_key"], str(r["Description"])), _EMPTY_MATCH
            ).get(c, _EMPTY_MATCH[c]),
            axis=1,
        ).values

    df_social.drop(columns=["_item_key"], inplace=True, errors="ignore")

    # ── 9. Match stage breakdown ───────────────────────────────────────────
    n_total = len(df_social)
    print(f"\n   Total rows kept: {n_total:,}  (no rows dropped — all respond.io sales included)")
    matched_kb = (df_social["Matched_Brand"] != "Other Social Sale").sum()
    print(f"   Matched to KB  : {matched_kb:,} rows ({matched_kb / n_total * 100:.1f}%)")
    print(f"   Unmatched      : {n_total - matched_kb:,} rows — labelled 'Other Social Sale'")
    print(f"\n   Match stage breakdown:")
    for stage, count in df_social["match_stage"].value_counts().items():
        print(f"      {stage:<30} {count:>6,} rows ({count / n_total * 100:.1f}%)")

    # ── 10. Acquisition source enrichment ─────────────────────────────────
    print("\n🎯 Enriching with Acquisition Source...")
    df_ads_lookup = load_ads_for_pos()
    df_social = enrich_with_acquisition_source(df_social, df_ads_lookup)

    paid    = (df_social["acquisition_source"] == "Paid Ads").sum()
    organic = (df_social["acquisition_source"] == "Organic / Direct").sum()
    print(f"   Paid Ads       : {paid:,} rows")
    print(f"   Organic/Direct : {organic:,} rows")

    # ── 11. Build final output ─────────────────────────────────────────────
    available = [c for c in KEEP_COLS if c in df_social.columns]
    df_final  = df_social[available].copy()

    # ── 12. Save ───────────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df_final.to_csv(OUTPUT_FILE, index=False)

    # ── 13. Console summary ────────────────────────────────────────────────
    now         = datetime.now()
    month_start = pd.Timestamp(now.year, now.month, 1)
    month_label = now.strftime("%B %Y")

    df_final["Sale_Date"] = pd.to_datetime(df_final["Sale_Date"], errors="coerce")
    df_month = df_final[df_final["Sale_Date"] >= month_start]

    # Exclude gap rows from revenue totals
    df_with_rev  = df_final[df_final["match_stage"] != "Gap - No Sales Row"]
    df_month_rev = df_month[df_month["match_stage"] != "Gap - No Sales Row"]

    total_rev  = df_with_rev["Total (Tax Ex)"].sum()
    total_txns = df_with_rev["Transaction ID"].nunique()
    month_rev  = df_month_rev["Total (Tax Ex)"].sum()
    month_txns = df_month_rev["Transaction ID"].nunique()

    print(f"\n{'═'*65}")
    print("📊  SOCIAL SALES SUMMARY  (Total Tax Ex — sales line items)")
    print(f"{'═'*65}")

    print(f"\n  ── All Time {'─'*45}")
    print(f"  Revenue (Tax Ex)  : KES {total_rev:>12,.0f}")
    print(f"  Transactions      :     {total_txns:>7,}")
    print(f"  Line items        :     {len(df_with_rev):>7,}")
    print(f"  Gap rows (kept)   :     {gap_rows:>7,}")
    print(f"  Paid Ads          :     {paid:>7,}")
    print(f"  Organic / Direct  :     {organic:>7,}")

    print(f"\n  ── {month_label} {'─'*40}")
    print(f"  Revenue (Tax Ex)  : KES {month_rev:>12,.0f}")
    print(f"  Transactions      :     {month_txns:>7,}")
    print(f"  Line items        :     {len(df_month_rev):>7,}")

    if not df_month_rev.empty:
        print(f"\n  ── Top 10 Brands ({month_label}) {'─'*28}")
        brand_rev = (
            df_month_rev.groupby("Matched_Brand")["Total (Tax Ex)"]
            .sum().sort_values(ascending=False).head(10)
        )
        for brand, rev in brand_rev.items():
            print(f"     {brand:<32} KES {rev:>10,.0f}")

    print(f"\n✅  Saved → {OUTPUT_FILE}")

    # ── 14. Concerns exploded report ───────────────────────────────────────
    # Split comma-separated concerns into one row per concern for Power BI drill-through
    df_concern = df_final.copy()
    df_concern["Matched_Concern"] = (
        df_concern["Matched_Concern"].fillna("General").str.split(",")
    )
    df_concern = df_concern.explode("Matched_Concern")
    df_concern["Matched_Concern"] = df_concern["Matched_Concern"].str.strip()
    df_concern = df_concern[
        df_concern["Matched_Concern"].notna()
        & (df_concern["Matched_Concern"] != "")
    ]
    df_concern.to_csv(CONCERN_OUTPUT_FILE, index=False)
    print(f"✅  Concern export → {CONCERN_OUTPUT_FILE}")
    print(f"   {len(df_concern):,} rows · {df_concern['Matched_Concern'].nunique():,} unique concerns")
    print(f"   Top concerns:\n{df_concern['Matched_Concern'].value_counts().head(10).to_string()}")


if __name__ == "__main__":
    run_social_sales_direct()