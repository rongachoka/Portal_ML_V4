"""
social_sales_direct.py
======================
Produces an accurate social media sales report by:
  1. Reading all_locations_sales_NEW.csv directly
  2. Filtering to rows where Ordered Via = respond.io
  3. Matching each POS description to the Knowledge Base for brand/product/category
  4. Outputting social_sales_direct.csv for Power BI

This bypasses the attribution waterfall entirely — no phone matching needed.
The Ordered Via field in the cashier report is the source of truth.

Acquisition source enrichment — two paths:
  Path 1 (strong): Respond Customer ID → ads folder, 4-day backward window
  Path 2 (fallback): norm_phone → fact_sessions_enriched (±2 days) → Contact ID
                     → ads folder on the same calendar day as the session
"""

import re
import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from difflib import SequenceMatcher
from Portal_ML_V4.src.config.settings import (
    BASE_DIR, MSG_HISTORY_RAW, META_ADS_DIR
)
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

SALES_FILE          = BASE_DIR / "data" / "03_processed" / "pos_data" / "all_locations_sales_NEW.csv"
KB_PATH             = BASE_DIR / "data" / "01_raw" / "Final_Knowledge_Base_PowerBI_New.csv"
OUTPUT_FILE         = BASE_DIR / "data" / "03_processed" / "sales_attribution" / "social_sales_direct.csv"
CONCERN_OUTPUT_FILE = OUTPUT_FILE.parent / "social_sales_by_concern.csv"
SESSIONS_PATH       = BASE_DIR / "data" / "03_processed" / "fact_sessions_enriched.csv"

# Date bounds — no future dates, no pre-2026 data
DATE_LOWER = pd.Timestamp("2025-01-01")
DATE_UPPER = pd.Timestamp.today().normalize()   # today at midnight, recomputed at runtime

SOCIAL_TAGS     = ['respond.io']
MATCH_THRESHOLD = 0.50

STOP_WORDS = {
    'FOR', 'WITH', 'OF', 'TO', 'IN', 'ON', 'AT', 'ML', 'GM', 'PCS',
    'TUBE', 'BOTTLE', 'CAPS', 'TABS', 'SYR'
}

_POS_SALE_DATE_FORMATS = [
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d",
    # New sales export format from the May 2026 file-system cutover.
    "%d-%b-%Y %I:%M:%S %p",
    "%d-%b-%Y %I:%M %p",
    "%d-%b-%Y",
    "%d-%b-%y %I:%M:%S %p",
    "%d-%b-%y %I:%M %p",
    "%d-%b-%y",
    # all_locations_sales_NEW inherits sales-report "Date Sold", which comes
    # through in US-style month/day order. Keep MDY ahead of DMY so ambiguous
    # slash dates like 2/6/26 stay aligned with the source export.
    "%m/%d/%Y %H:%M:%S",
    "%m/%d/%Y %H:%M",
    "%m/%d/%Y",
    "%m/%d/%y %H:%M:%S",
    "%m/%d/%y %H:%M",
    "%m/%d/%y",
    "%m/%d/%Y %I:%M:%S %p",
    "%m/%d/%Y %I:%M %p",
    "%m/%d/%y %I:%M:%S %p",
    "%m/%d/%y %I:%M %p",
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M",
    "%d/%m/%Y",
    "%d/%m/%y %H:%M:%S",
    "%d/%m/%y %H:%M",
    "%d/%m/%y",
    "%d/%m/%Y %I:%M:%S %p",
    "%d/%m/%Y %I:%M %p",
    "%d/%m/%y %I:%M:%S %p",
    "%d/%m/%y %I:%M %p",
]


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
    inter    = ta & tb
    jaccard  = len(inter) / len(ta | tb)
    coverage = len(inter) / min(len(ta), len(tb))
    token    = jaccard * 0.4 + coverage * 0.6
    return seq * 0.35 + token * 0.65


def _parse_pos_sale_timestamp(value):
    if pd.isna(value):
        return pd.NaT

    cleaned = str(value).replace('#', '').strip()
    cleaned = re.sub(r'\s+', ' ', cleaned)
    if not cleaned or cleaned.lower() == 'nan':
        return pd.NaT

    for fmt in _POS_SALE_DATE_FORMATS:
        try:
            return pd.Timestamp(datetime.strptime(cleaned, fmt))
        except ValueError:
            continue

    return pd.NaT


def _resolve_sale_dates(df: pd.DataFrame) -> pd.Series:
    sale_dates = pd.to_datetime(
        df["Sale_Date"].apply(_parse_pos_sale_timestamp),
        errors="coerce",
    ) if "Sale_Date" in df.columns else pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    if "Date Sold" not in df.columns:
        return sale_dates.dt.normalize()

    raw_dates = pd.to_datetime(
        df["Date Sold"].apply(_parse_pos_sale_timestamp),
        errors="coerce",
    )
    # Trust the normalized ETL-produced Sale_Date when present. Raw Date Sold
    # can vary by branch/file after the POS export cutovers and should only be
    # used as a fallback for older outputs that lack a usable Sale_Date.
    resolved = sale_dates.where(sale_dates.notna(), raw_dates)
    return pd.to_datetime(resolved, errors="coerce").dt.normalize()


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

        if re.search(r'\b' + re.escape(b) + r'\b', safe):
            found.append(brand)
        elif b_no_the != b and re.search(r'\b' + re.escape(b_no_the) + r'\b', safe):
            found.append(brand)
        else:
            # Multi-word brand fallback — catches "SOL DE JANEIRO" where "DE" is too short
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


# ── PRODUCT MATCHER ───────────────────────────────────────────────────────────

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

    # ── Stage 0: Prefix-based medicine override ───────────────────────────────
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

    # ── Stage 1: Exact barcode / item code join ───────────────────────────────
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

    # ── Stage 2 & 3: Brand detection → fuzzy name matching ───────────────────
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
        result["Matched_Brand"]    = str(best_brand).title()
        result["Matched_Product"]  = pos_desc
        result["Matched_Category"] = "General"
        result["match_stage"]      = "Stage 2 - Brand Only"

    return result



# ── META AD NAME MAP ──────────────────────────────────────────────────────────

def _clean_scientific_id(val) -> str:
    """Strips scientific notation and trailing .0 from Ad IDs."""
    if pd.isna(val) or str(val).strip() in ('', '-', 'nan', 'None'):
        return None
    try:
        return str(int(float(str(val).strip())))
    except ValueError:
        return str(val).strip().replace('.0', '')


def load_meta_ad_name_map() -> dict:
    """
    Builds Ad ID → Ad Name map from the Meta export CSV defined in settings.
    Falls back to empty dict silently so the rest of the pipeline always runs.
    Mirrors the same function in analytics_copy.py — kept in sync manually.
    """
    meta_path = Path(META_ADS_DIR)
    if not meta_path.exists():
        print(f"   Meta ads file not found at {meta_path} — Ad Names will be blank")
        return {}
    try:
        df_meta = pd.read_csv(meta_path, dtype=str)
        df_meta.columns = df_meta.columns.str.strip()
        if 'Ad ID' not in df_meta.columns or 'Ad name' not in df_meta.columns:
            print(" Meta ads CSV missing 'Ad ID' or 'Ad name' columns — Ad Names will be blank")
            return {}
        df_meta['Ad ID'] = df_meta['Ad ID'].apply(_clean_scientific_id)
        df_meta = df_meta.dropna(subset=['Ad ID'])
        name_map = df_meta.set_index('Ad ID')['Ad name'].to_dict()
        print(f"   ✅ Meta ad name map: {len(name_map):,} ads loaded from {meta_path.name}")
        return name_map
    except Exception as e:
        print(f"   Could not build Meta ad name map: {e} — Ad Names will be blank")
        return {}



# ── AD LOADER ─────────────────────────────────────────────────────────────────

def load_ads_for_pos() -> pd.DataFrame:
    """
    Loads the ads folder and returns a clean DataFrame keyed by Contact ID.
    The phone-bridge via contacts.csv has been removed — the two attribution
    paths in enrich_with_acquisition_source() work directly with Contact ID.

    Returns columns: Contact ID (Int64), Timestamp, Ad ID, Ad campaign ID, Ad Name
    """
    ADS_DIR   = Path(MSG_HISTORY_RAW).parent / "ads"
    all_files = (
        glob.glob(str(ADS_DIR / "contacts-added*.csv")) +
        glob.glob(str(ADS_DIR / "contacts-connected*.csv"))
    )

    if not all_files:
        print("   ⚠️ No ad files found — acquisition_source will be 'Organic / Direct' for all rows.")
        return pd.DataFrame()

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
            # Safety strip — Respond Customer ID should be a clean integer but normalise anyway
            cid = temp['Contact ID'].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
            temp['Contact ID'] = pd.to_numeric(cid, errors='coerce').astype('Int64')
            temp = temp.dropna(subset=['Contact ID', 'Timestamp'])
            keep = [c for c in ['Timestamp', 'Contact ID', 'Ad ID', 'Ad campaign ID', 'Ad group ID']
                    if c in temp.columns]
            dfs.append(temp[keep])
        except Exception as e:
            print(f"   ⚠️ Error reading {Path(f).name}: {e}")

    if not dfs:
        return pd.DataFrame()

    df_ads = pd.concat(dfs, ignore_index=True).drop_duplicates().sort_values('Timestamp')

    # Map Ad Name from static registry
    if 'Ad ID' in df_ads.columns:
        ad_name_map = load_meta_ad_name_map()
        # clean_id = df_ads['Ad ID'].astype(str).str.replace(r'\.0$', '', regex=True)
        clean_ids = df_ads['Ad ID'].apply(_clean_scientific_id)
        df_ads['Ad Name'] = clean_ids.map(ad_name_map)

        # Sanitize — Meta ad names can contain newlines that break CSV row structure
        df_ads['Ad Name'] = (
            df_ads['Ad Name']
            .astype(str)
            .str.replace(r'[\n\r\t]', ' ', regex=True)
            .str.replace(r'\s+', ' ', regex=True)
            .str.strip()
            .replace('nan', None)
        )

    print(f"   ✅ Ads loaded: {len(df_ads):,} rows · "
          f"{df_ads['Contact ID'].nunique():,} unique contact IDs")
    return df_ads


# ── ACQUISITION SOURCE ENRICHMENT — TWO PATHS ─────────────────────────────────

def enrich_with_acquisition_source(
    df: pd.DataFrame,
    df_ads: pd.DataFrame,
    sessions_path: Path,
    path1_days: int = 4,
    path2_session_days: int = 2,
) -> pd.DataFrame:
    """
    Tags each row with acquisition_source + ad metadata via two paths.

    Path 1 — Respond Customer ID (strong signal, 4-day lookback):
        Respond Customer ID in sales file == Contact ID in ads folder.
        For each sale with a populated Respond Customer ID, find the most
        recent ad click within `path1_days` days before the sale date.

    Path 2 — Phone fallback (fires only when Respond Customer ID is absent):
        1. Normalize phone from the sales row.
        2. In fact_sessions_enriched, find rows where phone_number matches
           AND session_date ≤ sale_date ≤ session_date + path2_session_days.
           Multiple hits → take the most recent session.
        3. Pull Contact ID from that session row.
        4. In the ads folder, look for that Contact ID on the SAME calendar
           day as the matched session (ads register at the same minute as
           the session is opened).

    Adds columns: acquisition_source, Ad campaign ID, Ad ID, Ad Name
    """
    # Always initialise columns so Power BI never sees missing fields
    for col in ['acquisition_source', 'Ad campaign ID', 'Ad ID', 'Ad Name']:
        if col not in df.columns:
            df[col] = None
    df['acquisition_source'] = 'Organic / Direct'

    if df_ads.empty:
        print("   ⚠️ No ads data — all rows default to 'Organic / Direct'.")
        return df

    # ── Shared prep: Sale_Date must be datetime ────────────────────────────────
    df['Sale_Date'] = _resolve_sale_dates(df)
    df_ads_cmp = df_ads.copy()
    df_ads_cmp['event_date'] = pd.to_datetime(df_ads_cmp['Timestamp'], errors='coerce').dt.normalize()
    df_ads_cmp = df_ads_cmp.dropna(subset=['Contact ID', 'Timestamp', 'event_date'])
    df_ads_cmp = df_ads_cmp.sort_values('Timestamp')

    # ─────────────────────────────────────────────────────────────────────────
    # PATH 1 — Respond Customer ID → ads, 4-day backward window
    # ─────────────────────────────────────────────────────────────────────────
    print("   🔗 Path 1: Respond Customer ID → ads lookup...")

    # Normalise Respond Customer ID to Int64 for safe join (strip .0 etc.)
    # Guard: column may be absent in older sales exports
    if 'Respond Customer ID' in df.columns:
        rcid_raw = df['Respond Customer ID'].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
    else:
        print("      ⚠️  'Respond Customer ID' column not found in sales file — Path 1 skipped.")
        rcid_raw = pd.Series([''] * len(df), index=df.index)
    df['_rcid'] = pd.to_numeric(rcid_raw, errors='coerce').astype('Int64')

    path1_mask = df['_rcid'].notna()
    p1_matched = 0

    if path1_mask.any():
        # One row per transaction — avoid merge_asof duplicating line items
        txn_p1 = (
            df.loc[path1_mask, ['Transaction ID', '_rcid', 'Sale_Date']]
            .drop_duplicates('Transaction ID')
            .rename(columns={'_rcid': 'Contact ID'})
            .sort_values('Sale_Date')
        )

        matched_p1 = pd.merge_asof(
            txn_p1,
            df_ads_cmp[
                ['Contact ID', 'event_date', 'Timestamp', 'Ad ID', 'Ad campaign ID', 'Ad Name']
            ],
            left_on='Sale_Date',
            right_on='event_date',
            by='Contact ID',
            tolerance=pd.Timedelta(days=path1_days),
            direction='backward',
        )
        matched_p1 = matched_p1.set_index('Transaction ID')

        for col in ['Ad campaign ID', 'Ad ID', 'Ad Name']:
            col_map = matched_p1[col].dropna().to_dict()
            if col_map:
                hit_mask = df['Transaction ID'].isin(col_map)
                df.loc[hit_mask, col] = df.loc[hit_mask, 'Transaction ID'].map(col_map)

        # Mark as Paid Ads only for path1 rows where we actually found an ad
        paid_p1_txns = matched_p1[matched_p1['Ad campaign ID'].notna()].index
        p1_matched   = df['Transaction ID'].isin(paid_p1_txns).sum()
        df.loc[df['Transaction ID'].isin(paid_p1_txns), 'acquisition_source'] = 'Paid Ads'

    print(f"      Path 1: {path1_mask.sum():,} rows had Respond Customer ID · "
          f"{p1_matched:,} matched to an ad")

    # ─────────────────────────────────────────────────────────────────────────
    # PATH 2 — Phone → fact_sessions_enriched → Contact ID → ads (same day)
    # Only fires for rows still on 'Organic / Direct' with no Respond Customer ID
    # ─────────────────────────────────────────────────────────────────────────
    print("   🔗 Path 2: phone → sessions → Contact ID → ads lookup...")

    path2_mask = (
        ~path1_mask &                               # no Respond Customer ID
        df['norm_phone'].notna() &                  # phone present
        (df['acquisition_source'] == 'Organic / Direct')  # not already tagged
    )
    p2_matched = 0

    if path2_mask.any() and sessions_path.exists():
        # Load only the columns we need from sessions
        df_sess = pd.read_csv(
            sessions_path,
            usecols=['Contact ID', 'phone_number', 'session_start'],
            dtype={'phone_number': str},
        )
        df_sess['session_start'] = pd.to_datetime(df_sess['session_start'], errors='coerce')
        df_sess['session_date']  = df_sess['session_start'].dt.normalize()
        df_sess['Contact ID']    = pd.to_numeric(df_sess['Contact ID'], errors='coerce').astype('Int64')
        df_sess['phone_number']  = df_sess['phone_number'].apply(normalize_phone)
        df_sess = df_sess.dropna(subset=['Contact ID', 'phone_number', 'session_start', 'session_date'])
        df_sess = df_sess.sort_values('session_date')

        # One transaction per row — merge_asof can't handle line-item duplicates
        txn_p2 = (
            df.loc[path2_mask, ['Transaction ID', 'norm_phone', 'Sale_Date']]
            .drop_duplicates('Transaction ID')
            .sort_values('Sale_Date')
        )

        # Condition: session_date ≤ sale_date ≤ session_date + 2 days
        # Equivalent to: sale_date - 2 days ≤ session_date ≤ sale_date
        # → direction='backward', tolerance=2 days
        # When multiple sessions exist for the same phone, merge_asof returns
        # the MOST RECENT session ≤ sale_date (which is what we want).
        merged_sess = pd.merge_asof(
            txn_p2,
            df_sess[['phone_number', 'session_start', 'session_date', 'Contact ID']],
            left_on='Sale_Date',
            right_on='session_date',
            left_by='norm_phone',
            right_by='phone_number',
            tolerance=pd.Timedelta(days=path2_session_days),
            direction='backward',
        )

        # Only rows where a session was found
        matched_sess = merged_sess.dropna(subset=['Contact ID']).copy()
        matched_sess['Contact ID']   = matched_sess['Contact ID'].astype('Int64')

        if not matched_sess.empty:
            # Collapse ads to one row per contact/day, then join on the session day.
            ads_daily = (
                df_ads_cmp
                .rename(columns={'event_date': 'ad_date'})
                .drop_duplicates(subset=['Contact ID', 'ad_date'], keep='last')
                [['Contact ID', 'ad_date', 'Ad ID', 'Ad campaign ID', 'Ad Name']]
            )

            updates = (
                matched_sess[['Transaction ID', 'Contact ID', 'session_date']]
                .merge(
                    ads_daily,
                    left_on=['Contact ID', 'session_date'],
                    right_on=['Contact ID', 'ad_date'],
                    how='left',
                )
                .drop(columns=['ad_date'])
                .drop_duplicates(subset=['Transaction ID'])
                .set_index('Transaction ID')
            )

            if not updates.empty:
                for col in ['Ad campaign ID', 'Ad ID', 'Ad Name']:
                    col_map = updates[col].dropna().to_dict()
                    if col_map:
                        hit_mask = df['Transaction ID'].isin(col_map)
                        df.loc[hit_mask, col] = df.loc[hit_mask, 'Transaction ID'].map(col_map)

                paid_p2_txns = updates[updates['Ad campaign ID'].notna()].index
                p2_matched   = df['Transaction ID'].isin(paid_p2_txns).sum()
                df.loc[df['Transaction ID'].isin(paid_p2_txns), 'acquisition_source'] = 'Paid Ads'

        print(f"      Path 2: {path2_mask.sum():,} rows eligible · "
              f"{len(matched_sess):,} sessions matched · "
              f"{p2_matched:,} resolved to Paid Ads")

    elif path2_mask.any() and not sessions_path.exists():
        print(f"      ⚠️ Path 2 skipped — sessions file not found: {sessions_path}")

    # Cleanup temp column
    df.drop(columns=['_rcid'], inplace=True, errors='ignore')
    return df


# ── MAIN ──────────────────────────────────────────────────────────────────────

def run_social_sales_direct():
    print("=" * 60)
    print("VERSION: No brand filter — all respond.io rows kept")
    print("=" * 60)

    # Recompute upper bound at runtime so the script stays correct day-to-day
    today = pd.Timestamp.today().normalize()

    # 1. Load sales file
    print(f"\n📥 Loading sales file...")
    df = pd.read_csv(SALES_FILE, low_memory=False, dtype={"Ordered Via": str})
    print(f"   Total rows: {len(df):,}")

    # 2. Filter to social media sales using the same cashier tags as the
    # broader attribution pipeline.
    mask = df["Ordered Via"].fillna("").astype(str).str.strip().str.lower().eq("respond.io")
    df_social = df[mask].copy()
    df_social["Sale_Date"] = _resolve_sale_dates(df_social)

    # Apply date bounds — no pre-2026, no future dates
    before_date_filter = len(df_social)
    pre_2026   = (df_social["Sale_Date"] < DATE_LOWER).sum()
    future     = (df_social["Sale_Date"] > today).sum()
    null_dates = df_social["Sale_Date"].isna().sum()
    df_social = df_social[
        (df_social["Sale_Date"] >= DATE_LOWER) &
        (df_social["Sale_Date"] <= today)
    ].copy()
    dropped_dates = before_date_filter - len(df_social)
    if dropped_dates:
        print(f"   ⚠️ Dropped {dropped_dates:,} rows outside [{DATE_LOWER.date()} – {today.date()}]:")
        if pre_2026:   print(f"      · {pre_2026:,} rows pre-2026 (2025 data in file)")
        if future:     print(f"      · {future:,} rows with future Sale_Date")
        if null_dates: print(f"      · {null_dates:,} rows with unparseable/null Sale_Date")

    print(f"   Social-tagged rows:        {len(df_social):,}")
    print(f"   Revenue (Total Tax Ex): KES {df_social['Total (Tax Ex)'].apply(pd.to_numeric, errors='coerce').sum():,.0f}")
    print(f"   Unique transactions:    {df_social['Transaction ID'].nunique():,}")
    print(f"   Sale_Date range:        {df_social['Sale_Date'].min().date()} → {df_social['Sale_Date'].max().date()}")

    if df_social.empty:
        print("\n❌ No social media rows found. Check 'Ordered Via' column values:")
        print(df["Ordered Via"].value_counts().head(10))
        return

    # 3. Load KB
    print(f"\n📚 Loading Knowledge Base...")
    kb_df, brands, kb_by_brand, kb_by_item_code = load_kb()

    # 4. Numeric prep
    df_social["Total (Tax Ex)"] = pd.to_numeric(df_social["Total (Tax Ex)"], errors="coerce").fillna(0)
    df_social["Qty Sold"]       = pd.to_numeric(df_social["Qty Sold"],       errors="coerce").fillna(1)
    if "Amount" in df_social.columns:
        df_social["Amount"] = pd.to_numeric(df_social["Amount"], errors="coerce").fillna(0)

    # Clean phone numbers (needed for Path 2)
    if "Phone Number" in df_social.columns:
        df_social["Phone Number"] = df_social["Phone Number"].apply(normalize_phone)
    df_social["norm_phone"] = df_social.get("Phone Number", pd.Series(dtype=str)).apply(normalize_phone)

    # Keep revenue exactly as it appears in all_locations_sales_NEW.
    # Missing sales-report fields are allowed to stay blank so branch upload
    # gaps and duplicate-ID issues remain visible in the dashboard output.
    df_social["revenue_source"] = "POS Total Tax Ex"

    # 5. Match each UNIQUE (Item, Description) pair to KB
    print(f"\n🔍 Matching unique items to KB...")
    df_social["_item_key"] = df_social["Item"].astype(str).str.strip().str.upper()
    unique_pairs = (
        df_social[["_item_key", "Description"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    print(f"   {len(unique_pairs):,} unique (Item, Description) pairs "
          f"(vs {len(df_social):,} rows) — "
          f"{len(df_social)//max(len(unique_pairs),1)}x speedup")

    pair_matches = {}
    for _, row in unique_pairs.iterrows():
        item_key  = row["_item_key"]
        desc      = row["Description"]
        item_code = item_key if item_key not in ("", "NAN", "NONE") else None
        pair_matches[(item_key, desc)] = match_product(
            pos_desc        = desc,
            brands          = brands,
            kb_by_brand     = kb_by_brand,
            kb_df           = kb_df,
            item_code       = item_code,
            kb_by_item_code = kb_by_item_code,
        )

    MATCH_COLS = ["Matched_Brand", "Matched_Product", "Matched_Category",
                  "Matched_Sub_Category", "Matched_Concern", "cost_status", "match_stage"]
    empty_match = {c: "General" for c in MATCH_COLS}
    empty_match["Matched_Brand"] = "Unknown"
    empty_match["cost_status"]   = "No Cost Data"
    empty_match["match_stage"]   = "Unmatched"

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

    # 6. Keep ALL social-tagged rows — label unmatched as "Other Social Sale"
    df_out["Matched_Brand"] = df_out["Matched_Brand"].apply(
        lambda b: b if b not in ["Unknown", "General", ""] else "Other Social Sale"
    )
    df_out["Matched_Category"] = df_out.apply(
        lambda r: r["Matched_Category"] if r["Matched_Brand"] != "Other Social Sale" else "Unmatched",
        axis=1
    )
    print(f"   Total rows kept: {len(df_out):,} (no rows dropped — all social-tagged sales included)")
    matched = (df_out["Matched_Brand"] != "Other Social Sale").sum()
    print(f"   Matched to KB:   {matched:,} rows ({matched/len(df_out)*100:.1f}%)")
    print(f"   Unmatched:       {len(df_out)-matched:,} rows — labelled 'Other Social Sale'")
    print(f"\n   Match stage breakdown:")
    for stage, count in df_out["match_stage"].value_counts().items():
        print(f"      {stage:<30} {count:>6,} rows ({count/len(df_out)*100:.1f}%)")

    # 7. Acquisition Source Enrichment
    print("\n🎯 Enriching with Acquisition Source...")
    df_ads_lookup = load_ads_for_pos()
    df_out = enrich_with_acquisition_source(
        df_out, df_ads_lookup, SESSIONS_PATH,
        path1_days=4, path2_session_days=2,
    )
    paid    = (df_out['acquisition_source'] == 'Paid Ads').sum()
    organic = (df_out['acquisition_source'] == 'Organic / Direct').sum()
    print(f"   Paid Ads:        {paid:,} rows")
    print(f"   Organic/Direct:  {organic:,} rows")

    # 8. Build clean output

    keep_cols = [
        "Sale_Date", "Location", "Transaction ID",
        "Description", "Item", "Qty Sold", "Total (Tax Ex)",
        "Unit Cost", "Tax Amount", "Total Sales Amount", "Total Cost",
        "Ordered Via", "Respond Customer ID", "Client Name",
        "Phone Number", "Sales Rep",          # from cashier
        "Sales Rep ID", "Sales Rep Name",     # from sales file
        "acquisition_source",
        "Ad Name", "Ad campaign ID", "Ad ID",
        "Matched_Brand", "Matched_Product", "Matched_Category",
        "Matched_Sub_Category", "Matched_Concern", "cost_status",
        "match_stage",
    ]

    # keep_cols = [
    #     "Sale_Date", "Location", "Transaction ID",
    #     "Receipt Txn No", "Audit_Status",
    #     "Description", "Qty Sold", "Total (Tax Ex)",
    #     "Amount", "revenue_source",
    #     "Ordered Via", "Respond Customer ID", "Client Name",
    #     "Phone Number", "Sales Rep", "acquisition_source",
    #     "Ad Name", "Ad campaign ID", "Ad ID",
    #     "Matched_Brand", "Matched_Product", "Matched_Category",
    #     "Matched_Sub_Category", "Matched_Concern", "cost_status",
    #     "match_stage", "Item"
    # ]
    available = [c for c in keep_cols if c in df_out.columns]
    df_final  = df_out[available].copy()

    # 9. Console summary
    now         = datetime.now()
    month_start = pd.Timestamp(now.year, now.month, 1)
    month_label = now.strftime("%B %Y")

    df_final["Sale_Date"] = _resolve_sale_dates(df_final)
    df_month = df_final[df_final["Sale_Date"] >= month_start]

    month_rev  = df_month["Total (Tax Ex)"].sum()
    month_txns = df_month["Transaction ID"].nunique()
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


    # ── Sanitize Ad Name before writing — newlines in Meta ad names break CSV rows ──
    _str_cols_to_clean = ['Ad Name', 'Ad campaign ID', 'Ad ID']
    for _col in _str_cols_to_clean:
        if _col in df_final.columns:
            df_final[_col] = (
                df_final[_col]
                .astype(str)
                .str.replace(r'[\n\r\t]', ' ', regex=True)  # collapse embedded newlines
                .str.replace(r'\s+', ' ', regex=True)        # normalize whitespace
                .str.strip()
                .replace('nan', '')
            )

    # 10. Save
    os.makedirs(OUTPUT_FILE.parent, exist_ok=True)
    df_final.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"\n✅ Saved to: {OUTPUT_FILE}")

    # 11. Concerns Exploded Report
    df_concern = df_final.copy()
    df_concern['Matched_Concern'] = df_concern['Matched_Concern'].fillna('General').str.split(',')
    df_concern = df_concern.explode('Matched_Concern')
    df_concern['Matched_Concern'] = df_concern['Matched_Concern'].str.strip()
    df_concern = df_concern[df_concern['Matched_Concern'].notna()]
    df_concern = df_concern[df_concern['Matched_Concern'] != '']
    df_concern.to_csv(CONCERN_OUTPUT_FILE, index=False)
    print(f"\n✅ Concern-level export saved: {len(df_concern):,} rows → {CONCERN_OUTPUT_FILE}")
    print(f"   Unique concerns: {df_concern['Matched_Concern'].nunique():,}")
    print(f"   Top concerns:\n{df_concern['Matched_Concern'].value_counts().head(10).to_string()}")


if __name__ == "__main__":
    run_social_sales_direct()
