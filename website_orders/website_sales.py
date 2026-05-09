"""
website_sales_etl.py
====================
Website sales report — reads from etl_local's output file.

Pipeline dependency:
    etl_local.py → all_locations_sales_NEW.csv → this script → website_sales_direct.csv

Matching waterfall per line item:
    Stage 0  — Prefix override  (PRE* → Prescription, ANT* → Antibiotics)
    Stage 1  — Exact barcode    (Item column → KB Item Code Final)
    Stage 2  — Brand Only       (brand detected, fuzzy below threshold)
    Stage 3  — Fuzzy            (brand + fuzzy name similarity above threshold)
    Unmatched — no brand found  → labelled "Other Website Sale"

REVENUE NOTE:
    Revenue = Total (Tax Ex) from sales line items.
    Cashier-only rows (no matching sales row) are KEPT with match_stage = "Gap - No Sales Row"
    so the gap is visible in Power BI rather than silently dropped.

No ad attribution — website orders don't have a reliable phone → ad-click chain.

Run:
    python -m Portal_ML_V4.src.pipelines.website_sales_etl
    — or —
    python website_sales_etl.py
"""

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
    )
except ImportError:
    BASE_DIR           = Path(r"D:\\Documents\\Portal ML Analys\\Portal_ML\\Portal_ML_V4")
    PROCESSED_DATA_DIR = BASE_DIR / "data" / "03_processed"

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


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

SALES_FILE  = PROCESSED_DATA_DIR / "pos_data" / "all_locations_sales_NEW.csv"
KB_PATH     = BASE_DIR / "data" / "01_raw" / "Final_Knowledge_Base_PowerBI_New.csv"
OUTPUT_DIR  = PROCESSED_DATA_DIR / "sales_attribution"
OUTPUT_FILE          = OUTPUT_DIR / "website_sales_direct.csv"
CONCERN_OUTPUT_FILE  = OUTPUT_DIR / "website_sales_by_concern.csv"
CATEGORY_OUTPUT_FILE = OUTPUT_DIR / "website_sales_by_category.csv"

WEBSITE_TAG     = "website"          # exact match, lowercased
DATE_CUTOFF     = pd.Timestamp("2025-01-01")
MATCH_THRESHOLD = 0.50

STOP_WORDS = {
    "FOR", "WITH", "OF", "TO", "IN", "ON", "AT", "ML", "GM", "PCS",
    "TUBE", "BOTTLE", "CAPS", "TABS", "SYR",
}

# Output column order
KEEP_COLS = [
    "Sale_Date", "Location", "Transaction ID",
    "Description", "Qty Sold", "Total (Tax Ex)",
    "Ordered Via", "Client Name", "Phone Number", "Sales Rep",
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
    "Matched_Brand":        "Other Website Sale",
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
    """Returns (kb_df, brands, kb_by_brand, kb_by_item_code)."""
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
            # Multi-word brand fallback (e.g. "SOL DE JANEIRO" where "DE" is too short)
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

    return list(dict.fromkeys(found))


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
        result["Matched_Product"]  = pos_desc
        result["Matched_Category"] = "General"
        result["match_stage"]      = "Stage 2 - Brand Only"

    return result


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run_website_sales_etl():
    print("=" * 65)
    print("  WEBSITE SALES ETL  (ordered_via = 'website' · Jan 2025+)")
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

    # ── 2. Filter to website + Jan 2025 onwards ────────────────────────────
    mask = (
        df["Ordered Via"]
        .fillna("").astype(str).str.strip().str.lower() == WEBSITE_TAG
    )
    df_web = df[mask].copy()

    df_web["Sale_Date"] = pd.to_datetime(df_web["Sale_Date"], errors="coerce")
    df_web = df_web[df_web["Sale_Date"] >= DATE_CUTOFF].copy()

    total_txns = df_web["Transaction ID"].nunique()
    print(f"   Website rows (Jan 2025+) : {len(df_web):,}")
    print(f"   Unique transaction IDs   : {total_txns:,}")

    if df_web.empty:
        print("\n❌  No website rows found. Check 'Ordered Via' column values:")
        print(df["Ordered Via"].value_counts().head(10).to_string())
        return

    # ── 3. Gap report ──────────────────────────────────────────────────────
    has_sales = (
        df_web["Description"].notna()
        & (df_web["Description"].astype(str).str.strip() != "")
    )
    gap_rows = (~has_sales).sum()
    gap_txns = df_web.loc[~has_sales, "Transaction ID"].nunique()

    print(f"\n📊  REVENUE GAP REPORT")
    print(f"    Website txns in cashier    : {total_txns:>6,}")
    print(f"    Txns with sales line items : {df_web.loc[has_sales, 'Transaction ID'].nunique():>6,}")
    if gap_txns:
        print(f"    ⚠ MISSING from sales       : {gap_txns:>6,} txns / {gap_rows:>6,} rows "
              f"← follow up with branch managers")
    else:
        print(f"    ✅ MISSING from sales       :      0  ← fully matched")

    # ── 4. Numeric prep ────────────────────────────────────────────────────
    df_web["Total (Tax Ex)"] = pd.to_numeric(
        df_web["Total (Tax Ex)"], errors="coerce"
    ).fillna(0)
    df_web["Qty Sold"] = pd.to_numeric(
        df_web["Qty Sold"], errors="coerce"
    ).fillna(1)

    # ── 5. Phone normalisation ─────────────────────────────────────────────
    if "Phone Number" in df_web.columns:
        df_web["Phone Number"] = df_web["Phone Number"].apply(normalize_phone)

    # ── 6. Apply gap labels before KB matching loop ────────────────────────
    for col, val in _GAP_MATCH.items():
        df_web.loc[~has_sales, col] = val

    # ── 7. Load KB ─────────────────────────────────────────────────────────
    print(f"\n📚 Loading Knowledge Base...")
    kb_df, brands, kb_by_brand, kb_by_item_code = load_kb()

    # ── 8. KB matching — unique (Item, Description) pairs ─────────────────
    print(f"\n🔍 Matching unique items to KB...")
    df_web["_item_key"] = df_web["Item"].astype(str).str.strip().str.upper()

    unique_pairs = (
        df_web.loc[has_sales, ["_item_key", "Description"]]
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

    rows_to_match = df_web[has_sales].copy()
    for col in MATCH_COLS:
        df_web.loc[has_sales, col] = rows_to_match.apply(
            lambda r, c=col: pair_matches.get(
                (r["_item_key"], str(r["Description"])), _EMPTY_MATCH
            ).get(c, _EMPTY_MATCH[c]),
            axis=1,
        ).values

    df_web.drop(columns=["_item_key"], inplace=True, errors="ignore")

    # ── 9. Relabel unmatched brands ────────────────────────────────────────
    df_web["Matched_Brand"] = df_web["Matched_Brand"].apply(
        lambda b: b if b not in ("Unknown", "General", "") else "Other Website Sale"
    )
    df_web["Matched_Category"] = df_web.apply(
        lambda r: r["Matched_Category"]
        if r["Matched_Brand"] != "Other Website Sale" else "Unmatched",
        axis=1,
    )

    # ── 10. Match stage breakdown ──────────────────────────────────────────
    n_total    = len(df_web)
    matched_kb = (df_web["Matched_Brand"] != "Other Website Sale").sum()
    print(f"\n   Total rows kept: {n_total:,}  (no rows dropped — all website sales included)")
    print(f"   Matched to KB  : {matched_kb:,} rows ({matched_kb / n_total * 100:.1f}%)")
    print(f"   Unmatched      : {n_total - matched_kb:,} rows — labelled 'Other Website Sale'")
    print(f"\n   Match stage breakdown:")
    for stage, count in df_web["match_stage"].value_counts().items():
        print(f"      {stage:<30} {count:>6,} rows ({count / n_total * 100:.1f}%)")

    # ── 11. Build final output ─────────────────────────────────────────────
    available = [c for c in KEEP_COLS if c in df_web.columns]
    df_final  = df_web[available].copy()

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
    print("📊  WEBSITE SALES SUMMARY  (Total Tax Ex — sales line items)")
    print(f"{'═'*65}")

    print(f"\n  ── All Time (Jan 2025+) {'─'*37}")
    print(f"  Revenue (Tax Ex)  : KES {total_rev:>12,.0f}")
    print(f"  Transactions      :     {total_txns:>7,}")
    print(f"  Line items        :     {len(df_with_rev):>7,}")
    if gap_rows:
        print(f"  Gap rows (kept)   :     {gap_rows:>7,}")

    print(f"\n  ── {month_label} {'─'*44}")
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

        print(f"\n  ── Top 10 Locations ({month_label}) {'─'*25}")
        if "Location" in df_month_rev.columns:
            loc_rev = (
                df_month_rev.groupby("Location")["Total (Tax Ex)"]
                .sum().sort_values(ascending=False).head(10)
            )
            for loc, rev in loc_rev.items():
                print(f"     {loc:<32} KES {rev:>10,.0f}")

    print(f"\n✅  Saved → {OUTPUT_FILE}")

    # ── 14. Concerns exploded report ───────────────────────────────────────
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

    # ── 15. Category exploded report ───────────────────────────────────────
    df_cat = df_final.copy()
    df_cat["Matched_Category"] = (
        df_cat["Matched_Category"].fillna("General").str.split(",")
    )
    df_cat = df_cat.explode("Matched_Category")
    df_cat["Matched_Category"] = df_cat["Matched_Category"].str.strip()
    df_cat = df_cat[
        df_cat["Matched_Category"].notna()
        & (df_cat["Matched_Category"] != "")
    ]
    df_cat.to_csv(CATEGORY_OUTPUT_FILE, index=False)
    print(f"\n✅  Category export → {CATEGORY_OUTPUT_FILE}")
    print(f"   {len(df_cat):,} rows · {df_cat['Matched_Category'].nunique():,} unique categories")
    print(f"   Category breakdown:")
    for cat, count in df_cat["Matched_Category"].value_counts().items():
        print(f"      {cat:<35} {count:>6,}")


if __name__ == "__main__":
    run_website_sales_etl()