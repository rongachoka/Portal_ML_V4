"""
ad_performance.py
=================
Builds the Ads Performance matrix for Power BI.

Sources:
  - META_ADS_DIR                → base ad metadata (spend, delivery, dates)
  - fact_sessions_enriched.csv  → Number of Inquiries per Ad ID
  - social_sales_direct.csv     → Converted Customers, Products Bought, Revenue

Outputs (written to data/03_processed/ads/):
  - fact_ad_performance.csv     → one row per ad  (main Power BI matrix)
  - fact_ad_products.csv        → one row per (ad × product line item)  (drill-through)

Filtering rules applied to the Meta Ads file:
  - Amount spent (USD) > 0
  - Ad delivery != 'not_delivering'   (keeps 'active' and 'inactive')

Pipeline position:
  Runs after run_social_sales_direct() — reads its output file.

Changes (2026-05):
  - fact_ad_products now carries product-level pre-aggregated stat columns:
      product_qty_sold, product_revenue_kes, product_num_transactions
    These are computed at the (Ad ID × Matched_Product) grain and stamped
    onto every line-item row so Power BI drill-through visuals can use them
    independently — without reading back the ad-level columns from
    fact_ad_performance and showing the same totals for every product.

  - fact_ad_performance gains a new column:
      products_bought — comma-separated list of unique Matched_Product names
    sold under each ad (all products included, even unmatched ones).
"""

import os
import pandas as pd
from pathlib import Path

# ── IMPORTS — graceful fallbacks so file can run standalone ───────────────
try:
    from Portal_ML_V4.src.config.settings import (
        BASE_DIR, PROCESSED_DATA_DIR, META_ADS_DIR,
    )
except ImportError:
    BASE_DIR           = Path(r"D:\\Documents\\Portal ML Analys\\Portal_ML\\Portal_ML_V4")
    PROCESSED_DATA_DIR = BASE_DIR / "data" / "03_processed"
    

# ══════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════

USD_TO_KES = 135          # conversion rate — update here when rate changes

# Input files
SESSIONS_FILE     = PROCESSED_DATA_DIR / "fact_sessions_enriched.csv"
SOCIAL_SALES_FILE = PROCESSED_DATA_DIR / "sales_attribution" / "social_sales_direct.csv"

# Output files
OUTPUT_DIR         = PROCESSED_DATA_DIR / "ads"
OUTPUT_PERFORMANCE = OUTPUT_DIR / "fact_ad_performance.csv"
OUTPUT_PRODUCTS    = OUTPUT_DIR / "fact_ad_products.csv"

# Ad delivery values to exclude from the matrix
EXCLUDE_DELIVERY = {"not_delivering"}

# Columns we want from the drill-through (products) output.
# product_qty_sold / product_revenue_kes / product_num_transactions are
# computed and appended in run_ad_performance() — listed here so the final
# column selection keeps them in a predictable order.
DRILL_COLS = [
    "Ad ID", "Ad Name",
    "Sale_Date", "Location", "Transaction ID",
    "Matched_Brand", "Matched_Product", "Matched_Category",
    "Matched_Sub_Category", "Matched_Concern",
    "Qty Sold", "Total (Tax Ex)",
    # ── product-level pre-aggregated stats ──
    "product_qty_sold",
    "product_revenue_kes",
    "product_num_transactions",
]


# ══════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════

def clean_ad_id(val) -> str | None:
    """
    Normalise Ad ID to a plain integer string.
    Handles scientific notation (e.g. 1.23456789e+17 → '123456789...').
    Returns None for blanks / invalid values so they can be dropped cleanly.
    """
    if pd.isna(val) or str(val).strip() in ("", "-", "nan", "None"):
        return None
    try:
        return str(int(float(str(val).strip())))
    except (ValueError, OverflowError):
        cleaned = str(val).strip().replace(".0", "")
        return cleaned if cleaned else None


def parse_meta_date(val) -> pd.Timestamp:
    """
    Parse Meta's 'Last significant edit' timestamp to a plain date.
    Input format:  2026-04-17T02:27:30+0300
    Output:        2026-04-17  (timezone dropped, time dropped)
    """
    if pd.isna(val):
        return pd.NaT
    try:
        # utc=True normalises any timezone offset; tz_localize(None) then strips tz
        return (
            pd.to_datetime(str(val), utc=True)
            .tz_localize(None)
            .normalize()           # zero out the time component → midnight
        )
    except Exception:
        return pd.NaT


# ══════════════════════════════════════════════════════════════════════════
# STEP 1 — META ADS BASE TABLE
# ══════════════════════════════════════════════════════════════════════════

def load_meta_ads() -> pd.DataFrame:
    """
    Load the Meta Ads Manager export and apply the two filter rules:
      1. Amount spent (USD) > 0
      2. Ad delivery is not 'not_delivering'

    Returns one row per unique Ad ID with spend and date columns.
    """
    meta_path = Path(META_ADS_DIR)
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Meta Ads file not found: {meta_path}\n"
            "Check META_ADS_DIR in settings.py."
        )

    df = pd.read_csv(meta_path, dtype=str)
    df.columns = df.columns.str.strip()
    print(f"   📥 Meta Ads raw rows: {len(df):,}")

    # ── Numeric spend ──────────────────────────────────────────────────
    df["Amount spent (USD)"] = pd.to_numeric(
        df["Amount spent (USD)"], errors="coerce"
    ).fillna(0)

    # ── Filter 1: spend > 0 ────────────────────────────────────────────
    df = df[df["Amount spent (USD)"] > 0].copy()
    print(f"   💰 After spend > 0 filter       : {len(df):,} rows")

    # ── Filter 2: exclude not_delivering ──────────────────────────────
    if "Ad delivery" in df.columns:
        df["Ad delivery"] = df["Ad delivery"].astype(str).str.strip().str.lower()
        before = len(df)
        df = df[~df["Ad delivery"].isin(EXCLUDE_DELIVERY)].copy()
        removed = before - len(df)
        print(f"   🚫 Removed {removed:,} 'not_delivering' ads        : {len(df):,} rows remain")

    # ── Clean Ad ID ────────────────────────────────────────────────────
    df["Ad ID"] = df["Ad ID"].apply(clean_ad_id)
    df = df.dropna(subset=["Ad ID"])

    # ── Parse last-edit date ───────────────────────────────────────────
    if "Last significant edit" in df.columns:
        df["last_edit_date"] = df["Last significant edit"].apply(parse_meta_date)
    else:
        print("   ⚠️  'Last significant edit' column not found — date will be null")
        df["last_edit_date"] = pd.NaT

    # ── Derived KES spend ──────────────────────────────────────────────
    df["Amount Spent KES"] = (df["Amount spent (USD)"] * USD_TO_KES).round(2)

    # ── Keep only what the matrix needs ───────────────────────────────
    keep = [
        "Ad ID", "Ad name", "Ad delivery", "last_edit_date",
        "Amount spent (USD)", "Amount Spent KES",
    ]
    df = df[[c for c in keep if c in df.columns]].copy()

    # Deduplicate — Meta exports can repeat an ad across date ranges
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["Ad ID"])
    if before_dedup > len(df):
        print(f"   ✂️  Deduped {before_dedup - len(df):,} repeated Ad ID rows (kept row with highest spend)")

    print(f"   ✅ Meta Ads loaded: {len(df):,} unique ads")
    return df


# ══════════════════════════════════════════════════════════════════════════
# STEP 2 — NUMBER OF INQUIRIES  (fact_sessions_enriched)
# ══════════════════════════════════════════════════════════════════════════

def load_inquiries() -> pd.DataFrame:
    """
    Count unique sessions per Ad ID from fact_sessions_enriched.csv.
    The Ad ID column in sessions is populated by analytics.py via
    merge_asof (6-hour tolerance on Contact ID × session_start).

    Returns: Ad ID | num_inquiries
    """
    if not SESSIONS_FILE.exists():
        print(f"   ⚠️  Sessions file not found — num_inquiries will be 0 for all ads")
        return pd.DataFrame(columns=["Ad ID", "num_inquiries"])

    # Only pull the two columns we need — keeps memory low on large session files
    df = pd.read_csv(SESSIONS_FILE, usecols=["session_id", "Ad ID"], dtype=str)
    df["Ad ID"] = df["Ad ID"].apply(clean_ad_id)
    df = df.dropna(subset=["Ad ID"])

    result = (
        df.groupby("Ad ID")["session_id"]
        .nunique()
        .reset_index()
        .rename(columns={"session_id": "num_inquiries"})
    )

    print(
        f"   ✅ Inquiries: {result['num_inquiries'].sum():,} sessions "
        f"across {len(result):,} ads"
    )
    return result


# ══════════════════════════════════════════════════════════════════════════
# STEP 3 — CONVERTED CUSTOMERS + REVENUE + PRODUCTS  (social_sales_direct)
# ══════════════════════════════════════════════════════════════════════════

def load_social_sales() -> tuple:
    """
    From social_sales_direct.csv, compute per Ad ID:
      - num_converted_customers  → unique Transaction IDs
        (Transaction ID has the fewest blanks across customer identifiers)
      - total_revenue_kes        → sum of Total (Tax Ex)

    Also returns the raw filtered rows for the products drill-through table.

    Returns: (agg_df, products_df)
    """
    if not SOCIAL_SALES_FILE.exists():
        print(" Social sales file not found — revenue/conversions will be 0")
        empty_agg = pd.DataFrame(
            columns=["Ad ID", "num_converted_customers", "total_revenue_kes"]
        )
        return empty_agg, pd.DataFrame()

    # Read only the columns we need — avoids loading the full wide file twice
    needed = [
        "Ad ID", "Transaction ID", "Total (Tax Ex)",
        "Matched_Brand", "Matched_Product", "Matched_Category",
        "Matched_Sub_Category", "Matched_Concern",
        "Qty Sold", "Sale_Date", "Location",
    ]
    # Peek at headers so we only request columns that actually exist
    available_cols = pd.read_csv(SOCIAL_SALES_FILE, nrows=0).columns.tolist()
    usecols = [c for c in needed if c in available_cols]

    df = pd.read_csv(SOCIAL_SALES_FILE, usecols=usecols, dtype=str, low_memory=False)

    # Clean Ad ID — social_sales_direct inherits the raw Meta value which can
    # still be in scientific notation depending on how the ads lookup was built
    df["Ad ID"] = df["Ad ID"].apply(clean_ad_id)
    df = df.dropna(subset=["Ad ID"])

    df["Total (Tax Ex)"] = pd.to_numeric(df["Total (Tax Ex)"], errors="coerce").fillna(0)
    df["Qty Sold"]       = pd.to_numeric(df["Qty Sold"],       errors="coerce").fillna(0)

    # ── Aggregation ────────────────────────────────────────────────────
    agg = (
        df.groupby("Ad ID")
        .agg(
            num_converted_customers=("Transaction ID", "nunique"),
            total_revenue_kes=("Total (Tax Ex)", "sum"),
        )
        .reset_index()
    )

    print(
        f"   ✅ Social sales: "
        f"{agg['num_converted_customers'].sum():,} transactions · "
        f"KES {agg['total_revenue_kes'].sum():,.0f} revenue "
        f"across {len(agg):,} ads"
    )
    return agg, df


# ══════════════════════════════════════════════════════════════════════════
# STEP 4 — PRODUCT-LEVEL STATS  (computed from social_sales_raw)
# ══════════════════════════════════════════════════════════════════════════

def compute_product_level_stats(df_sales_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-(Ad ID × Matched_Product) aggregates and stamp them back
    onto every line-item row.

    Why this fixes the Power BI drill-down:
    ----------------------------------------
    fact_ad_performance has num_inquiries, num_converted_customers, etc.
    pre-computed at the ad level. When Power BI drills into fact_ad_products
    it filters fact_ad_performance by Ad ID, so every product row inherits
    the same ad-level totals.

    By pre-computing product_qty_sold / product_revenue_kes /
    product_num_transactions here — at the (Ad ID × product) grain — and
    embedding them as plain columns on each line-item row, Power BI visuals
    in the drill-through page can read these directly without touching
    fact_ad_performance at all. Each product then shows its own numbers.

    Think of it like a receipt: the receipt header (fact_ad_performance)
    has the total spend for the whole order, while each line item
    (fact_ad_products) now also carries its own subtotal so a cashier
    can quote the cost of any single item without re-reading the header.
    """
    if df_sales_raw.empty:
        return df_sales_raw

    # ── Group at (Ad ID, Matched_Product) grain ────────────────────────
    prod_stats = (
        df_sales_raw.groupby(["Ad ID", "Matched_Product"], dropna=False)
        .agg(
            product_qty_sold        = ("Qty Sold",       "sum"),
            product_revenue_kes     = ("Total (Tax Ex)", "sum"),
            product_num_transactions= ("Transaction ID", "nunique"),
        )
        .reset_index()
    )

    prod_stats["product_qty_sold"]         = prod_stats["product_qty_sold"].round(0).astype(int)
    prod_stats["product_revenue_kes"]      = prod_stats["product_revenue_kes"].round(2)
    prod_stats["product_num_transactions"] = prod_stats["product_num_transactions"].astype(int)

    # ── Merge back onto line items ─────────────────────────────────────
    df_enriched = df_sales_raw.merge(
        prod_stats, on=["Ad ID", "Matched_Product"], how="left"
    )

    print(
        f"   ✅ Product-level stats computed: "
        f"{len(prod_stats):,} unique (Ad × Product) combinations"
    )
    return df_enriched


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def run_ad_performance():
    print("=" * 65)
    print("  AD PERFORMANCE ETL")
    print(f"  USD → KES rate : {USD_TO_KES}")
    print(f"  Output dir     : {OUTPUT_DIR}")
    print("=" * 65)

    # ── 1. Base ad metadata ────────────────────────────────────────────
    print("\n📥 Loading Meta Ads...")
    df_meta = load_meta_ads()

    # ── 2. Inquiries ───────────────────────────────────────────────────
    print("\n📊 Loading Inquiries (fact_sessions_enriched)...")
    df_inquiries = load_inquiries()

    # ── 3. Social sales (conversions + revenue) ────────────────────────
    print("\n💰 Loading Social Sales (social_sales_direct)...")
    df_sales_agg, df_sales_raw = load_social_sales()

    # ── 4. Merge everything onto the Meta ads spine ────────────────────
    # Meta ads is the spine — every ad in the filtered Meta file appears
    # in the output even if it has zero inquiries or zero revenue yet.
    df = df_meta.copy()
    df = df.merge(df_inquiries,  on="Ad ID", how="left")
    df = df.merge(df_sales_agg,  on="Ad ID", how="left")

    # Fill nulls for ads with no matches in sessions / sales yet
    df["num_inquiries"]           = df["num_inquiries"].fillna(0).astype(int)
    df["num_converted_customers"] = df["num_converted_customers"].fillna(0).astype(int)
    df["total_revenue_kes"]       = df["total_revenue_kes"].fillna(0).round(2)

    # ── 5. ROAS ────────────────────────────────────────────────────────
    # ROAS = Total Revenue KES / Amount Spent KES
    # Use pd.NA for ads with zero spend so Power BI shows blank rather than ∞
    df["ROAS"] = (
        df["total_revenue_kes"]
        / df["Amount Spent KES"].replace(0, pd.NA)
    ).round(2)

    # ── 6. products_bought column ──────────────────────────────────────
    # Comma-separated list of every unique Matched_Product sold under
    # each ad. All products are included (matched + unmatched).
    # Useful as a quick "what sold" label directly on the ad row.
    if not df_sales_raw.empty and "Matched_Product" in df_sales_raw.columns:
        products_per_ad = (
            df_sales_raw[df_sales_raw["Matched_Product"] != "Delivery Fee"]
            .groupby("Ad ID")["Matched_Product"]
            .apply(lambda s: ", ".join(
                sorted(s.dropna().astype(str).str.strip().unique())
            ))
            .reset_index()
            .rename(columns={"Matched_Product": "products_bought"})
        )
        df = df.merge(products_per_ad, on="Ad ID", how="left")
        df["products_bought"] = df["products_bought"].fillna("")
        print(f"   ✅ products_bought column built for {products_per_ad['Ad ID'].nunique():,} ads")
    else:
        df["products_bought"] = ""

    # ── 7. Final column order ──────────────────────────────────────────
    final_cols = [
        "Ad ID", "Ad name", "Ad delivery", "last_edit_date",
        "Amount spent (USD)", "Amount Spent KES",
        "num_inquiries", "num_converted_customers",
        "total_revenue_kes", "ROAS",
        "products_bought",
    ]
    df_final = df[[c for c in final_cols if c in df.columns]].copy()

    # ── 8. Save performance table ──────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df_final.to_csv(OUTPUT_PERFORMANCE, index=False)
    print(f"\n✅ fact_ad_performance.csv → {len(df_final):,} ads saved")

    # ── 9. Products drill-through table ───────────────────────────────
    if not df_sales_raw.empty:
        # ── 9a. Add Ad Name onto each line item ────────────────────────
        ad_name_map = df_meta.set_index("Ad ID")["Ad name"].to_dict()
        df_sales_raw["Ad Name"] = df_sales_raw["Ad ID"].map(ad_name_map)

        # ── 9b. Compute & stamp product-level stats ────────────────────
        # This is the key fix: each row in fact_ad_products now carries
        # product_qty_sold, product_revenue_kes, product_num_transactions
        # computed at the (Ad ID × Matched_Product) grain — NOT at the
        # ad level. Power BI drill-through visuals must reference these
        # columns (not num_inquiries / total_revenue_kes from the
        # fact_ad_performance table) to show per-product breakdowns.
        print("\n🔢 Computing product-level stats for drill-through...")
        df_sales_raw = compute_product_level_stats(df_sales_raw)

        df_products = df_sales_raw[
            [c for c in DRILL_COLS if c in df_sales_raw.columns]
        ].copy()

        df_products.to_csv(OUTPUT_PRODUCTS, index=False)
        print(f"✅ fact_ad_products.csv  → {len(df_products):,} product rows saved")

        # Quick sanity check — show per-product breakdown for top ad
        top_ad_id = (
            df_final[df_final["total_revenue_kes"] > 0]
            .sort_values("total_revenue_kes", ascending=False)
            .iloc[0]["Ad ID"]
            if not df_final[df_final["total_revenue_kes"] > 0].empty
            else None
        )
        if top_ad_id:
            top_ad_name = ad_name_map.get(top_ad_id, top_ad_id)
            sample = (
                df_products[df_products["Ad ID"] == top_ad_id]
                [["Matched_Product", "product_qty_sold", "product_revenue_kes", "product_num_transactions"]]
                .drop_duplicates("Matched_Product")
                .sort_values("product_revenue_kes", ascending=False)
                .head(8)
            )
            print(f"\n   Sample drill-down for top ad: '{top_ad_name}'")
            print(f"   {'Product':<40} {'Qty':>6} {'Revenue KES':>14} {'Txns':>6}")
            print(f"   {'─'*70}")
            for _, r in sample.iterrows():
                print(
                    f"   {str(r['Matched_Product'])[:40]:<40} "
                    f"{int(r['product_qty_sold']):>6,} "
                    f"KES {r['product_revenue_kes']:>10,.0f} "
                    f"{int(r['product_num_transactions']):>6,}"
                )
    else:
        print("⚠️  No social sales data — fact_ad_products.csv not written")

    # ── 10. Console summary ─────────────────────────────────────────────
    total_spend_usd = df_final["Amount spent (USD)"].sum()
    total_spend_kes = df_final["Amount Spent KES"].sum()
    total_inquiries = df_final["num_inquiries"].sum()
    total_converted = df_final["num_converted_customers"].sum()
    total_revenue   = df_final["total_revenue_kes"].sum()
    overall_roas    = total_revenue / total_spend_kes if total_spend_kes > 0 else 0

    print(f"\n{'═'*65}")
    print("📊  AD PERFORMANCE SUMMARY")
    print(f"{'═'*65}")
    print(f"  Ads tracked (active + inactive) : {len(df_final):>6,}")
    print(f"  Total Spend (USD)               : ${total_spend_usd:>10,.2f}")
    print(f"  Total Spend (KES @ {USD_TO_KES})       : KES {total_spend_kes:>10,.0f}")
    print(f"  Total Inquiries                 : {total_inquiries:>10,}")
    print(f"  Total Converted Customers       : {total_converted:>10,}")
    print(f"  Total Revenue (KES)             : KES {total_revenue:>10,.0f}")
    print(f"  Overall ROAS                    : {overall_roas:>10.2f}x")
    print(f"{'═'*65}")

    # ── 11. Per-ad breakdown (top 10 by revenue) ───────────────────────
    if not df_final.empty:
        print(f"\n  ── Top 10 Ads by Revenue {'─'*33}")
        top = (
            df_final[df_final["total_revenue_kes"] > 0]
            .sort_values("total_revenue_kes", ascending=False)
            .head(10)
        )
        for _, row in top.iterrows():
            print(
                f"  {str(row.get('Ad name', 'N/A'))[:40]:<42}"
                f"  KES {row['total_revenue_kes']:>10,.0f}"
                f"  ROAS {row['ROAS'] if pd.notna(row['ROAS']) else '—':>6}"
            )


if __name__ == "__main__":
    run_ad_performance()