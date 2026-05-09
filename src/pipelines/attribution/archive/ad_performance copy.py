"""
ad_performance.py
=================
Produces a per-ad ROAS report by joining:
  - Meta Ads export  (spend, impressions, reach per Ad ID)
  - social_sales_direct.csv  (attributed revenue per Ad ID)

Output:
  data/03_processed/ads/ad_roas.csv

Pipeline dependency:
    social_sales_etl.py → social_sales_direct.csv → this script → ad_roas.csv

Run:
    python -m Portal_ML_V4.src.pipelines.attribution.ad_performance
    — or —
    python ad_performance.py
"""

import os
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ══════════════════════════════════════════════════════════════════════════════
# IMPORTS — graceful fallbacks so the file runs standalone too
# ══════════════════════════════════════════════════════════════════════════════
try:
    from Portal_ML_V4.src.config.settings import (
        META_ADS_DIR,
        PROCESSED_DATA_DIR,
        USD_KES_EXCHANGE_RATE,
    )
except ImportError:
    # Standalone fallback — adjust paths if running directly
    META_ADS_DIR         = Path(r"D:\\Documents\\Portal ML Analys\\Portal_ML\\Portal_ML_V4\\data\\01_raw\\meta_ads")
    PROCESSED_DATA_DIR   = Path(r"D:\\Documents\\Portal ML Analys\\Portal_ML\\Portal_ML_V4\\data\\03_processed")
    USD_KES_EXCHANGE_RATE = 130

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

SOCIAL_SALES_FILE = PROCESSED_DATA_DIR / "sales_attribution" / "social_sales_direct.csv"
OUTPUT_DIR        = PROCESSED_DATA_DIR / "ads"
OUTPUT_FILE       = OUTPUT_DIR / "ad_roas.csv"

# Meta export column names — these come straight from the Facebook export header.
# 'Ad name' has a lowercase 'n' — do not change.
META_COL_AD_ID       = "Ad ID"
META_COL_AD_NAME     = "Ad name"
META_COL_SPEND_USD   = "Amount spent (USD)"
META_COL_IMPRESSIONS = "Impressions"
META_COL_REACH       = "Reach"
META_COL_RESULTS     = "Results"
META_COL_RESULT_IND  = "Result indicator"
META_COL_ADSET_NAME  = "Ad set name"
META_COL_START       = "Reporting starts"
META_COL_END         = "Reporting ends"


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def clean_scientific_id(val) -> str | None:
    """
    Converts scientific notation Ad IDs (e.g. 6.30697E+12) to plain strings.
    Redefined here to avoid importing analytics_copy.py (which pulls in sklearn).
    Source of truth lives in analytics_copy.clean_scientific_id — keep in sync.
    """
    if pd.isna(val) or str(val).strip() in ("", "-", "nan", "None"):
        return None
    try:
        return str(int(float(str(val).strip())))
    except ValueError:
        return str(val).strip().replace(".0", "")


# ══════════════════════════════════════════════════════════════════════════════
# LOADERS
# ══════════════════════════════════════════════════════════════════════════════

def load_meta_spend() -> pd.DataFrame:
    """
    Loads the Meta Ads Manager export.
    Cleans Ad IDs from scientific notation.
    Returns one row per Ad ID with spend and delivery metrics.
    """
    meta_path = Path(META_ADS_DIR)
    if not meta_path.exists():
        print(f"   ❌ Meta export not found: {meta_path}")
        return pd.DataFrame()

    # quotechar + engine='python' forces pandas to respect quoted fields.
    # Without this, ad names containing commas (e.g. "Product Awareness, Skincare")
    # shift all subsequent columns left, putting the ad name fragment into Ad ID.
    df = pd.read_csv(meta_path, dtype=str, quotechar='"', engine='python')
    df.columns = df.columns.str.strip()

    # Validate required columns exist
    required = [META_COL_AD_ID, META_COL_SPEND_USD]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"   ❌ Meta export missing columns: {missing}")
        print(f"      Found: {list(df.columns)}")
        return pd.DataFrame()

    # Clean Ad ID — scientific notation → plain string
    df["Ad_ID_Clean"] = df[META_COL_AD_ID].apply(clean_scientific_id)
    df = df.dropna(subset=["Ad_ID_Clean"])

    # Drop Meta summary / footer rows — Facebook exports often inject a "Total"
    # row or repeated header at the bottom. clean_scientific_id returns these
    # as-is (text), and dropna doesn't catch them because they're valid strings.
    # Filter to purely numeric IDs only.
    non_numeric = df[~df["Ad_ID_Clean"].str.isdigit()]
    if not non_numeric.empty:
        print(f"   ⚠️  Dropped {len(non_numeric)} non-numeric Ad ID rows "
              f"(Meta summary/footer): {non_numeric['Ad_ID_Clean'].tolist()}")
    df = df[df["Ad_ID_Clean"].str.isdigit()]

    # Numeric conversions
    df["Spend_USD"] = pd.to_numeric(df[META_COL_SPEND_USD], errors="coerce").fillna(0)
    df["Impressions"] = pd.to_numeric(df.get(META_COL_IMPRESSIONS, pd.Series(dtype=float)), errors="coerce").fillna(0)
    df["Reach"] = pd.to_numeric(df.get(META_COL_REACH, pd.Series(dtype=float)), errors="coerce").fillna(0)
    df["Results"] = pd.to_numeric(df.get(META_COL_RESULTS, pd.Series(dtype=float)), errors="coerce").fillna(0)

    # Optional descriptive columns — keep if present, blank string if not
    df["Ad_Name"]    = df.get(META_COL_AD_NAME, "")
    df["Adset_Name"] = df.get(META_COL_ADSET_NAME, "")
    df["Result_Indicator"] = df.get(META_COL_RESULT_IND, "")
    df["Reporting_Start"]  = df.get(META_COL_START, "")
    df["Reporting_End"]    = df.get(META_COL_END, "")

    # Aggregate — some exports have multiple rows per Ad ID (date breakdowns)
    df_spend = (
        df.groupby("Ad_ID_Clean", as_index=False)
        .agg(
            Ad_Name        = ("Ad_Name",        "first"),
            Adset_Name     = ("Adset_Name",      "first"),
            Result_Indicator = ("Result_Indicator", "first"),
            Reporting_Start  = ("Reporting_Start",  "min"),
            Reporting_End    = ("Reporting_End",    "max"),
            Spend_USD      = ("Spend_USD",       "sum"),
            Impressions    = ("Impressions",     "sum"),
            Reach          = ("Reach",           "sum"),
            Results        = ("Results",         "sum"),
        )
    )

    print(f"   ✅ Meta export loaded: {len(df_spend):,} ads · "
          f"Total spend USD {df_spend['Spend_USD'].sum():,.2f}")
    return df_spend


def load_attributed_revenue() -> pd.DataFrame:
    """
    Loads social_sales_direct.csv, filters to Paid Ads rows,
    and aggregates revenue + transaction count per Ad ID.
    """
    if not SOCIAL_SALES_FILE.exists():
        print(f"   ❌ Social sales file not found: {SOCIAL_SALES_FILE}")
        print("      Run social_sales_etl.py first.")
        return pd.DataFrame()

    df = pd.read_csv(SOCIAL_SALES_FILE, low_memory=False)
    df.columns = df.columns.str.strip()

    # Guard: columns we need
    for col in ["acquisition_source", "Ad ID", "Total (Tax Ex)"]:
        if col not in df.columns:
            print(f"   ❌ social_sales_direct.csv missing column: '{col}'")
            return pd.DataFrame()

    # Filter to paid rows only
    df_paid = df[df["acquisition_source"] == "Paid Ads"].copy()
    print(f"   ✅ Social sales loaded: {len(df_paid):,} paid rows "
          f"({df_paid['Transaction ID'].nunique():,} transactions) out of {len(df):,} total")

    if df_paid.empty:
        print("   ⚠️  No paid rows found — check acquisition_source values in social_sales_direct.csv")
        return pd.DataFrame()

    # Clean Ad ID to match Meta export format
    df_paid["Ad_ID_Clean"] = df_paid["Ad ID"].apply(clean_scientific_id)
    df_paid["Total (Tax Ex)"] = pd.to_numeric(df_paid["Total (Tax Ex)"], errors="coerce").fillna(0)

    # Aggregate per Ad ID
    df_rev = (
        df_paid.groupby("Ad_ID_Clean", as_index=False)
        .agg(
            Revenue_KES      = ("Total (Tax Ex)",  "sum"),
            Transactions     = ("Transaction ID",  "nunique"),
            Line_Items       = ("Transaction ID",  "count"),
        )
    )

    print(f"   ✅ Revenue aggregated: {len(df_rev):,} Ad IDs · "
          f"Total KES {df_rev['Revenue_KES'].sum():,.0f}")
    return df_rev


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run_ad_performance():
    print("=" * 65)
    print("  AD PERFORMANCE  (ROAS · per Meta Ad ID)")
    print(f"  FX Rate : 1 USD = {USD_KES_EXCHANGE_RATE} KES  (hardcoded — update in settings.py)")
    print(f"  Output  : {OUTPUT_FILE.name}")
    print("=" * 65)

    # ── 1. Load Meta spend ─────────────────────────────────────────────────
    print("\n📥 Loading Meta Ads export...")
    df_spend = load_meta_spend()
    if df_spend.empty:
        print("\n❌ Cannot build ROAS report — Meta export missing or invalid.")
        return

    # ── 2. Load attributed revenue ─────────────────────────────────────────
    print("\n📥 Loading attributed revenue from social_sales_direct.csv...")
    df_rev = load_attributed_revenue()
    if df_rev.empty:
        print("\n⚠️  No attributed revenue found. ROAS report will show spend only.")

    # ── 3. Join ────────────────────────────────────────────────────────────
    print("\n🔗 Joining spend and revenue on Ad ID...")
    if not df_rev.empty:
        df_roas = pd.merge(
            df_spend,
            df_rev,
            on="Ad_ID_Clean",
            how="left",   # keep all ads even if no revenue matched
        )
    else:
        df_roas = df_spend.copy()
        df_roas["Revenue_KES"] = 0.0
        df_roas["Transactions"] = 0
        df_roas["Line_Items"]   = 0

    df_roas["Revenue_KES"] = df_roas["Revenue_KES"].fillna(0)
    df_roas["Transactions"] = df_roas["Transactions"].fillna(0).astype(int)
    df_roas["Line_Items"]   = df_roas["Line_Items"].fillna(0).astype(int)

    # ── 4. Compute ROAS metrics ────────────────────────────────────────────
    df_roas["Spend_KES"] = (df_roas["Spend_USD"] * USD_KES_EXCHANGE_RATE).round(2)

    # ROAS = Revenue / Spend.  Guard against zero spend.
    df_roas["ROAS"] = df_roas.apply(
        lambda r: round(r["Revenue_KES"] / r["Spend_KES"], 2)
        if r["Spend_KES"] > 0 else None,
        axis=1,
    )

    # Cost per transaction (in KES) — blank if no transactions
    df_roas["Cost_Per_Txn_KES"] = df_roas.apply(
        lambda r: round(r["Spend_KES"] / r["Transactions"], 2)
        if r["Transactions"] > 0 else None,
        axis=1,
    )

    # Revenue per transaction — useful sanity check
    df_roas["Revenue_Per_Txn_KES"] = df_roas.apply(
        lambda r: round(r["Revenue_KES"] / r["Transactions"], 2)
        if r["Transactions"] > 0 else None,
        axis=1,
    )

    # ── 5. Clean up column order for Power BI ─────────────────────────────
    final_cols = [
        "Ad_ID_Clean", "Ad_Name", "Adset_Name",
        "Reporting_Start", "Reporting_End",
        "Spend_USD", "Spend_KES",
        "Revenue_KES", "Transactions", "Line_Items",
        "ROAS", "Cost_Per_Txn_KES", "Revenue_Per_Txn_KES",
        "Impressions", "Reach", "Results", "Result_Indicator",
    ]
    df_roas = df_roas[[c for c in final_cols if c in df_roas.columns]]
    df_roas = df_roas.sort_values("ROAS", ascending=False, na_position="last")

    # ── 6. Console summary ─────────────────────────────────────────────────
    matched_ads   = df_roas["ROAS"].notna().sum()
    unmatched_ads = df_roas["ROAS"].isna().sum()
    total_spend_kes = df_roas["Spend_KES"].sum()
    total_rev_kes   = df_roas["Revenue_KES"].sum()
    overall_roas    = round(total_rev_kes / total_spend_kes, 2) if total_spend_kes > 0 else None

    print(f"\n{'═'*65}")
    print("📊  AD ROAS SUMMARY")
    print(f"{'═'*65}")
    print(f"  Total ad spend     : USD {df_roas['Spend_USD'].sum():>10,.2f}  "
          f"(KES {total_spend_kes:>12,.0f})")
    print(f"  Total revenue attr : KES {total_rev_kes:>12,.0f}")
    print(f"  Overall ROAS       : {overall_roas}x" if overall_roas else "  Overall ROAS       : N/A (no spend)")
    print(f"  Ads with revenue   : {matched_ads:,}")
    print(f"  Ads with no match  : {unmatched_ads:,}  ← Ad ID not found in social_sales_direct.csv")

    if matched_ads > 0:
        print(f"\n  ── Top 5 Ads by ROAS {'─'*38}")
        top5 = df_roas[df_roas["ROAS"].notna()].head(5)
        for _, row in top5.iterrows():
            name = str(row.get("Ad_Name", ""))[:40] or str(row["Ad_ID_Clean"])
            print(f"     {name:<42} ROAS {row['ROAS']:>7.1f}x  "
                  f"KES {row['Revenue_KES']:>10,.0f}")

        print(f"\n  ── Bottom 5 Ads by ROAS (spending but not converting) {'─'*5}")
        bottom5 = df_roas[df_roas["ROAS"].notna()].tail(5)
        for _, row in bottom5.iterrows():
            name = str(row.get("Ad_Name", ""))[:40] or str(row["Ad_ID_Clean"])
            print(f"     {name:<42} ROAS {row['ROAS']:>7.1f}x  "
                  f"Spend KES {row['Spend_KES']:>8,.0f}")

    # ── 7. Save ────────────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df_roas.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"\n✅  Saved → {OUTPUT_FILE}")
    print(f"    {len(df_roas):,} rows  ·  columns: {list(df_roas.columns)}")


if __name__ == "__main__":
    run_ad_performance()