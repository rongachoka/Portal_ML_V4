"""
staff_performance_pos.py
========================
Produces staff performance data for Power BI across all branches.

Input : all_locations_sales_NEW.csv  (etl_local output)
Output:
    staff_sales_lineitems.csv   — one row per line item, Sales Rep cleaned
                                  Power BI uses this for revenue / units / category slices
    staff_transactions.csv      — one row per transaction
                                  Power BI uses this for avg basket size & transaction counts
    staff_daily_kpis.csv        — pre-aggregated (Sales Rep × Location × Date)
                                  optional fast-load table for summary visuals

Sales Rep cleaning rules:
    - Blank / null / whitespace-only → "Unassigned"   (rows are KEPT, not dropped)
    - Strip leading/trailing whitespace on all other values
    - Title-case normalisation so "FAITH" == "Faith"

Pipeline position (run_pipeline.py):
    Add after run_social_sales_direct() and before run_pos_loader()

    from Portal_ML_V4.src.pipelines.pos_finance.staff_performance_pos import run_staff_performance_pos
    ...
    print("Running Staff Performance POS\\n")
    run_staff_performance_pos()
""" 

import os
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ══════════════════════════════════════════════════════════════════════════════
# PATHS — graceful fallback so script runs standalone
# ══════════════════════════════════════════════════════════════════════════════
try:
    from Portal_ML_V4.src.config.settings import BASE_DIR, PROCESSED_DATA_DIR
except ImportError:
    BASE_DIR           = Path(r"D:\\Documents\\Portal ML Analys\\Portal_ML\\Portal_ML_V4")
    PROCESSED_DATA_DIR = BASE_DIR / "data" / "03_processed"

SALES_FILE   = PROCESSED_DATA_DIR / "pos_data" / "all_locations_sales_NEW.csv"
OUTPUT_DIR   = PROCESSED_DATA_DIR / "staff_performance"

OUTPUT_LINEITEMS    = OUTPUT_DIR / "staff_sales_lineitems.csv"
OUTPUT_TRANSACTIONS = OUTPUT_DIR / "staff_transactions.csv"
OUTPUT_DAILY_KPIS   = OUTPUT_DIR / "staff_daily_kpis.csv"

# ── Columns we want to carry through to Power BI ──────────────────────────────
# Add / remove as needed; the script will silently skip any that are absent.
KEEP_COLS = [
    "Sale_Date",
    "Location",
    "Transaction ID",
    "Item",
    "Description",
    "Qty Sold",
    "Total (Tax Ex)",
    "Ordered Via",
    "Client Name",
    "Phone Number",
    "Respond Customer ID",
    "Sales Rep",           # cleaned below
]

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

_JUNK_REP_VALUES = {"nan", "none", "n/a", "na", "null", "-", ""}

def clean_sales_rep(val) -> str:
    """
    Normalise a Sales Rep value.
    Blank / null / junk  → "Unassigned"  (row is kept)
    Everything else      → stripped + title-cased
    """
    if val is None:
        return "Unassigned"
    s = str(val).strip()
    if s.lower() in _JUNK_REP_VALUES:
        return "Unassigned"
    return s.title()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run_staff_performance_pos():
    print("=" * 65)
    print("  STAFF PERFORMANCE POS")
    print(f"  Input  : {SALES_FILE.name}")
    print(f"  Output : {OUTPUT_DIR}")
    print("=" * 65)

    if not SALES_FILE.exists():
        print(f"\n❌  Input file not found: {SALES_FILE}")
        print("    Run etl_local.py first to generate all_locations_sales_NEW.csv")
        return

    # ── 1. Load ───────────────────────────────────────────────────────────────
    print(f"\n📥 Loading {SALES_FILE.name}...")
    df = pd.read_csv(SALES_FILE, low_memory=False)
    print(f"   Raw rows  : {len(df):,}")
    print(f"   Columns   : {list(df.columns)}")

    # ── 2. Numeric coercion ───────────────────────────────────────────────────
    df["Total (Tax Ex)"] = pd.to_numeric(df.get("Total (Tax Ex)"), errors="coerce").fillna(0)
    df["Qty Sold"]       = pd.to_numeric(df.get("Qty Sold"),       errors="coerce").fillna(0)
    df["Sale_Date"]      = pd.to_datetime(df.get("Sale_Date"),     errors="coerce")

    # ── 3. Sales Rep cleaning ─────────────────────────────────────────────────
    # IMPORTANT: rows with blank Sales Rep are KEPT — labelled "Unassigned"
    before_unassigned = df["Sales Rep"].isna().sum() + (
        df["Sales Rep"].astype(str).str.strip()
        .str.lower().isin(_JUNK_REP_VALUES).sum()
    ) if "Sales Rep" in df.columns else 0

    if "Sales Rep" in df.columns:
        df["Sales Rep"] = df["Sales Rep"].apply(clean_sales_rep)
    else:
        print("   ⚠  'Sales Rep' column not found — setting all to 'Unassigned'")
        df["Sales Rep"] = "Unassigned"

    unassigned_count = (df["Sales Rep"] == "Unassigned").sum()
    print(f"\n   Sales Rep cleaning:")
    print(f"      Blank / null → 'Unassigned' : {before_unassigned:,} rows")
    print(f"      Total 'Unassigned' rows kept : {unassigned_count:,}")
    print(f"      Named staff rows             : {(df['Sales Rep'] != 'Unassigned').sum():,}")
    print(f"      Unique staff (incl Unassigned): {df['Sales Rep'].nunique():,}")
    print(f"\n   Staff breakdown:")
    for rep, count in df["Sales Rep"].value_counts().items():
        rev = df.loc[df["Sales Rep"] == rep, "Total (Tax Ex)"].sum()
        print(f"      {rep:<25} {count:>7,} rows   KES {rev:>12,.0f}")

    # ── 4. Keep only needed columns (silently drop missing ones) ──────────────
    available = [c for c in KEEP_COLS if c in df.columns]
    df = df[available].copy()

    # ── 5. OUTPUT A: Line-item level ──────────────────────────────────────────
    # One row per sales line item — Power BI aggregates revenue, units, etc.
    # This is the most flexible grain for slicers.
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(OUTPUT_LINEITEMS, index=False)
    print(f"\n✅ Line items saved  : {len(df):,} rows → {OUTPUT_LINEITEMS.name}")

    # ── 6. OUTPUT B: Transaction level ───────────────────────────────────────
    # One row per Transaction ID — for avg basket size visuals.
    # Basket size = total revenue for that transaction (sum across all line items).
    if "Transaction ID" in df.columns:
        txn_agg = (
            df.groupby(
                ["Transaction ID", "Sales Rep", "Location", "Sale_Date",
                 "Ordered Via", "Client Name"],
                dropna=False,
            )
            .agg(
                basket_revenue   = ("Total (Tax Ex)", "sum"),
                line_item_count  = ("Total (Tax Ex)", "count"),
                units_sold       = ("Qty Sold", "sum"),
            )
            .reset_index()
        )
        # Carry through optional phone / respond ID at txn level
        for extra_col in ["Phone Number", "Respond Customer ID"]:
            if extra_col in df.columns:
                phone_map = (
                    df.dropna(subset=[extra_col])
                    .drop_duplicates("Transaction ID")
                    .set_index("Transaction ID")[extra_col]
                )
                txn_agg[extra_col] = txn_agg["Transaction ID"].map(phone_map)

        txn_agg.to_csv(OUTPUT_TRANSACTIONS, index=False)
        avg_basket = txn_agg["basket_revenue"].mean()
        print(f"✅ Transactions saved : {len(txn_agg):,} rows → {OUTPUT_TRANSACTIONS.name}")
        print(f"   Overall avg basket : KES {avg_basket:,.0f}")
    else:
        print("   ⚠  'Transaction ID' column not found — skipping transaction output")

    # ── 7. OUTPUT C: Daily KPI summary ───────────────────────────────────────
    # Pre-aggregated (Sales Rep × Location × Sale_Date) — optional fast-load table.
    # Power BI can derive the same from Output A, but this speeds up large datasets.
    group_cols = [c for c in ["Sale_Date", "Location", "Sales Rep", "Ordered Via"]
                  if c in df.columns]

    daily = (
        df.groupby(group_cols, dropna=False)
        .agg(
            revenue       = ("Total (Tax Ex)", "sum"),
            units_sold    = ("Qty Sold",       "sum"),
            line_items    = ("Total (Tax Ex)", "count"),
        )
        .reset_index()
    )

    # Unique transaction count per group (can't do in one agg call cleanly)
    if "Transaction ID" in df.columns:
        txn_counts = (
            df.groupby(group_cols, dropna=False)["Transaction ID"]
            .nunique()
            .reset_index(name="transactions")
        )
        daily = daily.merge(txn_counts, on=group_cols, how="left")
        daily["avg_basket"] = (daily["revenue"] / daily["transactions"].replace(0, float("nan"))).round(2)

    daily.to_csv(OUTPUT_DAILY_KPIS, index=False)
    print(f"✅ Daily KPIs saved  : {len(daily):,} rows → {OUTPUT_DAILY_KPIS.name}")

    # ── 8. Console summary ────────────────────────────────────────────────────
    total_rev   = df["Total (Tax Ex)"].sum()
    total_units = df["Qty Sold"].sum()

    print(f"\n{'═'*65}")
    print("📊  STAFF PERFORMANCE SUMMARY  (all channels, all branches)")
    print(f"{'═'*65}")
    print(f"  Total Revenue (Tax Ex) : KES {total_rev:>12,.0f}")
    print(f"  Total Units Sold       :     {total_units:>10,.0f}")
    if "Transaction ID" in df.columns:
        total_txns = df["Transaction ID"].nunique()
        print(f"  Total Transactions     :     {total_txns:>10,}")

    print(f"\n  ── Revenue by Sales Rep {'─'*35}")
    rep_rev = (
        df.groupby("Sales Rep")["Total (Tax Ex)"]
        .sum().sort_values(ascending=False)
    )
    for rep, rev in rep_rev.items():
        txns = df.loc[df["Sales Rep"] == rep, "Transaction ID"].nunique() \
               if "Transaction ID" in df.columns else "-"
        pct  = rev / total_rev * 100 if total_rev > 0 else 0
        print(f"     {rep:<25} KES {rev:>12,.0f}   ({pct:.1f}%)   {txns} txns")

    print(f"\n  ── Revenue by Branch {'─'*38}")
    if "Location" in df.columns:
        branch_rev = (
            df.groupby("Location")["Total (Tax Ex)"]
            .sum().sort_values(ascending=False)
        )
        for branch, rev in branch_rev.items():
            pct = rev / total_rev * 100 if total_rev > 0 else 0
            print(f"     {str(branch):<25} KES {rev:>12,.0f}   ({pct:.1f}%)")

    print(f"\n✅  Done. Files in: {OUTPUT_DIR}")
    print("\n── Power BI Setup Guide ────────────────────────────────────────")
    print("1. Import all three CSVs into Power BI")
    print("2. Relate staff_sales_lineitems ← Transaction ID → staff_transactions")
    print("3. Key measures to create in DAX:")
    print("   Revenue        = SUM(staff_sales_lineitems[Total (Tax Ex)])")
    print("   Transactions   = DISTINCTCOUNT(staff_transactions[Transaction ID])")
    print("   Avg Basket     = DIVIDE([Revenue], [Transactions])")
    print("   Units Sold     = SUM(staff_sales_lineitems[Qty Sold])")
    print("4. Slicers: Sales Rep, Location, Sale_Date, Ordered Via")
    print("5. 'Unassigned' rows are included — filter them out in PBI if needed")
    print(f"{'═'*65}")


if __name__ == "__main__":
    run_staff_performance_pos()