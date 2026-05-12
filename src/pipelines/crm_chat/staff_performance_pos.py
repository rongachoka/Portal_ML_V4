"""
staff_performance_pos.py
========================
Builds dashboard-ready staff performance outputs directly from
all_locations_sales_NEW.csv.

This version keeps the existing transaction-level fact output and adds:
  - Profit and quantity metrics
  - Robust transaction keys
  - Daily target vs achievement summary
  - Monthly peer-comparison summary
  - Overall staff ranking summary

Primary outputs:
  - fact_staff_performance_pos_direct.csv
  - staff_performance_pos_daily_summary.csv
  - staff_performance_pos_monthly_summary.csv
  - staff_performance_pos_overall_summary.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


try:
    from Portal_ML_V4.src.config.settings import PROCESSED_DATA_DIR
except ImportError:
    PROCESSED_DATA_DIR = Path(
        r"D:\Documents\Portal ML Analys\Portal_ML\Portal_ML_V4\data\03_processed"
    )


SALES_FILE = PROCESSED_DATA_DIR / "pos_data" / "all_locations_sales_NEW.csv"
FACT_OUTPUT_FILE = PROCESSED_DATA_DIR / "fact_staff_performance_pos_direct.csv"
DAILY_OUTPUT_FILE = PROCESSED_DATA_DIR / "staff_performance_pos_daily_summary.csv"
MONTHLY_OUTPUT_FILE = PROCESSED_DATA_DIR / "staff_performance_pos_monthly_summary.csv"
OVERALL_OUTPUT_FILE = PROCESSED_DATA_DIR / "staff_performance_pos_overall_summary.csv"

SOCIAL_TAG = "respond.io"

# Same canonical staff map used elsewhere in the project.
STAFF_ID_MAP = {
    "845968": "Joy",
    "847526": "Ishmael",
    "860475": "Faith",
    "879396": "Nimmoh",
    "879430": "Rahab",
    "879438": "Brenda",
    "971945": "Jeff",
    "1000558": "Sharon",
    "1006108": "Jess",
    "962460": "Katie",
    "1052677": "Vivian",
}

# POS alias -> canonical team member.
SALES_REP_MAP = {
    "Cate": "Katie",
    "Emily": "Nimmoh",
}

SYSTEM_NAMES = {
    "System",
    "Bot",
    "Auto Assign",
    "Workflow",
    "Unknown",
    "Nan",
    "None",
    "Unassigned",
    "",
}

# Fill these when management shares targets.
# Example:
# STAFF_DAILY_SALES_TARGETS = {"Joy": 25000, "Faith": 18000}
STAFF_DAILY_SALES_TARGETS: dict[str, float] = {}
DEFAULT_DAILY_SALES_TARGET = np.nan

NON_PRODUCT_PREFIXES = (
    "DELIVERY FEE",
    "DELIVERY FEES",
    "GOODS",
    "PRODUCT VAT",
    "PRODUCT ZERO",
)

FACT_OUTPUT_COLS = [
    "Reporting_Date",
    "Reporting_Month",
    "Staff_Name",
    "Location",
    "Attribution_Bucket",
    "Is_Converted",
    "Customers_Handled",
    "Revenue",
    "Cost",
    "Profit",
    "Gross_Margin_Pct",
    "Qty_Sold",
    "Product_Qty_Sold",
    "Average_Basket_Value",
    "Qty_Per_Basket",
    "Time_To_Conversion_Mins",
    "Messages_To_Conversion",
    "session_id",
    "Transaction_ID_Source",
    "Receipt_Txn_No",
    "Contact ID",
    "Ordered_Via",
]


def normalize_sales_rep(name: str) -> str | None:
    """Normalize the raw POS sales rep value to the canonical dashboard name."""
    if pd.isna(name):
        return None
    clean = str(name).strip().title()
    clean = SALES_REP_MAP.get(clean, clean)
    if clean in SYSTEM_NAMES or len(clean) < 2:
        return None
    return clean


def clean_contact_id(val) -> str | None:
    """Reduce Respond Customer ID to digits only."""
    if pd.isna(val):
        return None
    digits = "".join(filter(str.isdigit, str(val).strip().replace(".0", "")))
    return digits or None


def first_non_blank(series: pd.Series):
    """Return the first non-null, non-empty value in a series."""
    cleaned = series.dropna().astype(str).str.strip()
    cleaned = cleaned[cleaned != ""]
    return cleaned.iloc[0] if not cleaned.empty else None


def coalesce_numeric(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Return the first populated numeric value from the provided columns."""
    result = pd.Series(np.nan, index=df.index, dtype="float64")
    for col in columns:
        if col in df.columns:
            parsed = pd.to_numeric(df[col], errors="coerce")
            result = result.fillna(parsed)
    return result


def build_transaction_key(row: pd.Series) -> str:
    """
    Prefer the real transaction id, but fall back safely when POS leaves it blank.
    This avoids collapsing unrelated baskets into one group.
    """
    for col in ("Transaction ID", "_receipt_date_key", "Receipt Txn No"):
        val = str(row.get(col, "")).strip()
        if val and val.lower() not in {"nan", "none"}:
            return val

    sale_date = str(row.get("Sale_Date_Str") or row.get("Sale_Date") or "").strip()
    client = str(row.get("Client Name") or "").strip()
    phone = str(row.get("Phone Number") or "").strip()
    return f"fallback::{sale_date}::{client}::{phone}"


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Vectorized divide that returns NaN instead of inf when denominator is zero."""
    denominator = denominator.replace({0: np.nan})
    return numerator / denominator


def classify_product_line(description: pd.Series, item: pd.Series) -> pd.Series:
    """
    Exclude delivery/goods style rows from product quantity metrics.
    Revenue still stays in the basket, but quantity-per-basket should reflect products.
    """
    desc = description.fillna("").astype(str).str.strip().str.upper()
    item_clean = item.fillna("").astype(str).str.strip()

    prefix_mask = pd.Series(False, index=desc.index)
    for prefix in NON_PRODUCT_PREFIXES:
        prefix_mask = prefix_mask | desc.str.startswith(prefix)

    blank_item_mask = item_clean.isin({"", "nan", "None", "NaN"})
    return ~(prefix_mask | blank_item_mask)


def build_daily_summary(txn: pd.DataFrame) -> pd.DataFrame:
    """One row per staff member per day for target-vs-achievement tracking."""
    daily = (
        txn.groupby(["Reporting_Date", "Staff_Name"], as_index=False)
        .agg(
            Revenue=("Revenue", "sum"),
            Cost=("Cost", "sum"),
            Profit=("Profit", "sum"),
            Unique_Transactions=("session_id", "nunique"),
            Total_Customers=("session_id", "nunique"),
            Qty_Sold=("Qty_Sold", "sum"),
            Product_Qty_Sold=("Product_Qty_Sold", "sum"),
            Active_Locations=("Location", lambda s: ", ".join(sorted(set(s.dropna())))),
        )
        .sort_values(["Reporting_Date", "Revenue"], ascending=[False, False])
    )

    daily["Reporting_Month"] = pd.to_datetime(daily["Reporting_Date"]).dt.to_period("M").dt.to_timestamp()
    daily["Average_Basket_Value"] = safe_divide(daily["Revenue"], daily["Unique_Transactions"])
    daily["Qty_Per_Basket"] = safe_divide(daily["Product_Qty_Sold"], daily["Unique_Transactions"])
    daily["Gross_Margin_Pct"] = safe_divide(daily["Profit"], daily["Revenue"])

    daily["Daily_Sales_Target"] = (
        daily["Staff_Name"].map(STAFF_DAILY_SALES_TARGETS).astype(float)
        if STAFF_DAILY_SALES_TARGETS
        else DEFAULT_DAILY_SALES_TARGET
    )
    daily["Daily_Target_Gap"] = daily["Revenue"] - daily["Daily_Sales_Target"]
    daily["Daily_Target_Achievement_Pct"] = safe_divide(
        daily["Revenue"], daily["Daily_Sales_Target"]
    )

    daily["Target_Status"] = np.select(
        [
            daily["Daily_Sales_Target"].isna(),
            daily["Revenue"] >= daily["Daily_Sales_Target"],
        ],
        [
            "Target Not Set",
            "Above Target",
        ],
        default="Below Target",
    )

    return daily


def add_peer_comparison(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Add peer averages and above/below status for dashboard comparison cards."""
    comparison_metrics = {
        "Revenue": "Sales",
        "Profit": "Profit",
        "Unique_Transactions": "Transactions",
        "Average_Basket_Value": "Basket_Value",
        "Qty_Per_Basket": "Qty_Per_Basket",
    }

    result = df.copy()
    for source_col, label in comparison_metrics.items():
        peer_avg_col = f"Peer_Avg_{label}"
        gap_col = f"{label}_Vs_Peer_Avg"
        status_col = f"{label}_Peer_Status"

        result[peer_avg_col] = result.groupby(group_col)[source_col].transform("mean")
        result[gap_col] = result[source_col] - result[peer_avg_col]
        result[status_col] = np.where(
            result[source_col] >= result[peer_avg_col],
            "Above Average",
            "Below Average",
        )

    result["Overall_Peer_Status"] = np.where(
        result["Revenue"] >= result["Peer_Avg_Sales"],
        "Above Average",
        "Below Average",
    )
    return result


def build_monthly_summary(txn: pd.DataFrame) -> pd.DataFrame:
    """One row per staff member per month for ranking and peer comparisons."""
    monthly = (
        txn.groupby(["Reporting_Month", "Staff_Name"], as_index=False)
        .agg(
            Revenue=("Revenue", "sum"),
            Cost=("Cost", "sum"),
            Profit=("Profit", "sum"),
            Unique_Transactions=("session_id", "nunique"),
            Total_Customers=("session_id", "nunique"),
            Qty_Sold=("Qty_Sold", "sum"),
            Product_Qty_Sold=("Product_Qty_Sold", "sum"),
            Active_Days=("Reporting_Date", "nunique"),
        )
        .sort_values(["Reporting_Month", "Revenue"], ascending=[False, False])
    )

    monthly["Average_Basket_Value"] = safe_divide(
        monthly["Revenue"], monthly["Unique_Transactions"]
    )
    monthly["Qty_Per_Basket"] = safe_divide(
        monthly["Product_Qty_Sold"], monthly["Unique_Transactions"]
    )
    monthly["Gross_Margin_Pct"] = safe_divide(monthly["Profit"], monthly["Revenue"])
    monthly["Contribution_Pct_Total_Sales"] = safe_divide(
        monthly["Revenue"],
        monthly.groupby("Reporting_Month")["Revenue"].transform("sum"),
    )
    monthly["Sales_Rank_In_Month"] = (
        monthly.groupby("Reporting_Month")["Revenue"]
        .rank(method="dense", ascending=False)
        .astype("Int64")
    )

    monthly = add_peer_comparison(monthly, "Reporting_Month")
    return monthly


def build_overall_summary(txn: pd.DataFrame) -> pd.DataFrame:
    """One row per staff member across the full dataset for the top-level leaderboard."""
    overall = (
        txn.groupby("Staff_Name", as_index=False)
        .agg(
            Revenue=("Revenue", "sum"),
            Cost=("Cost", "sum"),
            Profit=("Profit", "sum"),
            Unique_Transactions=("session_id", "nunique"),
            Total_Customers=("session_id", "nunique"),
            Qty_Sold=("Qty_Sold", "sum"),
            Product_Qty_Sold=("Product_Qty_Sold", "sum"),
            Active_Days=("Reporting_Date", "nunique"),
        )
        .sort_values("Revenue", ascending=False)
    )

    overall["Average_Basket_Value"] = safe_divide(
        overall["Revenue"], overall["Unique_Transactions"]
    )
    overall["Qty_Per_Basket"] = safe_divide(
        overall["Product_Qty_Sold"], overall["Unique_Transactions"]
    )
    overall["Gross_Margin_Pct"] = safe_divide(overall["Profit"], overall["Revenue"])
    overall["Contribution_Pct_Total_Sales"] = safe_divide(
        overall["Revenue"], pd.Series(overall["Revenue"].sum(), index=overall.index)
    )
    overall["Overall_Sales_Rank"] = (
        overall["Revenue"].rank(method="dense", ascending=False).astype("Int64")
    )

    comparison_metrics = {
        "Revenue": "Sales",
        "Profit": "Profit",
        "Unique_Transactions": "Transactions",
        "Average_Basket_Value": "Basket_Value",
        "Qty_Per_Basket": "Qty_Per_Basket",
    }
    for source_col, label in comparison_metrics.items():
        peer_avg = overall[source_col].mean()
        overall[f"Peer_Avg_{label}"] = peer_avg
        overall[f"{label}_Vs_Peer_Avg"] = overall[source_col] - peer_avg
        overall[f"{label}_Peer_Status"] = np.where(
            overall[source_col] >= peer_avg,
            "Above Average",
            "Below Average",
        )

    overall["Overall_Peer_Status"] = np.where(
        overall["Revenue"] >= overall["Peer_Avg_Sales"],
        "Above Average",
        "Below Average",
    )
    return overall


def run_staff_analysis_pos():
    print("=" * 72)
    print("STAFF PERFORMANCE - POS DASHBOARD PIPELINE")
    print(f"Input file    : {SALES_FILE}")
    print(f"Fact output   : {FACT_OUTPUT_FILE.name}")
    print(f"Daily output  : {DAILY_OUTPUT_FILE.name}")
    print(f"Monthly output: {MONTHLY_OUTPUT_FILE.name}")
    print(f"Overall output: {OVERALL_OUTPUT_FILE.name}")
    print("=" * 72)

    if not SALES_FILE.exists():
        print(f"\nInput file not found: {SALES_FILE}")
        print("Run the POS ETL first so all_locations_sales_NEW.csv exists.")
        return

    print(f"\nLoading {SALES_FILE.name} ...")
    df = pd.read_csv(SALES_FILE, low_memory=False, dtype=str)
    df.columns = df.columns.str.strip()
    print(f"Total raw rows: {len(df):,}")

    is_respond = df["Ordered Via"].fillna("").str.strip().str.lower().eq(SOCIAL_TAG)
    has_contact_id = df["Respond Customer ID"].fillna("").str.strip().str.len().gt(0)
    df_attr = df[is_respond | has_contact_id].copy()

    print(f"respond.io rows      : {is_respond.sum():,}")
    print(f"Rows with Contact ID : {has_contact_id.sum():,}")
    print(f"Attributed rows      : {len(df_attr):,}")

    if df_attr.empty:
        print("\nNo attributed POS rows found.")
        return

    df_attr["Sale_Date"] = pd.to_datetime(df_attr["Sale_Date"], errors="coerce")
    df_attr["Staff_Name"] = df_attr["Sales Rep"].apply(normalize_sales_rep)
    df_attr["_contact_id"] = df_attr["Respond Customer ID"].apply(clean_contact_id)
    df_attr["Transaction_Key"] = df_attr.apply(build_transaction_key, axis=1)

    df_attr["Line_Revenue"] = coalesce_numeric(
        df_attr, ["Total (Tax Ex)", "Total Sales Amount"]
    ).fillna(0)
    df_attr["Qty_Sold"] = coalesce_numeric(df_attr, ["Qty Sold"]).fillna(0)

    unit_cost = coalesce_numeric(df_attr, ["Unit Cost"])
    line_cost = coalesce_numeric(df_attr, ["Total Cost"])
    fallback_cost = unit_cost * df_attr["Qty_Sold"]
    df_attr["Line_Cost"] = line_cost.fillna(fallback_cost).fillna(0)
    df_attr["Line_Profit"] = df_attr["Line_Revenue"] - df_attr["Line_Cost"]

    is_product_line = classify_product_line(df_attr["Description"], df_attr["Item"])
    df_attr["Product_Qty_Sold"] = np.where(is_product_line, df_attr["Qty_Sold"], 0)

    print("\nAggregating to transaction level ...")
    txn = (
        df_attr.groupby("Transaction_Key", as_index=False)
        .agg(
            Reporting_Date=("Sale_Date", "min"),
            Revenue=("Line_Revenue", "sum"),
            Cost=("Line_Cost", "sum"),
            Profit=("Line_Profit", "sum"),
            Qty_Sold=("Qty_Sold", "sum"),
            Product_Qty_Sold=("Product_Qty_Sold", "sum"),
            Staff_Name=("Staff_Name", first_non_blank),
            Contact_ID_raw=("_contact_id", first_non_blank),
            Location=("Location", first_non_blank),
            Transaction_ID_Source=("Transaction ID", first_non_blank),
            Receipt_Txn_No=("Receipt Txn No", first_non_blank),
            Ordered_Via=("Ordered Via", first_non_blank),
        )
        .sort_values("Reporting_Date", ascending=False)
    )

    print(f"Transactions found: {len(txn):,}")

    known_staff = set(STAFF_ID_MAP.values()) | set(SALES_REP_MAP.values())
    unrecognized = txn[
        txn["Staff_Name"].notna() & ~txn["Staff_Name"].isin(known_staff)
    ]["Staff_Name"].value_counts()

    if not unrecognized.empty:
        print("\nUnrecognized staff names excluded from dashboard output:")
        for name, count in unrecognized.items():
            print(f"  {name:<25} {count:>6,} transactions")

    txn.loc[
        txn["Staff_Name"].notna() & ~txn["Staff_Name"].isin(known_staff),
        "Staff_Name",
    ] = None

    txn = txn[txn["Staff_Name"].notna()].copy()
    txn = txn[~txn["Staff_Name"].isin(SYSTEM_NAMES)].copy()

    if txn.empty:
        print("\nNo valid staff-attributed transactions remained after staff cleaning.")
        return

    txn["Reporting_Date"] = pd.to_datetime(txn["Reporting_Date"]).dt.date
    txn["Reporting_Month"] = (
        pd.to_datetime(txn["Reporting_Date"]).dt.to_period("M").dt.to_timestamp()
    )
    txn["Attribution_Bucket"] = "1. Active (<= 24h)"
    txn["Is_Converted"] = 1
    txn["Customers_Handled"] = 1
    txn["Average_Basket_Value"] = txn["Revenue"]
    txn["Qty_Per_Basket"] = txn["Product_Qty_Sold"]
    txn["Gross_Margin_Pct"] = safe_divide(txn["Profit"], txn["Revenue"])
    txn["Time_To_Conversion_Mins"] = np.nan
    txn["Messages_To_Conversion"] = np.nan
    txn["session_id"] = txn["Transaction_Key"]
    txn["Contact ID"] = txn["Contact_ID_raw"]

    fact_df = txn[FACT_OUTPUT_COLS].copy()
    daily_df = build_daily_summary(txn)
    monthly_df = build_monthly_summary(txn)
    overall_df = build_overall_summary(txn)

    FACT_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    fact_df.to_csv(FACT_OUTPUT_FILE, index=False)
    daily_df.to_csv(DAILY_OUTPUT_FILE, index=False)
    monthly_df.to_csv(MONTHLY_OUTPUT_FILE, index=False)
    overall_df.to_csv(OVERALL_OUTPUT_FILE, index=False)

    print("\nSummary")
    print("-" * 72)
    print(f"Revenue (tax ex) : KES {fact_df['Revenue'].sum():,.0f}")
    print(f"Profit           : KES {fact_df['Profit'].sum():,.0f}")
    print(f"Transactions     : {len(fact_df):,}")
    print(f"Daily rows       : {len(daily_df):,}")
    print(f"Monthly rows     : {len(monthly_df):,}")
    print(f"Staff rows       : {len(overall_df):,}")

    print("\nTop staff by revenue")
    for _, row in overall_df.head(10).iterrows():
        print(
            f"  #{int(row['Overall_Sales_Rank'])} "
            f"{row['Staff_Name']:<12} "
            f"Revenue KES {row['Revenue']:>12,.0f}   "
            f"Profit KES {row['Profit']:>11,.0f}"
        )

    print("\nSaved files:")
    print(f"  - {FACT_OUTPUT_FILE}")
    print(f"  - {DAILY_OUTPUT_FILE}")
    print(f"  - {MONTHLY_OUTPUT_FILE}")
    print(f"  - {OVERALL_OUTPUT_FILE}")
    print("-" * 72)


if __name__ == "__main__":
    run_staff_analysis_pos()
