"""
staff_performance.py
====================
Builds Power BI staff performance extracts across all branches.

Existing outputs are preserved and enhanced:
  - staff_sales_lineitems.csv
  - staff_transactions.csv
  - staff_daily_kpis.csv

New dashboard-ready outputs:
  - staff_daily_targets.csv
  - staff_monthly_kpis.csv
  - staff_overall_kpis.csv
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


try:
    from Portal_ML_V4.src.config.settings import BASE_DIR, PROCESSED_DATA_DIR
except ImportError:
    BASE_DIR = Path(r"D:\\Documents\\Portal ML Analys\\Portal_ML\\Portal_ML_V4")
    PROCESSED_DATA_DIR = BASE_DIR / "data" / "03_processed"


SALES_FILE = PROCESSED_DATA_DIR / "pos_data" / "all_locations_sales_NEW.csv"
OUTPUT_DIR = PROCESSED_DATA_DIR / "staff_performance"

OUTPUT_LINEITEMS = OUTPUT_DIR / "staff_sales_lineitems.csv"
OUTPUT_TRANSACTIONS = OUTPUT_DIR / "staff_transactions.csv"
OUTPUT_DAILY_KPIS = OUTPUT_DIR / "staff_daily_kpis.csv"
OUTPUT_DAILY_TARGETS = OUTPUT_DIR / "staff_daily_targets.csv"
OUTPUT_MONTHLY_KPIS = OUTPUT_DIR / "staff_monthly_kpis.csv"
OUTPUT_OVERALL_KPIS = OUTPUT_DIR / "staff_overall_kpis.csv"


SALES_REP_MAP = {
    "Cate": "Katie",
    "Emily": "Nimmoh",
}

JUNK_REP_VALUES = {"nan", "none", "n/a", "na", "null", "-", ""}

# Fill this when management confirms staff targets.
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

SALES_REP_SOURCE_COLUMNS = ["Sales Rep Name", "Sales Rep"]
SALE_DATE_SOURCE_COLUMNS = ["Date Sold", "Sale_Date"]
REVENUE_SOURCE_COLUMNS = ["Total Sales Amount", "Total (Tax Ex)"]


def clean_sales_rep(val) -> str:
    """Normalize blank or dirty sales rep names without dropping rows."""
    if val is None or pd.isna(val):
        return "Unassigned"
    clean = str(val).strip()
    if clean.lower() in JUNK_REP_VALUES:
        return "Unassigned"
    clean = clean.title()
    return SALES_REP_MAP.get(clean, clean)


def normalize_text(series: pd.Series) -> pd.Series:
    """Trim string-like fields and convert junk placeholders to nulls."""
    cleaned = series.astype("string").str.strip()
    lowered = cleaned.str.lower()
    return cleaned.mask(cleaned.isna() | lowered.isin(JUNK_REP_VALUES), pd.NA)


def coalesce_text(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Return the first populated text value from the given columns."""
    available = [col for col in columns if col in df.columns]
    if not available:
        return pd.Series(pd.NA, index=df.index, dtype="string")

    combined = pd.concat([normalize_text(df[col]).rename(col) for col in available], axis=1)
    return combined.bfill(axis=1).iloc[:, 0].astype("string")


def coalesce_numeric(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Return the first populated numeric value from the given columns."""
    result = pd.Series(np.nan, index=df.index, dtype="float64")
    for col in columns:
        if col in df.columns:
            parsed = pd.to_numeric(df[col], errors="coerce")
            result = result.fillna(parsed)
    return result


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide safely and return NaN where denominator is zero."""
    return numerator / denominator.replace({0: np.nan})


def build_transaction_keys(df: pd.DataFrame) -> pd.Series:
    """Create stable basket keys without row-by-row apply."""
    primary_key = coalesce_text(df, ["Transaction ID", "_receipt_date_key", "Receipt Txn No"])

    sale_date = coalesce_text(df, ["Sale_Date_Str", "Date Sold", "Sale_Date"]).fillna("")
    client = coalesce_text(df, ["Client Name"]).fillna("")
    phone = coalesce_text(df, ["Phone Number"]).fillna("")

    fallback_key = (
        "fallback::"
        + sale_date.astype("string")
        + "::"
        + client.astype("string")
        + "::"
        + phone.astype("string")
    )
    return primary_key.fillna(fallback_key)


def parse_mixed_sale_dates(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Return canonical sale dates and a cleaned Date Sold column without time values."""
    sale_date_base = pd.to_datetime(
        coalesce_text(df, ["Sale_Date"]),
        errors="coerce",
        format="%Y-%m-%d",
    )

    raw_date_sold = coalesce_text(df, ["Date Sold"])
    parsed_date_sold = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")

    slash_mask = raw_date_sold.str.contains("/", regex=False, na=False)
    dash_mask = raw_date_sold.str.contains("-", regex=False, na=False)

    if slash_mask.any():
        parsed_date_sold.loc[slash_mask] = pd.to_datetime(
            raw_date_sold.loc[slash_mask],
            errors="coerce",
            format="%d/%m/%Y %H:%M:%S",
        )

    if dash_mask.any():
        parsed_date_sold.loc[dash_mask] = pd.to_datetime(
            raw_date_sold.loc[dash_mask],
            errors="coerce",
            format="%d-%b-%y %I:%M:%S %p",
        )

    canonical_dates = sale_date_base.fillna(parsed_date_sold).dt.normalize()
    cleaned_date_sold = canonical_dates.dt.strftime("%Y-%m-%d")
    return canonical_dates.dt.date, cleaned_date_sold


def classify_product_line(description: pd.Series, item: pd.Series) -> pd.Series:
    """Flag whether a row looks like a real product line for qty-per-basket."""
    desc = description.fillna("").astype(str).str.strip().str.upper()
    item_clean = item.fillna("").astype(str).str.strip()

    prefix_mask = pd.Series(False, index=desc.index)
    for prefix in NON_PRODUCT_PREFIXES:
        prefix_mask = prefix_mask | desc.str.startswith(prefix)

    blank_item_mask = item_clean.isin({"", "nan", "None", "NaN"})
    return ~(prefix_mask | blank_item_mask)


def add_peer_comparison(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Add peer averages and above/below labels for the main performance metrics."""
    comparison_metrics = {
        "revenue": "sales",
        "profit": "profit",
        "transactions": "transactions",
        "avg_basket": "basket_value",
        "qty_per_basket": "qty_per_basket",
    }

    result = df.copy()
    for source_col, label in comparison_metrics.items():
        peer_avg_col = f"peer_avg_{label}"
        gap_col = f"{label}_vs_peer_avg"
        status_col = f"{label}_peer_status"

        result[peer_avg_col] = result.groupby(group_col)[source_col].transform("mean")
        result[gap_col] = result[source_col] - result[peer_avg_col]
        result[status_col] = np.where(
            result[source_col] >= result[peer_avg_col],
            "Above Average",
            "Below Average",
        )

    result["overall_peer_status"] = np.where(
        result["revenue"] >= result["peer_avg_sales"],
        "Above Average",
        "Below Average",
    )
    return result


def build_daily_targets(transactions: pd.DataFrame) -> pd.DataFrame:
    """One row per staff member per day for target vs achievement tracking."""
    daily = (
        transactions.groupby(["Sale_Date", "Sales Rep"], as_index=False)
        .agg(
            revenue=("basket_revenue", "sum"),
            cost=("basket_cost", "sum"),
            profit=("basket_profit", "sum"),
            transactions=("Transaction_Key", "size"),
            qty_sold=("units_sold", "sum"),
            product_qty_sold=("product_units_sold", "sum"),
            active_locations=("Location", lambda s: ", ".join(sorted(set(s.dropna())))),
        )
        .sort_values(["Sale_Date", "revenue"], ascending=[False, False])
    )

    daily["reporting_month"] = pd.to_datetime(daily["Sale_Date"]).dt.to_period("M").dt.to_timestamp()
    daily["avg_basket"] = safe_divide(daily["revenue"], daily["transactions"]).round(2)
    daily["qty_per_basket"] = safe_divide(daily["product_qty_sold"], daily["transactions"]).round(2)
    daily["gross_margin_pct"] = safe_divide(daily["profit"], daily["revenue"]).round(4)

    if STAFF_DAILY_SALES_TARGETS:
        daily["daily_sales_target"] = daily["Sales Rep"].map(STAFF_DAILY_SALES_TARGETS).astype(float)
    else:
        daily["daily_sales_target"] = DEFAULT_DAILY_SALES_TARGET

    daily["daily_target_gap"] = daily["revenue"] - daily["daily_sales_target"]
    daily["daily_target_achievement_pct"] = safe_divide(
        daily["revenue"], daily["daily_sales_target"]
    ).round(4)

    daily["target_status"] = np.select(
        [
            daily["daily_sales_target"].isna(),
            daily["revenue"] >= daily["daily_sales_target"],
        ],
        [
            "Target Not Set",
            "Above Target",
        ],
        default="Below Target",
    )
    return daily


def build_monthly_kpis(transactions: pd.DataFrame) -> pd.DataFrame:
    """One row per staff member per month for ranking and peer comparisons."""
    monthly = (
        transactions.groupby(["reporting_month", "Sales Rep"], as_index=False)
        .agg(
            revenue=("basket_revenue", "sum"),
            cost=("basket_cost", "sum"),
            profit=("basket_profit", "sum"),
            transactions=("Transaction_Key", "size"),
            total_customers=("Transaction_Key", "size"),
            qty_sold=("units_sold", "sum"),
            product_qty_sold=("product_units_sold", "sum"),
            active_days=("Sale_Date", "nunique"),
        )
        .sort_values(["reporting_month", "revenue"], ascending=[False, False])
    )

    monthly["avg_basket"] = safe_divide(monthly["revenue"], monthly["transactions"]).round(2)
    monthly["qty_per_basket"] = safe_divide(
        monthly["product_qty_sold"], monthly["transactions"]
    ).round(2)
    monthly["gross_margin_pct"] = safe_divide(monthly["profit"], monthly["revenue"]).round(4)
    monthly["contribution_pct_total_sales"] = safe_divide(
        monthly["revenue"], monthly.groupby("reporting_month")["revenue"].transform("sum")
    ).round(4)
    monthly["sales_rank_in_month"] = (
        monthly.groupby("reporting_month")["revenue"]
        .rank(method="dense", ascending=False)
        .astype("Int64")
    )

    return add_peer_comparison(monthly, "reporting_month")


def build_overall_kpis(transactions: pd.DataFrame) -> pd.DataFrame:
    """One row per staff member across the full dataset for leaderboard views."""
    overall = (
        transactions.groupby("Sales Rep", as_index=False)
        .agg(
            revenue=("basket_revenue", "sum"),
            cost=("basket_cost", "sum"),
            profit=("basket_profit", "sum"),
            transactions=("Transaction_Key", "size"),
            total_customers=("Transaction_Key", "size"),
            qty_sold=("units_sold", "sum"),
            product_qty_sold=("product_units_sold", "sum"),
            active_days=("Sale_Date", "nunique"),
        )
        .sort_values("revenue", ascending=False)
    )

    overall["avg_basket"] = safe_divide(overall["revenue"], overall["transactions"]).round(2)
    overall["qty_per_basket"] = safe_divide(
        overall["product_qty_sold"], overall["transactions"]
    ).round(2)
    overall["gross_margin_pct"] = safe_divide(overall["profit"], overall["revenue"]).round(4)
    overall["contribution_pct_total_sales"] = safe_divide(
        overall["revenue"], pd.Series(overall["revenue"].sum(), index=overall.index)
    ).round(4)
    overall["overall_sales_rank"] = (
        overall["revenue"].rank(method="dense", ascending=False).astype("Int64")
    )

    for source_col, label in {
        "revenue": "sales",
        "profit": "profit",
        "transactions": "transactions",
        "avg_basket": "basket_value",
        "qty_per_basket": "qty_per_basket",
    }.items():
        peer_avg = overall[source_col].mean()
        overall[f"peer_avg_{label}"] = peer_avg
        overall[f"{label}_vs_peer_avg"] = overall[source_col] - peer_avg
        overall[f"{label}_peer_status"] = np.where(
            overall[source_col] >= peer_avg,
            "Above Average",
            "Below Average",
        )

    overall["overall_peer_status"] = np.where(
        overall["revenue"] >= overall["peer_avg_sales"],
        "Above Average",
        "Below Average",
    )
    return overall


def run_staff_performance_pos():
    pipeline_start = perf_counter()
    print("=" * 72)
    print("STAFF PERFORMANCE POWER BI PIPELINE")
    print(f"Input          : {SALES_FILE}")
    print(f"Output folder  : {OUTPUT_DIR}")
    print("=" * 72)

    if not SALES_FILE.exists():
        print(f"\nInput file not found: {SALES_FILE}")
        print("Run etl_local.py first to generate all_locations_sales_NEW.csv")
        return

    print(f"\nLoading {SALES_FILE.name} ...")
    load_start = perf_counter()
    df = pd.read_csv(SALES_FILE, low_memory=False, dtype=str)
    df.columns = df.columns.str.strip()
    print(f"Raw rows: {len(df):,}")
    print(f"Load completed in     : {perf_counter() - load_start:.1f}s")

    prep_start = perf_counter()

    if any(col in df.columns for col in SALES_REP_SOURCE_COLUMNS):
        raw_sales_rep = coalesce_text(df, SALES_REP_SOURCE_COLUMNS)
        before_unassigned = raw_sales_rep.isna().sum()
        df["Sales Rep"] = raw_sales_rep.apply(clean_sales_rep)
    else:
        before_unassigned = len(df)
        df["Sales Rep"] = "Unassigned"

    df["Sales Rep Name"] = df["Sales Rep"]
    df["Sale_Date"], df["Date Sold"] = parse_mixed_sale_dates(df)
    df["reporting_month"] = pd.to_datetime(df["Sale_Date"]).dt.to_period("M").dt.to_timestamp()
    df["Transaction_Key"] = build_transaction_keys(df)

    df["Total Sales Amount"] = coalesce_numeric(df, REVENUE_SOURCE_COLUMNS).fillna(0)
    if "Total (Tax Ex)" in df.columns:
        df["Total (Tax Ex)"] = coalesce_numeric(df, ["Total (Tax Ex)"])
    df["Qty Sold"] = coalesce_numeric(df, ["Qty Sold"]).fillna(0)
    unit_cost = coalesce_numeric(df, ["Unit Cost"])
    total_cost = coalesce_numeric(df, ["Total Cost"])
    df["Cost"] = total_cost.fillna(unit_cost * df["Qty Sold"]).fillna(0)
    df["Profit"] = df["Total Sales Amount"] - df["Cost"]

    is_product_line = classify_product_line(
        df.get("Description", pd.Series(index=df.index, dtype=str)),
        df.get("Item", pd.Series(index=df.index, dtype=str)),
    )
    df["Is_Product_Line"] = is_product_line
    df["Product_Qty_Sold"] = np.where(df["Is_Product_Line"], df["Qty Sold"], 0)
    df["Gross_Margin_Pct"] = safe_divide(df["Profit"], df["Total Sales Amount"]).round(4)

    for col in [
        "Location",
        "Sales Rep ID",
        "Sales Rep Name",
        "Sales Rep",
        "Ordered Via",
        "Client Name",
        "Phone Number",
        "Respond Customer ID",
        "Transaction ID",
        "Receipt Txn No",
        "_receipt_date_key",
    ]:
        if col in df.columns:
            df[col] = normalize_text(df[col])

    unassigned_count = (df["Sales Rep"] == "Unassigned").sum()
    print("\nSales Rep cleaning")
    print(f"  Blank / null -> Unassigned : {before_unassigned:,} rows")
    print(f"  Total Unassigned kept      : {unassigned_count:,}")
    print(f"  Unique staff values        : {df['Sales Rep'].nunique():,}")
    print(f"Preparation completed in : {perf_counter() - prep_start:.1f}s")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    lineitem_cols = [
        "Sale_Date",
        "Date Sold",
        "reporting_month",
        "Location",
        "Department",
        "Category",
        "Transaction_Key",
        "Transaction ID",
        "Receipt Txn No",
        "_receipt_date_key",
        "Sales Rep ID",
        "Sales Rep Name",
        "Sales Rep",
        "Ordered Via",
        "Client Name",
        "Phone Number",
        "Respond Customer ID",
        "Item",
        "Description",
        "On Hand",
        "Unit Cost",
        "Last Sold",
        "Qty Sold",
        "Product_Qty_Sold",
        "Total Sales Amount",
        "Total (Tax Ex)",
        "Tax Amount",
        "Total Cost",
        "Cost",
        "Profit",
        "Gross_Margin_Pct",
        "Is_Product_Line",
    ]
    lineitem_cols = [col for col in lineitem_cols if col in df.columns]
    lineitem_start = perf_counter()
    lineitems = df[lineitem_cols].copy()
    lineitems.to_csv(OUTPUT_LINEITEMS, index=False)
    print(f"\nSaved line items     : {len(lineitems):,} rows -> {OUTPUT_LINEITEMS.name}")
    print(f"Line items completed in : {perf_counter() - lineitem_start:.1f}s")

    print("\nBuilding transactions ...")
    transactions_start = perf_counter()
    transaction_agg = {
        "Sale_Date": ("Sale_Date", "min"),
        "reporting_month": ("reporting_month", "min"),
        "Sales_Rep": ("Sales Rep", "first"),
        "basket_revenue": ("Total Sales Amount", "sum"),
        "basket_cost": ("Cost", "sum"),
        "basket_profit": ("Profit", "sum"),
        "units_sold": ("Qty Sold", "sum"),
        "product_units_sold": ("Product_Qty_Sold", "sum"),
        "line_item_count": ("Transaction_Key", "size"),
    }
    for output_col, source_col in {
        "Location": "Location",
        "Sales_Rep_ID": "Sales Rep ID",
        "Ordered_Via": "Ordered Via",
        "Client_Name": "Client Name",
        "Phone_Number": "Phone Number",
        "Respond_Customer_ID": "Respond Customer ID",
        "Transaction_ID": "Transaction ID",
        "Receipt_Txn_No": "Receipt Txn No",
    }.items():
        if source_col in df.columns:
            transaction_agg[output_col] = (source_col, "first")

    transactions = (
        df.groupby("Transaction_Key", as_index=False, sort=False)
        .agg(**transaction_agg)
        .rename(columns={"Sales_Rep": "Sales Rep", "Sales_Rep_ID": "Sales Rep ID"})
        .sort_values(["Sale_Date", "basket_revenue"], ascending=[False, False])
    )
    transactions["avg_item_value"] = safe_divide(
        transactions["basket_revenue"], transactions["line_item_count"]
    ).round(2)
    transactions["gross_margin_pct"] = safe_divide(
        transactions["basket_profit"], transactions["basket_revenue"]
    ).round(4)
    transactions.to_csv(OUTPUT_TRANSACTIONS, index=False)
    print(f"Saved transactions   : {len(transactions):,} rows -> {OUTPUT_TRANSACTIONS.name}")
    print(f"Transactions completed in: {perf_counter() - transactions_start:.1f}s")

    daily_start = perf_counter()
    daily_group_map = [
        ("Sale_Date", "Sale_Date"),
        ("Location", "Location"),
        ("Sales Rep", "Sales Rep"),
        ("Ordered_Via", "Ordered Via"),
    ]
    daily_group_cols = [source for source, _ in daily_group_map if source in transactions.columns]
    daily_kpis = (
        transactions.groupby(daily_group_cols, dropna=False)
        .agg(
            revenue=("basket_revenue", "sum"),
            cost=("basket_cost", "sum"),
            profit=("basket_profit", "sum"),
            units_sold=("units_sold", "sum"),
            product_units_sold=("product_units_sold", "sum"),
            line_items=("line_item_count", "sum"),
            transactions=("Transaction_Key", "size"),
        )
        .reset_index()
    )
    daily_kpis = daily_kpis.rename(
        columns={source: target for source, target in daily_group_map if source != target}
    )
    daily_kpis["avg_basket"] = safe_divide(daily_kpis["revenue"], daily_kpis["transactions"]).round(2)
    daily_kpis["qty_per_basket"] = safe_divide(
        daily_kpis["product_units_sold"], daily_kpis["transactions"]
    ).round(2)
    daily_kpis["gross_margin_pct"] = safe_divide(
        daily_kpis["profit"], daily_kpis["revenue"]
    ).round(4)
    daily_kpis.to_csv(OUTPUT_DAILY_KPIS, index=False)
    print(f"Saved daily KPIs     : {len(daily_kpis):,} rows -> {OUTPUT_DAILY_KPIS.name}")
    print(f"Daily KPIs completed in  : {perf_counter() - daily_start:.1f}s")

    dashboard_start = perf_counter()
    daily_targets = build_daily_targets(transactions)
    monthly_kpis = build_monthly_kpis(transactions)
    overall_kpis = build_overall_kpis(transactions)

    daily_targets.to_csv(OUTPUT_DAILY_TARGETS, index=False)
    monthly_kpis.to_csv(OUTPUT_MONTHLY_KPIS, index=False)
    overall_kpis.to_csv(OUTPUT_OVERALL_KPIS, index=False)

    print(f"Saved daily targets  : {len(daily_targets):,} rows -> {OUTPUT_DAILY_TARGETS.name}")
    print(f"Saved monthly KPIs   : {len(monthly_kpis):,} rows -> {OUTPUT_MONTHLY_KPIS.name}")
    print(f"Saved overall KPIs   : {len(overall_kpis):,} rows -> {OUTPUT_OVERALL_KPIS.name}")

    total_rev = df["Total Sales Amount"].sum()
    total_profit = df["Profit"].sum()
    total_units = df["Qty Sold"].sum()
    total_txns = transactions["Transaction_Key"].nunique()

    print("\nSummary")
    print("-" * 72)
    print(f"Revenue (gross)  : KES {total_rev:,.0f}")
    print(f"Profit           : KES {total_profit:,.0f}")
    print(f"Units sold       : {total_units:,.0f}")
    print(f"Transactions     : {total_txns:,}")

    print("\nTop staff by revenue")
    for _, row in overall_kpis.head(10).iterrows():
        print(
            f"  #{int(row['overall_sales_rank'])} "
            f"{row['Sales Rep']:<20} "
            f"KES {row['revenue']:>12,.0f}   "
            f"Profit KES {row['profit']:>11,.0f}"
        )

    print("\nPower BI notes")
    print("  1. Use Transaction_Key as the preferred basket key for distinct counts.")
    print("  2. staff_monthly_kpis.csv answers ranking, contribution, basket value, and peer average.")
    print("  3. staff_daily_targets.csv answers daily target vs achievement once targets are filled.")
    print("  4. staff_overall_kpis.csv gives the top-level leaderboard by rep.")
    print("-" * 72)
    print(f"Dashboard tables completed in: {perf_counter() - dashboard_start:.1f}s")
    print(f"Pipeline completed in : {perf_counter() - pipeline_start:.1f}s")


if __name__ == "__main__":
    run_staff_performance_pos()
