"""
audit_cashier_respondio.py
==========================
DIAGNOSTIC SCRIPT — Do not add to the main pipeline.

Purpose:
    Go directly to the raw cashier .xlsm source files (before etl_local merges
    anything) and sum the cashier `Amount` column for every row where
    `Ordered Via == 'respond.io'`.

    This gives a ground-truth revenue figure from the source of record,
    for comparison against what social_sales_etl.py reports via
    `Total (Tax Ex)` from the sales line items side.

    The gap between these two numbers is your suspected ~200k+ discrepancy.

Scope:   Jan 2026 – Apr 2026
Source:  {SP_BASE}/{branch}/cashier_reports/*.xlsm

Output:
    audit_cashier_respondio_by_date.csv    ← daily detail (date, branch, amount)
    audit_cashier_respondio_summary.csv    ← monthly rollup by branch
    Console summary printed on completion

Run:
    python audit_cashier_respondio.py
"""

import re
import warnings
from datetime import date, datetime
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG  — mirrors etl_local.py paths exactly
# ══════════════════════════════════════════════════════════════════════════════

SP_BASE = Path(
    r"D:\Documents\Portal ML Analys\Portal_ML\Portal_ML_V4"
    r"\data\01_raw\sharepoint_downloads"
)

OUTPUT_DIR = Path(
    r"D:\Documents\Portal ML Analys\Portal_ML\Portal_ML_V4"
    r"\data\03_processed\audit"
)

# The merged file that social_sales_etl.py reads — used for side-by-side comparison
MERGED_SALES_FILE = Path(
    r"D:\Documents\Portal ML Analys\Portal_ML\Portal_ML_V4"
    r"\data\03_processed\pos_data\all_locations_sales_NEW.csv"
)

BRANCHES = {
    "Galleria":     "GALLERIA",
    "ABC":          "PHARMART_ABC",
    "Milele":       "NGONG_MILELE",
    "Portal 2R":    "PORTAL_2R",
    "Portal CBD":   "PORTAL_CBD",
    "Centurion 2R": "CENTURION_2R",
}

# 2026 only — Jan through Apr
AUDIT_YEAR   = 2026
AUDIT_MONTHS = {1, 2, 3, 4}   # Jan, Feb, Mar, Apr

MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,  "may": 5,  "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

CASHIER_COLUMNS = [
    "Receipt Txn No", "Amount", "Txn Costs", "Time",
    "Txn Type", "Ordered Via", "Respond Customer ID",
    "Client Name", "Phone Number", "Sales Rep",
]

_CASHIER_V1 = {
    0: "Receipt Txn No", 1: "Amount",    2: "Txn Costs",
    3: "Time",           4: "Txn Type",  5: "Ordered Via",
    6: "Client Name",    7: "Phone Number", 8: "Sales Rep",
}
_CASHIER_V2 = {
    0: "Receipt Txn No", 1: "Amount",             2: "Txn Costs",
    3: "Time",           4: "Txn Type",            5: "Ordered Via",
    6: "Respond Customer ID",
    7: "Client Name",    8: "Phone Number",        9: "Sales Rep",
}


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS  — identical logic to etl_local.py
# ══════════════════════════════════════════════════════════════════════════════

def _extract_month_year(filename: str) -> tuple[int, int]:
    """Parse month + year from cashier filename, e.g. 'Portal CBD Cashier Jan 2026'."""
    m = re.search(r'([a-zA-Z]{3,})\s+(\d{4})', filename)
    if m:
        month = MONTH_MAP.get(m.group(1)[:3].lower())
        if month:
            return month, int(m.group(2))
    now = datetime.now()
    return now.month, now.year


def _is_day_sheet(sheet_name: str, month: int, year: int) -> int | None:
    """
    Returns day number if the sheet is a valid day sheet, else None.
    Accepts: '01'-'31', and 'DD-MM-YYYY' format.
    """
    s = sheet_name.strip()
    if re.fullmatch(r'0[1-9]|[12][0-9]|3[01]', s):
        return int(s)
    m = re.fullmatch(r'(\d{2})-(\d{2})-(\d{4})', s)
    if m:
        d, mo, yr = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if mo == month and yr == year:
            return d
    return None


def _parse_one_cashier_file(path: Path) -> pd.DataFrame:
    """
    Parse a single cashier .xlsm/.xlsx file.
    Reads all valid day-sheets, handles V1 (9-col) and V2 (10-col) layouts.
    Attaches _transaction_date from the sheet name + filename month/year.
    — Mirrors _parse_one_cashier_file in etl_local.py exactly.
    """
    month, year = _extract_month_year(path.name)

    # Skip files outside our audit window
    if year != AUDIT_YEAR or month not in AUDIT_MONTHS:
        return pd.DataFrame()

    try:
        xl = pd.ExcelFile(path, engine="openpyxl")
    except Exception as e:
        print(f"      ⚠ Cannot open {path.name}: {e}")
        return pd.DataFrame()

    frames = []
    for sheet_name in xl.sheet_names:
        day = _is_day_sheet(sheet_name, month, year)
        if day is None:
            continue

        try:
            df = xl.parse(sheet_name, dtype=str)
        except Exception:
            continue

        # Normalise column names
        df.columns = df.columns.str.strip()
        df = df.loc[:, ~df.columns.duplicated()]
        df = df.dropna(how="all")
        if df.empty:
            continue

        # Handle missing column names via positional rename (V1 / V2)
        missing = [c for c in CASHIER_COLUMNS if c not in df.columns]
        if missing:
            pos_map = _CASHIER_V2 if len(df.columns) >= 10 else _CASHIER_V1
            if len(df.columns) >= len(pos_map):
                rename = {
                    df.columns[pos]: name
                    for pos, name in pos_map.items()
                    if pos < len(df.columns)
                    and df.columns[pos] not in CASHIER_COLUMNS
                    and name not in df.columns
                }
                if rename:
                    df = df.rename(columns=rename)

        # Fill still-missing columns with None
        for col in CASHIER_COLUMNS:
            if col not in df.columns:
                df[col] = None

        df = df.loc[:, ~df.columns.duplicated()]
        df = df[CASHIER_COLUMNS].copy()
        df = df.dropna(how="all")
        if df.empty:
            continue

        # Attach date from sheet position
        try:
            txn_date = date(year, month, day)
        except ValueError:
            continue  # e.g. Feb 30 — skip silently

        df["_transaction_date"] = str(txn_date)
        df["_month"]  = month
        df["_year"]   = year
        frames.append(df)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN AUDIT
# ══════════════════════════════════════════════════════════════════════════════

def run_audit():
    print("=" * 65)
    print("  CASHIER RESPOND.IO REVENUE AUDIT")
    print(f"  Scope   : Jan {AUDIT_YEAR} – Apr {AUDIT_YEAR}")
    print(f"  Source  : {SP_BASE}")
    print("=" * 65)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = []

    for branch_folder, location_label in BRANCHES.items():
        branch_dir    = SP_BASE / branch_folder
        cashier_dir   = branch_dir / "cashier_reports"

        if not cashier_dir.exists():
            print(f"\n⚠  No cashier_reports/ folder for {branch_folder} — skipping")
            continue

        files = sorted([
            f for f in cashier_dir.iterdir()
            if f.suffix.lower() in {".xlsm", ".xlsx"}
            and not f.name.startswith("~$")
            and "template" not in f.name.lower()
        ])

        if not files:
            print(f"\n⚠  No .xlsm files found in {cashier_dir}")
            continue

        print(f"\n{'─'*60}")
        print(f"📍  {location_label}  ({len(files)} file(s))")
        print(f"{'─'*60}")

        for f in files:
            month, year = _extract_month_year(f.name)
            # Pre-filter: skip files outside our audit window before opening
            if year != AUDIT_YEAR or month not in AUDIT_MONTHS:
                print(f"      ⏭  Skipping {f.name} (outside Jan–Apr {AUDIT_YEAR})")
                continue

            df = _parse_one_cashier_file(f)
            if df.empty:
                print(f"      ⚠  {f.name}: no valid day-sheets found")
                continue

            # Clean Amount
            df["Amount"] = (
                df["Amount"].astype(str)
                .str.replace(",", "", regex=False)
                .str.replace(r"[^\d\.\-]", "", regex=True)
            )
            df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce").fillna(0.0)

            # Filter to respond.io rows only — exact match, case-insensitive, stripped
            mask = df["Ordered Via"].fillna("").astype(str).str.strip().str.lower() == "respond.io"
            df_social = df[mask].copy()

            total_rows   = len(df)
            social_rows  = len(df_social)
            social_amt   = df_social["Amount"].sum()

            print(f"      📋 {f.name}: "
                  f"{total_rows:,} total rows | "
                  f"{social_rows:,} respond.io rows | "
                  f"KES {social_amt:,.0f}")

            if df_social.empty:
                continue

            df_social["Branch"]   = location_label
            df_social["Filename"] = f.name
            all_rows.append(df_social)

    if not all_rows:
        print("\n❌  No respond.io cashier rows found across any branch.")
        return

    # ── Combine & build outputs ────────────────────────────────────────────
    combined = pd.concat(all_rows, ignore_index=True)
    combined["Sale_Date"] = pd.to_datetime(combined["_transaction_date"], errors="coerce")
    combined["Month"]     = combined["Sale_Date"].dt.to_period("M").astype(str)

    # ── Daily detail ───────────────────────────────────────────────────────
    daily = (
        combined.groupby(["Sale_Date", "Branch"])
        .agg(
            Cashier_Respond_Amount = ("Amount",          "sum"),
            Transaction_Count      = ("Receipt Txn No",  "nunique"),
            Row_Count              = ("Amount",          "count"),
        )
        .reset_index()
        .sort_values(["Sale_Date", "Branch"])
    )

    daily_path = OUTPUT_DIR / "audit_cashier_respondio_by_date.csv"
    daily.to_csv(daily_path, index=False)

    # ── Monthly summary ────────────────────────────────────────────────────
    monthly = (
        combined.groupby(["Month", "Branch"])
        .agg(
            Cashier_Respond_Amount = ("Amount",          "sum"),
            Transaction_Count      = ("Receipt Txn No",  "nunique"),
            Row_Count              = ("Amount",          "count"),
        )
        .reset_index()
        .sort_values(["Month", "Branch"])
    )

    monthly_path = OUTPUT_DIR / "audit_cashier_respondio_summary.csv"
    monthly.to_csv(monthly_path, index=False)

    # ── Side-by-side comparison against all_locations_sales_NEW.csv ───────
    if MERGED_SALES_FILE.exists():
        print(f"\n📊  Loading {MERGED_SALES_FILE.name} for side-by-side comparison...")
        try:
            df_merged = pd.read_csv(MERGED_SALES_FILE, low_memory=False,
                                    dtype={"Ordered Via": str})

            mask_social = (
                df_merged["Ordered Via"]
                .fillna("").astype(str).str.strip().str.lower() == "respond.io"
            )
            df_social_merged = df_merged[mask_social].copy()
            df_social_merged["Sale_Date"] = pd.to_datetime(
                df_social_merged["Sale_Date"], errors="coerce"
            )

            # Filter to 2026 Jan-Apr
            start = pd.Timestamp(f"{AUDIT_YEAR}-01-01")
            end   = pd.Timestamp(f"{AUDIT_YEAR}-04-30")
            df_social_merged = df_social_merged[
                (df_social_merged["Sale_Date"] >= start) &
                (df_social_merged["Sale_Date"] <= end)
            ]

            df_social_merged["Month"] = df_social_merged["Sale_Date"].dt.to_period("M").astype(str)
            df_social_merged["Total (Tax Ex)"] = pd.to_numeric(
                df_social_merged["Total (Tax Ex)"], errors="coerce"
            ).fillna(0)

            merged_monthly = (
                df_social_merged.groupby("Month")
                .agg(
                    TotalTaxEx_Revenue    = ("Total (Tax Ex)",  "sum"),
                    Merged_Txn_Count      = ("Transaction ID", "nunique"),
                )
                .reset_index()
            )

            cashier_monthly_total = (
                monthly.groupby("Month")["Cashier_Respond_Amount"]
                .sum()
                .reset_index()
                .rename(columns={"Cashier_Respond_Amount": "Cashier_Amount"})
            )

            comparison = pd.merge(
                cashier_monthly_total, merged_monthly, on="Month", how="outer"
            ).fillna(0)
            comparison["Discrepancy"] = (
                comparison["TotalTaxEx_Revenue"] - comparison["Cashier_Amount"]
            )
            comparison["Discrepancy_%"] = (
                comparison["Discrepancy"] / comparison["Cashier_Amount"].replace(0, 1) * 100
            ).round(1)

            comp_path = OUTPUT_DIR / "audit_cashier_vs_merged_comparison.csv"
            comparison.to_csv(comp_path, index=False)

            print("\n📊  MONTHLY COMPARISON — Cashier Amount vs Total (Tax Ex)")
            print(f"  {'Month':<12} {'Cashier Amount':>18} {'TaxEx Revenue':>16} {'Discrepancy':>14} {'%':>8}")
            print(f"  {'─'*12} {'─'*18} {'─'*16} {'─'*14} {'─'*8}")
            for _, r in comparison.iterrows():
                flag = " ⚠" if abs(r["Discrepancy"]) > 10_000 else ""
                print(
                    f"  {r['Month']:<12} "
                    f"KES {r['Cashier_Amount']:>12,.0f}  "
                    f"KES {r['TotalTaxEx_Revenue']:>10,.0f}  "
                    f"KES {r['Discrepancy']:>10,.0f}  "
                    f"{r['Discrepancy_%']:>7.1f}%{flag}"
                )
            total_gap = comparison["Discrepancy"].sum()
            print(f"\n  {'TOTAL':>42} KES {total_gap:>10,.0f}")
            print(f"  ℹ  Positive = TaxEx reports MORE than cashier source")
            print(f"     Negative = TaxEx reports LESS than cashier source")
            print(f"\n  Comparison saved → {comp_path}")

        except Exception as e:
            print(f"   ⚠  Could not load merged file for comparison: {e}")
    else:
        print(f"\n  ℹ  Merged sales file not found at {MERGED_SALES_FILE}")
        print(f"     Run etl_local.py first if you want the side-by-side comparison.")

    # ── Console summary ────────────────────────────────────────────────────
    grand_total = combined["Amount"].sum()
    grand_txns  = combined["Receipt Txn No"].nunique()

    print(f"\n{'═'*65}")
    print("📋  CASHIER SOURCE TOTALS  (raw .xlsm, Jan–Apr 2026)")
    print(f"{'═'*65}")

    print(f"\n  ── By Branch {'─'*45}")
    branch_totals = (
        combined.groupby("Branch")
        .agg(Amount=("Amount", "sum"), Transactions=("Receipt Txn No", "nunique"))
        .sort_values("Amount", ascending=False)
    )
    for branch, row in branch_totals.iterrows():
        print(f"  {branch:<20}  KES {row['Amount']:>12,.0f}   ({row['Transactions']:,} txns)")

    print(f"\n  ── By Month {'─'*46}")
    month_totals = (
        combined.groupby("Month")
        .agg(Amount=("Amount", "sum"), Transactions=("Receipt Txn No", "nunique"))
        .sort_index()
    )
    for month, row in month_totals.iterrows():
        print(f"  {month:<12}  KES {row['Amount']:>12,.0f}   ({row['Transactions']:,} txns)")

    print(f"\n  {'─'*57}")
    print(f"  {'GRAND TOTAL':<20}  KES {grand_total:>12,.0f}   ({grand_txns:,} txns)")
    print(f"\n  📂 Daily detail  → {daily_path}")
    print(f"  📂 Monthly summary → {monthly_path}")
    print(f"{'═'*65}")


if __name__ == "__main__":
    run_audit()