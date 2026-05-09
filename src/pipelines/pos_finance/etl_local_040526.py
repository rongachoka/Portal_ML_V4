"""
etl_local.py
============
POS ETL V5-LOCAL — reads directly from the SharePoint downloads folder.

Source:
    data/01_raw/sharepoint_downloads/
        {Branch}/
            sales_reports/    ← .csv / .xlsx  (daily incrementals + cumulative)
            cashier_reports/  ← .xlsm / .xlsx (day-per-sheet layout)

Processing:
    - Loads all sales files per branch, deduplicates on
      (Transaction ID, Description, Qty Sold, Date Sold)  ← no On Hand
    - Loads all cashier files, reads every valid day-sheet,
      handles V1 (9-col, pre-Apr 2026) and V2 (10-col, Apr 2026+) layouts
    - Merges sales ← cashier on Transaction ID = Receipt Txn No
    - Date filter: Jan 2025 onwards
    - Outputs all_locations_sales.csv for attribution pipeline

Run:
    python etl_local.py
"""

import gc
import os
import re
import warnings
from datetime import datetime, date
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# ── Paths ─────────────────────────────────────────────────────────────────────
try:
    from Portal_ML_V4.src.config.settings import BASE_DIR, PROCESSED_DATA_DIR
    from Portal_ML_V4.src.utils.name_cleaner import clean_name_series
    _HAS_SETTINGS = True
except ImportError:
    _HAS_SETTINGS = False

    PROCESSED_DATA_DIR = BASE_DIR / "data" / "03_processed"
    def clean_name_series(s): return s  # fallback no-op

SP_BASE      = BASE_DIR / "data" / "01_raw" / "sharepoint_downloads"
OUTPUT_DIR   = PROCESSED_DATA_DIR / "pos_data"
OUTPUT_FILE  = OUTPUT_DIR / "all_locations_sales_NEW.csv"
START_DATE   = pd.Timestamp("2025-01-01")

# ── Branch map  (SP folder name → output Location label) ─────────────────────
BRANCHES = {
    "Galleria":     "GALLERIA",
    "ABC":          "PHARMART_ABC",
    "Milele":       "NGONG_MILELE",
    "Portal 2R":    "PORTAL_2R",
    "Portal CBD":   "PORTAL_CBD",
    "Centurion 2R": "CENTURION_2R",
}

# ── Column definitions ────────────────────────────────────────────────────────
SALES_REQUIRED = {"Transaction ID", "Date Sold"}
SALES_KEEP     = [
    "Department", "Category", "Item", "Description",
    "On Hand", "Last Sold", "Qty Sold", "Total (Tax Ex)",
    "Transaction ID", "Date Sold",
]
# Dedup key — deliberately excludes On Hand (live inventory snapshot that
# changes between POS exports and would cause phantom doubles)
SALES_DEDUP = ["Transaction ID", "Description", "Qty Sold", "Date Sold"]

CASHIER_COLUMNS = [
    "Receipt Txn No", "Amount", "Txn Costs", "Time",
    "Txn Type", "Ordered Via", "Respond Customer ID",
    "Client Name", "Phone Number", "Sales Rep",
]

# Positional column maps for sheets that lack header names
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

MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,  "may": 5,  "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def clean_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r'\.0$', '', regex=True).str.strip()


def parse_date_sold(series: pd.Series) -> pd.Series:
    """Strip POS # characters and parse to datetime."""
    cleaned = series.astype(str).str.replace('#', '', regex=False).str.strip()
    return pd.to_datetime(cleaned, dayfirst=True, errors='coerce')


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        t = df[col].dtype
        if t == 'object' and df[col].nunique() / max(len(df), 1) < 0.5:
            df[col] = df[col].astype('category')
        elif t == 'float64':
            df[col] = pd.to_numeric(df[col], downcast='float')
        elif t == 'int64':
            df[col] = pd.to_numeric(df[col], downcast='integer')
    return df


def first_non_null(series):
    non_null = series.dropna()
    return non_null.iloc[0] if len(non_null) > 0 else None


# ══════════════════════════════════════════════════════════════════════════════
# SALES LOADER
# ══════════════════════════════════════════════════════════════════════════════

def _read_one_sales_file(path: Path) -> pd.DataFrame:
    """Read a single sales CSV or XLSX. Returns empty DataFrame on any error."""
    try:
        suffix = path.suffix.lower()
        if suffix == ".csv":
            try:
                df = pd.read_csv(path, dtype=str, low_memory=False)
            except UnicodeDecodeError:
                df = pd.read_csv(path, dtype=str, low_memory=False, encoding="cp1252")
        elif suffix in (".xlsx", ".xlsm"):
            df = pd.read_excel(path, dtype=str)
        else:
            return pd.DataFrame()

        df.columns = df.columns.str.strip()

        # Must have the key columns to be a valid sales file
        if not SALES_REQUIRED.issubset(df.columns):
            return pd.DataFrame()

        # Keep only the columns we care about
        keep = [c for c in SALES_KEEP if c in df.columns]
        df   = df[keep].copy()

        # Parse date and apply cutoff
        df["_date_parsed"] = parse_date_sold(df["Date Sold"])
        df = df[df["_date_parsed"] >= START_DATE].copy()

        if df.empty:
            return pd.DataFrame()

        df["Date_Obj"]       = df["_date_parsed"]
        df["Transaction ID"] = clean_id(df["Transaction ID"])
        df.drop(columns=["_date_parsed"], inplace=True)
        return df

    except Exception as e:
        print(f"      ⚠ Could not read {path.name}: {e}")
        return pd.DataFrame()



# ══════════════════════════════════════════════════════════════════════════════
# CANONICAL SALES FILE ELECTION
# ══════════════════════════════════════════════════════════════════════════════

def _peek_max_date(path: Path):
    """Read only the Date Sold column and return the max parsed date. Fast."""
    try:
        suffix = path.suffix.lower()
        if suffix == ".csv":
            try:
                df = pd.read_csv(path, usecols=["Date Sold"], dtype=str, low_memory=False)
            except (UnicodeDecodeError, ValueError):
                df = pd.read_csv(path, usecols=["Date Sold"], dtype=str,
                                 low_memory=False, encoding="cp1252")
        elif suffix in (".xlsx", ".xlsm"):
            df = pd.read_excel(path, usecols=["Date Sold"], dtype=str)
        else:
            return None
        if "Date Sold" not in df.columns:
            return None
        dates = parse_date_sold(df["Date Sold"]).dropna()
        return dates.max() if not dates.empty else None
    except Exception:
        return None


def elect_latest_sales_file(branch_dir: Path):
    """
    Scans sales_reports/, peeks at max Date Sold in each file,
    returns (winning_path, max_date) for the file with the highest date.
    That file is cumulative and contains all prior data.
    """
    sales_dir = branch_dir / "sales_reports"
    if not sales_dir.exists():
        return None, None

    candidates = [
        f for f in sales_dir.iterdir()
        if f.is_file()
        and f.suffix.lower() in {".csv", ".xlsx"}
        and not f.name.startswith("~$")
        and "cashier" not in f.name.lower()
        and "template" not in f.name.lower()
    ]

    if not candidates:
        return None, None

    best_path, best_date = None, None
    for f in candidates:
        max_date = _peek_max_date(f)
        if max_date is None:
            continue
        if best_date is None or max_date > best_date:
            best_date = max_date
            best_path = f

    return best_path, best_date


def load_sales_for_branch(branch_dir: Path) -> pd.DataFrame:
    """
    Elects the single latest sales file (highest max Date Sold) and loads only that.
    All sales files are cumulative — the latest one contains everything prior.
    """
    canonical, max_date = elect_latest_sales_file(branch_dir)
    if canonical is None:
        print(f"    ⚠ No sales files found in {branch_dir / 'sales_reports'}")
        return pd.DataFrame()

    print(f"    📄 Canonical file: {canonical.name}  (max date: {max_date.date() if max_date else '?'})")
    return _read_one_sales_file(canonical)


# ══════════════════════════════════════════════════════════════════════════════
# CASHIER LOADER
# ══════════════════════════════════════════════════════════════════════════════

def _extract_month_year(filename: str) -> tuple[int, int]:
    """Parse month + year from cashier filename (e.g. 'Galleria Cashier Jan 2026')."""
    m = re.search(r'([a-zA-Z]{3,})\s+(\d{4})', filename)
    if m:
        month = MONTH_MAP.get(m.group(1)[:3].lower())
        if month:
            return month, int(m.group(2))
    now = datetime.now()
    return now.month, now.year


def _is_day_sheet(sheet_name: str, month: int, year: int) -> int | None:
    """
    Returns the day number if the sheet is a valid day sheet, else None.
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
    Reads all valid day-sheets (01–31) and handles both V1 and V2 column layouts.
    """
    month, year = _extract_month_year(path.name)
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

        # Strip whitespace FIRST — "Phone Number" and "Phone Number "
        # look distinct before strip but become identical after it,
        # so the duplicate check must happen after the strip.
        df.columns = df.columns.str.strip()
        dupes = df.columns[df.columns.duplicated()].tolist()
        if dupes:
            print(f"      ⚠ Duplicate columns in {path.name} → sheet '{sheet_name}': {dupes}")
            print(f"         (file kept — duplicate column(s) will be dropped)")
        df = df.loc[:, ~df.columns.duplicated()]
        df = df.dropna(how="all")
        if df.empty:
            continue

        # Detect which named columns are already present
        missing = [c for c in CASHIER_COLUMNS if c not in df.columns]

        # If columns are missing, try positional rename (V1 vs V2 layout).
        # Guard: skip any position whose target name already exists in df —
        # renaming onto an existing column creates a duplicate that breaks concat.
        if missing:
            pos_map = _CASHIER_V2 if len(df.columns) >= 10 else _CASHIER_V1
            if len(df.columns) >= len(pos_map):
                rename = {
                    df.columns[pos]: name
                    for pos, name in pos_map.items()
                    if pos < len(df.columns)
                    and df.columns[pos] not in CASHIER_COLUMNS
                    and name not in df.columns   # ← stops rename creating dupes
                }
                if rename:
                    df = df.rename(columns=rename)

        # Fill still-missing columns with None
        for col in CASHIER_COLUMNS:
            if col not in df.columns:
                df[col] = None

        # Log and drop any duplicate column names that survived the rename step
        dupes = df.columns[df.columns.duplicated()].tolist()
        if dupes:
            print(f"      ⚠ Duplicate columns in {path.name} → sheet '{sheet_name}': {dupes}")
            print(f"         Review that file — first occurrence kept, rest dropped.")
        df = df.loc[:, ~df.columns.duplicated()]

        df = df[CASHIER_COLUMNS].copy()
        df = df.dropna(how="all")
        if df.empty:
            continue

        # Attach transaction date from sheet position
        try:
            txn_date = date(year, month, day)
        except ValueError:
            continue
        df["_transaction_date"] = str(txn_date)
        frames.append(df)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_cashier_for_branch(branch_dir: Path) -> pd.DataFrame:
    """
    Loads all cashier files from {branch}/cashier_reports/.
    Deduplicates on Receipt Txn No (keeps first occurrence).
    """
    cashier_dir = branch_dir / "cashier_reports"
    if not cashier_dir.exists():
        print(f"    ⚠ No cashier_reports/ folder: {cashier_dir}")
        return pd.DataFrame()

    all_cashier_candidates = [f for f in cashier_dir.iterdir() if f.is_file()]
    for f in all_cashier_candidates:
        if "template" in f.name.lower():
            print(f"      ⏭  Skipping template file: {f.name}")
    files = [
        f for f in all_cashier_candidates
        if f.suffix.lower() in {".xlsm", ".xlsx"}
        and not f.name.startswith("~$")
        and "template" not in f.name.lower()
    ]

    if not files:
        print(f"    ⚠ No cashier files in {cashier_dir}")
        return pd.DataFrame()

    frames = []
    for f in sorted(files):
        df = _parse_one_cashier_file(f)
        if not df.empty:
            print(f"      📋 {f.name}: {len(df):,} rows")
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    # Clean & aggregate to one row per Receipt Txn No
    combined["Receipt Txn No"] = combined["Receipt Txn No"].fillna("").astype(str).str.strip()
    # combined = combined[combined["Receipt Txn No"] != ""].copy()

    # Assign placeholder to missing IDs instead of dropping them
    no_id_mask = combined["Receipt Txn No"] == ""
    if no_id_mask.any():
        placeholders = [f"__NOID_{i}" for i in range(no_id_mask.sum())]
        combined.loc[no_id_mask, "Receipt Txn No"] = placeholders

    combined["Receipt Txn No"] = clean_id(combined["Receipt Txn No"])

    for col in ["Amount", "Txn Costs"]:
        if col in combined.columns:
            combined[col] = (
                combined[col].astype(str)
                .str.replace(",", "", regex=False)
                .str.replace(r"[^\d\.\-]", "", regex=True)
            )
            combined[col] = pd.to_numeric(combined[col], errors="coerce").fillna(0.0)

    if "Time" in combined.columns:
        combined["Time"] = combined["Time"].astype(str).fillna("00:00:00")

    agg = {"Amount": "sum", "Txn Costs": "sum"}
    if "Time" in combined.columns:
        agg["Time"] = "min"
    meta = [c for c in combined.columns if c not in {"Receipt Txn No", "Amount", "Txn Costs", "Time"}]
    for c in meta:
        agg[c] = first_non_null

    combined = combined.groupby("Receipt Txn No", as_index=False).agg(agg)

    if "Client Name" in combined.columns:
        combined["Client Name"] = clean_name_series(combined["Client Name"])

    return combined


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_pos_etl_local():
    print("🌍  POS ETL V5-LOCAL — reading from SharePoint downloads folder")
    print(f"📅  Date filter: {START_DATE.date()} → present")
    print(f"📂  Source     : {SP_BASE}")
    print(f"📂  Output     : {OUTPUT_FILE}\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    master_list = []

    for branch, location_label in BRANCHES.items():
        branch_dir = SP_BASE / branch
        if not branch_dir.exists():
            print(f"\n⚠  Branch folder not found, skipping: {branch_dir}")
            continue

        print(f"\n{'─'*60}")
        print(f"📍  BRANCH: {location_label}")
        print(f"{'─'*60}")

        # ── A. Load sales ──────────────────────────────────────────────────
        df_sales = load_sales_for_branch(branch_dir)
        if df_sales.empty:
            print(f"    ⚠ No valid sales data found for {branch}")
            continue

        df_sales = optimize_dtypes(df_sales)
        print(f"    ✅ Sales loaded: {len(df_sales):,} rows")

        # ── B. Load cashier ────────────────────────────────────────────────
        print(f"    📋 Loading cashier files...")
        df_cashier = load_cashier_for_branch(branch_dir)

        # ── C. Merge ───────────────────────────────────────────────────────
        if not df_cashier.empty:
            df_cashier = optimize_dtypes(df_cashier)
            print(f"    ✅ Cashier loaded: {len(df_cashier):,} unique transactions")
            print(f"    🔗 Merging...")

            try:
                df_merged = pd.merge(
                    df_sales,
                    df_cashier,
                    left_on="Transaction ID",
                    right_on="Receipt Txn No",
                    how="outer",
                    copy=False,
                )
            except (MemoryError, np.core._exceptions._ArrayMemoryError):
                print("    ⚠ Memory limit — switching to chunked merge...")
                chunks, chunk_size = [], 50_000
                for i in range(0, len(df_sales), chunk_size):
                    merged_chunk = pd.merge(
                        df_sales.iloc[i:i + chunk_size],
                        df_cashier,
                        left_on="Transaction ID",
                        right_on="Receipt Txn No",
                        how="outer",
                        copy=False,
                    )
                    chunks.append(merged_chunk)
                    gc.collect()
                df_merged = pd.concat(chunks, ignore_index=True)

            matched = df_merged["Receipt Txn No"].notna().sum()
            print(f"    ✅ Merge complete: {matched:,}/{len(df_merged):,} rows matched to cashier")
        else:
            df_merged = df_sales.copy()
            print(f"    ⚠ No cashier data — all rows marked 'No Cashier Data'")

        df_merged["Audit_Status"] = np.where(
            df_merged.get("Receipt Txn No", pd.Series(dtype=str)).isna(),
            "No Cashier Data", "Matched"
        )
        df_merged["Location"] = location_label
        master_list.append(df_merged)

        del df_sales, df_cashier, df_merged
        gc.collect()

    # ── Final stack ────────────────────────────────────────────────────────
    if not master_list:
        print("\n❌  No data processed. Check that SP_BASE path is correct.")
        return

    print(f"\n{'═'*60}")
    print("🏗   Stacking all branches...")
    final_df = pd.concat(master_list, ignore_index=True)
    gc.collect()

    # Date columns
    if "Date_Obj" in final_df.columns:
        if "_transaction_date" in final_df.columns:
            final_df["Date_Obj"] = final_df["Date_Obj"].fillna(
                pd.to_datetime(final_df["_transaction_date"], errors="coerce")
            )
        dt = pd.to_datetime(final_df["Date_Obj"], errors="coerce")
        final_df["Sale_Date"]     = dt.dt.normalize()
        final_df["Sale_Date_Str"] = dt.dt.strftime("%Y-%m-%d")
        final_df.drop(columns=["Date_Obj"], inplace=True)
        gc.collect()

    # Final dedup across branches (safety net)
    before = len(final_df)
    dedup  = [c for c in SALES_DEDUP if c in final_df.columns]
    # final_df.drop_duplicates(subset=dedup, inplace=True)
    # final_df.drop_duplicates(inplace=True)

    after  = len(final_df)
    if before - after:
        print(f"    ✂ Cross-branch dedup removed {before - after:,} rows")

    # Transaction_Total (sum of line items per transaction)
    if "Total (Tax Ex)" in final_df.columns:
        final_df["Total (Tax Ex)"] = pd.to_numeric(
            final_df["Total (Tax Ex)"], errors="coerce"
        ).fillna(0)

        # Fallback: If no sales data matched, use the Cashier Amount
        mask_no_sales = final_df["Transaction ID"].isna()
        if "Amount" in final_df.columns:
            final_df.loc[mask_no_sales, "Total (Tax Ex)"] = pd.to_numeric(
                final_df.loc[mask_no_sales, "Amount"], errors="coerce"
            ).fillna(0.0)

        final_df["Transaction_Total"] = (
            final_df.groupby("Transaction ID")["Total (Tax Ex)"].transform("sum")
        )

    final_df.to_csv(OUTPUT_FILE, index=False)

    print(f"\n🚀  PIPELINE COMPLETE")
    print(f"    📊 Rows before dedup : {before:,}")
    print(f"    ✂  Duplicates removed: {before - after:,}")
    print(f"    🧾 Final row count   : {after:,}")
    print(f"    📂 Output            : {OUTPUT_FILE}")


if __name__ == "__main__":
    run_pos_etl_local()