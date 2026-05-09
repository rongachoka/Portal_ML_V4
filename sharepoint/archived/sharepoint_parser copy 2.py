"""
sharepoint_parser.py
====================
Reads downloaded sales, cashier, and qty list files, applies deduplication,
and loads clean data into:
    stg_sales_reports / stg_cashier_reports / stg_qty_list

Called by sharepoint_downloader.py after files are downloaded.

Sales logic (V2 — canonical election + watermark):
  - Peeks inside every sales file to find its actual max Date Sold
  - Elects the file with the highest max Date Sold as the canonical source
  - Loads the canonical file in full on first run
  - On subsequent runs: canonical is only re-loaded if its max date is AFTER
    the stored branch watermark (i.e. it contains genuinely new data)
  - Incremental files are only loaded if their max date is AFTER the watermark
  - Files fully covered by the watermark are skipped without being read
  - After a successful load the branch watermark is updated

Cashier logic (unchanged):
  - Reads sheets 01-31 only (skips Summary, Sheet1 etc.)
  - Derives transaction_date from sheet number + month/year in filename
  - Time kept as plain string — not parsed (ambiguous formats e.g. '12.5')
  - Deduplication on cashier data happens downstream in load_to_postgres.py

Qty list logic (unchanged):
  - Detects files named 'Item Quantity List' or 'QTY LIST' (case-insensitive)
  - Strips junk header rows (lines starting with ';') before parsing
  - Date parsed from filename: full date > day-only > lastModifiedDateTime
  - Day-only filenames flagged in ingestion_files.notes for manual review
  - Loads into stg_qty_list; fact_inventory_snapshot built by load_to_postgres.py
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import re
from datetime import date, datetime, timezone, time as dt_time
from pathlib import Path
from typing import Any

import pandas as pd

from Portal_ML_V4.sharepoint.db import (
    bulk_insert,
    bulk_insert_safe,
    execute,
    get_connection,
    insert_returning_id,
    fetchone,
)

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

logger = logging.getLogger(__name__)

# ── CUTOFF: if latest transaction on a date is before this, day is incomplete ─
_DAY_COMPLETE_HOUR = 16   # 4pm — all branches close between 7pm and 8:30pm


# ── Column definitions ────────────────────────────────────────────────────────

SALES_COLUMNS = [
    "Department", "Category", "Item", "Description",
    "On Hand", "Last Sold", "Qty Sold", "Total (Tax Ex)",
    "Transaction ID", "Date Sold",
]

CASHIER_COLUMNS = [
    "Receipt Txn No", "Amount", "Txn Costs", "Time",
    "Txn Type", "Ordered Via", "Respond Customer ID", 
    "Client Name", "Phone Number", "Sales Rep",
]

# Composite dedup key — same item on same transaction with same qty/stock = duplicate
SALES_DEDUP_COLUMNS = [
    "Transaction ID", "Item", "On Hand", "Qty Sold", "Date Sold",
]

DAY_SHEETS = [f"{i:02d}" for i in range(1, 32)]

MONTH_MAP = {
    "jan": 1,  "feb": 2,  "mar": 3,  "apr": 4,
    "may": 5,  "jun": 6,  "jul": 7,  "aug": 8,
    "sep": 9,  "oct": 10, "nov": 11, "dec": 12,
}


# ══════════════════════════════════════════════════════════════════════════════
# FILE HELPERS
# ══════════════════════════════════════════════════════════════════════════════


def compute_file_hash(path: Path) -> str:
    """
    SHA-256 of file bytes. Reads in 1MB chunks.
    For files > 100MB uses filename + size + mtime to avoid MemoryError.
    """
    try:
        file_size = os.path.getsize(path)

        if file_size > 100 * 1024 * 1024:  # > 100MB
            stat = os.stat(path)
            identifier = f"{path.name}_{stat.st_size}_{stat.st_mtime}"
            return hashlib.sha256(identifier.encode()).hexdigest() + "_partial"

        sha = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                sha.update(chunk)
        return sha.hexdigest()

    except MemoryError:
        stat = os.stat(path)
        identifier = f"{path.name}_{stat.st_size}_{stat.st_mtime}"
        return hashlib.sha256(identifier.encode()).hexdigest() + "_fallback"

    except Exception as exc:
        logger.warning(f"Hash failed for {path.name}: {exc}")
        stat = os.stat(path)
        identifier = f"{path.name}_{stat.st_size}_{stat.st_mtime}"
        return hashlib.sha256(identifier.encode()).hexdigest() + "_error"
    

def is_file_already_loaded(file_hash: str) -> bool:
    """
    Returns True if a file with this exact hash was previously loaded
    successfully (status = 'loaded' in ingestion_files).
 
    This prevents re-processing cumulative files (e.g. 'ABC Jan 2023-Feb 2026
    Sales.xlsx') on every pipeline run, which was the root cause of the 17M
    duplicate rows found in April 2026.
    """
    try:
        row = fetchone(
            """
            SELECT id FROM ingestion_files
            WHERE file_hash = %s
              AND status    = 'loaded'
            LIMIT 1
            """,
            (file_hash,),
        )
        return row is not None
    except Exception as exc:
        # If the check itself fails, err on the side of processing the file
        # rather than silently skipping it.
        logger.warning(f"[HashCheck] Could not verify hash {file_hash[:12]}…: {exc}")
        return False


def count_rows(path: Path) -> int:
    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            try:
                df = pd.read_csv(path, usecols=[0], dtype=str)
            except UnicodeDecodeError:
                df = pd.read_csv(path, usecols=[0], dtype=str, encoding="cp1252")
        elif suffix in (".xlsx", ".xlsm"):
            df = pd.read_excel(path, usecols=[0], dtype=str)
        else:
            return 0
        return len(df.dropna())
    except Exception as exc:
        logger.warning(f"Could not count rows for {path.name}: {exc}")
        return 0


def get_max_date_sold(path: Path) -> date | None:
    """
    Peek inside a sales file and return the maximum Date Sold value.
    Peeks headers first, then reads only the matched Date Sold column — avoids
    loading the full file into memory while tolerating header whitespace.
    Returns None if the column is missing or all values are unparseable.
    """
    try:
        suffix = path.suffix.lower()
        if suffix == ".csv":
            try:
                header_df = pd.read_csv(path, dtype=str, nrows=0, low_memory=False)
            except UnicodeDecodeError:
                header_df = pd.read_csv(
                    path, dtype=str, nrows=0, low_memory=False, encoding="cp1252"
                )

            date_sold_col = next(
                (
                    col for col in header_df.columns
                    if isinstance(col, str) and col.strip() == "Date Sold"
                ),
                None,
            )
            if date_sold_col is None:
                return None

            try:
                df = pd.read_csv(
                    path, usecols=[date_sold_col], dtype=str, low_memory=False
                )
            except UnicodeDecodeError:
                df = pd.read_csv(
                    path, usecols=[date_sold_col], dtype=str,
                    low_memory=False, encoding="cp1252"
                )
        elif suffix in (".xlsx", ".xlsm"):
            header_df = pd.read_excel(path, dtype=str, nrows=0)
            date_sold_col = next(
                (
                    col for col in header_df.columns
                    if isinstance(col, str) and col.strip() == "Date Sold"
                ),
                None,
            )
            if date_sold_col is None:
                return None

            df = pd.read_excel(path, usecols=[date_sold_col], dtype=str)
        else:
            return None

        df = _strip_columns(df)
        if "Date Sold" not in df.columns:
            return None

        dates = df["Date Sold"].apply(_clean_date_sold).dropna()
        return dates.max() if not dates.empty else None

    except Exception as exc:
        logger.warning(f"Could not read max Date Sold from {path.name}: {exc}")
        return None


def classify_sales_file(filename: str) -> str:
    """
    Classify a sales file as 'historical' or 'incremental' based on filename.
    Historical: year range in name e.g. 'Jan 2023-Feb 2026', '2023-2026'
    Incremental: single date e.g. '150326', '14 Mar', 'sales 12 March'
    """
    name = filename.lower()
    if re.search(r'\d{4}\s*[-–]\s*\d{4}', name):
        return "historical"
    month_pattern = r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)'
    if re.search(month_pattern + r'.+\d{4}.+' + month_pattern, name):
        return "historical"
    return "incremental"


# ══════════════════════════════════════════════════════════════════════════════
# BRANCH WATERMARK
# ══════════════════════════════════════════════════════════════════════════════

# Maps SharePoint branch names → fact table location labels
_BRANCH_TO_LOCATION = {
    "ABC":          "PHARMART_ABC",
    "Centurion 2R": "CENTURION_2R",
    "Galleria":     "GALLERIA",
    "Milele":       "NGONG_MILELE",
    "Portal 2R":    "PORTAL_2R",
    "Portal CBD":   "PORTAL_CBD",
}


# def get_branch_watermark(branch: str) -> date | None:
#     """
#     Return the latest Date Sold already loaded for this branch.
#     Returns None if no watermark exists (first run).
#     """
#     location = _BRANCH_TO_LOCATION.get(branch, branch)
#     try:
#         with get_connection() as conn:
#             cur = conn.cursor()
#             cur.execute(
#                 "SELECT max_date_loaded FROM branch_watermarks WHERE branch = %s",
#                 (location,),
#             )
#             row = cur.fetchone()
#             cur.close()
#             return row[0] if row else None
#     except Exception as exc:
#         logger.warning(f"Could not read watermark for {branch}: {exc}")
#         return None

# def get_branch_watermark(branch: str) -> date | None:
#     location = _BRANCH_TO_LOCATION.get(branch, branch)
#     try:
#         row = fetchone(
#             "SELECT max_date_loaded FROM branch_watermarks WHERE branch = %s",
#             (location,)
#         )
#         return row["max_date_loaded"] if row else None
#     except Exception as exc:
#         logger.warning(f"Could not read watermark for {branch}: {exc}")
#         return None

def get_branch_watermark(branch: str) -> tuple[date | None, datetime | None]:
    """
    Return the watermark for this branch as (max_date_loaded, max_datetime_loaded).
 
    max_date_loaded    — DATE — always present once first load is done
    max_datetime_loaded — TIMESTAMPTZ — present for data loaded after the
                          datetime watermark was introduced (March 2026+)
 
    Returns (None, None) on first run.
    """
    location = _BRANCH_TO_LOCATION.get(branch, branch)
    try:
        row = fetchone(
            """
            SELECT max_date_loaded, max_datetime_loaded
            FROM branch_watermarks
            WHERE branch = %s
            """,
            (location,),
        )
        if row is None:
            return None, None
        return row["max_date_loaded"], row["max_datetime_loaded"]
    except Exception as exc:
        logger.warning(f"Could not read watermark for {branch}: {exc}")
        return None, None
 


# def update_branch_watermark(branch: str, max_date: date, run_id: int) -> None:
#     """
#     Upsert the branch watermark after a successful sales load.
#     Only updates if new max_date is AFTER the existing watermark.
#     """
#     location = _BRANCH_TO_LOCATION.get(branch, branch)
#     try:
#         execute(
#             """
#             INSERT INTO branch_watermarks (branch, max_date_loaded, last_updated_at, last_run_id)
#             VALUES (%s, %s, NOW(), %s)
#             ON CONFLICT (branch) DO UPDATE SET
#                 max_date_loaded = GREATEST(branch_watermarks.max_date_loaded, EXCLUDED.max_date_loaded),
#                 last_updated_at = NOW(),
#                 last_run_id     = EXCLUDED.last_run_id
#             """,
#             (location, max_date, run_id),
#         )
#         logger.info(f"[Watermark][{branch}] Updated to {max_date}")
#     except Exception as exc:
#         logger.warning(f"[Watermark][{branch}] Failed to update: {exc}")

# Applies the 4pm rule and stores max_datetime_loaded.
# =============================================================================
 
def update_branch_watermark(
    branch: str,
    max_date: date,
    run_id: int,
    merged_df: pd.DataFrame | None = None,
) -> None:
    """
    Upsert the branch watermark after a successful sales load.
 
    4pm rule:
      - Look at max(sale_datetime) for the latest date in the loaded data
      - If that time is >= 4pm  → watermark advances to max_date
      - If that time is <  4pm  → watermark stays at max_date - 1 day
        (day is incomplete; next run will re-check and pick up late transactions)
      - If sale_datetime is unavailable (historical) → old date-only logic
 
    Always stores max_datetime_loaded so the next run can filter precisely
    on datetime rather than date, catching late transactions on the boundary day.
    """
    location = _BRANCH_TO_LOCATION.get(branch, branch)
 
    # ── Determine effective watermark date and max datetime ───────────────────
    effective_date   = max_date
    max_datetime_val = None
 
    if merged_df is not None and "Date Sold Datetime" in merged_df.columns:
        # Get the max datetime across ALL loaded rows for the latest date
        latest_day_mask = merged_df["Date Sold"] == max_date
        latest_day_dts  = merged_df.loc[latest_day_mask, "Date Sold Datetime"].dropna()
 
        if not latest_day_dts.empty:
            max_dt = latest_day_dts.max()
 
            # Normalise timezone
            if hasattr(max_dt, "tzinfo") and max_dt.tzinfo is None:
                max_dt = max_dt.replace(tzinfo=timezone.utc)
 
            max_datetime_val = max_dt
            latest_time      = max_dt.time()
 
            if latest_time < dt_time(_DAY_COMPLETE_HOUR, 0):
                # Day is incomplete — step back one day
                from datetime import timedelta
                effective_date = max_date - timedelta(days=1)
                logger.info(
                    f"[Watermark][{branch}] Latest transaction on {max_date} was at "
                    f"{latest_time} — before 4pm cutoff. "
                    f"Watermark held at {effective_date} (day incomplete)."
                )
            else:
                logger.info(
                    f"[Watermark][{branch}] Latest transaction on {max_date} was at "
                    f"{latest_time} — day complete. Advancing watermark to {effective_date}."
                )
        else:
            # No datetimes for the latest date — historical data, use date only
            logger.info(
                f"[Watermark][{branch}] No sale_datetime for {max_date} "
                f"(historical data) — using date-only watermark."
            )
 
    try:
        execute(
            """
            INSERT INTO branch_watermarks
                (branch, max_date_loaded, max_datetime_loaded, last_updated_at, last_run_id)
            VALUES (%s, %s, %s, NOW(), %s)
            ON CONFLICT (branch) DO UPDATE SET
                max_date_loaded     = GREATEST(
                                        branch_watermarks.max_date_loaded,
                                        EXCLUDED.max_date_loaded
                                      ),
                max_datetime_loaded = CASE
                    WHEN EXCLUDED.max_datetime_loaded IS NOT NULL
                         AND (branch_watermarks.max_datetime_loaded IS NULL
                              OR EXCLUDED.max_datetime_loaded
                                 > branch_watermarks.max_datetime_loaded)
                    THEN EXCLUDED.max_datetime_loaded
                    ELSE branch_watermarks.max_datetime_loaded
                END,
                last_updated_at     = NOW(),
                last_run_id         = EXCLUDED.last_run_id
            """,
            (location, effective_date, max_datetime_val, run_id),
        )
        logger.info(
            f"[Watermark][{branch}] Stored — "
            f"date={effective_date}  datetime={max_datetime_val}"
        )
    except Exception as exc:
        logger.warning(f"[Watermark][{branch}] Failed to update: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# TYPE COERCION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _strip_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
    return df


# Ordered list of formats tried by _clean_date_sold / _clean_datetime_sold.
# ISO formats first (4-digit year is unambiguous and always matches before MDY).
# Using explicit strptime avoids pd.to_datetime's dayfirst ambiguity, which
# silently misreads ISO '2025-06-01' as January 6th when dayfirst=True.
#
# Date Sold format history:
#   Pre-May 2026  : #YYYY-MM-DD HH:MM:SS#  (Portal CBD still uses this)
#   May 2026+     : M/D/YYYY H:MM           (all other branches, no seconds)
_SALE_DATETIME_FORMATS = [
    "%Y-%m-%d %H:%M:%S",     # Old ISO with seconds:         2026-03-21 08:33:11
    "%Y-%m-%d %H:%M",        # Old ISO without seconds:      2026-03-21 08:33
    "%Y-%m-%d",              # Date-only ISO:                2026-03-21
    "%m/%d/%Y %H:%M:%S",     # MDY with seconds:             4/30/2026 10:00:00
    "%m/%d/%Y %H:%M",        # MDY no seconds:               4/30/2026 10:00
    "%m/%d/%Y",              # MDY date-only:                4/30/2026
    "%m/%d/%Y %I:%M:%S %p",  # MDY 12hr double-space:        6/1/2025  9:53:00 AM
    "%m/%d/%Y %I:%M %p",     # MDY 12hr no seconds:          6/1/2025  9:53 AM
]


def _clean_date_sold(value) -> date | None:
    """
    Parse Date Sold from POS format.
    Handles two formats:
      - Old ISO (with optional # guards): '#2026-03-21 08:33:11#' -> date(2026, 3, 21)
      - New MDY (M/D/YYYY H:MM):         '4/30/2026 10:00'        -> date(2026, 4, 30)
    Uses explicit strptime list -- avoids pd.to_datetime dayfirst ambiguity.
    """
    if pd.isna(value):
        return None
    cleaned = str(value).strip().strip("#").strip()
    if not cleaned:
        return None
    for fmt in _SALE_DATETIME_FORMATS:
        try:
            return datetime.strptime(cleaned, fmt).date()
        except ValueError:
            continue
    return None


def _clean_datetime_sold(value) -> datetime | None:
    """
    Parse full timestamp from Date Sold including time.
    Handles two formats:
      - Old ISO (with optional # guards): '#2026-03-21 08:33:11#' -> datetime(2026, 3, 21, 8, 33, 11)
      - New MDY (M/D/YYYY H:MM):         '4/30/2026 10:00'        -> datetime(2026, 4, 30, 10, 0)
    New MDY format has no seconds -- datetime will have seconds=0, which is fine
    for the 4pm completeness guard (compares hour only).
    Returns None if unparseable.
    """
    if pd.isna(value):
        return None
    cleaned = str(value).strip().strip("#").strip()
    if not cleaned:
        return None
    for fmt in _SALE_DATETIME_FORMATS:
        try:
            return datetime.strptime(cleaned, fmt)
        except ValueError:
            continue
    return None


def _to_date(value) -> date | None:
    if pd.isna(value):
        return None
    if isinstance(value, (date, datetime)):
        return value.date() if isinstance(value, datetime) else value
    try:
        return pd.to_datetime(value).date()
    except Exception:
        return None


def _to_numeric(value):
    if pd.isna(value):
        return None
    try:
        return float(str(value).replace(",", "").strip())
    except Exception:
        return None


def _to_str(value) -> str | None:
    """
    Clean string — strips whitespace, returns None for empty.
    Strips .0 float artefact e.g. '712345678.0' → '712345678'.
    """
    if pd.isna(value):
        return None
    s = str(value).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s if s else None


# ══════════════════════════════════════════════════════════════════════════════
# SALES PARSER
# ══════════════════════════════════════════════════════════════════════════════

def parse_sales_file(path: Path) -> pd.DataFrame:
    """
    Read a sales CSV or XLSX.
    - Strips column name whitespace
    - Validates all required columns are present
    - Cleans Date Sold (strips # characters)
    - Returns DataFrame with original column names intact
    """
    suffix = path.suffix.lower()

    if suffix == ".csv":
        try:
            df = pd.read_csv(path, dtype=str, low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(path, dtype=str, low_memory=False, encoding="cp1252")
    elif suffix in (".xlsx", ".xlsm"):
        df = pd.read_excel(path, dtype=str)
    else:
        raise ValueError(f"Unsupported sales file type: {suffix}")

    df = _strip_columns(df)

    missing = [c for c in SALES_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Sales file {path.name} missing columns: {missing}")

    df = df[SALES_COLUMNS].copy()

    df["Date Sold Raw"]      = df["Date Sold"].copy()   # keep original
    df["Date Sold Datetime"] = df["Date Sold Raw"].apply(_clean_datetime_sold)
    df["Date Sold"]          = df["Date Sold Datetime"].apply(
        lambda dt: dt.date() if dt is not None else None
    )
    df["On Hand"]   = pd.to_numeric(df["On Hand"],  errors="coerce")
    df["Qty Sold"]  = pd.to_numeric(df["Qty Sold"], errors="coerce")
    df["Transaction ID"] = df["Transaction ID"].astype(str).str.strip()
    df["Item"]           = df["Item"].astype(str).str.strip()

    return df


# def parse_sales_file_after_date(path: Path, after_date: date) -> pd.DataFrame:
#     """
#     Read a sales file and return ONLY rows with Date Sold > after_date.
#     Used for incremental files to avoid re-loading already-watermarked rows.
#     """
#     df = parse_sales_file(path)
#     if df.empty:
#         return df
#     return df[df["Date Sold"].apply(lambda d: d is not None and d > after_date)].copy()
# Filters on sale_datetime > watermark_datetime when available.
# Falls back to date comparison for rows where sale_datetime is NULL
# (historical bulk data loaded before datetime tracking was added).
# =============================================================================
 
def parse_sales_file_after_date(
    path: Path,
    after_date: date,
    after_datetime: datetime | None = None,
) -> pd.DataFrame:
    """
    Read a sales file and return only rows newer than the watermark.
 
    Priority:
      1. If after_datetime is set AND the row has a sale_datetime
         → keep row if sale_datetime > after_datetime
      2. If after_datetime is set but the row has no sale_datetime (historical)
         → keep row if Date Sold > after_date  (safe fallback)
      3. If after_datetime is None
         → keep row if Date Sold > after_date  (old behaviour, unchanged)
    """
    df = parse_sales_file(path)
    if df.empty:
        return df
 
    if after_datetime is None:
        # Old behaviour — date-only filter
        return df[
            df["Date Sold"].apply(lambda d: d is not None and d > after_date)
        ].copy()
 
    # Datetime-aware filter
    def _keep_row(row) -> bool:
        sale_dt = row.get("Date Sold Datetime")
        if sale_dt is not None:
            # Make both timezone-aware for comparison
            if sale_dt.tzinfo is None:
                sale_dt = sale_dt.replace(tzinfo=timezone.utc)
            wm = after_datetime
            if wm.tzinfo is None:
                wm = wm.replace(tzinfo=timezone.utc)
            return sale_dt > wm
        # Fallback for rows without a datetime (historical)
        d = row.get("Date Sold")
        return d is not None and d > after_date
 
    mask = df.apply(_keep_row, axis=1)
    return df[mask].copy()


# ══════════════════════════════════════════════════════════════════════════════
# CANONICAL ELECTION + MERGE (V2)
# ══════════════════════════════════════════════════════════════════════════════

def elect_canonical_file(
    file_records: list[dict],
    watermark: date | None,
) -> tuple[dict | None, list[dict], list[dict]]:
    """
    Elect the canonical file for a branch.

    Strategy:
      1. Peek inside every file to get its actual max Date Sold
      2. The file with the highest max Date Sold is canonical
         (regardless of filename classification — the data wins)
      3. Files whose max Date Sold <= watermark are skipped entirely
      4. Files with max Date Sold > watermark are candidates

    Returns:
        canonical    — the elected canonical file record (or None)
        incrementals — non-canonical files with dates > watermark
        skipped      — files fully covered by the watermark
    """
    enriched = []
    for rec in file_records:
        path = Path(rec["local_path"])
        max_date = get_max_date_sold(path)
        enriched.append({**rec, "_max_date": max_date})
        logger.info(
            f"[Sales] Peeked {rec['filename']}: max Date Sold = {max_date}"
        )

    # Files where max_date > watermark (or no watermark) are candidates
    if watermark:
        candidates = [
            r for r in enriched
            if r["_max_date"] is not None and r["_max_date"] > watermark
        ]
        skipped = [
            r for r in enriched
            if r["_max_date"] is None or r["_max_date"] <= watermark
        ]
    else:
        # No watermark — first run, all files are candidates
        candidates = [r for r in enriched if r["_max_date"] is not None]
        skipped    = [r for r in enriched if r["_max_date"] is None]

    if not candidates:
        return None, [], skipped

    # Elect canonical — the candidate with the highest max Date Sold
    candidates.sort(key=lambda r: r["_max_date"], reverse=True)
    canonical    = candidates[0]
    incrementals = candidates[1:]

    logger.info(
        f"[Sales] Canonical elected: {canonical['filename']} "
        f"(max Date Sold = {canonical['_max_date']})"
    )
    if incrementals:
        logger.info(
            f"[Sales] Incrementals with new data: "
            f"{[r['filename'] for r in incrementals]}"
        )
    if skipped:
        logger.info(
            f"[Sales] Skipping {len(skipped)} file(s) already covered by watermark "
            f"({watermark}): {[r['filename'] for r in skipped]}"
        )

    return canonical, incrementals, skipped


def merge_sales_files_v2(
    file_records: list[dict],
    watermark: date | None,
    watermark_dt: datetime | None = None, # New additon for datetime-aware merging
) -> tuple[pd.DataFrame, list[int], date | None]:
    """
    Load and merge sales files using canonical election + watermark logic.

    Loading strategy:
      - Canonical file: loaded in full if no watermark, or only rows after
        watermark if a watermark exists (avoids re-loading years of history)
      - Incremental files: only rows after watermark
      - Deduplication on SALES_DEDUP_COLUMNS (safety net for any overlap)

    Returns:
        merged DataFrame with '_source_file_id' and '_source_filename' columns
        list of file_ids that contributed rows
        new max Date Sold across all loaded rows (for watermark update)
    """
    canonical, incrementals, skipped = elect_canonical_file(file_records, watermark)

    # Mark skipped files in ingestion_files immediately
    for rec in skipped:
        mark_file_status(rec["file_id"], "skipped")
        logger.info(
            f"[Sales] Skipped (watermark covered): {rec['filename']}"
        )

    if canonical is None:
        logger.warning("[Sales] No files with new data found — nothing to load")
        return pd.DataFrame(), [], None

    frames        = []
    contributing  = []

    # ── Load canonical ────────────────────────────────────────────────────────
    try:
        path = Path(canonical["local_path"])
        if watermark:
            # Load only rows newer than the watermark — avoids re-reading 3 years
            # df_can = parse_sales_file_after_date(path, watermark)
            # load_reason = f"canonical_incremental (after {watermark})"
            df_can = parse_sales_file_after_date(path, watermark, watermark_dt)
            load_reason = (
                f"canonical_incremental (after datetime {watermark_dt})"
                if watermark_dt else
                f"canonical_incremental (after date {watermark})"
            )
            
        else:
            # First run — load everything
            df_can = parse_sales_file(path)
            load_reason = "canonical_full"

        if not df_can.empty:
            df_can["_source_file_id"]  = canonical["file_id"]
            df_can["_source_filename"] = canonical["filename"]
            df_can["_file_priority"]   = 1  # canonical wins on dedup conflict
            frames.append(df_can)
            contributing.append(canonical["file_id"])
            logger.info(
                f"[Sales] Loaded {load_reason}: {canonical['filename']} "
                f"({len(df_can):,} rows)"
            )
        else:
            logger.info(
                f"[Sales] Canonical {canonical['filename']} has no rows "
                f"after watermark {watermark} — skipping"
            )
            mark_file_status(canonical["file_id"], "skipped")
    except Exception as exc:
        logger.error(
            f"[Sales] Failed to load canonical {canonical['filename']}: {exc}"
        )
        mark_file_status(canonical["file_id"], "failed", str(exc))

    # ── Load incrementals (post-watermark rows only) ──────────────────────────
    for rec in incrementals:
        try:
            path = Path(rec["local_path"])
            # if watermark:
            #     df_inc = parse_sales_file_after_date(path, watermark)
            # else:
            #     df_inc = parse_sales_file(path)
            if watermark:
                df_inc = parse_sales_file_after_date(path, watermark, watermark_dt)
            else:
                df_inc = parse_sales_file(path)

            if not df_inc.empty:
                df_inc["_source_file_id"]  = rec["file_id"]
                df_inc["_source_filename"] = rec["filename"]
                df_inc["_file_priority"]   = 2  # incrementals load after canonical
                frames.append(df_inc)
                contributing.append(rec["file_id"])
                logger.info(
                    f"[Sales] Loaded incremental: {rec['filename']} ({len(df_inc):,} rows)"
                )
            else:
                logger.info(
                    f"[Sales] Incremental {rec['filename']} has no rows "
                    f"after watermark {watermark}"
                )
                mark_file_status(rec["file_id"], "skipped")
        except Exception as exc:
            logger.error(f"[Sales] Failed to load incremental {rec['filename']}: {exc}")
            mark_file_status(rec["file_id"], "failed", str(exc))

    if not frames:
        return pd.DataFrame(), [], None

    combined = pd.concat(frames, ignore_index=True)

    # Sort so canonical rows come first; incremental override on keep='last'
    combined = combined.sort_values("_file_priority")

    # Deduplicate — safety net for any overlap between canonical and incrementals
    missing_dedup = [c for c in SALES_DEDUP_COLUMNS if c not in combined.columns]
    if not missing_dedup:
        before = len(combined)
        combined = combined.drop_duplicates(subset=SALES_DEDUP_COLUMNS, keep="last")
        dropped = before - len(combined)
        if dropped > 0:
            logger.info(f"[Sales] Deduplication removed {dropped:,} duplicate rows")

    combined = combined.drop(columns=["_file_priority"])

    # New watermark = max date across all loaded rows
    new_max = combined["Date Sold"].dropna().max()

# ── 4pm completeness guard ────────────────────────────────────────────────
    # If the latest date's last transaction is before 4pm, the day is still
    # open. Strip those rows so partial-day data never reaches the DB.
    # The watermark stays at the previous day; next run picks up the full day.
    if new_max is not None:
        latest_day_mask = combined["Date Sold"] == new_max
        latest_day_dts  = combined.loc[latest_day_mask, "Date Sold Datetime"].dropna()

        if not latest_day_dts.empty:
            max_time = latest_day_dts.max().time()
            if max_time < dt_time(_DAY_COMPLETE_HOUR, 0):
                logger.info(
                    f"[Sales] Latest date {new_max} last txn at {max_time} — "
                    f"before 4pm cutoff. Excluding {new_max} rows from this load."
                )
                combined = combined[combined["Date Sold"] != new_max].copy()
                new_max  = combined["Date Sold"].dropna().max() if not combined.empty else None
            else:
                logger.info(
                    f"[Sales] Latest date {new_max} last txn at {max_time} — "
                    f"day complete, all rows included."
                )

    return combined, contributing, new_max


def _insert_sales_rows(conn, df: pd.DataFrame, branch: str) -> int:
    """Insert merged sales rows into stg_sales_reports."""
    columns = [
        "source_file_id", "source_filename", "branch",
        "department", "category", "item", "description",
        "on_hand", "last_sold", "qty_sold", "total_tax_ex",
        "transaction_id", "date_sold", "sale_time", "sale_datetime",
    ]
    rows = []
    for _, row in df.iterrows():
        rows.append((
            row.get("_source_file_id"),
            row.get("_source_filename"),
            branch,
            _to_str(row.get("Department")),
            _to_str(row.get("Category")),
            _to_str(row.get("Item")),
            _to_str(row.get("Description")),
            _to_numeric(row.get("On Hand")),
            _to_date(row.get("Last Sold")),
            _to_numeric(row.get("Qty Sold")),
            _to_numeric(row.get("Total (Tax Ex)")),
            _to_str(row.get("Transaction ID")),
            row.get("Date Sold"),
            row.get("Date Sold Datetime").time() if row.get("Date Sold Datetime") else None,
            row.get("Date Sold Datetime"),
        ))

    CONFLICT_COLS = [
        "transaction_id", "branch", "date_sold",
        "description", "qty_sold", "total_tax_ex",
    ]
    attempted, skipped = bulk_insert_safe(
        conn, "stg_sales_reports", columns, rows,
        conflict_columns=CONFLICT_COLS,
    )
    if skipped:
        logger.info(
            f"[Sales][{branch}] ON CONFLICT skipped {skipped:,} duplicate rows "
            f"(out of {attempted:,} attempted)"
        )
    return attempted - skipped

# ══════════════════════════════════════════════════════════════════════════════
# INGESTION RUN MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════

def start_run() -> int:
    sql = """
        INSERT INTO ingestion_runs (pipeline_name, started_at, status)
        VALUES ('sharepoint_ingestion', NOW(), 'running')
        RETURNING id
    """
    return insert_returning_id(sql)


def finish_run(run_id: int, stats: dict) -> None:
    sql = """
        UPDATE ingestion_runs SET
            finished_at      = NOW(),
            status           = %(status)s,
            files_seen       = %(files_seen)s,
            files_downloaded = %(files_downloaded)s,
            files_processed  = %(files_processed)s,
            files_failed     = %(files_failed)s,
            notes            = %(notes)s
        WHERE id = %(run_id)s
    """
    execute(sql, {**stats, "run_id": run_id})


def register_file(run_id: int, meta: dict[str, Any]) -> int:
    sql = """
        INSERT INTO ingestion_files (
            run_id, branch, report_type, file_type,
            filename, file_extension,
            sharepoint_item_id, sharepoint_path, sharepoint_last_modified,
            sharepoint_size_bytes, local_path,
            file_hash, row_count,
            is_canonical, canonical_reason,
            downloaded_at, status
        ) VALUES (
            %(run_id)s, %(branch)s, %(report_type)s, %(file_type)s,
            %(filename)s, %(file_extension)s,
            %(sharepoint_item_id)s, %(sharepoint_path)s, %(sharepoint_last_modified)s,
            %(sharepoint_size_bytes)s, %(local_path)s,
            %(file_hash)s, %(row_count)s,
            %(is_canonical)s, %(canonical_reason)s,
            NOW(), 'pending'
        )
        ON CONFLICT (run_id, branch, report_type, file_hash) 
        WHERE file_hash IS NOT NULL
        DO UPDATE SET status = EXCLUDED.status
        RETURNING id
    """
    return insert_returning_id(sql, {**meta, "run_id": run_id})


# def register_file(run_id: int, meta: dict[str, Any]) -> int:
#     sql = """
#         INSERT INTO ingestion_files (
#             run_id, branch, report_type, file_type,
#             filename, file_extension,
#             sharepoint_item_id, sharepoint_path, sharepoint_last_modified,
#             sharepoint_size_bytes, local_path,
#             file_hash, row_count,
#             is_canonical, canonical_reason,
#             downloaded_at, status
#         ) VALUES (
#             %(run_id)s, %(branch)s, %(report_type)s, %(file_type)s,
#             %(filename)s, %(file_extension)s,
#             %(sharepoint_item_id)s, %(sharepoint_path)s, %(sharepoint_last_modified)s,
#             %(sharepoint_size_bytes)s, %(local_path)s,
#             %(file_hash)s, %(row_count)s,
#             %(is_canonical)s, %(canonical_reason)s,
#             NOW(), 'pending'
#         )
#         RETURNING id
#     """
#     return insert_returning_id(sql, {**meta, "run_id": run_id})


def mark_file_status(file_id: int, status: str, error: str | None = None) -> None:
    execute(
        "UPDATE ingestion_files SET status=%s, processed_at=NOW(), error_message=%s WHERE id=%s",
        (status, error, file_id),
    )


def mark_canonical(file_id: int, reason: str) -> None:
    execute(
        "UPDATE ingestion_files SET is_canonical=TRUE, canonical_reason=%s WHERE id=%s",
        (reason, file_id),
    )


# ══════════════════════════════════════════════════════════════════════════════
# TOP-LEVEL PROCESS FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def process_sales_branch(
    run_id: int,
    branch: str,
    file_metas: list[dict[str, Any]],
) -> dict:
    """
    Process all sales files for a branch using canonical election + watermark.

      1. Read branch watermark from DB
      2. Hash, count and register all files in ingestion_files
      3. Elect canonical file by peeking at max Date Sold in each file
      4. Skip files fully covered by watermark
      5. Load canonical (post-watermark rows only on incremental runs)
      6. Load incremental files (post-watermark rows only)
      7. Deduplicate merged result
      8. Insert into stg_sales_reports
      9. Update branch watermark

    Returns stats dict.
    """
    stats = {"processed": 0, "failed": 0, "skipped": 0}

    # ── Step 1: Read watermark ─────────────────────────────────────────────
    # watermark = get_branch_watermark(branch)
    # if watermark:
    #     logger.info(f"[Sales][{branch}] Watermark: already loaded up to {watermark}")
    # else:
    #     logger.info(f"[Sales][{branch}] No watermark — first run, loading all data")

    watermark, watermark_dt = get_branch_watermark(branch)
    if watermark:
        logger.info(
            f"[Sales][{branch}] Watermark: loaded up to {watermark} "
            f"(datetime: {watermark_dt})"
        )
    else:
        logger.info(f"[Sales][{branch}] No watermark — first run, loading all data")

    # ── Step 2: Hash, count, register all files ────────────────────────────
    registered = []
    for meta in file_metas:
        path = Path(meta["local_path"])
        meta["file_type"]        = classify_sales_file(meta["filename"])
        meta["file_hash"]        = compute_file_hash(path)
        meta["row_count"]        = count_rows(path)
        meta["is_canonical"]     = False
        meta["canonical_reason"] = meta["file_type"]
 
        # ── Hash check: skip files already successfully loaded ────────────
        # Prevents cumulative files from being re-ingested on every run.
        # Root cause of 17M duplicate rows (April 2026 incident).
        if is_file_already_loaded(meta["file_hash"]):
            logger.info(
                f"[Sales][{branch}] SKIPPING {meta['filename']} — "
                f"hash already loaded in a previous run."
            )
            # Still register the file so we have a full audit trail,
            # but mark it skipped immediately without loading any rows.
            meta["is_canonical"]     = False
            meta["canonical_reason"] = "duplicate_hash"
            file_id = register_file(run_id, meta)
            mark_file_status(file_id, "skipped")
            stats["skipped"] += 1
            continue
        # ─────────────────────────────────────────────────────────────────
 
        file_id = register_file(run_id, meta)
        registered.append({**meta, "file_id": file_id})

    # ── Steps 3-7: Elect canonical, merge, deduplicate ─────────────────────
    # try:
    #     merged_df, contributing_ids, new_max_date = merge_sales_files_v2(
    #         registered, watermark
    #     )
    try:
        merged_df, contributing_ids, new_max_date = merge_sales_files_v2(
            registered, watermark, watermark_dt
        )
    except Exception as exc:
        logger.error(f"[Sales][{branch}] merge failed: {exc}")
        for rec in registered:
            mark_file_status(rec["file_id"], "failed", str(exc))
        stats["failed"] += len(registered)
        return stats

    if merged_df.empty:
        logger.warning(f"[Sales][{branch}] No new data after watermark — nothing loaded")
        for rec in registered:
            if rec["file_id"] not in {r["file_id"] for r in registered
                                       if r["file_id"] in contributing_ids}:
                mark_file_status(rec["file_id"], "skipped")
        stats["skipped"] += len(registered)
        return stats

    # ── Step 8: Insert ─────────────────────────────────────────────────────
    try:
        with get_connection() as conn:
            n = _insert_sales_rows(conn, merged_df, branch)

        for file_id in contributing_ids:
            file_type = next(
                (r["file_type"] for r in registered if r["file_id"] == file_id),
                "unknown",
            )
            mark_canonical(file_id, file_type)
            mark_file_status(file_id, "loaded")

        logger.info(
            f"[Sales][{branch}] Loaded {n:,} rows from "
            f"{len(contributing_ids)} file(s)"
        )
        stats["processed"] += len(contributing_ids)

        # ── Step 9: Update watermark ───────────────────────────────────────
        # if new_max_date is not None:
        #     update_branch_watermark(branch, new_max_date, run_id)
        if new_max_date is not None:
            update_branch_watermark(branch, new_max_date, run_id, merged_df)

    except Exception as exc:
        logger.error(f"[Sales][{branch}] DB insert failed: {exc}")
        for file_id in contributing_ids:
            mark_file_status(file_id, "failed", str(exc))
        stats["failed"] += len(contributing_ids)

    return stats


def process_cashier_file(
    run_id: int,
    branch: str,
    meta: dict[str, Any],
) -> dict:
    """
    Parse and load a single cashier file into stg_cashier_reports.
    Unchanged from V1.
    """
    stats = {"processed": 0, "failed": 0}

    path = Path(meta["local_path"])
    meta["file_type"]        = "cashier"
    meta["file_hash"]        = compute_file_hash(path)
    meta["row_count"]        = count_rows(path)
    meta["is_canonical"]     = True
    meta["canonical_reason"] = "only_file"
    file_id = register_file(run_id, meta)

    try:
        fallback_dt = meta.get("sharepoint_last_modified")
        df = parse_cashier_file(path, fallback_dt=fallback_dt)

        if df.empty:
            logger.warning(f"[Cashier][{branch}] No data found in {path.name}")
            mark_file_status(file_id, "skipped")
            return stats

        with get_connection() as conn:
            n = _insert_cashier_rows(conn, df, file_id, meta["filename"], branch)

        mark_file_status(file_id, "loaded")
        logger.info(f"[Cashier][{branch}] Loaded {n:,} rows from {path.name}")
        stats["processed"] += 1

    except Exception as exc:
        logger.error(f"[Cashier][{branch}] Failed to load {path.name}: {exc}")
        mark_file_status(file_id, "failed", str(exc))
        stats["failed"] += 1

    return stats


# ══════════════════════════════════════════════════════════════════════════════
# CASHIER PARSER (unchanged from V1)
# ══════════════════════════════════════════════════════════════════════════════

# Pre-Apr 2026: 9 columns, Sales Rep at position 8
CASHIER_COLUMN_POSITIONS_V1 = {
    0: "Receipt Txn No", 1: "Amount",    2: "Txn Costs",
    3: "Time",           4: "Txn Type",  5: "Ordered Via",
    6: "Client Name",    7: "Phone Number", 8: "Sales Rep",
}

CASHIER_COLUMN_POSITIONS_V2 = {
    0: "Receipt Txn No", 1: "Amount",             2: "Txn Costs",
    3: "Time",           4: "Txn Type",            5: "Ordered Via",
    6: "Respond Customer ID",
    7: "Client Name",    8: "Phone Number",        9: "Sales Rep",
}

# Column name variants across cashier report schema versions.
# Applied at the top of the sheet loop so all downstream logic sees
# canonical names regardless of when the file was exported.
#
# History:
#   Pre-Apr 2026 : no Respond Customer ID column at all  (V1, 9 cols)
#   Apr 2026     : "Respond Customer ID"                  (V2, 10 cols)
#   May 2026+    : "Customer ID Respond.io"               (V2, 10 cols)
#                   Some exports may still use "Customer ID [Respond.io]"
CASHIER_COLUMN_ALIASES = {
    "Customer ID [Respond.io]": "Respond Customer ID",
    "Customer ID Respond.io": "Respond Customer ID",
}


# def _get_cashier_position_map(df: pd.DataFrame) -> dict:
#     """
#     Select the correct positional column map based on column count.
#     10+ columns = V2 layout (Apr 2026+).
#     Anything fewer = V1 layout (pre-Apr 2026).
#     Sales Rep is always the last mapped position — extra columns to the right are ignored.
#     """
#     return CASHIER_COLUMN_POSITIONS_V2 if len(df.columns) >= 10 else CASHIER_COLUMN_POSITIONS_V1

def _get_cashier_position_map(df: pd.DataFrame) -> dict:
    """
    Detect cashier layout version by presence of 'Respond Customer ID',
    not by column count. Column count is unreliable because Excel sheets
    often carry trailing empty columns that inflate len(df.columns).

    V2 (Apr 2026+) : 'Respond Customer ID' present after alias normalisation
    V1 (pre-Apr 2026) : column absent
    """
    if "Respond Customer ID" in df.columns:
        return CASHIER_COLUMN_POSITIONS_V2
    return CASHIER_COLUMN_POSITIONS_V1


def extract_month_year(
    filename: str, fallback_dt: datetime | None = None
) -> tuple[int, int]:
    match = re.search(r"([a-zA-Z]{3,})\s+(\d{4})", filename)
    if match:
        month_str = match.group(1)[:3].lower()
        year      = int(match.group(2))
        month     = MONTH_MAP.get(month_str)
        if month:
            return month, year
    if fallback_dt:
        return fallback_dt.month, fallback_dt.year
    now = datetime.now()
    return now.month, now.year


def _get_sheet_day(
    sheet_name: str, expected_month: int, expected_year: int
) -> int | None:
    if re.fullmatch(r'0[1-9]|[12][0-9]|3[01]', sheet_name.strip()):
        return int(sheet_name.strip())
    m = re.fullmatch(r'(\d{2})-(\d{2})-(\d{4})', sheet_name.strip())
    if m:
        day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if month == expected_month and year == expected_year:
            return day
    return None


def parse_cashier_file(
    path: Path,
    fallback_dt: datetime | None = None,
) -> pd.DataFrame:
    month, year = extract_month_year(path.name, fallback_dt)
    xl          = pd.ExcelFile(path, engine="openpyxl")
    frames      = []

    for sheet_name in xl.sheet_names:
        day = _get_sheet_day(sheet_name, month, year)
        if day is None:
            continue

        try:
            df = xl.parse(sheet_name, dtype=str)
        except Exception as exc:
            logger.warning(f"Could not read sheet {sheet_name} in {path.name}: {exc}")
            continue

        df = _strip_columns(df)

        # Normalise column name variants before any other logic.
        # Handles May 2026+ header variants before positional fallback.
        df = df.rename(columns=CASHIER_COLUMN_ALIASES)

        present = [c for c in CASHIER_COLUMNS if c in df.columns]
        if not present:
            continue

        missing_by_name = [c for c in CASHIER_COLUMNS if c not in df.columns]
        # if missing_by_name and len(df.columns) >= len(CASHIER_COLUMN_POSITIONS):
        #     rename_map = {
        #         df.columns[pos]: name
        #         for pos, name in CASHIER_COLUMN_POSITIONS.items()
        #         if pos < len(df.columns) and df.columns[pos] not in CASHIER_COLUMNS
        #     }
        #     if rename_map:
        #         df = df.rename(columns=rename_map)
        #         logger.info(
        #             f"Sheet {sheet_name} in {path.name} — "
        #             f"positional rename applied for {list(rename_map.values())}"
        #         )
        if missing_by_name:
            position_map = _get_cashier_position_map(df)
            if len(df.columns) >= len(position_map):
                rename_map = {
                    df.columns[pos]: name
                    for pos, name in position_map.items()
                    if pos < len(df.columns) and df.columns[pos] not in CASHIER_COLUMNS
                }
                if rename_map:
                    df = df.rename(columns=rename_map)
                    layout = "V2 (Apr 2026+)" if position_map is CASHIER_COLUMN_POSITIONS_V2 else "V1 (pre-Apr 2026)"
                    logger.info(
                        f"Sheet {sheet_name} in {path.name} — "
                        f"positional rename applied [{layout}] for {list(rename_map.values())}"
                    )

        for col in CASHIER_COLUMNS:
            if col not in df.columns:
                logger.warning(
                    f"Sheet {sheet_name} in {path.name} missing '{col}' — filling None"
                )
                df[col] = None

        df = df[CASHIER_COLUMNS].copy()
        df = df.dropna(how="all")
        if df.empty:
            continue

        try:
            txn_date = date(year, month, day)
        except ValueError:
            continue

        df["_transaction_date"] = txn_date
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def _insert_cashier_rows(
    conn, df: pd.DataFrame, source_file_id: int,
    source_filename: str, branch: str,
) -> int:
    columns = [
        "source_file_id", "source_filename", "branch", "transaction_date",
        "receipt_txn_no", "amount", "txn_costs", "txn_time",
        "txn_type", "ordered_via","respond_customer_id",  
        "client_name", "phone_number", "sales_rep",
    ]
    rows = []
    for _, row in df.iterrows():
        rows.append((
            source_file_id,
            source_filename,
            branch,
            row.get("_transaction_date"),
            _to_str(row.get("Receipt Txn No")),
            _to_numeric(row.get("Amount")),
            _to_numeric(row.get("Txn Costs")),
            _to_str(row.get("Time")),
            _to_str(row.get("Txn Type")),
            _to_str(row.get("Ordered Via")),
            _to_str(row.get("Respond Customer ID")),
            _to_str(row.get("Client Name")),
            _to_str(row.get("Phone Number")),
            _to_str(row.get("Sales Rep")),
        ))
    return bulk_insert(conn, "stg_cashier_reports", columns, rows)


# ══════════════════════════════════════════════════════════════════════════════
# QTY LIST (unchanged from V1)
# ══════════════════════════════════════════════════════════════════════════════

QTY_COLUMNS = [
    "Department", "Category", "Item Lookup Code", "Description",
    "On-Hand", "Committed", "Reorder Pt.", "Restock Lvl.",
    "Qty to Order", "Supplier", "Reorder No.",
]

QTY_REQUIRED = {"Item Lookup Code", "Description", "On-Hand"}


def _parse_qty_date(
    filename: str, fallback_dt: datetime | None = None
) -> tuple[date | None, str]:
    name = filename
    m = re.search(r'(\d{2})\.(\d{2})\.(\d{2,4})', name)
    if m:
        day, month, year_raw = int(m.group(1)), int(m.group(2)), m.group(3)
        year = int(year_raw) + 2000 if len(year_raw) == 2 else int(year_raw)
        try:
            return date(year, month, day), 'filename_full'
        except ValueError:
            pass

    m = re.search(r'(?<![\d])([0-2]?[0-9]|3[01])(?![\d])', name)
    if m:
        day = int(m.group(1))
        if 1 <= day <= 31 and fallback_dt:
            try:
                inferred = date(fallback_dt.year, fallback_dt.month, day)
                return inferred, 'filename_day_inferred'
            except ValueError:
                pass

    if fallback_dt:
        return fallback_dt.date(), 'lastmodified'

    return date.today(), 'lastmodified'


def _detect_separator(lines: list[str]) -> str:
    for line in lines:
        stripped = line.lstrip().lstrip(";").strip()
        if not stripped:
            continue
        tabs   = stripped.count("\t")
        commas = stripped.count(",")
        if tabs > commas:
            return "\t"
        elif commas > tabs:
            return ","
        return "\t"
    return "\t"


def _clean_qty_csv(path: Path) -> pd.DataFrame:
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        with open(path, "r", encoding="cp1252") as f:
            lines = f.readlines()

    data_lines = []
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith(";"):
            content = stripped[1:].strip()
            if "Department" in content or "Item Lookup Code" in content:
                data_lines.append(stripped[1:])
        else:
            data_lines.append(line)

    sep     = _detect_separator(data_lines)
    cleaned = "".join(data_lines)
    try:
        df = pd.read_csv(io.StringIO(cleaned), dtype=str, sep=sep, low_memory=False)
    except Exception:
        df = pd.read_csv(path, dtype=str, sep=sep, on_bad_lines="skip")

    return df


def parse_qty_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()

    if suffix == ".csv":
        df = _clean_qty_csv(path)
    elif suffix in (".xlsx", ".xlsm"):
        df_check  = pd.read_excel(path, dtype=str, nrows=5, engine="openpyxl")
        first_val = str(df_check.iloc[0, 0] if not df_check.empty else "").strip()
        if "quantity list" in first_val.lower() or "filter" in first_val.lower():
            df = pd.read_excel(path, dtype=str, skiprows=2, engine="openpyxl")
        else:
            df = pd.read_excel(path, dtype=str, engine="openpyxl")
    else:
        raise ValueError(f"Unsupported qty file type: {suffix}")

    df = _strip_columns(df)

    missing_required = QTY_REQUIRED - set(df.columns)
    if missing_required:
        raise ValueError(
            f"Qty file {path.name} missing required columns: {missing_required}"
        )

    for col in QTY_COLUMNS:
        if col not in df.columns:
            logger.warning(f"[Qty] {path.name} missing optional column '{col}' — filling None")
            df[col] = None

    df = df[QTY_COLUMNS].copy()
    df = df.dropna(how="all")

    for col in ["On-Hand", "Committed", "Reorder Pt.", "Restock Lvl.", "Qty to Order"]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", "", regex=False),
                errors="coerce",
            )

    return df


def _insert_qty_rows(
    conn, df: pd.DataFrame, source_file_id: int,
    source_filename: str, branch: str,
    snapshot_date: date, snapshot_date_source: str,
) -> int:
    columns = [
        "source_file_id", "source_filename", "branch",
        "snapshot_date", "snapshot_date_source",
        "department", "category", "item_lookup_code", "description",
        "on_hand", "committed", "reorder_pt", "restock_lvl",
        "qty_to_order", "supplier", "reorder_no",
    ]
    rows = []
    for _, row in df.iterrows():
        rows.append((
            source_file_id, source_filename, branch,
            snapshot_date, snapshot_date_source,
            _to_str(row.get("Department")),
            _to_str(row.get("Category")),
            _to_str(row.get("Item Lookup Code")),
            _to_str(row.get("Description")),
            _to_numeric(row.get("On-Hand")),
            _to_numeric(row.get("Committed")),
            _to_numeric(row.get("Reorder Pt.")),
            _to_numeric(row.get("Restock Lvl.")),
            _to_numeric(row.get("Qty to Order")),
            _to_str(row.get("Supplier")),
            _to_str(row.get("Reorder No.")),
        ))
    return bulk_insert(conn, "stg_qty_list", columns, rows)


def _flag_inferred_date(file_id: int, snapshot_date: date, fallback_dt: datetime) -> None:
    note = (
        f"Date inferred: day {snapshot_date.day} from filename, "
        f"month/year from lastModifiedDateTime "
        f"({fallback_dt.year}-{fallback_dt.month:02d}). "
        f"Verify if correct."
    )
    execute(
        "UPDATE ingestion_files SET notes = %s WHERE id = %s",
        (note, file_id),
    )


def process_qty_file(
    run_id: int,
    branch: str,
    meta: dict[str, Any],
) -> dict:
    """
    Parse and load a single qty list file into stg_qty_list.
    Unchanged from V1.
    """
    stats = {"processed": 0, "failed": 0}

    path        = Path(meta["local_path"])
    fallback_dt = meta.get("sharepoint_last_modified")

    meta["file_type"]        = "qty_list"
    meta["file_hash"]        = compute_file_hash(path)
    meta["row_count"]        = count_rows(path)
    meta["is_canonical"]     = True
    meta["canonical_reason"] = "only_file"
    file_id = register_file(run_id, meta)

    try:
        snapshot_date, date_source = _parse_qty_date(meta["filename"], fallback_dt=fallback_dt)

        if date_source == "filename_day_inferred" and fallback_dt:
            _flag_inferred_date(file_id, snapshot_date, fallback_dt)
            logger.warning(
                f"[Qty][{branch}] {meta['filename']} — date inferred as "
                f"{snapshot_date} from day-only filename + lastModifiedDateTime. "
                f"Review if correct."
            )

        df = parse_qty_file(path)

        if df.empty:
            logger.warning(f"[Qty][{branch}] No data found in {path.name}")
            mark_file_status(file_id, "skipped")
            return stats

        with get_connection() as conn:
            n = _insert_qty_rows(
                conn, df, file_id, meta["filename"],
                branch, snapshot_date, date_source,
            )

        mark_file_status(file_id, "loaded")
        logger.info(
            f"[Qty][{branch}] Loaded {n:,} rows from {path.name} "
            f"(snapshot_date={snapshot_date}, source={date_source})"
        )
        stats["processed"] += 1

    except Exception as exc:
        logger.error(f"[Qty][{branch}] Failed to load {path.name}: {exc}")
        mark_file_status(file_id, "failed", str(exc))
        stats["failed"] += 1

    return stats
