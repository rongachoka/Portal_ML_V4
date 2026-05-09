"""
sharepoint_parser.py
====================
Reads downloaded sales, cashier, and qty list files, applies deduplication,
and loads clean data into:
    stg_sales_reports / stg_cashier_reports / stg_qty_list

Called by sharepoint_downloader.py after files are downloaded.

Sales logic:
  - Classifies each file as 'historical' or 'incremental'
  - Loads BOTH types and merges them (historical = full history,
    incremental = latest dates not yet in historical)
  - Deduplicates on: Transaction ID + Item + On Hand + Qty Sold + Date Sold
  - Incremental rows take priority on conflict
  - All files registered in ingestion_files regardless

Cashier logic:
  - Reads sheets 01-31 only (skips Summary, Sheet1 etc.)
  - Derives transaction_date from sheet number + month/year in filename
  - Time kept as plain string — not parsed (ambiguous formats e.g. '12.5')
  - Deduplication on cashier data happens downstream in load_to_postgres.py

Qty list logic:
  - Detects files named 'Item Quantity List' or 'QTY LIST' (case-insensitive)
  - Strips junk header rows (lines starting with ';') before parsing
  - Date parsed from filename: full date > day-only > lastModifiedDateTime
  - Day-only filenames flagged in ingestion_files.notes for manual review
  - Loads into stg_qty_list; fact_inventory_snapshot built by load_to_postgres.py
"""

from __future__ import annotations

import hashlib
import logging
import re
import io
import os
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from Portal_ML_V4.sharepoint.db import bulk_insert, execute, get_connection, insert_returning_id

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

logger = logging.getLogger(__name__)


# ── Column definitions ────────────────────────────────────────────────────────

SALES_COLUMNS = [
    "Department",
    "Category",
    "Item",
    "Description",
    "On Hand",
    "Last Sold",
    "Qty Sold",
    "Total (Tax Ex)",
    "Transaction ID",
    "Date Sold",
]

CASHIER_COLUMNS = [
    "Receipt Txn No",
    "Amount",
    "Txn Costs",
    "Time",
    "Txn Type",
    "Ordered Via",
    "Client Name",
    "Phone Number",
    "Sales Rep",
]

# Composite dedup key for sales line items
# Ensures same item on same transaction with same qty/stock = duplicate
SALES_DEDUP_COLUMNS = [
    "Transaction ID",
    "Item",
    "On Hand",
    "Qty Sold",
    "Date Sold",
]

DAY_SHEETS = [f"{i:02d}" for i in range(1, 32)]  # "01" … "31"

MONTH_MAP = {
    "jan": 1,  "feb": 2,  "mar": 3,  "apr": 4,
    "may": 5,  "jun": 6,  "jul": 7,  "aug": 8,
    "sep": 9,  "oct": 10, "nov": 11, "dec": 12,
}


# ── File helpers ──────────────────────────────────────────────────────────────

# def compute_file_hash(path: Path) -> str:
#     """SHA-256 of file bytes. Reads in 1 MB chunks — safe for large files."""
#     sha = hashlib.sha256()
#     with open(path, "rb") as f:
#         for chunk in iter(lambda: f.read(1024 * 1024), b""):
#             sha.update(chunk)
#     return sha.hexdigest()


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

def compute_file_hash(path: Path) -> str:
    """
    SHA-256 of file bytes. Reads in 1MB chunks.
    For files > 100MB uses filename + size + mtime to avoid MemoryError.
    """
    try:
        file_size = os.path.getsize(path)

        # Large file protection — avoid MemoryError on huge cashier files
        if file_size > 100 * 1024 * 1024:  # > 100MB
            stat = os.stat(path)
            identifier = f"{path.name}_{stat.st_size}_{stat.st_mtime}"
            return hashlib.sha256(identifier.encode()).hexdigest() + "_partial"

        # Normal files — full SHA-256
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


def classify_sales_file(filename: str) -> str:
    """
    Classify a sales file as 'historical' or 'incremental' based on filename.

    Historical patterns:  year range in name e.g. 'Jan 2023-Feb 2026',
                          '2023-2026'
    Incremental patterns: single date e.g. '150326', '14 Mar', 'sales 12 March'
    """
    name = filename.lower()

    # Year range → historical aggregate file
    if re.search(r'\d{4}\s*[-–]\s*\d{4}', name):
        return "historical"

    # Month-year range spanning two different months
    # e.g. "jan 2023-feb 2026"
    month_pattern = r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)'
    if re.search(month_pattern + r'.+\d{4}.+' + month_pattern, name):
        return "historical"

    return "incremental"


# ── Type coercion helpers ─────────────────────────────────────────────────────

def _strip_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from column names."""
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
    return df


def _clean_date_sold(value) -> date | None:
    """
    Parse Date Sold from POS format.
    Strips leading/trailing # characters before parsing.
    e.g. '#2025-06-01 10:15:22#' → date(2025, 6, 1)
    """
    if pd.isna(value):
        return None
    cleaned = str(value).strip().strip("#").strip()
    try:
        return pd.to_datetime(cleaned).date()
    except Exception:
        return None


def _to_date(value) -> date | None:
    """Generic date parser for non-Date Sold columns (e.g. Last Sold)."""
    if pd.isna(value):
        return None
    if isinstance(value, (date, datetime)):
        return value.date() if isinstance(value, datetime) else value
    try:
        return pd.to_datetime(value).date()
    except Exception:
        return None


def _to_numeric(value):
    """Parse numeric — strips commas and whitespace. Returns None on failure."""
    if pd.isna(value):
        return None
    try:
        return float(str(value).replace(",", "").strip())
    except Exception:
        return None


def _to_str(value) -> str | None:
    """
    Clean string — strips whitespace, returns None for empty.
    Also strips .0 float artefact that pandas introduces when reading
    numeric-looking strings e.g. '712345678.0' → '712345678'.
    Affects phone numbers, receipt numbers, transaction IDs.
    """
    if pd.isna(value):
        return None
    s = str(value).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s if s else None


# ── Sales parser ──────────────────────────────────────────────────────────────

def parse_sales_file(path: Path) -> pd.DataFrame:
    """
    Read a sales CSV or XLSX.
    - Strips column name whitespace
    - Validates all required columns are present
    - Cleans Date Sold (strips # characters)
    - Returns DataFrame with original column names intact
      (renaming happens at insert time)
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

    # Clean Date Sold at parse time so dedup comparisons are consistent
    df["Date Sold"] = df["Date Sold"].apply(_clean_date_sold)

    # Normalise numeric dedup columns to avoid float precision mismatches
    # e.g. "5.0" vs "5" would be treated as different without this
    df["On Hand"]  = pd.to_numeric(df["On Hand"],  errors="coerce")
    df["Qty Sold"] = pd.to_numeric(df["Qty Sold"], errors="coerce")

    # Strip whitespace from text dedup columns
    df["Transaction ID"] = df["Transaction ID"].astype(str).str.strip()
    df["Item"]           = df["Item"].astype(str).str.strip()

    return df


def merge_sales_files(file_records: list[dict]) -> tuple[pd.DataFrame, list[int]]:
    """
    Load and merge all sales files for a branch.

    - Historical file loaded first (base layer — full history)
    - Incremental file(s) loaded on top (latest dates)
    - Deduplication on SALES_DEDUP_COLUMNS
    - Incremental rows take priority on conflict (sort + keep='last')

    Returns:
        merged DataFrame with '_source_file_id' and '_source_filename' columns
        list of file_ids that contributed rows (all marked canonical)
    """
    historical  = [f for f in file_records if f.get("file_type") == "historical"]
    incremental = [f for f in file_records if f.get("file_type") == "incremental"]

    frames = []
    contributing_ids = []

    # Load historical first — it's the base layer
    for rec in historical:
        try:
            df = parse_sales_file(Path(rec["local_path"]))
            df["_source_file_id"]   = rec["file_id"]
            df["_source_filename"]  = rec["filename"]
            df["_file_type"]        = "historical"
            frames.append(df)
            contributing_ids.append(rec["file_id"])
            logger.info(f"[Sales] Loaded historical: {rec['filename']} ({len(df):,} rows)")
        except Exception as exc:
            logger.error(f"[Sales] Failed to load historical {rec['filename']}: {exc}")
            mark_file_status(rec["file_id"], "failed", str(exc))

    # Load incremental on top
    for rec in incremental:
        try:
            df = parse_sales_file(Path(rec["local_path"]))
            df["_source_file_id"]   = rec["file_id"]
            df["_source_filename"]  = rec["filename"]
            df["_file_type"]        = "incremental"
            frames.append(df)
            contributing_ids.append(rec["file_id"])
            logger.info(f"[Sales] Loaded incremental: {rec['filename']} ({len(df):,} rows)")
        except Exception as exc:
            logger.error(f"[Sales] Failed to load incremental {rec['filename']}: {exc}")
            mark_file_status(rec["file_id"], "failed", str(exc))

    if not frames:
        return pd.DataFrame(), []

    combined = pd.concat(frames, ignore_index=True)

    # Sort so incremental rows come last → keep='last' gives them priority
    combined["_type_order"] = combined["_file_type"].map(
        {"historical": 0, "incremental": 1}
    )
    combined = combined.sort_values("_type_order")

    # Deduplicate
    missing_dedup = [c for c in SALES_DEDUP_COLUMNS if c not in combined.columns]
    if missing_dedup:
        logger.warning(f"Missing dedup columns {missing_dedup} — skipping deduplication")
    else:
        before = len(combined)
        combined = combined.drop_duplicates(subset=SALES_DEDUP_COLUMNS, keep="last")
        dropped = before - len(combined)
        if dropped > 0:
            logger.info(f"[Sales] Deduplication removed {dropped:,} duplicate rows")

    combined = combined.drop(columns=["_type_order", "_file_type"])

    return combined, contributing_ids


def _insert_sales_rows(conn, df: pd.DataFrame, branch: str) -> int:
    """
    Insert merged sales rows into stg_sales_reports.
    source_file_id and source_filename are read per-row from the DataFrame
    (set during merge_sales_files so each row traces to its origin file).
    """
    columns = [
        "source_file_id", "source_filename", "branch",
        "department", "category", "item", "description",
        "on_hand", "last_sold", "qty_sold", "total_tax_ex",
        "transaction_id", "date_sold",
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
            # Already a date object from _clean_date_sold in parse_sales_file
        ))
    return bulk_insert(conn, "stg_sales_reports", columns, rows)


# ── Cashier parser ────────────────────────────────────────────────────────────

CASHIER_COLUMN_POSITIONS = {
    0: "Receipt Txn No",
    1: "Amount",
    2: "Txn Costs", 
    3: "Time",
    4: "Txn Type",
    5: "Ordered Via",
    6: "Client Name",
    7: "Phone Number",
    8: "Sales Rep",
}

def extract_month_year(filename: str, fallback_dt: datetime | None = None) -> tuple[int, int]:
    """
    Extract (month, year) from filename e.g. 'Mar 2026' or 'March 2026'.
    Falls back to fallback_dt (sharepoint_last_modified) if not found.
    Falls back to current month/year if neither available.
    """
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


def _get_sheet_day(sheet_name: str, expected_month: int, expected_year: int) -> int | None:
    """
    Extract the day number from a sheet name.
    Handles two formats:
        '01'            → day 1  (standard format)
        '01-05-2025'    → day 1  (full date format, validates month/year match)
    Returns None if the sheet name doesn't match either format.
    """
    # Standard format: '01' to '31'
    if re.fullmatch(r'0[1-9]|[12][0-9]|3[01]', sheet_name.strip()):
        return int(sheet_name.strip())

    # Full date format: 'DD-MM-YYYY'
    m = re.fullmatch(r'(\d{2})-(\d{2})-(\d{4})', sheet_name.strip())
    if m:
        day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        # Only accept if month and year match the file's expected period
        if month == expected_month and year == expected_year:
            return day
        else:
            logger.debug(
                f"Sheet '{sheet_name}' date {day}-{month}-{year} doesn't match "
                f"expected {expected_month}/{expected_year} — skipping"
            )
            return None

    return None


def parse_cashier_file(
    path: Path,
    fallback_dt: datetime | None = None,
) -> pd.DataFrame:
    """
    Read all day sheets (01-31) from a cashier XLSM/XLSX.
    - Skips sheets not named 01-31
    - Skips empty sheets
    - Derives transaction_date from sheet number + filename month/year
    - Time kept as plain string — not parsed
    Returns concatenated DataFrame of all non-empty sheets.
    """
    month, year = extract_month_year(path.name, fallback_dt)

    xl               = pd.ExcelFile(path, engine="openpyxl")

    frames = []
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

        # Skip if no cashier columns at all
        present = [c for c in CASHIER_COLUMNS if c in df.columns]
        if not present:
            continue

        # ── Positional fallback for mistyped/missing column names ─────────
        # If any expected columns are missing by name but the sheet has
        # enough columns, rename all unrecognised columns by position
        missing_by_name = [c for c in CASHIER_COLUMNS if c not in df.columns]
        if missing_by_name and len(df.columns) >= len(CASHIER_COLUMN_POSITIONS):
            rename_map = {
                df.columns[pos]: name
                for pos, name in CASHIER_COLUMN_POSITIONS.items()
                if pos < len(df.columns)
                and df.columns[pos] not in CASHIER_COLUMNS
            }
            if rename_map:
                df = df.rename(columns=rename_map)
                logger.info(
                    f"Sheet {sheet_name} in {path.name} — "
                    f"positional rename applied for {list(rename_map.values())}"
                )

        # ─────────────────────────────────────────────────────────────────

        # Fill missing columns with None rather than failing the whole file
        for col in CASHIER_COLUMNS:
            if col not in df.columns:
                logger.warning(
                    f"Sheet {sheet_name} in {path.name} missing '{col}' — filling with None"
                )
                df[col] = None

        df = df[CASHIER_COLUMNS].copy()

        # Drop completely empty rows
        df = df.dropna(how="all")
        if df.empty:
            continue

        try:
            txn_date = date(year, month, day)
        except ValueError:
            logger.debug(
                f"Skipping sheet {sheet_name}: invalid date {year}-{month}-{day}"
            )
            continue

        df["_transaction_date"] = txn_date
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def _insert_cashier_rows(
    conn,
    df: pd.DataFrame,
    source_file_id: int,
    source_filename: str,
    branch: str,
) -> int:
    """
    Insert cashier rows into stg_cashier_reports.
    Time is stored as a plain string — no parsing applied.
    """
    columns = [
        "source_file_id", "source_filename", "branch", "transaction_date",
        "receipt_txn_no", "amount", "txn_costs", "txn_time",
        "txn_type", "ordered_via", "client_name", "phone_number", "sales_rep",
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
            _to_str(row.get("Time")),       # plain string — no time parsing
            _to_str(row.get("Txn Type")),
            _to_str(row.get("Ordered Via")),
            _to_str(row.get("Client Name")),
            _to_str(row.get("Phone Number")),
            _to_str(row.get("Sales Rep")),
        ))
    return bulk_insert(conn, "stg_cashier_reports", columns, rows)


# ── Ingestion run management ──────────────────────────────────────────────────

def start_run() -> int:
    """Create a new ingestion_runs row and return its id."""
    sql = """
        INSERT INTO ingestion_runs (pipeline_name, started_at, status)
        VALUES ('sharepoint_ingestion', NOW(), 'running')
        RETURNING id
    """
    return insert_returning_id(sql)


def finish_run(run_id: int, stats: dict) -> None:
    """Update ingestion_runs row with final status and counts."""
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
    """
    Insert a row into ingestion_files and return its id.
    meta must contain all keys matching the INSERT columns below.
    """
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
        RETURNING id
    """
    return insert_returning_id(sql, {**meta, "run_id": run_id})


def mark_file_status(file_id: int, status: str, error: str | None = None) -> None:
    """Update status and processed_at on an ingestion_files row."""
    execute(
        "UPDATE ingestion_files SET status=%s, processed_at=NOW(), error_message=%s WHERE id=%s",
        (status, error, file_id),
    )


def mark_canonical(file_id: int, reason: str) -> None:
    """Flag a file as canonical with the given reason."""
    execute(
        "UPDATE ingestion_files SET is_canonical=TRUE, canonical_reason=%s WHERE id=%s",
        (reason, file_id),
    )


# ── Top-level process functions ───────────────────────────────────────────────

def process_sales_branch(
    run_id: int,
    branch: str,
    file_metas: list[dict[str, Any]],
) -> dict:
    """
    Process all sales files for a branch:
      1. Classify each as historical or incremental
      2. Register all in ingestion_files
      3. Merge historical + incremental, deduplicate
      4. Load merged result into stg_sales_reports
      5. Mark all contributing files as canonical

    Returns stats dict.
    """
    stats = {"processed": 0, "failed": 0, "skipped": 0}

    # Classify, hash, count, register all files
    registered = []
    for meta in file_metas:
        path = Path(meta["local_path"])
        meta["file_type"]       = classify_sales_file(meta["filename"])
        meta["file_hash"]       = compute_file_hash(path)
        meta["row_count"]       = count_rows(path)
        meta["is_canonical"]    = False
        meta["canonical_reason"] = meta["file_type"]
        file_id = register_file(run_id, meta)
        registered.append({**meta, "file_id": file_id})

    # Merge historical + incremental
    try:
        merged_df, contributing_ids = merge_sales_files(registered)
    except Exception as exc:
        logger.error(f"[Sales][{branch}] merge_sales_files failed: {exc}")
        for rec in registered:
            mark_file_status(rec["file_id"], "failed", str(exc))
        stats["failed"] += len(registered)
        return stats

    if merged_df.empty:
        logger.warning(f"[Sales][{branch}] No data after merge")
        for rec in registered:
            mark_file_status(rec["file_id"], "skipped")
        stats["skipped"] += len(registered)
        return stats

    # Mark non-contributing files as skipped
    contributing_set = set(contributing_ids)
    for rec in registered:
        if rec["file_id"] not in contributing_set:
            mark_file_status(rec["file_id"], "skipped")
            stats["skipped"] += 1

    # Load merged result
    try:
        with get_connection() as conn:
            n = _insert_sales_rows(conn, merged_df, branch)

        # Mark all contributing files as canonical + loaded
        for file_id in contributing_ids:
            file_type = next(
                (r["file_type"] for r in registered if r["file_id"] == file_id),
                "unknown"
            )
            mark_canonical(file_id, file_type)
            mark_file_status(file_id, "loaded")

        logger.info(
            f"[Sales][{branch}] Loaded {n:,} rows from "
            f"{len(contributing_ids)} file(s)"
        )
        stats["processed"] += len(contributing_ids)

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
    Returns stats dict.
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


# ── Qty list column definitions ───────────────────────────────────────────────

QTY_COLUMNS = [
    "Department",
    "Category",
    "Item Lookup Code",
    "Description",
    "On-Hand",
    "Committed",
    "Reorder Pt.",
    "Restock Lvl.",
    "Qty to Order",
    "Supplier",
    "Reorder No.",
]

# Columns we must have at minimum to consider the file valid
QTY_REQUIRED = {"Item Lookup Code", "Description", "On-Hand"}


# ── Qty list date parsing ─────────────────────────────────────────────────────

def _parse_qty_date(
    filename: str,
    fallback_dt: datetime | None = None,
) -> tuple[date | None, str]:
    """
    Extract snapshot date from a qty list filename.

    Priority:
        1. Full date  e.g. '04.03.26' or '04.03.2026' → date(2026, 3, 4)
        2. Day only   e.g. '15' → date(fallback.year, fallback.month, 15)
           ⚠ Flagged as 'filename_day_inferred'
        3. No date    → fallback_dt.date() or today

    Returns:
        (snapshot_date, snapshot_date_source)
    """
    name = filename

    # 1. Full date: DD.MM.YY or DD.MM.YYYY
    m = re.search(r'(\d{2})\.(\d{2})\.(\d{2,4})', name)
    if m:
        day, month, year_raw = int(m.group(1)), int(m.group(2)), m.group(3)
        year = int(year_raw) + 2000 if len(year_raw) == 2 else int(year_raw)
        try:
            return date(year, month, day), 'filename_full'
        except ValueError:
            pass

    # 2. Day only: standalone 1-2 digit number in the filename
    # Exclude matches that are part of longer numbers (e.g. year 2026)
    m = re.search(r'(?<![\d])([0-2]?[0-9]|3[01])(?![\d])', name)
    if m:
        day = int(m.group(1))
        if 1 <= day <= 31 and fallback_dt:
            try:
                inferred = date(fallback_dt.year, fallback_dt.month, day)
                return inferred, 'filename_day_inferred'
            except ValueError:
                pass

    # 3. Full fallback
    if fallback_dt:
        return fallback_dt.date(), 'lastmodified'

    return date.today(), 'lastmodified'


# ── Qty list file parser ──────────────────────────────────────────────────────

def _detect_separator(lines: list[str]) -> str:
    """
    Detect whether a qty CSV uses tabs or commas as separator.
    Looks at the first non-semicolon, non-empty line and counts
    tabs vs commas. Defaults to tab since qty files are tab-separated.
    """
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
    """
    Read a qty list CSV, stripping junk header rows.

    Some files have leading lines like:
        ;Item Quantity List
        ;Filter:(Inactive  =  No)
        ;Department  Category  ...

    These are detected by a leading semicolon and removed.
    The real header row is the first non-semicolon line.
    """
    # Try utf-8 first, fall back to cp1252
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
            # Only keep semicolon-prefixed lines that look like real headers
            # Skip junk lines like ';Item Quantity List' and ';Filter:...'
            if "Department" in content or "Item Lookup Code" in content:
                data_lines.append(stripped[1:])
            # else: drop the junk line entirely
        else:
            data_lines.append(line)

    sep = _detect_separator(data_lines)

    import io
    cleaned = "".join(data_lines)
    try:
        df = pd.read_csv(io.StringIO(cleaned), dtype=str, sep=sep, low_memory=False)
    except Exception:
        df = pd.read_csv(path, dtype=str, sep=sep, on_bad_lines="skip")

    return df


def parse_qty_file(path: Path) -> pd.DataFrame:
    """
    Read a qty list CSV or XLSX.
    - Strips junk header rows (lines starting with ';')
    - Validates required columns are present
    - Returns DataFrame with original column names intact
    """
    suffix = path.suffix.lower()

    # if suffix == ".csv":
    #     try:
    #         df = _clean_qty_csv(path)
    #     except UnicodeDecodeError:
    #         # Retry with cp1252 encoding
    #         import io
    #         with open(path, "r", encoding="cp1252", errors="replace") as f:
    #             lines = f.readlines()
    #         data_lines = []
    #         for line in lines:
    #             stripped = line.lstrip()
    #             data_lines.append(stripped[1:] if stripped.startswith(";") else line)
    #         df = pd.read_csv(io.StringIO("".join(data_lines)), dtype=str, sep='\t',
    #                          low_memory=False)

    if suffix == ".csv":
        df = _clean_qty_csv(path)

    elif suffix in (".xlsx", ".xlsm"):
        # Excel files don't have the semicolon issue — read normally
        # but check first few rows for junk and skip if needed
        df_check = pd.read_excel(path, dtype=str, nrows=5, engine="openpyxl")
        # If first column of first row looks like a junk header, skip rows
        first_val = str(df_check.iloc[0, 0] if not df_check.empty else "").strip()
        if "quantity list" in first_val.lower() or "filter" in first_val.lower():
            df = pd.read_excel(path, dtype=str, skiprows=2, engine="openpyxl")
        else:
            df = pd.read_excel(path, dtype=str, engine="openpyxl")
    else:
        raise ValueError(f"Unsupported qty file type: {suffix}")

    df = _strip_columns(df)

    # Validate minimum required columns
    missing_required = QTY_REQUIRED - set(df.columns)
    if missing_required:
        raise ValueError(
            f"Qty file {path.name} missing required columns: {missing_required}"
        )

    # Fill any optional missing columns with None
    for col in QTY_COLUMNS:
        if col not in df.columns:
            logger.warning(f"[Qty] {path.name} missing optional column '{col}' — filling None")
            df[col] = None

    df = df[QTY_COLUMNS].copy()

    # Drop completely empty rows
    df = df.dropna(how="all")

    # Normalise numeric columns
    for col in ["On-Hand", "Committed", "Reorder Pt.", "Restock Lvl.", "Qty to Order"]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", "", regex=False),
                errors="coerce",
            )

    return df


def _insert_qty_rows(
    conn,
    df: pd.DataFrame,
    source_file_id: int,
    source_filename: str,
    branch: str,
    snapshot_date: date,
    snapshot_date_source: str,
) -> int:
    """Insert qty list rows into stg_qty_list."""
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
            source_file_id,
            source_filename,
            branch,
            snapshot_date,
            snapshot_date_source,
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
    """
    Write a note to ingestion_files when the snapshot date was inferred
    from a day-only filename. Allows easy querying of ambiguous dates.
    """
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


# ── Top-level qty process function ────────────────────────────────────────────

def process_qty_file(
    run_id: int,
    branch: str,
    meta: dict[str, Any],
) -> dict:
    """
    Parse and load a single qty list file into stg_qty_list.

    Date parsing priority:
        1. Full date in filename e.g. 04.03.26
        2. Day only in filename + month/year from lastModifiedDateTime  ⚠ flagged
        3. Full fallback to lastModifiedDateTime

    Returns stats dict.
    """
    stats = {"processed": 0, "failed": 0}

    path       = Path(meta["local_path"])
    fallback_dt = meta.get("sharepoint_last_modified")

    meta["file_type"]        = "qty_list"
    meta["file_hash"]        = compute_file_hash(path)
    meta["row_count"]        = count_rows(path)
    meta["is_canonical"]     = True
    meta["canonical_reason"] = "only_file"
    file_id = register_file(run_id, meta)

    try:
        # Parse date from filename
        snapshot_date, date_source = _parse_qty_date(
            meta["filename"], fallback_dt=fallback_dt
        )

        # Flag ambiguous date in ingestion_files
        if date_source == "filename_day_inferred" and fallback_dt:
            _flag_inferred_date(file_id, snapshot_date, fallback_dt)
            logger.warning(
                f"[Qty][{branch}] {meta['filename']} — date inferred as "
                f"{snapshot_date} from day-only filename + lastModifiedDateTime. "
                f"Review if correct."
            )

        # Parse file
        df = parse_qty_file(path)

        if df.empty:
            logger.warning(f"[Qty][{branch}] No data found in {path.name}")
            mark_file_status(file_id, "skipped")
            return stats

        # Insert into staging
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