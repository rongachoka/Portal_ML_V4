from __future__ import annotations

import hashlib
import logging
import re
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from Portal_ML_V4.sharepoint.db import bulk_insert, get_connection, insert_returning_id

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

DAY_SHEETS = [f"{i:02d}" for i in range(1, 32)]  # "01" … "31"

MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "may": 5, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def compute_file_hash(path: Path) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def count_rows(path: Path) -> int:
    """Quick row count — used for canonical file selection."""
    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            df = pd.read_csv(path, usecols=[0], dtype=str)
        elif suffix in (".xlsx", ".xlsm"):
            df = pd.read_excel(path, usecols=[0], dtype=str)
        else:
            return 0
        return len(df.dropna())
    except Exception as exc:
        logger.warning(f"Could not count rows for {path.name}: {exc}")
        return 0


def select_canonical_file(file_records: list[dict]) -> dict:
    """
    Given a list of ingestion_file metadata dicts for the same branch/report_type,
    return the one with the highest row_count.
    Tie-break: most recent sharepoint_last_modified.
    """
    valid = [f for f in file_records if (f.get("row_count") or 0) > 0]
    if not valid:
        raise ValueError("No valid files to select canonical from.")
    return max(
        valid,
        key=lambda f: (
            f["row_count"],
            f.get("sharepoint_last_modified") or datetime.min.replace(tzinfo=timezone.utc),
        ),
    )


def extract_month_year(filename: str, fallback_dt: datetime | None = None) -> tuple[int, int]:
    """
    Try to extract (month, year) from filename like 'Mar 2026' or 'March 2026'.
    Falls back to fallback_dt (e.g. sharepoint_last_modified) if not found.
    """
    match = re.search(r"([a-zA-Z]{3,})\s+(\d{4})", filename)
    if match:
        month_str = match.group(1)[:3].lower()
        year = int(match.group(2))
        month = MONTH_MAP.get(month_str)
        if month:
            return month, year

    if fallback_dt:
        return fallback_dt.month, fallback_dt.year

    now = datetime.now()
    return now.month, now.year


def _strip_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from column names."""
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
    return df


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


def _to_time(value):
    if pd.isna(value):
        return None
    if isinstance(value, datetime):
        return value.time()
    try:
        return pd.to_datetime(str(value)).time()
    except Exception:
        return None


# ── Sales parser ──────────────────────────────────────────────────────────────

def parse_sales_file(path: Path) -> pd.DataFrame:
    """
    Read a sales CSV or XLSX.
    Returns a DataFrame with normalised column names or raises on failure.
    """
    suffix = path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(path, dtype=str)
    elif suffix in (".xlsx", ".xlsm"):
        df = pd.read_excel(path, dtype=str)
    else:
        raise ValueError(f"Unsupported sales file type: {suffix}")

    df = _strip_columns(df)

    missing = [c for c in SALES_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Sales file {path.name} missing columns: {missing}")

    return df[SALES_COLUMNS].copy()


def _insert_sales_rows(conn, df: pd.DataFrame, source_file_id: int, branch: str) -> int:
    columns = [
        "source_file_id", "branch",
        "department", "category", "item", "description",
        "on_hand", "last_sold", "qty_sold", "total_tax_ex",
        "transaction_id", "date_sold",
    ]
    rows = []
    for _, row in df.iterrows():
        rows.append((
            source_file_id,
            branch,
            row.get("Department") or None,
            row.get("Category") or None,
            row.get("Item") or None,
            row.get("Description") or None,
            _to_numeric(row.get("On Hand")),
            _to_date(row.get("Last Sold")),
            _to_numeric(row.get("Qty Sold")),
            _to_numeric(row.get("Total (Tax Ex)")),
            str(row.get("Transaction ID") or "").strip() or None,
            _to_date(row.get("Date Sold")),
        ))
    return bulk_insert(conn, "stg_sales_reports", columns, rows)


# ── Cashier parser ────────────────────────────────────────────────────────────

def parse_cashier_file(
    path: Path,
    fallback_dt: datetime | None = None,
) -> pd.DataFrame:
    """
    Read all day sheets (01-31) from a cashier XLSM/XLSX.
    Derives transaction_date from sheet number + month/year in filename.
    Returns a concatenated DataFrame of all non-empty sheets.
    """
    month, year = extract_month_year(path.name, fallback_dt)

    xl = pd.ExcelFile(path)
    available_sheets = set(xl.sheet_names)

    frames = []
    for sheet_name in DAY_SHEETS:
        if sheet_name not in available_sheets:
            continue

        try:
            df = xl.parse(sheet_name, dtype=str)
        except Exception as exc:
            logger.warning(f"Could not read sheet {sheet_name} in {path.name}: {exc}")
            continue

        df = _strip_columns(df)

        # Skip empty or missing-column sheets
        present = [c for c in CASHIER_COLUMNS if c in df.columns]
        if not present:
            continue

        # Keep only the columns we need (ignore extras)
        missing = [c for c in CASHIER_COLUMNS if c not in df.columns]
        if missing:
            logger.warning(f"Sheet {sheet_name} in {path.name} missing: {missing} — filling with None")
            for col in missing:
                df[col] = None

        df = df[CASHIER_COLUMNS].copy()

        # Drop completely empty rows
        df = df.dropna(how="all")
        if df.empty:
            continue

        # Derive transaction_date from sheet number
        day = int(sheet_name)
        try:
            txn_date = date(year, month, day)
        except ValueError:
            # e.g. sheet 31 in a 30-day month
            logger.debug(f"Skipping sheet {sheet_name}: invalid date {year}-{month}-{day}")
            continue

        df["_transaction_date"] = txn_date
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def _insert_cashier_rows(conn, df: pd.DataFrame, source_file_id: int, branch: str) -> int:
    columns = [
        "source_file_id", "branch", "transaction_date",
        "receipt_txn_no", "amount", "txn_costs", "txn_time",
        "txn_type", "ordered_via", "client_name", "phone_number", "sales_rep",
    ]
    rows = []
    for _, row in df.iterrows():
        rows.append((
            source_file_id,
            branch,
            row.get("_transaction_date"),
            str(row.get("Receipt Txn No") or "").strip() or None,
            _to_numeric(row.get("Amount")),
            _to_numeric(row.get("Txn Costs")),
            _to_time(row.get("Time")),
            str(row.get("Txn Type") or "").strip() or None,
            str(row.get("Ordered Via") or "").strip() or None,
            str(row.get("Client Name") or "").strip() or None,
            str(row.get("Phone Number") or "").strip() or None,
            str(row.get("Sales Rep") or "").strip() or None,
        ))
    return bulk_insert(conn, "stg_cashier_reports", columns, rows)


# ── Ingestion run management ──────────────────────────────────────────────────

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
            finished_at     = NOW(),
            status          = %(status)s,
            files_seen      = %(files_seen)s,
            files_downloaded= %(files_downloaded)s,
            files_processed = %(files_processed)s,
            files_failed    = %(files_failed)s,
            notes           = %(notes)s
        WHERE id = %(run_id)s
    """
    from Portal_ML_V4.sharepoint.db import execute
    execute(sql, {**stats, "run_id": run_id})


def register_file(run_id: int, meta: dict[str, Any]) -> int:
    """Insert a row into ingestion_files and return its id."""
    sql = """
        INSERT INTO ingestion_files (
            run_id, branch, report_type, filename, file_extension,
            sharepoint_item_id, sharepoint_path, sharepoint_last_modified,
            sharepoint_size_bytes, local_path, file_hash, row_count,
            is_canonical, canonical_reason, downloaded_at, status
        ) VALUES (
            %(run_id)s, %(branch)s, %(report_type)s, %(filename)s, %(file_extension)s,
            %(sharepoint_item_id)s, %(sharepoint_path)s, %(sharepoint_last_modified)s,
            %(sharepoint_size_bytes)s, %(local_path)s, %(file_hash)s, %(row_count)s,
            %(is_canonical)s, %(canonical_reason)s, NOW(), 'pending'
        )
        RETURNING id
    """
    return insert_returning_id(sql, {**meta, "run_id": run_id})


def mark_file_status(file_id: int, status: str, error: str | None = None) -> None:
    from Portal_ML_V4.sharepoint.db import execute
    execute(
        "UPDATE ingestion_files SET status=%s, processed_at=NOW(), error_message=%s WHERE id=%s",
        (status, error, file_id),
    )


def mark_canonical(file_id: int) -> None:
    from Portal_ML_V4.sharepoint.db import execute
    execute(
        "UPDATE ingestion_files SET is_canonical=TRUE, canonical_reason='max_row_count' WHERE id=%s",
        (file_id,),
    )


# ── Top-level process functions ───────────────────────────────────────────────

def process_sales_branch(
    run_id: int,
    branch: str,
    file_metas: list[dict[str, Any]],
) -> dict:
    """
    Given all sales file metadata for a branch, select the canonical file
    and load it into stg_sales_reports.
    Returns stats dict.
    """
    stats = {"processed": 0, "failed": 0, "skipped": 0}

    # Register all files in ingestion_files
    registered = []
    for meta in file_metas:
        path = Path(meta["local_path"])
        meta["file_hash"] = compute_file_hash(path)
        meta["row_count"] = count_rows(path)
        meta["is_canonical"] = False
        meta["canonical_reason"] = None
        file_id = register_file(run_id, meta)
        registered.append({**meta, "file_id": file_id})

    # Pick canonical
    try:
        canonical = select_canonical_file(registered)
    except ValueError as exc:
        logger.error(f"[Sales][{branch}] Cannot select canonical: {exc}")
        stats["failed"] += len(registered)
        return stats

    mark_canonical(canonical["file_id"])

    # Mark non-canonical as skipped
    for rec in registered:
        if rec["file_id"] != canonical["file_id"]:
            mark_file_status(rec["file_id"], "skipped")
            stats["skipped"] += 1

    # Parse and load canonical file
    path = Path(canonical["local_path"])
    try:
        df = parse_sales_file(path)
        with get_connection() as conn:
            n = _insert_sales_rows(conn, df, canonical["file_id"], branch)
        mark_file_status(canonical["file_id"], "loaded")
        logger.info(f"[Sales][{branch}] Loaded {n} rows from {path.name}")
        stats["processed"] += 1
    except Exception as exc:
        logger.error(f"[Sales][{branch}] Failed to load {path.name}: {exc}")
        mark_file_status(canonical["file_id"], "failed", str(exc))
        stats["failed"] += 1

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
    meta["file_hash"] = compute_file_hash(path)
    meta["row_count"] = count_rows(path)
    meta["is_canonical"] = True
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
            n = _insert_cashier_rows(conn, df, file_id, branch)
        mark_file_status(file_id, "loaded")
        logger.info(f"[Cashier][{branch}] Loaded {n} rows from {path.name}")
        stats["processed"] += 1

    except Exception as exc:
        logger.error(f"[Cashier][{branch}] Failed to load {path.name}: {exc}")
        mark_file_status(file_id, "failed", str(exc))
        stats["failed"] += 1

    return stats
