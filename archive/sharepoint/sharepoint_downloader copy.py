"""
sharepoint_downloader.py
========================
Downloads sales, cashier, and qty list files from SharePoint.
Passes downloaded files to sharepoint_parser.py for staging load.

Key changes from V1:
  - Combined historical files (e.g. 'Galleria Jan 2023-March 2026 Sales.xlsx')
    are NO LONGER skipped. They are downloaded and passed to the parser,
    which uses canonical election to decide how to use them.
  - Branch watermark is read from DB before processing and logged at startup
    so you can see at a glance what's already loaded vs what's new.
  - should_download() logic unchanged — file size + mtime check prevents
    re-downloading unchanged files.

Sales pipeline flow:
  downloader:
    1. Scan SharePoint branch folders
    2. Download new/changed files (size+mtime check)
    3. Pass all files to process_sales_branch()

  parser (process_sales_branch):
    1. Read branch watermark
    2. Peek inside each file to find max Date Sold
    3. Elect canonical file (highest max Date Sold)
    4. Skip files fully covered by watermark
    5. Load canonical + post-watermark incrementals
    6. Update watermark
"""

from __future__ import annotations

import logging
import os
import re
import pandas as pd
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from Portal_ML_V4.sharepoint.sharepoint_client import SharePointClient
from Portal_ML_V4.sharepoint.sharepoint_parser import (
    finish_run,
    get_branch_watermark,
    process_cashier_file,
    process_qty_file,
    process_sales_branch,
    start_run,
)
from Portal_ML_V4.src.config.settings import RAW_DATA_DIR, DRIVE_ID, PROCESSED_DATA_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

LOCAL_BASE_DIR = RAW_DATA_DIR / "sharepoint_downloads"

SALES_COMBINED_FILE   = PROCESSED_DATA_DIR / "pos_data" / "sharepoint_sales_combined.csv"
CASHIER_COMBINED_FILE = PROCESSED_DATA_DIR / "pos_data" / "sharepoint_cashier_combined.csv"


BRANCHES = [
    "Galleria",
    "ABC",
    "Milele",
    "Portal 2R",
    "Portal CBD",
    "Centurion 2R",
]

SALE_EXTENSIONS    = {".csv", ".xlsx"}
CASHIER_EXTENSIONS = {".xlsm"}
QTY_EXTENSIONS     = {".csv", ".xlsx"}

_MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    "january": 1, "february": 2, "march": 3, "april": 4, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10,
    "november": 11, "december": 12,
}

# Maps SharePoint branch names → fact table location labels (for watermark display)
_BRANCH_TO_LOCATION = {
    "ABC":          "PHARMART_ABC",
    "Centurion 2R": "CENTURION_2R",
    "Galleria":     "GALLERIA",
    "Milele":       "NGONG_MILELE",
    "Portal 2R":    "PORTAL_2R",
    "Portal CBD":   "PORTAL_CBD",
}


# ── Date Parsing ──────────────────────────────────────────────────────────────

def parse_effective_date(filename: str, last_modified: datetime | None) -> datetime:
    """
    Attempts to extract an effective date from a sales/qty filename.
    Falls back to lastModifiedDateTime if no date can be confidently parsed.

    Handles real-world formats across all six branches:
        SALES REPORT 18.03.26       → 2026-03-18  (DD.MM.YY)
        180326                      → 2026-03-18  (DDMMYY)
        S090326                     → 2026-03-09  (S + DDMMYY)
        6th Mar sales               → 2026-03-06  (day + month name)
        13th sales                  → fallback     (day only, no month)
        sales 05 march              → 2026-03-05  (word + day + month)
        sales060326                 → 2026-03-06  (word + DDMMYY)
        sales70326                  → 2026-03-07  (word + DMMYY)
        sales08 / sales15th         → fallback     (day only)
        salesmarch02                → 2026-03-02  (word + month + day)
        sales14march                → 2026-03-14  (word + day + month)
        sales 06032026              → 2026-03-06  (word + DDMMYYYY)
        05sales                     → fallback     (day only)
        06 mar sales 2026           → 2026-03-06  (day + month + year)
        QTY LIST 03.03.2026.xlsx    → 2026-03-03  (DD.MM.YYYY)
        Galleria Jan 2023-March 2026 Sales → fallback (range — uses lastModified)
    """
    name       = Path(filename).stem.lower()
    name_clean = re.sub(r'(\d+)(st|nd|rd|th)\b', r'\1', name)

    # Pattern 1: DD.MM.YYYY or DD.MM.YY
    m = re.search(r'\b(\d{1,2})\.(\d{1,2})\.(\d{2,4})\b', name_clean)
    if m:
        day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        year = _fix_year(year)
        if _valid_date(day, month, year):
            return datetime(year, month, day, tzinfo=timezone.utc)

    # Pattern 2: DD/MM/YYYY or DD/MM/YY
    m = re.search(r'\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b', name_clean)
    if m:
        day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        year = _fix_year(year)
        if _valid_date(day, month, year):
            return datetime(year, month, day, tzinfo=timezone.utc)

    # Pattern 3: 8-digit DDMMYYYY
    m = re.search(r'(?<!\d)(\d{2})(\d{2})(\d{4})(?!\d)', name_clean)
    if m:
        day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if _valid_date(day, month, year):
            return datetime(year, month, day, tzinfo=timezone.utc)

    # Pattern 4: 6-digit DDMMYY
    m = re.search(r'(?<!\d)(\d{2})(\d{2})(\d{2})(?!\d)', name_clean)
    if m:
        day, month, year = int(m.group(1)), int(m.group(2)), _fix_year(int(m.group(3)))
        if _valid_date(day, month, year):
            return datetime(year, month, day, tzinfo=timezone.utc)

    # Pattern 5: 5-digit DMMYY
    m = re.search(r'(?<!\d)(\d{1})(\d{2})(\d{2})(?!\d)', name_clean)
    if m:
        day, month, year = int(m.group(1)), int(m.group(2)), _fix_year(int(m.group(3)))
        if _valid_date(day, month, year):
            return datetime(year, month, day, tzinfo=timezone.utc)

    # Pattern 6: day + month name
    m = re.search(
        r'\b(\d{1,2})\s*(' + '|'.join(_MONTH_MAP.keys()) + r')\b', name_clean
    )
    if m:
        day   = int(m.group(1))
        month = _MONTH_MAP[m.group(2)]
        year  = _year_from_fallback(last_modified)
        if _valid_date(day, month, year):
            return datetime(year, month, day, tzinfo=timezone.utc)

    # Pattern 7: month name + day
    m = re.search(
        r'\b(' + '|'.join(_MONTH_MAP.keys()) + r')\s*(\d{1,2})\b', name_clean
    )
    if m:
        month = _MONTH_MAP[m.group(1)]
        day   = int(m.group(2))
        year  = _year_from_fallback(last_modified)
        if _valid_date(day, month, year):
            return datetime(year, month, day, tzinfo=timezone.utc)

    # Fallback: lastModifiedDateTime
    if last_modified:
        logger.debug(
            f"[DateParser] No date in '{filename}' — using lastModified: "
            f"{last_modified.date()}"
        )
        return last_modified

    return datetime(2000, 1, 1, tzinfo=timezone.utc)


def _fix_year(y: int) -> int:
    if y < 100:
        return 2000 + y if y < 50 else 1900 + y
    return y


def _valid_date(day: int, month: int, year: int) -> bool:
    try:
        datetime(year, month, day)
        return 2020 <= year <= 2035 and 1 <= month <= 12 and 1 <= day <= 31
    except ValueError:
        return False


def _year_from_fallback(last_modified: datetime | None) -> int:
    if last_modified:
        return last_modified.year
    return datetime.now(tz=timezone.utc).year


# ── Helpers ───────────────────────────────────────────────────────────────────

def is_template_file(filename: str) -> bool:
    return "template" in filename.lower()


def is_qty_file(filename: str) -> bool:
    name = filename.lower()
    return "qty list" in name or "item quantity list" in name


def find_root_folder_by_name(client: SharePointClient, folder_name: str) -> dict | None:
    target = folder_name.lower().replace(" ", "")
    for item in client.list_root_children():
        if not client.is_folder(item):
            continue
        actual = item["name"].lower().replace(" ", "")
        if actual == target:
            return item
    return None


def normalize_branch_name(name: str) -> str:
    return re.sub(r"\s+", " ", name).strip()


def detect_branch_from_filename(filename: str) -> str | None:
    lower_name = filename.lower()
    for branch in BRANCHES:
        if branch.lower() in lower_name:
            return branch
    return None


def ensure_branch_folder(branch: str) -> Path:
    return LOCAL_BASE_DIR / branch


def should_download(local_path: Path, remote_item: dict) -> bool:
    if not local_path.exists():
        return True

    remote_size = int(remote_item.get("size", 0))
    local_size  = local_path.stat().st_size
    if remote_size != local_size:
        return True

    remote_modified_raw = remote_item.get("lastModifiedDateTime")
    if not remote_modified_raw:
        return False

    remote_modified = datetime.fromisoformat(
        remote_modified_raw.replace("Z", "+00:00")
    )
    local_modified = datetime.fromtimestamp(
        local_path.stat().st_mtime, tz=timezone.utc
    )
    return remote_modified > local_modified


def stamp_local_mtime(local_path: Path, remote_item: dict) -> None:
    remote_modified_raw = remote_item.get("lastModifiedDateTime")
    if not remote_modified_raw:
        return
    remote_modified = datetime.fromisoformat(
        remote_modified_raw.replace("Z", "+00:00")
    )
    ts = remote_modified.timestamp()
    local_path.touch(exist_ok=True)
    os.utime(local_path, (ts, ts))


def _parse_sharepoint_dt(raw: str | None) -> datetime | None:
    if not raw:
        return None
    return datetime.fromisoformat(raw.replace("Z", "+00:00"))


def _build_meta(
    item: dict,
    local_path: Path,
    branch: str,
    report_type: str,
    sharepoint_path: str,
    newly_downloaded: bool,
) -> dict:
    return {
        "branch":                   branch,
        "report_type":              report_type,
        "filename":                 item["name"],
        "file_extension":           Path(item["name"]).suffix.lower(),
        "sharepoint_item_id":       item.get("id"),
        "sharepoint_path":          sharepoint_path,
        "sharepoint_last_modified": _parse_sharepoint_dt(
                                        item.get("lastModifiedDateTime")
                                    ),
        "sharepoint_size_bytes":    int(item.get("size", 0)),
        "local_path":               str(local_path),
        "newly_downloaded":         newly_downloaded,
        "file_hash":                None,
        "row_count":                None,
        "is_canonical":             False,
        "canonical_reason":         None,
        "file_type":                None,
    }


# ── Cashier download (unchanged) ──────────────────────────────────────────────

def download_cashier_reports(client: SharePointClient) -> list[dict]:
    root_path = "Finance Reports/Cashier Reports"
    logger.info(f"[Cashier] Scanning: {root_path}")

    all_metas: list[dict] = []
    root_items = client.list_children_by_path(root_path)

    for item in root_items:
        item_name = item["name"]
        if client.is_folder(item):
            if item_name.lower() == "closed":
                all_metas.extend(
                    _process_closed_cashier_folder(client, item["id"])
                )
            else:
                all_metas.extend(
                    _process_cashier_month_folder(
                        client, item["id"], item_name,
                        parent_path=root_path,
                    )
                )

    return all_metas


def _process_closed_cashier_folder(
    client: SharePointClient,
    closed_folder_id: str,
) -> list[dict]:
    logger.info("[Cashier] Entering Closed/")
    metas: list[dict] = []

    year_folders = client.list_children_by_item_id(closed_folder_id)
    for year_folder in year_folders:
        if not client.is_folder(year_folder):
            continue
        month_folders = client.list_children_by_item_id(year_folder["id"])
        for month_folder in month_folders:
            if client.is_folder(month_folder):
                metas.extend(
                    _process_cashier_month_folder(
                        client,
                        month_folder["id"],
                        month_folder["name"],
                        parent_path=(
                            f"Closed/{year_folder['name']}/{month_folder['name']}"
                        ),
                    )
                )
    return metas


def _process_cashier_month_folder(
    client: SharePointClient,
    folder_id: str,
    folder_name: str,
    parent_path: str,
) -> list[dict]:
    logger.info(f"[Cashier] Checking folder: {folder_name}")
    metas: list[dict] = []
    items = client.list_children_by_item_id(folder_id)

    for item in items:
        if not client.is_file(item):
            continue

        filename = item["name"]
        suffix   = Path(filename).suffix.lower()

        if is_template_file(filename):
            logger.info(f"[Cashier] Skipping template: {filename}")
            continue

        if suffix not in CASHIER_EXTENSIONS:
            continue

        branch = detect_branch_from_filename(filename)
        if not branch:
            logger.warning(f"[Cashier] Unknown branch, skipping: {filename}")
            continue

        local_path       = ensure_branch_folder(branch) / "cashier_reports" / filename
        newly_downloaded = should_download(local_path, item)

        if newly_downloaded:
            logger.info(f"[Cashier] Downloading: {filename}")
            client.download_file_by_item_id(item["id"], local_path)
            stamp_local_mtime(local_path, item)
        else:
            logger.info(f"[Cashier] Unchanged, skipping download: {filename}")

        metas.append(
            _build_meta(
                item             = item,
                local_path       = local_path,
                branch           = branch,
                report_type      = "cashier",
                sharepoint_path  = f"{parent_path}/{filename}",
                newly_downloaded = newly_downloaded,
            )
        )

    return metas


# ── Sales download ────────────────────────────────────────────────────────────

def download_sales_reports(client: SharePointClient) -> list[dict]:
    """
    Traverses:
        Sales&Orders Reports/<Branch>/Sales Reports/

    Downloads ALL sales files including combined historical files.
    Previously combined historical files were skipped here — that was wrong.
    The parser now uses canonical election to decide which file to use and
    how much of it to load, so all files must reach the parser.

    Returns a list of metadata dicts — one per file seen.
    """
    logger.info("[Sales] Scanning root for Sales&Orders Reports")

    sales_root = find_root_folder_by_name(client, "Sales&Orders Reports")
    if not sales_root:
        raise RuntimeError("Could not find root folder: Sales&Orders Reports")

    all_metas: list[dict] = []
    branch_folders = client.list_children_by_item_id(sales_root["id"])

    for branch_folder in branch_folders:
        if not client.is_folder(branch_folder):
            continue

        branch_name          = normalize_branch_name(branch_folder["name"])
        sales_reports_folder = _find_sales_reports_folder(client, branch_folder["id"])

        if not sales_reports_folder:
            logger.warning(
                f"[Sales] No 'Sales Reports' folder for branch: {branch_name}"
            )
            continue

        all_metas.extend(
            _process_sales_reports_folder(
                client,
                folder_id   = sales_reports_folder["id"],
                branch_name = branch_name,
                parent_path = (
                    f"Sales&Orders Reports/{branch_name}/Sales Reports"
                ),
            )
        )

    return all_metas


def _find_sales_reports_folder(
    client: SharePointClient,
    branch_folder_id: str,
) -> dict | None:
    children = client.list_children_by_item_id(branch_folder_id)
    for item in children:
        if (
            client.is_folder(item)
            and item["name"].strip().lower() == "sales reports"
        ):
            return item
    return None


def _process_sales_reports_folder(
    client: SharePointClient,
    folder_id: str,
    branch_name: str,
    parent_path: str,
) -> list[dict]:
    logger.info(f"[Sales] Checking branch: {branch_name}")
    metas: list[dict] = []
    items = client.list_children_by_item_id(folder_id)

    for item in items:
        if not client.is_file(item):
            continue

        filename = item["name"]
        suffix   = Path(filename).suffix.lower()

        if is_template_file(filename):
            logger.info(f"[Sales] Skipping template: {filename}")
            continue

        # 🟢 NOTE: Combined historical files (e.g. 'Galleria Jan 2023-March 2026')
        # are NO LONGER skipped here. They are downloaded and passed to the parser
        # which uses canonical election + watermark to decide how to use them.

        if suffix not in SALE_EXTENSIONS:
            continue

        local_path       = (
            ensure_branch_folder(branch_name) / "sales_reports" / filename
        )
        newly_downloaded = should_download(local_path, item)

        if newly_downloaded:
            logger.info(f"[Sales] Downloading: {filename}")
            client.download_file_by_item_id(item["id"], local_path)
            stamp_local_mtime(local_path, item)
        else:
            logger.info(f"[Sales] Unchanged, skipping download: {filename}")

        metas.append(
            _build_meta(
                item             = item,
                local_path       = local_path,
                branch           = branch_name,
                report_type      = "sales",
                sharepoint_path  = f"{parent_path}/{filename}",
                newly_downloaded = newly_downloaded,
            )
        )

    return metas


# ── Qty list download (unchanged) ─────────────────────────────────────────────

def download_qty_lists(client: SharePointClient) -> list[dict]:
    """
    Traverses:
        Sales&Orders Reports/<Branch>/

    SMART MODE: For each branch, only downloads the MOST RECENT qty file.
    Each qty file is a complete inventory snapshot — older snapshots are
    irrelevant once a newer one exists.
    """
    logger.info("[Qty] Scanning root for Sales&Orders Reports")

    sales_root = find_root_folder_by_name(client, "Sales&Orders Reports")
    if not sales_root:
        logger.error("[Qty] Could not find root folder: Sales&Orders Reports")
        return []

    all_metas: list[dict] = []
    branch_folders = client.list_children_by_item_id(sales_root["id"])

    for branch_folder in branch_folders:
        if not client.is_folder(branch_folder):
            continue

        branch_name = normalize_branch_name(branch_folder["name"])

        if branch_name not in BRANCHES:
            logger.debug(f"[Qty] Skipping non-branch folder: {branch_name}")
            continue

        logger.info(f"[Qty] Scanning branch folder: {branch_name}")

        items = client.list_children_by_item_id(branch_folder["id"])

        candidates: list[tuple[datetime, dict]] = []

        for item in items:
            if not client.is_file(item):
                continue

            filename = item["name"]
            suffix   = Path(filename).suffix.lower()

            if not is_qty_file(filename):
                continue

            if suffix not in QTY_EXTENSIONS:
                continue

            if is_template_file(filename):
                logger.info(f"[Qty] Skipping template: {filename}")
                continue

            last_modified  = _parse_sharepoint_dt(item.get("lastModifiedDateTime"))
            effective_date = parse_effective_date(filename, last_modified)

            logger.debug(
                f"[Qty][{branch_name}] {filename} → "
                f"effective_date={effective_date.date()}"
            )

            candidates.append((effective_date, item))

        if not candidates:
            logger.warning(f"[Qty][{branch_name}] No qty files found.")
            continue

        candidates.sort(key=lambda x: x[0], reverse=True)
        best_date, best_item = candidates[0]
        best_filename        = best_item["name"]

        skipped = [item["name"] for _, item in candidates[1:]]
        if skipped:
            logger.info(
                f"[Qty][{branch_name}] Selected latest: {best_filename} "
                f"(effective={best_date.date()}) | "
                f"Skipping {len(skipped)} older file(s): {skipped}"
            )
        else:
            logger.info(
                f"[Qty][{branch_name}] Only one file found: {best_filename}"
            )

        local_path       = (
            ensure_branch_folder(branch_name) / "qty_lists" / best_filename
        )
        newly_downloaded = should_download(local_path, best_item)

        if newly_downloaded:
            logger.info(f"[Qty] Downloading: {best_filename}")
            client.download_file_by_item_id(best_item["id"], local_path)
            stamp_local_mtime(local_path, best_item)
        else:
            logger.info(f"[Qty] Unchanged, skipping download: {best_filename}")

        all_metas.append(
            _build_meta(
                item             = best_item,
                local_path       = local_path,
                branch           = branch_name,
                report_type      = "qty_list",
                sharepoint_path  = (
                    f"Sales&Orders Reports/{branch_name}/{best_filename}"
                ),
                newly_downloaded = newly_downloaded,
            )
        )

    return all_metas


def build_combined_exports() -> None:
    """
    Walks all downloaded branch folders and produces two combined CSVs:
      - sharepoint_sales_combined.csv   (all sales report files across branches)
      - sharepoint_cashier_combined.csv (all cashier report files across branches)
    A 'Branch' column is added to each row so you can filter by branch in Power BI.
    """
    sales_dfs   = []
    cashier_dfs = []

    for branch in BRANCHES:
        branch_dir = LOCAL_BASE_DIR / branch

        if not branch_dir.exists():
            logger.warning(f"[Export] Branch folder not found: {branch_dir}")
            continue

        for f in branch_dir.iterdir():
            if not f.is_file():
                continue

            suffix = f.suffix.lower()
            name   = f.name.lower()

            # Skip temp files and qty lists subfolder files
            if name.startswith("~$"):
                continue

            try:
                # ── Cashier files (.xlsm / .xlsx with "cashier" in name) ──────
                if suffix in {".xlsm", ".xlsx"} and "cashier" in name:
                    xls = pd.ExcelFile(f)
                    for sheet in xls.sheet_names:
                        df_sheet = pd.read_excel(xls, sheet_name=sheet)
                        df_sheet.columns = df_sheet.columns.str.strip()
                        if "Receipt Txn No" in df_sheet.columns:
                            df_sheet["Branch"]    = branch
                            df_sheet["Source_File"] = f.name
                            cashier_dfs.append(df_sheet)

                # ── Sales files (.csv / .xlsx without "cashier" in name) ──────
                elif suffix == ".csv" and "cashier" not in name:
                    df = pd.read_csv(f, low_memory=False)
                    df.columns = df.columns.str.strip()
                    if "Transaction ID" in df.columns:
                        df["Branch"]      = branch
                        df["Source_File"] = f.name
                        sales_dfs.append(df)

                elif suffix == ".xlsx" and "cashier" not in name:
                    df = pd.read_excel(f)
                    df.columns = df.columns.str.strip()
                    if "Transaction ID" in df.columns:
                        df["Branch"]      = branch
                        df["Source_File"] = f.name
                        sales_dfs.append(df)

            except Exception as e:
                logger.warning(f"[Export] Could not read {f.name}: {e}")

    # ── Save sales combined ───────────────────────────────────────────────────
    SALES_COMBINED_FILE.parent.mkdir(parents=True, exist_ok=True)

    if sales_dfs:
        df_sales = pd.concat(sales_dfs, ignore_index=True)
        df_sales.drop_duplicates(
            subset=["Transaction ID", "Date Sold", "Description", "Qty Sold"],
            inplace=True
        )
        df_sales.to_csv(SALES_COMBINED_FILE, index=False)
        logger.info(
            f"[Export] Sales combined: {len(df_sales):,} rows → {SALES_COMBINED_FILE}"
        )
    else:
        logger.warning("[Export] No sales files found — sales combined not written.")

    # ── Save cashier combined ─────────────────────────────────────────────────
    if cashier_dfs:
        df_cashier = pd.concat(cashier_dfs, ignore_index=True)
        df_cashier.drop_duplicates(subset=["Receipt Txn No"], inplace=True)
        df_cashier.to_csv(CASHIER_COMBINED_FILE, index=False)
        logger.info(
            f"[Export] Cashier combined: {len(df_cashier):,} rows → {CASHIER_COMBINED_FILE}"
        )
    else:
        logger.warning("[Export] No cashier files found — cashier combined not written.")

# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    client = SharePointClient(drive_id=DRIVE_ID)
    LOCAL_BASE_DIR.mkdir(parents=True, exist_ok=True)

    run_id = start_run()
    logger.info(f"Ingestion run started — run_id={run_id}")

    # Log current watermarks so you can see at a glance what's already loaded
    logger.info("─" * 60)
    logger.info("Branch watermarks (max Date Sold already in DB):")
    for branch in BRANCHES:
        location  = _BRANCH_TO_LOCATION.get(branch, branch)
        watermark, watermark_dt = get_branch_watermark(branch)
        if watermark:
            logger.info(f"  {branch:<14} → loaded up to {watermark} (datetime: {watermark_dt})")
        else:
            logger.info(f"  {branch:<14} → no watermark (first run)")

    run_stats = {
        "status":           "success",
        "files_seen":       0,
        "files_downloaded": 0,
        "files_processed":  0,
        "files_failed":     0,
        "notes":            None,
    }

    try:
        # ── Step 1: Download ──────────────────────────────────────────────
        cashier_metas = download_cashier_reports(client)
        sales_metas   = download_sales_reports(client)
        qty_metas     = download_qty_lists(client)

        all_metas = cashier_metas + sales_metas + qty_metas
        run_stats["files_seen"]       = len(all_metas)
        run_stats["files_downloaded"] = sum(
            1 for f in all_metas if f.get("newly_downloaded")
        )

        # ── Step 2: Parse + load cashier files ───────────────────────────
        for meta in cashier_metas:
            result = process_cashier_file(run_id, meta["branch"], meta)
            run_stats["files_processed"] += result["processed"]
            run_stats["files_failed"]    += result["failed"]

        # ── Step 3: Parse + load sales files (per branch) ────────────────
        sales_by_branch: dict[str, list] = defaultdict(list)
        for meta in sales_metas:
            sales_by_branch[meta["branch"]].append(meta)

        for branch, metas in sales_by_branch.items():
            result = process_sales_branch(run_id, branch, metas)
            run_stats["files_processed"] += result["processed"]
            run_stats["files_failed"]    += result["failed"]

        # ── Step 4: Parse + load qty list files ──────────────────────────
        for meta in qty_metas:
            result = process_qty_file(run_id, meta["branch"], meta)
            run_stats["files_processed"] += result["processed"]
            run_stats["files_failed"]    += result["failed"]

        if run_stats["files_failed"] > 0:
            run_stats["status"] = "partial"

    except Exception as exc:
        logger.exception("Pipeline failed with unhandled error")
        run_stats["status"] = "failed"
        run_stats["notes"]  = str(exc)

    finally:
        finish_run(run_id, run_stats)
        logger.info(
            f"Run {run_id} finished — "
            f"status={run_stats['status']} | "
            f"seen={run_stats['files_seen']} | "
            f"downloaded={run_stats['files_downloaded']} | "
            f"processed={run_stats['files_processed']} | "
            f"failed={run_stats['files_failed']}"
        )

        # ── Step 5: Build combined exports ───────────────────────────────────
        build_combined_exports()


if __name__ == "__main__":
    main()