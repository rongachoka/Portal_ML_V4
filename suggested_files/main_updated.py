"""
Updated main() for sharepoint_downloader.py
Replace the existing main() at the bottom of your downloader with this.

The download logic (download_cashier_reports, download_sales_reports etc.)
stays exactly as-is. This just adds the parser/DB stage after downloading.
"""

import logging
from pathlib import Path
from collections import defaultdict

from Portal_ML_V4.sharepoint.sharepoint_client import SharePointClient
from Portal_ML_V4.sharepoint.sharepoint_parser import (
    finish_run,
    process_cashier_file,
    process_sales_branch,
    start_run,
)
from Portal_ML_V4.src.config.settings import RAW_DATA_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DRIVE_ID = "b!whL3rPzNh0-7qRe5yrHoftRvAUJj1gFFvAMiiq_bJDX64liSkv0CSZtdTu6bqccj"
LOCAL_BASE_DIR = RAW_DATA_DIR / "sharepoint_downloads"


def main() -> None:
    client = SharePointClient(drive_id=DRIVE_ID)
    LOCAL_BASE_DIR.mkdir(parents=True, exist_ok=True)

    # ── Start ingestion run ────────────────────────────────────────────────
    run_id = start_run()
    logger.info(f"Ingestion run started — run_id={run_id}")

    run_stats = {
        "status": "success",
        "files_seen": 0,
        "files_downloaded": 0,
        "files_processed": 0,
        "files_failed": 0,
        "notes": None,
    }

    try:
        # ── Step 1: Download (your existing logic, extended to return metadata) ──
        # download_cashier_reports and download_sales_reports now return
        # a list of metadata dicts for every file they downloaded or found unchanged.
        # See note below on what metadata each dict should contain.

        cashier_file_metas = download_cashier_reports(client, run_id)
        sales_file_metas   = download_sales_reports(client, run_id)

        run_stats["files_seen"]       = len(cashier_file_metas) + len(sales_file_metas)
        run_stats["files_downloaded"] = sum(1 for f in cashier_file_metas + sales_file_metas if f.get("newly_downloaded"))

        # ── Step 2: Parse & load cashier files ────────────────────────────────
        for meta in cashier_file_metas:
            result = process_cashier_file(run_id, meta["branch"], meta)
            run_stats["files_processed"] += result["processed"]
            run_stats["files_failed"]    += result["failed"]

        # ── Step 3: Parse & load sales files — canonical selection per branch ──
        # Group sales files by branch first
        sales_by_branch: dict[str, list] = defaultdict(list)
        for meta in sales_file_metas:
            sales_by_branch[meta["branch"]].append(meta)

        for branch, metas in sales_by_branch.items():
            result = process_sales_branch(run_id, branch, metas)
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


# ── What the download functions need to return ──────────────────────────────
#
# Each download function should return a list of dicts like:
#
# {
#     "branch":                   "Galleria",
#     "report_type":              "cashier",          # or "sales"
#     "filename":                 "Galleria Daily Cashier report Mar 2026.xlsm",
#     "file_extension":           ".xlsm",
#     "sharepoint_item_id":       item["id"],
#     "sharepoint_path":          "Finance Reports/Cashier Reports/Mar 2026/...",
#     "sharepoint_last_modified": datetime(...),      # parsed from item["lastModifiedDateTime"]
#     "sharepoint_size_bytes":    item["size"],
#     "local_path":               str(local_path),
#     "newly_downloaded":         True / False,       # True if actually re-downloaded this run
# }
#
# The easiest way to do this is to have download_cashier_reports and
# download_sales_reports build up a list and return it, appending a dict
# for each file they process (whether newly downloaded or unchanged).
#
# The metadata is already available from the Graph API item dict + your
# existing should_download() logic — it's mostly just collecting what
# you already have.


if __name__ == "__main__":
    main()
