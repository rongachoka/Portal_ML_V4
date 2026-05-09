from __future__ import annotations

import re
from pathlib import Path
from datetime import datetime, timezone

from Portal_ML_V4.sharepoint.sharepoint_client import SharePointClient
from Portal_ML_V4.src.config.settings import RAW_DATA_DIR

# Replace this with your real drive ID
DRIVE_ID = "b!whL3rPzNh0-7qRe5yrHoftRvAUJj1gFFvAMiiq_bJDX64liSkv0CSZtdTu6bqccj"

# Change this if you want another location
LOCAL_BASE_DIR = RAW_DATA_DIR/ "sharepoint_downloads"

# Known branch names from your org
BRANCHES = [
    "Galleria",
    "ABC",
    "Milele",
    "Portal 2R",
    "Portal CBD",
    "Centurion 2R",
]

SALE_EXTENSIONS = {".csv", ".xlsx"}
CASHIER_EXTENSIONS = {".xlsm"}


def is_template_file(filename: str) -> bool:
    return "template" in filename.lower()


def find_root_folder_by_name(client: SharePointClient, folder_name: str) -> dict | None:
    target = folder_name.lower().replace(" ", "")
    
    for item in client.list_root_children():
        if not client.is_folder(item):
            continue

        actual = item["name"].lower().replace(" ", "")
        print(f"[DEBUG] root folder: {repr(item['name'])}")

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
    local_size = local_path.stat().st_size
    if remote_size != local_size:
        return True

    remote_modified_raw = remote_item.get("lastModifiedDateTime")
    if not remote_modified_raw:
        return False

    # Graph timestamps look like 2026-03-16T14:20:11Z
    remote_modified = datetime.fromisoformat(remote_modified_raw.replace("Z", "+00:00"))
    local_modified = datetime.fromtimestamp(local_path.stat().st_mtime, tz=timezone.utc)

    # Allow a small tolerance
    return remote_modified > local_modified


def stamp_local_mtime(local_path: Path, remote_item: dict) -> None:
    remote_modified_raw = remote_item.get("lastModifiedDateTime")
    if not remote_modified_raw:
        return

    remote_modified = datetime.fromisoformat(remote_modified_raw.replace("Z", "+00:00"))
    ts = remote_modified.timestamp()
    local_path.touch(exist_ok=True)
    import os
    os.utime(local_path, (ts, ts))


def download_cashier_reports(client: SharePointClient) -> None:
    """
    Traverses:
    Documents/Finance Reports/Cashier Reports/

    Handles:
    - Jan 2026, Feb 2026, Mar 2026...
    - Closed/2025/Dec 2025/...
    """
    root_path = "Finance Reports/Cashier Reports"
    print(f"\n[Cashier] Scanning: {root_path}")

    root_items = client.list_children_by_path(root_path)

    for item in root_items:
        item_name = item["name"]

        if client.is_folder(item):
            if item_name.lower() == "closed":
                process_closed_cashier_folder(client, item["id"])
            else:
                process_cashier_month_folder(client, item["id"], item_name)


def process_closed_cashier_folder(client: SharePointClient, closed_folder_id: str) -> None:
    print("[Cashier] Entering Closed/")
    year_folders = client.list_children_by_item_id(closed_folder_id)

    for year_folder in year_folders:
        if not client.is_folder(year_folder):
            continue

        month_folders = client.list_children_by_item_id(year_folder["id"])
        for month_folder in month_folders:
            if client.is_folder(month_folder):
                process_cashier_month_folder(client, month_folder["id"], month_folder["name"])


def process_cashier_month_folder(client: SharePointClient, folder_id: str, folder_name: str) -> None:
    print(f"[Cashier] Checking folder: {folder_name}")
    items = client.list_children_by_item_id(folder_id)

    for item in items:
        if not client.is_file(item):
            continue

        filename = item["name"]
        suffix = Path(filename).suffix.lower()

        if is_template_file(filename):
            print(f"[Cashier] Skipping template file: {filename}")
            continue

        if suffix not in CASHIER_EXTENSIONS:
            continue

        branch = detect_branch_from_filename(filename)
        if not branch:
            print(f"[Cashier] Skipping unknown branch file: {filename}")
            continue

        local_path = ensure_branch_folder(branch) / "cashier_reports" / filename

        if should_download(local_path, item):
            print(f"[Cashier] Downloading: {filename} -> {local_path}")
            client.download_file_by_item_id(item["id"], local_path)
            stamp_local_mtime(local_path, item)
        else:
            print(f"[Cashier] Unchanged, skipping: {filename}")


def download_sales_reports(client: SharePointClient) -> None:
    """
    Traverses:
    Documents/Sales&Orders Reports/<Branch>/Sales Reports/
    """
    print("\n [Sales] Scanning root for Sales&Orders Reports")
    root_path = "Sales&Orders Reports"
    

    sales_root = find_root_folder_by_name(client, "Sales&Orders Reports")
    if not sales_root:
        raise RuntimeError("Could not find root folder:  Sales&Order Reports")

    branch_folders = client.list_children_by_item_id(sales_root["id"])

    for branch_folder in branch_folders:
        if not client.is_folder(branch_folder):
            continue

        branch_name = normalize_branch_name(branch_folder["name"])
        sales_reports_folder = find_sales_reports_folder(client, branch_folder["id"])

        if not sales_reports_folder:
            print(f"[Sales] No 'Sales Reports' folder found for branch: {branch_name}")
            continue

        process_sales_reports_folder(client, sales_reports_folder["id"], branch_name)


def find_sales_reports_folder(client: SharePointClient, branch_folder_id: str) -> dict | None:
    children = client.list_children_by_item_id(branch_folder_id)

    for item in children:
        if client.is_folder(item) and item["name"].strip().lower() == "sales reports":
            return item
    return None


def process_sales_reports_folder(client: SharePointClient, folder_id: str, branch_name: str) -> None:
    print(f"[Sales] Checking branch: {branch_name}")
    items = client.list_children_by_item_id(folder_id)

    for item in items:
        if not client.is_file(item):
            continue

        filename = item["name"]
        suffix = Path(filename).suffix.lower()

        if is_template_file(filename):
            print(f"[Cashier] Skipping template file: {filename}")
            continue

        if suffix not in SALE_EXTENSIONS:
            continue

        local_path = ensure_branch_folder(branch_name) / "sales_reports" / filename

        if should_download(local_path, item):
            print(f"[Sales] Downloading: {filename} -> {local_path}")
            client.download_file_by_item_id(item["id"], local_path)
            stamp_local_mtime(local_path, item)
        else:
            print(f"[Sales] Unchanged, skipping: {filename}")


def main() -> None:
    client = SharePointClient(drive_id=DRIVE_ID)

    LOCAL_BASE_DIR.mkdir(parents=True, exist_ok=True)

    download_cashier_reports(client)
    download_sales_reports(client)

    print("\nDone.")


if __name__ == "__main__":
    main()