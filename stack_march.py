"""
stack_march_cashier.py
======================
Stacks March 2026 cashier reports across all branches.
- Only reads sheets named/numbered 1-31
- Only keeps the 9 specified columns
- No deduplication
"""

import pandas as pd
from pathlib import Path

BASE_DIR = Path(r"D:\Documents\Portal ML Analys\Portal_ML\Portal_ML_V4\data\01_raw\sharepoint_downloads")
OUTPUT_FILE = BASE_DIR / "March_2026_Cashier_All_Branches.csv"

KEEP_COLS = [
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

BRANCH_FILES = {
    "Centurion 2R": "Centurion 2r  Daily Cashier report  Mar 2026.xlsm",
    "Galleria":     "Galleria Daily Cashier report Mar 2026.xlsm",
    "Milele":       "Milele  Daily Cashier report Mar 2026.xlsm",
    "ABC":          "Pharmart Abc Daily Cashier report  Mar 2026.xlsm",
    "Portal 2R":    "Portal 2R  Daily Cashier report  Mar 2026.xlsm",
    "Portal CBD":   "Portal CBD  Daily Cashier report  Mar 2026.xlsm",
}


def is_day_sheet(sheet_name: str) -> bool:
    try:
        n = int(str(sheet_name).strip())
        return 1 <= n <= 31
    except ValueError:
        return False


def load_branch_file(branch: str, filename: str) -> pd.DataFrame:
    filepath = BASE_DIR / branch / "cashier_reports" / filename

    if not filepath.exists():
        print(f"   ⚠️  [{branch}] File not found: {filename}")
        print(f"        Expected: {filepath}")
        return pd.DataFrame()

    print(f"   📍 [{branch}] Loading {filename}...")

    try:
        xls = pd.ExcelFile(filepath)
        all_sheets = xls.sheet_names
        day_sheets = [s for s in all_sheets if is_day_sheet(s)]

        print(f"      {len(all_sheets)} total sheets → {len(day_sheets)} day sheets (1-31)")

        if not day_sheets:
            print(f"      ⚠️  No sheets numbered 1-31 found")
            return pd.DataFrame()

        frames = []
        for sheet in day_sheets:
            df = pd.read_excel(xls, sheet_name=sheet, header=0)
            df.columns = df.columns.str.strip()

            available = [c for c in KEEP_COLS if c in df.columns]
            missing   = [c for c in KEEP_COLS if c not in df.columns]

            if missing and "Receipt Txn No" not in df.columns and len(df.columns) >= 9:
                print(f"      ⚠️  Sheet {sheet}: columns shifted — positional rename")
                col_map = {df.columns[i]: KEEP_COLS[i] for i in range(min(9, len(df.columns)))}
                df = df.rename(columns=col_map)
                available = [c for c in KEEP_COLS if c in df.columns]

            if not available:
                continue

            df = df[available].dropna(how="all").copy()

            if len(df) > 0:
                df["Sheet_Day"] = int(str(sheet).strip())
                frames.append(df)

        if not frames:
            return pd.DataFrame()

        branch_df = pd.concat(frames, ignore_index=True)
        branch_df["Branch"] = branch
        print(f"      ✅ {len(branch_df):,} rows across {len(frames)} sheets")
        return branch_df

    except Exception as e:
        print(f"      ❌ Error: {e}")
        return pd.DataFrame()


def stack_march_cashier():
    print("=" * 60)
    print("March 2026 Cashier Stack")
    print("=" * 60)

    all_frames = []

    for branch, filename in BRANCH_FILES.items():
        df = load_branch_file(branch, filename)
        if not df.empty:
            all_frames.append(df)
        print()

    if not all_frames:
        print("No data loaded. Check file paths above.")
        return

    print("Stacking all branches...")
    final_df = pd.concat(all_frames, ignore_index=True)
    final_df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nDone!")
    print(f"   Total rows : {len(final_df):,}")
    print(f"   Saved to   : {OUTPUT_FILE}")
    print(f"\nRows by branch:")
    for branch, count in final_df["Branch"].value_counts().items():
        print(f"   {branch:<20} {count:,}")


if __name__ == "__main__":
    stack_march_cashier()