"""
audit_social_cashier.py
=======================
Reads all branch "Daily Cashier report" .xlsm files for Jan / Feb / Mar 2026,
collects every row where Ordered Via == 'respond.io', and prints a summary.
"""

import re
from pathlib import Path
import pandas as pd

# ── CONFIG ────────────────────────────────────────────────────────────────────
POS_ROOT   = Path(r"D:\Documents\Portal ML Analys\Portal_ML\Portal_ML_V4\data\01_raw\pos_data")

TARGET_MONTHS = ["jan", "feb", "mar"]          # case-insensitive match against filename
TARGET_YEAR   = "2026"                         # must also appear in filename

SOCIAL_TAG    = "respond.io"                   # exact lower-case match
ORDERED_COL   = "Ordered Via"                  # column name in cashier sheets

DAY_SHEETS    = [f"{d:02d}" for d in range(1, 32)]   # "01" … "31"

# ── HELPERS ───────────────────────────────────────────────────────────────────

def is_cashier_file(path: Path) -> bool:
    return "daily cashier report" in path.name.lower() and path.suffix.lower() in (".xlsm", ".xlsx")

def month_tag(filename: str) -> str | None:
    """Return e.g. 'jan', 'feb', 'mar' if the filename matches a target month."""
    name_lower = filename.lower()
    for m in TARGET_MONTHS:
        if m in name_lower and TARGET_YEAR in filename:
            return m
    return None

def read_cashier_file(file_path: Path) -> pd.DataFrame:
    """Read all day sheets from one cashier file, return concatenated DataFrame."""
    frames = []
    try:
        xls = pd.ExcelFile(file_path, engine="openpyxl")
    except Exception as e:
        print(f"      ⚠️  Cannot open {file_path.name}: {e}")
        return pd.DataFrame()

    available_sheets = [s for s in xls.sheet_names if s.strip() in DAY_SHEETS]

    if not available_sheets:
        print(f"      ⚠️  No day sheets (01-31) found in {file_path.name} — tabs: {xls.sheet_names[:5]}")
        return pd.DataFrame()

    for sheet in available_sheets:
        try:
            df = pd.read_excel(xls, sheet_name=sheet, dtype=str)
            df.columns = df.columns.str.strip()
            df["_sheet"] = sheet
            frames.append(df)
        except Exception as e:
            pass   # blank/corrupt day sheet — skip silently

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  RESPOND.IO CASHIER AUDIT  —  Jan / Feb / Mar 2026")
    print("=" * 65)

    all_rows = []
    branch_dirs = [d for d in POS_ROOT.iterdir() if d.is_dir()]

    for branch_dir in sorted(branch_dirs):
        branch_name = branch_dir.name.upper()
        cashier_files = [f for f in branch_dir.iterdir() if is_cashier_file(f)]

        # Keep only Jan / Feb / Mar 2026 files
        target_files = [(f, month_tag(f.name)) for f in cashier_files if month_tag(f.name)]

        if not target_files:
            continue

        print(f"\n📍 {branch_name}")

        for file_path, month in sorted(target_files, key=lambda x: TARGET_MONTHS.index(x[1])):
            print(f"   📄 {file_path.name}  [{month.upper()}]")
            df = read_cashier_file(file_path)

            if df.empty or ORDERED_COL not in df.columns:
                if not df.empty:
                    print(f"      ⚠️  '{ORDERED_COL}' column not found. Columns: {list(df.columns)[:8]}")
                continue

            # Filter respond.io rows (case-insensitive, strip whitespace)
            mask = df[ORDERED_COL].fillna("").str.strip().str.lower() == SOCIAL_TAG
            social_df = df[mask].copy()
            social_df["_branch"] = branch_name
            social_df["_month"]  = month.upper()
            social_df["_file"]   = file_path.name
            all_rows.append(social_df)

            total_rows = mask.sum()
            print(f"      ✅ {total_rows:,} respond.io rows found")

    # ── SUMMARY ───────────────────────────────────────────────────────────────
    if not all_rows:
        print("\n❌ No respond.io rows found across any file.")
        return

    combined = pd.concat(all_rows, ignore_index=True)

    # Normalise Amount column — try common column names
    amount_col = next(
        (c for c in combined.columns if c.lower() in ("amount", "total", "sale amount", "txn amount")),
        None
    )
    if amount_col:
        combined["_amount"] = pd.to_numeric(
            combined[amount_col].str.replace(",", "", regex=False),
            errors="coerce"
        ).fillna(0)
    else:
        combined["_amount"] = 0
        print(f"\n   ⚠️  No Amount column found — revenue totals will be 0. Columns present: {list(combined.columns)[:10]}")

    # Unique transaction count — try common ID column names
    txn_col = next(
        (c for c in combined.columns if "receipt" in c.lower() or "txn no" in c.lower() or "transaction" in c.lower()),
        None
    )

    month_order = {m.upper(): i for i, m in enumerate(TARGET_MONTHS)}

    print("\n" + "=" * 65)
    print("  SUMMARY BY MONTH")
    print("=" * 65)
    print(f"  {'Month':<8} {'Rows':>8}  {'Transactions':>14}  {'Revenue (KES)':>14}")
    print(f"  {'-'*60}")

    grand_rows = grand_txns = grand_rev = 0

    for month in [m.upper() for m in TARGET_MONTHS]:
        grp = combined[combined["_month"] == month]
        if grp.empty:
            continue
        rows = len(grp)
        txns = grp[txn_col].nunique() if txn_col else "n/a"
        rev  = grp["_amount"].sum()
        print(f"  {month:<8} {rows:>8,}  {str(txns):>14}  KES {rev:>10,.0f}")
        grand_rows += rows
        if isinstance(txns, int): grand_txns += txns
        grand_rev  += rev

    print(f"  {'-'*60}")
    print(f"  {'TOTAL':<8} {grand_rows:>8,}  {str(grand_txns):>14}  KES {grand_rev:>10,.0f}")

    print("\n" + "=" * 65)
    print("  SUMMARY BY BRANCH")
    print("=" * 65)
    print(f"  {'Branch':<22} {'Month':<6} {'Rows':>6}  {'Transactions':>14}  {'Revenue (KES)':>14}")
    print(f"  {'-'*65}")

    for branch in sorted(combined["_branch"].unique()):
        for month in [m.upper() for m in TARGET_MONTHS]:
            grp = combined[(combined["_branch"] == branch) & (combined["_month"] == month)]
            if grp.empty:
                continue
            rows = len(grp)
            txns = grp[txn_col].nunique() if txn_col else "n/a"
            rev  = grp["_amount"].sum()
            print(f"  {branch:<22} {month:<6} {rows:>6,}  {str(txns):>14}  KES {rev:>10,.0f}")

    print(f"\n  Grand Total Respond.io line items: {grand_rows:,}")
    if txn_col:
        print(f"  Grand Total Unique Transactions:   {combined[txn_col].nunique():,}")
    print(f"  Grand Revenue (KES):               {grand_rev:,.0f}")
    print()

if __name__ == "__main__":
    main()