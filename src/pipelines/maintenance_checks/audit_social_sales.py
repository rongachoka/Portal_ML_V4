"""
diagnose_social_gap.py
======================
Compares respond.io rows in the raw cashier files vs what survived
into all_locations_sales_Jan25-Jan26.csv after the ETL merge.
Tells you exactly where the gap is.
"""

from pathlib import Path
import pandas as pd

# ── CONFIG ────────────────────────────────────────────────────────────────────
POS_ROOT     = Path(r"D:\Documents\Portal ML Analys\Portal_ML\Portal_ML_V4\data\01_raw\pos_data")
PROCESSED_CSV = Path(r"D:\Documents\Portal ML Analys\Portal_ML\Portal_ML_V4\data\03_processed\pos_data\all_locations_sales_Jan25-Jan26.csv")

TARGET_MONTHS = ["jan", "feb", "mar"]
TARGET_YEAR   = "2026"
SOCIAL_TAG    = "respond.io"
DAY_SHEETS    = [f"{d:02d}" for d in range(1, 32)]

# ── STEP 1: Pull all respond.io Receipt Txn Nos from RAW cashier files ────────

def is_cashier_file(path: Path) -> bool:
    return "daily cashier report" in path.name.lower() and path.suffix.lower() in (".xlsm", ".xlsx")

def month_tag(filename: str):
    name_lower = filename.lower()
    for m in TARGET_MONTHS:
        if m in name_lower and TARGET_YEAR in filename:
            return m
    return None

print("=" * 65)
print("  STEP 1 — Reading respond.io rows from RAW cashier files")
print("=" * 65)

raw_txn_rows = []

for branch_dir in sorted(d for d in POS_ROOT.iterdir() if d.is_dir()):
    cashier_files = [(f, month_tag(f.name)) for f in branch_dir.iterdir()
                     if is_cashier_file(f) and month_tag(f.name)]

    for file_path, month in cashier_files:
        try:
            xls = pd.ExcelFile(file_path, engine="openpyxl")
        except Exception as e:
            print(f"  ⚠️  Cannot open {file_path.name}: {e}")
            continue

        for sheet in [s for s in xls.sheet_names if s.strip() in DAY_SHEETS]:
            try:
                df = pd.read_excel(xls, sheet_name=sheet, dtype=str)
                df.columns = df.columns.str.strip()

                ordered_col = next((c for c in df.columns if "ordered" in c.lower()), None)
                txn_col     = next((c for c in df.columns if "receipt" in c.lower() or "txn no" in c.lower()), None)
                amount_col  = next((c for c in df.columns if c.lower() == "amount"), None)

                if not ordered_col:
                    continue

                mask = df[ordered_col].fillna("").str.strip().str.lower() == SOCIAL_TAG
                if mask.any():
                    grp = df[mask].copy()
                    grp["_branch"] = branch_dir.name.upper()
                    grp["_month"]  = month.upper()
                    grp["_raw_txn_no"] = grp[txn_col].astype(str).str.replace(r'\.0$', '', regex=True).str.strip() if txn_col else None
                    grp["_raw_amount"] = pd.to_numeric(
                        grp[amount_col].str.replace(",", "", regex=False), errors="coerce"
                    ).fillna(0) if amount_col else 0
                    raw_txn_rows.append(grp[["_branch", "_month", "_raw_txn_no", "_raw_amount"]])
            except Exception:
                pass

if not raw_txn_rows:
    print("  ❌ No respond.io rows found in raw files. Check file paths.")
    exit()

df_raw = pd.concat(raw_txn_rows, ignore_index=True)
print(f"  Raw cashier respond.io rows:         {len(df_raw):,}")
print(f"  Raw cashier unique transaction nos:  {df_raw['_raw_txn_no'].nunique():,}")
print(f"  Raw cashier revenue:                 KES {df_raw['_raw_amount'].sum():,.0f}")

# ── STEP 2: Check what made it into the processed CSV ─────────────────────────

print("\n" + "=" * 65)
print("  STEP 2 — Checking processed CSV")
print("=" * 65)

if not PROCESSED_CSV.exists():
    print(f"  ❌ Processed CSV not found: {PROCESSED_CSV}")
    exit()

df_proc = pd.read_csv(PROCESSED_CSV, low_memory=False, dtype=str)
df_proc.columns = df_proc.columns.str.strip()

# Check Ordered Via population
ordered_col = next((c for c in df_proc.columns if "ordered" in c.lower()), None)
txn_col     = next((c for c in df_proc.columns if "transaction" in c.lower() and "id" in c.lower()), None)
audit_col   = next((c for c in df_proc.columns if "audit" in c.lower()), None)

print(f"  Total rows in processed CSV:  {len(df_proc):,}")

if ordered_col:
    social_mask = df_proc[ordered_col].fillna("").str.strip().str.lower() == SOCIAL_TAG
    df_proc_social = df_proc[social_mask]
    print(f"  Rows with Ordered Via = respond.io: {len(df_proc_social):,}")
    print(f"  Unique transactions (respond.io):   {df_proc_social[txn_col].nunique() if txn_col else 'n/a'}")
else:
    print("  ⚠️  'Ordered Via' column not found in processed CSV!")
    print(f"  Columns present: {list(df_proc.columns)}")

# Audit status breakdown (shows how many rows had no cashier data)
if audit_col:
    print(f"\n  Audit_Status breakdown (all rows):")
    for status, count in df_proc[audit_col].value_counts().items():
        pct = count / len(df_proc) * 100
        print(f"    {status:<25} {count:>8,}  ({pct:.1f}%)")

# ── STEP 3: Find which raw txn nos are MISSING from processed ─────────────────

print("\n" + "=" * 65)
print("  STEP 3 — Transaction-level gap analysis")
print("=" * 65)

if txn_col and ordered_col:
    proc_social_txns = set(
        df_proc[social_mask][txn_col].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
    )
    raw_txns = set(df_raw["_raw_txn_no"].dropna())

    missing_from_proc  = raw_txns - proc_social_txns
    extra_in_proc      = proc_social_txns - raw_txns

    print(f"  Txn nos in raw cashier files:        {len(raw_txns):,}")
    print(f"  Txn nos in processed CSV (social):   {len(proc_social_txns):,}")
    print(f"  Missing from processed CSV:          {len(missing_from_proc):,}  ← these are your gap")
    print(f"  In processed but not in raw:         {len(extra_in_proc):,}")

    if missing_from_proc:
        # Check if those txn nos exist in processed at ALL (just with no Ordered Via)
        all_proc_txns = set(df_proc[txn_col].astype(str).str.replace(r'\.0$', '', regex=True).str.strip())
        found_but_no_tag = missing_from_proc & all_proc_txns
        truly_absent     = missing_from_proc - all_proc_txns

        print(f"\n  Of the {len(missing_from_proc):,} missing:")
        print(f"    Exist in CSV but Ordered Via is blank/lost:  {len(found_but_no_tag):,}  ← merge dropped the column")
        print(f"    Completely absent from CSV (ETL skipped):    {len(truly_absent):,}  ← ETL didn't load the file")

        if found_but_no_tag:
            print("\n  Sample txn nos that lost their Ordered Via tag:")
            for t in list(found_but_no_tag)[:10]:
                print(f"    {t}")

        if truly_absent:
            print("\n  Sample txn nos completely missing from CSV:")
            for t in list(truly_absent)[:10]:
                print(f"    {t}")

# ── STEP 4: Per-branch, per-month gap ─────────────────────────────────────────

print("\n" + "=" * 65)
print("  STEP 4 — Gap by branch × month")
print("=" * 65)
print(f"  {'Branch':<22} {'Month':<6} {'Raw':>6}  {'In CSV':>8}  {'Gap':>6}")
print(f"  {'-'*55}")

for branch in sorted(df_raw["_branch"].unique()):
    for month in [m.upper() for m in TARGET_MONTHS]:
        grp_raw = df_raw[(df_raw["_branch"] == branch) & (df_raw["_month"] == month)]
        if grp_raw.empty:
            continue
        raw_count = len(grp_raw)

        if txn_col and ordered_col:
            branch_txns = set(grp_raw["_raw_txn_no"].dropna())
            in_csv = len(branch_txns & proc_social_txns)
        else:
            in_csv = "?"

        gap = raw_count - in_csv if isinstance(in_csv, int) else "?"
        flag = "  ⚠️" if isinstance(gap, int) and gap > 0 else ""
        print(f"  {branch:<22} {month:<6} {raw_count:>6,}  {str(in_csv):>8}  {str(gap):>6}{flag}")

        print("\n  Branch breakdown of the 6 absent transactions:")
        absent_rows = df_raw[df_raw["_raw_txn_no"].isin(truly_absent)]
        print(absent_rows[["_raw_txn_no", "_branch", "_month"]].to_string(index=False))