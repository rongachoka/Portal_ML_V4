"""
ELT LOAD VALIDATOR
==================
Compares row counts between source files and PostgreSQL.
Produces a summary table + a detailed CSV for investigation.

Run after database_loader_v1.py completes.
"""
import os
import glob
import re
import warnings
import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine, text
from datetime import datetime

warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

try:
    from Portal_ML_V4.src.config.settings import BASE_DIR, DB_CONNECTION_STRING
except ImportError:
    print("❌ Could not load settings. Check your settings.py.")
    exit()

RAW_DIR    = BASE_DIR / "data" / "01_raw" / "pos_data"
CLOSED_DIR = RAW_DIR / "Closed"
DEC_DIR    = RAW_DIR / "Dec 2025"
LOG_DIR    = BASE_DIR / "data" / "logs" / "validation"
os.makedirs(LOG_DIR, exist_ok=True)

LOCATION_MAP = {
    "centurion_2R": ["centurion", "c2r"],
    "galleria":     ["galleria"],
    "ngong_milele": ["milele"],
    "pharmart_abc": ["abc", "pharmart abc"],
    "portal_2R":    ["portal 2r", "portal_2r", "portal 2 rivers"],
    "portal_cbd":   ["cbd", "portal cbd"],
}

MONTH_MAP = {
    'jan': 'Jan', 'feb': 'Feb', 'mar': 'Mar', 'apr': 'Apr',
    'may': 'May', 'jun': 'Jun', 'jul': 'Jul', 'aug': 'Aug',
    'sep': 'Sep', 'oct': 'Oct', 'nov': 'Nov', 'dec': 'Dec',
    'january': 'Jan', 'february': 'Feb', 'march': 'Mar', 'april': 'Apr',
    'june': 'Jun', 'july': 'Jul', 'august': 'Aug', 'september': 'Sep',
    'october': 'Oct', 'november': 'Nov', 'december': 'Dec',
}

# ── HELPERS ────────────────────────────────────────────────────────────────

def is_numbered_sheet(sheet_name):
    return bool(re.fullmatch(r'0[1-9]|[12][0-9]|3[01]', str(sheet_name).strip()))

def is_cashier_file(fname, branch_keywords):
    return any(k in fname.lower() for k in branch_keywords)

def count_sales_rows_in_file(fpath):
    """Count rows with Transaction ID + Date Sold in a sales file."""
    try:
        fname = os.path.basename(fpath).lower()
        if fname.endswith('.csv'):
            try:    df = pd.read_csv(fpath, low_memory=False)
            except: df = pd.read_csv(fpath, low_memory=False, encoding='cp1252')
        else:
            df = pd.read_excel(fpath)
        df.columns = df.columns.astype(str).str.strip()
        if 'Transaction ID' not in df.columns or 'Date Sold' not in df.columns:
            return 0, 'no_txn_col'
        valid = df.dropna(subset=['Transaction ID'])
        return len(valid), 'ok'
    except Exception as e:
        return 0, str(e)

def count_cashier_rows_in_file(fpath):
    """Count rows with Receipt Txn No across all numbered sheets."""
    total = 0
    try:
        if str(fpath).endswith('.csv'):
            df = pd.read_csv(fpath, low_memory=False)
            df.columns = df.columns.astype(str).str.strip()
            if 'Receipt Txn No' in df.columns:
                total = df['Receipt Txn No'].notna().sum()
        else:
            xls = pd.ExcelFile(fpath)
            for sheet in xls.sheet_names:
                if not is_numbered_sheet(sheet):
                    continue
                df = pd.read_excel(xls, sheet_name=sheet)
                df.columns = df.columns.astype(str).str.strip()
                if 'Receipt Txn No' in df.columns:
                    total += df['Receipt Txn No'].notna().sum()
    except Exception as e:
        return 0, str(e)
    return total, 'ok'

# ── MAIN VALIDATION ────────────────────────────────────────────────────────

def run_validation():
    print("=" * 65)
    print("🔍 ELT LOAD VALIDATOR")
    print("=" * 65)

    engine = create_engine(DB_CONNECTION_STRING)
    ts     = datetime.now().strftime('%Y%m%d_%H%M%S')
    rows   = []

    for folder_name, branch_keywords in LOCATION_MAP.items():
        loc_path    = RAW_DIR / folder_name
        branch_label = folder_name.upper()

        if not loc_path.exists():
            print(f"\n⚠️  {branch_label}: folder not found, skipping.")
            continue

        print(f"\n📍 {branch_label}")

        # ── DB counts ─────────────────────────────────────────────────────
        with engine.connect() as conn:
            db_sales = conn.execute(
                text("SELECT COUNT(*) FROM raw_sales WHERE location = :loc"),
                {"loc": branch_label}
            ).scalar()
            db_cashier = conn.execute(
                text("SELECT COUNT(*) FROM raw_cashier WHERE location = :loc"),
                {"loc": branch_label}
            ).scalar()

        # ── SALES: count source rows ───────────────────────────────────────
        src_sales = 0
        sales_files = [
            f for f in glob.glob(str(loc_path / "*"))
            if not os.path.basename(f).lower().startswith('~$')
            and "cashier" not in os.path.basename(f).lower()
            and (f.endswith('.csv') or f.endswith('.xlsx'))
        ]
        for fpath in sales_files:
            n, status = count_sales_rows_in_file(fpath)
            src_sales += n
            rows.append({
                'branch': branch_label, 'type': 'sales',
                'file': os.path.basename(fpath),
                'source_rows': n, 'status': status,
            })

        # ── CASHIER: current files ─────────────────────────────────────────
        src_cashier = 0
        cashier_files = (
            list(loc_path.glob("*Cashier*.xlsm")) +
            list(loc_path.glob("*Cashier*.xlsx"))
        )
        # Historical files
        for hist_dir in [DEC_DIR, CLOSED_DIR]:
            if hist_dir.exists():
                for f in hist_dir.rglob("*.xlsm"):
                    if is_cashier_file(f.name, branch_keywords):
                        cashier_files.append(f)
                for f in hist_dir.rglob("*.xlsx"):
                    if is_cashier_file(f.name, branch_keywords):
                        cashier_files.append(f)

        for fpath in cashier_files:
            if "~$" in fpath.name:
                continue
            n, status = count_cashier_rows_in_file(fpath)
            src_cashier += n
            rows.append({
                'branch': branch_label, 'type': 'cashier',
                'file': fpath.name,
                'source_rows': n, 'status': status,
            })

        # ── COMPARE ───────────────────────────────────────────────────────
        sales_diff   = src_sales   - db_sales
        cashier_diff = src_cashier - db_cashier

        sales_flag   = "✅" if sales_diff   == 0 else ("⚠️ " if sales_diff   > 0 else "➕")
        cashier_flag = "✅" if cashier_diff == 0 else ("⚠️ " if cashier_diff > 0 else "➕")

        print(f"    Sales:   source={src_sales:>8,}  db={db_sales:>8,}  diff={sales_diff:>+8,}  {sales_flag}")
        print(f"    Cashier: source={src_cashier:>8,}  db={db_cashier:>8,}  diff={cashier_diff:>+8,}  {cashier_flag}")

        rows.append({
            'branch': branch_label, 'type': 'SUMMARY_SALES',
            'file': '— TOTAL —',
            'source_rows': src_sales,
            'db_rows': db_sales,
            'diff': sales_diff,
            'status': sales_flag,
        })
        rows.append({
            'branch': branch_label, 'type': 'SUMMARY_CASHIER',
            'file': '— TOTAL —',
            'source_rows': src_cashier,
            'db_rows': db_cashier,
            'diff': cashier_diff,
            'status': cashier_flag,
        })

    # ── SAVE REPORT ───────────────────────────────────────────────────────
    out_path = LOG_DIR / f"validation_{ts}.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)

    print(f"\n{'=' * 65}")
    print(f"📄 Full file-level report: {out_path}")
    print(f"{'=' * 65}")
    print("\nLEGEND:  ✅ exact match  |  ⚠️  source > db (possible gap)  |  ➕ db > source (dedup removed extras)")
    engine.dispose()

if __name__ == "__main__":
    run_validation()