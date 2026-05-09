"""
ITEM VALUE LIST LOADER
======================
Reads cost/price files from:
    data/01_raw/pos_data/item_value_list/

Expected file naming:
    Centurion 2r Item Value List.xlsx
    Galleria Item Value List.xlsx
    etc.

Column headers are on ROW 3 (index 2) — rows above are skipped.
Loads into dim_products table (truncate + reload on each run).
"""
import os
import re
import warnings
import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine, text

warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

try:
    from Portal_ML_V4.src.config.settings import BASE_DIR, DB_CONNECTION_STRING
except ImportError:
    print("❌ Could not load settings.")
    exit()

ITEM_VALUE_DIR = BASE_DIR / "data" / "01_raw" / "pos_data" / "item_value_list"

# Maps keywords in filename → branch label.
# Keywords are intentionally specific to avoid cross-branch matches
# e.g. "centurion_2r" won't match a file containing just "centurion milele"
BRANCH_FILE_MAP = {
    "CENTURION_2R": ["centurion_2r", "centurion 2r"],
    "GALLERIA":     ["galleria"],
    "NGONG_MILELE": ["ngong_milele", "ngong milele", "milele"],
    "PHARMART_ABC": ["pharmart_abc", "pharmart abc"],
    "PORTAL_2R":    ["portal_2r", "portal 2r"],
    "PORTAL_CBD":   ["portal_cbd", "portal cbd"],
}

# Column mapping: raw file name → DB column name
COL_MAP = {
    'Department':       'department',
    'Category':         'category',
    'Supplier':         'supplier',
    'Item':             'item_barcode',
    'Description':      'description',
    'Cost':             'cost_price',
    'Price':            'selling_price',
}


def detect_branch(filepath: Path) -> str | None:
    fname = filepath.stem.lower()
    for branch, keywords in BRANCH_FILE_MAP.items():
        if any(k in fname for k in keywords):
            return branch
    return None


def load_item_value_file(fpath: Path, branch: str) -> pd.DataFrame:
    print(f"    📂 Loading: {fpath.name}")
    try:
        # Headers are on row 3 (0-indexed row 2) — skip the first 2 rows.
        # CSVs are read with pandas read_csv; Excel files with read_excel.
        fname_lower = fpath.name.lower()
        if fname_lower.endswith('.csv'):
            try:
                df = pd.read_csv(fpath, header=2, low_memory=False)
            except UnicodeDecodeError:
                df = pd.read_csv(fpath, header=2, low_memory=False, encoding='cp1252')
        else:
            df = pd.read_excel(fpath, header=2)
        df.columns = df.columns.astype(str).str.strip()

        # Keep only columns we care about
        existing = {k: v for k, v in COL_MAP.items() if k in df.columns}
        if 'Item' not in existing:
            print(f"    ⚠️  No 'Item' (barcode) column found — skipping.")
            return pd.DataFrame()

        df = df[list(existing.keys())].rename(columns=existing).copy()

        # ── Clean item_barcode ────────────────────────────────────────────
        # Handles scientific notation (6.81E+11 → 681000000000)
        def clean_barcode(val):
            if pd.isna(val):
                return None
            try:
                # Float with scientific notation
                f = float(val)
                if f == int(f):
                    return str(int(f))
                return str(f)
            except (ValueError, TypeError):
                return str(val).strip()

        df['item_barcode'] = df['item_barcode'].apply(clean_barcode)

        # Drop rows with no barcode or no prices at all
        df = df.dropna(subset=['item_barcode'])
        df = df[df['item_barcode'].str.strip() != '']

        # ── Clean numeric columns ─────────────────────────────────────────
        for col in ['cost_price', 'selling_price']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Drop rows where both cost and price are 0 or null — no useful data
        if 'cost_price' in df.columns and 'selling_price' in df.columns:
            before = len(df)
            df = df[~(
                (df['cost_price'].fillna(0) == 0) &
                (df['selling_price'].fillna(0) == 0)
            )]
            dropped = before - len(df)
            if dropped > 0:
                print(f"    🧹 Dropped {dropped:,} rows with zero cost and price")

        # ── Clean text columns ────────────────────────────────────────────
        for col in ['department', 'category', 'supplier', 'description']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
                df[col] = df[col].replace({'nan': None, '': None})

        df['location'] = branch
        print(f"    ✅ {len(df):,} valid products loaded from {fpath.name}")
        return df

    except Exception as e:
        print(f"    ❌ Error loading {fpath.name}: {e}")
        return pd.DataFrame()


# Maps subfolder names → branch label (same convention as pos_data)
SUBFOLDER_BRANCH_MAP = {
    "centurion_2r": "CENTURION_2R",
    "galleria":     "GALLERIA",
    "ngong_milele": "NGONG_MILELE",
    "pharmart_abc": "PHARMART_ABC",
    "portal_2r":    "PORTAL_2R",
    "portal_cbd":   "PORTAL_CBD",
}


def collect_files_with_branches() -> list[tuple[Path, str]]:
    """
    Two-mode file discovery — mirrors how raw_sales works:

    Mode 1 — Subfolder per branch (preferred):
        item_value_list/centurion_2r/Centurion Item Value List.xlsx
        item_value_list/galleria/Galleria Item Value List.xlsx

    Mode 2 — All files flat in item_value_list/ (branch from filename):
        item_value_list/Centurion 2r Item Value List.xlsx
        item_value_list/Galleria Item Value List.xlsx

    Both modes can coexist — subfolder match takes priority.
    """
    results = []

    # Mode 1: walk subfolders
    for subfolder in ITEM_VALUE_DIR.iterdir():
        if not subfolder.is_dir():
            continue
        branch = SUBFOLDER_BRANCH_MAP.get(subfolder.name.lower())
        if branch is None:
            continue
        for ext in ("*.xlsx", "*.xls", "*.csv"):
            for fpath in subfolder.glob(ext):
                if not fpath.name.startswith("~$"):
                    results.append((fpath, branch))

    # Mode 2: flat files in root of item_value_list/
    for ext in ("*.xlsx", "*.xls", "*.csv"):
        for fpath in ITEM_VALUE_DIR.glob(ext):
            if fpath.name.startswith("~$"):
                continue
            branch = detect_branch(fpath)
            if branch:
                results.append((fpath, branch))

    return results


def run_item_value_loader():
    print("=" * 60)
    print("💰 ITEM VALUE LIST LOADER")
    print("=" * 60)

    if not ITEM_VALUE_DIR.exists():
        print(f"❌ Folder not found: {ITEM_VALUE_DIR}")
        print("   Create this folder and place your Item Value List files inside.")
        print("   You can organise by subfolder (centurion_2r/, galleria/ etc.)")
        print("   or just drop files flat and location is read from the filename.")
        return

    file_branch_pairs = collect_files_with_branches()

    if not file_branch_pairs:
        print(f"❌ No files found in {ITEM_VALUE_DIR}")
        return

    engine = create_engine(DB_CONNECTION_STRING)

    try:
        # Truncate first — item value lists are full replacements, not incremental
        with engine.begin() as conn:
            conn.execute(text("TRUNCATE TABLE dim_products"))
        print("🗑️  Cleared old cost data.\n")

        all_dfs    = []
        unmatched  = []

        for fpath, branch in file_branch_pairs:
            print(f"\n📍 {branch}")
            df = load_item_value_file(fpath, branch)
            if not df.empty:
                all_dfs.append(df)

        if not all_dfs:
            print("\n❌ No data loaded.")
            return

        final_df = pd.concat(all_dfs, ignore_index=True)

        # Dedup: if same barcode + location appears twice, keep last (most recent file wins)
        before = len(final_df)
        final_df = final_df.drop_duplicates(
            subset=['item_barcode', 'location'], keep='last'
        )
        dupes = before - len(final_df)
        if dupes > 0:
            print(f"\n🧹 Removed {dupes:,} duplicate barcode+location rows (kept latest)")

        final_df.to_sql(
            'dim_products', engine,
            if_exists='append', index=False, chunksize=10_000
        )

        print(f"\n{'=' * 60}")
        print(f"✅ LOAD COMPLETE")
        print(f"   📦 Total products loaded: {len(final_df):,}")

        # Coverage report — how many stg_sales_reports barcodes have cost data
        with engine.connect() as conn:
            total_barcodes = conn.execute(
                text("SELECT COUNT(DISTINCT item) FROM stg_sales_reports")
            ).scalar()
            costed_barcodes = conn.execute(
                text("""
                    SELECT COUNT(DISTINCT s.item)
                    FROM stg_sales_reports s
                    JOIN dim_products p
                      ON s.item     = p.item_barcode
                     AND s.branch   = p.location
                """)
            ).scalar()

        pct = round(costed_barcodes / total_barcodes * 100, 1) if total_barcodes else 0
        print(f"   📊 Cost coverage: {costed_barcodes:,} / {total_barcodes:,} "
              f"unique barcodes ({pct}%)")

        if unmatched:
            print(f"\n   ⚠️  Files with no branch detected (not loaded):")
            for f in unmatched:
                print(f"      - {f}")

        print(f"{'=' * 60}")

    finally:
        engine.dispose()
        print("🔌 Database connections closed.")


if __name__ == "__main__":
    run_item_value_loader()