"""
check_categories.py
===================
Compares the Matched_Category values in social_sales_direct.csv against
the approved categories list from the Knowledge Base copy file.

Sources:
  Approved list : Final_Knowledge_Base_PowerBI - Copy V3(Sheet1).csv  → column 'Categories'
  Actuals       : sales_attribution/social_sales_direct.csv            → column 'Matched_Category'

NOTE: Matched_Category can contain comma-separated values (e.g. "Skincare, Medicine & Treatment").
      These are exploded to one category per row before auditing so each category
      is assessed individually.

Run:
    python check_categories.py
"""

from pathlib import Path
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
try:
    from Portal_ML_V4.src.config.settings import BASE_DIR, PROCESSED_DATA_DIR
except ImportError:
    BASE_DIR           = Path(r"D:\Documents\Portal ML Analys\Portal_ML\Portal_ML_V4")
    PROCESSED_DATA_DIR = BASE_DIR / "data" / "03_processed"

RAW_DATA_DIR = BASE_DIR / "data" / "01_raw"

CATEGORIES_FILE   = RAW_DATA_DIR / "Final_Knowledge_Base_PowerBI - Copy V3(Sheet1).csv"
SOCIAL_SALES_FILE = PROCESSED_DATA_DIR / "sales_attribution" / "social_sales_direct.csv"

# ── Load approved categories (column B = 'Categories') ────────────────────────
print(f"📋 Loading approved categories from:")
print(f"   {CATEGORIES_FILE.name}\n")

if not CATEGORIES_FILE.exists():
    print(f"❌ File not found: {CATEGORIES_FILE}")
    raise SystemExit(1)

df_cats = pd.read_csv(CATEGORIES_FILE, usecols=["Categories"], dtype=str)
approved = set(
    df_cats["Categories"]
    .dropna()
    .str.strip()
    .loc[lambda s: s != ""]
    .unique()
)
print(f"   {len(approved)} approved categories loaded:")
for c in sorted(approved):
    print(f"      • {c}")

# ── Load and explode Matched_Category ─────────────────────────────────────────
print(f"\n📊 Loading Matched_Category from:")
print(f"   {SOCIAL_SALES_FILE.name}\n")

if not SOCIAL_SALES_FILE.exists():
    print(f"❌ File not found: {SOCIAL_SALES_FILE}")
    raise SystemExit(1)

df_sales = pd.read_csv(
    SOCIAL_SALES_FILE,
    usecols=["Matched_Category"],
    dtype=str,
    low_memory=False,
)

# Explode comma-separated categories — same logic as in social_sales_etl.py
df_exploded = df_sales.copy()
df_exploded["Matched_Category"] = (
    df_exploded["Matched_Category"].fillna("(blank)").str.split(",")
)
df_exploded = df_exploded.explode("Matched_Category")
df_exploded["Matched_Category"] = df_exploded["Matched_Category"].str.strip()
df_exploded = df_exploded[
    df_exploded["Matched_Category"].notna()
    & (df_exploded["Matched_Category"] != "")
]

raw_rows      = len(df_sales)
exploded_rows = len(df_exploded)
multi_cat_rows = raw_rows - df_sales["Matched_Category"].str.contains(",", na=False).sum()

print(f"   Raw rows          : {raw_rows:,}")
print(f"   After explode     : {exploded_rows:,}  ({exploded_rows - raw_rows:+,} from multi-category rows)")

actual_counts = df_exploded["Matched_Category"].value_counts()
total_cats    = actual_counts.sum()

# ── Compare ───────────────────────────────────────────────────────────────────
matched_cats   = {c: n for c, n in actual_counts.items() if c in approved}
unmatched_cats = {c: n for c, n in actual_counts.items() if c not in approved}
unused_approved = approved - set(actual_counts.index)

print(f"\n{'='*60}")
print(f"  CATEGORY AUDIT  ({total_cats:,} category rows after explode)")
print(f"{'='*60}")

print(f"\n✅  IN approved list ({len(matched_cats)} values):")
print(f"   {'Category':<35} {'Rows':>8}  {'%':>6}")
print(f"   {'-'*52}")
for cat, n in sorted(matched_cats.items(), key=lambda x: -x[1]):
    print(f"   {cat:<35} {n:>8,}  {n/total_cats*100:>5.1f}%")

print(f"\n❌  NOT in approved list ({len(unmatched_cats)} values) — need mapping:")
print(f"   {'Category':<35} {'Rows':>8}  {'%':>6}")
print(f"   {'-'*52}")
for cat, n in sorted(unmatched_cats.items(), key=lambda x: -x[1]):
    print(f"   {cat:<35} {n:>8,}  {n/total_cats*100:>5.1f}%")

if unused_approved:
    print(f"\n📭  Approved categories with zero rows in output ({len(unused_approved)}):")
    for c in sorted(unused_approved):
        print(f"      • {c}")

print(f"\n{'='*60}")
rows_ok  = sum(matched_cats.values())
rows_bad = sum(unmatched_cats.values())
print(f"  Rows on approved list : {rows_ok:>8,}  ({rows_ok/total_cats*100:.1f}%)")
print(f"  Rows needing mapping  : {rows_bad:>8,}  ({rows_bad/total_cats*100:.1f}%)")
print(f"{'='*60}")