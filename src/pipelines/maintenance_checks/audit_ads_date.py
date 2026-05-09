"""
diagnose_ads.py
─────────────────────────────────────────────────────────────
Audit the contacts-added / contacts-connected ad CSV files.

Run from project root:
    python diagnose_ads.py

Outputs
  1. Per-file date range + row count
  2. Monthly coverage heatmap (gaps show up as empty cells)
  3. Overlap: how many Contact IDs appear in BOTH file types
  4. Contacts-only-in-added vs contacts-only-in-connected
─────────────────────────────────────────────────────────────
"""

import glob
import re
from pathlib import Path

import pandas as pd

# ── Point this at your ads folder ──────────────────────────
from Portal_ML_V4.src.config.settings import MSG_HISTORY_RAW
ADS_DIR = Path(MSG_HISTORY_RAW).parent / "ads"
# ────────────────────────────────────────────────────────────


def load_file(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df.columns = df.columns.str.strip()

    if "Timestamp" not in df.columns:
        print(f"   ⚠️  No 'Timestamp' column in {Path(path).name} — skipping")
        return pd.DataFrame()

    s = df["Timestamp"].str.strip().str.replace("T", " ", regex=False)
    df["Timestamp"] = pd.to_datetime(s, errors="coerce", format="mixed")
    df["Contact ID"] = pd.to_numeric(
        df["Contact ID"].str.strip().str.replace(r"\.0$", "", regex=True),
        errors="coerce",
    ).astype("Int64")

    return df.dropna(subset=["Timestamp", "Contact ID"])


def file_type(path: str) -> str:
    name = Path(path).name.lower()
    if "added" in name:
        return "added"
    if "connected" in name:
        return "connected"
    return "other"


# ── 1. LOAD ALL FILES ───────────────────────────────────────
added_files     = sorted(glob.glob(str(ADS_DIR / "contacts-added*.csv")))
connected_files = sorted(glob.glob(str(ADS_DIR / "contacts-connected*.csv")))
all_files       = added_files + connected_files

if not all_files:
    print(f"❌ No ad files found in: {ADS_DIR}")
    raise SystemExit

rows_added     = []
rows_connected = []
file_summaries = []

for f in all_files:
    ftype = file_type(f)
    df    = load_file(f)
    if df.empty:
        continue

    min_ts = df["Timestamp"].min()
    max_ts = df["Timestamp"].max()
    n_rows = len(df)
    n_ids  = df["Contact ID"].nunique()

    file_summaries.append(
        {
            "File":       Path(f).name,
            "Type":       ftype,
            "Rows":       n_rows,
            "Unique IDs": n_ids,
            "First Date": min_ts.date(),
            "Last Date":  max_ts.date(),
            "Span (days)": (max_ts - min_ts).days,
        }
    )

    if ftype == "added":
        rows_added.append(df)
    elif ftype == "connected":
        rows_connected.append(df)

df_added     = pd.concat(rows_added,     ignore_index=True) if rows_added     else pd.DataFrame()
df_connected = pd.concat(rows_connected, ignore_index=True) if rows_connected else pd.DataFrame()


# ── 2. PER-FILE SUMMARY ─────────────────────────────────────
sep = "─" * 90
print(f"\n{sep}")
print("  PER-FILE DATE RANGES")
print(sep)
summary_df = pd.DataFrame(file_summaries)
print(summary_df.to_string(index=False))


# ── 3. AGGREGATE RANGE BY TYPE ──────────────────────────────
print(f"\n{sep}")
print("  AGGREGATE DATE RANGE BY FILE TYPE")
print(sep)

for label, df in [("contacts-added", df_added), ("contacts-connected", df_connected)]:
    if df.empty:
        print(f"  {label:30s}  NO DATA")
        continue
    print(
        f"  {label:30s}  "
        f"{df['Timestamp'].min().date()}  →  {df['Timestamp'].max().date()}  "
        f"({df['Timestamp'].max() - df['Timestamp'].min()})  |  "
        f"{len(df):,} rows  |  {df['Contact ID'].nunique():,} unique contacts"
    )


# ── 4. MONTHLY COVERAGE HEATMAP ─────────────────────────────
print(f"\n{sep}")
print("  MONTHLY ROW COUNT  (blank = no data that month — likely a gap)")
print(sep)

def monthly_counts(df, label):
    if df.empty:
        return pd.Series(dtype=int, name=label)
    df = df.copy()
    df["YearMonth"] = df["Timestamp"].dt.to_period("M")
    counts = df.groupby("YearMonth").size()
    counts.name = label
    return counts

s_added     = monthly_counts(df_added,     "added")
s_connected = monthly_counts(df_connected, "connected")

heat = pd.concat([s_added, s_connected], axis=1).fillna(0).astype(int)
heat.index = heat.index.astype(str)
heat["total"] = heat.sum(axis=1)

print(heat.to_string())

# Flag months where one file type is completely absent
added_months     = set(s_added.index.astype(str))
connected_months = set(s_connected.index.astype(str))
all_months       = added_months | connected_months

gaps_added     = sorted(all_months - added_months)
gaps_connected = sorted(all_months - connected_months)

if gaps_added:
    print(f"\n  ⚠️  Months with NO 'added' data:     {', '.join(gaps_added)}")
if gaps_connected:
    print(f"  ⚠️  Months with NO 'connected' data:  {', '.join(gaps_connected)}")


# ── 5. OVERLAP ANALYSIS ─────────────────────────────────────
print(f"\n{sep}")
print("  CONTACT ID OVERLAP (added ∩ connected)")
print(sep)

if not df_added.empty and not df_connected.empty:
    ids_added     = set(df_added["Contact ID"].dropna())
    ids_connected = set(df_connected["Contact ID"].dropna())
    ids_both      = ids_added & ids_connected
    ids_only_added     = ids_added - ids_connected
    ids_only_connected = ids_connected - ids_added

    total_unique = len(ids_added | ids_connected)

    print(f"  Total unique Contact IDs (union):        {total_unique:,}")
    print(f"  In BOTH added & connected:               {len(ids_both):,}  "
          f"({len(ids_both)/total_unique*100:.1f}% of union)")
    print(f"  Only in contacts-added:                  {len(ids_only_added):,}  "
          f"({len(ids_only_added)/total_unique*100:.1f}%)")
    print(f"  Only in contacts-connected:              {len(ids_only_connected):,}  "
          f"({len(ids_only_connected)/total_unique*100:.1f}%)")

    # Contacts in both — how far apart are their timestamps?
    if ids_both:
        first_added = (
            df_added[df_added["Contact ID"].isin(ids_both)]
            .groupby("Contact ID")["Timestamp"].min()
            .rename("ts_added")
        )
        first_connected = (
            df_connected[df_connected["Contact ID"].isin(ids_both)]
            .groupby("Contact ID")["Timestamp"].min()
            .rename("ts_connected")
        )
        joined = pd.concat([first_added, first_connected], axis=1).dropna()
        joined["lag_hours"] = (
            joined["ts_connected"] - joined["ts_added"]
        ).dt.total_seconds() / 3600

        print(f"\n  For IDs in both files — lag (connected - added):")
        print(f"    Median : {joined['lag_hours'].median():.1f} h")
        print(f"    Mean   : {joined['lag_hours'].mean():.1f} h")
        print(f"    Min    : {joined['lag_hours'].min():.1f} h")
        print(f"    Max    : {joined['lag_hours'].max():.1f} h")
        print(f"    (Negative = connected timestamp came BEFORE added — worth investigating)")
else:
    print("  ⚠️  One or both file types is empty — overlap cannot be computed.")


# ── 6. DAILY DENSITY WITHIN EACH FILE ───────────────────────
print(f"\n{sep}")
print("  DAILY ROW DENSITY  (top 10 busiest days per file type)")
print(sep)

for label, df in [("added", df_added), ("connected", df_connected)]:
    if df.empty:
        print(f"  {label}: no data")
        continue
    daily = df.groupby(df["Timestamp"].dt.date).size().sort_values(ascending=False)
    print(f"\n  [{label}]")
    print(daily.head(10).to_string())

print(f"\n{sep}\n")