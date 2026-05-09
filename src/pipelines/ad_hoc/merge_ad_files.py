"""
merge_ad_files.py
=================
Merges all Meta Ads CSV exports into a single consolidated ads file.

Scans for all Meta Ads CSV files matching the expected naming pattern,
deduplicates on Ad ID, applies AD_NAME_MAP to normalise campaign names,
and writes the merged result for use by ad_performance.py.

Input:  data/01_raw/meta_ads/Caroline-Mwangi-Ads-*.csv  (multiple date-range files)
Output: data/03_processed/ads/all_ads_merged.csv

Run manually after downloading a new Meta Ads export.
"""

import glob
from pathlib import Path

import pandas as pd

from Portal_ML_V4.src.config.ad_name_map import AD_NAME_MAP
from Portal_ML_V4.src.config.settings import MSG_HISTORY_RAW, PROCESSED_DATA_DIR

ADS_OUTPUT_FILE = PROCESSED_DATA_DIR / "ads" / "all_ads_merged.csv"


def clean_scientific_id(val):
    """Converts scientific notation strings (6.95E+12) back to flat integer strings."""
    if pd.isna(val) or str(val).strip() in ('', '-', 'nan', 'None'):
        return pd.NA
    try:
        return str(int(float(str(val).strip())))
    except ValueError:
        return str(val).strip().replace('.0', '')


def load_ads_for_analytics():
    ads_dir = Path(MSG_HISTORY_RAW).parent / "ads"
    all_files = glob.glob(str(ads_dir / "contacts-*.csv"))
    
    if not all_files:
        return pd.DataFrame(), set(), set()

    dfs = []
    cols = ['Timestamp', 'Contact ID', 'Source', 'Ad campaign ID', 'Ad group ID', 'Ad ID']

    for f in all_files:
        try:
            t = pd.read_csv(f, dtype=str, keep_default_na=False)
            t['Timestamp'] = pd.to_datetime(t['Timestamp'], errors='coerce', format='mixed')
            
            t['Contact ID'] = pd.to_numeric(
                t['Contact ID'].astype(str).str.strip().str.replace(r'\.0$', '', regex=True),
                errors='coerce'
            ).astype('Int64')

            avail = [c for c in cols if c in t.columns]
            dfs.append(t.dropna(subset=['Contact ID', 'Timestamp'])[avail])

        except Exception as e:
            print(f"   Could not read {Path(f).name}: {e}")

    if not dfs:
        return pd.DataFrame(), set(), set()

    # Sort DESCENDING to view newest data at the top
    df = pd.concat(dfs, ignore_index=True).sort_values('Timestamp', ascending=False)

    # Clean the IDs to remove scientific notation
    id_cols = ['Ad campaign ID', 'Ad group ID', 'Ad ID']
    for c in id_cols:
        if c in df.columns:
            df[c] = df[c].apply(clean_scientific_id)

    df = df.drop_duplicates(subset=['Contact ID', 'Timestamp', 'Ad campaign ID', 'Ad ID'])

    # ── Map Ad Names ──────────────────────────────────────────────────────────
    if 'Ad ID' in df.columns:
        df['Ad Name'] = df['Ad ID'].map(AD_NAME_MAP)
        
        # Reorder columns to place Ad Name immediately after Ad ID
        all_cols = list(df.columns)
        all_cols.remove('Ad Name')
        ad_id_index = all_cols.index('Ad ID')
        all_cols.insert(ad_id_index + 1, 'Ad Name')
        df = df[all_cols]

    # ── Save merged & deduplicated ads ────────────────────────────────────────
    ADS_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ADS_OUTPUT_FILE, index=False)
    print(f"   ✅ Merged ads saved: {len(df):,} rows → {ADS_OUTPUT_FILE}")

    # Build set of Contact IDs that came from Paid Ads
    if 'Source' in df.columns:
        paid_contact_ids = set(
            df[df['Source'].fillna('').str.lower().str.strip() == 'paid ads']['Contact ID']
            .dropna()
            .astype(int)
            .tolist()
        )
        organic_contact_ids = set(
            df[df['Source'].fillna('').str.lower().str.strip().isin(['incoming message', 'echo message'])]['Contact ID']
            .dropna()
            .astype(int)
            .tolist()
        )
    else:
        paid_contact_ids = set(
            df[df['Ad campaign ID'].notna()]['Contact ID']
            .dropna()
            .astype(int)
            .tolist()
        )
        organic_contact_ids = set()

    print(f"   Ads loaded: {len(df):,} records | "
          f"{len(paid_contact_ids):,} paid | {len(organic_contact_ids):,} organic")

    return df, paid_contact_ids, organic_contact_ids


if __name__ == "__main__":
    load_ads_for_analytics()