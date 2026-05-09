
import pandas as pd, glob
from pathlib import Path

ADS_DIR = Path(r'D:\Documents\Portal ML Analys\Portal_ML\Portal_ML_V4\data\01_raw\ads')
ENRICHED = Path(r'D:\Documents\Portal ML Analys\Portal_ML\Portal_ML_V4\data\03_processed\fact_sessions_enriched.csv')

files = glob.glob(str(ADS_DIR / 'contacts-*.csv'))
dfs = []
for f in files:
    t = pd.read_csv(f, dtype=str, keep_default_na=False)
    t['Timestamp'] = pd.to_datetime(t['Timestamp'], errors='coerce', format='mixed')
    t['Contact ID'] = pd.to_numeric(t['Contact ID'], errors='coerce')
    dfs.append(t.dropna(subset=['Contact ID','Timestamp']))
df_ads = pd.concat(dfs).sort_values('Timestamp')
df_ads_first = df_ads.groupby('Contact ID')['Timestamp'].min().reset_index()
df_ads_first.columns = ['Contact ID', 'first_ad_click']

df = pd.read_csv(ENRICHED, low_memory=False, usecols=['Contact ID','session_start','acquisition_source'])
df['session_start'] = pd.to_datetime(df['session_start'], errors='coerce')
df['Contact ID'] = pd.to_numeric(df['Contact ID'], errors='coerce')

march = df[(df['session_start'] >= '2026-03-01') & (df['session_start'] <= '2026-03-31')]
organic_march = march[march['acquisition_source'].isin(['Organic / Direct','Inbound / Unknown'])]

merged = organic_march.merge(df_ads_first, on='Contact ID', how='inner')
merged = merged[merged['session_start'] >= merged['first_ad_click']].copy()
merged['gap_hours'] = (merged['session_start'] - merged['first_ad_click']).dt.total_seconds() / 3600

print('Gap stats (hours):')
print(merged['gap_hours'].describe().round(1))
print()
buckets = [0,1,6,24,48,72,96,168,float('inf')]
labels  = ['0-1h','1-6h','6-24h','1-2d','2-3d','3-4d','4-7d','7d+']
merged['bucket'] = pd.cut(merged['gap_hours'], bins=buckets, labels=labels)
print(merged['bucket'].value_counts().sort_index())
