"""
audit_again.py
==============
Diagnostic: reconciles M-Pesa total from sessions against social sales matches.

Loads fact_sessions_enriched.csv, sums all M-Pesa amounts for converted
sessions, then splits the total into matched vs unmatched against the
social_sales_Jan25_Jan26.csv attribution output.

Inputs:
    data/03_processed/fact_sessions_enriched.csv
    data/03_processed/sales_attribution/social_sales_Jan25_Jan26.csv
Output: console report of matched vs unmatched M-Pesa revenue
"""

import pandas as pd
from Portal_ML_V4.src.config.settings import PROCESSED_DATA_DIR

df_sess  = pd.read_csv(PROCESSED_DATA_DIR / "fact_sessions_enriched.csv")
df_social = pd.read_csv(PROCESSED_DATA_DIR / "sales_attribution" / "social_sales_Jan25_Jan26.csv")

df_conv = df_sess[(df_sess['is_converted']==1) | (df_sess['mpesa_amount']>0)].copy()
df_conv['mpesa_amount'] = pd.to_numeric(df_conv['mpesa_amount'], errors='coerce').fillna(0)

total_mpesa    = df_conv['mpesa_amount'].sum()
matched_ids    = set(df_social['session_id'].dropna()) if 'session_id' in df_social.columns else set()
matched        = df_conv[df_conv['session_id'].isin(matched_ids)]
unmatched      = df_conv[~df_conv['session_id'].isin(matched_ids)]

print(f"Total M-Pesa (all converted sessions):  KES {total_mpesa:>12,.0f}")
print(f"In social_sales (POS matched):          KES {matched['mpesa_amount'].sum():>12,.0f}")
print(f"GAP (confirmed but unmatched):          KES {unmatched['mpesa_amount'].sum():>12,.0f}")
print(f"\nGap breakdown by acquisition source:")
print(unmatched.groupby('acquisition_source')['mpesa_amount'].sum().sort_values(ascending=False).to_string())
print(f"\nGap breakdown by channel:")
print(unmatched.groupby('channel_name')['mpesa_amount'].sum().sort_values(ascending=False).to_string())
print(f"\nGap breakdown by matched_brand:")
print(unmatched.groupby('matched_brand')['mpesa_amount'].sum().sort_values(ascending=False).head(15).to_string())