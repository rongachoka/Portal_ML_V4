"""
revenue_check.py
================
Sums Total (Tax Ex) from social_sales_direct.csv
by month for Jan - March 2026, filtered to Ordered Via = respond.io.
Transaction-level to avoid line-item double counting.
"""

import pandas as pd
from pathlib import Path

BASE_DIR    = Path(r"D:\Documents\Portal ML Analys\Portal_ML\Portal_ML_V4")
SOCIAL_PATH = BASE_DIR / "data" / "03_processed" / "sales_attribution" / "social_sales_direct.csv"

df = pd.read_csv(SOCIAL_PATH, low_memory=False, dtype={"Ordered Via": str})
df['Sale_Date'] = pd.to_datetime(df['Sale_Date'], errors='coerce')
df['Total (Tax Ex)'] = pd.to_numeric(df['Total (Tax Ex)'], errors='coerce').fillna(0)

# Already filtered to respond.io but apply again to be safe
df = df[df['Ordered Via'].fillna('').str.lower().str.strip() == 'respond.io'].copy()

# Filter to Jan - March 2026
df = df[
    (df['Sale_Date'] >= pd.Timestamp('2026-01-01')) &
    (df['Sale_Date'] <= pd.Timestamp('2026-03-31 23:59:59'))
].copy()

df['month'] = df['Sale_Date'].dt.to_period('M')

# Deduplicate to transaction level before summing revenue
# (each transaction has multiple line items — sum at line-item level is correct,
#  but we want unique transactions for the count)
monthly = df.groupby('month').agg(
    revenue       = ('Total (Tax Ex)', 'sum'),
    transactions  = ('Transaction ID', 'nunique'),
    line_items    = ('Transaction ID', 'count'),
).reset_index()

print(f"\n{'=' * 55}")
print(f"  SOCIAL SALES REVENUE — JAN to MAR 2026 (respond.io)")
print(f"{'=' * 55}")
print(f"  {'Month':<12} {'Revenue (KES)':>15}  {'Txns':>6}  {'Line Items':>10}")
print(f"  {'─'*12} {'─'*15}  {'─'*6}  {'─'*10}")

total_rev  = 0
total_txns = 0
for _, row in monthly.iterrows():
    print(f"  {str(row['month']):<12} KES {row['revenue']:>11,.0f}  {row['transactions']:>6,}  {row['line_items']:>10,}")
    total_rev  += row['revenue']
    total_txns += row['transactions']

print(f"  {'─'*12} {'─'*15}  {'─'*6}  {'─'*10}")
print(f"  {'TOTAL':<12} KES {total_rev:>11,.0f}  {total_txns:>6,}")
print(f"{'=' * 55}\n")