"""
audit_jamieson.py
=================
Diagnostic: investigates brand alias matching for a specific brand (Jamieson).

Filters the attribution output for a target brand and prints all matched
transactions, then tests the brand alias regex patterns. Used as a template
for investigating any specific brand's match quality.

Input:  data/03_processed/sales_attribution/social_sales_Jan25_Jan26.csv
Output: console report of brand-specific transaction rows
"""

# import pandas as pd
# from Portal_ML_V4.src.config.settings import PROCESSED_DATA_DIR

# df = pd.read_csv(PROCESSED_DATA_DIR / "sales_attribution" / "social_sales_Jan25_Jan26.csv")

# jamieson = df[df['Matched_Brand'] == 'Jamieson']
# print(jamieson[['Transaction ID', 'Description', 'Qty Sold',
#                 'Total (Tax Ex)', 'Matched_Brand', 'Matched_Product']].to_string())


import pandas as pd
from Portal_ML_V4.src.config.settings import PROCESSED_DATA_DIR
from Portal_ML_V4.src.config.brands import BRAND_ALIASES
import re

# Read the enriched file (has Matched_Brand and Matched_Product)
df_enriched = pd.read_csv(PROCESSED_DATA_DIR / "sales_attribution" / "social_sales_Jan25_Jan26.csv")
# Read the waterfall file (all rows before filtering)
df_waterfall = pd.read_csv(PROCESSED_DATA_DIR / "sales_attribution" / "attributed_sales_waterfall_v7.csv")

print(f"Waterfall rows:  {len(df_waterfall)}")
print(f"Social sales rows: {len(df_enriched)}")
print(f"Dropped by filter: {len(df_waterfall) - len(df_enriched)}")

# Find what was in waterfall but not in social_sales
merged = df_waterfall[['Transaction ID', 'Description']].merge(
    df_enriched[['Transaction ID', 'Description', 'Matched_Brand', 'Matched_Product']],
    on=['Transaction ID', 'Description'],
    how='left',
    indicator=True
)

dropped = merged[merged['_merge'] == 'left_only']
print(f"\nTotal dropped: {len(dropped)}")
print("\nSample dropped rows:")
print(dropped[['Transaction ID', 'Description']].head(30).to_string())