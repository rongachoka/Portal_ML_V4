"""
product_check.py
================
Diagnostic: reports unmatched products and brands in the attribution output.

Prints the top 20 POS descriptions where Matched_Product == "Unknown" and
where Matched_Brand == "Unknown", to guide KB expansion or alias additions.

Input:  data/03_processed/sales_attribution/final_enriched_social_sales_Jan25_Jan26.csv
Output: console report

Run manually after a pipeline run to check KB coverage.
"""

import pandas as pd
from Portal_ML_V4.src.config.settings import PROCESSED_DATA_DIR

# Load your new file
file_path = PROCESSED_DATA_DIR / "sales_attribution" / "final_enriched_social_sales_Jan25_Jan26.csv"
df = pd.read_csv(file_path)

# Filter for Unknown Products
unknowns = df[df['Matched_Product'] == 'Unknown']

# Count the most frequent unmatched descriptions
print("--- TOP 20 UNMATCHED ITEMS ---")
print(unknowns['Description'].value_counts().head(20))

# Filter for Unknown Brands
unknown_brands = df[df['Matched_Brand'] == 'Unknown']
print("\n--- TOP 20 UNMATCHED BRANDS (Context) ---")
print(unknown_brands['Description'].value_counts().head(20))