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