"""
audit_fuzzy.py
==============
Diagnostic: spot-checks the fuzzy matching logic in enrich_attribution_products.py.

Runs a side-by-side comparison of POS description vs KB description through
the clean_text_for_matching and fuzzy_similarity functions and prints the
similarity score. Also reports top brand distribution in the attribution output.

Input:  data/03_processed/sales_attribution/social_sales_Jan25_Jan26.csv
Output: console report of match scores and brand value counts
"""

# # from Portal_ML_V4.src.pipelines.attribution.enrich_attribution_products import *

# # a = clean_text_for_matching("CERAVE MOIST LOTION DRY/ V.DRY")
# # b = clean_text_for_matching("CeraVe Moisturizing Lotion For Dry To Very Dry Skin 8oz")
# # print(f"POS clean:  {a}")
# # print(f"KB clean:   {b}")
# # print(f"Score:      {fuzzy_similarity(a, b):.3f}")

# import pandas as pd
# from Portal_ML_V4.src.config.settings import PROCESSED_DATA_DIR

# df = pd.read_csv(PROCESSED_DATA_DIR / "sales_attribution" / "social_sales_Jan25_Jan26.csv")
# print(df['Matched_Brand'].value_counts().head(20))
# print(f"\nUnknown brands: {(df['Matched_Brand'] == 'Unknown').sum():,}")
# print(f"General products: {df['Matched_Product'].str.startswith('General').sum():,}")
# print(f"Exact matches: {(~df['Matched_Product'].str.startswith('General') & (df['Matched_Brand'] != 'Unknown')).sum():,}")


import pandas as pd
from Portal_ML_V4.src.config.settings import PROCESSED_DATA_DIR

df = pd.read_csv(PROCESSED_DATA_DIR / "fact_sessions_enriched.csv", nrows=1)
print(f"Total columns: {len(df.columns)}")
print(list(df.columns))