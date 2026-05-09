"""
audit_reusable.py
=================
Diagnostic: inspects column schema of the POS sales CSV.

Loads the first row of all_locations_sales_Jan25-Jan26.csv and prints the
column list. Used to verify ETL output column names after a schema change.

Input:  data/03_processed/pos_data/all_locations_sales_Jan25-Jan26.csv
Output: console print of column names
"""

import pandas as pd
from Portal_ML_V4.src.config.settings import PROCESSED_DATA_DIR

POS_DATA_PATH = PROCESSED_DATA_DIR / "pos_data" / "all_locations_sales_Jan25-Jan26.csv"
df_peek = pd.read_csv(POS_DATA_PATH, nrows=0)
print(df_peek.columns.tolist())