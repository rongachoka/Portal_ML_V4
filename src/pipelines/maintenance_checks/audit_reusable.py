import pandas as pd
from Portal_ML_V4.src.config.settings import PROCESSED_DATA_DIR

POS_DATA_PATH = PROCESSED_DATA_DIR / "pos_data" / "all_locations_sales_Jan25-Jan26.csv"
df_peek = pd.read_csv(POS_DATA_PATH, nrows=0)
print(df_peek.columns.tolist())