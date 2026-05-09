# import pandas as pd
# from Portal_ML_V4.src.config.settings import PROCESSED_DATA_DIR

# df = pd.read_csv(PROCESSED_DATA_DIR / "fact_sessions_enriched.csv", nrows=1)

# for col in df.columns:
#     issues = []
#     if len(col) > 80:
#         issues.append("too long")
#     if any(c in col for c in ['[', ']', '#', '@']):
#         issues.append("special chars")
#     if col != col.strip():
#         issues.append("leading/trailing space")
#     if issues:
#         print(f"⚠️  '{col}': {issues}")

# print("\nAll columns:")
# for i, col in enumerate(df.columns):
#     print(f"{i+1:3}. '{col}'")


import pandas as pd

# Paste the exact path PowerBI is using
path = r"D:\\Documents\\Portal ML Analys\\Portal_ML\\Portal_ML_V4\\data\\03_processed\\fact_sessions_enriched.csv"  

df = pd.read_csv(path, nrows=1)
print(f"Columns in THIS file: {len(df.columns)}")
print(list(df.columns))