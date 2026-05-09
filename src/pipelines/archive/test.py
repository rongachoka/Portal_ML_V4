import pandas as pd
df = pd.read_parquet("data/02_interim/final_tagged_sessions.parquet")
print(df['full_context'].str.contains('{').value_counts())