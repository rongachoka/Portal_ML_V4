import pandas as pd
import os
from pathlib import Path

# SETTINGS - UPDATE IF NEEDED
BASE_DIR = Path(os.getcwd()) # Assumes you run from root
MSG_PARQUET = BASE_DIR / "data" / "02_interim" / "cleaned_messages.parquet"
SESS_PARQUET = BASE_DIR / "data" / "03_processed" / "final_tagged_sessions.parquet"

def inspect():
    print("🔍 INSPECTING DATA STRUCTURE...\n")

    # 1. CHECK MESSAGES
    if os.path.exists(MSG_PARQUET):
        print(f"📂 Found Messages File: {MSG_PARQUET}")
        try:
            df_msg = pd.read_parquet(MSG_PARQUET)
            print(f"   - Columns: {list(df_msg.columns)}")
            print(f"   - Row Count: {len(df_msg)}")
            print("   - Sample Row (First 1):")
            print(df_msg.iloc[0].to_dict())
        except Exception as e:
            print(f"   ❌ Error reading parquet: {e}")
    else:
        print(f"❌ Messages file missing at: {MSG_PARQUET}")

    print("-" * 30)

    # 2. CHECK SESSIONS
    if os.path.exists(SESS_PARQUET):
        print(f"📂 Found Sessions File: {SESS_PARQUET}")
        try:
            df_sess = pd.read_parquet(SESS_PARQUET)
            print(f"   - Columns: {list(df_sess.columns)}")
            print(f"   - Row Count: {len(df_sess)}")
        except Exception as e:
            print(f"   ❌ Error reading parquet: {e}")
    else:
        print(f"❌ Sessions file missing at: {SESS_PARQUET}")

if __name__ == "__main__":
    inspect()