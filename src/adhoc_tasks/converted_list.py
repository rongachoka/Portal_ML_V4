import pandas as pd
import os
from pathlib import Path

# ✅ IMPORT PATHS
from Portal_ML_V4.src.config.settings import PROCESSED_DATA_DIR

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_FILE = PROCESSED_DATA_DIR / "fact_sessions_enriched.csv"
OUTPUT_FILE = PROCESSED_DATA_DIR / "Converted_Customers_List.csv"

def run_extraction():
    print("🚀 Starting Ad-Hoc Extraction: Converted Customer List...")

    if not INPUT_FILE.exists():
        print(f"❌ Error: Could not find {INPUT_FILE}")
        print("   👉 Please run 'run_pipeline.py' first to generate the base data.")
        return

    # 1. LOAD DATA
    df = pd.read_csv(INPUT_FILE)
    print(f"   📥 Loaded {len(df):,} total sessions.")

    # 2. FILTER: CONVERTED ONLY
    # We check if they are flagged as converted OR have revenue > 0
    df_converted = df[
        (df['is_converted'] == 1) | 
        (df['mpesa_amount'] > 0)
    ].copy()

    # 3. SELECT & RENAME COLUMNS
    # We try to get Location from 'zone_name' (extracted from tags like "Zone: Westlands")
    
    # Ensure columns exist before selecting
    available_cols = df_converted.columns.tolist()
    
    target_columns = {
        'session_start': 'Date',
        'contact_name': 'Customer Name',
        'phone_number': 'Phone Number',
        'mpesa_amount': 'Purchase Value (KES)',
        'zone_name': 'Location (Zone)',
        'primary_category': 'Product Category Interest',
        'sales_owner': 'Staff Handling'
    }

    # Only select columns that actually exist in your data
    final_cols = {k: v for k, v in target_columns.items() if k in available_cols}
    
    df_final = df_converted[list(final_cols.keys())].rename(columns=final_cols)

    # 4. FORMATTING
    # Format Date
    if 'Date' in df_final.columns:
        df_final['Date'] = pd.to_datetime(df_final['Date']).dt.strftime('%Y-%m-%d %H:%M')

    # Fill Empty Locations
    if 'Location (Zone)' in df_final.columns:
        df_final['Location (Zone)'] = df_final['Location (Zone)'].fillna("Unknown")

    # Sort by most recent
    df_final = df_final.sort_values('Date', ascending=False)

    # 5. EXPORT
    df_final.to_csv(OUTPUT_FILE, index=False)
    
    print(f"   ✅ Found {len(df_final):,} converted transactions.")
    print(f"   📂 Spreadsheet saved to: {OUTPUT_FILE}")
    print("   👉 You can send this file immediately.")

if __name__ == "__main__":
    run_extraction()