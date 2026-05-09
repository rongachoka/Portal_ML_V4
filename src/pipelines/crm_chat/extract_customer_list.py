import pandas as pd
import numpy as np
import os
from pathlib import Path

# ✅ V3 PRODUCTION IMPORT
try:
    from Portal_ML_V4.src.config.settings import PROCESSED_DATA_DIR
except ImportError:
    print("⚠️ Could not import project settings. Using manual fallback...")
    PROCESSED_DATA_DIR = Path(r"D:\Documents\Portal ML Analys\Portal_ML\data\03_processed")

INPUT_FILE = PROCESSED_DATA_DIR / "fact_sessions_enriched.csv"
OUTPUT_FILE = PROCESSED_DATA_DIR / "customer_lifetime_value_export.csv"

def run_customer_export():
    print("-" * 60)
    print("🚀 GENERATING CUSTOMER LIFETIME EXPORT (PURCHASE COUNT)")
    print(f"📂 Looking in: {PROCESSED_DATA_DIR}")
    print("-" * 60)

    if not INPUT_FILE.exists():
        print(f"❌ CRITICAL ERROR: Input file not found at {INPUT_FILE}")
        return

    print("📖 Loading Session Data...")
    try:
        df = pd.read_csv(INPUT_FILE, low_memory=False)
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return

    # 1. ENSURE NUMERIC SPEND
    if 'mpesa_amount' in df.columns:
        df['mpesa_amount'] = pd.to_numeric(df['mpesa_amount'], errors='coerce').fillna(0)

    # 2. DEFINE "PURCHASE" (Crucial Step)
    # If 'is_converted' exists from analytics.py, use it. 
    # Otherwise, fallback to checking if money was spent (> 0).
    if 'is_converted' not in df.columns:
        print("   ⚠️ 'is_converted' column missing. Calculating from M-Pesa Amount...")
        df['is_converted'] = np.where(df['mpesa_amount'] > 0, 1, 0)
    else:
        # Ensure it's numeric so we can sum it
        df['is_converted'] = pd.to_numeric(df['is_converted'], errors='coerce').fillna(0)

    print("🔄 Aggregating by Customer...")
    
    # 3. AGGREGATION LOGIC
    customer_df = df.groupby('Contact ID').agg({
        'contact_name': 'first',              
        'phone_number': 'first',              
        'channel_name': lambda x: x.mode()[0] if not x.mode().empty else "Unknown", 
        'lifetime_tier_history': 'first',      
        'session_start': 'max',               
        'mpesa_amount': 'sum',                
        'is_converted': 'sum'  # ✅ CHANGED: Summing conversions (Purchases), not counting sessions
    }).reset_index()

    # 4. RENAME COLUMNS
    customer_df.rename(columns={
        'Contact ID': 'Customer ID',
        'contact_name': 'Customer Name',
        'phone_number': 'Phone Number',
        'channel_name': 'Preferred Platform',
        'lifetime_tier_history': 'Lifetime Tier', 
        'session_start': 'Last Interaction Date',
        'mpesa_amount': 'Total Lifetime Spend',
        'is_converted': 'Total Purchases' # ✅ NEW NAME
    }, inplace=True)

    # 5. FORMATTING
    # Sort by Spend (High to Low)
    customer_df = customer_df.sort_values('Total Lifetime Spend', ascending=False)
    
    # Format Date
    if 'Last Interaction Date' in customer_df.columns:
        customer_df['Last Interaction Date'] = pd.to_datetime(customer_df['Last Interaction Date']).dt.date

    # 6. SAVE
    print(f"💾 Saving {len(customer_df):,} customers to CSV...")
    try:
        customer_df.to_csv(OUTPUT_FILE, index=False)
        print(f"✅ EXPORT COMPLETE: {OUTPUT_FILE}")
    except PermissionError:
        print(f"❌ Error: Could not save file. Is '{OUTPUT_FILE.name}' open in Excel?")

    # Preview
    print("\n🏆 Top 5 Lifetime VIPs (By Spend):")
    print(customer_df[['Customer Name', 'Total Purchases', 'Total Lifetime Spend', 'Lifetime Tier']].head(5))

if __name__ == "__main__":
    run_customer_export()