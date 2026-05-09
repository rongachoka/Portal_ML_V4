"""
extract_customers_pos_history.py
=================================
Exports a multi-sheet POS customer Excel report from the full sales history.

Reads all_locations_sales_FULL_HISTORY.csv and produces an Excel workbook
with one sheet per branch, each containing customer lifetime spend grouped
by normalised phone number.

Input:  data/03_processed/pos_data/all_locations_sales_FULL_HISTORY.csv
Output: data/03_processed/customer_pos_lifetime_report.xlsx

Entry point: run_pos_excel_export() (run manually as needed)
"""

import pandas as pd
import os
from pathlib import Path
from Portal_ML_V4.src.config.settings import PROCESSED_DATA_DIR


# INPUT: Full History CSV
INPUT_FILE = PROCESSED_DATA_DIR / "pos_data" / "all_locations_sales_FULL_HISTORY.csv"
# OUTPUT: Excel File (Required for multiple sheets)
OUTPUT_FILE = PROCESSED_DATA_DIR / "customer_pos_lifetime_report.xlsx"

def run_pos_excel_export():
    print("-" * 60)
    print("🚀 GENERATING POS CUSTOMER EXCEL REPORT (MULTI-SHEET)")
    print(f"📂 Reading: {INPUT_FILE}")
    print("-" * 60)

    if not INPUT_FILE.exists():
        print(f"❌ Error: Sales history file not found at {INPUT_FILE}")
        return

    # 1. LOAD DATA
    print("📖 Loading Sales Data...")
    cols_needed = [
        'Transaction ID', 'Phone Number', 'Client Name', 
        'Total (Tax Ex)', 'Date Sold', 'Location', 'Transaction_Total'
    ]
    try:
        df = pd.read_csv(INPUT_FILE, usecols=lambda c: c in cols_needed, low_memory=False)
    except ValueError:
        df = pd.read_csv(INPUT_FILE, low_memory=False)

    # 2. CLEANING
    print("🧹 Cleaning Data...")
    df = df.dropna(subset=['Phone Number'])
    df = df[df['Phone Number'].astype(str).str.len() > 5] 
    df['Total (Tax Ex)'] = pd.to_numeric(df['Total (Tax Ex)'], errors='coerce').fillna(0)
    df['Date Sold'] = pd.to_datetime(df['Date Sold'], errors='coerce')

    # 3. STAGE 1: COMPRESS TO TRANSACTION LEVEL (Avoid Double Counting)
    print("🔄 Stage 1: Compressing Line Items...")
    txn_df = df.groupby('Transaction ID').agg({
        'Phone Number': 'first',
        'Client Name': 'first',
        'Location': 'first',
        'Date Sold': 'max',
        'Total (Tax Ex)': 'sum' 
    }).reset_index()
    txn_df.rename(columns={'Total (Tax Ex)': 'Real_Transaction_Value'}, inplace=True)

    # 4. STAGE 2: AGGREGATE BY CUSTOMER
    print("👤 Stage 2: Grouping by Customer...")
    customer_df = txn_df.groupby('Phone Number').agg({
        'Client Name': 'first',
        'Location': lambda x: x.mode()[0] if not x.mode().empty else "Unknown", 
        'Real_Transaction_Value': 'sum',      
        'Transaction ID': 'count',            
        'Date Sold': 'max'                    
    }).reset_index()

    # 5. TIERS & FORMATTING
    # Tier Logic
    def get_tier(spend):
        if spend > 20000: return "Platinum"
        if spend > 7000: return "Gold"
        return "Silver"
    
    customer_df['Lifetime Tier'] = customer_df['Real_Transaction_Value'].apply(get_tier)
    customer_df['Date Sold'] = customer_df['Date Sold'].dt.date

    # 6. RENAME TO MATCH YOUR REQUEST EXACTLY
    # Note: 'Preferred Platform' will show the Branch Name (e.g. "GALLERIA")
    customer_df.rename(columns={
        'Phone Number': 'Customer ID',  # Using Phone as ID for POS
        'Client Name': 'Customer Name',
        # 'Phone Number' is technically the index/ID now, so we duplicate it if you need both columns
        'Location': 'Preferred Platform', 
        'Date Sold': 'Last Interaction Date',
        'Real_Transaction_Value': 'Total Lifetime Spend',
        'Transaction ID': 'Total Purchases'
    }, inplace=True)
    
    # Ensure "Phone Number" exists as a separate column if ID is used
    customer_df['Phone Number'] = customer_df['Customer ID']
    
    # Reorder columns to match your exact list
    final_cols = [
        'Customer ID', 'Customer Name', 'Phone Number', 'Preferred Platform', 
        'Lifetime Tier', 'Last Interaction Date', 'Total Lifetime Spend', 'Total Purchases'
    ]
    customer_df = customer_df[final_cols]
    
    # Sort
    customer_df = customer_df.sort_values('Total Lifetime Spend', ascending=False)

    # 7. WRITE TO EXCEL (MULTI-SHEET)
    print(f"💾 Creating Excel Report with Tabs...")
    
    try:
        with pd.ExcelWriter(OUTPUT_FILE, engine='xlsxwriter') as writer:
            # Sheet 1: Master List
            customer_df.to_excel(writer, sheet_name='All Customers', index=False)
            
            # Sheet 2: Platinum
            print("   Creating 'Platinum' tab...")
            plat_df = customer_df[customer_df['Lifetime Tier'] == 'Platinum']
            plat_df.to_excel(writer, sheet_name='Platinum', index=False)
            
            # Sheet 3: Gold
            print("   Creating 'Gold' tab...")
            gold_df = customer_df[customer_df['Lifetime Tier'] == 'Gold']
            gold_df.to_excel(writer, sheet_name='Gold', index=False)
            
            # Sheet 4: Silver
            print("   Creating 'Silver' tab...")
            silver_df = customer_df[customer_df['Lifetime Tier'] == 'Silver']
            silver_df.to_excel(writer, sheet_name='Silver', index=False)
            
        print("-" * 60)
        print(f"✅ SUCCESS: Excel Report saved at {OUTPUT_FILE}")
        print("-" * 60)
        
    except Exception as e:
        print(f"❌ Error Saving Excel: {e}")
        print("   (Do you have the file open? Close it and try again.)")

if __name__ == "__main__":
    run_pos_excel_export()