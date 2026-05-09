import pandas as pd
import os
import re
from pathlib import Path

# ✅ V3 CONFIGURATION
try:
    from Portal_ML_V4.src.config.settings import PROCESSED_DATA_DIR
except ImportError:
    print("⚠️ Using manual fallback path...")
    PROCESSED_DATA_DIR = Path(r"D:\Portal_Analytics_HQ\processed_data")

INPUT_FILE = PROCESSED_DATA_DIR / "pos_data" / "all_locations_sales_FULL_HISTORY.csv"
OUTPUT_FILE = PROCESSED_DATA_DIR / "crm" /"customer_pos_lifetime_report_updateds.xlsx"
DISCREPANCY_FILE = PROCESSED_DATA_DIR / "crm" / "pos_cashier_discrepancies.csv"

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def clean_phone_number(val):
    if pd.isna(val): return None
    s = str(val).strip()
    if s.endswith('.0'): s = s[:-2]
    s = ''.join(filter(str.isdigit, s))
    if len(s) < 8: return None 
    if s.startswith('254'): s = '0' + s[3:]
    if len(s) == 9: s = '0' + s
    return f"'{s}"

# ==========================================
# MAIN EXPORT LOGIC
# ==========================================
def run_pos_excel_export():
    print("-" * 60)
    print("🚀 GENERATING POS CUSTOMER REPORT & SMART AUDIT")
    print(f"📂 Reading: {INPUT_FILE}")
    print("-" * 60)

    if not INPUT_FILE.exists():
        print(f"❌ Error: Sales history file not found at {INPUT_FILE}")
        return

    # 1. LOAD DATA
    print("📖 Loading Sales Data...")
    cols_needed = [
        'Transaction ID', 'Phone Number', 'Client Name', 
        'Total (Tax Ex)', 'Date Sold', 'Location', 'Item', 'Amount'
    ]
    try:
        df = pd.read_csv(INPUT_FILE, usecols=lambda c: c in cols_needed, low_memory=False)
    except ValueError:
        df = pd.read_csv(INPUT_FILE, low_memory=False)

    # 2. CLEANING
    print("🧹 Cleaning Data & Formatting Phones...")
    df['Phone Number'] = df['Phone Number'].apply(clean_phone_number)
    df = df.dropna(subset=['Phone Number'])
    
    df['Total (Tax Ex)'] = pd.to_numeric(df['Total (Tax Ex)'], errors='coerce').fillna(0)
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce').fillna(0)
    df['Date Sold'] = df['Date Sold'].astype(str)

    # STRICT DEDUPLICATION
    before_rows = len(df)
    df = df.drop_duplicates(subset=['Transaction ID', 'Item'], keep='first')
    after_rows = len(df)
    print(f"   ✂️ Dropped {before_rows - after_rows:,} ghost duplicates.")

    # 3. STAGE 1: COMPRESS TO TRANSACTION LEVEL (Named Aggregation)
    print("🔄 Stage 1: Compressing Line Items to Transactions...")
    
    # We grab BOTH the Sum and the Max to outsmart the POS formatting glitch
    txn_df = df.groupby('Transaction ID').agg(
        Phone_Number=('Phone Number', 'first'),
        Client_Name=('Client Name', 'first'),
        Location=('Location', 'first'),
        Date_Sold=('Date Sold', 'max'),
        POS_Sum=('Total (Tax Ex)', 'sum'),
        POS_Max=('Total (Tax Ex)', 'max'),
        Amount=('Amount', 'first')
    ).reset_index()
    
    # Rename back to standard names
    txn_df.rename(columns={'Phone_Number': 'Phone Number', 'Client_Name': 'Client Name', 'Date_Sold': 'Date Sold'}, inplace=True)
    
    # 🚨 SMART DISCREPANCY AUDIT
    print("🔍 Auditing for True POS vs Cashier Discrepancies...")
    
    # It is ONLY a discrepancy if Cashier exists, AND it doesn't match the Sum, AND it doesn't match the Max
    discrepancy_mask = (
        (txn_df['Amount'] != 0) & 
        (abs(txn_df['POS_Sum'] - txn_df['Amount']) > 1) & 
        (abs(txn_df['POS_Max'] - txn_df['Amount']) > 1)
    )
    discrepancy_df = txn_df[discrepancy_mask].copy()
    
    if not discrepancy_df.empty:
        discrepancy_export = discrepancy_df[[
            'Date Sold', 'Location', 'Client Name', 'Transaction ID', 'POS_Sum', 'POS_Max', 'Amount'
        ]].rename(columns={
            'Date Sold': 'Date',
            'Location': 'Branch',
            'Client Name': 'Customer Name',
            'POS_Sum': 'Sales Amount (POS Sum)',
            'POS_Max': 'Sales Amount (POS Max Line Item)',
            'Amount': 'Cashier Report Amount'
        })
        
        discrepancy_export['Difference_From_Sum'] = abs(discrepancy_export['Sales Amount (POS Sum)'] - discrepancy_export['Cashier Report Amount'])
        discrepancy_export = discrepancy_export.sort_values('Difference_From_Sum', ascending=False)
        
        discrepancy_export.to_csv(DISCREPANCY_FILE, index=False)
        print(f"   ⚠️ Found {len(discrepancy_export):,} TRUE discrepancies! Saved to {DISCREPANCY_FILE.name}")
    else:
        print("   ✅ No major discrepancies found.")
        
    # SAFETY LOGIC: True Customer Lifetime Value
    def calculate_true_value(row):
        cashier_amt = row['Amount']
        pos_sum = row['POS_Sum']
        return cashier_amt if cashier_amt != 0 else pos_sum
        
    txn_df['Real_Transaction_Value'] = txn_df.apply(calculate_true_value, axis=1)

    # 4. STAGE 2: AGGREGATE BY CUSTOMER
    print("👤 Stage 2: Grouping by Customer Phone Number...")
    customer_df = txn_df.groupby('Phone Number').agg({
        'Client Name': 'first',
        'Location': lambda x: x.mode()[0] if not x.mode().empty else "Unknown", 
        'Real_Transaction_Value': 'sum',      
        'Transaction ID': 'count',            
        'Date Sold': 'max'                    
    }).reset_index()

    # 5. TIERS & FORMATTING
    def get_tier(spend):
        if spend > 20000: return "Platinum"
        if spend > 7000: return "Gold"
        return "Silver"
    
    customer_df['Lifetime Tier'] = customer_df['Real_Transaction_Value'].apply(get_tier)

    # 6. RENAME TO MATCH YOUR REQUEST
    customer_df.rename(columns={
        'Phone Number': 'Customer ID',
        'Client Name': 'Customer Name',
        'Location': 'Preferred Platform', 
        'Date Sold': 'Last Interaction Date',
        'Real_Transaction_Value': 'Total Lifetime Spend',
        'Transaction ID': 'Total Purchases'
    }, inplace=True)
    
    customer_df['Phone Number'] = customer_df['Customer ID']
    
    final_cols = [
        'Customer ID', 'Customer Name', 'Phone Number', 'Preferred Platform', 
        'Lifetime Tier', 'Last Interaction Date', 'Total Lifetime Spend', 'Total Purchases'
    ]
    customer_df = customer_df[final_cols]
    customer_df = customer_df.sort_values('Total Lifetime Spend', ascending=False)

    # 7. WRITE TO EXCEL (MULTI-SHEET)
    print(f"💾 Creating Excel Report with Tabs...")
    try:
        with pd.ExcelWriter(OUTPUT_FILE, engine='xlsxwriter') as writer:
            customer_df.to_excel(writer, sheet_name='All Customers', index=False)
            customer_df[customer_df['Lifetime Tier'] == 'Platinum'].to_excel(writer, sheet_name='Platinum', index=False)
            customer_df[customer_df['Lifetime Tier'] == 'Gold'].to_excel(writer, sheet_name='Gold', index=False)
            customer_df[customer_df['Lifetime Tier'] == 'Silver'].to_excel(writer, sheet_name='Silver', index=False)
            
        print("-" * 60)
        print(f"✅ SUCCESS: Excel Report saved at {OUTPUT_FILE}")
        print("-" * 60)
        
    except Exception as e:
        print(f"❌ Error Saving Excel: {e}")

if __name__ == "__main__":
    run_pos_excel_export()