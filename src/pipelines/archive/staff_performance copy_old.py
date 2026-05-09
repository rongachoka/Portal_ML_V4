import pandas as pd
import numpy as np
from pathlib import Path

# ✅ IMPORT PATHS
from Portal_ML_V4.src.config.settings import (
    BASE_DIR,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
)

# ==========================================
# 1. CONFIGURATION
# ==========================================
CONVERSATIONS_FILE = INTERIM_DATA_DIR / "cleaned_conversations.csv"
SALES_SOURCE_FILE = PROCESSED_DATA_DIR / "fact_sessions_enriched.csv"
OUTPUT_FILE = PROCESSED_DATA_DIR / "report_staff_performance_detailed_v9.csv"

STAFF_ID_MAP = {
    '845968': 'Joy', '847526': 'Ishmael', '860475': 'Faith', 
    '879396': 'Nimmoh', '879430': 'Rahab', '879438': 'Brenda', 
    '971945': 'Jeffery', '1000558': 'Sharon', '1006108': 'Jess',
    '962460' : 'Katie'
}

FORWARD_GRACE_PERIOD_HOURS = 24 
EXCLUDED_STAFF = ['System', 'Bot', 'Auto Assign', 'nan', 'None']

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def normalize_id(contact_id):
    s = str(contact_id).replace('.0', '') 
    s = ''.join(filter(str.isdigit, s))
    if len(s) >= 9:
        return s[-9:]
    return s

# ==========================================
# 3. MAIN LOGIC
# ==========================================
def run_staff_analysis():
    print("🕵️  STARTING STAFF PERFORMANCE ANALYSIS (V9.0 - Transaction Grouping)...")

    if not CONVERSATIONS_FILE.exists() or not SALES_SOURCE_FILE.exists():
        print(f"❌ Missing input files.")
        return

    # --- LOAD CONVERSATIONS ---
    print("   📥 Loading Conversations...")
    df_conv = pd.read_csv(CONVERSATIONS_FILE)
    df_conv['start_time'] = pd.to_datetime(df_conv['DateTime Conversation Started'], errors='coerce')
    df_conv['Reporting_Date'] = df_conv['start_time'].dt.date
    df_conv['match_id'] = df_conv['Contact ID'].apply(normalize_id)

    # --- STAFF IDENTIFICATION ---
    print("   👤 Resolving Staff Identities...")
    def find_human(row):
        candidates = [str(row['Last Assignee']), str(row['Assignee']), str(row['First Response By']), str(row['First Assignee'])]
        for cand in candidates:
            cand_clean = cand.replace('.0', '')
            if cand_clean in STAFF_ID_MAP: return STAFF_ID_MAP[cand_clean]
            if cand_clean not in ['nan', 'None', 'System', 'Bot', 'Auto Assign'] and len(cand_clean) > 2: return cand_clean
        return "Unassigned"

    df_conv['Staff_Name'] = df_conv.apply(find_human, axis=1)

    # --- LOAD & AGGREGATE SALES ---
    print("   💰 Loading and Aggregating Sales...")
    df_sales = pd.read_csv(SALES_SOURCE_FILE)
    
    # Filter Converted Only
    df_sales = df_sales[df_sales['is_converted'] == 1].copy()

    # Create Grouping Keys
    df_sales['sale_time'] = pd.to_datetime(df_sales['session_start']) 
    df_sales['match_id'] = df_sales['Contact ID'].apply(normalize_id)
    
    # Round time to nearest minute to catch duplicate rows with slight drift
    df_sales['sale_time_minute'] = df_sales['sale_time'].dt.round('min')

    # 🚨 THE FIX: GROUP BY TRANSACTION 🚨
    # If the same person, same time (minute), and same amount appears twice, it is ONE sale.
    unique_sales = df_sales.groupby(['match_id', 'sale_time_minute', 'mpesa_amount']).first().reset_index()

    # Restore the exact sale time from the first row found
    unique_sales['sale_time'] = unique_sales['session_start'] # Assuming first row kept has correct time
    unique_sales['sale_time'] = pd.to_datetime(unique_sales['sale_time'])

    total_loaded_revenue = unique_sales['mpesa_amount'].sum()
    print(f"   📊 VALIDATION CHECK: Total Unique Revenue to Attribute: {total_loaded_revenue:,.0f}")
    
    # Sort for processing
    sales_log = unique_sales[['match_id', 'mpesa_amount', 'sale_time']].dropna().sort_values('sale_time')
    df_conv = df_conv.sort_values('start_time')

    # --- ATTRIBUTION LOOP ---
    print("   🔗 Linking Sales...")
    
    df_conv['Revenue Generated'] = 0.0
    df_conv['Is Converted'] = 0

    matched_count = 0
    total_rev = 0
    
    for _, sale in sales_log.iterrows():
        s_id = sale['match_id']
        s_time = sale['sale_time']
        s_amount = sale['mpesa_amount']
        
        candidates = df_conv[
            (df_conv['match_id'] == s_id) & 
            (df_conv['start_time'] <= s_time + pd.Timedelta(hours=FORWARD_GRACE_PERIOD_HOURS))
        ]
        
        if not candidates.empty:
            winner_idx = candidates['start_time'].idxmax()
            
            df_conv.at[winner_idx, 'Revenue Generated'] += s_amount
            df_conv.at[winner_idx, 'Is Converted'] = 1
            df_conv.at[winner_idx, 'Reporting_Date'] = s_time.date()
            
            matched_count += 1
            total_rev += s_amount

    print(f"   ✅ Linked {matched_count} sales.")
    print(f"   💵 Total Revenue Attributed: {total_rev:,.0f} (Should not exceed {total_loaded_revenue:,.0f})")

    # --- METRICS ---
    df_conv['Number of Incoming Messages'] = df_conv['Number of Incoming Messages'].fillna(0)
    df_conv['Number of Outgoing Messages'] = df_conv['Number of Outgoing Messages'].fillna(0)
    df_conv['Messages Handled'] = df_conv['Number of Incoming Messages'] + df_conv['Number of Outgoing Messages']
    df_conv['Number of Responses'] = df_conv['Number of Responses'].fillna(0)
    df_conv['Messages to Conversion'] = np.where(df_conv['Is Converted'] == 1, df_conv['Messages Handled'], np.nan)

    # --- EXPORT ---
    final_cols = [
        'Conversation ID',
        'Reporting_Date',   
        'Staff_Name',
        'Messages Handled',            
        'Number of Responses',
        'Messages to Conversion',
        'Revenue Generated',           
        'Is Converted'                 
    ]
    
    export_df = df_conv[final_cols].rename(columns={'Staff_Name': 'Staff Name'})
    export_df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Final Staff Report saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    run_staff_analysis()