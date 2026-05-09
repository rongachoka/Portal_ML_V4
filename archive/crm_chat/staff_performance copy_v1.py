import pandas as pd
import numpy as np
import re
from pathlib import Path

# ✅ IMPORT PATHS
from Portal_ML_V4.src.config.settings import (
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
)

# ==========================================
# 1. CONFIGURATION
# ==========================================
CONVERSATIONS_FILE = INTERIM_DATA_DIR / "cleaned_conversations.csv"
SALES_SOURCE_FILE = PROCESSED_DATA_DIR / "fact_sessions_enriched.csv"
OUTPUT_FILE = PROCESSED_DATA_DIR / "report_staff_performance_v9.csv"

# ✅ TEAM MAP (Same as Analytics.py)
STAFF_ID_MAP = {
    '845968': 'Joy', '847526': 'Ishmael', '860475': 'Faith', 
    '879396': 'Nimmoh', '879430': 'Rahab', '879438': 'Brenda', 
    '971945': 'Jeff', '1000558': 'Sharon', '1006108': 'Jess',
    '962460' : 'Katie'
}

EXCLUDED_STAFF = ['System', 'Bot', 'Auto Assign', 'nan', 'None']

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

# ✅ NEW: Text Extraction Logic (Matches Analytics.py)
def extract_assigned_staff(text):
    if pd.isna(text): return None
    # Pattern: "assigned to [Name]"
    # Grabs the last occurrence in the session
    matches = re.findall(r"assigned to\s+([a-zA-Z]+)", str(text), re.IGNORECASE)
    
    if matches:
        found = matches[-1].title() # Take the most recent assignment
        # Filter system keywords
        if found.lower() in ['workflow', 'bot', 'system', 'me', 'you', 'undefined']: 
            return None
        return found
    return None

def normalize_id(contact_id):
    s = str(contact_id).replace('.0', '') 
    s = ''.join(filter(str.isdigit, s))
    if len(s) >= 9:
        return s[-9:]
    return s

def resolve_staff_closer_logic(row):
    """
    Priority: 1. Last Assignee, 2. Assignee, 3. First Response By
    """
    def clean(val):
        s = str(val).replace('.0', '').strip()
        return s if s not in ['nan', 'None', '', 'System', 'Bot'] else None

    candidates = [
        clean(row.get('Last Assignee')),
        clean(row.get('Assignee')),
        clean(row.get('First Response By')),
        clean(row.get('First Assignee'))
    ]

    for cand in candidates:
        if not cand: continue
        if cand in STAFF_ID_MAP: return STAFF_ID_MAP[cand]
        if cand not in EXCLUDED_STAFF and len(cand) > 2: return cand

    return "Unassigned"

# ==========================================
# 3. MAIN LOGIC
# ==========================================
def run_staff_analysis():
    print("🕵️  STARTING STAFF ANALYSIS (V15 - Text Fallback Integration)...")

    if not CONVERSATIONS_FILE.exists() or not SALES_SOURCE_FILE.exists():
        print(f"❌ Missing input files.")
        return

    # --- LOAD CONVERSATIONS ---
    print("   📥 Loading Conversations...")
    df_conv = pd.read_csv(CONVERSATIONS_FILE)
    df_conv['start_time'] = pd.to_datetime(df_conv['DateTime Conversation Started'], errors='coerce')
    df_conv['match_id'] = df_conv['Contact ID'].apply(normalize_id)
    # This gets the metadata-based staff name
    df_conv['Staff_Metadata'] = df_conv.apply(resolve_staff_closer_logic, axis=1)

    # --- LOAD SALES ---
    print("   💰 Loading Sales...")
    df_sales = pd.read_csv(SALES_SOURCE_FILE)
    
    # Filter purely on revenue > 0.
    df_sales = df_sales[df_sales['mpesa_amount'] > 0].copy()

    df_sales['sale_time'] = pd.to_datetime(df_sales['session_start']) 
    df_sales['match_id'] = df_sales['Contact ID'].apply(normalize_id)

    # ✅ APPLY TEXT PARSING TO SALES SOURCE
    if 'full_context' in df_sales.columns:
        print("   📝 Extracting 'Assigned To' names from full context...")
        df_sales['extracted_staff'] = df_sales['full_context'].apply(extract_assigned_staff)
    else:
        df_sales['extracted_staff'] = None

    total_loaded_revenue = df_sales['mpesa_amount'].sum()
    print(f"   📊 TOTAL SOURCE REVENUE: {total_loaded_revenue:,.0f}")
    
    # --- ATTRIBUTION LOOP ---
    print("   🔗 Linking Sales...")
    
    df_conv = df_conv.sort_values('start_time')
    sales_results = []

    for _, sale in df_sales.iterrows():
        s_id = sale['match_id']
        s_time = sale['sale_time']
        s_amount = sale['mpesa_amount']
        s_session_id = sale.get('session_id', 'Unknown')
        s_extracted = sale.get('extracted_staff') # The text-based fallback
        
        # Look for chat that started *before* or *during* the sale (1 hour buffer)
        candidates = df_conv[
            (df_conv['match_id'] == s_id) & 
            (df_conv['start_time'] <= s_time + pd.Timedelta(hours=1))
        ]
        
        if not candidates.empty:
            winner = candidates.iloc[-1]
            last_chat_time = winner['start_time']
            metadata_staff = winner['Staff_Metadata']
            
            # 🚨 HYBRID LOGIC:
            # 1. Use Metadata (if valid)
            # 2. Use Text Extraction (if Metadata failed)
            final_staff = metadata_staff
            if final_staff in ["Unassigned", "System", "Bot", None]:
                if s_extracted: 
                    final_staff = s_extracted # Fallback used!

            time_diff = s_time - last_chat_time
            days_diff = time_diff.total_seconds() / (3600 * 24)
            
            if days_diff <= 1: bucket = "1. Active (≤ 24h)"
            elif days_diff <= 7: bucket = "2. Recent (1-7 Days)"
            elif days_diff <= 30: bucket = "3. Cold (7-30 Days)"
            elif days_diff <= 60: bucket = "4. Legacy (30-60 Days)"
            else: bucket = "5. Dormant (> 60 Days)"
            
            sales_results.append({
                'Session ID': s_session_id,
                'Conversation ID': winner['Conversation ID'],
                'Staff_Name': final_staff,
                'Sale_Amount': s_amount,
                'Sale_Date': s_time,
                'Last_Chat_Date': last_chat_time.date(),
                'Days_Since_Chat': round(days_diff, 1),
                'Attribution_Bucket': bucket
            })
        else:
            # GHOST SALE (No Chat Found)
            # Try to save it with Text Extraction even if no chat ID found
            ghost_staff = s_extracted if s_extracted else 'System / No Chat Found'
            
            sales_results.append({
                'Session ID': s_session_id,
                'Conversation ID': 'NO_HISTORY',
                'Staff_Name': ghost_staff,
                'Sale_Amount': s_amount,
                'Sale_Date': s_time,
                'Last_Chat_Date': np.nan,
                'Days_Since_Chat': np.nan,
                'Attribution_Bucket': "6. Ghost (No Data)"
            })

    # --- AGGREGATE RESULTS ---
    df_results = pd.DataFrame(sales_results)

    # 🚨 V14 FIX: dropna=False to keep ghosts
    rev_per_conv = df_results.groupby(
        ['Conversation ID', 'Session ID', 'Attribution_Bucket', 'Last_Chat_Date', 'Days_Since_Chat', 'Staff_Name', 'Sale_Date'],
        dropna=False 
    )['Sale_Amount'].sum().reset_index()
    
    rev_per_conv.rename(columns={'Sale_Amount': 'Revenue Generated'}, inplace=True)

    # --- MERGE CONTEXT ---
    cols_to_merge = [
        'Conversation ID', 
        'Number of Incoming Messages', 
        'Number of Outgoing Messages', 
        'Number of Responses'
    ]
    
    context_source = df_conv[cols_to_merge].drop_duplicates(subset=['Conversation ID'])
    
    final_df = pd.merge(rev_per_conv, context_source, on='Conversation ID', how='left')
    final_df['Staff_Name'] = final_df['Staff_Name'].fillna('System / No Chat Found')
    
    final_df['Reporting_Date'] = pd.to_datetime(final_df['Sale_Date']).dt.date
    
    # --- METRICS ---
    metric_cols = ['Number of Incoming Messages', 'Number of Outgoing Messages', 'Number of Responses']
    final_df[metric_cols] = final_df[metric_cols].fillna(0)

    final_df['Messages Handled'] = final_df['Number of Incoming Messages'] + final_df['Number of Outgoing Messages']
    final_df['Is Converted'] = 1 
    final_df['Messages to Conversion'] = final_df['Messages Handled']

    # --- EXPORT ---
    export_cols = [
        'Conversation ID',
        'Reporting_Date', 
        'Last_Chat_Date', 
        'Staff_Name',
        'Attribution_Bucket',
        'Days_Since_Chat',
        'Messages Handled',
        'Number of Responses',
        'Messages to Conversion', 
        'Revenue Generated',
        'Is Converted'
    ]
    
    final_export = final_df[export_cols].copy()
    final_export = final_export.sort_values('Reporting_Date')

    # 🔍 VERIFICATION PRINT
    print("\n📅 MONTHLY REVENUE CHECK:")
    final_export['Month'] = pd.to_datetime(final_export['Reporting_Date']).dt.strftime('%Y-%m')
    print(final_export.groupby('Month')['Revenue Generated'].sum().apply(lambda x: f"{x:,.0f}"))

    total_final = final_export['Revenue Generated'].sum()
    print(f"\n   💵 Total Attributed: {total_final:,.0f}")
    
    diff = total_loaded_revenue - total_final
    if abs(diff) > 100:
        print(f"⚠️ WARNING: Lost {diff:,.0f} revenue! Check your logic.")
    else:
        print("✅ REVENUE MATCHES PERFECTLY.")

    final_export.drop(columns=['Month'], inplace=True, errors='ignore')
    final_export.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Final Staff Report saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    run_staff_analysis()