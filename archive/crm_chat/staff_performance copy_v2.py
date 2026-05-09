import pandas as pd
import numpy as np
import re
from pathlib import Path

# ✅ IMPORT PATHS
from Portal_ML_V4.src.config.settings import (
    PROCESSED_DATA_DIR,
    INTERIM_DATA_DIR
)

# ==========================================
# 1. CONFIGURATION
# ==========================================
SESSION_DATA_PATH = PROCESSED_DATA_DIR / "fact_sessions_enriched.csv"
# 🚨 INPUT: The granular message logs (one row per message)
MESSAGES_PATH = INTERIM_DATA_DIR / "cleaned_messages.csv" 
OUTPUT_FILE = PROCESSED_DATA_DIR / "fact_staff_performance_v6_granular.csv"

# Staff Cleaning Lists
SYSTEM_NAMES = ['System', 'Bot', 'Auto Assign', 'Workflow', 'Unknown', 'nan', 'None', 'Unassigned']

# ✅ RESTORED TEAM MAP
STAFF_ID_MAP = {
    '845968': 'Joy', '847526': 'Ishmael', '860475': 'Faith', 
    '879396': 'Nimmoh', '879430': 'Rahab', '879438': 'Brenda', 
    '971945': 'Jeff', '1000558': 'Sharon', '1006108': 'Jess',
    '962460': 'Katie'
}

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def robust_extract_staff(text):
    if pd.isna(text): return None
    matches = re.findall(r"assigned to\s+([a-zA-Z0-9_]+)", str(text), re.IGNORECASE)
    if matches:
        found = matches[-1].title()
        if found in ['Me', 'You', 'Undefined'] or len(found) < 3: return None
        return found
    return None

def resolve_performance_staff(row):
    active = str(row.get('active_staff', ''))
    # Clean ID
    if active.replace('.0', '') in STAFF_ID_MAP:
        return STAFF_ID_MAP[active.replace('.0', '')]
    
    if active not in SYSTEM_NAMES and len(active) > 2 and not active.replace('.', '').isdigit():
        return active.title()
    
    fresh = row.get('fresh_extracted_staff')
    if fresh and str(fresh) not in SYSTEM_NAMES: return str(fresh).title()

    old = str(row.get('extracted_staff_name', ''))
    if old not in SYSTEM_NAMES and len(old) > 2: return old.title()
        
    return "Unassigned"

def calculate_bucket(days):
    if pd.isna(days) or days < 0: return "1. Active (≤ 24h)"
    if days <= 1: return "1. Active (≤ 24h)"
    if days <= 7: return "2. Recent (1-7 Days)"
    if days <= 30: return "3. Cold (7-30 Days)"
    if days <= 60: return "4. Legacy (30-60 Days)"
    return "5. Dormant (> 60 Days)"

# ==========================================
# 3. MAIN LOGIC
# ==========================================
def run_staff_analysis():
    print("👩‍⚕️ RUNNING STAFF PERFORMANCE V6.1 (Converted Metrics Only)...")

    # --- 1. LOAD SESSIONS ---
    print("   📥 Loading Sessions (The Truth)...")
    if not SESSION_DATA_PATH.exists():
        print("❌ Error: Session data missing.")
        return
    df_sess = pd.read_csv(SESSION_DATA_PATH)
    
    # Filter Ghosts
    def is_valid_session(row):
        ctx = str(row.get('full_context', ''))
        if ctx.strip().startswith('{') and len(ctx) < 500 and row.get('is_converted', 0) == 0:
            return False
        return True
    
    df_sess = df_sess[df_sess.apply(is_valid_session, axis=1)].copy()
    
    # Resolve Staff Names
    df_sess['fresh_extracted_staff'] = df_sess['full_context'].apply(robust_extract_staff)
    df_sess['Staff_Name'] = df_sess.apply(resolve_performance_staff, axis=1)

    # Contact Names
    if 'contact_name' not in df_sess.columns:
        df_sess['contact_name'] = "Unknown"
    
    # Dates & Revenue
    df_sess['session_start'] = pd.to_datetime(df_sess['session_start'])
    df_sess['Revenue'] = pd.to_numeric(df_sess['mpesa_amount'], errors='coerce').fillna(0)
    df_sess['Contact ID'] = df_sess['Contact ID'].astype(str).str.replace(r'\.0$', '', regex=True)
    
    # Ensure Is_Converted is int
    df_sess['Is_Converted'] = df_sess['is_converted'].fillna(0).astype(int)

    # --- 1B. BACKFILL IS_CONVERTED FROM SOCIAL SALES DIRECT ---
    print("   🔗 Backfilling conversions from social_sales_direct...")
    SOCIAL_SALES_PATH = PROCESSED_DATA_DIR / "sales_attribution" / "social_sales_direct.csv"

    if SOCIAL_SALES_PATH.exists():
        df_social = pd.read_csv(SOCIAL_SALES_PATH, usecols=["Phone Number", "Transaction ID", "Sale_Date"])
        
        # Normalize phone numbers for matching
        from Portal_ML_V4.src.utils.phone import normalize_phone
        df_social["norm_phone"] = df_social["Phone Number"].apply(normalize_phone)
        df_sess["norm_phone"]   = df_sess["phone_number"].apply(normalize_phone)
        
        # Get set of phones that made a confirmed POS purchase
        confirmed_buyers = set(df_social["norm_phone"].dropna().unique())
        
        # Mark as converted if phone appears in social_sales_direct
        df_sess["Is_Converted"] = df_sess.apply(
            lambda r: 1 if (
                r.get("is_converted", 0) == 1 or
                (str(r.get("norm_phone", "")) in confirmed_buyers and 
                str(r.get("norm_phone", "")) != "None")
            ) else 0,
            axis=1
        )
        
        original = df_sess["is_converted"].sum()
        updated  = df_sess["Is_Converted"].sum()
        print(f"   ✅ is_converted: {original} → {updated} sessions (+{updated-original} recovered)")
    else:
        df_sess["Is_Converted"] = df_sess["is_converted"].fillna(0).astype(int)
        print("   ⚠️ social_sales_direct not found — using original is_converted")

    # --- 2. LOAD MESSAGES (The Granularity) ---
    print("   💬 Loading Raw Messages (This may take a moment)...")
    if not MESSAGES_PATH.exists():
        print(f"❌ Error: Messages file missing at {MESSAGES_PATH}")
        return

    # Load only necessary columns
    msg_cols = ['Contact ID', 'Date & Time'] 
    df_msgs = pd.read_csv(MESSAGES_PATH, usecols=msg_cols)
    
    # Standardize Message Data
    df_msgs['Contact ID'] = df_msgs['Contact ID'].astype(str).str.replace(r'\.0$', '', regex=True)
    df_msgs['msg_time'] = pd.to_datetime(df_msgs['Date & Time'], dayfirst=True, errors='coerce')
    
    # Optimization: Filter messages to only those Contacts present in Sessions
    valid_contacts = set(df_sess['Contact ID'].unique())
    df_msgs = df_msgs[df_msgs['Contact ID'].isin(valid_contacts)].copy()

    print(f"   🔍 Analyzing {len(df_msgs):,} messages across {len(df_sess):,} sessions...")

    # --- 3. REHYDRATE SESSIONS (The Join) ---
    # We join Messages to Sessions on Contact ID, then filter by time window
    
    df_sess_lite = df_sess[['session_id', 'Contact ID', 'session_start']].copy()
    merged = pd.merge(df_msgs, df_sess_lite, on='Contact ID', how='inner')
    
    # 🚨 THE LOGIC: Filter Messages belonging to this 3-Day Session Window
    merged['time_diff_hours'] = (merged['msg_time'] - merged['session_start']).dt.total_seconds() / 3600
    
    # Window: -1 to 72 hours (3 days)
    session_msgs = merged[(merged['time_diff_hours'] >= -1) & (merged['time_diff_hours'] <= 72)].copy()
    
    # --- 4. CALCULATE PRECISE METRICS ---
    print("   ⏱️  Calculating Precise Durations...")
    
    metrics = session_msgs.groupby('session_id').agg(
        First_Msg=('msg_time', 'min'),
        Last_Msg=('msg_time', 'max'),
        Msg_Count=('msg_time', 'count')
    ).reset_index()
    
    # Calculate Duration (Last - First)
    metrics['Duration_Minutes'] = (metrics['Last_Msg'] - metrics['First_Msg']).dt.total_seconds() / 60
    metrics['Duration_Minutes'] = metrics['Duration_Minutes'].clip(lower=1)

    # Map back to main DF
    metric_map_dur = metrics.set_index('session_id')['Duration_Minutes'].to_dict()
    metric_map_cnt = metrics.set_index('session_id')['Msg_Count'].to_dict()
    
    # Apply Map
    df_sess['Time_To_Conversion_Mins'] = df_sess['session_id'].map(metric_map_dur).fillna(0)
    df_sess['Messages_To_Conversion'] = df_sess['session_id'].map(metric_map_cnt).fillna(0)

    # This ensures "Average" calculations ignore non-converts
    non_converted_mask = df_sess['Is_Converted'] != 1
    
    df_sess.loc[non_converted_mask, 'Time_To_Conversion_Mins'] = np.nan
    df_sess.loc[non_converted_mask, 'Messages_To_Conversion'] = np.nan

    # --- 5. FINALIZE & EXPORT ---
    
    df_sess['Reporting_Date'] = df_sess['session_start'].dt.date
    
    if 'session_days_to_convert' not in df_sess.columns: df_sess['session_days_to_convert'] = 0
    df_sess['Attribution_Bucket'] = df_sess['session_days_to_convert'].apply(calculate_bucket)
    df_sess['Customers_Handled'] = 1

    cols = [
        'Reporting_Date',
        'Staff_Name',
        'Attribution_Bucket',
        'Is_Converted',
        'Customers_Handled',
        'Revenue',
        'Time_To_Conversion_Mins',    # Only for Converted=1
        'Messages_To_Conversion',     # Only for Converted=1
        'session_id',                 
        'Contact ID'                 
    ]
    
    final_df = df_sess[cols].sort_values('Reporting_Date', ascending=False)
    
    # Clean Filter
    final_df = final_df[~final_df['Staff_Name'].isin(['System', 'Bot', 'Workflow', 'Auto Assign'])]
    
    final_df.to_csv(OUTPUT_FILE, index=False)
    
    print("-" * 60)
    print("🚀 STAFF PERFORMANCE REPORT V6.1 GENERATED")
    print(f"📂 Output: {OUTPUT_FILE}")
    print("-" * 60)
    print(final_df[['Reporting_Date', 'Staff_Name', 'Revenue', 'Time_To_Conversion_Mins', 'Messages_To_Conversion']].head(10))

if __name__ == "__main__":
    run_staff_analysis()