"""
staff_performance.py
====================
Generates per-agent performance metrics by cross-referencing chat sessions
with POS cashier data.

Inputs:
    data/03_processed/fact_sessions_enriched.csv  — session data with assignee
    data/02_interim/cleaned_messages.csv          — granular message log

Output:
    data/03_processed/Staff_Performance_Test.csv
    — one row per staff member, with session count, conversion rate,
      average response time, M-Pesa collections, and POS revenue cross-ref

Entry point: run_staff_analysis()
"""

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
OUTPUT_FILE = PROCESSED_DATA_DIR / "Staff_Performance_Test.csv"

# Staff Cleaning Lists
SYSTEM_NAMES = ['System', 'Bot', 'Auto Assign', 'Workflow', 'Unknown', 'nan', 'None', 'Unassigned']

# RESTORED TEAM MAP
STAFF_ID_MAP = {
    '845968': 'Joy', '847526': 'Ishmael', '860475': 'Faith', 
    '879396': 'Nimmoh', '879430': 'Rahab', '879438': 'Brenda', 
    '971945': 'Jeff', '1000558': 'Sharon', '1006108': 'Jess',
    '962460': 'Katie', '1052677': 'Vivian'
}

# Maps POS Sales Rep names → canonical Staff_Name.
# Only entries that differ from the canonical name need to be listed.
# Everyone else is used as-is (title-cased).
SALES_REP_MAP = {
    "Cate":  "Katie",
    "Emily": "Nimmoh",
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
    df_sess['Revenue']       = pd.to_numeric(df_sess['mpesa_amount'], errors='coerce').fillna(0)
    df_sess['Contact ID']    = df_sess['Contact ID'].astype(str).str.replace(r'\.0$', '', regex=True)

    # Start with original is_converted
    df_sess['Is_Converted'] = df_sess['is_converted'].fillna(0).astype(int)

    # --- 1B. BACKFILL IS_CONVERTED + SALES REP FROM SOCIAL SALES DIRECT ---
    print("   🔗 Backfilling conversions and Sales Rep from social_sales_direct...")
    SOCIAL_SALES_PATH = PROCESSED_DATA_DIR / "sales_attribution" / "social_sales_direct.csv"

    if SOCIAL_SALES_PATH.exists():
        from Portal_ML_V4.src.utils.phone import normalize_phone

        df_social = pd.read_csv(
            SOCIAL_SALES_PATH,
            usecols=["Phone Number", "Transaction ID", "Sale_Date", "Sales Rep"],
            dtype=str,
        )
        df_social["norm_phone"] = df_social["Phone Number"].apply(normalize_phone)

        # Normalize Sales Rep names using SALES_REP_MAP.
        # Title-case first, then apply explicit overrides.
        df_social["Sales Rep"] = (
            df_social["Sales Rep"]
            .fillna("")
            .str.strip()
            .str.title()
            .map(lambda n: SALES_REP_MAP.get(n, n) if n else n)
        )

        # Build (norm_phone, sale_date) -> Sales Rep lookup.
        # Keyed on both phone AND date so we only attribute a rep to the session
        # where they actually processed the sale -- no cross-date guessing.
        # We also allow a +1 day tolerance: sale could be processed the morning
        # after the WhatsApp session that converted the customer.
        df_social["sale_date_str"] = pd.to_datetime(
            df_social["Sale_Date"], errors="coerce"
        ).dt.strftime("%Y-%m-%d")

        phone_date_rep_map = {}
        for _, row in df_social[df_social["Sales Rep"].str.len() > 1].iterrows():
            phone = row["norm_phone"]
            date  = row["sale_date_str"]
            rep   = row["Sales Rep"]
            if pd.notna(phone) and pd.notna(date) and phone and date:
                # Keep first rep seen for this (phone, date) pair --
                # if there are duplicates on the same day the first is fine.
                phone_date_rep_map.setdefault((phone, date), rep)

        print(f"   📋 Sales Rep lookup built: {len(phone_date_rep_map):,} (phone, date) pairs")

        # Find phone column in sessions safely
        phone_col = next(
            (c for c in df_sess.columns if 'phone' in c.lower() and 'number' not in c.lower()),
            next((c for c in df_sess.columns if 'phone' in c.lower()), None)
        )

        if phone_col:
            df_sess["norm_phone"] = df_sess[phone_col].apply(normalize_phone)
        else:
            df_sess["norm_phone"] = None
            print("   ⚠️ No phone column found in sessions — phone-based backfill skipped")

        # Set of phones confirmed as POS buyers
        confirmed_buyers = set(df_social["norm_phone"].dropna().unique())

        original = int(df_sess["Is_Converted"].sum())

        df_sess["Is_Converted"] = df_sess.apply(
            lambda r: 1 if (
                r["Is_Converted"] == 1 or
                (r.get("norm_phone") is not None and
                 str(r.get("norm_phone")) not in ("None", "nan", "") and
                 str(r.get("norm_phone")) in confirmed_buyers)
            ) else 0,
            axis=1
        )

        updated = int(df_sess["Is_Converted"].sum())
        print(f"   ✅ is_converted: {original:,} → {updated:,} sessions (+{updated - original:,} recovered)")

        # Canonical staff names — the only Sales Rep values we trust from the POS.
        # Anything outside this set is treated as unrecognized and we fall back
        # to the session-resolved staff from fact_sessions_enriched.
        KNOWN_STAFF = set(STAFF_ID_MAP.values()) | set(SALES_REP_MAP.values())

        # Override Staff_Name with POS Sales Rep for confirmed conversions.
        # Lookup key: (norm_phone, sale_date). We try the session date first,
        # then session date + 1 day (sale processed the following morning).
        #
        # Waterfall for converted sessions:
        #   1. POS Sales Rep (phone + date match, must be in KNOWN_STAFF) → use it
        #   2. Session staff from fact_sessions_enriched (must be in KNOWN_STAFF) → use it
        #   3. Neither recognised → "Other Staff"
        #
        # Non-converted sessions are untouched — no KNOWN_STAFF enforcement applied.
        def apply_sales_rep(row):
            if row["Is_Converted"] != 1:
                return row["Staff_Name"]

            phone = str(row.get("norm_phone", ""))
            session_date = pd.Timestamp(row["session_start"])

            # Stage 1 — POS Sales Rep via phone + date lookup
            if phone not in ("None", "nan", ""):
                for delta in [0, 1]:
                    date_str = (session_date + pd.Timedelta(days=delta)).strftime("%Y-%m-%d")
                    rep = phone_date_rep_map.get((phone, date_str))
                    if rep and rep in KNOWN_STAFF:
                        return rep

            # Stage 2 — session staff from fact_sessions_enriched
            session_staff = row["Staff_Name"]
            if session_staff in KNOWN_STAFF:
                return session_staff

            # Stage 3 — neither source is a recognised team member
            return "Other Staff"

        original_staff = df_sess["Staff_Name"].copy()
        df_sess["Staff_Name"] = df_sess.apply(apply_sales_rep, axis=1)

        converted_mask = df_sess["Is_Converted"] == 1
        rep_overrides    = (df_sess.loc[converted_mask, "Staff_Name"] != original_staff[converted_mask]).sum()
        rep_session_used = (
            (df_sess.loc[converted_mask, "Staff_Name"] == original_staff[converted_mask]) &
            (df_sess.loc[converted_mask, "Staff_Name"] != "Other Staff")
        ).sum()
        rep_other_staff  = (df_sess.loc[converted_mask, "Staff_Name"] == "Other Staff").sum()

        print(f"   ✅ Staff from POS Sales Rep (known):        {rep_overrides:,} converted sessions")
        print(f"   ℹ️  Kept from fact_sessions_enriched:        {rep_session_used:,} converted sessions")
        print(f"   ⚠️  Assigned 'Other Staff' (unrecognised):  {rep_other_staff:,} converted sessions")


    else:
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