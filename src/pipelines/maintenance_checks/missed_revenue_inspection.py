"""
missed_revenue_inspection.py
============================
Diagnostic: investigates "ghost sales" — converted sessions with no matched staff.

Loads fact_sessions_enriched.csv and identifies sessions marked as converted
(is_converted=1 or mpesa_amount > 0) but missing from the staff performance report,
flagging them for investigation. Outputs a debug CSV for manual review.

Inputs:
    data/02_interim/cleaned_conversations.csv
    data/03_processed/fact_sessions_enriched.csv
Output: data/03_processed/debug_ghost_sales.csv

Run manually when ghost sale counts are unexpectedly high.
"""

import pandas as pd
from pathlib import Path
from Portal_ML_V4.src.config.settings import INTERIM_DATA_DIR, PROCESSED_DATA_DIR

# FILES
CONVERSATIONS_FILE = INTERIM_DATA_DIR / "cleaned_conversations.csv"
SALES_SOURCE_FILE = PROCESSED_DATA_DIR / "fact_sessions_enriched.csv"
OUTPUT_DEBUG_FILE = PROCESSED_DATA_DIR / "debug_ghost_sales.csv"

def normalize_id(contact_id):
    s = str(contact_id).replace('.0', '') 
    s = ''.join(filter(str.isdigit, s))
    if len(s) >= 9: return s[-9:]
    return s

def run_debug():
    print("👻 STARTING GHOST SALE INVESTIGATION...")
    
    # 1. Load Sales & Normalize
    df_sales = pd.read_csv(SALES_SOURCE_FILE)
    df_sales = df_sales[df_sales['is_converted'] == 1].copy()
    df_sales['sale_time'] = pd.to_datetime(df_sales['session_start'])
    df_sales['match_id'] = df_sales['Contact ID'].apply(normalize_id)
    
    # 2. Load Conversations & Normalize
    df_conv = pd.read_csv(CONVERSATIONS_FILE)
    df_conv['start_time'] = pd.to_datetime(df_conv['DateTime Conversation Started'], errors='coerce')
    df_conv['match_id'] = df_conv['Contact ID'].apply(normalize_id)

    # Get the earliest date in your conversation history
    min_chat_date = df_conv['start_time'].min()
    print(f"📅 Conversation History starts from: {min_chat_date}")

    # 3. Find the Ghosts
    ghost_rows = []
    
    # Get set of all known chat IDs for speed
    known_chat_ids = set(df_conv['match_id'].unique())

    for _, sale in df_sales.iterrows():
        s_id = sale['match_id']
        s_time = sale['sale_time']
        
        # Check 1: Does this ID exist AT ALL in conversations?
        id_exists = s_id in known_chat_ids
        
        # Check 2: If yes, is there a chat BEFORE the sale?
        chat_exists_before = False
        if id_exists:
            matches = df_conv[df_conv['match_id'] == s_id]
            if not matches[matches['start_time'] <= s_time].empty:
                chat_exists_before = True
        
        # If no valid chat found, it's a Ghost
        if not chat_exists_before:
            reason = "ID Not Found in Chats" if not id_exists else "Chat Exists but AFTER Sale"
            
            ghost_rows.append({
                'Sale Date': s_time,
                'Raw Contact ID': sale['Contact ID'],
                'Normalized ID': s_id,
                'Amount': sale['mpesa_amount'],
                'Reason': reason,
                'History Start Date': min_chat_date
            })

    # 4. Export
    if ghost_rows:
        df_ghost = pd.DataFrame(ghost_rows)
        df_ghost.to_csv(OUTPUT_DEBUG_FILE, index=False)
        print(f"❌ Found {len(df_ghost)} Ghost Sales totaling KES {df_ghost['Amount'].sum():,.0f}")
        print(f"📂 Open this file to see them: {OUTPUT_DEBUG_FILE}")
        
        # Immediate Diagnosis
        print("\n🔍 DIAGNOSIS SUMMARY:")
        print(df_ghost['Reason'].value_counts())
        
        # Check if Sale Date is before Chat History
        early_sales = df_ghost[df_ghost['Sale Date'] < min_chat_date]
        if not early_sales.empty:
            print(f"\n⚠️ WARNING: {len(early_sales)} sales happened BEFORE your conversation export starts.")
            print("   -> Solution: Export older conversation history from Respond.io.")
    else:
        print("✅ No Ghost Sales found! All sales matched.")

if __name__ == "__main__":
    run_debug()