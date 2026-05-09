import pandas as pd
import os
from pathlib import Path
from Portal_ML_V4.src.config.settings import (
    MSG_HISTORY_RAW, CONV_HISTORY_RAW, CONTACTS_HISTORY_RAW,
    MSG_INTERIM_PARQUET, MSG_INTERIM_CSV,
    CONV_INTERIM_PARQUET, CONV_INTERIM_CSV,
    CONTACTS_INTERIM_PARQUET, CONTACTS_INTERIM_CSV
)


def run_production_cleaning():
    """
    V3 AUDIT-READY CLEANER + ADS MERGE
    """
    os.makedirs(MSG_INTERIM_PARQUET.parent, exist_ok=True)
    
    print("-" * 60)
    print("🚀 V3 AUDIT: DATA STANDARDIZATION, HEALING & AD MERGING")
    print("-" * 60)

    # --- 1. PROCESS MESSAGES (Source of Truth) ---
    print("📖 Processing Messages...")
    msg_list = []
    msg_cols = ['Date & Time', 'Sender Type', 'Content', 'Contact ID', 
                'Channel ID', 'Message ID']
    
    chunks = pd.read_csv(MSG_HISTORY_RAW, chunksize=50000, low_memory=False)
    for i, chunk in enumerate(chunks):
        # Standardize Types
        chunk['Date & Time'] = pd.to_datetime(chunk['Date & Time'], errors='coerce')
        chunk['Contact ID'] = pd.to_numeric(chunk['Contact ID'], errors='coerce').astype('Int64')
        chunk['Channel ID'] = pd.to_numeric(chunk['Channel ID'], errors='coerce').astype('Int64')

        if 'Message ID' in chunk.columns:
            chunk['Message ID'] = chunk['Message ID'].astype(str).str.replace(r'\.0$', '', regex=True)
        
        mask = chunk['Sender Type'].isin(['contact', 'user', 'echo', 'workflow', 'broadcast'])
        clean_chunk = chunk[mask].dropna(subset=['Contact ID'])[msg_cols].copy()
        
        msg_list.append(clean_chunk)
        print(f"   Chunk {i+1}: Kept {len(clean_chunk):,} human messages.")

    df_msg = pd.concat(msg_list)
    
    # Ad merge removed — ad attribution is handled in analytics.py
    # via load_ads_for_analytics() which reads the ads folder directly.
    # Merging here was wasted work: ml_inference.py's groupby dropped
    # all ad columns during sessionization before they reached analytics.

    # Save Messages
    df_msg.to_parquet(MSG_INTERIM_PARQUET, index=False)
    df_msg.to_csv(MSG_INTERIM_CSV, index=False)
    print(f"✅ Messages Cleaned & Enriched: {len(df_msg):,} rows.")

    # --- 2. PROCESS CONTACTS (Strict Columns) ---
    print(f"\n📖 Processing Contacts...")
    cont_cols = [
        'ContactID', 'FirstName', 'LastName', 'PhoneNumber', 'Email', 
        'Country', 'Tags', 'Status', 'Lifecycle', 'DateTimeCreated', 'Channels'
    ]
    df_cont = pd.read_csv(CONTACTS_HISTORY_RAW)
    df_cont['ContactID'] = pd.to_numeric(df_cont['ContactID'], errors='coerce').astype('Int64')
    df_cont['DateTimeCreated'] = pd.to_datetime(df_cont['DateTimeCreated'], errors='coerce')
    df_cont = df_cont.dropna(subset=['ContactID'])[cont_cols].copy()
    
    df_cont.to_parquet(CONTACTS_INTERIM_PARQUET, index=False)
    df_cont.to_csv(CONTACTS_INTERIM_CSV, index=False)
    print(f"✅ Contacts Standardized: {len(df_cont):,} records.")

    # --- 3. PROCESS CONVERSATIONS & HEAL CHANNELS ---
    print("\n📖 Processing Conversations...")
    conv_cols = [
        'Conversation ID', 'Contact ID', 'DateTime Conversation Started', 
        'DateTime Conversation Resolved', 'Opened By Source', 'Opened By Channel',
        'Assignee', 'First Assignee', 'Last Assignee', 'First Response By', 'Closed By',
        'First Response Time', 'Resolution Time', 'Average Response Time',
        'Number of Incoming Messages', 'Number of Outgoing Messages', 'Number of Responses',
        'Number of Assignments', 'Conversation Category', 'Closing Note Summary'
    ]
    df_conv = pd.read_csv(CONV_HISTORY_RAW)
    df_conv['Contact ID'] = pd.to_numeric(df_conv['Contact ID'], errors='coerce').astype('Int64')
    df_conv['Conversation ID'] = pd.to_numeric(df_conv['Conversation ID'], errors='coerce').astype('Int64')
    df_conv['DateTime Conversation Started'] = pd.to_datetime(df_conv['DateTime Conversation Started'], errors='coerce')
    df_conv['Opened By Channel'] = pd.to_numeric(df_conv['Opened By Channel'], errors='coerce').astype('Int64')
    
    # Healing Logic
    missing_mask = df_conv['Opened By Channel'].isna()
    if missing_mask.any():
        print(f"   🔍 Healing {missing_mask.sum()} blank Channels using Message History...")
        df_msg['time_key'] = df_msg['Date & Time'].dt.floor('min')
        channel_map = df_msg.drop_duplicates(['Contact ID', 'time_key']).set_index(['Contact ID', 'time_key'])['Channel ID']
        
        def heal(row):
            key = (row['Contact ID'], row['DateTime Conversation Started'].floor('min'))
            return channel_map.get(key, row['Opened By Channel'])

        df_conv.loc[missing_mask, 'Opened By Channel'] = df_conv[missing_mask].apply(heal, axis=1)

    df_conv = df_conv[df_conv['Opened By Source'] != 'workflow'][conv_cols].copy()
    
    df_conv.to_parquet(CONV_INTERIM_PARQUET, index=False)
    df_conv.to_csv(CONV_INTERIM_CSV, index=False)
    
    print("-" * 60)
    print(f"CLEANING SUCCESSFUL. All evidence in /02_interim")
    print("-" * 60)

if __name__ == "__main__":
    run_production_cleaning()