import pandas as pd
import os
import glob
from pathlib import Path
from Portal_ML_V4.src.config.settings import (
    MSG_HISTORY_RAW, CONV_HISTORY_RAW, CONTACTS_HISTORY_RAW,
    MSG_INTERIM_PARQUET, MSG_INTERIM_CSV,
    CONV_INTERIM_PARQUET, CONV_INTERIM_CSV,
    CONTACTS_INTERIM_PARQUET, CONTACTS_INTERIM_CSV
)

def load_and_prep_ads(ads_folder_path: Path) -> pd.DataFrame:
    all_files = []
    all_files += glob.glob(str(ads_folder_path / "contacts-added*.csv"))
    all_files += glob.glob(str(ads_folder_path / "contacts-connected*.csv"))

    if not all_files:
        print("   ⚠️ No Ad files found in /ads folder. Skipping Ad merge.")
        return pd.DataFrame()

    print(f"   📂 Found {len(all_files)} Ad files. Aggregating...")
    print("   Example ad file:", Path(all_files[0]).name)

    ad_cols_to_keep = [
        'Timestamp', 'Contact ID', 'Ad campaign ID', 'Ad group ID', 'Ad ID',
        'Source', 'Sub Source', 'is_ad_contact'
    ]

    dfs = []

    for f in all_files:
        try:
            temp = pd.read_csv(f, dtype=str, keep_default_na=False)

            # Ensure required columns exist
            if 'Timestamp' not in temp.columns or 'Contact ID' not in temp.columns:
                print(f"   ⚠️ Missing Timestamp/Contact ID in {Path(f).name}. Skipping.")
                continue

            temp['is_ad_contact'] = temp['Source'].str.strip().str.lower() == 'paid ads'

            # Clean dash placeholders
            for col in ['Ad ID', 'Ad campaign ID', 'Ad group ID']:
                if col in temp.columns:
                    temp[col] = temp[col].replace(['-', ' -', ''], pd.NA)

            # Parse Timestamp (tolerant)
            s = temp['Timestamp'].astype(str).str.strip().str.replace('T', ' ', regex=False)
            temp['Timestamp'] = pd.to_datetime(s, errors='coerce', format='mixed', cache=True)

            # Normalize Contact ID
            cid = temp['Contact ID'].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
            temp['Contact ID'] = pd.to_numeric(cid, errors='coerce').astype('Int64')

            # Debug: show drop reasons
            before = len(temp)
            bad_ts = temp['Timestamp'].isna().sum()
            bad_id = temp['Contact ID'].isna().sum()
            print(f"   {Path(f).name}: rows={before:,} bad_timestamp={bad_ts:,} bad_contact_id={bad_id:,}")

            available_cols = [c for c in ad_cols_to_keep if c in temp.columns]
            temp = temp.dropna(subset=['Contact ID', 'Timestamp'])[available_cols]

            if not temp.empty:
                dfs.append(temp)

        except Exception as e:
            print(f"   ❌ Error reading {Path(f).name}: {e}")

    if not dfs:
        return pd.DataFrame()

    df_ads = pd.concat(dfs, ignore_index=True)

    # Dedupe (prefer Ad ID if present)
    dedupe_cols = ['Contact ID', 'Timestamp']
    if 'Ad ID' in df_ads.columns:
        dedupe_cols.append('Ad ID')
    elif 'Ad campaign ID' in df_ads.columns:
        dedupe_cols.append('Ad campaign ID')

    before_dedup = len(df_ads)
    df_ads = df_ads.drop_duplicates(subset=dedupe_cols)
    print(f"   ✨ Ads Deduplicated: {before_dedup:,} -> {len(df_ads):,} rows.")

    df_ads = df_ads.sort_values('Timestamp')
    return df_ads

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
    
    # # --- 1.5 MERGE ADS DATA (The Ad Hoc Request) ---
    # ADS_DIR = Path(MSG_HISTORY_RAW).parent / "ads" 
    
    # df_ads = load_and_prep_ads(ADS_DIR)
    
    # if not df_ads.empty:
    #     print("\n Merging Ad Campaign Data")
        
    #     # Sort messages for merge_asof
    #     df_msg = df_msg.sort_values('Date & Time')
        
    #     # Perform Fuzzy Time Merge
    #     df_merged = pd.merge_asof(
    #         df_msg, 
    #         df_ads, 
    #         left_on='Date & Time', 
    #         right_on='Timestamp', 
    #         by='Contact ID', 
    #         #tolerance=pd.Timedelta('1h'), 
    #         direction='nearest', 
    #         suffixes=('', '_ad')
    #     )
        
    #     # --- AUDIT REPORTING ---
    #     total_ad_contacts = df_ads['Contact ID'].nunique()
    #     matched_contacts = df_merged[df_merged['Ad campaign ID'].notna()]['Contact ID'].nunique()
        
    #     print(f"   📊 AD MATCHING REPORT:")
    #     print(f"      - Total Contacts in Ad Files: {total_ad_contacts:,}")
    #     print(f"      - Contacts Matched to Messages: {matched_contacts:,}")
    #     print(f"      - Unmatched (Ghost Leads): {total_ad_contacts - matched_contacts:,}")
        
    #     df_msg = df_merged.drop(columns=['Timestamp'], errors='ignore')
    # else:
    #     print("   ⚠️ Skipping Ad Merge (No Data).")

    # --- 1.5 MERGE ADS DATA (The Ad Hoc Request) ---
    ADS_DIR = Path(MSG_HISTORY_RAW).parent / "ads" 
    
    df_ads = load_and_prep_ads(ADS_DIR)
    
    if not df_ads.empty:
        print("\n🔗 Merging Ad Campaign Data (Contact-Level Attribution)...")
        
        # Deduplicate to get the earliest ad interaction per contact
        # This ensures one-to-many merges don't artificially duplicate messages
        df_ads_unique = df_ads.sort_values('Timestamp').drop_duplicates(
            subset=['Contact ID'], 
            keep='first'
        )
        
        # Perform a direct left merge strictly on Contact ID
        df_merged = pd.merge(
            df_msg, 
            df_ads_unique, 
            on='Contact ID', 
            how='left', 
            suffixes=('', '_ad')
        )
        
        # --- AUDIT REPORTING ---
        total_ad_contacts = df_ads['Contact ID'].nunique()
        mask_matched = df_merged['Ad campaign ID'].notna()
        matched_contacts = df_merged[mask_matched]['Contact ID'].nunique()
        
        print("   📊 AD MATCHING REPORT:")
        print(f"      - Total Contacts in Ad Files: {total_ad_contacts:,}")
        print(f"      - Contacts Matched to Messages: {matched_contacts:,}")
        print(f"      - Unmatched (Ghost Leads): {total_ad_contacts - matched_contacts:,}")
        
        if 'Timestamp' in df_merged.columns:
            df_msg = df_merged.drop(columns=['Timestamp'])
        else:
            df_msg = df_merged
            
    else:
        print("   ⚠️ Skipping Ad Merge (No Data).")

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
    print(f"🎉 CLEANING SUCCESSFUL. All evidence in /02_interim")
    print("-" * 60)

if __name__ == "__main__":
    run_production_cleaning()