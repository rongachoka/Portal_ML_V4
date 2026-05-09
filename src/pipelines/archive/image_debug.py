import pandas as pd
import os
import re
from pathlib import Path

# --- 1. HARDCODED PATH (From your previous message) ---
# We use the raw string r"..." to handle backslashes correctly
RAW_MSG_PATH = r"C:\\Users\\Portal Pharmacy\\Documents\\Portal ML Analys\\Ron's Work (2)\\Portal_ML\\Portal_ML_V4\\data\\02_interim\\cleaned_messages.parquet"
SESSION_PATH = r"C:\\Users\\Portal Pharmacy\\Documents\\Portal ML Analys\\Ron's Work (2)\\Portal_ML\\Portal_ML_V4\\data\\03_processed\\final_tagged_sessions.parquet"

def extract_url_debug(text):
    if not isinstance(text, str): return None
    match = re.search(r'"url"\s*:\s*"(https?://[^"]+)"', text)
    if match:
        url = match.group(1)
        if "/stickers/" in url: return None
        if any(ext in url.lower() for ext in ['.jpg', '.png', '.jpeg']):
            return url
    return None

def run_debug():
    print("🕵️ STARTING DIAGNOSTIC...\n")

    # --- CHECK MESSAGE FILE ---
    if os.path.exists(RAW_MSG_PATH):
        print(f"✅ Found Message File: {RAW_MSG_PATH}")
        df_msg = pd.read_parquet(RAW_MSG_PATH)
        print(f"   - Columns: {list(df_msg.columns)}")
        print(f"   - Row Count: {len(df_msg)}")
        
        # Check Content
        # We try to find the content column
        content_col = next((c for c in ['message_content', 'content', 'body', 'text'] if c in df_msg.columns), None)
        if content_col:
            print(f"   - Using Content Column: '{content_col}'")
            
            # Test Extraction
            df_msg['debug_url'] = df_msg[content_col].astype(str).apply(extract_url_debug)
            valid_imgs = df_msg[df_msg['debug_url'].notna()]
            print(f"   - 📸 Found {len(valid_imgs)} valid images in raw messages.")
            
            if len(valid_imgs) > 0:
                print(f"   - Sample URL: {valid_imgs.iloc[0]['debug_url']}")
                print(f"   - IDs in Image Rows: {valid_imgs[['session_id', 'contact_id', 'Contact ID'] if 'Contact ID' in df_msg.columns else df_msg.columns].head(1).to_dict()}")
            else:
                print("   ❌ Extraction failed. Regex did not match any rows.")
                print(f"   - Sample Raw Content: {df_msg[content_col].iloc[0][:100]}")
        else:
            print("   ❌ Could not find a content column (message_content/content/body).")

    else:
        print(f"❌ Message File NOT found at: {RAW_MSG_PATH}")

    print("\n" + "="*30 + "\n")

    # --- CHECK SESSION FILE ---
    if os.path.exists(SESSION_PATH):
        print(f"✅ Found Session File: {SESSION_PATH}")
        df_sess = pd.read_parquet(SESSION_PATH)
        print(f"   - Columns: {list(df_sess.columns)}")
        
        # CHECK FOR COMMON IDs
        if 'df_msg' in locals() and len(valid_imgs) > 0:
            print("\n🤝 CHECKING MERGE COMPATIBILITY:")
            
            # Check ID overlap
            msg_cols = set(df_msg.columns)
            sess_cols = set(df_sess.columns)
            common_cols = msg_cols.intersection(sess_cols)
            print(f"   - Common Columns: {common_cols}")
            
            if 'session_id' in common_cols:
                overlap = df_msg['session_id'].isin(df_sess['session_id']).sum()
                print(f"   - 'session_id' Matches: {overlap} rows match.")
            
            if 'Contact ID' in sess_cols:
                # Check against likely message contact columns
                for c in ['contact_id', 'Contact ID', 'phone']:
                    if c in df_msg.columns:
                        # Try converting to string for safety
                        s_ids = df_sess['Contact ID'].astype(str)
                        m_ids = df_msg[c].astype(str)
                        overlap = m_ids.isin(s_ids).sum()
                        print(f"   - '{c}' (Msg) vs 'Contact ID' (Sess) Matches: {overlap}")

    else:
        print(f"❌ Session File NOT found at: {SESSION_PATH}")

if __name__ == "__main__":
    run_debug()