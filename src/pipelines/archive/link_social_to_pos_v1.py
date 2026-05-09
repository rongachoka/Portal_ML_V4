import pandas as pd
import numpy as np
import re
from datetime import timedelta

# ✅ IMPORT PATHS
from Portal_ML_V4.src.config.settings import (
    BASE_DIR,
    PROCESSED_DATA_DIR,
)

# ==========================================
# 1. CONFIGURATION
# ==========================================
CHAT_DATA_PATH = PROCESSED_DATA_DIR / "fact_sessions_enriched.csv"
POS_DATA_PATH = PROCESSED_DATA_DIR / "pos_data" / "fact_all_locations_sales.csv"

OUTPUT_DIR = PROCESSED_DATA_DIR / "sales_attribution"
OUTPUT_FILE = OUTPUT_DIR / "fact_social_sales_attribution.csv"

# Time window for matching (e.g., Chat must happen within 1 hour of Sale)
TIME_WINDOW_MINUTES = 60

# Regex to find money (2200, 2,200, 2.2k)
MONEY_PATTERN = r'(?:Ksh\.?|Kes\.?)?\s*(\d{1,3}(?:,\d{3})*|\d{3,})'

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def extract_potential_amounts(text):
    """
    Scans full context for ANY financial amount mentioned by Customer OR Staff.
    Returns the largest valid amount found.
    """
    if pd.isna(text): return None
    
    # Remove phone numbers first (numbers starting with 07 or +254 and long)
    # This prevents identifying "0712345678" as a price of 71 million.
    clean_text = re.sub(r'(?:07|\+254)\d{8,}', '', str(text))
    
    matches = re.findall(MONEY_PATTERN, clean_text, re.IGNORECASE)
    valid_amounts = []
    
    for m in matches:
        try:
            val = float(m.replace(',', ''))
            # Filter: Reasonable price range (50 KES to 500,000 KES)
            if 50 <= val < 500000: 
                valid_amounts.append(val)
        except: continue
            
    if valid_amounts:
        # We assume the largest number discussed is the Final Total
        return max(valid_amounts)
    return None

def check_payment_confirmation(text):
    """
    Checks if staff confirmed payment, even if no MPESA code exists.
    """
    if pd.isna(text): return False
    keywords = ['payment received', 'payment well received', 'received with thanks', 'paid', 'confirmed']
    text_lower = str(text).lower()
    return any(k in text_lower for k in keywords)

# ==========================================
# 3. MAIN PIPELINE
# ==========================================
def run_attribution_pipeline():
    print("🔗 STARTING SOCIAL-TO-POS ATTRIBUTION...")

    if not CHAT_DATA_PATH.exists() or not POS_DATA_PATH.exists():
        print("❌ Error: Missing Input Files.")
        return

    # --- STEP 1: LOAD DATA (Fixed Column Names) ---
    print("   📥 Loading Datasets...")
    
    # 1. Load Chats (Using lowercase 'session_start')
    df_chats = pd.read_csv(CHAT_DATA_PATH, parse_dates=['session_start'])
    
    # 2. Load POS
    df_pos = pd.read_csv(POS_DATA_PATH)

    # --- STEP 2: PREPARE POS DATA ---
    # Construct Full DateTime for POS (Date + Time)
    print("   ⚙️  Preparing POS Dates...")
    
    # Ensure Sale_Date is datetime
    df_pos['Sale_Date'] = pd.to_datetime(df_pos['Sale_Date'], errors='coerce')
    
    def combine_pos_datetime(row):
        try:
            # Convert Time string (e.g. "14:30") to actual time components
            t = pd.to_datetime(str(row['Time']), format='%H:%M').time()
            # Combine with Sale Date
            return pd.Timestamp.combine(row['Sale_Date'].date(), t)
        except:
            return pd.NaT

    df_pos['POS_Full_DateTime'] = df_pos.apply(combine_pos_datetime, axis=1)
    
    # Drop rows where time parsing failed
    df_pos = df_pos.dropna(subset=['POS_Full_DateTime'])


    # --- STEP 3: PREPARE CHAT DATA ---
    print("   🕵️ Scanning Chats for Payment Signals...")
    
    # 1. Extract Amount from 'full_context'
    df_chats['Extracted_Amount'] = df_chats['full_context'].apply(extract_potential_amounts)
    
    # 2. Check for Staff Confirmation
    df_chats['Staff_Confirmed'] = df_chats['full_context'].apply(check_payment_confirmation)
    
    # 3. Filter: Keep sessions that have an Amount OR a Staff Confirmation
    # (If we have an amount, we can match. If we have NO amount but confirmed, we can't match easily yet, 
    # but we keep them for manual review if needed).
    # FOR NOW: We only keep rows with an Amount because we need a Key to Join.
    
    potential_sales = df_chats.dropna(subset=['Extracted_Amount']).copy()
    
    print(f"   ✅ Found {len(potential_sales)} sessions with detectable prices.")


    # --- STEP 4: THE MATCH (Date + Amount) ---
    print("   🤝 Matching Chats to Receipts...")
    
    # Join Keys
    potential_sales['Join_Date'] = potential_sales['session_start'].dt.date
    df_pos['Join_Date'] = df_pos['Sale_Date'].dt.date
    
    # INNER JOIN on Date + Amount
    # Matches: Chat mentioning "2200" <-> Cashier Receipt for "2200" on same day
    candidates = pd.merge(
        potential_sales,
        df_pos,
        left_on=['Join_Date', 'Extracted_Amount'],
        right_on=['Join_Date', 'Amount'],
        how='inner',
        suffixes=('_Chat', '_POS')
    )

    # --- STEP 5: FILTER BY TIME (The "Soft" Constraint) ---
    # Calculate difference between Chat Start and POS Print Time
    candidates['Time_Diff_Minutes'] = (
        candidates['POS_Full_DateTime'] - candidates['session_start']
    ).abs().dt.total_seconds() / 60.0
    
    # Keep matches within window
    matches = candidates[candidates['Time_Diff_Minutes'] <= TIME_WINDOW_MINUTES].copy()


    # --- STEP 6: FORMAT OUTPUT (Your Wishlist Columns) ---
    print("   📝 Formatting Final Report...")
    
    cols_to_keep = [
        # --- FROM POS (The Hard Facts) ---
        'Transaction ID',
        'Receipt Txn No',
        'Location',
        'Sale_Date',
        'Time',
        'Department',
        'Category',
        'Item',              # Product Code
        'Description',       # Product Name
        'Total (Tax Ex)',    # Unit Price
        'Amount',            # Receipt Total
        'Client Name',       # Name on Receipt
        'Phone Number',      # Phone on Receipt
        
        # --- FROM CHAT (The Context) ---
        'session_id',
        'contact_name',      # Customer Name (Social)
        'session_start',     # Chat Time
        'Extracted_Amount',
        'Staff_Confirmed',   # Did staff say "Received"?
        'full_context',      # The Chat Log
        'channel_name',      # e.g. WhatsApp
        'active_staff',
        
        # --- VALIDATION ---
        'Time_Diff_Minutes'
    ]
    
    # Select available columns
    available = [c for c in cols_to_keep if c in matches.columns]
    final_df = matches[available]
    
    # Sort by Time Diff (Best matches first)
    final_df = final_df.sort_values('Time_Diff_Minutes')

    # Export
    final_df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"🚀 SUCCESS: Linked {len(final_df)} Chat Sessions to Sales.")
    print(f"   📂 Output: {OUTPUT_FILE}")
    
    # Preview
    if not final_df.empty:
        print("\n🔎 Match Preview:")
        preview_cols = ['Location', 'Amount', 'Description', 'contact_name', 'Time_Diff_Minutes']
        print(final_df[[c for c in preview_cols if c in final_df.columns]].head())
    else:
        print("\n⚠️  No matches found. Check date formats or widen time window.")

if __name__ == "__main__":
    run_attribution_pipeline()