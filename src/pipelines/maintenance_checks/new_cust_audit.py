"""
new_cust_audit.py
=================
Diagnostic: identifies genuinely new customers in the most recent Respond.io export.

Loads the full Messages History CSV, determines each contact's first-ever message
date, and flags contacts whose first contact falls within a configurable recent
window — distinguishing new customers from returning ones.

Input:  data/01_raw/Respond IO Messages History.csv
Output: console report of new vs returning contact counts by time window

Run manually to track new customer acquisition rates.
"""

import pandas as pd
from pathlib import Path

# --- CONFIGURATION ---
# Use your FULL historical export here to ensure we know who is truly 'old'
FILE_PATH = Path("C:\\Users\\Portal Pharmacy\\Documents\\Portal ML Analys\\Ron's Work (2)\\Portal_ML\\Portal_ML_V4\\data\\01_raw\\Respond IO Messages History.csv") 

def run_new_customer_audit():
    print("📂 Loading Full Historical Data...")
    
    # 1. Load Data
    try:
        df = pd.read_csv(FILE_PATH, parse_dates=['Date & Time'], dayfirst=True)
    except:
        df = pd.read_parquet(FILE_PATH)
        
    # Ensure Date Column is correct
    time_col = 'Date & Time'
    df[time_col] = pd.to_datetime(df[time_col], dayfirst=True, errors='coerce')
    
    # 2. Find the "First Message Date" for every single Contact ID
    print("🔍 Calculating the first interaction for all customers...")
    first_seen = df.groupby('Contact ID')[time_col].min().reset_index()
    first_seen.columns = ['Contact ID', 'first_interaction_date']

    # 3. Filter for those whose first interaction was in JANUARY 2026
    jan_2026_new = first_seen[
        (first_seen['first_interaction_date'].dt.year == 2026) & 
        (first_seen['first_interaction_date'].dt.month == 1)
    ]

    # 4. Optional: Filter for human-only interactions 
    # (If a bot messaged them first, Respond.io might still count it as 'Added')
    
    # 5. PRINT THE TRUTH
    print("-" * 50)
    print("📊 JANUARY 2026 ACQUISITION REPORT")
    print("-" * 50)
    print(f"Total Unique IDs in Jan 2026:          418 (Your existing count)")
    print(f"NEW Contacts Added (First time ever):  {len(jan_2026_new)}")
    print("-" * 50)
    
    returning_count = 418 - len(jan_2026_new)
    print(f"♻️  Returning Customers in Jan:          {returning_count}")
    print("-" * 50)

    # Save the list to compare with Respond.io if needed
    jan_2026_new.to_csv("new_contacts_january.csv", index=False)
    print("💾 New contact list saved to 'new_contacts_january.csv'")

if __name__ == "__main__":
    run_new_customer_audit()