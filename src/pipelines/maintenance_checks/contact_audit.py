import pandas as pd
from pathlib import Path

# --- CONFIGURATION ---
# Update this path to point to your raw message file (Parquet or CSV)
# If using the interim file from your pipeline:
FILE_PATH = Path("C:\\Users\\Portal Pharmacy\\Documents\\Portal ML Analys\\Ron's Work (2)\\Portal_ML\\Portal_ML_V4\\data\\01_raw\\Respond IO Messages History.csv") # <--- CHANGE THIS to your actual file
# Or if you use the processed parquet:
# FILE_PATH = Path("data/02_processed/messages_interim.parquet")

def run_audit():
    print(f"📂 Loading data from {FILE_PATH}...")
    
    # 1. Load Data
    if str(FILE_PATH).endswith('.parquet'):
        df = pd.read_parquet(FILE_PATH)
    else:
        df = pd.read_csv(FILE_PATH)
        
    # 2. Ensure Date Parsing
    # Change 'Date & Time' to your actual date column name
    date_col = 'Date & Time' 
    if date_col not in df.columns:
        print(f"❌ Could not find column '{date_col}'. Available: {df.columns.tolist()}")
        return

    df[date_col] = pd.to_datetime(df[date_col])

    # 3. Filter for JANUARY 2026
    jan_mask = (df[date_col].dt.year == 2026) & (df[date_col].dt.month == 1)
    df_jan = df[jan_mask].copy()
    
    print(f"📅 Total Rows in Jan 2026: {len(df_jan):,}")

    # 4. Filter for SENDER = CONTACT
    # Change 'Sender Type' and 'contact' if your column names differ
    contact_mask = df_jan['Sender Type'].str.lower() == 'contact'
    df_customer = df_jan[contact_mask]

    # 5. The "Ghost" Calculation
    # Count how many messages each Contact ID sent
    counts = df_customer.groupby('Contact ID').size()

    total_unique = len(counts)
    one_hit_wonders = len(counts[counts == 1])
    engaged_customers = len(counts[counts > 1])
    
    # 6. PRINT RESULTS
    print("-" * 40)
    print("📊 JANUARY 2026 AUDIT REPORT")
    print("-" * 40)
    print(f"Total Unique IDs (Raw):      {total_unique}")
    print(f"1-Message Only (Ghosts):    -{one_hit_wonders}")
    print("-" * 40)
    print(f"✅ Engaged Customers (>1 msg): {engaged_customers}")
    print("-" * 40)
    
    # Optional: Export the ghosts to check manually
    if one_hit_wonders > 0:
        ghost_ids = counts[counts == 1].index.tolist()
        pd.DataFrame(ghost_ids, columns=['Ghost_IDs']).to_csv("ghost_contacts_jan.csv", index=False)
        print("👻 List of Ghost IDs saved to 'ghost_contacts_jan.csv'")

if __name__ == "__main__":
    run_audit()
