"""
customer_check.py
=================
Diagnostic: validates staff performance session count against total session count.

Compares the number of unique contacts in fact_sessions_enriched.csv vs
fact_staff_performance*.csv. The difference represents bot/ghost sessions
not attributable to a human agent.

Inputs:
    data/03_processed/fact_sessions_enriched.csv
    data/03_processed/fact_staff_performance_v6_granular.csv
Output: console report with contact counts and difference breakdown

Run manually to verify staff performance completeness.
"""

import pandas as pd
from Portal_ML_V4.src.config.settings import PROCESSED_DATA_DIR

# Load the raw source
df_raw = pd.read_csv(PROCESSED_DATA_DIR / "fact_sessions_enriched.csv")
total_raw = df_raw['Contact ID'].nunique()

# Load the staff report
df_staff = pd.read_csv(PROCESSED_DATA_DIR / "fact_staff_performance_v6_granular.csv")
total_staff = df_staff['Contact ID'].nunique()

print(f"1. Total Traffic (fact_sessions): {total_raw}")
print(f"2. Human Traffic (staff_performance): {total_staff}")
print(f"3. Difference (Bots/Ghosts): {total_raw - total_staff}")

# --- PROOF: See what was dropped ---
# Filter raw data for System/Bots to see if that equals the difference
ghosts = df_raw[df_raw['full_context'].str.startswith('{', na=False)]
bots = df_raw[df_raw['active_staff'].isin(['System', 'Bot', 'Workflow'])]

print(f"   - Ghost Sessions Dropped: {len(ghosts)}")
print(f"   - Bot Sessions Dropped: {len(bots)}")