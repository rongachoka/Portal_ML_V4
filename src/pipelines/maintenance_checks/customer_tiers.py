"""
customer_tiers.py
=================
Diagnostic: plots customer spend distribution from the RFV (Recency-Frequency-Value) table.

Loads dim_customers_rfv.csv and renders a scatter plot of total spend per customer
with tier threshold lines (20k and 50k KES). Useful for validating tier boundaries.

Input:  data/03_processed/dim_customers_rfv.csv
Output: matplotlib plot rendered to screen (not saved to disk)

Run manually to inspect tier distributions.
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

# Load your RFV file
file_path = "C:\\Users\\Portal Pharmacy\\Documents\\Portal ML Analys\\Ron's Work (2)\\Portal_ML\\Portal_ML_V4\\data\\03_processed\\dim_customers_rfv.csv"

if os.path.exists(file_path):
    df = pd.read_csv(file_path)
    df_sorted = df.sort_values('monetary_value', ascending=False).reset_index()

    plt.figure(figsize=(12, 6))
    plt.scatter(df_sorted.index, df_sorted['monetary_value'], alpha=0.5, c='blue')
    
    # Updated Higher Thresholds
    plt.axhline(y=20000, color='r', linestyle='--', label='20k Threshold')
    plt.axhline(y=50000, color='g', linestyle='--', label='50k Threshold')
    
    plt.title('Customer Spend Distribution (High Spenders Focus)')
    plt.ylabel('Total Spend (KES)')
    plt.xlabel('Customer Rank (High to Low)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # THE FIX: This line saves the file to your current folder
    plt.savefig('customer_spend_analysis.png')
    print("✅ Image saved as 'customer_spend_analysis.png' in your folder.")
    plt.show()
else:
    print("❌ File not found. Run the analytics pipeline first.")