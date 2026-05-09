"""
ollama_kb_update.py
===================
Uses a local Ollama LLM to enrich the Knowledge Base with concern mappings.

For each product in the KB, queries the Ollama model (local LLM) with the
product name and category to suggest which customer concern(s) the product
addresses. Results are written back to the KB CSV for use in tag_rules.py.

Inputs:
    data/01_raw/Final_Knowledge_Base_PowerBI.csv
    data/01_raw/Concerns List V1(Sheet1).csv  — master concern taxonomy
Output: Updated KB CSV with a new Concerns column

Requires Ollama running locally (ollama serve). Run manually when expanding
the KB with new products or new concern categories.
"""

import pandas as pd
import ollama
from pathlib import Path
from tqdm import tqdm
from Portal_ML_V4.src.config.settings import RAW_DATA_DIR

# 1. Define Paths
KB_PATH = RAW_DATA_DIR / "Final_Knowledge_Base_PowerBI.csv"
CONCERNS_LIST_PATH = RAW_DATA_DIR / "Concerns List V1(Sheet1).csv"

# 2. Dynamically Load the Boss's Concerns List
def load_dynamic_concerns():
    try:
        df_concerns = pd.read_csv(CONCERNS_LIST_PATH)
        # Grabs the first column, drops blanks, strips spaces, and gets unique values
        concerns = df_concerns.iloc[:, 0].dropna().astype(str).str.strip().unique().tolist()
        return [c for c in concerns if c.lower() not in ["", "nan", "none"]]
    except Exception as e:
        print(f"❌ CRITICAL ERROR: Could not load the Concerns CSV from {CONCERNS_LIST_PATH}")
        print(f"Error details: {e}")
        return []

KNOWN_CONCERNS = load_dynamic_concerns()

def clean_llm_response(response: str) -> str:
    """Forces the LLM output to match our exact list."""
    clean_resp = response.strip().strip("'\"")
    for concern in KNOWN_CONCERNS:
        if concern.lower() in clean_resp.lower():
            return concern
    return "General Care"  # Fallback if the AI hallucinates

def run_kb_updater():
    print("-" * 65)
    print("🤖 OLLAMA KNOWLEDGE BASE UPDATER (Llama 3.2)")
    print("-" * 65)

    if not KNOWN_CONCERNS:
        print("Aborting: Known concerns list is empty.")
        return

    if not KB_PATH.exists():
        print(f"❌ Could not find KB at: {KB_PATH}")
        return

    # Load the KB
    df = pd.read_csv(KB_PATH)
    initial_rows = len(df)
    
    # Identify rows that need fixing (Blank, NaN, or "General Care")
    needs_update = df['Concerns'].isna() | df['Concerns'].astype(str).str.strip().str.lower().isin(
        ['', 'nan', 'none', 'general care', 'general']
    )
    
    df_to_fix = df[needs_update].copy()
    print(f"🔍 Found {len(df_to_fix)} out of {initial_rows} products that need AI categorization.")
    print(f"📋 Loaded {len(KNOWN_CONCERNS)} unique concerns from the boss's CSV.")
    
    if df_to_fix.empty:
        print("✅ Your Knowledge Base is fully mapped!")
        return

    # We only want to ask the AI about unique products to save time
    unique_products = df_to_fix[['Brand', 'Name']].drop_duplicates()
    print(f"🧠 Querying Llama 3.2 for {len(unique_products)} unique products...\n")

    ai_mappings = {}
    
    # Loop through and ask the local LLM
    for _, row in tqdm(unique_products.iterrows(), total=len(unique_products), desc="Asking Ollama"):
        brand = str(row['Brand']).strip()
        name = str(row['Name']).strip()
        
        # The strict prompt keeps the AI from being chatty
        prompt = f"""You are a pharmaceutical database assistant. 
        What health concern does the product "{brand} {name}" treat?
        You must reply with EXACTLY ONE category from this list, and absolutely nothing else:
        {", ".join(KNOWN_CONCERNS)}
        """
        
        try:
            response = ollama.chat(model='llama3.2', messages=[
                {'role': 'user', 'content': prompt}
            ])
            raw_answer = response['message']['content']
            ai_mappings[f"{brand}_{name}"] = clean_llm_response(raw_answer)
        except Exception as e:
            print(f"\n Error connecting to Ollama: {e}")
            return

    # Apply the AI's answers back to the main dataframe
    print("\n💾 Applying AI insights and saving to CSV...")
    def map_back(row):
        if needs_update[row.name]:
            key = f"{str(row['Brand']).strip()}_{str(row['Name']).strip()}"
            return ai_mappings.get(key, "General Care")
        return row['Concerns']
        
    df['Concerns'] = df.apply(map_back, axis=1)
    
    # Save it back
    df.to_csv(KB_PATH, index=False)
    
    print("-" * 65)
    print(f"✅ KNOWLEDGE BASE UPDATED SUCCESSFULLY")
    print("-" * 65)

if __name__ == "__main__":
    run_kb_updater()