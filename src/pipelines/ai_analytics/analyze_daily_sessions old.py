"""
analyze_daily_sessions_ai.py
============================
Scans daily WhatsApp conversations starting from a specific date.
Uses a local Llama 3.2 LLM to read the `full_context` and infer the
primary customer concern based on the canonical concerns list.

Features:
- State Saving: Appends row-by-row so progress is never lost.
- Smart Resume: Skips session_ids that have already been analyzed.
"""

import os
import pandas as pd
import ollama
from pathlib import Path
from tqdm import tqdm

# Import dynamic paths from config
from Portal_ML_V4.src.config.settings import RAW_DATA_DIR, PROCESSED_DATA_DIR

# ── PATHS & SETTINGS ──────────────────────────────────────────────────────────
INPUT_CSV = Path(PROCESSED_DATA_DIR) / "fact_sessions_enriched.csv"
OUTPUT_CSV = Path(PROCESSED_DATA_DIR) / "ai_daily_session_concerns.csv"
CONCERNS_LIST_PATH = Path(RAW_DATA_DIR) / "Concerns List V1(Sheet1).csv"

START_DATE = "2026-04-01"
LLM_MODEL = "llama3.2"

# ── DYNAMIC CANONICAL CONCERN LIST ────────────────────────────────────────────
def _load_dynamic_concerns() -> list[str]:
    """Loads the canonical concerns list dynamically from the boss's CSV."""
    try:
        df_concerns = pd.read_csv(CONCERNS_LIST_PATH)
        concerns = df_concerns.iloc[:, 0].dropna().astype(str).str.strip().unique().tolist()
        return [c for c in concerns if c.lower() not in ["", "nan", "none"]]
    except Exception as e:
        print(f"❌ CRITICAL ERROR: Could not load the Concerns CSV: {e}")
        return ["General Care"] 

KNOWN_CONCERNS: list[str] = _load_dynamic_concerns()

# ── LLM HELPER ────────────────────────────────────────────────────────────────
def clean_llm_response(response: str) -> str:
    """Forces the LLM output to match our exact list."""
    clean_resp = response.strip().strip("'\"*.")
    for concern in KNOWN_CONCERNS:
        # Exact match preferred
        if concern.lower() == clean_resp.lower():
            return concern
    for concern in KNOWN_CONCERNS:
        # Substring fallback
        if concern.lower() in clean_resp.lower() or clean_resp.lower() in concern.lower():
            return concern
    return "General Care"

# ── MAIN ENGINE ───────────────────────────────────────────────────────────────
def run_daily_ai_analysis():
    print("-" * 65)
    print(f"🧠 Llama 3.2 DAILY CONVERSATION ANALYZER (From {START_DATE})")
    print("-" * 65)

    if not INPUT_CSV.exists():
        print(f"❌ Error: Cannot find {INPUT_CSV.name}")
        return

    # 1. Load the main dataset
    print(f"📂 Loading {INPUT_CSV.name}...")
    # Use low_memory=False to avoid DtypeWarnings on large files
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    initial_count = len(df)

    # 2. Convert dates and filter strictly to >= April 1, 2026
    # dayfirst=True because your sample date format is DD/MM/YYYY (e.g., 16/06/2025)
    df["session_start_dt"] = pd.to_datetime(df["session_start"], dayfirst=True, errors="coerce")
    
    # Drop rows where date couldn't be parsed or full_context is empty
    df = df.dropna(subset=["session_start_dt", "full_context", "session_id"])
    
    # Filter for the target date range
    target_mask = df["session_start_dt"] >= pd.to_datetime(START_DATE)
    df_target = df[target_mask].copy()

    print(f"📅 Filtered {initial_count:,} total sessions down to {len(df_target):,} sessions since {START_DATE}.")

    if df_target.empty:
        print("✅ No new sessions found matching the date criteria.")
        return

    # 3. Check what we've already processed to allow pausing/resuming
    processed_ids = set()
    if OUTPUT_CSV.exists():
        try:
            df_existing = pd.read_csv(OUTPUT_CSV)
            processed_ids = set(df_existing["session_id"].astype(str))
            print(f"⏭️  Found {len(processed_ids):,} sessions already processed. Resuming...")
        except Exception as e:
            print(f"⚠️ Could not read existing output file: {e}")

    # 4. Filter out already processed rows
    df_to_process = df_target[~df_target["session_id"].astype(str).isin(processed_ids)].copy()

    if df_to_process.empty:
        print("✅ All sessions in the date range have already been analyzed!")
        return

    print(f"🚀 Initializing Llama 3.2 for {len(df_to_process):,} pending sessions...\n")

    # 5. Open output file in Append mode so we save after every row
    # If the file doesn't exist, write headers first
    file_exists = OUTPUT_CSV.exists()
    
    with open(OUTPUT_CSV, mode="a", encoding="utf-8") as f:
        if not file_exists:
            f.write("session_id,ai_inferred_concern,ai_reasoning\n")

        # Iterate over the rows with a progress bar
        # Iterate over the rows with a progress bar
        for _, row in tqdm(df_to_process.iterrows(), total=len(df_to_process), desc="Analyzing Chats"):
            session_id = str(row["session_id"])
            chat_text = str(row["full_context"])
            
            # ── NEW: Pull in Staff Context ──
            tags = str(row.get("final_tags", ""))
            product = str(row.get("matched_product", ""))

            # Clean up extreme lengths just in case
            if len(chat_text) > 1500:
                chat_text = chat_text[-1500:]

            # ── NEW: Highly Tuned Strict Prompt ──
            prompt = f"""You are a pharmaceutical data analyst reading a customer service chat.
            What is the primary health concern or underlying intent of the customer?
            
            Chat Transcript: "{chat_text}"
            Staff Tags: "{tags}"
            Product Mentioned: "{product}"
            
            STRICT RULES:
            1. You must reply with EXACTLY ONE category from the Allowed List below, and nothing else.
            2. If the user only asks about an image (e.g. "Do you have this? [image]") and lists no symptoms, look at the Staff Tags and Product to determine the concern.
            3. If there is ZERO clear mention of a health condition in the Chat, Tags, or Product, you MUST reply with "General Care". 
            4. DO NOT GUESS. If it is purely logistical (stock check, delivery price) and the product doesn't give it away, use "General Care".
            
            Allowed List: {", ".join(KNOWN_CONCERNS)}
            """

            try:
                response = ollama.chat(model=LLM_MODEL, messages=[
                    {'role': 'user', 'content': prompt}
                ])
                raw_answer = response['message']['content'].replace('\n', ' ')
                mapped_concern = clean_llm_response(raw_answer)

                # Save immediately to CSV
                # Ensure no commas in the text break the CSV format
                safe_raw = raw_answer.replace(',', ';').replace('"', "'")
                f.write(f'"{session_id}","{mapped_concern}","{safe_raw}"\n')
                f.flush() # Force write to disk immediately

            except Exception as e:
                print(f"\nError connecting to Ollama on session {session_id}: {e}")
                # We break here so the user can restart without losing progress
                break

    print("\n" + "-" * 65)
    print(f"✅ DAILY AI ANALYSIS COMPLETE. Results saved to {OUTPUT_CSV.name}")
    print("-" * 65)

if __name__ == "__main__":
    run_daily_ai_analysis()