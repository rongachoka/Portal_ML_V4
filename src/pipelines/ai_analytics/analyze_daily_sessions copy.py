"""
analyze_daily_sessions_ai.py
============================
Scans daily WhatsApp conversations starting from a specific date.
Uses a local Llama 3.2 LLM to read the `full_context` and infer the
primary customer concern based on the canonical concerns list.

Features:
- State Saving: Appends row-by-row so progress is never lost.
- Smart Resume: Skips session_ids that have already been analyzed.
- Force Rerun: Wipe previous progress to re-analyze from scratch.
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

# 🛑 Set to True if you want to wipe the previous results and start completely over
FORCE_RERUN = True 

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
# def clean_llm_response(response: str) -> str:
#     """Forces the LLM output to match our exact list."""
#     clean_resp = response.strip().strip("'\"*.")
#     for concern in KNOWN_CONCERNS:
#         # Exact match preferred
#         if concern.lower() == clean_resp.lower():
#             return concern
#     for concern in KNOWN_CONCERNS:
#         # Substring fallback
#         if concern.lower() in clean_resp.lower() or clean_resp.lower() in concern.lower():
#             return concern
#     return "General Care"

def clean_llm_response(response: str) -> str:
    """Parses the LLM output to extract multiple concerns matching our exact list."""
    found_concerns = set()
    
    # Split the AI's response by commas or newlines
    parts = [p.strip().strip("'\"*.") for p in response.replace('\n', ',').split(',')]
    
    for part in parts:
        if not part:
            continue
        # Check against our canonical list
        for concern in KNOWN_CONCERNS:
            # Exact match
            if concern.lower() == part.lower():
                found_concerns.add(concern)
            # Substring match (e.g., if AI outputs "Acne and Dry Skin")
            elif concern.lower() in part.lower():
                found_concerns.add(concern)
                
    if not found_concerns or "General Care" in found_concerns:
        return "General Care"
        
    # Return alphabetically sorted, pipe-separated multiple concerns
    return " | ".join(sorted(list(found_concerns)))

# ── MAIN ENGINE ───────────────────────────────────────────────────────────────
def run_daily_ai_analysis():
    print("-" * 65)
    print(f"🧠 Llama 3.2 DAILY CONVERSATION ANALYZER (From {START_DATE})")
    print("-" * 65)

    if not INPUT_CSV.exists():
        print(f"❌ Error: Cannot find {INPUT_CSV.name}")
        return

    # Handle Force Rerun
    if FORCE_RERUN and OUTPUT_CSV.exists():
        print("⚠️  FORCE_RERUN is enabled. Wiping previous AI results to start fresh...")
        OUTPUT_CSV.unlink() # Deletes the existing file

    # 1. Load the main dataset
    print(f"📂 Loading {INPUT_CSV.name}...")
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    initial_count = len(df)

    # 2. Convert dates and filter strictly to >= April 1, 2026
    # format='mixed' silences the Pandas warning about date formats
    df["session_start_dt"] = pd.to_datetime(df["session_start"], format='mixed', dayfirst=True, errors="coerce")
    
    df = df.dropna(subset=["session_start_dt", "full_context", "session_id"])
    
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
    file_exists = OUTPUT_CSV.exists()
    
    with open(OUTPUT_CSV, mode="a", encoding="utf-8") as f:
        if not file_exists:
            f.write("session_id,ai_inferred_concern,ai_reasoning\n")

        for _, row in tqdm(df_to_process.iterrows(), total=len(df_to_process), desc="Analyzing Chats"):
            session_id = str(row["session_id"])
            chat_text = str(row["full_context"])
            
            # Pull in Staff Context to prevent hallucination on Image-only queries
            tags = str(row.get("final_tags", ""))
            product = str(row.get("matched_product", ""))

            if len(chat_text) > 1500:
                chat_text = chat_text[-1500:]

            prompt = f"""You are a pharmaceutical data expert categorizing customer service chats.
            Identify ALL health concerns or intents requested by the customer.
            
            Chat Transcript: "{chat_text}"
            Staff Tags: "{tags}"
            Product Mentioned: "{product}"
            
            STRICT RULES:
            1. MULTIPLE CONCERNS: You may return multiple concerns separated by commas if the product treats multiple issues (e.g. Azelaic Acid).
            2. ALLOWED LIST ONLY: You must ONLY use categories from the Allowed List below. Do not invent new categories.
            3. LOGISTICAL / BLANK: If the chat is just "Can I make an order?", "Hello", or asking for delivery prices with no specific product/symptom, you MUST reply with ONLY "General Care". Do not guess.
            
            EXAMPLES OF CORRECT BEHAVIOR:
            Chat: "Can I make an order? Yes of course!" -> General Care
            Chat: "Do you have CeraVe SA cleanser?" -> Acne
            Chat: "Hi how much for the anua azelaic acid serum?" -> Acne, Hyperpigmentation, Sensitive Skin, Redness
            
            Allowed List: {", ".join(KNOWN_CONCERNS)}
            
            OUTPUT FORMAT: Reply strictly with the comma-separated categories.
            """

            try:
                response = ollama.chat(model=LLM_MODEL, messages=[
                    {'role': 'user', 'content': prompt}
                ])
                raw_answer = response['message']['content'].replace('\n', ' ')
                mapped_concern = clean_llm_response(raw_answer)

                safe_raw = raw_answer.replace(',', ';').replace('"', "'")
                f.write(f'"{session_id}","{mapped_concern}","{safe_raw}"\n')
                f.flush() 

            except Exception as e:
                print(f"\n Error connecting to Ollama on session {session_id}: {e}")
                break

    print("\n" + "-" * 65)
    print(f"✅ DAILY AI ANALYSIS COMPLETE. Results saved to {OUTPUT_CSV.name}")
    print("-" * 65)

if __name__ == "__main__":
    run_daily_ai_analysis()