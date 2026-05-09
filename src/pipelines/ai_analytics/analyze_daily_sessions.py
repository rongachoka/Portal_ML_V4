"""
analyze_daily_sessions_ai.py
============================
Scans daily WhatsApp conversations starting from a specific date.
Uses a local LLM to read the `full_context` and infer the
primary customer concern based on the canonical concerns list.
"""

import os
import time
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
# LLM_MODEL = "gemma2:2b"
LLM_MODEL = "gemma2:2b"

FORCE_RERUN = True 

# ── DYNAMIC CANONICAL CONCERN LIST ────────────────────────────────────────────
def _load_dynamic_concerns() -> list[str]:
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
    """Parses the LLM output to extract multiple text concerns matching our exact list."""
    found_concerns = set()
    
    parts = [p.strip().strip("'\"*.:-") for p in response.replace('\n', ',').split(',')]
    
    for part in parts:
        if not part:
            continue
        part_lower = part.lower()
        for concern in KNOWN_CONCERNS:
            # Check if the concern name appears inside the LLM output part
            # e.g. LLM says "Acne Prone Skin" → matches concern "Acne"
            if concern.lower() == part_lower or concern.lower() in part_lower:
                found_concerns.add(concern)

    # If real concerns were found alongside General Care, drop General Care —
    # it was just the LLM hedging, not a meaningful classification
    if len(found_concerns) > 1:
        found_concerns.discard("General Care")

    # Only fall back to General Care if nothing at all was matched
    if not found_concerns:
        return "General Care"
        
    # Sort alphabetically and strictly enforce a MAXIMUM of 2 concerns
    final_concerns = sorted(list(found_concerns))[:2]
    return " | ".join(final_concerns)

# ── MAIN ENGINE ───────────────────────────────────────────────────────────────
def run_daily_ai_analysis():
    print("-" * 65)
    print(f"🧠 Gemma 4 (e2B) DAILY CONVERSATION ANALYZER (From {START_DATE})")
    print(f"   Concerns loaded: {len(KNOWN_CONCERNS)}")
    print("-" * 65)

    if not INPUT_CSV.exists():
        print(f"❌ Error: Cannot find {INPUT_CSV.name}")
        return

    if FORCE_RERUN and OUTPUT_CSV.exists():
        print("⚠️  FORCE_RERUN enabled. Wiping previous results...")
        OUTPUT_CSV.unlink() 

    print(f"📂 Loading {INPUT_CSV.name}...")
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    initial_count = len(df)

    df["session_start_dt"] = pd.to_datetime(df["session_start"], format='mixed', dayfirst=True, errors="coerce")
    df = df.dropna(subset=["session_start_dt", "full_context", "session_id"])
    
    target_mask = df["session_start_dt"] >= pd.to_datetime(START_DATE)
    df_target = df[target_mask].copy()

    print(f"📅 Filtered {initial_count:,} → {len(df_target):,} sessions since {START_DATE}.")

    if df_target.empty:
        print("✅ No new sessions found matching the date criteria.")
        return

    processed_ids = set()
    if OUTPUT_CSV.exists():
        try:
            df_existing = pd.read_csv(OUTPUT_CSV)
            processed_ids = set(df_existing["session_id"].astype(str))
            print(f"⏭️  Found {len(processed_ids):,} sessions already processed. Resuming...")
        except Exception as e:
            pass

    df_to_process = df_target[~df_target["session_id"].astype(str).isin(processed_ids)].copy()

    if df_to_process.empty:
        print("✅ All sessions in the date range have already been analyzed!")
        return

    print(f"🚀 Processing {len(df_to_process):,} sessions...\n")

    file_exists = OUTPUT_CSV.exists()
    
    with open(OUTPUT_CSV, mode="a", encoding="utf-8") as f:
        if not file_exists:
            f.write("session_id,ai_inferred_concern,ai_reasoning\n")

        for _, row in tqdm(df_to_process.iterrows(), total=len(df_to_process), desc="Analyzing"):
            session_id = str(row["session_id"])
            chat_text = str(row["full_context"])
            
            tags = str(row.get("final_tags", ""))
            product = str(row.get("matched_product", ""))

            if len(chat_text) > 1500:
                chat_text = chat_text[-1500:]

            # TEXT-BASED PROMPT (No Numbers allowed!)
            prompt = f"""You are a pharmaceutical data analyst classifying customer chats.
            
            ALLOWED LIST OF CONCERNS: 
            {", ".join(KNOWN_CONCERNS)}
            
            CHAT TRANSCRIPT: "{chat_text}"
            PRODUCT MENTIONED: "{product}"
            STAFF TAGS: "{tags}"
            
            STRICT RULES:
            1. Output a MAXIMUM OF 2 categories from the Allowed List. NEVER output more than 2.
            2. If the user mentions a specific condition (e.g., "dry skin", "dull skin", "pimples"), output that exact category.
            3. If they only mention a brand (e.g., "La Roche Posay", "Uncover", "shampoo") without a symptom, output the broad category (e.g., "Skin Care", "Hair Health"). 
            4. If the chat is purely about delivery, pricing, or ordering with ZERO health products mentioned, output: General Care
            5. Output ONLY the exact text categories separated by commas. No explanations.
            
            YOUR ANSWER (Max 2 categories):"""

            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = ollama.chat(
                        model=LLM_MODEL, 
                        messages=[
                            {
                                'role': 'system',
                                'content': (
                                    'You are a strict classifier. '
                                    'Output a maximum of two categories from the allowed list, separated by commas. '
                                    'Do not copy the whole list. '
                                    'NO Introductionry text allowed. '
                                )
                            },
                            {'role': 'user', 'content': prompt}
                        ],
                        options={
                            'temperature': 0.0,
                            'top_p': 1.0,
                            'num_predict': 100 # Strictly limits the output length to prevent alphabet soup
                        }
                    )
                    
                    raw_answer = response['message']['content'].strip().replace('\n', ', ')
                    
                    if not raw_answer or raw_answer.lower() == "none":
                        raw_answer = "General Care"
                        
                    mapped_concern = clean_llm_response(raw_answer)

                    safe_raw = raw_answer.replace(',', ';').replace('"', "'")
                    f.write(f'"{session_id}","{mapped_concern}","{safe_raw}"\n')
                    f.flush() 
                    break 
                    
                except Exception as e:
                    if attempt < max_retries - 1:
                        time.sleep(2)
                    else:
                        print(f"\n❌ Ollama error on session {session_id}: {e}")
                        break
            else:
                continue
                        

    print("\n" + "-" * 65)
    print(f"✅ COMPLETE. Results saved to {OUTPUT_CSV.name}")
    print("-" * 65)

if __name__ == "__main__":
    run_daily_ai_analysis()