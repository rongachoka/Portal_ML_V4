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
import re
import pandas as pd
import ollama
from pathlib import Path
from tqdm import tqdm

from Portal_ML_V4.src.config.settings import RAW_DATA_DIR, PROCESSED_DATA_DIR

# ── PATHS & SETTINGS ──────────────────────────────────────────────────────────
INPUT_CSV          = Path(PROCESSED_DATA_DIR) / "fact_sessions_enriched.csv"
OUTPUT_CSV         = Path(PROCESSED_DATA_DIR) / "ai_daily_session_concerns.csv"
CONCERNS_LIST_PATH = Path(RAW_DATA_DIR) / "Concerns List V1(Sheet1).csv"

START_DATE  = "2026-04-01"
# LLM_MODEL   = "llama3.2"
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

# Build numbered lookup once at startup — e.g. {1: "Acne", 2: "Dry Skin", ...}
# "General Care" is always option 0 so the model has a clear logistical escape hatch
CONCERN_INDEX: dict[int, str] = {0: "General Care"}
CONCERN_INDEX.update({
    i + 1: c for i, c in enumerate(c for c in KNOWN_CONCERNS if c != "General Care")
})
# Reverse map for prompt display
NUMBERED_LIST = "\n".join(f"  {k}. {v}" for k, v in CONCERN_INDEX.items())

# ── PRE-FILTER: skip LLM for obviously logistical sessions ────────────────────
_LOGISTICAL_PATTERNS = re.compile(
    r'\b(hello|hi|hey|good morning|good evening|delivery|shipping|location|branch|'
    r'how much|price|payment|mpesa|confirm|order|available|stock|okay|sawa|thank)\b',
    re.IGNORECASE
)
_PRODUCT_SIGNAL = re.compile(
    r'\b(cerave|la roche|ordinary|neutrogena|anua|serum|cleanser|moistur|vitamin|'
    r'supplement|retinol|niacinamide|spf|sunscreen|acne|skin|hair|baby|pain|'
    r'tablet|capsule|lotion|cream|gel|spray|shampoo)\b',
    re.IGNORECASE
)

def is_logistical_only(chat_text: str) -> bool:
    """Returns True if the session has no product/health signal — skip LLM."""
    if len(chat_text.strip()) < 30:
        return True
    has_product_signal = bool(_PRODUCT_SIGNAL.search(chat_text))
    return not has_product_signal


# ── RESPONSE PARSER ───────────────────────────────────────────────────────────
MAX_CONCERNS = 3  # Hard cap — prevents the model from hedging with everything

def parse_numbered_response(response: str) -> str:
    """
    Extracts numbers from the LLM response and maps back to concern names.
    Handles both comma and semicolon separated outputs.
    Caps at MAX_CONCERNS to prevent over-classification.
    """
    found = []

    numbers = re.findall(r'\b(\d+)\b', response)
    for n in numbers:
        idx = int(n)
        if idx == 0:
            continue  # Skip General Care — handled separately below
        if idx in CONCERN_INDEX and CONCERN_INDEX[idx] not in found:
            found.append(CONCERN_INDEX[idx])

    # If nothing valid or model only returned 0
    if not found:
        return "General Care"

    # Cap at MAX_CONCERNS — the model lists most confident first
    found = found[:MAX_CONCERNS]
    return " | ".join(sorted(found))


# ── MAIN ENGINE ───────────────────────────────────────────────────────────────
def run_daily_ai_analysis():
    print("-" * 65)
    print(f"🧠 Llama 3.2 DAILY CONVERSATION ANALYZER (From {START_DATE})")
    print(f"   Concerns loaded: {len(KNOWN_CONCERNS)}")
    print("-" * 65)

    if not INPUT_CSV.exists():
        print(f"❌ Error: Cannot find {INPUT_CSV.name}")
        return

    if FORCE_RERUN and OUTPUT_CSV.exists():
        print("⚠️  FORCE_RERUN enabled. Wiping previous results...")
        OUTPUT_CSV.unlink()

    # 1. Load
    print(f"📂 Loading {INPUT_CSV.name}...")
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    initial_count = len(df)

    # 2. Filter by date
    df["session_start_dt"] = pd.to_datetime(
        df["session_start"], format='mixed', dayfirst=True, errors="coerce"
    )
    df = df.dropna(subset=["session_start_dt", "full_context", "session_id"])
    df_target = df[df["session_start_dt"] >= pd.to_datetime(START_DATE)].copy()
    print(f"📅 Filtered {initial_count:,} → {len(df_target):,} sessions since {START_DATE}.")

    if df_target.empty:
        print("✅ No sessions found for the date range.")
        return

    # 3. Resume logic
    processed_ids = set()
    if OUTPUT_CSV.exists():
        try:
            processed_ids = set(pd.read_csv(OUTPUT_CSV)["session_id"].astype(str))
            print(f"⏭️  Resuming — {len(processed_ids):,} already processed.")
        except Exception as e:
            print(f"⚠️ Could not read existing output: {e}")

    df_to_process = df_target[
        ~df_target["session_id"].astype(str).isin(processed_ids)
    ].copy()

    if df_to_process.empty:
        print("✅ All sessions already analyzed.")
        return

    print(f"🚀 Processing {len(df_to_process):,} sessions...\n")

    file_exists = OUTPUT_CSV.exists()
    skipped = 0

    with open(OUTPUT_CSV, mode="a", encoding="utf-8") as f:
        if not file_exists:
            f.write("session_id,ai_inferred_concern,ai_reasoning\n")

        for _, row in tqdm(df_to_process.iterrows(), total=len(df_to_process), desc="Analyzing"):
            session_id = str(row["session_id"])
            chat_text  = str(row["full_context"])
            tags       = str(row.get("final_tags", ""))
            product    = str(row.get("matched_product", ""))

            # ── PRE-FILTER: skip LLM for logistical-only sessions ─────────────
            if is_logistical_only(chat_text):
                f.write(f'"{session_id}","General Care","pre-filter: no product signal"\n')
                f.flush()
                skipped += 1
                continue

            # Trim — take first 800 chars (context/opening) + last 700 (resolution)
            # This preserves both what they asked and what was resolved
            if len(chat_text) > 1500:
                chat_text = chat_text[:800] + " ... " + chat_text[-700:]

            # ── NUMBERED PROMPT ───────────────────────────────────────────────
            # Asking for numbers instead of names eliminates freeform hallucination.
            # The model can ONLY return numbers from the list below.
            prompt = f"""You are a strict data classifier for a pharmacy. Classify this customer chat.
                CHAT: "{chat_text}"
                PRODUCT MENTIONED: "{product}"

                TASK:
                - Pick 1 or 2 numbers from the list below that match what the customer is asking about.
                - MAXIMUM 2 numbers. Do not pick more even if unsure.
                - Only pick a concern if it is DIRECTLY mentioned or clearly implied by the product.
                - If the chat is just about delivery, payment, hello, or ordering with no health topic → reply: 0
                - If you are unsure or the health topic is not clearly listed → reply: 0
                - You MUST NOT leave the answer blank. You must output at least one number.
                - Do NOT explain. Output numbers only, comma separated.

                CONCERNS:
                {NUMBERED_LIST}

                YOUR ANSWER (1-2 numbers only):"""

            try:
                response = ollama.chat(
                    model=LLM_MODEL,
                    messages=[
                        {
                            'role': 'system',
                            'content': (
                                'You are a strict data classifier. '
                                'You only output numbers separated by commas. '
                                'You never explain, never add words, never invent categories, '
                                'and you NEVER leave the response blank. If unsure, output 0.'
                            )
                        },
                        {'role': 'user', 'content': prompt}
                    ],
                    options={
                        'temperature': 0.0,    # deterministic — no creativity
                        'top_p': 1.0,
                        'num_predict': 20,     # numbers only — no need for long output
                    }
                )
                
                raw_answer = response['message']['content'].strip()
                
                # Python-level safety net: if the model still blanks, force it to '0'
                if not raw_answer:
                    raw_answer = "0"
                mapped_concern = parse_numbered_response(raw_answer)
                safe_raw       = raw_answer.replace(',', ';').replace('"', "'")
                f.write(f'"{session_id}","{mapped_concern}","{safe_raw}"\n')
                f.flush()

            except Exception as e:
                print(f"\n❌ Ollama error on session {session_id}: {e}")
                break

    print("\n" + "-" * 65)
    print(f"✅ COMPLETE. Pre-filtered (no LLM): {skipped:,} sessions")
    print(f"📂 Results: {OUTPUT_CSV}")
    print("-" * 65)


if __name__ == "__main__":
    run_daily_ai_analysis()