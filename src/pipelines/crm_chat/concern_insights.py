"""
analyze_daily_sessions.py
=========================
Scans daily WhatsApp conversations starting from a specific date.
Uses lightning-fast Regex keyword matching to read the `full_context`
and infer the primary customer concern.
"""

import re
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Import dynamic paths and your new centralized Regex Dictionary
from Portal_ML_V4.src.config.settings import RAW_DATA_DIR, PROCESSED_DATA_DIR
from Portal_ML_V4.src.config.concerns import CONCERN_KEYWORDS

# ── PATHS & SETTINGS ─────────────────────────────────────────────────────────

INPUT_CSV = Path(PROCESSED_DATA_DIR) / "fact_sessions_enriched.csv"
OUTPUT_CSV = Path(PROCESSED_DATA_DIR) / "daily_session_concerns.csv"

START_DATE = "2026-01-01"
FORCE_RERUN = True


# ── REGEX HELPER ─────────────────────────────────────────────────────────────

def extract_concerns_via_regex(chat_text: str) -> str:
    """
    Scans a chat string against the CONCERN_KEYWORDS dictionary using regex.
    Returns a pipe-separated string of matched concerns, or 'General Care'.
    """
    if not isinstance(chat_text, str) or not chat_text.strip():
        return "General Care"

    chat_lower = chat_text.lower()
    matched_concerns = set()

    for concern, pattern_list in CONCERN_KEYWORDS.items():
        for pattern in pattern_list:
            if re.search(pattern, chat_lower, flags=re.IGNORECASE):
                matched_concerns.add(concern)
                break  # Stop checking other patterns for this specific concern

    if not matched_concerns:
        return "General"

    final_concerns = sorted(list(matched_concerns))
    return " | ".join(final_concerns)


# ── MAIN ENGINE ──────────────────────────────────────────────────────────────

def run_daily_analysis() -> None:
    """Run the main daily session analysis workflow using Regex."""
    print("-" * 65)
    print(f"🔍 REGEX CONVERSATION ANALYZER (From {START_DATE})")
    print(f"   Keywords loaded for {len(CONCERN_KEYWORDS)} categories")
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

    df["session_start_dt"] = pd.to_datetime(
        df["session_start"],
        format='mixed',
        dayfirst=True,
        errors="coerce"
    )
    df = df.dropna(subset=["session_start_dt", "full_context", "session_id"])

    target_mask = df["session_start_dt"] >= pd.to_datetime(START_DATE)
    df_target = df[target_mask].copy()

    print(
        f"📅 Filtered {initial_count:,} → {len(df_target):,} "
        f"sessions since {START_DATE}."
    )

    if df_target.empty:
        print("✅ No new sessions found matching the date criteria.")
        return

    processed_ids = set()
    if OUTPUT_CSV.exists():
        try:
            df_existing = pd.read_csv(OUTPUT_CSV)
            processed_ids = set(df_existing["session_id"].astype(str))
            print(
                f"⏭️  Found {len(processed_ids):,} sessions already "
                "processed. Resuming..."
            )
        except Exception:
            pass

    unprocessed_mask = ~df_target["session_id"].astype(str).isin(processed_ids)
    df_to_process = df_target[unprocessed_mask].copy()

    if df_to_process.empty:
        print("✅ All sessions in the date range have already been analyzed!")
        return

    print(f"🚀 Processing {len(df_to_process):,} sessions...\n")

    file_exists = OUTPUT_CSV.exists()

    with open(OUTPUT_CSV, mode="a", encoding="utf-8") as f:
        # We keep the 'ai_reasoning' column header so it doesn't break 
        # your Power BI schema, but we'll just fill it with "Regex Match"
        if not file_exists:
            f.write("session_id,ai_inferred_concern,ai_reasoning\n")

        for _, row in tqdm(
            df_to_process.iterrows(),
            total=len(df_to_process),
            desc="Analyzing"
        ):
            session_id = str(row["session_id"])
            chat_text = str(row["full_context"])

            # 1. Extract using the fast Regex function
            mapped_concern = extract_concerns_via_regex(chat_text)

            # Split into individual concerns and write one row each
            concerns = [c.strip() for c in mapped_concern.split("|") if c.strip()]
            for concern in concerns:
                safe_raw = f"Regex Match: {concern}"
                f.write(f'"{session_id}","{concern}","{safe_raw}"\n')
            
    print("\n" + "-" * 65)
    print(f"✅ COMPLETE. Results saved to {OUTPUT_CSV.name}")
    print("-" * 65)


if __name__ == "__main__":
    run_daily_analysis()