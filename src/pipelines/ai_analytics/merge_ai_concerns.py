"""
merge_ai_concerns.py
====================
Joins ai_daily_session_concerns.csv onto fact_sessions_enriched.csv
by session_id, adding ai_inferred_concern and final_concern columns.

final_concern priority:
    1. matched_concern (KB product-level, highest precision)
    2. ai_inferred_concern (LLM session-level, broadest coverage)
    3. 'Not Analyzed' (fallback)

Also patches final_concern into fact_session_concerns.csv by session_id.
"""

import pandas as pd
from pathlib import Path
from Portal_ML_V4.src.config.settings import PROCESSED_DATA_DIR

# ── PATHS ─────────────────────────────────────────────────────────────────────
SESSIONS_PATH       = PROCESSED_DATA_DIR / "fact_sessions_enriched.csv"
AI_CONCERNS_PATH    = PROCESSED_DATA_DIR / "ai_daily_session_concerns.csv"
SESSION_CONCERNS_PATH = PROCESSED_DATA_DIR / "fact_session_concerns.csv"

# Values treated as "not populated" for matched_concern
GENERIC_CONCERN_VALUES = {
    '', 'nan', 'none', 'general', 'general care',
    'general inquiry', 'unknown'
}


def _is_concern_populated(val: str) -> bool:
    """Returns True if val contains at least one non-generic concern."""
    if not val or str(val).strip().lower() in GENERIC_CONCERN_VALUES:
        return False
    # Handle comma-separated: "Acne, Hyperpigmentation, Oily Skin"
    parts = [p.strip().lower() for p in str(val).split(',')]
    return any(p and p not in GENERIC_CONCERN_VALUES for p in parts)


def _is_ai_concern_populated(val: str) -> bool:
    """Returns True if ai_inferred_concern is a real value."""
    if not val or str(val).strip().lower() in GENERIC_CONCERN_VALUES | {'not analyzed'}:
        return False
    # Handle pipe-separated: "Acne | Dry Skin"
    parts = [p.strip().lower() for p in str(val).split('|')]
    return any(p and p not in GENERIC_CONCERN_VALUES for p in parts)


def resolve_final_concern(row) -> str:
    matched = str(row.get('matched_concern', ''))
    ai      = str(row.get('ai_inferred_concern', ''))

    if _is_concern_populated(matched):
        return matched.strip()
    if _is_ai_concern_populated(ai):
        return ai.strip()
    return 'Not Analyzed'


def run_merge_ai_concerns():
    print("-" * 60)
    print("🔗 MERGING AI CONCERNS → fact_sessions_enriched")
    print("-" * 60)

    # 1. LOAD
    print("Loading sessions...")
    df_sessions = pd.read_csv(SESSIONS_PATH, low_memory=False)
    print(f"   Sessions loaded: {len(df_sessions):,} rows")

    print("Loading AI concerns...")
    df_concerns = pd.read_csv(AI_CONCERNS_PATH)
    print(f"   AI concerns loaded: {len(df_concerns):,} rows")

    # 2. CLEAN
    df_concerns = df_concerns.dropna(subset=["session_id", "ai_inferred_concern"])
    df_concerns["session_id"] = df_concerns["session_id"].astype(str).str.strip()
    df_sessions["session_id"] = df_sessions["session_id"].astype(str).str.strip()

    # 3. DEDUPLICATE AI concerns
    before = len(df_concerns)
    df_concerns = df_concerns.drop_duplicates(subset=["session_id"], keep="first")
    if before != len(df_concerns):
        print(f"   Deduplicated AI concerns: {before:,} → {len(df_concerns):,}")

    # 4. DROP old columns if they exist (clean re-merge)
    for col in ["ai_inferred_concern", "final_concern"]:
        if col in df_sessions.columns:
            df_sessions = df_sessions.drop(columns=[col])
            print(f"   ♻️  Dropped existing '{col}' column for clean re-merge")

    # 5. MERGE ai_inferred_concern
    df_merged = df_sessions.merge(
        df_concerns[["session_id", "ai_inferred_concern"]],
        on="session_id",
        how="left"
    )
    df_merged["ai_inferred_concern"] = df_merged["ai_inferred_concern"].fillna("Not Analyzed")

    # 6. BUILD final_concern
    df_merged["final_concern"] = df_merged.apply(resolve_final_concern, axis=1)

    # 7. REPORT
    total        = len(df_merged)
    analyzed     = (df_merged["ai_inferred_concern"] != "Not Analyzed").sum()
    general_care = (df_merged["ai_inferred_concern"] == "General Care").sum()
    not_analyzed = (df_merged["ai_inferred_concern"] == "Not Analyzed").sum()
    kb_sourced   = df_merged["final_concern"].apply(_is_concern_populated).sum()
    ai_sourced   = (
        (~df_merged["final_concern"].apply(_is_concern_populated)) &
        (df_merged["final_concern"] != "Not Analyzed")
    ).sum()

    print("\n   MERGE REPORT:")
    print(f"      Total sessions:           {total:,}")
    print(f"      AI analyzed:              {analyzed:,} ({analyzed/total*100:.1f}%)")
    print(f"      → General Care:           {general_care:,}")
    print(f"      → Specific AI concern:    {analyzed - general_care:,}")
    print(f"      Not yet analyzed:         {not_analyzed:,}")
    print(f"\n      final_concern source:")
    print(f"      → From KB (matched):      {kb_sourced:,}")
    print(f"      → From AI (fallback):     {ai_sourced:,}")
    print(f"      → Not Analyzed:           {total - kb_sourced - ai_sourced:,}")

    print("\n   Top 10 final concerns:")
    top = (
        df_merged[df_merged["final_concern"] != "Not Analyzed"]["final_concern"]
        .value_counts()
        .head(10)
    )
    for concern, count in top.items():
        print(f"      {concern:<40} {count:>5,}")

    # 8. SAVE fact_sessions_enriched
    df_merged.to_csv(SESSIONS_PATH, index=False)
    print(f"\n✅ Saved sessions to: {SESSIONS_PATH}")

    # 9. PATCH fact_session_concerns with final_concern
    if SESSION_CONCERNS_PATH.exists():
        print("\n🔗 Patching fact_session_concerns.csv with final_concern...")
        df_sc = pd.read_csv(SESSION_CONCERNS_PATH)
        df_sc["session_id"] = df_sc["session_id"].astype(str).str.strip()

        # Build session_id → final_concern map
        concern_map = df_merged.set_index("session_id")["final_concern"].to_dict()

        # Drop old final_concern if exists
        if "final_concern" in df_sc.columns:
            df_sc = df_sc.drop(columns=["final_concern"])

        df_sc["final_concern"] = df_sc["session_id"].map(concern_map).fillna("Not Analyzed")

        df_sc.to_csv(SESSION_CONCERNS_PATH, index=False)
        print(f"✅ Saved session concerns to: {SESSION_CONCERNS_PATH}")
        print(f"   Rows patched: {df_sc['final_concern'].ne('Not Analyzed').sum():,}")
    else:
        print(f"⚠️  {SESSION_CONCERNS_PATH.name} not found — skipping concern patch")

    print("-" * 60)


if __name__ == "__main__":
    run_merge_ai_concerns()