"""
ml_inference.py
===============
V4 — Incremental session append + single CrossEncoder load.

What changed from V3:
  - CrossEncoder loaded ONCE at the top (was loaded twice — at step 1
    and again at step 5 — wasting GPU memory and ~30s on every run).
  - FINAL_TAGGED_DATA is now appended to, not overwritten.
    Dedup key: (Contact ID, session_start) — session_start is the min
    timestamp within a session so it is stable across runs.
    For active contacts, open sessions that gained new messages get their
    full_context and tags refreshed (keep='last' on dedup).
    Sessions for inactive contacts (not in the parquet) are untouched.
  - Early exit if MSG_INTERIM_PARQUET is empty — nothing to do.

Interface to cleaning.py and analytics.py is unchanged:
  - Reads:  MSG_INTERIM_PARQUET
  - Writes: FINAL_TAGGED_DATA (.parquet + .csv)
"""

import os
import gc

import pandas as pd
import torch
from tqdm.auto import tqdm
from sentence_transformers import CrossEncoder

# V3 PRODUCTION IMPORTS
from Portal_ML_V4.src.config.settings import (
    SESSION_GAP_HOURS, CROSS_ENCODER_MODEL,
    CATEGORY_CONFIDENCE_THRESHOLD, MSG_INTERIM_PARQUET,
    FINAL_TAGGED_DATA, HIGH_VALUE_THRESHOLD,
)
from Portal_ML_V4.src.config.constants import ML_LABELS, ML_TO_RESP
from Portal_ML_V4.src.core.mpesa_engine import detect_payment_converted_v2
from Portal_ML_V4.src.core.signal_detectors import (
    extract_locations_zones, detect_price_quote, detect_brands,
    detect_price_objection, infer_concerns_from_text, _normalise_tags,
)
from Portal_ML_V4.src.utils.text_cleaner import (
    extract_message_text, is_low_signal_text, is_system_message,
)
from Portal_ML_V4.src.config.tag_rules import enrich_canonical_categories_from_text


# ── Constants ─────────────────────────────────────────────────────────────────

OUR_BRANCHES = [
    "two rivers", "two rivers, ruaka",
    "abc place", "abc place, waiyaki way",
    "galleria", "galleria mall", "galleria mall, langata road",
    "milele mall", "milele mall, ngong road",
    "cbd", "banda street", "cbd, banda street",
]

OUTPUT_COLS = [
    'session_start', 'Contact ID', 'session_id', 'Channel ID',
    'full_context', 'final_tags', 'mpesa_amount', 'mpesa_code',
    'customer_contact_id',
]


# ── Model loader (called once) ────────────────────────────────────────────────

def sanitize_text_for_csv_export(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten embedded newlines in string columns only for CSV exports.
    Excel/Power BI can misparse quoted multiline cells as row breaks,
    which shifts session data into the wrong columns.
    """
    export_df = df.copy()
    text_cols = export_df.select_dtypes(include=['object', 'string']).columns
    for col in text_cols:
        mask = export_df[col].notna()
        export_df.loc[mask, col] = (
            export_df.loc[mask, col]
            .astype(str)
            .str.replace(r'\r\n|\r|\n', ' ', regex=True)
        )
    return export_df


def _load_model() -> tuple[CrossEncoder, str]:
    """
    Load CrossEncoder onto GPU if available, fall back to CPU on OOM.
    Returns (model, device_str).
    Called exactly once per pipeline run.
    """
    if torch.cuda.is_available():
        print("   🧹 Cleaning GPU cache before model load...")
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   ⚙️  Device: {device}")
    print(f"   🧠 Loading CrossEncoder: {CROSS_ENCODER_MODEL}")

    try:
        model = CrossEncoder(CROSS_ENCODER_MODEL, device=device)
    except Exception as e:
        if "out of memory" in str(e).lower():
            print("   ⚠️  GPU OOM — falling back to CPU (slower but stable)")
            torch.cuda.empty_cache()
            device = "cpu"
            model = CrossEncoder(CROSS_ENCODER_MODEL, device="cpu")
        else:
            raise

    return model, device


# ── Append logic ──────────────────────────────────────────────────────────────

def _append_to_final(new_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge new sessions into FINAL_TAGGED_DATA.

    Strategy:
      - Dedup key: (Contact ID, session_start)
      - For active contacts, open sessions that got new messages will have
        the same session_start but updated full_context/tags — keep='last'
        ensures the freshest version wins.
      - Sessions for inactive contacts (not processed this run) are
        preserved untouched from the existing file.

    Returns the combined DataFrame that gets written to disk.
    """
    if not os.path.exists(FINAL_TAGGED_DATA):
        print("   📄 No existing FINAL_TAGGED_DATA — writing fresh.")
        return new_df

    print("   🔗 Appending to existing FINAL_TAGGED_DATA...")
    existing = pd.read_parquet(FINAL_TAGGED_DATA)
    print(f"      Existing: {len(existing):,} sessions")
    print(f"      New/updated: {len(new_df):,} sessions")

    combined = pd.concat([existing, new_df], ignore_index=True)

    # Normalise types before dedup so timestamps compare correctly
    combined['session_start'] = pd.to_datetime(combined['session_start'], errors='coerce')
    combined['Contact ID']    = pd.to_numeric(combined['Contact ID'],    errors='coerce')

    before = len(combined)
    combined = (
        combined
        .sort_values('session_start')              # oldest first so 'last' = freshest
        .drop_duplicates(
            subset=['Contact ID', 'session_start'],
            keep='last',                           # keep refreshed version for open sessions
        )
        .reset_index(drop=True)
    )
    refreshed = before - len(combined)
    print(
        f"      After dedup: {len(combined):,} sessions "
        f"({refreshed:,} refreshed / deduplicated)"
    )
    return combined


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_ml_inference(batch_size: int = 128) -> None:
    print("-" * 65)
    print("🚀 PORTAL V4 ML INFERENCE: INCREMENTAL MODE")
    print("-" * 65)

    # ── Guard: nothing to process ─────────────────────────────────────────────
    if not os.path.exists(MSG_INTERIM_PARQUET):
        print(f"❌ Missing {MSG_INTERIM_PARQUET}. Run cleaning.py first.")
        return

    df = pd.read_parquet(MSG_INTERIM_PARQUET)

    if df.empty:
        print("✅ MSG_INTERIM_PARQUET is empty — no new messages to process.")
        return

    print(f"📖 Loaded {len(df):,} messages for active contacts.")

    # ── Load model ONCE ───────────────────────────────────────────────────────
    model, device = _load_model()

    # ── Step 1: Separate activity ID from customer identity ───────────────────
    df['activity_id'] = df['Contact ID']

    if 'Sender Type' in df.columns:
        df['customer_contact_id'] = df.apply(
            lambda x: x['Contact ID']
            if str(x.get('Sender Type', '')).lower() == 'contact'
            else None,
            axis=1,
        )
    else:
        df['customer_contact_id'] = df['Contact ID']

    # ── Step 2: Chronological sessionization ──────────────────────────────────
    df['Date & Time'] = pd.to_datetime(df['Date & Time'])
    df = df.sort_values(['Contact ID', 'Date & Time'])

    df['prev_time']   = df.groupby('Contact ID')['Date & Time'].shift(1)
    df['gap_hours']   = (df['Date & Time'] - df['prev_time']).dt.total_seconds() / 3600.0
    df['new_session'] = (df['gap_hours'] > SESSION_GAP_HOURS) | df['prev_time'].isna()
    df['session_id']  = df.groupby('Contact ID')['new_session'].cumsum().astype(int)

    # ── Step 3: Context aggregation & noise removal ───────────────────────────
    print("🔬 Aggregating session context and stripping noise...")
    df['content_extracted'] = df['Content'].apply(extract_message_text)
    df = df[~df['content_extracted'].apply(is_system_message)].copy()

    sessions = (
        df.groupby(['Contact ID', 'session_id'])
        .agg(
            full_context        = ('content_extracted',   lambda x: ' '.join(str(s) for s in x if s)),
            session_start       = ('Date & Time',         'min'),
            Channel_ID          = ('Channel ID',          'first'),
            customer_contact_id = ('customer_contact_id', 'max'),
        )
        .rename(columns={'Channel_ID': 'Channel ID'})
        .reset_index()
    )

    print(f"   📦 {len(sessions):,} sessions built from {len(df):,} messages")

    # ── Step 4: Heuristic & deterministic signal detection ────────────────────
    print("📡 Extracting Business Intelligence signals...")
    sessions['tags']         = [set() for _ in range(len(sessions))]
    sessions['mpesa_amount'] = None
    sessions['mpesa_code']   = None
    sessions['primary_zone']    = ''
    sessions['secondary_zones'] = ''

    sorted_branches = sorted(OUR_BRANCHES, key=len, reverse=True)

    for idx, row in tqdm(sessions.iterrows(), total=len(sessions), desc="Heuristics"):
        ctx  = row['full_context']
        tags = set()

        # A. Zones
        clean_ctx = ctx.lower()
        for branch in sorted_branches:
            clean_ctx = clean_ctx.replace(branch, "[INTERNAL_LOC]")

        zones = extract_locations_zones(ctx)
        if zones:
            primary_zone = zones[-1]
            tags.add(f"Zone: {primary_zone}")
            sessions.at[idx, 'primary_zone'] = primary_zone
            if len(zones) > 1:
                secondary_list = zones[:-1]
                sessions.at[idx, 'secondary_zones'] = ' | '.join(secondary_list)
                for sz in secondary_list:
                    tags.add(f"Secondary Zone: {sz}")

        # B. Brands
        tags.update(detect_brands(ctx))

        # C. Categories
        tags.update(enrich_canonical_categories_from_text(ctx, existing=tags, source='chat'))

        # D. Concerns
        for c in infer_concerns_from_text(ctx):
            tags.add(c)

        # E. Pricing
        if detect_price_quote(ctx):    
            tags.add("Funnel: Price Quoted")
        if detect_price_objection(ctx): 
            tags.add("Concern: Price Objection")

        # F. M-Pesa
        mpesa = detect_payment_converted_v2(ctx)
        if mpesa['is_converted']:
            tags.add("Converted")
            sessions.at[idx, 'mpesa_amount'] = mpesa['amount']
            sessions.at[idx, 'mpesa_code']   = (
                '|'.join(mpesa['tx_code']) if mpesa['tx_code'] else None
            )
        elif mpesa['is_instruction']:
            tags.add("Funnel: Payment Instruction Sent")

        sessions.at[idx, 'tags'] = tags

    # ── Step 5: AI classification (CrossEncoder already loaded above) ─────────
    sessions['is_low_signal'] = sessions['full_context'].apply(is_low_signal_text)
    to_classify = sessions[~sessions['is_low_signal']].copy()

    if not to_classify.empty:
        print(f"🧠 Classifying {len(to_classify):,} high-signal sessions...")
        texts = to_classify['full_context'].tolist()

        for i in tqdm(range(0, len(texts), batch_size), desc="AI Inference"):
            batch_texts   = texts[i : i + batch_size]
            batch_indices = to_classify.index[i : i + batch_size]

            for text_idx, text in enumerate(batch_texts):
                pairs      = [[text, lbl] for lbl in ML_LABELS]
                scores     = model.predict(pairs, show_progress_bar=False)
                best_score = max(scores)
                best_label = ML_LABELS[scores.index(best_score)] if hasattr(scores, 'index') \
                             else ML_LABELS[scores.tolist().index(best_score)]

                if best_score >= CATEGORY_CONFIDENCE_THRESHOLD:
                    mapped = ML_TO_RESP.get(best_label, "Product Inquiry - Others")
                    sessions.at[batch_indices[text_idx], 'tags'].add(mapped)
    else:
        print("⏭️  All sessions are low-signal — skipping AI classification.")

    # ── Step 6: Finalise tags ─────────────────────────────────────────────────
    print("💾 Finalising tags...")
    sessions['final_tags'] = sessions['tags'].apply(
        lambda x: ' | '.join(_normalise_tags(list(x)))
    )

    # Keep only the output columns that exist
    available_output_cols = [c for c in OUTPUT_COLS if c in sessions.columns]
    new_sessions = sessions[available_output_cols].copy()

    # ── Step 7: Append to FINAL_TAGGED_DATA ───────────────────────────────────
    combined = _append_to_final(new_sessions)

    # ── Step 8: Write dual format ─────────────────────────────────────────────
    os.makedirs(FINAL_TAGGED_DATA.parent, exist_ok=True)
    combined.to_parquet(FINAL_TAGGED_DATA, index=False)
    # Flatten multiline chat text for CSV only so Excel/Power BI do not break rows.
    combined_csv = sanitize_text_for_csv_export(combined)
    combined_csv.to_csv(FINAL_TAGGED_DATA.with_suffix('.csv'), index=False)

    print("-" * 65)
    print(" V4 INFERENCE COMPLETE")
    print(f"   New sessions processed : {len(new_sessions):,}")
    print(f"   Total sessions on file : {len(combined):,}")
    print(f"   📍 {FINAL_TAGGED_DATA}")
    print("-" * 65)


if __name__ == "__main__":
    run_ml_inference()
