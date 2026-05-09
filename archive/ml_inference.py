import pandas as pd
import os
import torch
import re
from tqdm.auto import tqdm
from sentence_transformers import CrossEncoder

# V3 PRODUCTION IMPORTS (Absolute Package Paths)
from Portal_ML_V4.src.config.settings import (
    SESSION_GAP_HOURS, CROSS_ENCODER_MODEL, 
    CATEGORY_CONFIDENCE_THRESHOLD, MSG_INTERIM_PARQUET,
    FINAL_TAGGED_DATA, HIGH_VALUE_THRESHOLD
)
from Portal_ML_V4.src.config.constants import ML_LABELS, ML_TO_RESP
from Portal_ML_V4.src.core.mpesa_engine import detect_payment_converted_v2
from Portal_ML_V4.src.core.signal_detectors import (
    extract_locations_zones, detect_price_quote, detect_brands,
    detect_price_objection, infer_concerns_from_text, _normalise_tags
)
from Portal_ML_V4.src.utils.text_cleaner import (
    extract_message_text, text_for_matching, is_low_signal_text, is_system_message
)
from Portal_ML_V4.src.config.tag_rules import enrich_canonical_categories_from_text

def run_ml_inference(batch_size=128):
    print("-" * 65)
    print("🚀 PORTAL V3 ML INFERENCE: NUCLEAR PRODUCTION MODE")
    print("-" * 65)

    # 1. LOAD INTERIM DATA
    if not os.path.exists(MSG_INTERIM_PARQUET):
        print(f"❌ Error: Missing {MSG_INTERIM_PARQUET}. Run cleaning.py first.")
        return
    
    df = pd.read_parquet(MSG_INTERIM_PARQUET)
    print(f"📖 Loaded {len(df):,} human-signal messages.")

    # 2. CHRONOLOGICAL SESSIONIZATION
    df['Date & Time'] = pd.to_datetime(df['Date & Time'])
    df = df.sort_values(['Contact ID', 'Date & Time'])
    
    # Calculate time gaps to separate "Hello" on Monday from "Hello" on Friday
    df["prev_time"] = df.groupby("Contact ID")["Date & Time"].shift(1)
    df["gap_hours"] = (df["Date & Time"] - df["prev_time"]).dt.total_seconds() / 3600.0
    df["new_session"] = (df["gap_hours"] > SESSION_GAP_HOURS) | df["prev_time"].isna()
    df["session_id"] = df.groupby("Contact ID")["new_session"].cumsum().astype(int)

    # 3. CONTEXT AGGREGATION & NUCLEAR CLEANING
    print("🔬 Aggregating session context and stripping noise...")
    df["content_extracted"] = df["Content"].apply(extract_message_text)
    
    # Filter out system-level noise (WhatsApp template metadata, etc.)
    df = df[~df["content_extracted"].apply(is_system_message)].copy()
    
    # Concatenate all messages in a session into a single 'story' for the AI
    sessions = df.groupby(["Contact ID", "session_id"]).agg({
        "content_extracted": lambda x: " ".join([str(s) for s in x if s]),
        "Date & Time": "min",
        "Channel ID": "first"
    }).rename(columns={"content_extracted": "full_context", "Date & Time": "session_start"}).reset_index()

    # 4. HEURISTIC & DETERMINISTIC SIGNAL DETECTION
    print("📡 Extracting Business Intelligence signals...")
    sessions['tags'] = [set() for _ in range(len(sessions))]
    sessions['mpesa_amount'] = None
    sessions['mpesa_code'] = None

    for idx, row in tqdm(sessions.iterrows(), total=len(sessions), desc="Heuristics"):
        ctx = row['full_context']
        tags = set()
        
        # A. Detect Zones (Rongai, Mombasa, etc.)
        zones = extract_locations_zones(ctx)
        for z in zones: tags.add(f"Zone: {z}")
        
        # B. Detect Brands (Explicitly)
        brands = detect_brands(ctx)
        tags.update(brands) # Adds "CeraVe", "La Roche Posay" directly to tags

        # C. Detect Categories (Using Tag Rules)
        cat_tags = enrich_canonical_categories_from_text(ctx, existing=tags, source="chat")
        tags.update(cat_tags)

        # D. Detect Concerns (Using new cleaner logic)
        concerns = infer_concerns_from_text(ctx)
        for c in concerns: tags.add(c) # Adds "Acne" directly (no prefix)
        
        # E. Pricing Intelligence
        if detect_price_quote(ctx): tags.add("Funnel: Price Quoted")
        if detect_price_objection(ctx): tags.add("Concern: Price Objection")
        
        # F. Nuclear M-Pesa Detection
        mpesa = detect_payment_converted_v2(ctx)
        if mpesa['is_converted']:
            tags.add("Converted")
            sessions.at[idx, 'mpesa_amount'] = mpesa['amount']
            sessions.at[idx, 'mpesa_code'] = "|".join(mpesa['tx_code']) if mpesa['tx_code'] else None
        elif mpesa['is_instruction']:
            tags.add("Funnel: Payment Instruction Sent")
            
        sessions.at[idx, 'tags'] = tags

    # 5. AI CLASSIFICATION (Cross-Encoder Batch Inference)
    print(f"🤖 Initializing Cross-Encoder: {CROSS_ENCODER_MODEL}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CrossEncoder(CROSS_ENCODER_MODEL, device=device)

    # Only process high-signal sessions to save GPU/Time
    sessions['is_low_signal'] = sessions['full_context'].apply(is_low_signal_text)
    to_classify = sessions[~sessions['is_low_signal']].copy()

    if not to_classify.empty:
        print(f"🧠 Classifying {len(to_classify):,} sessions...")
        
        texts = to_classify['full_context'].tolist()
        
        for i in tqdm(range(0, len(texts), batch_size), desc="AI Inference"):
            batch_texts = texts[i : i + batch_size]
            batch_indices = to_classify.index[i : i + batch_size]
            
            # Predict labels for the batch
            for text_idx, text in enumerate(batch_texts):
                # We compare the context against every possible label in the taxonomy
                pairs = [[text, lbl] for lbl in ML_LABELS]
                scores = model.predict(pairs)
                
                # Identify the highest confidence label
                best_score = 0
                best_label = None
                for lbl_idx, score in enumerate(scores):
                    if score > best_score:
                        best_score = score
                        best_label = ML_LABELS[lbl_idx]
                
                # Apply Confidence Threshold
                if best_score >= CATEGORY_CONFIDENCE_THRESHOLD:
                    mapped_category = ML_TO_RESP.get(best_label, "Product Inquiry - Others")
                    sessions.at[batch_indices[text_idx], 'tags'].add(mapped_category)

    # 6. ORDER-PRESERVING NORMALIZATION & EXPORT
    print("💾 Finalizing tags and exporting...")
    # Convert sets to sorted, pipe-separated strings for Excel/Looker Studio compatibility
    sessions['final_tags'] = sessions['tags'].apply(lambda x: " | ".join(_normalise_tags(list(x))))
    
    # Preserve original columns + new AI insights
    output_cols = ['session_start', 'Contact ID', 'session_id', 'Channel ID', 
                   'full_context', 'final_tags', 'mpesa_amount', 'mpesa_code']
    final_df = sessions[output_cols]

    # Save Dual Format
    os.makedirs(FINAL_TAGGED_DATA.parent, exist_ok=True)
    final_df.to_parquet(FINAL_TAGGED_DATA, index=False)
    final_df.to_csv(FINAL_TAGGED_DATA.with_suffix(".csv"), index=False)
    
    print("-" * 65)
    print(f"✅ V3 PRODUCTION PIPELINE COMPLETE")
    print(f"📍 Location: {FINAL_TAGGED_DATA}")
    print("-" * 65)

if __name__ == "__main__":
    run_ml_inference()