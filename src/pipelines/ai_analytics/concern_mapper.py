"""
concern_mapper.py
=================
Maps every WhatsApp session to a concern from the canonical concern list
using a blazing fast Bi-Encoder Semantic AI on the GPU, followed by deterministic fallbacks.

Priority waterfall:
  Layer 1   Semantic AI Inference from full_context (GPU Accelerated)
  Layer 2   matched_concern from KB product match
  Layer 3   Concern tags already in final_tags (Exact/Substring match)
  Layer 3a  primary_category → category-level fallback
  Layer 3b  matched_brand → brand's most common KB concern
  Layer 3c  sub_category / matched_product keyword scan
  Layer 4   Flag for review (genuine unknowns with health/product intent)
"""

import re
import gc
import torch
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
from tqdm.auto import tqdm

# Import dynamic paths from config
from Portal_ML_V4.src.config.settings import RAW_DATA_DIR, PROCESSED_DATA_DIR

# Initialize tqdm for Pandas so we can use .progress_apply()
tqdm.pandas()

# ── DYNAMIC CANONICAL CONCERN LIST ────────────────────────────────────────────

CONCERNS_LIST_PATH = Path(RAW_DATA_DIR) / "Concerns List V1(Sheet1).csv"

def _load_dynamic_concerns() -> list[str]:
    """Loads the canonical concerns list dynamically from the boss's CSV."""
    try:
        df_concerns = pd.read_csv(CONCERNS_LIST_PATH)
        # Grab the first column, drop blanks, strip whitespace, and return unique values
        concerns = df_concerns.iloc[:, 0].dropna().astype(str).str.strip().unique().tolist()
        return [c for c in concerns if c.lower() not in ["", "nan", "none"]]
    except Exception as e:
        print(f"❌ CRITICAL ERROR: Could not load the Concerns CSV from {CONCERNS_LIST_PATH}")
        print(f"Error details: {e}")
        return ["General Care"] 

KNOWN_CONCERNS: list[str] = _load_dynamic_concerns()
print(f"📋 Loaded {len(KNOWN_CONCERNS)} unique concerns from external CSV.")

# ── CATEGORY → CONCERN FALLBACK ───────────────────────────────────────────────

CATEGORY_CONCERN_MAP: dict[str, str] = {
    "Skincare":               "Skin Care",
    "Baby Care":              "General Care",
    "Hair Care":              "Hair",
    "Oral Care":              "Oral Care",
    "Supplements":            "General Care",
    "Medicine":               "General Care",
    "Medical Devices & Kits": "General Care",
    "Homeopathy":             "General Care",
    "Men Care":               "Men",
    "Women's Health":         "Hormonal Balance",
    "Perfumes":               "General Care",
    "Lip Care":               "General Care",
    "Sexual Health":          "Sexual Health",
    "General Inquiry":        "General Care",
}

# ── LOGISTICAL — explicitly NOT a health concern ──────────────────────────
_LOGISTICAL_PATTERNS: list[str] = [
    r"(what\s+(time|are)\s+(do\s+)?you\s+(open|close|clos))",
    r"(open(ing)?\s+hour)",
    r"(clos(ing|ed)\s+(time|at|hour))",
    r"(where\s+(are\s+you|is\s+(the\s+)?(shop|branch|store|pharmacy)))",
    r"(how\s+do\s+i\s+get\s+to)",
    r"(directions?\s+to)",
    r"(how\s+long\s+(will|does)\s+(delivery|shipping)\s+take)",
    r"(when\s+will\s+(it|my\s+order)\s+(arrive|be\s+delivered))",
    r"(track(ing)?\s+(my\s+)?order)",
    r"(delivery\s+(fee|cost|charge|time|status))",
    r"(do\s+you\s+deliver\s+to)",
    r"(can\s+you\s+deliver\s+to)",
    r"(mpesa\s+(till|number|paybill))",
    r"(till\s+number)",
    r"(paybill\s+number)",
]

def _is_logistical(text: str) -> bool:
    if not text: return False
    t = text.lower()
    logistical_hits = sum(1 for p in _LOGISTICAL_PATTERNS if re.search(p, t, re.IGNORECASE))
    if logistical_hits == 0: return False
    
    health_signals = [
        r"skin|hair|pain|fever|cough|acne|spot|rash|itch|wound|period|hormonal"
        r"|blood sugar|uti|malaria|weight|sleep|anxiety|stress|supplement|vitamin"
    ]
    has_health = any(re.search(p, t, re.IGNORECASE) for p in health_signals)
    return logistical_hits >= 1 and not has_health

# ── SEMANTIC AI ENGINE ────────────────────────────────────────────────────────
def _infer_concerns_semantically(
    df: pd.DataFrame, 
    known_concerns: list[str], 
    text_col: str = "full_context", 
    threshold: float = 0.42, 
    batch_size: int = 128
) -> pd.DataFrame:
    """
    Uses a fast Bi-Encoder to map customer messages to canonical concerns
    via cosine similarity on the GPU.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   🧠 Loading Semantic Mapper (Bi-Encoder) on {device}...")
    
    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    
    print(f"   📐 Embedding {len(known_concerns)} canonical concerns...")
    concern_embeddings = model.encode(
        known_concerns, 
        convert_to_tensor=True, 
        device=device,
        show_progress_bar=False 
    )
    
    unmapped_mask = df["session_concern"].isna()
    texts = df.loc[unmapped_mask, text_col].fillna("").astype(str).tolist()
    
    if not texts:
        print("   ✅ No unresolved sessions to map semantically.")
        return df
        
    print(f"   🚀 Encoding {len(texts):,} customer sessions in batches...")
    text_embeddings = model.encode(
        texts, 
        batch_size=batch_size, 
        convert_to_tensor=True, 
        device=device,
        show_progress_bar=True 
    )
    
    print("   🔍 Calculating Cosine Similarity Matrix...")
    cosine_scores = util.cos_sim(text_embeddings, concern_embeddings)
    
    best_scores, best_indices = torch.max(cosine_scores, dim=1)
    
    best_scores = best_scores.cpu().numpy()
    best_indices = best_indices.cpu().numpy()
    
    inferred_concerns = []
    for i in range(len(texts)):
        if best_scores[i] >= threshold:
            inferred_concerns.append(known_concerns[best_indices[i]])
        else:
            inferred_concerns.append(None)
            
    df.loc[unmapped_mask, "session_concern"] = inferred_concerns
    
    mapped_now_mask = unmapped_mask & df["session_concern"].notna()
    df.loc[mapped_now_mask, "concern_source"] = "Semantic AI"
    
    if device == "cuda":
        del model
        del concern_embeddings
        del text_embeddings
        torch.cuda.empty_cache()
        gc.collect()
        
    return df

# ── EXACT/SUBSTRING MATCHING HELPERS (Fast Fallbacks) ─────────────────────────
_NON_CONCERN_TAGS = {
    "converted", "funnel:", "zone:", "secondary zone:",
    "product inquiry", "recommendation", "price quoted",
    "payment instruction", "stock inquiry",
}

def _fast_substring_match(text: str) -> str | None:
    """Instantly matches against known concerns using basic string operations."""
    if not text or not text.strip(): return None
    t = text.lower().strip()
    
    for concern in KNOWN_CONCERNS:
        if concern.lower() == t:
            return concern
            
    for concern in KNOWN_CONCERNS:
        if concern.lower() in t or t in concern.lower():
            return concern
    return None

def _concerns_from_tags(tags_str: str) -> str | None:
    if not isinstance(tags_str, str) or not tags_str.strip(): return None
    
    for tag in tags_str.split("|"):
        tag = tag.strip()
        if not tag or len(tag) < 3: continue
        if any(skip in tag.lower() for skip in _NON_CONCERN_TAGS): continue
        
        matched = _fast_substring_match(tag)
        if matched: return matched
    return None

def _build_brand_concern_map(kb_path: Path) -> dict[str, str]:
    if not kb_path or not kb_path.exists(): return {}
    try:
        df = pd.read_csv(kb_path)
        df = df.dropna(subset=["Brand", "Concerns"])
        brand_map = (
            df.groupby("Brand")["Concerns"]
            .agg(lambda x: x.mode()[0] if not x.mode().empty else None)
            .dropna()
            .to_dict()
        )
        return {str(k).upper().strip(): str(v).strip() for k, v in brand_map.items()}
    except Exception as e:
        print(f"   ⚠️  concern_mapper: could not build brand map: {e}")
        return {}

# ── KEYWORD SCAN ──────────────────────────────────────────────────────────────
_KEYWORD_CONCERN_MAP: list[tuple[str, str]] = [
    # Women's Health & Intimate Care 
    (r'\bazo\b|boric acid|suppositor|yoni|thrush|candida|yeast', "UTI's"),
    (r'\bplan b\b|p2|postinor|contraceptive', "Hormonal Balance"),
    (r'\bperiod\b|cramp|menstrual', "Period Pain"),
    
    # Skincare
    (r'\bacne\b|pimple|breakout|effaclar|salicylic', "Acne-Prone Skin"),
    (r'\bhyperpigment|dark spot|brighten|vitamin c serum', "Hyperpigmentation"),
    (r'\bdry skin\b|cerave hydrating|moisturiz', "Dry Skin"),
    (r'\boily skin\b|sebum|matte', "Oily Skin"),
    (r'\bsensitive skin\b|rosacea|redness', "Sensitive Skin"),
    (r'\beczema\b|dermatitis', "Eczema"),
    (r'\banti.?ag|retinol|wrinkle|fine line', "Anti-aging"),
    (r'\bspf\b|sunscreen|sunblock|anthelios', "SPF"),
    (r'\bstretch mark|bio oil', "Stretch Marks"),
    
    # Hair & Body
    (r'\bdandruff\b|flaky scalp', "Dandruff"),
    (r'\bhair loss\b|minoxidil|thinning|receding', "Hair Loss"),
    (r'\bshampoo\b|conditioner|hair mask', "Hair"),
    (r'\bwound\b|cut|burn|antiseptic|betadine', "Wound Care"),
    (r'\bdiaper\b|nappy|sudocrem', "Dry Skin, Damaged Skin, Diaper Rash"),
    (r'\boral\b|dental|teeth|toothpaste|mouthwash', "Oral Care"),
    
    # General Medicine
    (r'\bpain\b|panadol|headache|paracetamol|ibuprofen', "Pain Relief"),
    (r'\bfever\b|temperature', "Fever"),
    (r'\bcough\b|syrup|benylin', "Cough Syrup"),
    (r'\bnasal\b|congestion|blocked nose|sinus', "Nasal Congestion"),
    (r'\bsleep\b|insomnia|melatonin', "Sleep"),
    (r'\bweight\b|slim|fat burn', "Weight Management"),
    (r'\bdigest\b|bloat|acid reflux|heartburn|gaviscon', "Digestion"),
    (r'\bimmunit|vitamin c|zinc', "Immunity"),
    (r'\bcollagen\b|joint|bone', "Collagen"),
    (r'\bvitamin\b|supplement', "General Care"),
]

def _concern_from_keywords(row) -> str | None:
    text1, text2 = str(row.get("sub_category", "")), str(row.get("matched_product", ""))
    t = (text1 + " " + text2).lower()
    if t.strip() in ("", "general", "nan", "none"): return None
    
    for pattern, concern in _KEYWORD_CONCERN_MAP:
        if re.search(pattern, t, re.IGNORECASE): return concern
    return None

# ── MAIN FUNCTION ─────────────────────────────────────────────────────────────

def enrich_session_concerns(
    df: pd.DataFrame,
    output_dir: Path,
    kb_path: Path | None = None,
) -> pd.DataFrame:
    print("\n🔬 CONCERN MAPPER: Enriching session concerns (Semantic AI Mode)...")

    df = df.copy()
    df["session_concern"] = None
    df["concern_source"]  = None

    brand_concern_map = _build_brand_concern_map(kb_path) if kb_path else {}
    print(f"   📚 Brand→concern map: {len(brand_concern_map)} brands loaded")

    # ── LAYER 1: Semantic AI Inference (GPU) ──────────────────────────────────
    print("   Initiating Layer 1: Semantic AI Inference...")
    df = _infer_concerns_semantically(
        df=df, 
        known_concerns=KNOWN_CONCERNS,
        threshold=0.42  
    )
    mapped_by_ai = (df["concern_source"] == "Semantic AI").sum()
    print(f"   Layer 1 (Semantic AI):    {mapped_by_ai:,} sessions mapped")

    # ── LAYER 2: KB product match ─────────────────────────────────────────────
    unresolved = df["session_concern"].isna()
    kb_mask = (
        unresolved &
        df["matched_concern"].notna() &
        ~df["matched_concern"].str.strip().str.lower().isin(["", "general", "general care", "nan", "none"])
    )
    df.loc[kb_mask, "session_concern"] = df.loc[kb_mask, "matched_concern"].str.strip()
    df.loc[kb_mask, "concern_source"]  = "KB Product Match"
    print(f"   Layer 2 (KB product):     {kb_mask.sum():,} sessions")

    # ── LAYER 3: Concerns in final_tags ───────────────────────────────────────
    unresolved = df["session_concern"].isna()
    tqdm.pandas(desc="Layer 3: Scanning final_tags")
    tag_concerns = df.loc[unresolved, "final_tags"].progress_apply(_concerns_from_tags)
    tag_mask = unresolved & tag_concerns.notna()
    df.loc[tag_mask, "session_concern"] = tag_concerns[tag_mask]
    df.loc[tag_mask, "concern_source"]  = "Tag Rules"
    print(f"   Layer 3 (tag rules):      {tag_mask.sum():,} sessions")

    # ── LAYER 3a: primary_category fallback ───────────────────────────────────
    unresolved = df["session_concern"].isna()
    cat_concerns = df.loc[unresolved, "primary_category"].map(CATEGORY_CONCERN_MAP)
    cat_mask = unresolved & cat_concerns.notna()
    df.loc[cat_mask, "session_concern"] = cat_concerns[cat_mask]
    df.loc[cat_mask, "concern_source"]  = "Category Fallback"
    print(f"   Layer 3a (category):      {cat_mask.sum():,} sessions")

    # ── LAYER 3b: matched_brand → KB concern ──────────────────────────────────
    unresolved = df["session_concern"].isna()
    if brand_concern_map and "matched_brand" in df.columns:
        def _brand_lookup(brand_val: str) -> str | None:
            if not isinstance(brand_val, str): return None
            raw = brand_concern_map.get(brand_val.upper().strip())
            return _fast_substring_match(raw) if raw else None

        tqdm.pandas(desc="Layer 3b: Mapping Brands")
        brand_concerns = df.loc[unresolved, "matched_brand"].progress_apply(_brand_lookup)
        brand_mask = unresolved & brand_concerns.notna()
        df.loc[brand_mask, "session_concern"] = brand_concerns[brand_mask]
        df.loc[brand_mask, "concern_source"]  = "Brand→KB Concern"
        print(f"   Layer 3b (brand KB):      {brand_mask.sum():,} sessions")

    # ── LAYER 3c: keyword scan ────────────────────────────────────────────────
    unresolved = df["session_concern"].isna()

    tqdm.pandas(desc="Layer 3c: Keyword Scan")
    kw_concerns = df[unresolved].progress_apply(_concern_from_keywords, axis=1)
    kw_mask = unresolved & kw_concerns.notna()
    df.loc[kw_mask, "session_concern"] = kw_concerns[kw_mask]
    df.loc[kw_mask, "concern_source"]  = "Keyword Scan"
    print(f"   Layer 3c (keywords):      {kw_mask.sum():,} sessions")

    # ── LAYER 4: Triage remaining sessions ───────────────────────────────────
    unresolved = df["session_concern"].isna()

    if unresolved.any():
        tqdm.pandas(desc="Layer 4: Triage Logistical Sessions")
        is_logistical = df.loc[unresolved, "full_context"].progress_apply(_is_logistical)

        logistical_mask = unresolved & is_logistical
        review_mask     = unresolved & ~is_logistical

        df.loc[logistical_mask, "session_concern"] = "General Care"
        df.loc[logistical_mask, "concern_source"]  = "Logistical Session"

        flagged_count = review_mask.sum()
        logistical_count = logistical_mask.sum()
        print(f"   Layer 4 (logistical):     {logistical_count:,} sessions → General Care")
        print(f"   Layer 4 (for review):     {flagged_count:,} sessions")

        if flagged_count > 0:
            review_cols = [
                "session_id", "session_start", "Contact ID",
                "primary_category", "matched_brand", "matched_product",
                "sub_category", "final_tags", "acquisition_source",
            ]
            available = [c for c in review_cols if c in df.columns]
            df_review = df[review_mask][available].copy()
            df_review["suggested_concern"] = ""
            df_review["approved"] = "No"

            review_path = output_dir / "concerns_for_review.csv"
            if review_path.exists():
                existing = pd.read_csv(review_path)
                new_rows = df_review[
                    ~df_review["session_id"].isin(existing["session_id"])
                ]
                if not new_rows.empty:
                    df_review = pd.concat([existing, new_rows], ignore_index=True)
                    df_review.to_csv(review_path, index=False)
                    print(f"\n   📋 Review file updated: {review_path} (+{len(new_rows):,} new rows)")
                else:
                    print(f"\n   📋 Review file unchanged — all flagged sessions already present")
            else:
                df_review.to_csv(review_path, index=False)
                print(f"\n   📋 Review file created: {review_path} ({flagged_count:,} rows)")

            df.loc[review_mask, "session_concern"] = "General Care"
            df.loc[review_mask, "concern_source"]  = "Flagged for Review"

    # ── SUMMARY ───────────────────────────────────────────────────────────────
    print("\n   📊 Concern source breakdown:")
    for src, count in df["concern_source"].value_counts().items():
        pct = count / len(df) * 100
        print(f"      {src:<35} {count:>6,}  ({pct:.1f}%)")

    print(f"\n   ✅ All {len(df):,} sessions have a concern.")

    # ── CSV EXPORT ────────────────────────────────────────────────────────────
    export_cols = [
        "session_id", "session_start", "Contact ID",
        "session_concern", "concern_source",
        "acquisition_source", "channel_name",
        "active_staff", "mpesa_amount", "is_converted",
        "primary_category", "matched_brand", "matched_product",
    ]
    available_export = [c for c in export_cols if c in df.columns]
    df_export = df[available_export].copy()
 
    export_path = output_dir / "concern_sessions_in_progress.csv"
    df_export.to_csv(export_path, index=False)
 
    print(f"\n   💾 Concerns export saved: {export_path}")
    print(f"      Rows: {len(df_export):,} | Columns: {len(available_export)}")
 
    print("\n   📊 Top 15 concerns in export:")
    top = df_export["session_concern"].value_counts().head(15)
    for concern, count in top.items():
        pct = count / len(df_export) * 100
        print(f"      {concern:<45} {count:>6,}  ({pct:.1f}%)")
 
    return df

# ── STANDALONE EXECUTION BLOCK ────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    
    # 1. Define Paths using dynamic config
    KB_PATH = Path(RAW_DATA_DIR) / "Final_Knowledge_Base_PowerBI.csv"
    input_file = Path(PROCESSED_DATA_DIR) / "fact_sessions_enriched.csv"
    
    print("-" * 65)
    print("🚀 STARTING STANDALONE CONCERN MAPPER")
    print("-" * 65)
    
    # 2. Check if the input data exists
    if not input_file.exists():
        print(f"❌ Error: Cannot find input data at {input_file}")
        print("Please ensure your main analytics pipeline has run and generated this file.")
    else:
        # 3. Load the data
        print(f"📂 Loading sessions from: {input_file.name}...")
        df_sessions = pd.read_csv(input_file)
        
        if 'session_start' in df_sessions.columns:
            df_sessions['session_start'] = pd.to_datetime(df_sessions['session_start'])
            
        # 4. Run the engine
        enrich_session_concerns(
            df=df_sessions,
            output_dir=Path(PROCESSED_DATA_DIR),
            kb_path=KB_PATH if KB_PATH.exists() else None
        )
        
        print("-" * 65)
        print("✅ STANDALONE CONCERN MAPPER COMPLETE")
        print("-" * 65)