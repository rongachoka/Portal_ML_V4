"""
text_cleaner.py
===============
Cleans and filters raw Respond.io message content for ML processing.

Functions:
    extract_message_text(content) → str
        Parses raw message content (plain text, JSON media payloads, URLs)
        into a clean string suitable for session context aggregation.
    is_system_message(text) → bool
        Returns True for automated/workflow messages that should be excluded
        from session context (matched via SYSTEM_PATTERNS from constants.py).
    is_low_signal_text(text) → bool
        Returns True for sessions with too little content to classify
        (e.g. single-word replies, greetings, emoji-only).

Input:  raw message Content strings from Respond.io CSV
Output: cleaned text strings and boolean filter flags, consumed by ml_inference.py
"""

# Portal_ML_V4/src/utils/text_cleaner.py

import os
import re
import json
import pandas as pd
from typing import Iterable, Any
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE

from Portal_ML_V4.src.config.constants import (
    SYSTEM_PATTERNS, 
    LOW_SIGNAL_PHRASES,
    SUPPLEMENT_KEYWORDS,
    SKINCARE_KEYWORDS,
    BABY_KEYWORDS
)
from Portal_ML_V4.src.config.tag_rules import enrich_canonical_categories_from_text, CONCERN_RULES

# Dynamic Imports for Zones and Brands
try:
    from Portal_ML_V4.src.config.zones import ZONE_MAPPING
except ImportError:
    ZONE_MAPPING = {}

try:
    from Portal_ML_V4.src.config.brands import BRAND_LIST
    KNOWN_BRANDS = set(str(b).lower().strip() for b in BRAND_LIST)
except ImportError:
    KNOWN_BRANDS = set()

KNOWN_CONCERNS = set(str(k).lower().strip() for k in CONCERN_RULES.keys())
ILLEGAL_CTRL_RE = re.compile(r"[\x00-\x08\x0B-\x0C\x0E-\x1F]")

# --- TEXT NORMALIZATION ---

def normalize_text_simple(text):
    """Basic whitespace and newline cleanup."""
    if pd.isna(text):
        return ""
    return str(text).replace("\n", " ").replace("\r", " ").replace("\t", " ").strip()

def text_for_matching(s):
    """Deep cleanup for fuzzy matching: lowercase and alphanumeric only."""
    t = normalize_text_simple(s).lower()
    t = re.sub(r"[^\w\s\+\-]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

def extract_message_text(raw):
    """Parses Respond.io JSON content blobs into human-readable strings."""
    if raw is None: 
        return ""
    if not isinstance(raw, str): 
        raw = str(raw)

    try:
        raw = raw.encode('raw_unicode_escape').decode('utf-8')
    except:
        pass
    raw = raw.encode("ascii", "ignore").decode("ascii")
    
    raw = raw.strip()
    if not raw:
        return ""
    
    raw = raw.strip()
    if not raw: 
        return ""

    if raw.startswith("{") and raw.endswith("}"):
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                # Text message
                t = obj.get("type")
                if t == "text":
                    txt = obj.get("text", "")
                    if isinstance(txt, dict):
                        txt = txt.get("body", "") or txt.get("text", "")
                    return str(txt)
                # Attachment message
                if t == "attachment":
                    att = obj.get("attachment", {}) or {}
                    caption = att.get("caption") or att.get("text") or ""
                    if caption: 
                        return str(caption)
                    att_type = att.get("type", "attachment")
                    file_name = att.get("fileName") or ""
                    return f"[{att_type} {file_name}]" if file_name else f"[{att_type} attachment]"
        except Exception:
            return raw
    return raw

# --- AUDIT HEURISTICS ---

def is_system_message(text: str) -> bool:
    """
    HEURISTIC FILTER - PRODUCTION V3
    Strictly removes system noise, metadata, and WhatsApp template dumps 
    to ensure 100% accurate human-signal extraction.
    """
    if not isinstance(text, str):
        return True

    t = text.lower()
    t = re.sub(r"\s+", " ", t).strip()

    if not t:
        return True
    
    # WHITELIST EXCEPTION
    whitelist_phrases = [
        "you have now been assigned to",
        "has been assigned to" 
    ]

    if any(phrase in t for phrase in whitelist_phrases):
        return False

    # 1. Broad Metadata & Echo Patterns
    # Restored: signature ayf, asset_id, and lookaside CDN links
    meta_keywords = [
        "sender id", "recipient id", "timestamp", "message mid", 
        "attachments type", "unsupported_type", "type unsupported payload", 
        "is_echo true", "lookaside fbsbx com ig_messaging_cdn", 
        "asset_id", "signature ayf", "workflow",
        "google.com", "mail-settings", "forwarding confirmation"
    ]
    if any(k in t for k in meta_keywords):
        return True

    # 2. JSON/Data Blob Detection (Audit-Grade)
    # Restored: digits vs words ratio check
    id_like = len(re.findall(r"\b(id|asset_id|sender|recipient)\b", t))
    colons = t.count(":")
    digits = len(re.findall(r"\d", t))
    words = len(t.split())

    if id_like >= 3 and colons >= 3 and digits > words:
        return True

    # 3. Platform & Automation Phrases
    # Restored: auto-reply and closed-by phrases
    system_phrases = [
        "conversation closed by", 
        "user joined the chat", "auto-respond", "auto reply", 
        "automated message",
    ]
    if any(p in t for p in system_phrases):
        return True
    
    # 4. WhatsApp Template Structural Noise
    # Restored: Full list to prevent "CeraVe error" recurrence
    whatsapp_junk = [
        "whatsapp_template", "components type body", "namespace", 
        "element_name", "languagecode en", "example body_text" 
    ]
    if any(k in t for k in whatsapp_junk):
        return True

    return False


def is_low_signal_text(text: str) -> bool:
    """
    Only flags a message as Low Signal if it is EXCLUSIVELY a greeting \n
    AND contains no metadata, brand, or operational keywords.
    """
    if not text:
        return True

    t = text.lower()
    # Remove punctuation for better tokenization
    t_clean = re.sub(r"[^\w\s]", " ", t)
    t_clean = re.sub(r"\s+", " ", t_clean).strip()
    tokens = t_clean.split()
    
    if not tokens:
        return True

    # --- RULE 1: THE SIGNAL OVERRIDE ---
    # If it contains ANY of these, we NEVER mark it as low signal, 
    # even if the message is only one word long.
    if any(b in t for b in KNOWN_BRANDS): return False
    if any(c in t for c in KNOWN_CONCERNS): return False
    
    
    ops_keywords = [
        "niacinamide", "retinol", "salicylic", "glycolic", "azelaic",
        "peptide", "hyaluronic", "arbutin", "minoxidil", "regaine",
        "buy", "order", "cost", "price", "ksh", "pay", "stock", 
        "delivery", "insurance", "card", "location", "visit"
    ]
    if any(kw in t for kw in ops_keywords): return False

    # --- RULE 2: THE "ONLY GREETING" CHECK ---
    # We only filter if the message consists ONLY of these words.
    generic_fillers = {
        "hello", "hi", "hey", "hallo", "morning", "afternoon", "evening", 
        "good", "thanks", "thank", "please", "ok", "okay", "noted",
        "yes", "no", "can", "get", "dear", "sir", "madam"
    }
    
    # Are there any words that are NOT in the filler list?
    meaningful_tokens = [w for w in tokens if w not in generic_fillers]
    
    # If meaningful_tokens exists, it means the user said something 
    # outside of a basic greeting (e.g., "Hello, send price")
    if len(meaningful_tokens) > 0:
        # One last check: If the 'meaningful' word is just a single letter, it's noise
        if len(meaningful_tokens) == 1 and len(meaningful_tokens[0]) <= 1:
            return True
        return False # High signal found

    # If we reached here, the message was ONLY generic fillers
    return True


def normalize_text_simple(text):
    """Deep cleanup for whitespace and encoding artifacts."""
    if pd.isna(text):
        return ""
    
    # 1. Fix UTF-8 Mojibake (Youâ€™re -> You're)
    t = str(text)
    try:
        t = t.encode('raw_unicode_escape').decode('utf-8')
    except:
        pass

    # 2. Remove non-ASCII artifacts (ðŸ™ðŸ»)
    t = t.encode("ascii", "ignore").decode("ascii")

    # 3. Cleanup whitespace
    t = re.sub(r"\s+", " ", t)
    return t.strip()