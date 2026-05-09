# ----------------------------
# Utilities
# ----------------------------

# Imports
import os
import re
import json
import pandas as pd
import datetime
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
from Portal_ML_V2.config.constants import (
    SYSTEM_PATTERNS, 
    LOW_SIGNAL_PHRASES,
    SUPPLEMENT_KEYWORDS,
    SKINCARE_KEYWORDS,
    BABY_KEYWORDS)
from typing import Iterable, Any
from Portal_ML_V2.config.tag_rules import enrich_canonical_categories_from_text
# Location Zones
try:
    from Portal_ML_V2.config.zones import ZONE_MAPPING
except ImportError:
    ZONE_MAPPING = {}
# Brand List
try:
    from Portal_ML_V2.config.brands import BRAND_LIST
    # Pre-process brands for speed: Lowercase and strip
    # We maintain this as a global variable so we don't rebuild it 1000 times
    KNOWN_BRANDS = set(str(b).lower().strip() for b in BRAND_LIST)
except ImportError:
    KNOWN_BRANDS = set()
# Concerns
try:
    from Portal_ML_V2.config.tag_rules import CONCERN_RULES
    # Extract just the KEYS (e.g. "Acne", "Sleep") as high-signal words
    KNOWN_CONCERNS = set(str(k).lower().strip() for k in CONCERN_RULES.keys())
except ImportError:
    KNOWN_CONCERNS = set()

ILLEGAL_CTRL_RE = re.compile(r"[\x00-\x08\x0B-\x0C\x0E-\x1F]")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def extract_locations_zones(text: str) -> list[str]:
    """
    Extract Pricing Zones using the imported ZONE_MAPPING.
    """
    if not text or not ZONE_MAPPING:
        return []
    
    found_zones = set()
    t_lower = text.lower()
    
    # 1. Direct Search (Fast)
    # Check if any known zone appears in the text
    for zone_key, official_name in ZONE_MAPPING.items():
        # Use word boundaries to prevent partial matches (e.g. matching 'bar' in 'barber')
        # We perform a simple check first for speed
        if zone_key in t_lower:
            # Confirm with regex word boundary for safety
            if re.search(r"\b" + re.escape(zone_key) + r"\b", t_lower):
                found_zones.add(official_name)
            
    return list(found_zones)

def detect_price_quote(text: str) -> bool:
    """
    Did the AGENT give a price?
    Looks for: 'Ksh 500', '500/=', '500sh'
    """
    if not text: 
        return False
    
    # Regex: Currency followed by digits OR Digits followed by currency
    # Matches: "Ksh 500", "500/=", "500shs", "500 ksh"
    pat = r'(ksh|kes|shs|sh)\.?\s?[\d,]{2,}|[\d,]{2,}\s?(ksh|shs|/=|kes)'
    return bool(re.search(pat, str(text), re.IGNORECASE))


def detect_price_objection(text: str) -> bool:
    """
    Did the CUSTOMER complain about price?
    """
    if not text: 
        return False
    
    triggers = [
        "too expensive", "high price", "costly", 
        "last price", "best price", "discount",
        "reduce", "budget", "hapana", "expe", "lower"
    ]
    t = str(text).lower()
    return any(trig in t for trig in triggers)


def sanitize_scalar_for_excel(x):
    """
    Strip control chars and make exotic objects safe for Excel.
    """
    # Leave proper numbers and datetimes alone
    if isinstance(x, (int, float, pd.Timestamp)):
        return x

    # Everything else → string (or NaN)
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return x

    s = str(x)
    s = ILLEGAL_CTRL_RE.sub("", s)
    return s


def sanitize_df_for_excel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply sanitize_scalar_for_excel to every cell.
    """
    return df.map(sanitize_scalar_for_excel)


def _normalise_tags(value: Any) -> list[str]:
    """
    Take whatever is in session_tags/final_tags and turn it into a clean list of tags.

    Handles: list, set, str with '|', ',', ';', or a single tag.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []

    if isinstance(value, (list, set, tuple)):
        tags = list(value)
    elif isinstance(value, str):
        # split on | or , or ;, keep non-empty
        parts = re.split(r"\s*\|\s*|\s*,\s*|\s*;\s*", value.strip())
        tags = [p for p in parts if p]
    else:
        # fallback: just cast to str
        tags = [str(value).strip()]

    # strip whitespace, drop empties, dedupe preserving order
    seen = set()
    clean = []
    for t in tags:
        t = t.strip()
        if t and t not in seen:
            seen.add(t)
            clean.append(t)
    return clean


def _serialise_tags(tags: Iterable[str]) -> str:
    """
    Turn list of tags back into a single string for Excel export.
    Adjust the joiner if you prefer ',' instead of '|'.
    """
    tags = [t.strip() for t in tags if t and isinstance(t, str)]
    return " | ".join(dict.fromkeys(tags))  # dedupe while preserving order


def normalize_text_simple(text):
    #Cleans text
    if pd.isna(text):
        return ""
    return str(text).replace("\n", " ").replace("\r", " ").replace("\t", " ").strip()


def text_for_matching(s):
    t = normalize_text_simple(s).lower()
    t = re.sub(r"[^\w\s\+\-]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def is_system_message(text: str) -> bool:
    """
    Heuristic filter for system / technical messages that should not be
    classified or shown as customer content.

    We treat as system if:
      - It looks like JSON/metadata from the platform (sender id, recipient id, timestamps)
      - It describes unsupported attachments / payloads
      - It is clearly an echo or delivery receipt
    """
    if not isinstance(text, str):
        return True

    t = text.lower()
    t = re.sub(r"\s+", " ", t).strip()

    if not t:
        return True

    # 1) Very common IG / FB metadata patterns
    meta_keywords = [
        "sender id",
        "recipient id",
        "timestamp",
        "message mid",
        "attachments type",
        "unsupported_type",
        "type unsupported payload",
        "is_echo true",
        "lookaside fbsbx com ig_messaging_cdn",
        "asset_id",
        "signature ayf",  # starts of those long signatures
        "assigned to",
        "workflow"
    ]
    if any(k in t for k in meta_keywords):
        return True

    # 2) Generic JSON / metadata blobs (lots of colons / ids, little natural language)
    # Count how many "id" words + how many colons and digits
    id_like = len(re.findall(r"\b(id|asset_id|sender|recipient)\b", t))
    colons = t.count(":")
    digits = len(re.findall(r"\d", t))
    words = len(t.split())

    if id_like >= 3 and colons >= 3 and digits > words:
        return True

    # 3) Known chat-platform system texts (you can extend this list over time)
    system_phrases = [
        "conversation closed by",
        "user joined the chat",
        "auto-respond",
        "auto reply",
        "automated message",
    ]
    if any(p in t for p in system_phrases):
        return True
    
    # 4) NEW: Remove WhatsApp Template Raw Dumps
    # Detects: "type whatsapp_template template" 
    # Detects: "example body_text" (The source of the CeraVe error)
    whatsapp_junk = [
        "whatsapp_template",
        "components type body",
        "namespace",
        "element_name",
        "languagecode en",
        "example body_text" 
    ]

    if any(k in t for k in whatsapp_junk):
        return True

    return False


def is_low_signal_text(text: str) -> bool:
    """
    Return True if the message is too generic (Low Signal).
    Return False if it contains ANY Brand, Concern, Zone, or Operational Keyword.
    """
    if not text:
        return True

    t = text.lower()
    t_clean = re.sub(r"[^\w\s]", " ", t)
    t_clean = re.sub(r"\s+", " ", t_clean).strip()
    tokens = t_clean.split()
    
    if not tokens:
        return True

    # ---------------------------------------------------------
    # 1. DYNAMIC CHECKS (Brands, Concerns, Zones)
    # ---------------------------------------------------------
    
    # Check A: Is a Known Brand present? (e.g. "CeraVe", "Regaine")
    # We check if the brand string appears in the text
    if any(b in t for b in KNOWN_BRANDS):
        return False

    # Check B: Is a Known Concern Key present? (e.g. "Acne", "Sleep")
    if any(c in t for c in KNOWN_CONCERNS):
        return False

    # Check C: Is a Known Zone present? (e.g. "Rongai", "Mombasa")
    if ZONE_MAPPING:
        # Optimization: Only check short texts to save time
        if len(tokens) < 15: 
            for zone_key in ZONE_MAPPING.keys():
                if zone_key in t:
                    return False

    # ---------------------------------------------------------
    # 2. MANUAL OPS LIST (Ingredients, Intent, Logistics)
    # ---------------------------------------------------------
    # Things that are NOT brands or concerns but still important.
    
    ops_keywords = [
        # Ingredients (Since BRAND_LIST doesn't have ingredients)
        "niacinamide", "retinol", "salicylic", "glycolic", "azelaic",
        "peptide", "hyaluronic", "arbutin", "minoxidil", "regaine",
        "vitamin", "zinc", "magnesium", "collagen", "omega", "biotic",
        "mg", "ml", "tablet", "cap", "cream", "gel", "serum", "toner", 
        "wash", "cleanser", "moisturizer", "sunscreen",
        
        # Intent / Commerce
        "buy", "order", "purchase", "cost", "price", "ksh", "pay",
        "stock", "available", "have this", "need", "want", "looking for",
        
        # Logistics / Ops
        "location", "located", "branch", "where", "visit", "open", "close",
        "delivery", "deliver", "shipping", "transport", "courier", "send",
        "insurance", "cover", "card"
    ]
    
    if any(kw in t for kw in ops_keywords):
        return False

    # ---------------------------------------------------------
    # 3. THE NOISE FILTER (Generic Greeting Check)
    # ---------------------------------------------------------
    
    generic_fillers = {
        "hello", "hi", "hey", "hallo", "morning", "afternoon", "evening", "good",
        "how", "are", "you", "doing", "today", "hope", "well", "fine",
        "thanks", "thank", "please", "kindly", "ok", "okay", "noted",
        "yes", "no", "can", "get", "help", "assist", "reach", "out",
        "tell", "me", "more", "info", "information", "query", "question",
        "contact", "chat", "dear", "sir", "madam", "waiting"
    }
    
    # Remove generic words to see if anything substantive is left
    meaningful_tokens = [w for w in tokens if w not in generic_fillers]
    
    # If 0 meaningful words left -> It was 100% fluff -> Low Signal
    if len(meaningful_tokens) == 0:
        return True
        
    # If only 1 meaningful word left and it's tiny -> Likely noise
    if len(meaningful_tokens) == 1 and len(meaningful_tokens[0]) <= 2:
        return True

    # Otherwise, it passed all checks -> High Signal
    return False

# Output would confuse between 'supplement' and 'baby' tags; so we use keyword rules to adjust
def adjust_product_inquiry_tags(full_context: str, tags: set[str]) -> set[str]:
    """
    LEgacy Heuristic for 'Product Inquiry ...'.
    We now have:
        - product_Features + product_matcher
        - shared CANONICAL_cATEGORY_RULES (config/tag_rules)
        - ML Fall back
    """

    # Just return tags unchanged
    return set(tags) if tags is not None else set()


def extract_message_text(raw):
    """
    Respond.io `Content` often looks like:
      {"type":"text","text":"hello"}
    or
      {"type":"attachment","attachment":{...}}

    This extracts a human-readable text string.
    """
    if raw is None:
        return ""
    if not isinstance(raw, str):
        raw = str(raw)
    raw = raw.strip()
    if not raw:
        return ""

    # Try JSON
    if raw.startswith("{") and raw.endswith("}"):
        try:
            obj = json.loads(raw)
        except Exception:
            # Not valid JSON; just return raw
            return raw

        if isinstance(obj, dict):
            t = obj.get("type")

            # Text messages
            if t == "text":
                txt = obj.get("text", "")
                if isinstance(txt, dict):
                    txt = txt.get("body", "") or txt.get("text", "")
                return str(txt)

            # Attachments (image/file/etc.)
            if t == "attachment":
                att = obj.get("attachment", {}) or {}
                caption = att.get("caption") or att.get("text") or ""
                if caption:
                    return str(caption)
                att_type = att.get("type", "attachment")
                file_name = att.get("fileName") or ""
                if file_name:
                    return f"[{att_type} {file_name}]"
                return f"[{att_type} attachment]"

        # Unexpected JSON structure -> return original
        return raw

    # Not JSON, return as-is
    return raw

def has_any_regex(patterns, text):
    return any(re.search(p, text) for p in patterns)


import re

# Simple mapping from keywords in text -> concern labels
TEXT_KEYWORD_TO_CONCERNS = {
    # Acne
    "acne": {"Acne"},
    "pimple": {"Acne"},
    "pimples": {"Acne"},
    "breakout": {"Acne"},
    "breakouts": {"Acne"},
    "whitehead": {"Acne"},
    "whiteheads": {"Acne"},
    "blackhead": {"Acne"},
    "blackheads": {"Acne"},
    "spots": {"Acne"},  # be careful, can be noisy but fine to start

    # Hyperpigmentation
    "dark spot": {"Hyperpigmentation"},
    "dark spots": {"Hyperpigmentation"},
    "hyperpigmentation": {"Hyperpigmentation"},
    "dark mark": {"Hyperpigmentation"},
    "dark marks": {"Hyperpigmentation"},
    "uneven tone": {"Hyperpigmentation"},
    "uneven skin tone": {"Hyperpigmentation"},

    # Sleep
    "insomnia": {"Sleep"},
    "sleep": {"Sleep"},
    "sleeping": {"Sleep"},
    "restless night": {"Sleep"},
    "restless nights": {"Sleep"},
    "can't sleep": {"Sleep"},
    "cant sleep": {"Sleep"},
}


def infer_concerns_from_text(text: str) -> set[str]:
    """
    Scan a text blob (e.g. session full_context) for concern keywords
    and return a set of concern labels like {"Acne", "Hyperpigmentation"}.
    """
    if not isinstance(text, str):
        return set()

    t = text.lower()
    # Normalise whitespace
    t = re.sub(r"\s+", " ", t)

    concerns = set()
    for kw, con_set in TEXT_KEYWORD_TO_CONCERNS.items():
        if kw in t:
            concerns.update(con_set)

    return concerns

def infer_conc_categories_from_text(text: str) -> set[str]:
    '''
    Infer canonical "Product Inquiry - ..." categories from chat text, using
    shared CANONICAL_CATEGORY_RULES (config.tag_rules)
    '''
    cats = enrich_canonical_categories_from_text(
        text=text or "",
        existing=set(),
        source="chat"
    )
    return cats