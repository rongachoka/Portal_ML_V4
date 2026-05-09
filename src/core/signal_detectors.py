# src/core/signal_detectors.py
import re
from typing import Set, List
from Portal_ML_V4.src.config.tag_rules import extract_concerns_from_text
from Portal_ML_V4.src.config.brands import BRAND_LIST

# Dynamic Imports for Zones
try:
    from src.config.zones import ZONE_MAPPING
except ImportError:
    ZONE_MAPPING = {}

KNOWN_BRANDS = set(b.lower() for b in BRAND_LIST)

def extract_locations_zones(text: str) -> List[str]:
    """Extract Pricing Zones."""
    if not text or not ZONE_MAPPING: return []
    found_zones = set()
    t_lower = text.lower()
    for zone_key, official_name in ZONE_MAPPING.items():
        if zone_key in t_lower:
            if re.search(r"\b" + re.escape(zone_key) + r"\b", t_lower):
                found_zones.add(official_name)
    return list(found_zones)

def detect_brands(text: str) -> Set[str]:
    """Explicitly finds brands in text."""
    if not text: return set()
    t_lower = text.lower()
    found = set()
    for brand in KNOWN_BRANDS:
        if brand in t_lower:
            if re.search(r"\b" + re.escape(brand) + r"\b", t_lower):
                found.add(brand.title()) 
    return found

def detect_price_quote(text: str) -> bool:
    if not text: return False
    pat = r'(ksh|kes|shs|sh)\.?\s?[\d,]{2,}|[\d,]{2,}\s?(ksh|shs|/=|kes)'
    return bool(re.search(pat, str(text), re.IGNORECASE))

def detect_price_objection(text: str) -> bool:
    if not text: return False
    triggers = ["too expensive", "high price", "costly", "last price", "best price", "discount", "reduce", "budget", "lower"]
    return any(trig in str(text).lower() for trig in triggers)

def infer_concerns_from_text(text: str) -> Set[str]:
    return extract_concerns_from_text(text, source="chat")

def _normalise_tags(value) -> List[str]:
    if not value: return []
    if isinstance(value, str): 
        tags = [p.strip() for p in value.split("|") if p.strip()]
    else: 
        tags = list(value)
    seen = set()
    clean = []
    for t in tags:
        t = t.strip()
        if t and t not in seen:
            seen.add(t)
            clean.append(t)
    return clean