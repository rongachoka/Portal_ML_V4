# Portal_ML_V4/src/core/mpesa_engine.py
import re
from Portal_ML_V4.src.utils.text_cleaner import normalize_text_simple

# --- REGEX DEFINITIONS ---
# Matches 10-12 character alphanumeric transaction codes
TX_CODE_RE = re.compile(r"\b([A-Za-z0-9]{10,12})\b", re.IGNORECASE)
# Matches currency markers followed by numbers
CURRENCY_RE = re.compile(r"(?:ksh|kes|kshs|shs|sh)\.?\s*([\d][\d\s,\.]*)", re.IGNORECASE)
# Matches numbers ending with the /- suffix (e.g., 2950/-)
CENTS_SUFFIX_RE = re.compile(r"\b(\d{2,6})\s*/-") 

# --- KEYWORD BANKS ---
MPESA_INSTRUCTION_KEYWORDS = [
    "paybill", "till", "use till", "acc no", "please pay", "pay using",
    "send to", "account no", "account number"
]

# Restored: Full Un-simplified Keyword Bank
MPESA_CONFIRM_KEYWORDS = [
    "transaction confirmed", "payment received",
    "sent to equity paybill account",
    "paid to pharmart galleria chemist",
    "paid to centurion pharmacy",
    "paid to portal pharmacy", 
    "paid to pharmart abc chemist",
    "for account 666222", # Ngong Milele
    "paid to", "received", "well received", "payment received",
    "received with thanks"
]

def _parse_ksh_amount(num_str: str):
    """Refined parser: stops at the first space or punctuation after digits."""
    if not num_str: return 0.0
    # Replace non-breaking spaces and commas
    clean = num_str.replace('\xa0', ' ').replace(',', '')
    # Extract only the first contiguous number found
    match = re.search(r"(\d+(\.\d{1,2})?)", clean)
    if not match: return 0.0
    
    try:
        val = float(match.group(1))
        return val if 10 <= val <= 1000000 else 0.0
    except: return 0.0

def detect_payment_converted_v2(text: str) -> dict:
    t = normalize_text_simple(text)
    low = t.lower()
    tx_codes = []
    found_amounts = []

    # 1. Mask Codes
    text_masked = t
    for m in TX_CODE_RE.finditer(t):
        c = m.group(1)
        if any(char.isdigit() for char in c) and any(char.isalpha() for char in c):
            tx_codes.append(c)
            text_masked = text_masked.replace(c, "XXXXXXXXXX")

    # 2. Extract Currency Candidates
    for m in CURRENCY_RE.finditer(text_masked.lower()):
        val = _parse_ksh_amount(m.group(1))
        if val > 0: found_amounts.append(val)

    for m in CENTS_SUFFIX_RE.finditer(text_masked.lower()):
        val = _parse_ksh_amount(m.group(1))
        if val > 0: found_amounts.append(val)

    # 3. PRIORITY LOGIC: Confirmation SMS vs Conversation
    has_confirm = any(k in low for k in MPESA_CONFIRM_KEYWORDS)
    
    # Check for official SMS patterns first (Highest Priority)
    # Pattern: "Ksh8,950.00 paid to" or "Ksh6,600.00 sent to"
    sms_pattern = re.search(r"(?:ksh|kes)\.?\s*([\d,]+\.\d{2})\s*(?:paid|sent|confirmed)", low)
    if sms_pattern:
        final_amount = _parse_ksh_amount(sms_pattern.group(1))
    elif found_amounts and has_confirm:
        # If no SMS found, pick the MAX to avoid repetitions (920 920 -> 920)
        final_amount = max(found_amounts)
    else:
        final_amount = 0.0

    return {
        "is_converted": final_amount > 0 or len(tx_codes) > 0,
        "amount": final_amount,
        "tx_code": tx_codes,
        "is_instruction": "paybill" in low and not has_confirm
    }