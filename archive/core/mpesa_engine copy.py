import re
from Portal_ML_V4.src.utils.text_cleaner import normalize_text_simple

# --- REGEX DEFINITIONS ---
TX_CODE_RE = re.compile(r"\b([A-Z0-9]{10,12})\b", re.IGNORECASE)
CURRENCY_RE = re.compile(r"(?:ksh|kes|kshs|shs|sh)\.?\s*([\d][\d\s,\.]*)", re.IGNORECASE)
CENTS_SUFFIX_RE = re.compile(r"\b(\d{2,6})\s*/-")  

# --- KEYWORD BANKS ---
MPESA_INSTRUCTION_KEYWORDS = [
    "paybill", "till", "use till", "acc no", "please pay", "pay using",
    "send to", "account no", "account number"
]

MPESA_CONFIRM_KEYWORDS = [
    "transaction confirmed", "payment received", "payment well received",
    "sent to equity paybill account", "paid to pharmart galleria chemist",
    "paid to centurion pharmacy", "paid to portal pharmacy", 
    "paid to pharmart abc chemist", "for account 666222", 
    "received with thanks"
]

def _parse_ksh_amount(num_str: str):
    if not num_str: return 0.0
    clean = num_str.replace('\xa0', ' ').replace(',', '').strip()
    match = re.search(r"(\d+(\.\d{1,2})?)", clean)
    if not match: return 0.0
    try:
        val = float(match.group(1))
        return val if 1.0 <= val <= 1000000 else 0.0
    except: return 0.0

def detect_payment_converted_v2(text: str) -> dict:
    t = normalize_text_simple(text)
    low = t.lower()
    
    # --- 1. CONFIGURATION: BUSINESS HANDLE MAPPING ---
    # UPDATED: Added '666226' (3 sixes) to match your text example
    PAYBILL_MAP = {
        "247247": ["666226", "662226", "217004", "666222"], # Equity Paybill accounts
        "552800": ["222666"] # Other Paybills
    }
    # Tills: Added generic ones if needed
    TILL_NUMBERS = ["894353", "727721", "633450"] 
    
    # --- 2. NOISE MASKING (Safeguard) ---
    noise_patterns = [
        r"new m-pesa balance is\s*(?:ksh|kes)?\s*[\d,.]+",
        r"transaction cost,\s*(?:ksh|kes)?\s*[\d,.]+",
        r"amount you can transact within the day is\s*[\d,.]+"
    ]
    clean_text = low
    for pattern in noise_patterns:
        clean_text = re.sub(pattern, "SAFEGUARD_MASK", clean_text, flags=re.I)

    # --- 3. ANCHORED SMS EXTRACTION (Priority) ---
    found_payments = {}
    pattern_a = re.compile(r"([A-Z0-9]{10,12})\s+confirmed\.\s*(?:ksh|kes)\.?\s*([\d,]+\.\d{2})", re.I)
    pattern_b = re.compile(r"(?:ksh|kes)\.?\s*([\d,]+\.\d{2}).*?ref\.\s*([A-Z0-9]{10,12})", re.I)

    for m in pattern_a.finditer(clean_text):
        found_payments[m.group(1).upper()] = float(m.group(2).replace(',', ''))
    for m in pattern_b.finditer(clean_text):
        found_payments[m.group(2).upper()] = float(m.group(1).replace(',', ''))

    final_amount = sum(found_payments.values())
    tx_codes = list(found_payments.keys())

    # --- 4. STRICT FALLBACK (The Fix for Riverside) ---
    if final_amount == 0:
        # A. Check for Business Handles (Paybill/Account/Till)
        valid_handle = False
        for pb, accounts in PAYBILL_MAP.items():
            if pb in low and any(acc in low for acc in accounts):
                valid_handle = True
                break
        
        if not valid_handle:
            valid_handle = any(till in low for till in TILL_NUMBERS)

        # B. Check for Confirmation Signals
        # We look for "Paid" or "Well received"
        confirmed_signal = any(k in low for k in ["payment received", "received with thanks",
                                                  "payment well received", "transaction confirmed",
                                                  "well received"])

        # C. Global Amount Search
        # If we found the correct Account AND a Confirmation, we scan the whole text for the price
        if valid_handle and confirmed_signal:
            # Matches: "amount kes 12000" or "kes 12000" anywhere in the text
            # This regex allows the amount to be BEFORE the confirmation keyword
            amounts = re.findall(r"(?:amount|total|kes|ksh)\.?\s*([0-9,]{3,}\.?\d*)", clean_text)
            
            if amounts:
                try:
                    # We usually trust the LAST mentioned amount in a negotiation/instruction
                    amt_str = amounts[-1].replace(',', '').strip()
                    val = float(amt_str)
                    
                    # Range Check: 100 to 1M to avoid grabbing phone numbers/dates
                    if 100 <= val <= 1000000:
                        final_amount = val
                except ValueError:
                    pass

    return {
        "is_converted": final_amount > 0,
        "amount": final_amount,
        "tx_code": tx_codes,
        "is_instruction": "paybill" in low and final_amount == 0
    }