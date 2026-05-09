import re
import hashlib

# --- REGEX DEFINITIONS ---
# 1. Standard M-Pesa Code (10 alphanumeric characters)
TX_CODE_RE = re.compile(r"\b([A-Z0-9]{10})\b", re.IGNORECASE)

# 2. Strict Money Pattern (Ksh 1,000.00)
CURRENCY_RE = re.compile(r"(?:ksh|kes|ush|tzs)\.?\s*([\d,]+\.?\d*)", re.IGNORECASE)

# --- KEYWORD BANKS ---
MPESA_CONFIRM_KEYWORDS = [
    "payment received", "amount received", "payment well received", 
    "amount is well received", "amount well received",
    "received with thanks", "well received"
]

def normalize_text_simple(text: str) -> str:
    if not isinstance(text, str): return ""
    return re.sub(r'\s+', ' ', str(text)).strip()

def detect_payment_converted_v2(text: str) -> dict:
    """
    Extracts M-Pesa payments while ignoring costs, balances, and duplicate messages.
    Returns: {'is_converted': bool, 'amount': float, 'tx_code': list}
    """
    t = normalize_text_simple(text)
    low = t.lower()
    
    # ==========================================================
    # 1. AGGRESSIVE CLEANING (The "Masking" Phase)
    # ==========================================================
    # We replace these patterns with empty space so they are never detected as revenue.
    
    noise_patterns = [
        # Mask Account Numbers (Avoids "Account 217004" being read as 217,004 KES)
        r"(?:account|acc)\s*(?:no|number|num)?[:\.]?\s*\d+",
        r"paybill\s*(?:no|number|num)?[:\.]?\s*\d+",

        # Transactin charge (Handles "Transaction cost, Ksh20.00")
        r"transaction\s*(?:cost|charge)\s*[:.,-]?\s*(?:ksh|kes|kshs)?\.?\s*[\d,]+(?:\.\d{2})?",
        
        # Mask Balances (Handles "New M-PESA balance is Ksh342.93")
        r"(?:new)?\s*m-?pesa\s*balance\s*(?:is|:)?\s*(?:ksh|kes)?\.?\s*[\d,]+(?:\.\d{2})?",
        
        # Mask Daily Limits (Handles "Amount you can transact within the day is...")
        r"amount\s*you\s*can\s*transact.*?(?:ksh|kes)?\.?\s*[\d,]+(?:\.\d{2})?",
        
        # Alerrts
        r"(?:sms\s*)?alert\s*charge\s*(?:ksh|kes|kshs)?\.?\s*[\d,]+(?:\.\d{2})?",
    ]
    
    clean_text = low
    for pattern in noise_patterns:
        clean_text = re.sub(pattern, " [NOISE] ", clean_text, flags=re.I)

    # ==========================================================
    # 2. DICTIONARY EXTRACTION (Deduplication)
    # ==========================================================
    # We store found amounts in a dict {TxCode: Amount}. 
    # If the same code appears twice in history, it overwrites (counts once).
    
    found_payments = {}

    # PATTERN A: Standard SMS 
    # "TL3C1BVSMD Confirmed. Ksh1,650.00 sent to..."
    # We allow flexible spacing/punctuation between Code and Confirmed
    pattern_standard = re.compile(
        r"([A-Z0-9]{10})\s+.*confirmed.*?(?:ksh|kes)\.?\s*([\d,]+\.\d{2})", 
        re.I
    )
    
    # PATTERN B: Reverse Style 
    # "Ksh 1,000 sent to... Ref: QWE123456"
    pattern_reverse = re.compile(
        r"(?:ksh|kes)\.?\s*([\d,]+\.\d{2}).*?ref[:\.]?\s*([A-Z0-9]{10})", 
        re.I
    )

    # Execute Regex
    for m in pattern_standard.finditer(clean_text):
        try:
            code = m.group(1).upper()
            amt = float(m.group(2).replace(',', ''))
            found_payments[code] = amt
        except: pass

    for m in pattern_reverse.finditer(clean_text):
        try:
            code = m.group(2).upper()
            amt = float(m.group(1).replace(',', ''))
            found_payments[code] = amt
        except: pass

    # ===================
    # EQUITY PAYMENTS
    # ===================
    pattern_equity = re.compile(
        r"confirmed\s+k?sh[s]?\.?\s*([\d,]+\.?\d*)\s+sent\s+to.*?ref[:\.]?\s*([A-Z0-9]{10,15})",
        re.I
    )
    for m in pattern_equity.finditer(clean_text):
        try:
            code = m.group(2).upper()
            amt = float(m.group(1).replace(',', ''))
            found_payments[code] = amt
        except: pass


    # ==========================================================
    # 3. FALLBACK: CODE-LESS CONFIRMATION (Manual Entry)
    # ==========================================================
    # Only runs if NO valid M-Pesa code was found.
    # Checks for: "Payment received" + "Ksh 2500" nearby
    
    if not found_payments:
        # A. Check for Strong Confirmation Keywords
        is_confirmed = any(k in clean_text for k in MPESA_CONFIRM_KEYWORDS)
        
        # B. Find Candidates for Amount (ignoring the masked noise)
        amounts = re.findall(r"(?:ksh|kes|amount)\.?\s*([\d,]+\.?\d*)", clean_text)
        
        if is_confirmed and amounts:
            try:
                # We grab the LAST valid amount mentioned 
                # (Agent: "Total is 3500?" -> Cust: "Paid 2500" -> Agent: "Received")
                valid_amts = []
                for a in amounts:
                    try:
                        val = float(a.replace(',', ''))
                        # Sanity Check: Revenue > 50 and < 300k. Avoid years like 2025.
                        if 50 < val < 300000 and val not in [2024, 2025, 2026]:
                            valid_amts.append(val)
                    except: continue
                
                if valid_amts:
                    final_amt = max(valid_amts) # Take largest amount — avoids capturing fees like SMS charges
                    
                    # Create a dummy hash so this "manual" payment is unique-ish
                    # but deduplicates if the EXACT SAME text appears twice.
                    dummy_code = "MANUAL_" + hashlib.md5(clean_text.encode()).hexdigest()[:8].upper()
                    found_payments[dummy_code] = final_amt
            except: pass

    # ==========================================================
    # 4. FINAL AGGREGATION
    # ==========================================================
    final_amount = sum(found_payments.values())
    tx_codes = list(found_payments.keys())

    return {
        "is_converted": final_amount > 0,
        "amount": final_amount,
        "tx_code": tx_codes,
        "is_instruction": "paybill" in low and final_amount == 0
    }