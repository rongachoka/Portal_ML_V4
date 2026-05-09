
# Imports
import re

# categories mapping used for cross-encoder labels
ML_LABELS = [
    "skincare",
    "haircare",
    "perfume",
    "supplement",
    "baby",
    "lip",
    "oral",
    "men",
    "first_aid",
    "medicine",
    "medical_device",
    "others",
    "lady"
]
ML_TO_RESP = {
    "skincare": "Product Inquiry - Skincare",
    "haircare": "Product Inquiry - Haircare",
    "perfume": "Product Inquiry - Perfumes",
    "supplement": "Product Inquiry - Supplements",
    "baby": "Product Inquiry - Baby Care",
    "lip": "Product Inquiry - Lip Care",
    "oral": "Product Inquiry - Oral Care",
    "men": "Product Inquiry - Men Care",
    "first_aid": "Product Inquiry - First Aid",
    "medicine": "Product Inquiry - Medicine",
    "medical_device": "Product Inquiry - Medical Devices and Kits",
    "lady": "Product Inquiry - Women's Health",
    "others": "Product Inquiry - Others",
}

# System patterns to remove
SYSTEM_PATTERNS = [
    "conversation opened",
    "conversation closed",
    "assigned to",
    "workflow",
    "tag added",
    "tag removed",
    "bot started",
    "bot ended",
    "delivered",
    "read",
    "subscribed",
    "channel linked",
    "lifecycle stage",
    "lifecycle stage new lead updated",
    "lifecycle stage to be delivered updated",
    "workflow send capi for converted purchases started",
    "workflow send capi for converted purchases ended",
    "workflow send capi for converted leads started",
    "workflow send capi for converted leads ended",
    "workflow get channel ids started",
    "workflow get channel ids ended",
    "new lead added",
    "to be delivered by",
    "product delivered by",
]

MPESA_CONFIRM_KEYWORDS = [
    # very generic confirmation words
    "confirmed",                      # e.g. "tgm72krbon confirmed ksh2 700 00..."
    "transaction confirmed",
    "payment received",
    "payment well received",
    "you have received",
    "you have paid",
    "you have sent",
    "has been received",
    "has been paid",
    "has been sent",
    "dispathed"

    # phrases around recipient / movement
    "paid to",
    "sent to",                        # past tense = confirmation in mpesa/bank sms
    "credited to",
    "debited from",

    # success phrases
    "your transaction is successful",
    "transaction successful",
    "successful transaction",

    # bill payments
    "bill payment to",

    # bank / your specific merchant names
    "sent to equity paybill account",
    "paid to pharmart galleria chemist",
    "paid to centurion pharmacy",
    "paid to"
]

# ----------------------------------------------------
# Instruction / request keywords
# - ALL lowercase
# - Used to identify "please pay via ..." type messages
# ----------------------------------------------------
MPESA_INSTRUCTION_KEYWORDS = [
    "paybill",
    "till number",
    "till no",
    "till",
    "use till",
    "acc no",
    "account number",
    "please pay",
    "kindly pay",
    "pay using",
    "pay via mpesa",
    "pay via m-pesa",
    "pay through mpesa",
    "use paybill",
    "mpesa number",
    "lipa na mpesa",
    "go to mpesa",
    "select lipa na mpesa",
    "enter business number",
    "enter account number",
    "send to",       # imperative (instructions) – NOT "sent to"
]


MPESA_INSTRUCTION_KEYWORDS = [
    "paybill", "till", "use till", "acc no", "please pay", "pay using",
    "send to"
]

SUPPLEMENT_KEYWORDS = [
    "vitamin", "supplement", "magnesium", "omega", "effervescent",
    "multivitamin", "collagen", "calcium", "probiotic", "iron",
    "folic"
]

SKINCARE_KEYWORDS = [
    "serum", "moisturizer", "cleanser", "toner", "exfoliant",
    "acne", "moisturiser", "sunscreen", "sun screen", "face wash",
    "retinol", "niacinamide", "hydrating"
]

BABY_KEYWORDS = [
    "baby", "infant", "diaper", "formula", "kids", "kid", "child", "children"
]

# ----------------------------------------------------
# Transaction code detection
# - Slightly more relaxed, but still alphanumeric and 10–12 chars 
# Mpesa 10 chars, bank 12 chars
# - We already post-filter in mpesa.py to require both letters and digits
# ----------------------------------------------------
TX_CODE_RE = re.compile(r"\b([A-Za-z0-9]{10,12})\b")

# ----------------------------------------------------
# Amount detection
# - Now strictly tied to a currency word: ksh, kes, kshs, shs, sh
# - We mostly use our own Ksh parser in detect_payment_converted_v2,
#   but keep this in case other modules rely on AMOUNT_RE.
# ----------------------------------------------------
AMOUNT_RE = re.compile(
    r"(?:ksh|kes|kshs|shs|sh)\.?\s*([0-9][0-9\s,\.]*)",
    re.I,
)

LOW_SIGNAL_PHRASES = {
    "hi",
    "hello",
    "hey",
    "thanks",
    "thank you",
    "ok",
    "okay",
    "welcome",
    "noted"
}

_ILLEGAL_XL_RE = re.compile(
    "[" +
    "\x00-\x08" +
    "\x0B-\x0C" +
    "\x0E-\x1f" +
    "]s"
    )
