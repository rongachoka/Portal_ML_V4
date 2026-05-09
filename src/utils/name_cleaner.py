"""
name_cleaner.py
===============
Shared utility for cleaning junk client names that come from
POS bank payment methods (Loop, Coop, NCBA) being written into
the client name field instead of the actual customer name.

Location: Portal_ML_V4/src/utils/name_cleaner.py

Usage:
    from Portal_ML_V4.src.utils.name_cleaner import clean_client_name, clean_name_series
"""

import re
import pandas as pd


# ==========================================
# 1. JUNK PATTERN DEFINITIONS
# ==========================================

# Any name containing these strings (case-insensitive) is payment method noise
BANK_PAYMENT_PATTERNS = [
    # Loop variants
    r'\bloop\b',
    r'\blooop\b',
    r'\bloop\s*b2c\b',
    r'\bloop\s*btc\b',
    r'\bncba\s*loop\b',
    r'\bloop\s*business\b',

    # Coop variants
    r'\bcoop\b',
    r'\bco-?op\b',
    r'\bcoop\s*bank\b',
    r'\bcoop\s*to\s*paybill\b',
    r'\bcoop\s*paybill\b',
    r'\bcoop\s*payabill\b',
    r'\bcoop\s*till\b',
    r'\bcoop\s*to\s*till\b',

    # NCBA variants
    r'\bncba\b',
]

# A name is junk if it contains ONLY symbols (no letters or digits)
SYMBOLS_ONLY_PATTERN = re.compile(r'^[^a-zA-Z0-9]+$')

# A name is junk if it's one of these exact strings
EXACT_JUNK_NAMES = {
    'nil', 'n/a', 'na', 'none', 'unknown', 'nan',
    '//', '**', '--', '***', '/', '\\',
    'cash', 'card',   # payment methods sometimes written as names
}

# Compile all bank patterns into one combined regex for efficiency
_BANK_PATTERN_COMPILED = re.compile(
    '|'.join(BANK_PAYMENT_PATTERNS),
    flags=re.IGNORECASE
)


# ==========================================
# 2. CORE CLEANING FUNCTION
# ==========================================

def clean_client_name(name) -> str | None:
    """
    Returns None if the name is payment-method noise or junk.
    Returns the cleaned name string if it's a real name.

    Rules applied in order:
      1. Null / empty / too short  -> None
      2. Exact junk match          -> None
      3. Symbols-only              -> None
      4. Bank payment pattern      -> None
      5. Otherwise                 -> return stripped title-cased name
    
    Examples:
        'LOOP'              -> None
        'NCBA LOOP'         -> None
        'loop btc'          -> None
        'COOP BANK'         -> None
        '//'                -> None
        '***'               -> None
        'Nil'               -> None
        'Jane Doe'          -> 'Jane Doe'
        'MARY WANJIKU'      -> 'Mary Wanjiku'
    """
    if pd.isna(name):
        return None

    s = str(name).strip()

    # 1. Empty or too short to be a real name
    if len(s) < 2:
        return None

    # 2. Exact junk match (case-insensitive)
    if s.lower() in EXACT_JUNK_NAMES:
        return None

    # 3. Symbols only — no letters or digits at all
    if SYMBOLS_ONLY_PATTERN.match(s):
        return None

    # 4. Bank / payment method pattern
    if _BANK_PATTERN_COMPILED.search(s):
        return None

    # 5. Looks like a real name — clean up formatting
    return s.strip()


# ==========================================
# 3. SERIES-LEVEL HELPER
# ==========================================

def clean_name_series(series: pd.Series) -> pd.Series:
    """
    Apply clean_client_name to an entire DataFrame column.
    Junk names are replaced with pd.NA (not empty string)
    so downstream .isna() checks work correctly.

    Usage:
        df['Client Name'] = clean_name_series(df['Client Name'])
    """
    return series.apply(clean_client_name).where(
        series.apply(clean_client_name).notna(), other=pd.NA
    )


# ==========================================
# 4. COALESCE HELPER (for client list)
# ==========================================

def resolve_best_name(primary_name, fallback_name=None) -> str:
    """
    Picks the best available name between a primary and a fallback.
    Used in run_client_export.py when the CRM name is junk but
    a name was extracted from conversation text (name_audit_flag).

    Usage:
        df['resolved_name'] = df.apply(
            lambda r: resolve_best_name(r['client_name'], r['name_audit_flag']),
            axis=1
        )
    """
    clean_primary  = clean_client_name(primary_name)
    clean_fallback = clean_client_name(fallback_name) if fallback_name is not None else None

    if clean_primary:
        return clean_primary
    if clean_fallback:
        return clean_fallback
    return "Unknown"


# ==========================================
# 5. DIAGNOSTIC HELPER
# ==========================================

def audit_junk_names(df: pd.DataFrame, name_col: str) -> pd.DataFrame:
    """
    Returns a summary DataFrame showing which junk name patterns
    appear in your data and how many times.

    Useful to run once to confirm the cleaning is working correctly.

    Usage:
        audit = audit_junk_names(df, 'Client Name')
        print(audit)
    """
    results = []
    for _, row in df.iterrows():
        raw = str(row.get(name_col, '')).strip()
        cleaned = clean_client_name(raw)
        if cleaned is None and raw not in ('', 'nan'):
            results.append({'raw_name': raw, 'reason': _classify_junk_reason(raw)})

    if not results:
        print("✅ No junk names found.")
        return pd.DataFrame()

    audit_df = (
        pd.DataFrame(results)
        .groupby(['raw_name', 'reason'])
        .size()
        .reset_index(name='count')
        .sort_values('count', ascending=False)
    )
    return audit_df


def _classify_junk_reason(name: str) -> str:
    """Internal helper — explains WHY a name was flagged as junk."""
    s = str(name).strip()
    if s.lower() in EXACT_JUNK_NAMES:
        return 'exact_junk_match'
    if SYMBOLS_ONLY_PATTERN.match(s):
        return 'symbols_only'
    if _BANK_PATTERN_COMPILED.search(s):
        # Identify which bank pattern matched
        if re.search(r'\bloop\b|\blooop\b', s, re.I):
            return 'bank_loop'
        if re.search(r'\bcoop\b|\bco-?op\b', s, re.I):
            return 'bank_coop'
        if re.search(r'\bncba\b', s, re.I):
            return 'bank_ncba'
    return 'other_junk'


# ==========================================
# 6. QUICK TEST (run directly to verify)
# ==========================================

if __name__ == "__main__":
    test_cases = [
        # Should all return None
        ("LOOP",              None),
        ("loop btc",          None),
        ("LOOP B2C",          None),
        ("NCBA LOOP",         None),
        ("loop business",     None),
        ("LOOOP",             None),
        ("COOP",              None),
        ("coop to paybill",   None),
        ("Coop bank",         None),
        ("COOP TILL",         None),
        ("coop payabill",     None),
        ("NCBA",              None),
        ("//",                None),
        ("***",               None),
        ("**",                None),
        ("Nil",               None),
        ("NIL",               None),
        ("nan",               None),
        # Should return cleaned name
        ("Jane Doe",          "Jane Doe"),
        ("Sherlock Holmes",   "Sherlock Holmes")
    ]

    print("🧪 Running name_cleaner tests...\n")
    all_passed = True
    for name, expected in test_cases:
        result = clean_client_name(name)
        status = "✅" if result == expected else "❌"
        if result != expected:
            all_passed = False
        print(f"  {status}  Input: {repr(name):<30} Expected: {repr(expected):<15} Got: {repr(result)}")

    print(f"\n{'✅ All tests passed.' if all_passed else '❌ Some tests failed — check above.'}")