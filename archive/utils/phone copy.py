"""
Portal_ML_V4/src/utils/phone.py

Single source of truth for all phone number normalization across the project.

Import with:
    from Portal_ML_V4.src.utils.phone import clean_id, normalize_phone, clean_id_excel_safe
"""

import re

# ── COUNTRY CODE LOOKUP ───────────────────────────────────────────────────────
# Add new countries here only — nothing else needs to change
# Format: prefix → (country_name, expected_total_digits, format_function)

def _fmt_us(d):
    return f"+1 {d[1:4]} {d[4:7]} {d[7:]}"       # +1 212 555 0123

def _fmt_uk(d):
    return f"+44 {d[2:6]} {d[6:9]} {d[9:]}"       # +44 7700 900 982

COUNTRY_CODES = {
    "44": ("UK",        12, _fmt_uk),
    "1":  ("US/Canada", 11, _fmt_us),
    # Add more as needed:
    # "971": ("UAE",   12, _fmt_generic),
    # "966": ("Saudi", 12, _fmt_generic),
}


def clean_id(val):
    """
    For Contact IDs only — strips non-digits and .0 suffix.
    Simple pass-through, no phone-specific logic.
    """
    if val is None: return None
    s = str(val).strip().replace('.0', '')
    s = ''.join(filter(str.isdigit, s))
    if len(s) == 0: return None
    return s  # return as-is — no phone pattern matching


def normalize_phone(val):
    """
    For phone numbers only — full country-aware normalization.
    Use this everywhere a phone number needs to be matched or stored.
    """
    if val is None:
        return None
    s = str(val).strip().replace('.0', '')
    s = ''.join(filter(str.isdigit, s))
    if len(s) == 0:
        return None

    # Kenyan
    if s.startswith('254') and len(s) == 12:
        return s[-9:]
    if s.startswith('0') and len(s) == 10:
        return s[-9:]
    if len(s) == 9 and s.startswith(('7', '1')):
        return s

    # Non-Kenyan
    if len(s) >= 11 and not s.startswith('254'):
        for prefix in ["44", "1"]:
            if s.startswith(prefix):
                _, expected_len, fmt_fn = COUNTRY_CODES[prefix]
                if len(s) == expected_len:
                    return fmt_fn(s)
                elif len(s) > expected_len:
                    return f"INVALID:{s}"
        if len(s) > 13:
            return f"INVALID:{s}"
        return f"+{s}"

    return None  # genuinely unrecognizable


def clean_id_excel_safe(val):
    result = normalize_phone(val)
    if result is None: 
        return None
    if result.startswith('INVALID') or result.startswith('+'): 
        return result
    return f"'{result}"

def is_valid_phone(val):
    """
    Returns True if val can be normalized to a valid phone number.
    Returns False for None, empty, unparseable, or INVALID: numbers.
    """
    cleaned = normalize_phone(val)
    return cleaned is not None and not str(cleaned).startswith('INVALID:')