"""
Quick test for clean_id phone normalisation.
Run with: python test_clean_id.py
No dependencies on the rest of the project.
"""

# ── COUNTRY CODE LOOKUP ───────────────────────────────────────────────────────
def fmt_us(d):
    # +1 212 555 0123  (1 + 3 + 3 + 4)
    return f"+1{d[1:4]}{d[4:7]}{d[7:]}"

def fmt_uk(d):
    # +44 7700 900 982  (44 + 4 + 3 + 3)
    return f"+44{d[2:6]}{d[6:9]}{d[9:]}"

COUNTRY_CODES = {
    "44": ("UK",        12, fmt_uk),
    "1":  ("US/Canada", 11, fmt_us),
}

def clean_id(val):
    if val is None: return None
    s = str(val).strip().replace('.0', '')
    s = ''.join(filter(str.isdigit, s))
    if len(s) == 0: return None

    # ── KENYAN — normalise to 9 digits ────────────────────────────────────────
    if s.startswith('254') and len(s) == 12:
        return s[-9:]
    if s.startswith('0') and len(s) == 10:
        return s[-9:]
    if len(s) == 9 and s.startswith(('7', '1')):
        return s

    # ── NON-KENYAN — check country code lookup ────────────────────────────────
    if len(s) >= 11 and not s.startswith('254'):
        # Try 2-digit codes before 1-digit to avoid '1' matching '44' numbers
        for prefix in ["44", "1"]:
            if s.startswith(prefix):
                name, expected_len, fmt_fn = COUNTRY_CODES[prefix]
                if len(s) == expected_len:
                    return fmt_fn(s)        # well-formed → format nicely
                elif len(s) > expected_len:
                    return f"INVALID:{s}"   # too long → flag it
                # too short → fall through

        # Generic foreign number — no specific formatter
        if len(s) > 13:
            return f"INVALID:{s}"
        return f"+{s}"

    return s


# ── TEST CASES ────────────────────────────────────────────────────────────────
tests = [
    # (input,                 expected,                   description)
    ("'0722000000",           "722000000",                "Kenyan local with apostrophe"),
    ("'722000000",            "722000000",                "Kenyan 9-digit with apostrophe"),
    ("'254722000000",         "722000000",                "Kenyan +254 with apostrophe"),
    ("+254722000000",         "722000000",                "Kenyan +254 with plus sign"),
    ("254722000000",          "722000000",                "Kenyan 254 no prefix"),
    ("722000000",             "722000000",                "Kenyan 9-digit clean"),
    ("0722000000",            "722000000",                "Kenyan 10-digit local"),
    # UK
    ("'+447700900982",        "+447700900982",         "UK with apostrophe"),
    ("447700900982",          "+447700900982",         "UK clean"),
    ("'+447700900982.0",      "+447700900982",         "UK with .0 and apostrophe"),
    ("'+4477911123456.0",     "INVALID:4477911123456",    "UK malformed — too long"),
    # US
    ("'+12125550123",         "+12125550123",          "US with apostrophe"),
    ("12125550123",           "+12125550123",          "US clean"),
    ("'+12125550123.0",       "+12125550123",          "US with .0 and apostrophe"),
    # Tanzania / other
    ("255712440805",          "+255712440805",            "Tanzania"),
    ("'255712440805",         "+255712440805",            "Tanzania with apostrophe"),
    ("'255712440805.0",       "+255712440805",            "Tanzania with .0 and apostrophe"),
    ("33637814944",           "+33637814944",             "France"),
    # Edge cases
    (None,                    None,                       "None input"),
    ("",                      None,                       "Empty string"),
    ("nan",                   None,                       "String nan"),
]

# ── RUN ───────────────────────────────────────────────────────────────────────
SEP = "-" * 80
print(f"\n{'='*80}")
print(f"  clean_id() — Phone Normalisation Test")
print(f"{'='*80}")
print(f"  {'Input':<26}  {'Expected':<28}  {'Got':<28}  Status")
print(SEP)

passed = 0
failed = 0

for val, expected, desc in tests:
    result = clean_id(val)
    ok = result == expected
    status = "✅ PASS" if ok else "❌ FAIL"
    if ok: passed += 1
    else:  failed += 1
    print(f"  {str(val):<26}  {str(expected):<28}  {str(result):<28}  {status}  ({desc})")

print(SEP)
print(f"  {passed} passed  |  {failed} failed  |  {len(tests)} total")
print(f"{'='*80}\n")