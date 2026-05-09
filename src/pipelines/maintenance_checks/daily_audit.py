"""
daily_match_audit.py
====================
Takes a single day and tries to match converted sessions from
fact_sessions_enriched to social_sales_direct using three signals:
  1. Phone number (exact)
  2. Amount proximity (within 10%)
  3. Context keyword overlap (brand/product mentions)

Helps diagnose why the overall match rate is low.
"""

import pandas as pd
import re
from pathlib import Path

# ── PATHS ─────────────────────────────────────────────────────────────────
BASE_DIR      = Path(r"D:\Documents\Portal ML Analys\Portal_ML\Portal_ML_V4")
SESSIONS_PATH = BASE_DIR / "data" / "03_processed" / "fact_sessions_enriched.csv"
SOCIAL_PATH   = BASE_DIR / "data" / "03_processed" / "sales_attribution" / "social_sales_direct.csv"
CONTACTS_PATH = BASE_DIR / "data" / "02_interim" / "cleaned_contacts.csv"

AUDIT_DATE    = pd.Timestamp("2026-04-20")  # change to any day you want to inspect
AMOUNT_TOL    = 0.10                        # 10% price tolerance
# ──────────────────────────────────────────────────────────────────────────

def normalize_phone(val):
    if pd.isna(val): return None
    s = str(val).strip().replace('.0', '').lstrip("'")
    digits = ''.join(filter(str.isdigit, s))
    return digits[-9:] if len(digits) >= 9 else None

def normalize_cid(series):
    return pd.to_numeric(
        series.astype(str).str.strip().str.replace(r'\.0$', '', regex=True),
        errors='coerce'
    ).astype('Int64')

# ── 1. Load sessions for that day ─────────────────────────────────────────
print(f"📂 Loading sessions for {AUDIT_DATE.date()}...")
df_sess = pd.read_csv(SESSIONS_PATH, low_memory=False)
df_sess['session_start'] = pd.to_datetime(df_sess['session_start'], errors='coerce')
df_sess['Contact ID'] = normalize_cid(df_sess['Contact ID'])
df_sess['mpesa_amount'] = pd.to_numeric(df_sess['mpesa_amount'], errors='coerce').fillna(0)

day_mask = df_sess['session_start'].dt.date == AUDIT_DATE.date()
df_day = df_sess[day_mask].copy()
df_converted = df_day[df_day['mpesa_amount'] > 0].copy()

print(f"   All sessions on day       : {len(df_day):,}")
print(f"   Converted (mpesa > 0)     : {len(df_converted):,}")
print(f"   Non-converted             : {len(df_day) - len(df_converted):,}")

# ── 2. Bridge Contact ID → phone ──────────────────────────────────────────
df_cont = pd.read_csv(CONTACTS_PATH, low_memory=False)
df_cont['ContactID'] = normalize_cid(df_cont['ContactID'])
df_cont['norm_phone'] = df_cont['PhoneNumber'].apply(normalize_phone)
phone_map = df_cont.dropna(subset=['norm_phone']).set_index('ContactID')['norm_phone'].to_dict()

df_converted['norm_phone'] = df_converted['Contact ID'].map(phone_map)
has_phone = df_converted['norm_phone'].notna().sum()
print(f"\n   Converted with phone      : {has_phone:,} ({has_phone/max(len(df_converted),1)*100:.1f}%)")
print(f"   Converted no phone        : {len(df_converted) - has_phone:,}")

# ── 3. Load social_sales_direct for same day ──────────────────────────────
print(f"\n📂 Loading social_sales_direct for {AUDIT_DATE.date()}...")
df_social = pd.read_csv(SOCIAL_PATH, low_memory=False)
df_social['Sale_Date'] = pd.to_datetime(df_social['Sale_Date'], errors='coerce')
df_social['Total (Tax Ex)'] = pd.to_numeric(df_social['Total (Tax Ex)'], errors='coerce').fillna(0)
df_social['norm_phone'] = df_social['Phone Number'].apply(normalize_phone)

df_social_day = df_social[df_social['Sale_Date'].dt.date == AUDIT_DATE.date()].copy()

# Transaction-level (not line-item) for matching
df_social_txns = (
    df_social_day.groupby('Transaction ID').agg(
        norm_phone    = ('norm_phone', 'first'),
        total_amount  = ('Total (Tax Ex)', 'sum'),
        descriptions  = ('Description', lambda x: ' '.join(x.astype(str)).lower())
    ).reset_index()
)

print(f"   Social sales transactions  : {df_social_txns['Transaction ID'].nunique():,}")
print(f"   Social sales line items    : {len(df_social_day):,}")
print(f"   Social txns with phone     : {df_social_txns['norm_phone'].notna().sum():,}")

# ── 4. Match converted sessions to social sales ───────────────────────────
print(f"\n🔍 Matching converted sessions to social sales...")

results = []
for _, sess in df_converted.iterrows():
    sess_phone  = sess['norm_phone']
    sess_amount = sess['mpesa_amount']
    sess_brand  = str(sess.get('matched_brand', '')).lower()
    sess_product = str(sess.get('matched_product', '')).lower()

    match_type  = 'No Match'
    matched_txn = None

    # Signal 1: phone match
    if pd.notna(sess_phone):
        phone_matches = df_social_txns[df_social_txns['norm_phone'] == sess_phone]
        if not phone_matches.empty:
            match_type  = 'Phone'
            matched_txn = phone_matches.iloc[0]['Transaction ID']

    # Signal 2: amount proximity (if no phone match)
    if match_type == 'No Match' and sess_amount > 0:
        amt_mask = df_social_txns['total_amount'].between(
            sess_amount * (1 - AMOUNT_TOL),
            sess_amount * (1 + AMOUNT_TOL)
        )
        amt_matches = df_social_txns[amt_mask]
        if len(amt_matches) == 1:  # only accept if unambiguous
            match_type  = 'Amount'
            matched_txn = amt_matches.iloc[0]['Transaction ID']
        elif len(amt_matches) > 1:
            # Signal 3: amount + context keyword overlap
            for _, txn in amt_matches.iterrows():
                desc = txn['descriptions']
                if (sess_brand and sess_brand not in ('unknown', '') and sess_brand in desc) or \
                   (sess_product and sess_product not in ('unknown', '') and
                    any(w in desc for w in sess_product.split() if len(w) > 3)):
                    match_type  = 'Amount + Context'
                    matched_txn = txn['Transaction ID']
                    break

    results.append({
        'session_id':       sess['session_id'],
        'Contact ID':       sess['Contact ID'],
        'mpesa_amount':     sess_amount,
        'has_phone':        pd.notna(sess_phone),
        'acquisition_source': sess.get('acquisition_source', ''),
        'match_type':       match_type,
        'matched_txn':      matched_txn,
    })

df_results = pd.DataFrame(results)

# ── 5. Summary ────────────────────────────────────────────────────────────
total_conv = len(df_results)
print(f"\n{'=' * 60}")
print(f"  DAILY MATCH AUDIT — {AUDIT_DATE.date()}")
print(f"  Converted sessions: {total_conv:,}")
print(f"{'=' * 60}")

print(f"\n── Match breakdown ───────────────────────────────────")
for mt, cnt in df_results['match_type'].value_counts().items():
    print(f"   {mt:<25} {cnt:>4,}  ({cnt/total_conv*100:.1f}%)")

matched_total = (df_results['match_type'] != 'No Match').sum()
print(f"\n   Total matched             : {matched_total:,}  ({matched_total/total_conv*100:.1f}%)")
print(f"   Total unmatched           : {total_conv - matched_total:,}  ({(total_conv-matched_total)/total_conv*100:.1f}%)")

print(f"\n── Unmatched — why? ──────────────────────────────────")
df_unmatched = df_results[df_results['match_type'] == 'No Match']
no_phone  = (~df_unmatched['has_phone']).sum()
has_phone = df_unmatched['has_phone'].sum()
print(f"   No phone number           : {no_phone:,}")
print(f"   Has phone, still no match : {has_phone:,}  ← phone not in POS cashier report")

print(f"\n── Unmatched sessions with phone (investigate these) ─")
df_phone_unmatched = df_unmatched[df_unmatched['has_phone']].copy()
if not df_phone_unmatched.empty:
    print(df_phone_unmatched[['Contact ID', 'mpesa_amount', 'acquisition_source']].to_string(index=False))
else:
    print("   None — all phone-matched contacts found a social sale ✅")

print(f"\n{'=' * 60}\n")