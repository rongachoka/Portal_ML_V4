import pandas as pd
from Portal_ML_V4.src.config.settings import PROCESSED_DATA_DIR
from Portal_ML_V4.src.utils.phone import normalize_phone

df_sess   = pd.read_csv(PROCESSED_DATA_DIR / "fact_sessions_enriched.csv")
df_social = pd.read_csv(PROCESSED_DATA_DIR / "sales_attribution" / "social_sales_Jan25_Jan26.csv")

df_sess['session_start'] = pd.to_datetime(df_sess['session_start'], errors='coerce')
df_sess['mpesa_amount']  = pd.to_numeric(df_sess['mpesa_amount'], errors='coerce').fillna(0)

# February only
df_feb = df_sess[
    (df_sess['session_start'].dt.year  == 2026) &
    (df_sess['session_start'].dt.month == 2)
].copy()

df_conv = df_feb[(df_feb['is_converted']==1) | (df_feb['mpesa_amount']>0)].copy()

matched_ids = set(df_social['session_id'].dropna()) if 'session_id' in df_social.columns else set()
matched     = df_conv[df_conv['session_id'].isin(matched_ids)]
unmatched   = df_conv[~df_conv['session_id'].isin(matched_ids)]

print(f"FEBRUARY 2026 — ATTRIBUTION GAP ANALYSIS")
print(f"{'='*55}")
print(f"Total M-Pesa (Feb converted sessions):  KES {df_conv['mpesa_amount'].sum():>10,.0f}")
print(f"In social_sales (POS matched):          KES {matched['mpesa_amount'].sum():>10,.0f}")
print(f"GAP (confirmed but unmatched):          KES {unmatched['mpesa_amount'].sum():>10,.0f}")

print(f"\nGap by acquisition source:")
print(unmatched.groupby('acquisition_source')['mpesa_amount'].sum().sort_values(ascending=False).to_string())

print(f"\nGap by channel:")
print(unmatched.groupby('channel_name')['mpesa_amount'].sum().sort_values(ascending=False).to_string())

print(f"\nGap by matched_brand:")
print(unmatched.groupby('matched_brand')['mpesa_amount'].sum().sort_values(ascending=False).head(15).to_string())

# Key question: of the WhatsApp unmatched, how many actually have a phone number?
wa_unmatched = unmatched[unmatched['channel_name'] == 'WhatsApp'].copy()



wa_unmatched['norm_phone'] = wa_unmatched['phone_number'].apply(normalize_phone)

has_phone    = wa_unmatched['norm_phone'].notna().sum()
no_phone     = wa_unmatched['norm_phone'].isna().sum()
rev_w_phone  = wa_unmatched[wa_unmatched['norm_phone'].notna()]['mpesa_amount'].sum()
rev_no_phone = wa_unmatched[wa_unmatched['norm_phone'].isna()]['mpesa_amount'].sum()

print(f"\nWHATSAPP UNMATCHED — PHONE NUMBER BREAKDOWN (Feb):")
print(f"{'='*55}")
print(f"  Sessions WITH phone number:  {has_phone:>5,}  →  KES {rev_w_phone:>10,.0f}  ← waterfall should have caught these")
print(f"  Sessions WITHOUT phone:      {no_phone:>5,}  →  KES {rev_no_phone:>10,.0f}  ← genuinely unresolvable without POS entry")

print(f"\nFor sessions WITH phone that still didn't match — top session examples:")
sample = wa_unmatched[wa_unmatched['norm_phone'].notna()][
    ['session_id','session_start','contact_name','phone_number','mpesa_amount','matched_brand']
].head(10)
print(sample.to_string(index=False))