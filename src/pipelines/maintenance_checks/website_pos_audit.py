"""
check_website_pos_duplicates.py

Checks for true duplicates between website orders and POS data.
A true duplicate = same phone + same product + same date + same amount.

Run with:
    py -m Portal_ML_V4.check_website_pos_duplicates
"""
import pandas as pd
from sqlalchemy import create_engine, text
from Portal_ML_V4.src.config.settings import PROCESSED_DATA_DIR, DB_CONNECTION_STRING
from Portal_ML_V4.src.utils.phone import normalize_phone

WEBSITE_ITEM_PATH = PROCESSED_DATA_DIR / "website_data" / "website_item_history.csv"
WEBSITE_CLIENT_PATH = PROCESSED_DATA_DIR / "website_data" / "website_dim_clients.csv"

print("\n" + "="*60)
print("🔍 WEBSITE vs POS DUPLICATE CHECK")
print("="*60)

# ── 1. Load website items ─────────────────────────────────────
if not WEBSITE_ITEM_PATH.exists():
    print(f"❌ Website item history not found: {WEBSITE_ITEM_PATH}")
    print("   Run website_orders.py first.")
    exit()

df_web = pd.read_csv(WEBSITE_ITEM_PATH)
print(f"\n📦 Website item rows   : {len(df_web):,}")
print(f"   Columns: {df_web.columns.tolist()}")

# Show all columns so we can see exactly what's there
print(f"\n   All columns: {df_web.columns.tolist()}")

# Normalise phone — strip to 9-digit Kenyan format for comparison
phone_col = next((c for c in ['phone_clean', 'phone_number', 'Phone'] if c in df_web.columns), None)
if phone_col:
    df_web['phone_norm'] = df_web[phone_col].apply(normalize_phone)
    # Show sample of raw vs normalized so we can verify
    print(f"\n   Phone sample (raw → normalized):")
    sample = df_web[[phone_col, 'phone_norm']].dropna().head(5)
    for _, r in sample.iterrows():
        print(f"      {r[phone_col]}  →  {r['phone_norm']}")
else:
    print("❌ No phone column found in website items")
    exit()

# Normalise key fields — check actual column names present
date_col    = next((c for c in ['First_Order_Date', 'Last_Order_Date', 'All_Order_Dates',
                                 'order_date', 'date', 'Date', 'Order Date'] if c in df_web.columns), None)
product_col = next((c for c in ['product_bought', 'product', 'Product',
                                 'description', 'item_name'] if c in df_web.columns), None)
amount_col  = next((c for c in ['Total_Spend', 'Unit_Price', 'amount',
                                 'Amount', 'price', 'total'] if c in df_web.columns), None)

print(f"\n   Using: phone={phone_col}, date={date_col}, product={product_col}, amount={amount_col}")

if date_col:
    df_web['date_norm'] = pd.to_datetime(df_web[date_col], errors='coerce').dt.date
if product_col:
    df_web['product_norm'] = df_web[product_col].astype(str).str.upper().str.strip()
if amount_col:
    df_web['amount_norm'] = pd.to_numeric(df_web[amount_col], errors='coerce').round(0)

# ── 2. Load POS line items from DB ────────────────────────────
engine = create_engine(DB_CONNECTION_STRING)
df_pos = pd.read_sql(text("""
    SELECT
        phone_number,
        sale_date               AS date_norm,
        UPPER(TRIM(description)) AS product_norm,
        ROUND(total_tax_ex::numeric, 0) AS amount_norm,
        client_name,
        transaction_id,
        location
    FROM fact_sales_lineitems
    WHERE phone_number IS NOT NULL
      AND TRIM(phone_number) <> ''
      AND description IS NOT NULL
"""), engine)
engine.dispose()

# Normalize POS phones the same way as website phones
df_pos['phone_norm'] = df_pos['phone_number'].apply(normalize_phone)

print(f"🏪 POS line item rows  : {len(df_pos):,}")
print(f"\n   POS phone sample (raw → normalized):")
sample_pos = df_pos[['phone_number', 'phone_norm']].dropna().drop_duplicates().head(5)
for _, r in sample_pos.iterrows():
    print(f"      {r['phone_number']}  →  {r['phone_norm']}")

# ── 3. Check phone overlap at client level ────────────────────
web_phones = set(df_web['phone_norm'].dropna().unique())
pos_phones = set(df_pos['phone_norm'].dropna().unique())

both        = web_phones & pos_phones
web_only    = web_phones - pos_phones
pos_only    = pos_phones - web_phones

print(f"\n📊 PHONE OVERLAP:")
print(f"   Website clients          : {len(web_phones):,}")
print(f"   POS clients              : {len(pos_phones):,}")
print(f"   In BOTH (need checking)  : {len(both):,}")
print(f"   Website-only             : {len(web_only):,}  ← safe to add directly")
print(f"   POS-only                 : {len(pos_only):,}  ← unaffected")

# ── 4. True duplicate check — same phone + product + date + amount ──
if not all([date_col, product_col, amount_col]):
    print("\n⚠️  Cannot do product-level check — missing date/product/amount column in website data")
    exit()

# Only check clients that appear in both
df_web_overlap = df_web[df_web['phone_norm'].isin(both)][
    ['phone_norm', 'date_norm', 'product_norm', 'amount_norm']
].dropna()

df_pos_overlap = df_pos[df_pos['phone_norm'].isin(both)][
    ['phone_norm', 'date_norm', 'product_norm', 'amount_norm']
].dropna()

# ── 4. True duplicate check — same phone + product + amount ──────
# Note: website data is aggregated (First/Last order dates, total spend)
# so we match on phone + product + amount rather than exact date
if not all([product_col, amount_col]):
    print("\n⚠️  Cannot do product-level check — missing product/amount column in website data")
    print(f"   Available columns: {df_web.columns.tolist()}")
else:
    df_web['product_norm'] = df_web[product_col].astype(str).str.upper().str.strip()
    df_web['amount_norm']  = pd.to_numeric(df_web[amount_col], errors='coerce').round(0)

    # Aggregate POS to same level as website (per phone + product + total spend)
    df_pos_agg = (
        df_pos.groupby(['phone_norm', 'product_norm'], as_index=False)
        .agg(amount_norm=('amount_norm', 'sum'))
    )
    df_pos_agg['amount_norm'] = df_pos_agg['amount_norm'].round(0)

    df_web_check = df_web[['phone_norm', 'product_norm', 'amount_norm']].dropna()

    dupes = df_web_check.merge(
        df_pos_agg,
        on=['phone_norm', 'product_norm', 'amount_norm'],
        how='inner'
    )

    print(f"\n🔴 TRUE DUPLICATES (same phone + product + total amount):")
    print(f"   Count: {len(dupes):,}")

    if not dupes.empty:
        print("\n   Sample:")
        print(dupes.head(10).to_string(index=False))
        dupes.to_csv(PROCESSED_DATA_DIR / "duplicate_check_results.csv", index=False)
        print(f"\n   Full list saved to: duplicate_check_results.csv")
    else:
        print("   ✅ No true duplicates found — safe to merge")

    # ── 5. Phone + product match (different amount — could be partial purchase) ──
    partial = df_web_check[['phone_norm', 'product_norm']].merge(
        df_pos_agg[['phone_norm', 'product_norm']].drop_duplicates(),
        on=['phone_norm', 'product_norm'],
        how='inner'
    )
    print(f"\n🟡 SAME CLIENT + PRODUCT (any amount — includes legitimate repeats):")
    print(f"   Count: {len(partial):,}")
    if not partial.empty:
        print("\n   Sample:")
        print(partial.head(5).to_string(index=False))

print("\n✅ Check complete.")