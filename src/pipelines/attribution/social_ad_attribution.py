"""
social_ad_attribution.py
========================
Links social_sales_direct.csv transactions to CRM sessions to answer:
  "How many contacts came through each ad, and what revenue did they generate?"

Matching waterfall (per transaction):
  Layer 1 — Respond Customer ID → sessions.Contact ID  (exact, grows from Apr 2026)
  Layer 2 — normalize(Phone Number) → sessions.phone_number
            → most recent session on or before Sale_Date
  Layer 3 — Unmatched: row kept, ad columns null

Output: social_ad_attribution.csv  (transaction-level, one row per Transaction ID)
        → feeds Power BI for per-ad, per-date contact + revenue reporting
"""

import os
import pandas as pd
from pathlib import Path
from Portal_ML_V4.src.config.settings import PROCESSED_DATA_DIR

try:
    from Portal_ML_V4.src.utils.phone import normalize_phone
except ImportError:
    def normalize_phone(val):
        if val is None: return None
        s = str(val).strip().replace('.0', '')
        s = ''.join(filter(str.isdigit, s))
        return s[-9:] if len(s) >= 9 else None

# ── CONFIG ────────────────────────────────────────────────────────────────────

ATTRIBUTION_DIR = PROCESSED_DATA_DIR / "sales_attribution"

SOCIAL_SALES_FILE = ATTRIBUTION_DIR / "social_sales_direct.csv"
SESSIONS_FILE     = PROCESSED_DATA_DIR / "fact_sessions_enriched.csv"
OUTPUT_FILE       = ATTRIBUTION_DIR / "social_ad_attribution.csv"

# Columns we pull from sessions to enrich each matched transaction
SESSION_COLS = [
    "Contact ID", "phone_number", "session_id", "session_start",
    "acquisition_source", "Ad Name", "Ad campaign ID", "Ad ID",
    "clean_ad_id",
]


# ── HELPERS ───────────────────────────────────────────────────────────────────

def load_sessions(path: Path) -> pd.DataFrame:
    """
    Load fact_sessions_enriched, keep only the columns we need,
    normalize phone, and sort chronologically for merge_asof.
    """
    usecols = [c for c in SESSION_COLS if c != "clean_ad_id"]  # derived, not stored

    try:
        df = pd.read_csv(path, low_memory=False, usecols=lambda c: c in SESSION_COLS + ["phone_number"])
    except Exception as e:
        print(f"   ❌ Could not load sessions: {e}")
        return pd.DataFrame()

    df["session_start"]  = pd.to_datetime(df["session_start"], errors="coerce")
    df["Contact ID"]     = df["Contact ID"].astype(str).str.strip()
    df["norm_phone"]     = df["phone_number"].apply(normalize_phone)

    # Keep only sessions that have some ad attribution value
    # (don't restrict — organic sessions are kept so Layer 2 can still match,
    #  acquisition_source tells Power BI whether it was paid or organic)
    df = df.dropna(subset=["session_start"])
    df = df.sort_values("session_start")
    return df


def load_social_sales(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, low_memory=False, dtype={"Respond Customer ID": str})
    except Exception as e:
        print(f"   ❌ Could not load social sales: {e}")
        return pd.DataFrame()

    df["Sale_Date"]    = pd.to_datetime(df["Sale_Date"], errors="coerce")
    df["norm_phone"]   = df["Phone Number"].apply(normalize_phone)

    # Clean Respond Customer ID — strip floats, whitespace, nulls
    if "Respond Customer ID" in df.columns:
        df["Respond Customer ID"] = (
            df["Respond Customer ID"]
            .astype(str)
            .str.strip()
            .str.replace(r"\.0$", "", regex=True)
        )
        df["Respond Customer ID"] = df["Respond Customer ID"].replace(
            ["nan", "None", "", "NaN"], None
        )
    else:
        df["Respond Customer ID"] = None

    return df


# ── MATCHING ENGINE ───────────────────────────────────────────────────────────

def run_attribution(df_sales: pd.DataFrame, df_sess: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a transaction-level DataFrame with session + ad attribution columns.
    One row per unique Transaction ID.
    """

    # Work at transaction level — deduplicate line items first
    txn = (
        df_sales[["Transaction ID", "Sale_Date", "norm_phone",
                  "Respond Customer ID", "Client Name", "Location",
                  "Total (Tax Ex)"]]
        .copy()
    )

    # Aggregate revenue per transaction (sum of line items)
    rev = (
        df_sales.groupby("Transaction ID")["Total (Tax Ex)"]
        .sum()
        .reset_index()
        .rename(columns={"Total (Tax Ex)": "transaction_revenue"})
    )
    txn = (
        txn.drop_duplicates("Transaction ID")
           .merge(rev, on="Transaction ID", how="left")
    )
    txn["Total (Tax Ex)"] = pd.to_numeric(txn["Total (Tax Ex)"], errors="coerce")

    # Initialise output columns — all null until a match fills them
    for col in ["matched_contact_id", "match_layer", "session_id",
                "session_start", "acquisition_source",
                "Ad Name", "Ad campaign ID", "Ad ID"]:
        txn[col] = None

    # ── Layer 1: Respond Customer ID → sessions.Contact ID ───────────────────
    print("   🔵 Layer 1: Respond Customer ID match...")
    has_rcid = txn["Respond Customer ID"].notna()

    if has_rcid.any():
        # Build a Contact ID → best session map
        # "best" = most recent session on or before the sale date
        # We do this as a merge then filter, keeping the closest match per txn

        rcid_txns = txn[has_rcid].copy()

        # sessions indexed by Contact ID — we want most recent session <= sale date
        sess_for_l1 = df_sess[df_sess["Contact ID"].notna()].copy()

        # Merge on Contact ID (Respond Customer ID = Contact ID in sessions)
        l1 = rcid_txns.merge(
            sess_for_l1.rename(columns={"Contact ID": "Respond Customer ID"}),
            on="Respond Customer ID",
            how="inner",
            suffixes=("", "_sess"),
        )

        if not l1.empty:
            # Keep only sessions on or before sale date
            l1 = l1[l1["session_start"] <= l1["Sale_Date"]]
            # Take the most recent session per transaction
            l1 = (
                l1.sort_values("session_start", ascending=False)
                  .drop_duplicates("Transaction ID", keep="first")
            )

            # Write results back into txn
            l1_map = l1.set_index("Transaction ID")
            for col in ["session_id", "session_start", "acquisition_source",
                        "Ad Name", "Ad campaign ID", "Ad ID"]:
                if col in l1_map.columns:
                    txn.loc[txn["Transaction ID"].isin(l1_map.index), col] = (
                        txn.loc[txn["Transaction ID"].isin(l1_map.index), "Transaction ID"]
                           .map(l1_map[col])
                    )
            txn.loc[txn["Transaction ID"].isin(l1_map.index), "matched_contact_id"] = (
                txn.loc[txn["Transaction ID"].isin(l1_map.index), "Transaction ID"]
                   .map(l1_map["Respond Customer ID"])
            )
            txn.loc[txn["Transaction ID"].isin(l1_map.index), "match_layer"] = "Layer 1 - Respond ID"

            l1_count = txn["match_layer"].eq("Layer 1 - Respond ID").sum()
            print(f"      ✅ Layer 1 matched: {l1_count:,} transactions")
        else:
            print("      ℹ️  Layer 1: no session matches found (expected today — column is new)")
    else:
        print("      ℹ️  Layer 1: no Respond Customer IDs present yet (expected — column is new)")

    # ── Layer 2: Phone number → sessions.phone_number ────────────────────────
    print("   🟡 Layer 2: Phone number match...")

    # Only run on transactions not already matched by Layer 1
    unmatched_mask = txn["match_layer"].isna()
    has_phone_mask = txn["norm_phone"].notna()
    l2_candidates  = txn[unmatched_mask & has_phone_mask].copy()

    if not l2_candidates.empty and df_sess["norm_phone"].notna().any():
        # Sessions with a phone number
        sess_with_phone = df_sess[df_sess["norm_phone"].notna()].copy()

        # merge_asof: for each transaction (sorted by Sale_Date),
        # find the most recent session on or before Sale_Date, matched by phone
        l2_candidates = l2_candidates.sort_values("Sale_Date")
        sess_with_phone = sess_with_phone.sort_values("session_start")

        l2 = pd.merge_asof(
            l2_candidates[["Transaction ID", "Sale_Date", "norm_phone"]],
            sess_with_phone[[
                "norm_phone", "session_start", "Contact ID",
                "session_id", "acquisition_source",
                "Ad Name", "Ad campaign ID", "Ad ID",
            ]],
            left_on="Sale_Date",
            right_on="session_start",
            by="norm_phone",
            direction="backward",   # session must be on or before the sale
        )

        # Drop rows where merge_asof found no session (session_start is null)
        l2 = l2.dropna(subset=["session_start"])

        if not l2.empty:
            l2_map = l2.set_index("Transaction ID")

            for col in ["session_id", "session_start", "acquisition_source",
                        "Ad Name", "Ad campaign ID", "Ad ID"]:
                if col in l2_map.columns:
                    matched_idx = txn["Transaction ID"].isin(l2_map.index) & unmatched_mask
                    txn.loc[matched_idx, col] = (
                        txn.loc[matched_idx, "Transaction ID"].map(l2_map[col])
                    )

            matched_idx = txn["Transaction ID"].isin(l2_map.index) & unmatched_mask
            txn.loc[matched_idx, "matched_contact_id"] = (
                txn.loc[matched_idx, "Transaction ID"].map(l2_map["Contact ID"])
            )
            txn.loc[matched_idx, "match_layer"] = "Layer 2 - Phone"

            l2_count = txn["match_layer"].eq("Layer 2 - Phone").sum()
            print(f"      ✅ Layer 2 matched: {l2_count:,} transactions")
        else:
            print("      ℹ️  Layer 2: no phone matches found")
    else:
        print("      ℹ️  Layer 2: no phone numbers available to match")

    # ── Layer 3: Unmatched ────────────────────────────────────────────────────
    txn["match_layer"] = txn["match_layer"].fillna("Unmatched")
    txn["acquisition_source"] = txn["acquisition_source"].fillna("Unmatched")

    return txn


# ── MAIN ──────────────────────────────────────────────────────────────────────

def run_social_ad_attribution():
    print("=" * 60)
    print("📊 SOCIAL AD ATTRIBUTION")
    print("=" * 60)

    # 1. Load inputs
    print("\n📥 Loading social sales...")
    df_sales = load_social_sales(SOCIAL_SALES_FILE)
    if df_sales.empty:
        print("❌ No social sales data. Run social_sales_direct.py first.")
        return
    print(f"   {len(df_sales):,} line items · {df_sales['Transaction ID'].nunique():,} transactions")

    print("\n📥 Loading sessions...")
    df_sess = load_sessions(SESSIONS_FILE)
    if df_sess.empty:
        print("❌ No session data. Run analytics pipeline first.")
        return
    print(f"   {len(df_sess):,} sessions · {df_sess['norm_phone'].notna().sum():,} with phone")

    # 2. Run matching waterfall
    print("\n🔗 Running matching waterfall...")
    df_out = run_attribution(df_sales, df_sess)

    # 3. Summary
    print("\n📊 Match Summary:")
    print(f"   {'Layer':<30} {'Txns':>6}  {'Revenue (KES)':>14}")
    print(f"   {'-'*55}")
    total_rev = df_out["transaction_revenue"].sum()
    for layer in ["Layer 1 - Respond ID", "Layer 2 - Phone", "Unmatched"]:
        grp = df_out[df_out["match_layer"] == layer]
        if grp.empty:
            continue
        rev = grp["transaction_revenue"].sum()
        print(f"   {layer:<30} {len(grp):>6,}  KES {rev:>10,.0f}")
    print(f"   {'-'*55}")
    print(f"   {'TOTAL':<30} {len(df_out):>6,}  KES {total_rev:>10,.0f}")

    # Ad breakdown — paid only
    paid = df_out[df_out["acquisition_source"] == "Paid Ads"]
    if not paid.empty:
        print(f"\n   By Ad (Paid, all time):")
        print(f"   {'Ad Name':<35} {'Contacts':>8}  {'Txns':>6}  {'Revenue (KES)':>14}")
        print(f"   {'-'*68}")
        ad_summary = (
            paid.groupby("Ad Name", dropna=False)
                .agg(
                    contacts=("matched_contact_id", "nunique"),
                    txns=("Transaction ID", "nunique"),
                    revenue=("transaction_revenue", "sum"),
                )
                .sort_values("revenue", ascending=False)
        )
        for ad_name, row in ad_summary.iterrows():
            label = str(ad_name) if pd.notna(ad_name) else "Unknown Ad"
            print(f"   {label:<35} {int(row['contacts']):>8,}  {int(row['txns']):>6,}  KES {row['revenue']:>10,.0f}")

    # 4. Save
    os.makedirs(OUTPUT_FILE.parent, exist_ok=True)
    df_out.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ Saved: {len(df_out):,} transactions → {OUTPUT_FILE}")
    print(f"   Columns: {list(df_out.columns)}")


if __name__ == "__main__":
    run_social_ad_attribution()