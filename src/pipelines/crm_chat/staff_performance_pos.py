"""
staff_performance_pos.py
========================
Staff performance report built directly from all_locations_sales_NEW.csv.

Replaces the session-based approach for converted sales.
No dependency on fact_sessions_enriched or cleaned_messages.

Attribution logic:
  - Filters to respond.io rows OR any row with a Respond Customer ID
    (staff started entering Contact IDs in the cashier report ~5 days ago)
  - Aggregates to one row per Transaction ID
  - Sales Rep → Staff_Name via SALES_REP_MAP then STAFF_ID_MAP
  - All rows are Is_Converted = 1 (these are actual POS sales)
  - Revenue = sum of Total (Tax Ex) per transaction
  - Time_To_Conversion_Mins / Messages_To_Conversion = null
    (no message log — Power BI handles null in averages correctly)
  - Attribution_Bucket = "1. Active (≤ 24h)" for all rows
    (direct POS sale, no session lag to measure)

Output columns are identical to fact_staff_performance_v6_granular.csv
so Power BI needs no changes.

Run:
    python -m Portal_ML_V4.src.pipelines.staff_performance_pos
    — or —
    python staff_performance_pos.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── Imports ───────────────────────────────────────────────────────────────────
try:
    from Portal_ML_V4.src.config.settings import PROCESSED_DATA_DIR
except ImportError:
    PROCESSED_DATA_DIR = Path(r"D:\Documents\Portal ML Analys\Portal_ML\Portal_ML_V4\data\03_processed")

# ── Config ────────────────────────────────────────────────────────────────────
SALES_FILE  = PROCESSED_DATA_DIR / "pos_data" / "all_locations_sales_NEW.csv"
OUTPUT_FILE = PROCESSED_DATA_DIR / "fact_staff_performance_pos_direct.csv"

SOCIAL_TAG = "respond.io"

# Canonical staff ID map — same as staff_performance.py
STAFF_ID_MAP = {
    "845968": "Joy",     "847526": "Ishmael", "860475": "Faith",
    "879396": "Nimmoh",  "879430": "Rahab",   "879438": "Brenda",
    "971945": "Jeff",    "1000558": "Sharon",  "1006108": "Jess",
    "962460": "Katie",   "1052677": "Vivian"
}

# POS name → canonical name — same as staff_performance.py
SALES_REP_MAP = {
    "Cate":  "Kate",
    "Emily": "Nimmoh",
}

SYSTEM_NAMES = {
    "System", "Bot", "Auto Assign", "Workflow",
    "Unknown", "Nan", "None", "Unassigned", "",
}

# Output column order — identical to fact_staff_performance_v6_granular.csv
OUTPUT_COLS = [
    "Reporting_Date",
    "Staff_Name",
    "Attribution_Bucket",
    "Is_Converted",
    "Customers_Handled",
    "Revenue",
    "Time_To_Conversion_Mins",   # null — no message log
    "Messages_To_Conversion",    # null — no message log
    "session_id",                # Transaction ID
    "Contact ID",                # Respond Customer ID
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def normalize_sales_rep(name: str) -> str:
    """
    1. Strip and title-case the raw Sales Rep value.
    2. Apply SALES_REP_MAP overrides (Cate → Kate, Emily → Nimmoh).
    3. Return the canonical name, or None if blank / system value.
    """
    if pd.isna(name):
        return None
    clean = str(name).strip().title()
    clean = SALES_REP_MAP.get(clean, clean)
    if clean in SYSTEM_NAMES or len(clean) < 2:
        return None
    return clean


def clean_contact_id(val) -> str | None:
    """Strip the Respond Customer ID to a plain digit string."""
    if pd.isna(val):
        return None
    s = str(val).strip().replace(".0", "")
    s = "".join(filter(str.isdigit, s))
    return s if s else None


# ── Main ──────────────────────────────────────────────────────────────────────

def run_staff_analysis_pos():
    print("=" * 60)
    print("  STAFF PERFORMANCE — POS DIRECT")
    print(f"  Input  : {SALES_FILE.name}")
    print(f"  Output : {OUTPUT_FILE.name}")
    print("=" * 60)

    if not SALES_FILE.exists():
        print(f"\n❌  Input file not found: {SALES_FILE}")
        print("    Run etl_local.py first to generate all_locations_sales_NEW.csv")
        return

    # ── 1. Load ───────────────────────────────────────────────────────────────
    print(f"\n📥 Loading {SALES_FILE.name}...")
    df = pd.read_csv(SALES_FILE, low_memory=False, dtype=str)
    df.columns = df.columns.str.strip()
    print(f"   Total rows: {len(df):,}")

    # ── 2. Filter to chat-attributed sales ────────────────────────────────────
    # Keep respond.io rows OR any row where the cashier entered a Contact ID.
    # Staff started populating Respond Customer ID ~5 days ago so both signals
    # are needed to capture the full attribution window.
    is_respond = (
        df["Ordered Via"].fillna("").str.strip().str.lower() == SOCIAL_TAG
    )
    has_contact_id = (
        df["Respond Customer ID"].fillna("").str.strip().str.len() > 0
    )
    df_attr = df[is_respond | has_contact_id].copy()

    print(f"   respond.io rows        : {is_respond.sum():,}")
    print(f"   Rows with Contact ID   : {has_contact_id.sum():,}")
    print(f"   Combined (de-duped)    : {len(df_attr):,}")

    if df_attr.empty:
        print("\n❌  No attributed rows found.")
        return

    # ── 3. Numeric clean ──────────────────────────────────────────────────────
    df_attr["Total (Tax Ex)"] = pd.to_numeric(
        df_attr["Total (Tax Ex)"], errors="coerce"
    ).fillna(0)
    df_attr["Sale_Date"] = pd.to_datetime(
        df_attr["Sale_Date"], errors="coerce"
    )

    # ── 4. Normalize Sales Rep → Staff_Name ───────────────────────────────────
    df_attr["Staff_Name"] = df_attr["Sales Rep"].apply(normalize_sales_rep)

    # ── 5. Normalize Respond Customer ID ──────────────────────────────────────
    df_attr["_contact_id"] = df_attr["Respond Customer ID"].apply(clean_contact_id)

    # ── 6. Aggregate to transaction level ─────────────────────────────────────
    # Revenue = sum of line items per transaction.
    # Staff_Name / Sale_Date / Contact ID: take first non-null per transaction
    # (all line items in one transaction share the same cashier metadata).
    print("\n🔢 Aggregating to transaction level...")

    def first_non_null(s):
        non_null = s.dropna()
        return non_null.iloc[0] if not non_null.empty else None

    txn = df_attr.groupby("Transaction ID", as_index=False).agg(
        Reporting_Date   = ("Sale_Date",     "min"),
        Revenue          = ("Total (Tax Ex)", "sum"),
        Staff_Name       = ("Staff_Name",     first_non_null),
        Contact_ID_raw   = ("_contact_id",    first_non_null),
        Location         = ("Location",       first_non_null),
    )

    total_txns = len(txn)
    print(f"   Transactions: {total_txns:,}")

    # ── 7. Staff Name validation ───────────────────────────────────────────────
    # Only names in KNOWN_STAFF are valid. Unrecognized names are nulled out
    # and excluded from the output. They are logged so you can add them to
    # SALES_REP_MAP if they turn out to be real team members.
    KNOWN_STAFF = set(STAFF_ID_MAP.values()) | set(SALES_REP_MAP.values())

    unrecognized = txn[
        txn["Staff_Name"].notna()
        & ~txn["Staff_Name"].isin(KNOWN_STAFF)
    ]["Staff_Name"].value_counts()
    if not unrecognized.empty:
        print(f"\n   ⚠️  Excluded (not in KNOWN_STAFF):")
        for name, count in unrecognized.items():
            print(f"      {name:<25} {count:>5,} txns — add to SALES_REP_MAP if valid")

    # Null out unrecognized names so they are dropped in step 9
    txn.loc[
        txn["Staff_Name"].notna() & ~txn["Staff_Name"].isin(KNOWN_STAFF),
        "Staff_Name"
    ] = None

    no_staff = txn["Staff_Name"].isna().sum()
    print(f"\n   Staff resolved         : {total_txns - no_staff:,} transactions")
    print(f"   Excluded / no rep      : {no_staff:,} transactions")


    # ── 8. Build output columns ───────────────────────────────────────────────
    txn["Attribution_Bucket"]       = "1. Active (≤ 24h)"   # direct POS sale
    txn["Is_Converted"]             = 1
    txn["Customers_Handled"]        = 1
    txn["Time_To_Conversion_Mins"]  = np.nan   # no message log
    txn["Messages_To_Conversion"]   = np.nan   # no message log
    txn["session_id"]               = txn["Transaction ID"]
    txn["Contact ID"]               = txn["Contact_ID_raw"]
    txn["Reporting_Date"]           = txn["Reporting_Date"].dt.date

    # Remove system / blank staff rows
    txn = txn[~txn["Staff_Name"].isin(SYSTEM_NAMES)].copy()
    txn = txn[txn["Staff_Name"].notna()].copy()

    # ── 9. Final column selection & sort ─────────────────────────────────────
    final_df = txn[OUTPUT_COLS].sort_values("Reporting_Date", ascending=False)

    # ── 10. Save ──────────────────────────────────────────────────────────────
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(OUTPUT_FILE, index=False)

    # ── 11. Summary ───────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  Revenue (Tax Ex) : KES {final_df['Revenue'].sum():>12,.0f}")
    print(f"  Transactions     :     {len(final_df):>7,}")

    print(f"\n  Staff breakdown:")
    staff_rev = (
        final_df.groupby("Staff_Name")["Revenue"]
        .sum().sort_values(ascending=False)
    )
    for staff, rev in staff_rev.items():
        txn_count = (final_df["Staff_Name"] == staff).sum()
        print(f"     {staff:<20} {txn_count:>5,} txns   KES {rev:>10,.0f}")

    print(f"\n✅  Saved → {OUTPUT_FILE}")
    print(f"{'─'*60}")


if __name__ == "__main__":
    run_staff_analysis_pos()