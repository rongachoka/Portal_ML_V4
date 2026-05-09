"""
load_to_postgres.py
====================
Loads cleaned ETL output into the PostgreSQL warehouse tables:
  - fact_sales_lineitems      (one row per line item)
  - fact_sales_transactions   (one row per transaction)

Then refreshes the materialized views:
  - mv_transaction_master
  - mv_client_list

Location: Portal_ML_V4/src/scripts/pos_finance/load_to_postgres.py

Run order in pipeline:
  etl.py → load_to_postgres.py → run_client_export.py
"""

import pandas as pd
import gc
import numpy as np
import re
import os
from pathlib import Path
from sqlalchemy import create_engine, text

from Portal_ML_V4.src.config.settings import (
    BASE_DIR,
    PROCESSED_DATA_DIR,
    DB_CONNECTION_STRING,
)
from Portal_ML_V4.src.utils.name_cleaner import clean_name_series

# ==========================================
# 1. CONFIGURATION
# ==========================================

# ETL output files (produced by etl.py)
ETL_OUTPUT_FILE      = PROCESSED_DATA_DIR / "pos_data" / "all_locations_sales_Jan25-Jan26.csv"
ETL_FULL_HIST_FILE   = PROCESSED_DATA_DIR / "pos_data" / "all_locations_sales_FULL_HISTORY.csv"

# Load full history into the DW — captures all data not just 2025+
SOURCE_FILE = ETL_FULL_HIST_FILE

# DB table names
TABLE_LINEITEMS     = "fact_sales_lineitems"
TABLE_TRANSACTIONS  = "fact_sales_transactions"

# ==========================================
# 2. HELPERS
# ==========================================

def normalise_phone(val):
    """Standardises to +254XXXXXXXXX. Consistent with analytics.py."""
    if pd.isna(val):
        return None
    s = re.sub(r'[^\d]', '', str(val).strip())
    if len(s) == 0:
        return None
    if s.startswith('254') and len(s) == 12:
        return f'+{s}'
    if s.startswith('0') and len(s) == 10:
        return f'+254{s[1:]}'
    if len(s) == 9:
        return f'+254{s}'
    return s


def rename_csv_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Maps ETL CSV column names (with spaces and special chars)
    to clean snake_case DB column names.
    """
    col_map = {
        'Department':          'department',
        'Category':            'category',
        'Item':                'item',
        'Description':         'description',
        'On Hand':             'on_hand',
        'Last Sold':           'last_sold',
        'Qty Sold':            'qty_sold',
        'Total (Tax Ex)':      'total_tax_ex',
        'Transaction ID':      'transaction_id',
        'Date Sold':           'date_sold',
        'Sale_Date':           'sale_date',
        'Sale_Date_Str':       'sale_date_str',
        'Transaction_Total':   'transaction_total',
        'Location':            'location',
        'Audit_Status':        'audit_status',
        # Cashier columns
        'Client Name':         'client_name',
        'Phone Number':        'phone_number',
        'Sales Rep':           'sales_rep',
        'Txn Type':            'txn_type',
        'Ordered Via':         'ordered_via',
        'Amount':              'cashier_amount',
        'Txn Costs':           'txn_costs',
        'Time':                'txn_time',
        'Receipt Txn No':      'receipt_txn_no',
    }
    return df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})


# ==========================================
# 3. LOAD LINE ITEMS
# ==========================================

def build_lineitems(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares the line-item level DataFrame for insertion.
    One row = one product line in a transaction.
    """
    # Columns for line items table
    lineitem_cols = [
        'location', 'transaction_id',
        'department', 'category', 'item', 'description',
        'qty_sold', 'total_tax_ex',
        'date_sold', 'sale_date', 'sale_date_str',
        'client_name', 'phone_number', 'sales_rep',
        'txn_type', 'ordered_via',
        'cashier_amount', 'transaction_total', 'audit_status',
    ]

    available = [c for c in lineitem_cols if c in df.columns]
    out = df[available].copy()

    # Cast transaction_id and location to string — DB columns are TEXT
    # but CSV may read them as integers causing type mismatch in Postgres
    for col in ['transaction_id', 'location']:
        if col in out.columns:
            out[col] = out[col].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)

    # Numeric cleanup
    for col in ['qty_sold', 'total_tax_ex', 'cashier_amount', 'transaction_total']:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors='coerce')

    # Date cleanup
    if 'sale_date' in out.columns:
        out['sale_date'] = pd.to_datetime(out['sale_date'], errors='coerce').dt.date

    # Phone normalisation
    if 'phone_number' in out.columns:
        out['phone_number'] = out['phone_number'].apply(normalise_phone)

    # Name cleaning
    if 'client_name' in out.columns:
        out['client_name'] = clean_name_series(out['client_name'])

    # Replace NaN with None for clean DB nulls
    out = out.where(pd.notna(out), other=None)

    return out


# ==========================================
# 4. BUILD TRANSACTIONS (aggregated from line items)
# ==========================================

def build_transactions(df_lineitems: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates line items up to one row per transaction.
    Applies the same priority logic as mv_transaction_master:
      real_value = cashier_amount if > 0 else pos_txn_sum
    """
    # Ensure transaction_id is string in the aggregation input too
    df_lineitems = df_lineitems.copy()
    df_lineitems['transaction_id'] = df_lineitems['transaction_id'].astype(str).str.strip().str.replace(r'.0$', '', regex=True)

    grp = df_lineitems.groupby(['location', 'transaction_id'], as_index=False)

    txns = grp.agg(
        sale_date        = ('sale_date',         'max'),
        sale_date_str    = ('sale_date_str',      'first'),
        client_name      = ('client_name',        'first'),
        phone_number     = ('phone_number',        'first'),
        sales_rep        = ('sales_rep',           'first'),
        txn_type         = ('txn_type',            'first'),
        ordered_via      = ('ordered_via',         'first'),
        pos_txn_sum      = ('total_tax_ex',        'sum'),
        cashier_amount   = ('cashier_amount',      'first'),
        transaction_total= ('transaction_total',   'first'),
        item_count       = ('description',         'nunique'),
        audit_status     = ('audit_status',        'first'),
        products_in_txn  = ('description',
                            lambda x: ' | '.join(
                                sorted(x.dropna().astype(str).unique())
                            )),
    )

    # Priority: cashier amount if > 0, else POS sum
    txns['cashier_amount'] = pd.to_numeric(txns['cashier_amount'], errors='coerce').fillna(0)
    txns['pos_txn_sum']    = pd.to_numeric(txns['pos_txn_sum'],    errors='coerce').fillna(0)

    txns['real_transaction_value'] = np.where(
        txns['cashier_amount'] > 0,
        txns['cashier_amount'],
        txns['pos_txn_sum']
    )

    # Audit status
    txns['audit_status'] = np.where(
        (txns['cashier_amount'] > 0) &
        (abs(txns['pos_txn_sum'] - txns['cashier_amount']) > 1),
        'TRUE DISCREPANCY',
        np.where(txns['cashier_amount'] == 0, 'NO CASHIER DATA', 'MATCH')
    )

    txns = txns.where(pd.notna(txns), other=None)

    return txns


# ==========================================
# 5. LOAD TO POSTGRES
# ==========================================

def load_lineitems_to_db(df: pd.DataFrame, engine, mode: str = 'upsert'):
    """
    Loads line items into fact_sales_lineitems.

    mode='upsert' (default):
        - Pulls existing transaction IDs from DB
        - Anti-join: only new transactions get inserted (fast, no deletes)
        - Correction check: if cashier_amount changed on an existing transaction,
          flag it in a corrections log and update those rows
    mode='replace':
        - Truncates table and reloads everything (full reset)
    """
    if df.empty:
        print("      ⚠️  No line items to load.")
        return

    if mode == 'replace':
        print(f"      🗑️  Truncating {TABLE_LINEITEMS}...")
        with engine.begin() as conn:
            conn.execute(text(f"TRUNCATE TABLE {TABLE_LINEITEMS} RESTART IDENTITY;"))
        df.to_sql(
            TABLE_LINEITEMS, engine, if_exists='append',
            index=False, method='multi', chunksize=5000,
        )
        print(f"      ✅ {len(df):,} line item rows loaded into {TABLE_LINEITEMS}")
        return

    # ── UPSERT MODE ──────────────────────────────────────────────────
    print("      🔍 Fetching existing transactions from DB...")
    existing = pd.read_sql(
        f"SELECT location, transaction_id, cashier_amount FROM {TABLE_LINEITEMS}",
        engine
    )

    # Dedupe existing to one row per transaction for the amount comparison
    # (cashier_amount is the same across all line items for a transaction)
    existing_txns = (
        existing.groupby(['location', 'transaction_id'], as_index=False)['cashier_amount']
        .first()
    )

    # Incoming CSV — one row per transaction for comparison
    incoming_txns = (
        df.groupby(['location', 'transaction_id'], as_index=False)['cashier_amount']
        .first()
    )

    # ── STEP 1: Find genuinely new transactions (anti-join) ──────────
    merged = incoming_txns.merge(
        existing_txns[['location', 'transaction_id']],
        on=['location', 'transaction_id'],
        how='left',
        indicator=True
    )
    new_txn_ids = merged[merged['_merge'] == 'left_only'][['location', 'transaction_id']]

    df_new = df.merge(new_txn_ids, on=['location', 'transaction_id'], how='inner')

    # ── STEP 2: Find corrections (existing txn, amount changed) ──────
    amount_check = incoming_txns.merge(
        existing_txns,
        on=['location', 'transaction_id'],
        how='inner',
        suffixes=('_new', '_db')
    )
    amount_check['cashier_amount_new'] = pd.to_numeric(amount_check['cashier_amount_new'], errors='coerce').fillna(0)
    amount_check['cashier_amount_db']  = pd.to_numeric(amount_check['cashier_amount_db'],  errors='coerce').fillna(0)

    corrections = amount_check[
        abs(amount_check['cashier_amount_new'] - amount_check['cashier_amount_db']) > 1
    ][['location', 'transaction_id', 'cashier_amount_db', 'cashier_amount_new']]

    # ── STEP 3: Insert new transactions ──────────────────────────────
    if not df_new.empty:
        df_new.to_sql(
            TABLE_LINEITEMS, engine, if_exists='append',
            index=False, method='multi', chunksize=5000,
        )
        print(f"      ✅ {len(df_new):,} NEW line item rows inserted")
    else:
        print("      ✅ No new line items to insert — DB is up to date")

    # ── STEP 4: Handle corrections ─────────────────────────────────────────────
    # "Corrections" = existing transaction where cashier_amount genuinely changed.
    # We ignore 0 → value transitions (those are late cashier matches, not edits).
    real_corrections = corrections[corrections['cashier_amount_db'] > 1].copy()
    late_matches     = corrections[corrections['cashier_amount_db'] <= 1].copy()

    if not late_matches.empty:
        print("")
        print("      ℹ️  " + str(len(late_matches)) + " late cashier matches (0 → value). Updating silently...")
        # Merge these into corrections so they still get reloaded below
        real_corrections = corrections.copy()

    if not real_corrections.empty:
        n = len(real_corrections)
        print("")
        print("      ⚠️  " + str(n) + " CORRECTIONS DETECTED (cashier amount changed):")

        # ── Print summary by location (not every single row) ─────────────
        summary = (
            real_corrections
            .groupby('location')
            .agg(
                txn_count=('transaction_id', 'count'),
                total_old=('cashier_amount_db', 'sum'),
                total_new=('cashier_amount_new', 'sum'),
            )
            .reset_index()
        )
        print("      " + "-" * 72)
        print("      {:<20} {:>10} {:>14} {:>14} {:>10}".format(
            "Location", "Txns", "Old Total", "New Total", "Diff"
        ))
        print("      " + "-" * 72)
        for _, r in summary.iterrows():
            diff = r['total_new'] - r['total_old']
            sign = "+" if diff >= 0 else ""
            print("      {:<20} {:>10,} KES {:>10,.0f} KES {:>10,.0f}  {}{}".format(
                str(r['location']), int(r['txn_count']),
                r['total_old'], r['total_new'],
                sign, abs(diff)
            ))
        print("      " + "-" * 72)
        print("      (Run corrections_log.csv for full row-level detail)")

        # ── Delete corrected rows using a temp table (avoids stack overflow) ──
        df_corrections = df.merge(
            real_corrections[['location', 'transaction_id']],
            on=['location', 'transaction_id'],
            how='inner'
        )
        with engine.begin() as conn:
            conn.execute(text("""
                CREATE TEMP TABLE _corrections_to_delete (
                    location TEXT,
                    transaction_id TEXT
                ) ON COMMIT DROP;
            """))
            # Insert the pairs in chunks of 1000
            chunk_size = 1000
            pairs_list = list(real_corrections[['location', 'transaction_id']].itertuples(index=False))
            for i in range(0, len(pairs_list), chunk_size):
                batch = pairs_list[i:i + chunk_size]
                values_sql = ", ".join(
                    [f"('{r.location}', '{r.transaction_id}')" for r in batch]
                )
                conn.execute(text(
                    f"INSERT INTO _corrections_to_delete VALUES {values_sql};"
                ))
            conn.execute(text(f"""
                DELETE FROM {TABLE_LINEITEMS} t
                USING _corrections_to_delete d
                WHERE t.location = d.location
                  AND t.transaction_id = d.transaction_id;
            """))

        # Reinsert with corrected values
        df_corrections.to_sql(
            TABLE_LINEITEMS, engine, if_exists='append',
            index=False, method='multi', chunksize=5000,
        )
        print("      ✅ " + str(n) + " corrected transactions updated in DB")

        # ── Save corrections log (full detail goes here, not the screen) ──
        corrections_log = Path(str(ETL_FULL_HIST_FILE).replace(
            ETL_FULL_HIST_FILE.name, 'corrections_log.csv'
        ))
        real_corrections['detected_at'] = pd.Timestamp.now()
        if corrections_log.exists():
            existing_log = pd.read_csv(corrections_log)
            real_corrections = pd.concat([existing_log, real_corrections], ignore_index=True)
        real_corrections.to_csv(corrections_log, index=False)
        print("      📋 Full detail logged to: " + corrections_log.name)

    else:
        print("      ✅ No corrections detected — all amounts match")


def load_transactions_to_db(df: pd.DataFrame, engine):
    """
    Loads transactions into fact_sales_transactions.
    Anti-join: fetches existing (location, transaction_id) pairs from DB
    and only inserts rows that are genuinely new — same pattern as line items.
    ON CONFLICT DO UPDATE is kept as a safety net for the rare case where
    two pipeline runs overlap.
    """
    if df.empty:
        print("      ⚠️  No transactions to load.")
        return

    # ── STEP 1: Fetch existing keys from DB ──────────────────────
    print("      🔍 Fetching existing transactions from DB...")
    existing = pd.read_sql(
        f"SELECT location, transaction_id FROM {TABLE_TRANSACTIONS}",
        engine
    )

    if existing.empty:
        new_txns = df.copy()
    else:
        merged = df.merge(
            existing[['location', 'transaction_id']],
            on=['location', 'transaction_id'],
            how='left',
            indicator=True
        )
        new_txns = merged[merged['_merge'] == 'left_only'].drop(columns='_merge')

    if new_txns.empty:
        print("      ✅ No new transactions to insert — DB is up to date")
        return

    print(f"      ➕ {len(new_txns):,} new transactions to insert (skipping {len(df) - len(new_txns):,} already in DB)")

    # ── STEP 2: Insert only new rows ─────────────────────────────
    tmp_table = "_tmp_transactions"
    cols = [c for c in new_txns.columns if c not in ('id', 'loaded_at')]
    col_list    = ', '.join(cols)
    update_list = ', '.join([f"{c} = EXCLUDED.{c}" for c in cols if c not in ('location', 'transaction_id')])

    with engine.begin() as conn:
        new_txns[cols].to_sql(tmp_table, conn, if_exists='replace', index=False, method='multi', chunksize=5000)

        conn.execute(text(f"""
            INSERT INTO {TABLE_TRANSACTIONS} ({col_list})
            SELECT {col_list} FROM {tmp_table}
            ON CONFLICT (location, transaction_id)
            DO UPDATE SET {update_list}, loaded_at = NOW();
        """))

        conn.execute(text(f"DROP TABLE IF EXISTS {tmp_table};"))

    print(f"      ✅ {len(new_txns):,} transaction rows inserted into {TABLE_TRANSACTIONS}")


def refresh_materialized_views(engine):
    """
    Refreshes mv_transaction_master then mv_client_list.
    Uses standard REFRESH (not CONCURRENTLY) — safe for batch pipeline use.
    CONCURRENTLY requires a unique index on the view itself; standard refresh
    does not. The view locks briefly (~seconds) during refresh which is fine
    since nothing queries it mid-pipeline.
    """
    print("\n   🔄 Refreshing materialized views...")
    views = ["mv_transaction_master", "mv_client_list"]
    with engine.begin() as conn:
        for view in views:
            try:
                conn.execute(text(f"REFRESH MATERIALIZED VIEW {view};"))
                print(f"      ✅ {view} refreshed")
            except Exception as e:
                print(f"      ⚠️  Could not refresh {view}: {e}")
                print(f"         (Non-fatal — view may not exist yet, run warehouse_setup.sql first)")


# ==========================================
# 6. MAIN
# ==========================================

def run_pos_loader(source_file: Path = None, mode: str = 'upsert'):
    """
    Full loader pipeline:
      1. Read ETL CSV output
      2. Rename columns to snake_case
      3. Build line items DataFrame
      4. Build transactions DataFrame (aggregated)
      5. Load both to Postgres
      6. Refresh materialized views

    Args:
        source_file: Path to the ETL CSV. Defaults to SOURCE_FILE.
        mode: 'upsert' (safe re-runs) or 'replace' (full reload).
    """
    print("\n" + "="*60)
    print("🐘 POS → POSTGRES LOADER")
    print("="*60)

    file_to_load = source_file or SOURCE_FILE

    if not file_to_load.exists():
        print(f"   ❌ Source file not found: {file_to_load}")
        print(f"      Run etl.py first.")
        return

    # --- Connect first so chunks can be loaded as they are built ---
    print("\n   Connecting to PostgreSQL...")
    try:
        engine = create_engine(DB_CONNECTION_STRING)
    except Exception as e:
        print(f"\n   Connection failed: {e}")
        raise e

    # --- Read & process in chunks to avoid OOM on 900k row file ---
    print(f"   Reading: {file_to_load.name} (chunked)")
    CHUNK_SIZE = 100000
    total_lineitems = 0
    total_transactions = 0
    chunk_num = 0

    try:
        for df_raw in pd.read_csv(file_to_load, low_memory=False, chunksize=CHUNK_SIZE):
            chunk_num += 1
            print(f"\n   Chunk {chunk_num}: {len(df_raw):,} rows")

            df = rename_csv_cols(df_raw)
            del df_raw
            gc.collect()

            df_lineitems = build_lineitems(df)
            del df
            gc.collect()

            df_transactions = build_transactions(df_lineitems)

            load_lineitems_to_db(df_lineitems, engine, mode=mode)
            load_transactions_to_db(df_transactions, engine)

            total_lineitems += len(df_lineitems)
            total_transactions += len(df_transactions)

            del df_lineitems, df_transactions
            gc.collect()

            # After first chunk, switch mode to upsert so subsequent
            # chunks don't wipe what was just loaded
            if mode == 'replace':
                mode = 'upsert'

        refresh_materialized_views(engine)

        print(f"\n POS LOADER COMPLETE.")
        print(f"   fact_sales_lineitems    : {total_lineitems:,} rows")
        print(f"   fact_sales_transactions : {total_transactions:,} rows")

    except Exception as e:
        print(f"\n   Loader failed: {e}")
        raise e
    finally:
        try:
            engine.dispose()
        except:
            pass


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Load POS ETL output to PostgreSQL")
    parser.add_argument(
        '--mode',
        choices=['upsert', 'replace'],
        default='upsert',
        help="upsert = safe re-run (default) | replace = full reload"
    )
    parser.add_argument(
        '--full-history',
        action='store_true',
        help="Load full history file instead of recent file"
    )
    args = parser.parse_args()

    source = ETL_FULL_HIST_FILE if args.full_history else ETL_OUTPUT_FILE
    run_pos_loader(source_file=source, mode=args.mode)