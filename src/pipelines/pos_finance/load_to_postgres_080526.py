"""
load_to_postgres.py
===================
Reads clean data from staging tables and loads into fact tables:
    - fact_sales_lineitems      (one row per sales line item)
    - fact_sales_transactions   (one row per transaction, aggregated)
    - fact_inventory_snapshot   (one row per product per branch per date)

Then refreshes:
    - mv_transaction_master
    - mv_client_list

Key behaviours:
    - Reads from stg_sales_reports / stg_cashier_reports / stg_qty_list
    - Cashier dedup key: Receipt Txn No + Amount + Phone Number + Txn Type + Time
    - Time kept as plain string throughout — never parsed
    - POS_Txn_Sum grouped by Transaction ID + Date Sold + Client Name
    - real_transaction_value = POS_Txn_Sum always (cashier amount unreliable)
    - Phone numbers normalised via normalize_phone() after merge
    - All three fact tables use ON CONFLICT DO UPDATE — safe to rerun incrementally
    - Upserts only write to disk when column values have actually changed (WHERE IS DISTINCT FROM)
    - Cashier staging filtered by the same watermark as sales staging

Branch name normalisation:
    Staging tables use raw SharePoint names e.g. 'Portal CBD', 'Galleria'
    Fact tables use normalised labels e.g. 'PORTAL_CBD', 'GALLERIA'
    (consistent with existing data in the DB)
"""

from __future__ import annotations

import gc
import os
import sys
import warnings
from datetime import datetime, date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import execute_values
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool

from Portal_ML_V4.src.utils.name_cleaner import clean_name_series
from Portal_ML_V4.src.utils.phone import normalize_phone
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

try:
    from Portal_ML_V4.src.config.settings import BASE_DIR, PROCESSED_DATA_DIR, POWERBI_CACHE_DIR
except ImportError:
    BASE_DIR          = Path(os.getcwd())
    PROCESSED_DATA_DIR = BASE_DIR / "data" / "03_processed"


sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# ── Config ────────────────────────────────────────────────────────────────────

env_path = os.getenv("ENV_FILE") or str(BASE_DIR / ".env")
load_dotenv(dotenv_path=env_path, override=True)

OUTPUT_DIR   = PROCESSED_DATA_DIR / "pos_data"
DISCREPANCY_LOG = OUTPUT_DIR / "discrepancy_log.csv"
WATERMARK_LOOKBACK_DAYS = max(0, int(os.getenv("POS_WATERMARK_LOOKBACK_DAYS", "1")))
MERGE_CHUNK_SIZE = max(1, int(os.getenv("POS_MERGE_CHUNK_SIZE", "50000")))
UPSERT_PAGE_SIZE = max(100, int(os.getenv("POS_UPSERT_PAGE_SIZE", "5000")))
EXPORT_POWERBI_CACHE_ENABLED = (
    os.getenv("EXPORT_POWERBI_CACHE", "1").strip().lower()
    not in {"0", "false", "no"}
)
FORCE_POWERBI_CACHE_EXPORT = (
    os.getenv("FORCE_POWERBI_CACHE_EXPORT", "0").strip().lower()
    in {"1", "true", "yes"}
)

DB_CONFIG = {
    "host":     os.getenv("DB_HOST",  "localhost"),
    "port":     int(os.getenv("DB_PORT", 5432)),
    "database": os.getenv("DB_NAME"),
    "user":     os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
}

# Maps staging branch names → normalised fact table location labels
# Add new branches here as the org grows
BRANCH_LABEL_MAP = {
    "Galleria":     "GALLERIA",
    "ABC":          "PHARMART_ABC",
    "Milele":       "NGONG_MILELE",
    "Portal 2R":    "PORTAL_2R",
    "Portal CBD":   "PORTAL_CBD",
    "Centurion 2R": "CENTURION_2R",
}

# Cashier dedup key — agreed:
# same receipt + same amount + same client + same payment type + same time = duplicate
CASHIER_DEDUP_COLS = [
    "receipt_txn_no",
    "amount",
    "phone_number",
    "txn_type",
    "txn_time",
]

# Stronger sales-line signature used to remove duplicated staging rows.
# This keeps legitimate multi-item transactions while dropping repeated loads.
SALES_LINE_DEDUP_COLS = [
    "transaction_id",
    "date_sold",
    "_dedup_item",
    "_dedup_description",
    "_dedup_qty_sold",
    "_dedup_total_tax_ex",
]


def branch_to_location(branch: str) -> str:
    return BRANCH_LABEL_MAP.get(branch, branch.upper().replace(" ", "_"))


def qty_watermark_key(branch: str) -> str:
    return f"QTY::{branch_to_location(branch)}"


def watermark_start_date(max_date: date | None) -> date | None:
    if max_date is None:
        return None
    return max_date - timedelta(days=WATERMARK_LOOKBACK_DAYS)


# ── DB helpers ────────────────────────────────────────────────────────────────

def get_conn() -> psycopg2.extensions.connection:
    return psycopg2.connect(
        host     = DB_CONFIG["host"],
        port     = DB_CONFIG["port"],
        dbname   = DB_CONFIG["database"],
        user     = DB_CONFIG["user"],
        password = DB_CONFIG["password"],
        sslmode = "disable"
        # sslmode  = "require"
    )


def get_engine():
    return create_engine(
        "postgresql+psycopg2://",
        creator   = get_conn,
        poolclass = NullPool,
    )




def get_watermark(key: str) -> date | None:
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT max_date_loaded FROM fact_load_watermarks WHERE branch = %s",
                    (key,)
                )
                row = cur.fetchone()
                return row[0] if row else None
    except Exception as e:
        print(f"    Could not read watermark for {key}: {e}")
        return None


def update_watermark(key: str, max_date) -> None:
    if max_date is None:
        return
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO fact_load_watermarks (branch, max_date_loaded, last_updated_at)
                    VALUES (%s, %s, NOW())
                    ON CONFLICT (branch) DO UPDATE SET
                        max_date_loaded = GREATEST(
                            fact_load_watermarks.max_date_loaded,
                            EXCLUDED.max_date_loaded
                        ),
                        last_updated_at = NOW()
                """, (key, max_date))
            conn.commit()
        print(f"    Watermark updated: {key} -> {max_date}")
    except Exception as e:
        print(f"    Could not update watermark for {key}: {e}")


def get_qty_watermark(branch: str) -> date | None:
    return get_watermark(qty_watermark_key(branch))


def update_qty_watermark(branch: str, max_date) -> None:
    update_watermark(qty_watermark_key(branch), max_date)


def ensure_runtime_db_objects() -> None:
    """Create lightweight runtime helpers the loader depends on."""
    ddl = [
        """
        CREATE TABLE IF NOT EXISTS fact_load_watermarks (
            branch TEXT PRIMARY KEY,
            max_date_loaded DATE NOT NULL,
            last_updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_stg_sales_branch_date_sold
        ON stg_sales_reports (branch, date_sold)
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_stg_cashier_branch_transaction_date
        ON stg_cashier_reports (branch, transaction_date)
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_stg_qty_branch_snapshot_date
        ON stg_qty_list (branch, snapshot_date)
        """,
    ]
    try:
        with get_conn() as conn:
            conn.autocommit = True
            with conn.cursor() as cur:
                for statement in ddl:
                    cur.execute(statement)
    except Exception as e:
        print(f"    Could not ensure runtime DB objects: {e}")


def dataframe_to_rows(df: pd.DataFrame, columns: list[str]) -> list[tuple]:
    """Convert a dataframe slice to Python tuples with NULL-safe values."""
    # reindex() materializes any missing optional columns as all-null so
    # branches with no cashier-side fields can still serialize cleanly.
    payload = df.reindex(columns=columns).astype(object)
    payload = payload.where(pd.notnull(payload), None)
    return list(payload.itertuples(index=False, name=None))


def execute_values_count(cur, sql: str, rows: list[tuple], page_size: int) -> int:
    """
    Execute batched VALUES upserts and return the number of rows Postgres
    actually inserted or updated across all batches.
    """
    if not rows:
        return 0

    total_affected = 0
    for start in range(0, len(rows), page_size):
        batch = rows[start : start + page_size]
        execute_values(cur, sql, batch, page_size=len(batch))
        total_affected += max(cur.rowcount, 0)
    return total_affected

# ── Read from watermark ───────────────────────────────────────────────────────
def get_fact_watermark(branch: str) -> date | None:
    """Read the latest date already loaded into fact_sales_transactions."""
    return get_watermark(branch_to_location(branch))


def update_fact_watermark(branch: str, max_date) -> None:
    """Update fact_load_watermarks after successful load."""
    update_watermark(branch_to_location(branch), max_date)


def read_sales_staging(engine, branch: str, after_date=None) -> pd.DataFrame:
    if after_date:
        sql = text("""
            SELECT
                branch, department, category, item, description,
                on_hand, last_sold, qty_sold, total_tax_ex,
                transaction_id, date_sold, sale_time, sale_datetime
            FROM stg_sales_reports
            WHERE branch = :branch
              AND date_sold >= :after_date
        """)
        df = pd.read_sql(sql, engine, params={"branch": branch, "after_date": after_date})
    else:
        sql = text("""
            SELECT
                branch, department, category, item, description,
                on_hand, last_sold, qty_sold, total_tax_ex,
                transaction_id, date_sold, sale_time, sale_datetime
            FROM stg_sales_reports
            WHERE branch = :branch
        """)
        df = pd.read_sql(sql, engine, params={"branch": branch})

    print(f"    📥 Sales staging rows: {len(df):,}")
    return df


def read_cashier_staging(engine, branch: str, after_date=None) -> pd.DataFrame:
    if after_date:
        sql = text("""
            SELECT
                branch, transaction_date, receipt_txn_no, amount, txn_costs,
                txn_time, txn_type, ordered_via, client_name, phone_number, sales_rep
            FROM stg_cashier_reports
            WHERE branch = :branch
              AND transaction_date >= :after_date
        """)
        df = pd.read_sql(sql, engine, params={"branch": branch, "after_date": after_date})
    else:
        sql = text("""
            SELECT
                branch, transaction_date, receipt_txn_no, amount, txn_costs,
                txn_time, txn_type, ordered_via, client_name, phone_number, sales_rep
            FROM stg_cashier_reports
            WHERE branch = :branch
        """)
        df = pd.read_sql(sql, engine, params={"branch": branch})

    print(f"     Cashier staging rows: {len(df):,}")
    return df


def read_qty_staging(engine, branch: str, after_date=None) -> pd.DataFrame:
    """
    Read qty list rows for a branch from stg_qty_list.
    Includes snapshot_date and snapshot_date_source for traceability.
    """
    if after_date:
        sql = text("""
            SELECT
                source_file_id,
                branch,
                snapshot_date,
                snapshot_date_source,
                department,
                category,
                item_lookup_code,
                description,
                on_hand,
                committed,
                reorder_pt,
                restock_lvl,
                qty_to_order,
                supplier,
                reorder_no
            FROM stg_qty_list
            WHERE branch = :branch
              AND snapshot_date >= :after_date
        """)
        df = pd.read_sql(sql, engine, params={"branch": branch, "after_date": after_date})
    else:
        sql = text("""
            SELECT
                source_file_id,
                branch,
                snapshot_date,
                snapshot_date_source,
                department,
                category,
                item_lookup_code,
                description,
                on_hand,
                committed,
                reorder_pt,
                restock_lvl,
                qty_to_order,
                supplier,
                reorder_no
            FROM stg_qty_list
            WHERE branch = :branch
        """)
        df = pd.read_sql(sql, engine, params={"branch": branch})
    print(f"    Qty staging rows: {len(df):,}")
    return df


# ── Cashier preparation ───────────────────────────────────────────────────────

def dedup_sales_lineitems(df_sales: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicate sales rows on a strict, normalized line-item key.
    This protects against repeated staging loads of the same source rows.
    """
    if df_sales.empty:
        return df_sales

    df_sales = df_sales.copy()

    if "transaction_id" in df_sales.columns:
        df_sales["transaction_id"] = (
            df_sales["transaction_id"].astype(str)
            .str.replace(r"\.0$", "", regex=True)
            .str.strip()
        )

    df_sales["_dedup_item"] = (
        df_sales.get("item", pd.Series("", index=df_sales.index))
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )
    df_sales["_dedup_description"] = (
        df_sales.get("description", pd.Series("", index=df_sales.index))
        .fillna("")
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .str.upper()
    )
    df_sales["_dedup_qty_sold"] = pd.to_numeric(
        df_sales.get("qty_sold", pd.Series(0, index=df_sales.index)),
        errors="coerce",
    ).fillna(0).round(4)
    df_sales["_dedup_total_tax_ex"] = pd.to_numeric(
        df_sales.get("total_tax_ex", pd.Series(0, index=df_sales.index)),
        errors="coerce",
    ).fillna(0).round(2)

    dedup_cols = [c for c in SALES_LINE_DEDUP_COLS if c in df_sales.columns]
    before = len(df_sales)
    df_sales = df_sales.drop_duplicates(subset=dedup_cols, keep="first")
    removed = before - len(df_sales)
    if removed > 0:
        print(f"    Sales line dedup: {removed:,} duplicate rows removed")

    df_sales.drop(
        columns=[
            "_dedup_item",
            "_dedup_description",
            "_dedup_qty_sold",
            "_dedup_total_tax_ex",
        ],
        inplace=True,
        errors="ignore",
    )

    return df_sales


def prepare_cashier(df_cashier: pd.DataFrame) -> pd.DataFrame:
    """
    1. Dedup on agreed key (Receipt Txn No + Amount + Phone Number +
       Txn Type + Time)
    2. Clean Client Name
    3. Aggregate to one row per Receipt Txn No:
       Amount → sum, Txn Costs → sum, Time → min (as string),
       everything else → first
    """
    if df_cashier.empty:
        return df_cashier

    # ── Clean amount columns ──────────────────────────────────────────────────
    for col in ["amount", "txn_costs"]:
        if col in df_cashier.columns:
            df_cashier[col] = (
                df_cashier[col].astype(str)
                .str.replace(",", "", regex=False)
                .str.replace(r"[^\d\.\-]", "", regex=True)
            )
            df_cashier[col] = pd.to_numeric(df_cashier[col], errors="coerce").fillna(0.0)

    # ── Clean Client Name ─────────────────────────────────────────────────────
    if "client_name" in df_cashier.columns:
        df_cashier["client_name"] = clean_name_series(df_cashier["client_name"])

    # Normalize dedup-key fields before dropping duplicates.
    if "receipt_txn_no" in df_cashier.columns:
        df_cashier["receipt_txn_no"] = (
            df_cashier["receipt_txn_no"].astype(str)
            .str.replace(r"\.0$", "", regex=True)
            .str.strip()
        )
    if "phone_number" in df_cashier.columns:
        df_cashier["phone_number"] = df_cashier["phone_number"].apply(normalize_phone)
    if "txn_type" in df_cashier.columns:
        df_cashier["txn_type"] = (
            df_cashier["txn_type"].fillna("").astype(str).str.strip().str.upper()
        )
    if "txn_time" in df_cashier.columns:
        df_cashier["txn_time"] = (
            df_cashier["txn_time"].fillna("").astype(str).str.strip()
        )

    # ── Dedup — agreed key ────────────────────────────────────────────────────
    # txn_time is already a plain string from staging — no parsing needed
    dedup_cols = [c for c in CASHIER_DEDUP_COLS if c in df_cashier.columns]
    before = len(df_cashier)
    df_cashier = df_cashier.drop_duplicates(subset=dedup_cols, keep="first")
    removed = before - len(df_cashier)
    if removed > 0:
        print(f"     Cashier dedup: {removed:,} duplicate rows removed")

    # ── Aggregate to one row per Receipt Txn No ───────────────────────────────
    if "receipt_txn_no" not in df_cashier.columns:
        return df_cashier

    agg_rules = {
        "amount":    "sum",
        "txn_costs": "sum",
    }
    # Time kept as string — take min lexicographically (earliest as string)
    if "txn_time" in df_cashier.columns:
        agg_rules["txn_time"] = "min"

    # All other columns → first value
    meta_cols = [
        c for c in df_cashier.columns
        if c not in ["receipt_txn_no", "amount", "txn_costs", "txn_time"]
    ]
    for c in meta_cols:
        agg_rules[c] = "first"

    df_cashier = df_cashier.groupby("receipt_txn_no", as_index=False).agg(agg_rules)
    print(f"     Cashier after aggregation: {len(df_cashier):,} unique receipts")

    return df_cashier


# ── Merge & transform ─────────────────────────────────────────────────────────

def merge_and_transform(
    df_sales: pd.DataFrame,
    df_cashier: pd.DataFrame,
    branch_label: str,
    output_dir: Path,
) -> pd.DataFrame:
    """
    Merge sales + cashier, dedup, calculate POS_Txn_Sum, set audit fields.
    Returns df_merged ready for loading into fact tables.
    """
    # ── Clean transaction_id ──────────────────────────────────────────────────
    df_sales = dedup_sales_lineitems(df_sales)

    # ── Chunked merge to prevent memory spikes ────────────────────────────────
    if not df_cashier.empty:
        print("    🔗 Running chunked merge...")
        chunks     = []
        chunk_size = MERGE_CHUNK_SIZE

        for i in range(0, len(df_sales), chunk_size):
            chunk = df_sales.iloc[i : i + chunk_size]
            merged_chunk = pd.merge(
                chunk,
                df_cashier,
                left_on  = "transaction_id",
                right_on = "receipt_txn_no",
                how      = "left",
                sort     = False,
            )
            # Prevent ArrowString/category OOM when stacking chunks
            object_like_cols = merged_chunk.select_dtypes(
                include=["string", "category"]
            ).columns
            if len(object_like_cols) > 0:
                merged_chunk[object_like_cols] = merged_chunk[object_like_cols].astype(object)

            chunks.append(merged_chunk)
            del merged_chunk

        df_merged = pd.concat(chunks, ignore_index=True)
        del chunks
        gc.collect()

        df_merged["audit_status"] = np.where(
            df_merged["receipt_txn_no"].isna(), "No Cashier Data", "Matched"
        )
    else:
        df_merged = df_sales.copy()
        df_merged["audit_status"] = "No Cashier Data"

    df_merged["location"] = branch_label

    # ── Normalise phone numbers ────────────────────────────────────────────
    # phone_number is already normalised in prepare_cashier(), so avoid a
    # second full-column Python apply here.

    # ── Derive sale_date from date_sold (already a DATE from staging) ─────────
    df_merged["sale_date"]     = pd.to_datetime(df_merged["date_sold"], errors="coerce")
    df_merged["sale_date_str"] = df_merged["sale_date"].dt.strftime("%Y-%m-%d")
    df_merged["sale_date"]     = df_merged["sale_date"].dt.date

    # ── Branch-level dedup ────────────────────────────────────────────────────
    df_merged["_dedup_item"] = (
        df_merged.get("item", pd.Series("", index=df_merged.index))
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )
    df_merged["_dedup_description"] = (
        df_merged.get("description", pd.Series("", index=df_merged.index))
        .fillna("")
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .str.upper()
    )
    df_merged["_dedup_qty_sold"] = pd.to_numeric(
        df_merged.get("qty_sold", pd.Series(0, index=df_merged.index)),
        errors="coerce",
    ).fillna(0).round(4)
    df_merged["_dedup_total_tax_ex"] = pd.to_numeric(
        df_merged.get("total_tax_ex", pd.Series(0, index=df_merged.index)),
        errors="coerce",
    ).fillna(0).round(2)

    dedup_cols = [c for c in SALES_LINE_DEDUP_COLS if c in df_merged.columns]
    actual_dedup = [c for c in dedup_cols if c in df_merged.columns]
    before = len(df_merged)
    df_merged.drop_duplicates(subset=actual_dedup, inplace=True)
    print(f"    🧹 Branch dedup: {before - len(df_merged):,} duplicates removed")
    df_merged.drop(
        columns=[
            "_dedup_item",
            "_dedup_description",
            "_dedup_qty_sold",
            "_dedup_total_tax_ex",
        ],
        inplace=True,
        errors="ignore",
    )

    # ── POS_Txn_Sum — agreed groupby key ─────────────────────────────────────
    # Transaction ID + Date Sold + Client Name
    # Client Name nulls filled with 'UNKNOWN' so they group correctly
    df_merged["total_tax_ex"] = pd.to_numeric(
        df_merged["total_tax_ex"], errors="coerce"
    ).fillna(0)

    if "amount" not in df_merged.columns:
        df_merged["amount"] = 0.0
    else:
        df_merged["amount"] = pd.to_numeric(
            df_merged["amount"], errors="coerce"
        ).fillna(0.0)

    df_merged["_client_for_groupby"] = df_merged.get("client_name", pd.Series()).fillna("UNKNOWN")

    df_merged["pos_txn_sum"] = df_merged.groupby(
        ["transaction_id", "date_sold", "_client_for_groupby"]
    )["total_tax_ex"].transform("sum")

    # real_transaction_value = always POS sum (cashier amount unreliable)
    df_merged["real_transaction_value"] = df_merged["pos_txn_sum"]

    df_merged = df_merged.drop(columns=["_client_for_groupby"])

    # ── Discrepancy audit log ─────────────────────────────────────────────────
    # Log transactions where cashier amount != POS sum (>1 KES tolerance)
    audit_cols = [
        c for c in
        ["transaction_id", "sale_date", "description", "amount",
         "pos_txn_sum", "phone_number", "client_name"]
        if c in df_merged.columns
    ]
    audit_src = df_merged[audit_cols].drop_duplicates(subset=["transaction_id"])
    discrepancy_mask = (
        (audit_src["amount"] != 0) &
        (abs(audit_src["pos_txn_sum"] - audit_src["amount"]) > 1)
    )
    discrepant = audit_src[discrepancy_mask].copy()
    if not discrepant.empty:
        discrepant["location"]    = branch_label
        discrepant["discrepancy"] = discrepant["pos_txn_sum"] - discrepant["amount"]
        write_header = not output_dir.exists() or not DISCREPANCY_LOG.exists()
        os.makedirs(output_dir, exist_ok=True)
        discrepant.to_csv(
            DISCREPANCY_LOG, mode="a", index=False, header=write_header
        )
        print(f"      Discrepancies logged: {len(discrepant):,} transactions")
    else:
        print("     No discrepancies found")

    return df_merged


# ── Load into fact tables ─────────────────────────────────────────────────────

def load_fact_lineitems(cur, df: pd.DataFrame) -> int:
    """
    Insert rows into fact_sales_lineitems.
    One row per sales line item (most granular level).
    """

    # Dedup before insert — prevents ON CONFLICT hitting same row twice
    before = len(df)
    df = df.drop_duplicates(
        subset=["location", "transaction_id", "description", "total_tax_ex"],
        keep="last"
    )
    removed = before - len(df)
    if removed > 0:
        print(f"     Lineitems dedup: {removed:,} duplicate rows removed before insert")

    INSERT_SQL = """
        INSERT INTO fact_sales_lineitems (
            location, transaction_id,
            department, category, item, description,
            qty_sold, total_tax_ex,
            date_sold, sale_date, sale_date_str,
            client_name, phone_number, sales_rep,
            txn_type, ordered_via,
            cashier_amount, transaction_total, audit_status
        ) VALUES %s
        ON CONFLICT (location, transaction_id, description, total_tax_ex) DO UPDATE SET
            department        = EXCLUDED.department,
            category          = EXCLUDED.category,
            item              = EXCLUDED.item,
            qty_sold          = EXCLUDED.qty_sold,
            date_sold         = EXCLUDED.date_sold,
            sale_date         = EXCLUDED.sale_date,
            sale_date_str     = EXCLUDED.sale_date_str,
            client_name       = EXCLUDED.client_name,
            phone_number      = EXCLUDED.phone_number,
            sales_rep         = EXCLUDED.sales_rep,
            txn_type          = EXCLUDED.txn_type,
            ordered_via       = EXCLUDED.ordered_via,
            cashier_amount    = EXCLUDED.cashier_amount,
            transaction_total = EXCLUDED.transaction_total,
            audit_status      = EXCLUDED.audit_status,
            loaded_at         = NOW()
        WHERE
            fact_sales_lineitems.department        IS DISTINCT FROM EXCLUDED.department
            OR fact_sales_lineitems.category          IS DISTINCT FROM EXCLUDED.category
            OR fact_sales_lineitems.item              IS DISTINCT FROM EXCLUDED.item
            OR fact_sales_lineitems.qty_sold          IS DISTINCT FROM EXCLUDED.qty_sold
            OR fact_sales_lineitems.date_sold         IS DISTINCT FROM EXCLUDED.date_sold
            OR fact_sales_lineitems.sale_date         IS DISTINCT FROM EXCLUDED.sale_date
            OR fact_sales_lineitems.sale_date_str     IS DISTINCT FROM EXCLUDED.sale_date_str
            OR fact_sales_lineitems.client_name       IS DISTINCT FROM EXCLUDED.client_name
            OR fact_sales_lineitems.phone_number      IS DISTINCT FROM EXCLUDED.phone_number
            OR fact_sales_lineitems.sales_rep         IS DISTINCT FROM EXCLUDED.sales_rep
            OR fact_sales_lineitems.txn_type          IS DISTINCT FROM EXCLUDED.txn_type
            OR fact_sales_lineitems.ordered_via       IS DISTINCT FROM EXCLUDED.ordered_via
            OR fact_sales_lineitems.cashier_amount    IS DISTINCT FROM EXCLUDED.cashier_amount
            OR fact_sales_lineitems.transaction_total IS DISTINCT FROM EXCLUDED.transaction_total
            OR fact_sales_lineitems.audit_status      IS DISTINCT FROM EXCLUDED.audit_status
    """


    # ── Vectorized column prep — eliminates per-row Python overhead ───────────
    for col in ("qty_sold", "total_tax_ex", "amount", "pos_txn_sum"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # date_sold kept as audit string; sale_date already a Python date from merge
    df["_date_sold_str"] = df["date_sold"].astype(str).where(
        df["date_sold"].notna(), ""
    )

    load_cols = [
        "location",
        "transaction_id",
        "department",
        "category",
        "item",
        "description",
        "qty_sold",
        "total_tax_ex",
        "_date_sold_str",
        "sale_date",
        "sale_date_str",
        "client_name",
        "phone_number",
        "sales_rep",
        "txn_type",
        "ordered_via",
        "amount",
        "pos_txn_sum",
        "audit_status",
    ]
    rows = dataframe_to_rows(df, load_cols)
    return execute_values_count(cur, INSERT_SQL, rows, page_size=UPSERT_PAGE_SIZE)


def load_fact_transactions(cur, df: pd.DataFrame) -> int:
    """
    Aggregate line items to transaction level and insert into
    fact_sales_transactions. One row per (location, transaction_id).
    Handles the UNIQUE constraint with ON CONFLICT DO UPDATE.
    """

    CASHIER_COLS = ["client_name", "phone_number", "sales_rep", "txn_type", "ordered_via"]

    for col in CASHIER_COLS:
        if col not in df.columns:
            df[col] = None
    # Aggregate per transaction
    agg_dict = {
    "sale_date":              ("sale_date",              "first"),
    "sale_date_str":          ("sale_date_str",          "first"),
    "client_name":            ("client_name",            "first"),
    "phone_number":           ("phone_number",           "first"),
    "sales_rep":              ("sales_rep",              "first"),
    "txn_type":               ("txn_type",               "first"),
    "ordered_via":            ("ordered_via",            "first"),
    "pos_txn_sum":            ("pos_txn_sum",            "first"),
    "cashier_amount":         ("amount",                 "first"),
    "real_transaction_value": ("real_transaction_value", "first"),
    "products_in_txn":        ("description",            lambda x: " | ".join(
                                  pd.unique(x.dropna().astype(str))
                              )),
    "item_count":             ("description",            "count"),
    "audit_status":           ("audit_status",           "first"),
    }

    # Only add time columns if they exist in df
    if "sale_time" in df.columns:
        agg_dict["sale_time"]     = ("sale_time",     "first")
    if "sale_datetime" in df.columns:
        agg_dict["sale_datetime"] = ("sale_datetime", "first")

    agg = df.groupby(["location", "transaction_id"], as_index=False).agg(**agg_dict)

    # Ensure columns exist even if not in staging yet
    if "sale_time" not in agg.columns:
        agg["sale_time"] = None
    if "sale_datetime" not in agg.columns:
        agg["sale_datetime"] = None


    INSERT_SQL = """
    INSERT INTO fact_sales_transactions (
        location, transaction_id,
        sale_date, sale_date_str,
        sale_time, sale_datetime,
        client_name, phone_number, sales_rep,
        txn_type, ordered_via,
        pos_txn_sum, cashier_amount, real_transaction_value,
        products_in_txn, item_count, audit_status
    ) VALUES %s
    ON CONFLICT (location, transaction_id) DO UPDATE SET
        sale_date              = EXCLUDED.sale_date,
        sale_date_str          = EXCLUDED.sale_date_str,
        sale_time              = EXCLUDED.sale_time,
        sale_datetime          = EXCLUDED.sale_datetime,
        client_name            = EXCLUDED.client_name,
        phone_number           = EXCLUDED.phone_number,
        sales_rep              = EXCLUDED.sales_rep,
        txn_type               = EXCLUDED.txn_type,
        ordered_via            = EXCLUDED.ordered_via,
        pos_txn_sum            = EXCLUDED.pos_txn_sum,
        cashier_amount         = EXCLUDED.cashier_amount,
        real_transaction_value = EXCLUDED.real_transaction_value,
        products_in_txn        = EXCLUDED.products_in_txn,
        item_count             = EXCLUDED.item_count,
        audit_status           = EXCLUDED.audit_status,
        loaded_at              = NOW()
    WHERE
        fact_sales_transactions.sale_date              IS DISTINCT FROM EXCLUDED.sale_date
        OR fact_sales_transactions.sale_date_str          IS DISTINCT FROM EXCLUDED.sale_date_str
        OR fact_sales_transactions.sale_time              IS DISTINCT FROM EXCLUDED.sale_time
        OR fact_sales_transactions.sale_datetime          IS DISTINCT FROM EXCLUDED.sale_datetime
        OR fact_sales_transactions.client_name            IS DISTINCT FROM EXCLUDED.client_name
        OR fact_sales_transactions.phone_number           IS DISTINCT FROM EXCLUDED.phone_number
        OR fact_sales_transactions.sales_rep              IS DISTINCT FROM EXCLUDED.sales_rep
        OR fact_sales_transactions.txn_type               IS DISTINCT FROM EXCLUDED.txn_type
        OR fact_sales_transactions.ordered_via            IS DISTINCT FROM EXCLUDED.ordered_via
        OR fact_sales_transactions.pos_txn_sum            IS DISTINCT FROM EXCLUDED.pos_txn_sum
        OR fact_sales_transactions.cashier_amount         IS DISTINCT FROM EXCLUDED.cashier_amount
        OR fact_sales_transactions.real_transaction_value IS DISTINCT FROM EXCLUDED.real_transaction_value
        OR fact_sales_transactions.products_in_txn        IS DISTINCT FROM EXCLUDED.products_in_txn
        OR fact_sales_transactions.item_count             IS DISTINCT FROM EXCLUDED.item_count
        OR fact_sales_transactions.audit_status           IS DISTINCT FROM EXCLUDED.audit_status
    """

    # ── Vectorized column prep ────────────────────────────────────────────────
    for col in ("pos_txn_sum", "cashier_amount", "real_transaction_value"):
        if col in agg.columns:
            agg[col] = pd.to_numeric(agg[col], errors="coerce")

    load_cols = [
        "location",
        "transaction_id",
        "sale_date",
        "sale_date_str",
        "sale_time",
        "sale_datetime",
        "client_name",
        "phone_number",
        "sales_rep",
        "txn_type",
        "ordered_via",
        "pos_txn_sum",
        "cashier_amount",
        "real_transaction_value",
        "products_in_txn",
        "item_count",
        "audit_status",
    ]
    rows = dataframe_to_rows(agg, load_cols)
    return execute_values_count(cur, INSERT_SQL, rows, page_size=UPSERT_PAGE_SIZE)


def load_fact_inventory_snapshot(cur, df: pd.DataFrame, branch_label: str) -> int:
    """
    Load qty list rows into fact_inventory_snapshot.
    Uses ON CONFLICT DO UPDATE so rerunning the pipeline updates existing
    rows rather than duplicating them.
    Branch label normalised to match fact table convention e.g. GALLERIA.
    """
    if df.empty:
        return 0

    df = df.drop_duplicates(
        subset=["item_lookup_code", "snapshot_date"], 
        keep="last"
    ).copy()

    INSERT_SQL = """
        INSERT INTO fact_inventory_snapshot (
            branch, snapshot_date, snapshot_date_source,
            department, category, item_lookup_code, description,
            on_hand, committed, reorder_pt, restock_lvl,
            qty_to_order, supplier, reorder_no,
            source_file_id
        ) VALUES %s
        ON CONFLICT (branch, item_lookup_code, snapshot_date) DO UPDATE SET
            snapshot_date_source = EXCLUDED.snapshot_date_source,
            department           = EXCLUDED.department,
            category             = EXCLUDED.category,
            description          = EXCLUDED.description,
            on_hand              = EXCLUDED.on_hand,
            committed            = EXCLUDED.committed,
            reorder_pt           = EXCLUDED.reorder_pt,
            restock_lvl          = EXCLUDED.restock_lvl,
            qty_to_order         = EXCLUDED.qty_to_order,
            supplier             = EXCLUDED.supplier,
            reorder_no           = EXCLUDED.reorder_no,
            source_file_id       = EXCLUDED.source_file_id,
            loaded_at            = NOW()
        WHERE
            fact_inventory_snapshot.snapshot_date_source IS DISTINCT FROM EXCLUDED.snapshot_date_source
            OR fact_inventory_snapshot.department           IS DISTINCT FROM EXCLUDED.department
            OR fact_inventory_snapshot.category             IS DISTINCT FROM EXCLUDED.category
            OR fact_inventory_snapshot.description          IS DISTINCT FROM EXCLUDED.description
            OR fact_inventory_snapshot.on_hand              IS DISTINCT FROM EXCLUDED.on_hand
            OR fact_inventory_snapshot.committed            IS DISTINCT FROM EXCLUDED.committed
            OR fact_inventory_snapshot.reorder_pt           IS DISTINCT FROM EXCLUDED.reorder_pt
            OR fact_inventory_snapshot.restock_lvl          IS DISTINCT FROM EXCLUDED.restock_lvl
            OR fact_inventory_snapshot.qty_to_order         IS DISTINCT FROM EXCLUDED.qty_to_order
            OR fact_inventory_snapshot.supplier             IS DISTINCT FROM EXCLUDED.supplier
            OR fact_inventory_snapshot.reorder_no           IS DISTINCT FROM EXCLUDED.reorder_no
            OR fact_inventory_snapshot.source_file_id       IS DISTINCT FROM EXCLUDED.source_file_id
    """

    # ── Vectorized column prep ────────────────────────────────────────────────
    for col in ("on_hand", "committed", "reorder_pt", "restock_lvl", "qty_to_order"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["branch"] = branch_label
    load_cols = [
        "branch",
        "snapshot_date",
        "snapshot_date_source",
        "department",
        "category",
        "item_lookup_code",
        "description",
        "on_hand",
        "committed",
        "reorder_pt",
        "restock_lvl",
        "qty_to_order",
        "supplier",
        "reorder_no",
        "source_file_id",
    ]
    rows = dataframe_to_rows(df, load_cols)
    return execute_values_count(cur, INSERT_SQL, rows, page_size=UPSERT_PAGE_SIZE)
    

# ── Type helpers ──────────────────────────────────────────────────────────────

def _safe_numeric(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_date(value):
    if value is None or pd.isna(value):
        return None
    return value


def _safe_time(value):
    """Convert time value to Python time or None. Handles pandas NaT."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _safe_datetime(value):
    """Convert datetime value to Python datetime or None. Handles pandas NaT."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def export_powerbi_cache(engine):
    """Export key tables to CSV for Power BI import mode."""
    cache_root = os.getenv("POWERBI_CACHE_DIR")
    if not cache_root:
        cache_root = str(POWERBI_CACHE_DIR) if "POWERBI_CACHE_DIR" in globals() else "D:/PowerBI_Cache"
    cache_dir = Path(cache_root)
    cache_dir.mkdir(parents=True, exist_ok=True)

    exports = {
        "vw_sales_base.csv":         "SELECT * FROM vw_sales_base",
        "vw_sales_with_margin.csv":  "SELECT * FROM vw_sales_with_margin",
        "mv_transaction_master.csv": "SELECT * FROM mv_transaction_master",
        "mv_client_list.csv":        "SELECT * FROM mv_client_list",
        "vw_dead_stock.csv":         "SELECT * FROM vw_dead_stock",
    }

    for filename, sql in exports.items():
        print(f"    Exporting {filename}...")
        df = pd.read_sql(text(sql), engine)
        df.to_csv(cache_dir / filename, index=False)
        print(f"       {len(df):,} rows -> {filename}")

def run_pos_loader():
    print("=" * 65)
    print(" POS LOADER — reading from staging tables")
    print("=" * 65)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Clear previous discrepancy log so each run starts fresh
    if DISCREPANCY_LOG.exists():
        DISCREPANCY_LOG.unlink()

    engine = get_engine()
    ensure_runtime_db_objects()

    # Get distinct branches from staging
    with engine.connect() as conn:
        branches_in_staging = pd.read_sql(
            text("SELECT DISTINCT branch FROM stg_sales_reports ORDER BY branch"),
            conn,
        )["branch"].tolist()

    if not branches_in_staging:
        print(" No data found in stg_sales_reports. Run the downloader first.")
        return

    print(f"\n Branches found in staging: {branches_in_staging}\n")
        

    total_lineitems    = 0
    total_transactions = 0
    total_snapshots    = 0
    changes_loaded     = False

    # ── Process each branch — sales + cashier ─────────────────────────────────
    for branch in branches_in_staging:
        branch_label = branch_to_location(branch)
        print(f"\n PROCESSING SALES: {branch} - {branch_label}")

        watermark = get_fact_watermark(branch)
        load_after = watermark_start_date(watermark)
        if watermark:
            print(f"    📅 Watermark: loading staging rows after {watermark}")

        # A. Read from staging
        df_sales   = read_sales_staging(engine, branch, after_date=load_after)
        df_cashier = read_cashier_staging(engine, branch, after_date=load_after)

        if df_sales.empty:
            print(f"     No sales data in staging for {branch}")
            continue

        # B. Prepare cashier (dedup + aggregate)
        df_cashier = prepare_cashier(df_cashier)

        # C. Merge, transform, calculate POS_Txn_Sum
        df_merged = merge_and_transform(
            df_sales     = df_sales,
            df_cashier   = df_cashier,
            branch_label = branch_label,
            output_dir   = OUTPUT_DIR,
        )

        if df_merged.empty:
            print(f"     No data after merge for {branch}")
            continue

        # D. Load to sales fact tables
        with get_conn() as conn:
            with conn.cursor() as cur:
                n_lines = load_fact_lineitems(cur, df_merged)
                n_txns  = load_fact_transactions(cur, df_merged)
            conn.commit()

        total_lineitems    += n_lines
        total_transactions += n_txns
        changes_loaded = changes_loaded or bool(n_lines or n_txns)
        print(f"     Loaded {n_lines:,} line items | {n_txns:,} transactions")

        if "sale_date" in df_merged.columns:
            max_loaded = df_merged["sale_date"].dropna().max()
            update_fact_watermark(branch, max_loaded)

        del df_sales, df_cashier, df_merged
        gc.collect()

    # ── Process each branch — qty lists ──────────────────────────────────────
    # Get branches that have qty data (may differ from sales branches)
    with engine.connect() as conn:
        qty_branches = pd.read_sql(
            text("SELECT DISTINCT branch FROM stg_qty_list ORDER BY branch"),
            conn,
        )["branch"].tolist()

    for branch in qty_branches:
        branch_label = branch_to_location(branch)
        print(f"\n PROCESSING QTY: {branch} - {branch_label}")

        qty_watermark = get_qty_watermark(branch)
        qty_load_after = watermark_start_date(qty_watermark)
        df_qty = read_qty_staging(engine, branch, after_date=qty_load_after)

        if df_qty.empty:
            print(f"     No qty data in staging for {branch}")
            continue

        # Flag any inferred dates so they're visible in the output
        inferred = df_qty[df_qty["snapshot_date_source"] == "filename_day_inferred"]
        if not inferred.empty:
            dates = inferred["snapshot_date"].unique()
            print(
                f"      {len(inferred):,} rows have inferred dates "
                f"(day-only filename): {sorted(dates)}"
            )
            print(f"       Query ingestion_files WHERE notes LIKE 'Date inferred%' to review.")

        with get_conn() as conn:
            with conn.cursor() as cur:
                n_snap = load_fact_inventory_snapshot(cur, df_qty, branch_label)
            conn.commit()

        total_snapshots += n_snap
        changes_loaded = changes_loaded or bool(n_snap)
        print(f"     Loaded {n_snap:,} inventory snapshot rows")

        max_qty_loaded = df_qty["snapshot_date"].dropna().max()
        update_qty_watermark(branch, max_qty_loaded)

        del df_qty
        gc.collect()

    # ── Refresh materialized views ────────────────────────────────────────────
    if changes_loaded:
        print("\n Refreshing materialized views...")
        with get_conn() as conn:
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute("REFRESH MATERIALIZED VIEW mv_transaction_master;")
                print("     mv_transaction_master refreshed")
                cur.execute("REFRESH MATERIALIZED VIEW mv_client_list;")
                print("     mv_client_list refreshed")
    else:
        print("\n No fact changes detected; skipping materialized view refresh.")

    print(f"\n{'=' * 65}")
    print(f" PIPELINE COMPLETE")
    print(f"   Line items loaded:        {total_lineitems:,}")
    print(f"   Transactions loaded:      {total_transactions:,}")
    print(f"   Inventory snapshots:      {total_snapshots:,}")
    print(f"{'=' * 65}")

    if not (EXPORT_POWERBI_CACHE_ENABLED and (changes_loaded or FORCE_POWERBI_CACHE_EXPORT)):
        print("\n Skipping Power BI cache export.")
        engine.dispose()
        return

    # ── Refresh pwer bi cache ────────────────────────────────────────────
    print("\n📤 Exporting Power BI cache files...")
    export_powerbi_cache(engine)

    engine.dispose()


if __name__ == "__main__":
    run_pos_loader()
