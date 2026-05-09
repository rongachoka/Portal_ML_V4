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
    - fact_inventory_snapshot uses ON CONFLICT DO UPDATE — safe to rerun
      (fact_sales_lineitems and fact_sales_transactions are truncated + reloaded)

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
from datetime import datetime, date
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

# ── Read from watermark ───────────────────────────────────────────────────────
def get_fact_watermark(branch: str) -> date | None:
    """Read the latest date already loaded into fact_sales_transactions."""
    location = BRANCH_LABEL_MAP.get(branch, branch.upper().replace(" ", "_"))
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT max_date_loaded FROM fact_load_watermarks WHERE branch = %s",
                    (location,)
                )
                row = cur.fetchone()
                return row[0] if row else None
    except Exception as e:
        print(f"    ⚠️ Could not read fact watermark for {branch}: {e}")
        return None


def update_fact_watermark(branch: str, max_date) -> None:
    """Update fact_load_watermarks after successful load."""
    if max_date is None:
        return
    location = BRANCH_LABEL_MAP.get(branch, branch.upper().replace(" ", "_"))
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
                """, (location, max_date))
            conn.commit()
        print(f"    💾 Fact watermark updated: {location} → {max_date}")
    except Exception as e:
        print(f"    ⚠️ Could not update fact watermark for {branch}: {e}")


# ── Read from staging ─────────────────────────────────────────────────────────

def read_sales_staging(engine, branch: str, after_date=None) -> pd.DataFrame:
    if after_date:
        sql = text("""
            SELECT
                branch, department, category, item, description,
                on_hand, last_sold, qty_sold, total_tax_ex,
                transaction_id, date_sold, sale_time, sale_datetime
            FROM stg_sales_reports
            WHERE branch = :branch
              AND date_sold > :after_date
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
              AND transaction_date > :after_date
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


def read_qty_staging(engine, branch: str) -> pd.DataFrame:
    """
    Read all qty list rows for a branch from stg_qty_list.
    Includes snapshot_date and snapshot_date_source for traceability.
    """
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
        chunk_size = 50_000

        for i in range(0, len(df_sales), chunk_size):
            chunk = df_sales.iloc[i : i + chunk_size].copy()
            merged_chunk = pd.merge(
                chunk,
                df_cashier,
                left_on  = "transaction_id",
                right_on = "receipt_txn_no",
                how      = "left",
            )
            # Prevent ArrowString/category OOM when stacking chunks
            for col in merged_chunk.columns:
                if (
                    merged_chunk[col].dtype.name in ("category", "string")
                    or hasattr(merged_chunk[col], "cat")
                ):
                    merged_chunk[col] = merged_chunk[col].astype(object)

            chunks.append(merged_chunk)
            del chunk, merged_chunk
            gc.collect()

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
    # Applies after merge so both sales-side and cashier-side phone numbers
    # are covered. normalize_phone handles all formats:
    # +254XXXXXXXXX, 0XXXXXXXXX, 254XXXXXXXXX, with/without apostrophe prefix
    # All normalise to 9-digit Kenyan format e.g. 722000000
    if "phone_number" in df_merged.columns:
        df_merged["phone_number"] = df_merged["phone_number"].apply(normalize_phone)

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
    """

    rows = []
    for _, row in df.iterrows():
        rows.append((
            row.get("location"),
            row.get("transaction_id"),
            row.get("department"),
            row.get("category"),
            row.get("item"),
            row.get("description"),
            _safe_numeric(row.get("qty_sold")),
            _safe_numeric(row.get("total_tax_ex")),
            str(row.get("date_sold") or ""),       # raw date string for audit
            _safe_date(row.get("sale_date")),
            row.get("sale_date_str"),
            row.get("client_name"),
            row.get("phone_number"),
            row.get("sales_rep"),
            row.get("txn_type"),
            row.get("ordered_via"),
            _safe_numeric(row.get("amount")),
            _safe_numeric(row.get("pos_txn_sum")), # transaction_total = POS sum
            row.get("audit_status"),
        ))

    execute_values(cur, INSERT_SQL, rows, page_size=1000)
    return len(rows)


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
                                  x.dropna().astype(str).unique()
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
    """

    rows = []
    for row in agg.itertuples(index=False):
        rows.append((
            row.location,
            row.transaction_id,
            _safe_date(row.sale_date),
            row.sale_date_str,
            _safe_time(row.sale_time),         
            _safe_datetime(row.sale_datetime),
            row.client_name,
            row.phone_number,
            row.sales_rep,
            row.txn_type,
            row.ordered_via,
            _safe_numeric(row.pos_txn_sum),
            _safe_numeric(row.cashier_amount),
            _safe_numeric(row.real_transaction_value),
            row.products_in_txn,
            row.item_count,
            row.audit_status,
        ))


    execute_values(cur, INSERT_SQL, rows, page_size=1000)
    return len(rows)


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
    """

    rows = []
    for _, row in df.iterrows():
        rows.append((
            branch_label,
            _safe_date(row.get("snapshot_date")),
            row.get("snapshot_date_source"),
            row.get("department"),
            row.get("category"),
            row.get("item_lookup_code"),
            row.get("description"),
            _safe_numeric(row.get("on_hand")),
            _safe_numeric(row.get("committed")),
            _safe_numeric(row.get("reorder_pt")),
            _safe_numeric(row.get("restock_lvl")),
            _safe_numeric(row.get("qty_to_order")),
            row.get("supplier"),
            row.get("reorder_no"),
            row.get("source_file_id"),
        ))

    execute_values(cur, INSERT_SQL, rows, page_size=1000)
    return len(rows)


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
        CACHE_DIR = Path(os.getenv("POWERBI_CACHE_DIR", "D:/PowerBI_Cache"))
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        exports = {
            "vw_sales_base.csv":         "SELECT * FROM vw_sales_base",
            "vw_sales_with_margin.csv":  "SELECT * FROM vw_sales_with_margin",
            "mv_transaction_master.csv": "SELECT * FROM mv_transaction_master",
            "mv_client_list.csv":        "SELECT * FROM mv_client_list",
            "vw_dead_stock.csv":         "SELECT * FROM vw_dead_stock",
        }

        for filename, sql in exports.items():
            print(f"    📤 Exporting {filename}...")
            df = pd.read_sql(text(sql), engine)
            df.to_csv(CACHE_DIR / filename, index=False)
            print(f"       ✅ {len(df):,} rows → {filename}")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_pos_loader():
    print("=" * 65)
    print(" POS LOADER — reading from staging tables")
    print("=" * 65)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Clear previous discrepancy log so each run starts fresh
    if DISCREPANCY_LOG.exists():
        DISCREPANCY_LOG.unlink()

    engine = get_engine()

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

    # ── Process each branch — sales + cashier ─────────────────────────────────
    for branch in branches_in_staging:
        branch_label = BRANCH_LABEL_MAP.get(branch, branch.upper().replace(" ", "_"))
        print(f"\n PROCESSING SALES: {branch} - {branch_label}")

        watermark = get_fact_watermark(branch)
        if watermark:
            print(f"    📅 Watermark: loading staging rows after {watermark}")

        # A. Read from staging
        df_sales   = read_sales_staging(engine, branch, after_date=watermark)
        df_cashier = read_cashier_staging(engine, branch)

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
        branch_label = BRANCH_LABEL_MAP.get(branch, branch.upper().replace(" ", "_"))
        print(f"\n PROCESSING QTY: {branch} - {branch_label}")

        df_qty = read_qty_staging(engine, branch)

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
        print(f"     Loaded {n_snap:,} inventory snapshot rows")

        del df_qty
        gc.collect()

    # ── Refresh materialized views ────────────────────────────────────────────
    print("\n Refreshing materialized views...")
    with get_conn() as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("REFRESH MATERIALIZED VIEW mv_transaction_master;")
            print("     mv_transaction_master refreshed")
            cur.execute("REFRESH MATERIALIZED VIEW mv_client_list;")
            print("     mv_client_list refreshed")

    print(f"\n{'=' * 65}")
    print(f" PIPELINE COMPLETE")
    print(f"   Line items loaded:        {total_lineitems:,}")
    print(f"   Transactions loaded:      {total_transactions:,}")
    print(f"   Inventory snapshots:      {total_snapshots:,}")
    print(f"{'=' * 65}")

    # ── Refresh pwer bi cache ────────────────────────────────────────────
    print("\n📤 Exporting Power BI cache files...")
    export_powerbi_cache(engine)

    engine.dispose()


if __name__ == "__main__":
    

    run_pos_loader()
