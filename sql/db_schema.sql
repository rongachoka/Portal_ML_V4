-- ============================================================
-- PORTAL ML — INGESTION SCHEMA
-- Run once to create the new ingestion and staging tables.
-- Existing tables (fact_sales_transactions, fact_sales_lineitems,
-- mv_transaction_master, mv_client_list, dim_products etc.)
-- are NOT touched by this script.
-- ============================================================


-- ============================================================
-- 1. ingestion_runs
--    One row per pipeline execution.
-- ============================================================
CREATE TABLE IF NOT EXISTS ingestion_runs (
    id                  SERIAL          PRIMARY KEY,
    pipeline_name       TEXT            NOT NULL DEFAULT 'sharepoint_ingestion',
    started_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    finished_at         TIMESTAMPTZ,
    status              TEXT            NOT NULL DEFAULT 'running',
    -- running | success | partial | failed
    files_seen          INTEGER         DEFAULT 0,
    files_downloaded    INTEGER         DEFAULT 0,
    files_processed     INTEGER         DEFAULT 0,
    files_failed        INTEGER         DEFAULT 0,
    notes               TEXT
);


-- ============================================================
-- 2. ingestion_files
--    One row per file encountered per run.
--    All files are registered here — canonical or not.
-- ============================================================
CREATE TABLE IF NOT EXISTS ingestion_files (
    id                          SERIAL          PRIMARY KEY,
    run_id                      INTEGER         REFERENCES ingestion_runs(id),

    -- Source identity
    branch                      TEXT            NOT NULL,
    report_type                 TEXT            NOT NULL,
    -- sales | cashier
    file_type                   TEXT,
    -- historical | incremental (sales only)

    -- File details
    filename                    TEXT            NOT NULL,
    file_extension              TEXT,

    -- SharePoint metadata
    sharepoint_item_id          TEXT,
    sharepoint_path             TEXT,
    sharepoint_last_modified    TIMESTAMPTZ,
    sharepoint_size_bytes       BIGINT,

    -- Local copy
    local_path                  TEXT,

    -- Content fingerprint
    file_hash                   TEXT,
    row_count                   INTEGER,

    -- Canonical selection
    is_canonical                BOOLEAN         DEFAULT FALSE,
    canonical_reason            TEXT,
    -- e.g. 'historical', 'incremental', 'max_row_count', 'only_file'

    -- Tracking
    downloaded_at               TIMESTAMPTZ     DEFAULT NOW(),
    processed_at                TIMESTAMPTZ,
    status                      TEXT            DEFAULT 'pending',
    -- pending | loaded | skipped | failed
    error_message               TEXT,
    notes                       TEXT
    -- Used to flag non-error conditions e.g. ambiguous date inference
);

-- Prevent loading the same file content twice in the same run
CREATE UNIQUE INDEX IF NOT EXISTS uq_ingestion_files_run_hash
    ON ingestion_files (run_id, branch, report_type, file_hash)
    WHERE file_hash IS NOT NULL;


-- ============================================================
-- 3. stg_sales_reports
--    One row per line item from a sales CSV or XLSX.
--    Deduplication applied upstream (canonical file selection
--    + composite dedup on Transaction ID, Item, On Hand,
--    Qty Sold, Date Sold).
--    Date Sold is stored clean (# characters stripped).
-- ============================================================
CREATE TABLE IF NOT EXISTS stg_sales_reports (
    id                  SERIAL          PRIMARY KEY,

    -- Audit (inline — no join needed for basic tracing)
    source_file_id      INTEGER         REFERENCES ingestion_files(id),
    source_filename     TEXT,

    -- Location
    branch              TEXT            NOT NULL,

    -- Sales columns (mapped from source)
    department          TEXT,
    category            TEXT,
    item                TEXT,
    -- Item barcode from source 'Item' column
    description         TEXT,
    on_hand             NUMERIC,
    last_sold           DATE,
    qty_sold            NUMERIC,
    total_tax_ex        NUMERIC,
    transaction_id      TEXT,
    date_sold           DATE,
    -- Cleaned: # characters stripped, parsed to DATE

    -- Load timestamp
    loaded_at           TIMESTAMPTZ     DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_stg_sales_branch
    ON stg_sales_reports (branch);
CREATE INDEX IF NOT EXISTS idx_stg_sales_transaction_id
    ON stg_sales_reports (transaction_id);
CREATE INDEX IF NOT EXISTS idx_stg_sales_date_sold
    ON stg_sales_reports (date_sold);
CREATE INDEX IF NOT EXISTS idx_stg_sales_source_file
    ON stg_sales_reports (source_file_id);


-- ============================================================
-- 4. stg_cashier_reports
--    One row per cashier transaction line.
--    Sheets 01-31 only. Empty sheets skipped.
--    transaction_date derived from sheet number + month/year
--    extracted from filename (fallback: sharepoint_last_modified).
--    Time kept as TEXT — not parsed to avoid ambiguous formats
--    e.g. '12.5' could be 12:05 or 12:50.
-- ============================================================
CREATE TABLE IF NOT EXISTS stg_cashier_reports (
    id                  SERIAL          PRIMARY KEY,

    -- Audit (inline)
    source_file_id      INTEGER         REFERENCES ingestion_files(id),
    source_filename     TEXT,

    -- Location
    branch              TEXT            NOT NULL,

    -- Derived date (sheet number + filename month/year)
    transaction_date    DATE,

    -- Cashier columns (mapped from source)
    receipt_txn_no      TEXT,
    amount              NUMERIC,
    txn_costs           NUMERIC,
    txn_time            TEXT,
    -- Kept as string — ambiguous formats e.g. '12.5'
    txn_type            TEXT,
    ordered_via         TEXT,
    client_name         TEXT,
    phone_number        TEXT,
    sales_rep           TEXT,

    -- Load timestamp
    loaded_at           TIMESTAMPTZ     DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_stg_cashier_branch
    ON stg_cashier_reports (branch);
CREATE INDEX IF NOT EXISTS idx_stg_cashier_transaction_date
    ON stg_cashier_reports (transaction_date);
CREATE INDEX IF NOT EXISTS idx_stg_cashier_receipt_txn_no
    ON stg_cashier_reports (receipt_txn_no);
CREATE INDEX IF NOT EXISTS idx_stg_cashier_source_file
    ON stg_cashier_reports (source_file_id);


-- ============================================================
-- 5. stg_qty_list
--    One row per product per branch per snapshot date.
--    Source: Item Quantity List / QTY LIST files from SharePoint.
--    snapshot_date_source flags how the date was derived —
--    'filename_day_inferred' rows should be reviewed manually.
--
--    Header cleaning: some files have leading semicolons and
--    two junk rows above the real header — stripped at parse time.
-- ============================================================
CREATE TABLE IF NOT EXISTS stg_qty_list (
    id                      SERIAL          PRIMARY KEY,

    -- Audit (inline)
    source_file_id          INTEGER         REFERENCES ingestion_files(id),
    source_filename         TEXT,

    -- Location
    branch                  TEXT            NOT NULL,

    -- Snapshot date + how it was derived
    snapshot_date           DATE,
    snapshot_date_source    TEXT,
    -- 'filename_full'         → full date parsed from filename e.g. 04.03.26
    -- 'filename_day_inferred' → day from filename, month/year from lastModifiedDateTime ⚠ REVIEW
    -- 'lastmodified'          → full fallback to lastModifiedDateTime

    -- Qty list columns (mapped from source)
    department              TEXT,
    category                TEXT,
    item_lookup_code        TEXT,
    -- Same as 'item' in stg_sales_reports — enables join
    description             TEXT,
    on_hand                 NUMERIC,
    committed               NUMERIC,
    -- Meaning TBC — kept as-is
    reorder_pt              NUMERIC,
    restock_lvl             NUMERIC,
    qty_to_order            NUMERIC,
    supplier                TEXT,
    reorder_no              TEXT,

    -- Load timestamp
    loaded_at               TIMESTAMPTZ     DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_stg_qty_branch
    ON stg_qty_list (branch);
CREATE INDEX IF NOT EXISTS idx_stg_qty_snapshot_date
    ON stg_qty_list (snapshot_date);
CREATE INDEX IF NOT EXISTS idx_stg_qty_item_lookup_code
    ON stg_qty_list (item_lookup_code);
CREATE INDEX IF NOT EXISTS idx_stg_qty_date_source
    ON stg_qty_list (snapshot_date_source);
CREATE INDEX IF NOT EXISTS idx_stg_qty_source_file
    ON stg_qty_list (source_file_id);


-- ============================================================
-- 6. fact_inventory_snapshot
--    One row per product per branch per date.
--    Populated from stg_qty_list by load_to_postgres.py.
--    UNIQUE constraint on (branch, item_lookup_code, snapshot_date)
--    means rerunning the pipeline on a corrected file updates
--    the row rather than duplicating it.
--    Enables audit: compare On-Hand change vs Qty Sold per day.
-- ============================================================
CREATE TABLE IF NOT EXISTS fact_inventory_snapshot (
    id                      SERIAL          PRIMARY KEY,

    -- Identity
    branch                  TEXT            NOT NULL,
    snapshot_date           DATE            NOT NULL,
    snapshot_date_source    TEXT,
    -- Carried from staging for traceability

    -- Product
    department              TEXT,
    category                TEXT,
    item_lookup_code        TEXT,
    description             TEXT,

    -- Stock levels
    on_hand                 NUMERIC,
    committed               NUMERIC,
    reorder_pt              NUMERIC,
    restock_lvl             NUMERIC,
    qty_to_order            NUMERIC,
    supplier                TEXT,
    reorder_no              TEXT,

    -- Audit
    source_file_id          INTEGER         REFERENCES ingestion_files(id),
    loaded_at               TIMESTAMPTZ     DEFAULT NOW(),

    -- Prevent duplicates — rerun = update, not insert
    CONSTRAINT uq_inventory_snapshot
        UNIQUE (branch, item_lookup_code, snapshot_date)
);

CREATE INDEX IF NOT EXISTS idx_fact_inv_branch
    ON fact_inventory_snapshot (branch);
CREATE INDEX IF NOT EXISTS idx_fact_inv_snapshot_date
    ON fact_inventory_snapshot (snapshot_date);
CREATE INDEX IF NOT EXISTS idx_fact_inv_item
    ON fact_inventory_snapshot (item_lookup_code);
CREATE INDEX IF NOT EXISTS idx_fact_inv_date_source
    ON fact_inventory_snapshot (snapshot_date_source);


-- ============================================================
-- PIPELINE ORDER (for reference)
-- ============================================================
--
--   1. sharepoint_downloader.py  → downloads files, populates
--                                   ingestion_runs + ingestion_files
--
--   2. sharepoint_parser.py      → reads downloaded files, applies
--                                   dedup logic, loads into
--                                   stg_sales_reports + stg_cashier_reports
--                                   + stg_qty_list
--
--   3. load_to_postgres.py       → reads from stg_sales_reports +
--                                   stg_cashier_reports, merges,
--                                   loads fact_sales_lineitems +
--                                   fact_sales_transactions +
--                                   fact_inventory_snapshot,
--                                   refreshes mv_transaction_master +
--                                   mv_client_list
--
-- Existing tables downstream of fact_sales_transactions are
-- untouched by this script and require no changes.
-- ============================================================