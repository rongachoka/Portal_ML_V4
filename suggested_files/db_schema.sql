-- ============================================================
-- Portal ML Pipeline — Ingestion Schema
-- ============================================================

-- 1. ingestion_runs
--    One row per pipeline execution
-- ============================================================
CREATE TABLE IF NOT EXISTS ingestion_runs (
    id                  SERIAL PRIMARY KEY,
    pipeline_name       TEXT        NOT NULL DEFAULT 'sharepoint_ingestion',
    started_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at         TIMESTAMPTZ,
    status              TEXT        NOT NULL DEFAULT 'running',   -- running | success | partial | failed
    files_seen          INTEGER     DEFAULT 0,
    files_downloaded    INTEGER     DEFAULT 0,
    files_processed     INTEGER     DEFAULT 0,
    files_failed        INTEGER     DEFAULT 0,
    notes               TEXT
);


-- ============================================================
-- 2. ingestion_files
--    One row per downloaded file per run
-- ============================================================
CREATE TABLE IF NOT EXISTS ingestion_files (
    id                          SERIAL PRIMARY KEY,
    run_id                      INTEGER     REFERENCES ingestion_runs(id),
    branch                      TEXT        NOT NULL,
    report_type                 TEXT        NOT NULL,             -- sales | cashier
    filename                    TEXT        NOT NULL,
    file_extension              TEXT,
    sharepoint_item_id          TEXT,
    sharepoint_path             TEXT,
    sharepoint_last_modified    TIMESTAMPTZ,
    sharepoint_size_bytes       BIGINT,
    local_path                  TEXT,
    file_hash                   TEXT,
    row_count                   INTEGER,
    is_canonical                BOOLEAN     DEFAULT FALSE,        -- TRUE = this file fed the DB
    canonical_reason            TEXT,                            -- e.g. 'max_row_count'
    downloaded_at               TIMESTAMPTZ DEFAULT NOW(),
    processed_at                TIMESTAMPTZ,
    status                      TEXT        DEFAULT 'pending',   -- pending | loaded | skipped | failed
    error_message               TEXT
);

-- Prevent same file content being loaded twice in the same run
CREATE UNIQUE INDEX IF NOT EXISTS uq_ingestion_files_run_hash
    ON ingestion_files (run_id, branch, report_type, file_hash)
    WHERE file_hash IS NOT NULL;


-- ============================================================
-- 3. stg_sales_reports
--    One row per sales transaction line
-- ============================================================
CREATE TABLE IF NOT EXISTS stg_sales_reports (
    id                  SERIAL PRIMARY KEY,
    source_file_id      INTEGER     REFERENCES ingestion_files(id),
    branch              TEXT        NOT NULL,
    department          TEXT,
    category            TEXT,
    item                TEXT,
    description         TEXT,
    on_hand             NUMERIC,
    last_sold           DATE,
    qty_sold            NUMERIC,
    total_tax_ex        NUMERIC,
    transaction_id      TEXT,
    date_sold           DATE,
    loaded_at           TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_stg_sales_branch       ON stg_sales_reports (branch);
CREATE INDEX IF NOT EXISTS idx_stg_sales_date_sold    ON stg_sales_reports (date_sold);
CREATE INDEX IF NOT EXISTS idx_stg_sales_source_file  ON stg_sales_reports (source_file_id);


-- ============================================================
-- 4. stg_cashier_reports
--    One row per cashier transaction line
-- ============================================================
CREATE TABLE IF NOT EXISTS stg_cashier_reports (
    id                  SERIAL PRIMARY KEY,
    source_file_id      INTEGER     REFERENCES ingestion_files(id),
    branch              TEXT        NOT NULL,
    transaction_date    DATE,                                    -- derived from sheet number + filename month/year
    receipt_txn_no      TEXT,
    amount              NUMERIC,
    txn_costs           NUMERIC,
    txn_time            TIME,
    txn_type            TEXT,
    ordered_via         TEXT,
    client_name         TEXT,
    phone_number        TEXT,
    sales_rep           TEXT,
    loaded_at           TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_stg_cashier_branch       ON stg_cashier_reports (branch);
CREATE INDEX IF NOT EXISTS idx_stg_cashier_date         ON stg_cashier_reports (transaction_date);
CREATE INDEX IF NOT EXISTS idx_stg_cashier_source_file  ON stg_cashier_reports (source_file_id);
