-- ══════════════════════════════════════════════════════════════════════════════
-- PORTAL ML V3 — DATA REMEDIATION  (run: 2026-05-02)
-- ══════════════════════════════════════════════════════════════════════════════
-- Scope   : Remove all data dated 2026-04-15 onwards.
--           This catches BOTH legitimate May data (needs re-loading with
--           the fixed parser) AND the fake December 2026 rows that the old
--           parser created from DD/MM/YYYY dates (e.g. 4/12/2026 → Dec 4).
-- Boundary: Everything through 2026-04-14 inclusive is untouched.
-- Watermark reset: all branches → 2026-04-14, datetime → NULL.
--           NULL datetime forces the pipeline back to date-only filtering,
--           so the next run loads date_sold > 2026-04-14 (April 15 onwards).
--
-- INSTRUCTIONS:
--   Run STEP 0 first — inspect the row counts.
--   Only run STEP 1 onwards once you are satisfied.
-- ══════════════════════════════════════════════════════════════════════════════


-- ── STEP 0: PRE-FLIGHT AUDIT ──────────────────────────────────────────────────
-- Run this block alone first. Check the numbers look right.
-- (No data is changed here.)

SELECT
    table_name,
    COUNT(*)           AS rows_to_delete,
    MIN(date_col)      AS earliest_affected,
    MAX(date_col)      AS latest_affected
FROM (
    SELECT 'stg_sales_reports'    AS table_name, date_sold    AS date_col FROM stg_sales_reports    WHERE date_sold    >= '2026-04-15' OR date_sold    > CURRENT_DATE
    UNION ALL
    SELECT 'fact_sales_lineitems'    ,            sale_date               FROM fact_sales_lineitems    WHERE sale_date    >= '2026-04-15' OR sale_date    > CURRENT_DATE
    UNION ALL
    SELECT 'fact_sales_transactions' ,            sale_date               FROM fact_sales_transactions WHERE sale_date    >= '2026-04-15' OR sale_date    > CURRENT_DATE
) t
GROUP BY table_name
ORDER BY table_name;


-- ── STEP 1: DELETE CORRUPTED DATA ─────────────────────────────────────────────

BEGIN;

-- Staging
DELETE FROM stg_sales_reports
WHERE date_sold >= '2026-04-15'
   OR date_sold  > CURRENT_DATE;      -- belt-and-braces: catches any future-dated corruption

-- Fact line items
DELETE FROM fact_sales_lineitems
WHERE sale_date >= '2026-04-15'
   OR sale_date  > CURRENT_DATE;

-- Fact transactions
DELETE FROM fact_sales_transactions
WHERE sale_date >= '2026-04-15'
   OR sale_date  > CURRENT_DATE;


-- ── STEP 2: RESET BRANCH WATERMARKS ──────────────────────────────────────────
-- max_datetime_loaded → NULL so next run uses safe date-only filtering.

UPDATE branch_watermarks
SET
    max_date_loaded     = '2026-04-14',
    max_datetime_loaded = NULL,
    last_updated_at     = NOW()
WHERE branch IN (
    'GALLERIA',
    'PHARMART_ABC',
    'NGONG_MILELE',
    'PORTAL_2R',
    'PORTAL_CBD',
    'CENTURION_2R'
);


-- ── STEP 3: RESET FACT LOAD WATERMARKS ───────────────────────────────────────

UPDATE fact_load_watermarks
SET
    max_date_loaded = '2026-04-14',
    last_updated_at = NOW()
WHERE branch IN (
    'GALLERIA',
    'PHARMART_ABC',
    'NGONG_MILELE',
    'PORTAL_2R',
    'PORTAL_CBD',
    'CENTURION_2R'
);


-- ── STEP 4: UNBLOCK INGESTION FILES ──────────────────────────────────────────
-- Files processed in today's corrupted run are currently status='loaded'.
-- The hash-check in sharepoint_parser.py skips any file marked 'loaded',
-- so without this reset the pipeline would never re-process them.
-- Only today's files are touched — older 'loaded' rows are left alone.

UPDATE ingestion_files
SET status = 'pending'
WHERE status      = 'loaded'
  AND processed_at >= CURRENT_DATE;   -- CURRENT_DATE = 2026-05-02

COMMIT;


-- ── STEP 5: POST-FLIGHT VERIFICATION ──────────────────────────────────────────
-- Run after commit. All six branches should show 2026-04-14 / NULL.

SELECT
    branch,
    max_date_loaded,
    max_datetime_loaded
FROM branch_watermarks
WHERE branch IN (
    'GALLERIA', 'PHARMART_ABC', 'NGONG_MILELE',
    'PORTAL_2R', 'PORTAL_CBD', 'CENTURION_2R'
)
ORDER BY branch;

-- Quick sanity check — no rows should come back.
SELECT 'stg_remaining_post_cutoff' AS check_name, COUNT(*) AS row_count
FROM stg_sales_reports
WHERE date_sold >= '2026-04-15' OR date_sold > CURRENT_DATE;

SELECT 'fact_lineitems_remaining_post_cutoff' AS check_name, COUNT(*) AS row_count
FROM fact_sales_lineitems
WHERE sale_date >= '2026-04-15' OR sale_date > CURRENT_DATE;
