-- ============================================================
-- branch_watermarks
-- Stores the latest Date Sold successfully loaded per branch.
-- Used by the sales pipeline to skip files already covered.
-- Run once against portal_pharmacy before next nightly run.
-- ============================================================

CREATE TABLE IF NOT EXISTS branch_watermarks (
    branch              TEXT        PRIMARY KEY,
    max_date_loaded     DATE        NOT NULL,
    last_updated_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_run_id         INTEGER,
    notes               TEXT
);

-- Seed with current max dates from fact so first run is correct
INSERT INTO branch_watermarks (branch, max_date_loaded, notes)
SELECT
    location                        AS branch,
    MAX(sale_date)                  AS max_date_loaded,
    'seeded from fact_sales_transactions on initial setup' AS notes
FROM fact_sales_transactions
GROUP BY location
ON CONFLICT (branch) DO UPDATE SET
    max_date_loaded = EXCLUDED.max_date_loaded,
    last_updated_at = NOW(),
    notes           = EXCLUDED.notes;

-- Verify
SELECT branch, max_date_loaded, last_updated_at
FROM branch_watermarks
ORDER BY branch;