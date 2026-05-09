-- ============================================================
-- PORTAL PHARMACY DATA WAREHOUSE SETUP
-- ============================================================
-- Run this script ONCE to set up the warehouse layer.
-- After this, load_to_postgres.py handles all data loading.
--
-- Order of execution:
--   1. Create base tables (fact_sales_lineitems, fact_sales_transactions)
--   2. Drop and recreate mv_transaction_master (now reads from tables)
--   3. Drop and recreate mv_client_list (now includes Products_Bought)
-- ============================================================


-- ============================================================
-- STEP 1: FACT TABLES
-- ============================================================

-- ----------------------------------------------------------
-- 1A. fact_sales_lineitems
--     One row per line item in a transaction.
--     This is the highest granularity table — never aggregate
--     data at this level, only read from it.
--     Source: etl.py → all_locations_sales_Jan25-Jan26.csv
-- ----------------------------------------------------------
CREATE TABLE IF NOT EXISTS fact_sales_lineitems (
    id                  SERIAL PRIMARY KEY,

    -- Identity
    location            TEXT        NOT NULL,
    transaction_id      TEXT        NOT NULL,

    -- Product details
    department          TEXT,
    category            TEXT,
    item                TEXT,
    description         TEXT,           -- product name
    qty_sold            NUMERIC(10, 2),
    total_tax_ex        NUMERIC(12, 2), -- line item value

    -- Date
    date_sold           TEXT,           -- raw string from POS (kept for audit)
    sale_date           DATE,           -- parsed clean date
    sale_date_str       TEXT,

    -- Transaction-level context (from cashier merge)
    client_name         TEXT,
    phone_number        TEXT,
    sales_rep           TEXT,
    txn_type            TEXT,
    ordered_via         TEXT,
    cashier_amount      NUMERIC(12, 2),
    transaction_total   NUMERIC(12, 2), -- sum of all line items in txn
    audit_status        TEXT,

    -- Metadata
    loaded_at           TIMESTAMP DEFAULT NOW()
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_fsl_transaction_id  ON fact_sales_lineitems (transaction_id);
CREATE INDEX IF NOT EXISTS idx_fsl_phone_number    ON fact_sales_lineitems (phone_number);
CREATE INDEX IF NOT EXISTS idx_fsl_location        ON fact_sales_lineitems (location);
CREATE INDEX IF NOT EXISTS idx_fsl_sale_date       ON fact_sales_lineitems (sale_date);
CREATE INDEX IF NOT EXISTS idx_fsl_description     ON fact_sales_lineitems (description);


-- ----------------------------------------------------------
-- 1B. fact_sales_transactions
--     One row per transaction (aggregated from line items).
--     Use this for revenue, visit counts, and client stats.
--     Populated by load_to_postgres.py — never edited manually.
-- ----------------------------------------------------------
CREATE TABLE IF NOT EXISTS fact_sales_transactions (
    id                  SERIAL PRIMARY KEY,

    -- Identity
    location            TEXT        NOT NULL,
    transaction_id      TEXT        NOT NULL,

    -- Dates
    sale_date           DATE,
    sale_date_str       TEXT,

    -- Client (from cashier)
    client_name         TEXT,
    phone_number        TEXT,
    sales_rep           TEXT,
    txn_type            TEXT,
    ordered_via         TEXT,

    -- Financials
    pos_txn_sum         NUMERIC(12, 2), -- sum of line item values
    cashier_amount      NUMERIC(12, 2), -- what cashier recorded
    real_transaction_value NUMERIC(12, 2), -- pos_sum or cashier (priority logic)

    -- Products (rolled up from line items)
    products_in_txn     TEXT,           -- pipe-separated product list
    item_count          INT,            -- number of distinct line items

    -- Audit
    audit_status        TEXT,
    loaded_at           TIMESTAMP DEFAULT NOW(),

    -- Unique constraint prevents duplicate loads
    CONSTRAINT uq_fst_location_txn UNIQUE (location, transaction_id)
);

CREATE INDEX IF NOT EXISTS idx_fst_transaction_id  ON fact_sales_transactions (transaction_id);
CREATE INDEX IF NOT EXISTS idx_fst_phone_number    ON fact_sales_transactions (phone_number);
CREATE INDEX IF NOT EXISTS idx_fst_location        ON fact_sales_transactions (location);
CREATE INDEX IF NOT EXISTS idx_fst_sale_date       ON fact_sales_transactions (sale_date);


-- ============================================================
-- STEP 2: UPDATED mv_transaction_master
--         Now reads from fact_sales_transactions instead of
--         raw_sales + raw_cashier directly. Much simpler.
-- ============================================================

-- Drop dependent view first, then the view itself
DROP MATERIALIZED VIEW IF EXISTS mv_client_list;
DROP MATERIALIZED VIEW IF EXISTS mv_transaction_master;

CREATE MATERIALIZED VIEW mv_transaction_master AS

SELECT
    location,
    transaction_id,
    sale_date           AS date,
    client_name,
    phone_number,
    sales_rep,
    txn_type,
    ordered_via,
    pos_txn_sum,
    cashier_amount,
    real_transaction_value,
    products_in_txn,
    item_count,
    audit_status
FROM fact_sales_transactions
WHERE real_transaction_value > 0;

-- Indexes
CREATE INDEX ON mv_transaction_master (transaction_id);
CREATE INDEX ON mv_transaction_master (phone_number);
CREATE INDEX ON mv_transaction_master (location);


-- ============================================================
-- STEP 3: UPDATED mv_client_list
--         Now includes branches_visited and products_bought,
--         and uses the new 4-tier system.
-- ============================================================

CREATE MATERIALIZED VIEW mv_client_list AS

WITH cleaned_txns AS (
    SELECT DISTINCT ON (transaction_id, location)
        transaction_id,
        location,
        real_transaction_value,
        -- Clean junk bank payment names at the DB level
        CASE
            WHEN client_name ~* '\m(loop|coop|co-op|ncba)\M' THEN NULL
            WHEN client_name ~  '^[^a-zA-Z0-9]+$'           THEN NULL
            WHEN TRIM(client_name) = ''                       THEN NULL
            ELSE initcap(TRIM(client_name))
        END AS client_name,
        phone_number,
        date AS txn_date
    FROM mv_transaction_master
    WHERE
        real_transaction_value > 0
        AND phone_number IS NOT NULL
        AND TRIM(phone_number) <> ''
    ORDER BY
        transaction_id,
        location,
        initcap(TRIM(client_name))
),

deduped_txns AS (
    SELECT
        transaction_id,
        location,
        real_transaction_value,
        client_name,
        phone_number,
        txn_date,
        lag(txn_date) OVER (
            PARTITION BY transaction_id ORDER BY txn_date
        ) AS prev_txn_date
    FROM cleaned_txns
),

unique_txns AS (
    SELECT
        transaction_id,
        location,
        real_transaction_value,
        client_name,
        phone_number,
        txn_date,
        CASE
            WHEN prev_txn_date IS NULL          THEN 1
            WHEN (txn_date - prev_txn_date) > 2 THEN 1
            ELSE 0
        END AS is_unique
    FROM deduped_txns
),

client_stats AS (
    SELECT
        phone_number,

        -- Best non-junk name across all transactions
        mode() WITHIN GROUP (ORDER BY client_name)      AS client_name,

        -- Most visited branch
        mode() WITHIN GROUP (ORDER BY location)         AS preferred_location,

        -- All branches ever visited
        string_agg(DISTINCT location, ' | '
            ORDER BY location)                          AS branches_visited,

        -- Latest transaction date
        max(txn_date)                                   AS last_interaction_date,

        -- First transaction date
        min(txn_date)                                   AS first_interaction_date,

        -- Total spend (unique transactions only)
        sum(CASE WHEN is_unique = 1
            THEN real_transaction_value ELSE 0 END)     AS total_lifetime_spend,

        -- Total purchases (unique transactions only)
        count(CASE WHEN is_unique = 1
            THEN 1 ELSE NULL END)                       AS total_purchases,

        -- Name audit helpers
        count(DISTINCT client_name)                     AS distinct_name_count,
        string_agg(DISTINCT client_name, ' | ')         AS all_names

    FROM unique_txns
    GROUP BY phone_number
),

-- Products bought: join to line items for full product history per client
client_products AS (
    SELECT
        fsl.phone_number,
        string_agg(
            DISTINCT fsl.description,
            ' | '
            ORDER BY fsl.description
        ) AS products_bought

    FROM fact_sales_lineitems fsl
    WHERE
        fsl.phone_number IS NOT NULL
        AND TRIM(fsl.phone_number) <> ''
        AND fsl.description IS NOT NULL
    GROUP BY fsl.phone_number
)

SELECT
    cs.client_name,
    cs.phone_number,
    cs.preferred_location,
    cs.branches_visited,
    cp.products_bought,
    cs.first_interaction_date,
    cs.last_interaction_date,
    cs.total_lifetime_spend,
    cs.total_purchases,

    -- 4-tier loyalty system
    CASE
        WHEN cs.total_lifetime_spend > 20000 THEN 'Platinum'
        WHEN cs.total_lifetime_spend > 13000 THEN 'Gold'
        WHEN cs.total_lifetime_spend > 7000  THEN 'Silver'
        WHEN cs.total_lifetime_spend > 0     THEN 'Bronze'
        ELSE 'No Spend'
    END AS lifetime_tier,

    -- Name audit flag
    CASE
        WHEN cs.distinct_name_count > 1
            THEN 'Review: ' || cs.all_names
        ELSE 'Clean'
    END AS name_audit_flag

FROM client_stats cs
LEFT JOIN client_products cp
    ON cs.phone_number = cp.phone_number

ORDER BY cs.total_lifetime_spend DESC;

-- Indexes for fast lookups
CREATE UNIQUE INDEX ON mv_client_list (phone_number);
CREATE        INDEX ON mv_client_list (lifetime_tier);


-- ============================================================
-- STEP 4: REFRESH HELPER
--         Run this after every pipeline execution to update
--         the materialized views with fresh data.
-- ============================================================

-- You can call this manually or from load_to_postgres.py:
--
--   REFRESH MATERIALIZED VIEW CONCURRENTLY mv_transaction_master;
--   REFRESH MATERIALIZED VIEW CONCURRENTLY mv_client_list;
--
-- CONCURRENTLY means the views stay readable during refresh
-- (requires the UNIQUE INDEX we created above).