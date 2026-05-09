-- ============================================================
-- PAGE 1: EXECUTIVE OVERVIEW — SQL VIEWS
-- Portal Pharmacy Data Warehouse
--
-- KEY DESIGN PRINCIPLE: "aligned_date"
--   All date comparisons use the minimum of all branches' latest
--   available date. If 4 branches have data to Mar 9 and 2 have
--   Mar 10, everything reports to Mar 9. This ensures DoD/MoM/YoY
--   comparisons are always apples-to-apples across branches.
--
-- mv_transaction_master columns:
--   location, transaction_id, "date", phone_number,
--   real_transaction_value, client_name, sales_rep, audit_status
-- ============================================================


-- ============================================================
-- HELPER VIEW: vw_aligned_date
-- Single row. The universal cutoff date used by all other views.
-- = minimum of each branch's most recent date in the data.
-- ============================================================

DROP VIEW IF EXISTS vw_aligned_date CASCADE;

CREATE VIEW vw_aligned_date AS

WITH branch_dates AS (
    SELECT location, MAX("date") AS branch_last_date
    FROM mv_transaction_master
    GROUP BY location
),
totals AS (
    SELECT
        MIN(branch_last_date) AS aligned_date,
        MAX(branch_last_date) AS latest_branch_date,
        COUNT(*)              AS branch_count
    FROM branch_dates
)

SELECT
    t.aligned_date,
    t.latest_branch_date,
    t.branch_count,
    STRING_AGG(
        b.location || ' (' || TO_CHAR(b.branch_last_date, 'DD Mon') || ')',
        ', ' ORDER BY b.location
    ) AS lagging_branches
FROM totals t
LEFT JOIN branch_dates b ON b.branch_last_date < t.latest_branch_date
GROUP BY t.aligned_date, t.latest_branch_date, t.branch_count;


-- ============================================================
-- VIEW 1: vw_exec_kpis  (single-row KPI cards)
-- ============================================================

DROP VIEW IF EXISTS vw_exec_kpis CASCADE;

CREATE VIEW vw_exec_kpis AS

WITH ref AS (
    SELECT
        -- aligned_date = last day all branches have submitted data for
        ad.aligned_date                                                     AS yesterday,
        DATE_TRUNC('month',  ad.aligned_date)::DATE                         AS month_start,
        (DATE_TRUNC('month', ad.aligned_date) - INTERVAL '1 month')::DATE   AS prior_month_start,
        (DATE_TRUNC('month', ad.aligned_date) - INTERVAL '1 year')::DATE    AS prior_yr_month_start,
        DATE_TRUNC('year',   ad.aligned_date)::DATE                         AS year_start,
        (DATE_TRUNC('year',  ad.aligned_date) - INTERVAL '1 year')::DATE    AS prior_year_start,
        EXTRACT(DAY FROM ad.aligned_date)::INT                              AS days_elapsed,
        ad.aligned_date                                                     AS last_full_day,
        ad.lagging_branches
    FROM vw_aligned_date ad
),

base AS (
    SELECT t.transaction_id, t.location, t."date",
           t.real_transaction_value AS txn_value
    FROM mv_transaction_master t
    WHERE t.real_transaction_value > 0
)

SELECT
    CURRENT_DATE                                                            AS report_date,
    (SELECT last_full_day    FROM ref)                                      AS last_full_day,
    (SELECT lagging_branches FROM ref)                                      AS lagging_branches,

    -- YTD
    ROUND(COALESCE(SUM(txn_value) FILTER (
        WHERE "date" >= (SELECT year_start FROM ref)
          AND "date" <= (SELECT yesterday  FROM ref)
    ), 0)::NUMERIC, 0)                                                      AS revenue_ytd,

    COUNT(DISTINCT transaction_id) FILTER (
        WHERE "date" >= (SELECT year_start FROM ref)
          AND "date" <= (SELECT yesterday  FROM ref)
    )                                                                       AS transactions_ytd,

    -- Prior YTD (same calendar days last year)
    ROUND(COALESCE(SUM(txn_value) FILTER (
        WHERE "date" >= (SELECT prior_year_start FROM ref)
          AND "date" <  (SELECT prior_year_start FROM ref)
                        + ((SELECT yesterday FROM ref)
                           - (SELECT year_start FROM ref) + 1) * INTERVAL '1 day'
    ), 0)::NUMERIC, 0)                                                      AS revenue_prior_ytd,

    -- YTD YoY %
    ROUND(
        CASE WHEN COALESCE(SUM(txn_value) FILTER (
                    WHERE "date" >= (SELECT prior_year_start FROM ref)
                      AND "date" <  (SELECT prior_year_start FROM ref)
                                    + ((SELECT yesterday FROM ref)
                                       - (SELECT year_start FROM ref) + 1) * INTERVAL '1 day'
                 ), 0) = 0 THEN NULL
             ELSE (
                 COALESCE(SUM(txn_value) FILTER (
                     WHERE "date" >= (SELECT year_start FROM ref)
                       AND "date" <= (SELECT yesterday  FROM ref)), 0)
                 - COALESCE(SUM(txn_value) FILTER (
                     WHERE "date" >= (SELECT prior_year_start FROM ref)
                       AND "date" <  (SELECT prior_year_start FROM ref)
                                     + ((SELECT yesterday FROM ref)
                                        - (SELECT year_start FROM ref) + 1) * INTERVAL '1 day'), 0)
             )::NUMERIC / NULLIF(SUM(txn_value) FILTER (
                 WHERE "date" >= (SELECT prior_year_start FROM ref)
                   AND "date" <  (SELECT prior_year_start FROM ref)
                                  + ((SELECT yesterday FROM ref)
                                     - (SELECT year_start FROM ref) + 1) * INTERVAL '1 day'
             ), 0) * 100
        END::NUMERIC
    , 1)                                                                    AS ytd_yoy_pct,

    -- Current MTD
    ROUND(COALESCE(SUM(txn_value) FILTER (
        WHERE "date" >= (SELECT month_start FROM ref)
          AND "date" <= (SELECT yesterday   FROM ref)
    ), 0)::NUMERIC, 0)                                                      AS revenue_mtd,

    COUNT(DISTINCT transaction_id) FILTER (
        WHERE "date" >= (SELECT month_start FROM ref)
          AND "date" <= (SELECT yesterday   FROM ref)
    )                                                                       AS transactions_mtd,

    -- Prior MTD (same days last month)
    ROUND(COALESCE(SUM(txn_value) FILTER (
        WHERE "date" >= (SELECT prior_month_start FROM ref)
          AND "date" <  (SELECT prior_month_start FROM ref)
                        + ((SELECT days_elapsed FROM ref) || ' days')::INTERVAL
    ), 0)::NUMERIC, 0)                                                      AS revenue_prior_mtd,

    -- MoM %
    ROUND(
        CASE WHEN COALESCE(SUM(txn_value) FILTER (
                    WHERE "date" >= (SELECT prior_month_start FROM ref)
                      AND "date" <  (SELECT prior_month_start FROM ref)
                                    + ((SELECT days_elapsed FROM ref) || ' days')::INTERVAL
                 ), 0) = 0 THEN NULL
             ELSE (
                 COALESCE(SUM(txn_value) FILTER (
                     WHERE "date" >= (SELECT month_start FROM ref)
                       AND "date" <= (SELECT yesterday   FROM ref)), 0)
                 - COALESCE(SUM(txn_value) FILTER (
                     WHERE "date" >= (SELECT prior_month_start FROM ref)
                       AND "date" <  (SELECT prior_month_start FROM ref)
                                     + ((SELECT days_elapsed FROM ref) || ' days')::INTERVAL), 0)
             )::NUMERIC / NULLIF(SUM(txn_value) FILTER (
                 WHERE "date" >= (SELECT prior_month_start FROM ref)
                   AND "date" <  (SELECT prior_month_start FROM ref)
                                  + ((SELECT days_elapsed FROM ref) || ' days')::INTERVAL
             ), 0) * 100
        END::NUMERIC
    , 1)                                                                    AS mom_growth_pct,

    -- Same MTD last year
    ROUND(COALESCE(SUM(txn_value) FILTER (
        WHERE "date" >= (SELECT prior_yr_month_start FROM ref)
          AND "date" <  (SELECT prior_yr_month_start FROM ref)
                        + ((SELECT days_elapsed FROM ref) || ' days')::INTERVAL
    ), 0)::NUMERIC, 0)                                                      AS revenue_prior_year_mtd,

    -- YoY MTD %
    ROUND(
        CASE WHEN COALESCE(SUM(txn_value) FILTER (
                    WHERE "date" >= (SELECT prior_yr_month_start FROM ref)
                      AND "date" <  (SELECT prior_yr_month_start FROM ref)
                                    + ((SELECT days_elapsed FROM ref) || ' days')::INTERVAL
                 ), 0) = 0 THEN NULL
             ELSE (
                 COALESCE(SUM(txn_value) FILTER (
                     WHERE "date" >= (SELECT month_start FROM ref)
                       AND "date" <= (SELECT yesterday   FROM ref)), 0)
                 - COALESCE(SUM(txn_value) FILTER (
                     WHERE "date" >= (SELECT prior_yr_month_start FROM ref)
                       AND "date" <  (SELECT prior_yr_month_start FROM ref)
                                     + ((SELECT days_elapsed FROM ref) || ' days')::INTERVAL), 0)
             )::NUMERIC / NULLIF(SUM(txn_value) FILTER (
                 WHERE "date" >= (SELECT prior_yr_month_start FROM ref)
                   AND "date" <  (SELECT prior_yr_month_start FROM ref)
                                  + ((SELECT days_elapsed FROM ref) || ' days')::INTERVAL
             ), 0) * 100
        END::NUMERIC
    , 1)                                                                    AS yoy_mtd_pct,

    -- Yesterday (aligned)
    ROUND(COALESCE(SUM(txn_value) FILTER (
        WHERE "date" = (SELECT yesterday FROM ref)
    ), 0)::NUMERIC, 0)                                                      AS revenue_yesterday,

    COUNT(DISTINCT transaction_id) FILTER (
        WHERE "date" = (SELECT yesterday FROM ref)
    )                                                                       AS transactions_yesterday,

    -- Day before
    ROUND(COALESCE(SUM(txn_value) FILTER (
        WHERE "date" = (SELECT yesterday FROM ref) - 1
    ), 0)::NUMERIC, 0)                                                      AS revenue_day_before,

    -- DoD %
    ROUND(
        CASE WHEN COALESCE(SUM(txn_value) FILTER (
                    WHERE "date" = (SELECT yesterday FROM ref) - 1), 0) = 0 THEN NULL
             ELSE (
                 COALESCE(SUM(txn_value) FILTER (WHERE "date" = (SELECT yesterday FROM ref)), 0)
                 - COALESCE(SUM(txn_value) FILTER (WHERE "date" = (SELECT yesterday FROM ref) - 1), 0)
             )::NUMERIC / NULLIF(SUM(txn_value) FILTER (
                 WHERE "date" = (SELECT yesterday FROM ref) - 1), 0) * 100
        END::NUMERIC
    , 1)                                                                    AS dod_growth_pct,

    -- Today partial (branches that submitted today)
    ROUND(COALESCE(SUM(txn_value) FILTER (
        WHERE "date" > (SELECT yesterday FROM ref)
    ), 0)::NUMERIC, 0)                                                      AS revenue_today_partial,

    -- Avg basket MTD
    ROUND(
        COALESCE(SUM(txn_value) FILTER (
            WHERE "date" >= (SELECT month_start FROM ref)
              AND "date" <= (SELECT yesterday   FROM ref)), 0)::NUMERIC /
        NULLIF(COUNT(DISTINCT transaction_id) FILTER (
            WHERE "date" >= (SELECT month_start FROM ref)
              AND "date" <= (SELECT yesterday   FROM ref)), 0)
    , 0)                                                                    AS avg_basket_mtd,

    COUNT(DISTINCT location) FILTER (
        WHERE "date" >= (SELECT month_start FROM ref)
    )                                                                       AS active_branches

FROM base;


-- ============================================================
-- VIEW 2: vw_revenue_by_branch_month
-- ============================================================

DROP VIEW IF EXISTS vw_revenue_by_branch_month CASCADE;

CREATE VIEW vw_revenue_by_branch_month AS

WITH monthly_branch AS (
    SELECT
        location,
        DATE_TRUNC('month', "date")::DATE                   AS month_start,
        TO_CHAR(DATE_TRUNC('month', "date"), 'Mon YY')      AS year_month,
        ROUND(SUM(real_transaction_value)::NUMERIC, 0)      AS revenue,
        COUNT(DISTINCT transaction_id)                       AS txn_count,
        COUNT(DISTINCT phone_number)
            FILTER (WHERE phone_number IS NOT NULL
                      AND TRIM(phone_number::TEXT) <> '')   AS unique_clients
    FROM mv_transaction_master
    WHERE "date" IS NOT NULL AND real_transaction_value > 0
    GROUP BY location, DATE_TRUNC('month', "date")
),
monthly_totals AS (
    SELECT month_start, SUM(revenue) AS total_all_branches
    FROM monthly_branch GROUP BY month_start
)

SELECT
    mb.location,
    mb.month_start,
    mb.year_month,
    mb.revenue,
    mb.txn_count,
    mb.unique_clients,
    ROUND(mb.revenue::NUMERIC / NULLIF(mt.total_all_branches, 0) * 100, 1) AS contribution_pct,
    ROUND(mb.revenue::NUMERIC / NULLIF(mb.txn_count, 0), 0)                AS avg_basket
FROM monthly_branch mb
JOIN monthly_totals mt USING (month_start)
ORDER BY mb.month_start, mb.location;


-- ============================================================
-- VIEW 3: vw_branch_performance  (DoD / MoM / YoY per branch)
-- ============================================================

DROP VIEW IF EXISTS vw_branch_mom_growth CASCADE;
DROP VIEW IF EXISTS vw_branch_performance CASCADE;

CREATE VIEW vw_branch_performance AS

WITH ref AS (
    SELECT
        ad.aligned_date                                                     AS yesterday,
        DATE_TRUNC('month',  ad.aligned_date)::DATE                         AS month_start,
        (DATE_TRUNC('month', ad.aligned_date) - INTERVAL '1 month')::DATE   AS prior_month_start,
        (DATE_TRUNC('month', ad.aligned_date) - INTERVAL '1 year')::DATE    AS prior_yr_month_start,
        EXTRACT(DAY FROM ad.aligned_date)::INT                              AS days_elapsed
    FROM vw_aligned_date ad
),

today_rev AS (
    -- "today" = any data beyond the aligned date (early submitters)
    SELECT location, ROUND(SUM(real_transaction_value)::NUMERIC, 0) AS revenue
    FROM mv_transaction_master, ref
    WHERE "date" > ref.yesterday AND real_transaction_value > 0
    GROUP BY location
),
yesterday_rev AS (
    SELECT location, ROUND(SUM(real_transaction_value)::NUMERIC, 0) AS revenue
    FROM mv_transaction_master, ref
    WHERE "date" = ref.yesterday AND real_transaction_value > 0
    GROUP BY location
),
day_before_rev AS (
    SELECT location, ROUND(SUM(real_transaction_value)::NUMERIC, 0) AS revenue
    FROM mv_transaction_master, ref
    WHERE "date" = ref.yesterday - 1 AND real_transaction_value > 0
    GROUP BY location
),
current_mtd AS (
    SELECT location,
        ROUND(SUM(real_transaction_value)::NUMERIC, 0)  AS revenue,
        COUNT(DISTINCT transaction_id)                   AS txn_count
    FROM mv_transaction_master, ref
    WHERE "date" >= ref.month_start AND "date" <= ref.yesterday
      AND real_transaction_value > 0
    GROUP BY location
),
prior_mtd AS (
    SELECT location, ROUND(SUM(real_transaction_value)::NUMERIC, 0) AS revenue
    FROM mv_transaction_master, ref
    WHERE "date" >= ref.prior_month_start
      AND "date" <  ref.prior_month_start + (ref.days_elapsed || ' days')::INTERVAL
      AND real_transaction_value > 0
    GROUP BY location
),
prior_year_mtd AS (
    SELECT location, ROUND(SUM(real_transaction_value)::NUMERIC, 0) AS revenue
    FROM mv_transaction_master, ref
    WHERE "date" >= ref.prior_yr_month_start
      AND "date" <  ref.prior_yr_month_start + (ref.days_elapsed || ' days')::INTERVAL
      AND real_transaction_value > 0
    GROUP BY location
),

all_branches AS (SELECT DISTINCT location FROM mv_transaction_master),
grand_total   AS (SELECT SUM(revenue) AS total FROM current_mtd)

SELECT
    ab.location                                                             AS branch,

    -- The aligned cutoff date (same for all branches)
    (SELECT yesterday FROM ref)                                             AS data_as_at,

    -- DoD: aligned date vs the day before it
    COALESCE(yd.revenue,  0)                                                AS revenue_aligned_date,
    COALESCE(dby.revenue, 0)                                                AS revenue_prior_day,
    ROUND(CASE WHEN COALESCE(dby.revenue, 0) = 0 THEN NULL
               ELSE (COALESCE(yd.revenue, 0) - dby.revenue)::NUMERIC
                    / dby.revenue * 100
          END::NUMERIC, 1)                                                  AS dod_growth_pct,

    -- MoM
    COALESCE(cm.revenue,  0)                                                AS revenue_mtd,
    COALESCE(pm.revenue,  0)                                                AS revenue_prior_mtd,
    ROUND(CASE WHEN COALESCE(pm.revenue, 0) = 0 THEN NULL
               ELSE (COALESCE(cm.revenue, 0) - pm.revenue)::NUMERIC
                    / pm.revenue * 100
          END::NUMERIC, 1)                                                  AS mom_growth_pct,

    -- YoY
    COALESCE(py.revenue,  0)                                                AS revenue_prior_year_mtd,
    ROUND(CASE WHEN COALESCE(py.revenue, 0) = 0 THEN NULL
               ELSE (COALESCE(cm.revenue, 0) - py.revenue)::NUMERIC
                    / py.revenue * 100
          END::NUMERIC, 1)                                                  AS yoy_growth_pct,

    -- Context
    COALESCE(cm.txn_count, 0)                                               AS txn_count_mtd,
    ROUND(COALESCE(cm.revenue, 0)::NUMERIC / NULLIF(cm.txn_count, 0), 0)   AS avg_basket_mtd,
    ROUND(COALESCE(cm.revenue, 0)::NUMERIC / NULLIF(gt.total, 0) * 100, 1) AS contribution_pct,

    -- Revenue submitted beyond the aligned cutoff (branch is ahead — not included in comparisons)
    COALESCE(td.revenue, 0)                                                 AS revenue_submitted_ahead

FROM all_branches ab
CROSS JOIN grand_total gt
LEFT JOIN today_rev      td  USING (location)
LEFT JOIN yesterday_rev  yd  USING (location)
LEFT JOIN day_before_rev dby USING (location)
LEFT JOIN current_mtd    cm  USING (location)
LEFT JOIN prior_mtd      pm  USING (location)
LEFT JOIN prior_year_mtd py  USING (location)
ORDER BY COALESCE(cm.revenue, 0) DESC;


-- ============================================================
-- VIEW 4: vw_daily_revenue  (rolling 90 days, DoD sparklines)
-- Also capped at aligned_date so partial days don't appear
-- ============================================================

DROP VIEW IF EXISTS vw_daily_revenue CASCADE;

CREATE VIEW vw_daily_revenue AS

WITH daily AS (
    SELECT
        t.location,
        t."date"                                                AS sale_date,
        ROUND(SUM(t.real_transaction_value)::NUMERIC, 0)       AS revenue,
        COUNT(DISTINCT t.transaction_id)                        AS txn_count
    FROM mv_transaction_master t
    -- Cap at aligned_date so no branch shows partial-day data
    JOIN vw_aligned_date ad ON t."date" <= ad.aligned_date
    WHERE t."date" >= CURRENT_DATE - 90
      AND t.real_transaction_value > 0
    GROUP BY t.location, t."date"
),
daily_totals AS (
    SELECT sale_date, SUM(revenue) AS all_branches_revenue
    FROM daily GROUP BY sale_date
)

SELECT
    d.location,
    d.sale_date,
    TO_CHAR(d.sale_date, 'DD Mon')                                          AS day_label,
    TO_CHAR(d.sale_date, 'Dy')                                              AS day_name,
    EXTRACT(DOW FROM d.sale_date)::INT                                      AS day_of_week,
    d.revenue,
    d.txn_count,
    ROUND(d.revenue::NUMERIC / NULLIF(d.txn_count, 0), 0)                  AS avg_basket,
    ROUND(d.revenue::NUMERIC / NULLIF(dt.all_branches_revenue, 0) * 100, 1) AS daily_contribution_pct,
    COALESCE(LAG(d.revenue) OVER (PARTITION BY d.location ORDER BY d.sale_date), 0) AS prior_day_revenue,
    ROUND(
        CASE WHEN LAG(d.revenue) OVER (PARTITION BY d.location ORDER BY d.sale_date) IS NULL
                  OR LAG(d.revenue) OVER (PARTITION BY d.location ORDER BY d.sale_date) = 0
             THEN NULL
             ELSE (d.revenue - LAG(d.revenue) OVER (PARTITION BY d.location ORDER BY d.sale_date))::NUMERIC
                  / LAG(d.revenue) OVER (PARTITION BY d.location ORDER BY d.sale_date) * 100
        END::NUMERIC
    , 1)                                                                    AS dod_growth_pct
FROM daily d
JOIN daily_totals dt USING (sale_date)
ORDER BY d.sale_date DESC, d.location;


-- ============================================================
-- SANITY CHECKS
-- ============================================================
-- Check the aligned date and who is lagging:
-- SELECT * FROM vw_aligned_date;

-- Check KPIs (last_full_day should show the aligned date):
-- SELECT report_date, last_full_day, lagging_branches, revenue_mtd, mom_growth_pct FROM vw_exec_kpis;

-- Branch performance (all branches should have real numbers, no -100% DoD):
-- SELECT * FROM vw_branch_performance;

-- Daily trend capped at aligned date:
-- SELECT * FROM vw_daily_revenue ORDER BY sale_date DESC, location LIMIT 20;