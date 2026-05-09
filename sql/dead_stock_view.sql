CREATE OR REPLACE VIEW vw_dead_stock AS
WITH branch_map AS (
    SELECT * FROM (VALUES
        ('ABC',          'PHARMART_ABC'),
        ('Centurion 2R', 'CENTURION_2R'),
        ('Galleria',     'GALLERIA'),
        ('Milele',       'NGONG_MILELE'),
        ('Portal 2R',    'PORTAL_2R'),
        ('Portal CBD',   'PORTAL_CBD')
    ) AS t(qty_branch, sales_branch)
),
latest_snapshot AS (
    SELECT branch, MAX(snapshot_date) AS snapshot_date
    FROM stg_qty_list
    GROUP BY branch
),
current_stock AS (
    SELECT DISTINCT ON (q.branch, q.description, q.item_lookup_code)
        q.branch,
        bm.sales_branch,
        q.description,
        q.department,
        q.item_lookup_code,
        q.on_hand,
        q.supplier,
        l.snapshot_date
    FROM stg_qty_list q
    JOIN latest_snapshot l 
        ON q.branch = l.branch 
        AND q.snapshot_date = l.snapshot_date
    JOIN branch_map bm
        ON q.branch = bm.qty_branch
    WHERE q.on_hand > 0
      AND q.item_lookup_code IS NOT NULL
      AND LOWER(TRIM(q.item_lookup_code)) NOT IN ('nan','none','')
    ORDER BY q.branch, q.description, q.item_lookup_code, q.on_hand DESC
),
deduped_costs AS (
    SELECT DISTINCT ON (description, location)
        description,
        location,
        cost_price,
        selling_price,
        margin_pct
    FROM dim_products
    WHERE cost_price IS NOT NULL
      AND cost_price > 0
    ORDER BY description, location, cost_price DESC NULLS LAST
),
last_sale_by_barcode AS (
    SELECT 
        item                AS item_lookup_code,
        location            AS sales_branch,
        MAX(sale_date)      AS last_sold
    FROM fact_sales_lineitems
    WHERE item IS NOT NULL
      AND item != '#NULL#'
      AND description NOT ILIKE '%GOODS%'
    GROUP BY item, location
),
last_sale_by_description AS (
    SELECT 
        UPPER(TRIM(description))    AS description,
        location                    AS sales_branch,
        MAX(sale_date)              AS last_sold
    FROM fact_sales_lineitems
    WHERE description IS NOT NULL
      AND description NOT ILIKE '%GOODS%'
    GROUP BY UPPER(TRIM(description)), location
)
SELECT 
    s.branch,
    s.sales_branch,
    s.description,
    s.department,
    s.item_lookup_code,
    s.on_hand,
    s.supplier,
    s.snapshot_date,
    dp.cost_price,
    dp.selling_price,
    dp.margin_pct,
    ROUND(s.on_hand * COALESCE(dp.cost_price, 0), 2)    AS stock_value,
    COALESCE(lb.last_sold, ld.last_sold)                 AS last_sold,
    CASE 
        WHEN COALESCE(lb.last_sold, ld.last_sold) IS NULL 
            THEN 999
        ELSE CURRENT_DATE - COALESCE(lb.last_sold, ld.last_sold)
    END                                                  AS days_since_sold,
    CASE
        WHEN COALESCE(lb.last_sold, ld.last_sold) IS NULL 
            THEN 'Never Sold'
        WHEN CURRENT_DATE - COALESCE(lb.last_sold, ld.last_sold) >= 90 
            THEN 'Dead Stock'
        WHEN CURRENT_DATE - COALESCE(lb.last_sold, ld.last_sold) >= 60 
            THEN 'Stagnant'
        WHEN CURRENT_DATE - COALESCE(lb.last_sold, ld.last_sold) >= 30 
            THEN 'Slow Mover'
        ELSE 'Active'
    END                                                  AS stock_status,
    CASE
        WHEN dp.cost_price IS NOT NULL THEN 'Costed'
        ELSE 'No Cost Data'
    END                                                  AS cost_status
FROM current_stock s
LEFT JOIN deduped_costs dp
    ON UPPER(TRIM(s.description)) = UPPER(TRIM(dp.description))
    AND s.sales_branch = dp.location
LEFT JOIN last_sale_by_barcode lb
    ON s.item_lookup_code = lb.item_lookup_code
    AND s.sales_branch = lb.sales_branch
LEFT JOIN last_sale_by_description ld
    ON UPPER(TRIM(s.description)) = ld.description
    AND s.sales_branch = ld.sales_branch
WHERE 
    COALESCE(lb.last_sold, ld.last_sold) IS NULL 
    OR CURRENT_DATE - COALESCE(lb.last_sold, ld.last_sold) >= 70;