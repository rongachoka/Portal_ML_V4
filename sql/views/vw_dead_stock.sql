-- ============================================================
-- VIEW: vw_dead_stock
-- Database: portal_pharmacy
-- ============================================================

CREATE OR REPLACE VIEW vw_dead_stock AS
 WITH branch_map AS (
         SELECT t.qty_branch,
            t.sales_branch
           FROM ( VALUES
               ('ABC'::text,          'PHARMART_ABC'::text),
               ('Centurion 2R'::text, 'CENTURION_2R'::text),
               ('Galleria'::text,     'GALLERIA'::text),
               ('Milele'::text,       'NGONG_MILELE'::text),
               ('Portal 2R'::text,    'PORTAL_2R'::text),
               ('Portal CBD'::text,   'PORTAL_CBD'::text)
           ) t(qty_branch, sales_branch)
        ), latest_snapshot AS (
         SELECT stg_qty_list.branch,
            max(stg_qty_list.snapshot_date) AS snapshot_date
           FROM stg_qty_list
          GROUP BY stg_qty_list.branch
        ), current_stock AS (
         SELECT DISTINCT ON (q.branch, q.description, q.item_lookup_code)
            q.branch,
            bm.sales_branch,
            q.description,
            q.department,
            q.item_lookup_code,
            q.on_hand,
            q.supplier,
            l.snapshot_date
           FROM ((stg_qty_list q
             JOIN latest_snapshot l ON (((q.branch = l.branch) AND (q.snapshot_date = l.snapshot_date))))
             JOIN branch_map bm ON ((q.branch = bm.qty_branch)))
          WHERE ((q.on_hand > (0)::numeric)
            AND (q.item_lookup_code IS NOT NULL)
            AND (lower(TRIM(BOTH FROM q.item_lookup_code)) <> ALL (ARRAY['nan'::text, 'none'::text, ''::text]))
            AND UPPER(q.department) NOT IN ('BRANCH', 'BRANCHES'))
          ORDER BY q.branch, q.description, q.item_lookup_code, q.on_hand DESC
        ), deduped_costs AS (
         SELECT DISTINCT ON (dim_products.description, dim_products.location)
            dim_products.description,
            dim_products.location,
            dim_products.cost_price,
            dim_products.selling_price,
            dim_products.margin_pct
           FROM dim_products
          WHERE ((dim_products.cost_price IS NOT NULL) AND (dim_products.cost_price > (0)::numeric))
          ORDER BY dim_products.description, dim_products.location, dim_products.cost_price DESC NULLS LAST
        ), last_sale_by_barcode AS (
         SELECT fact_sales_lineitems.item AS item_lookup_code,
            fact_sales_lineitems.location AS sales_branch,
            max(fact_sales_lineitems.sale_date) AS last_sold
           FROM fact_sales_lineitems
          WHERE ((fact_sales_lineitems.item IS NOT NULL)
            AND (fact_sales_lineitems.item <> '#NULL#'::text)
            AND UPPER(fact_sales_lineitems.department) NOT IN ('BRANCH', 'BRANCHES'))
          GROUP BY fact_sales_lineitems.item, fact_sales_lineitems.location
        ), last_sale_by_description AS (
         SELECT upper(TRIM(BOTH FROM fact_sales_lineitems.description)) AS description,
            fact_sales_lineitems.location AS sales_branch,
            max(fact_sales_lineitems.sale_date) AS last_sold
           FROM fact_sales_lineitems
          WHERE ((fact_sales_lineitems.description IS NOT NULL)
            AND UPPER(fact_sales_lineitems.department) NOT IN ('BRANCH', 'BRANCHES'))
          GROUP BY (upper(TRIM(BOTH FROM fact_sales_lineitems.description))), fact_sales_lineitems.location
        )
 SELECT s.branch,
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
    round((s.on_hand * COALESCE(dp.cost_price, (0)::numeric)), 2) AS stock_value,
    COALESCE(lb.last_sold, ld.last_sold) AS last_sold,
    CASE
        WHEN (COALESCE(lb.last_sold, ld.last_sold) IS NULL) THEN 999
        ELSE (CURRENT_DATE - COALESCE(lb.last_sold, ld.last_sold))
    END AS days_since_sold,
    CASE
        WHEN (COALESCE(lb.last_sold, ld.last_sold) IS NULL)                          THEN 'Never Sold'::text
        WHEN ((CURRENT_DATE - COALESCE(lb.last_sold, ld.last_sold)) >= 90)           THEN 'Dead Stock'::text
        WHEN ((CURRENT_DATE - COALESCE(lb.last_sold, ld.last_sold)) >= 60)           THEN 'Stagnant'::text
        WHEN ((CURRENT_DATE - COALESCE(lb.last_sold, ld.last_sold)) >= 30)           THEN 'Slow Mover'::text
        ELSE 'Active'::text
    END AS stock_status,
    CASE
        WHEN (dp.cost_price IS NOT NULL) THEN 'Costed'::text
        ELSE 'No Cost Data'::text
    END AS cost_status
   FROM (((current_stock s
     LEFT JOIN deduped_costs dp
        ON (((upper(TRIM(BOTH FROM s.description)) = upper(TRIM(BOTH FROM dp.description)))
            AND (s.sales_branch = dp.location))))
     LEFT JOIN last_sale_by_barcode lb
        ON (((s.item_lookup_code = lb.item_lookup_code) AND (s.sales_branch = lb.sales_branch))))
     LEFT JOIN last_sale_by_description ld
        ON (((upper(TRIM(BOTH FROM s.description)) = ld.description) AND (s.sales_branch = ld.sales_branch))))
  WHERE ((COALESCE(lb.last_sold, ld.last_sold) IS NULL)
     OR ((CURRENT_DATE - COALESCE(lb.last_sold, ld.last_sold)) >= 70));
