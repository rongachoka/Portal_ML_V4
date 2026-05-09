CREATE OR REPLACE VIEW vw_sales_with_margin AS
WITH deduped_costs AS (
    SELECT DISTINCT ON (dim_products.description, dim_products.location)
        dim_products.description,
        dim_products.location,
        dim_products.cost_price,
        dim_products.selling_price,
        dim_products.selling_price_incl_vat,
        dim_products.margin_pct,
        dim_products.item_barcode,
        dim_products.department_mapped
    FROM dim_products
    ORDER BY dim_products.description, dim_products.location, dim_products.cost_price DESC NULLS LAST
)
SELECT
    f.location,
    f.transaction_id,
    f.sale_date,
    l.description,
    f.client_name,
    f.phone_number,
    f.sales_rep,
    f.txn_type,
    f.ordered_via,
    l.qty_sold,
    l.total_tax_ex                      AS line_revenue,
    f.real_transaction_value            AS transaction_revenue,
    f.cashier_amount,
    f.audit_status,
    c.canonical_name,
    COALESCE(c.department, dp.department_mapped) AS department,
    c.category,
    c.supplier,
    dp.cost_price,
    dp.selling_price,
    dp.selling_price_incl_vat,
    dp.department_mapped,
    CASE
        WHEN dp.cost_price IS NOT NULL AND dp.cost_price > 0
            AND dp.selling_price_incl_vat IS NOT NULL AND dp.selling_price_incl_vat > 0
        THEN ROUND(
            (dp.selling_price_incl_vat - dp.cost_price) / dp.selling_price_incl_vat * 100,
            2
        )
        ELSE dp.margin_pct
    END                                 AS margin_pct,
    CASE
        WHEN dp.cost_price IS NOT NULL AND dp.cost_price > 0
        THEN l.total_tax_ex - (dp.cost_price * l.qty_sold)
        ELSE NULL
    END                                 AS gross_profit,
    CASE
        WHEN dp.item_barcode IS NOT NULL THEN 'Costed'
        ELSE 'No Cost Data'
    END                                 AS cost_status
FROM fact_sales_transactions f
JOIN fact_sales_lineitems l
    ON f.transaction_id = l.transaction_id
    AND f.location = l.location
LEFT JOIN dim_product_catalogue c
    ON ltrim(l.item, '0') = c.item_lookup_code
LEFT JOIN deduped_costs dp
    ON l.description = dp.description
    AND f.location = dp.location
WHERE f.real_transaction_value > 0
  AND l.description NOT ILIKE '%GOODS%'
  AND l.description NOT ILIKE '%#NULL#%'
  AND l.description IS NOT NULL;

-- -- Sales column was vat exclusive

-- DROP VIEW vw_sales_with_margin;

-- CREATE VIEW vw_sales_with_margin AS
-- WITH deduped_costs AS (
--     SELECT DISTINCT ON (description, location)
--         description,
--         location,
--         cost_price,
--         selling_price,
--         margin_pct,
--         item_barcode
--     FROM dim_products
--     ORDER BY description, location, cost_price DESC NULLS LAST
-- )
-- SELECT
--     f.location,
--     f.transaction_id,
--     f.sale_date,
--     f.products_in_txn                               AS description,
--     f.client_name,
--     f.phone_number,
--     f.sales_rep,
--     f.txn_type,
--     f.ordered_via,
--     f.item_count                                    AS qty_sold,
--     f.pos_txn_sum                                   AS line_revenue,
--     f.real_transaction_value                        AS transaction_revenue,
--     f.cashier_amount,
--     f.audit_status,
--     c.canonical_name,
--     c.department,
--     c.category,
--     c.supplier,
--     dp.cost_price,
--     dp.selling_price,
--     dp.margin_pct,
--     CASE
--         WHEN dp.cost_price IS NOT NULL AND dp.cost_price > 0
--         THEN f.pos_txn_sum - (dp.cost_price * f.item_count)
--         ELSE NULL
--     END                                             AS gross_profit,
--     CASE
--         WHEN dp.item_barcode IS NOT NULL THEN 'Costed'
--         ELSE 'No Cost Data'
--     END                                             AS cost_status
-- FROM fact_sales_transactions f
-- LEFT JOIN fact_sales_lineitems l
--     ON  f.transaction_id = l.transaction_id
--     AND f.location       = l.location
-- LEFT JOIN dim_product_catalogue c
--     ON  LTRIM(l.item, '0') = c.item_lookup_code
-- LEFT JOIN deduped_costs dp
--     ON  f.products_in_txn = dp.description
--     AND f.location        = dp.location
-- WHERE f.real_transaction_value > 0
--   AND f.products_in_txn NOT ILIKE '%GOODS%'
--   AND f.products_in_txn NOT ILIKE '%#NULL#%';