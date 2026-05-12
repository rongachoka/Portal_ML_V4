-- ============================================================
-- VIEW: vw_inventory_snapshot
-- Database: portal_pharmacy
-- ============================================================

CREATE OR REPLACE VIEW vw_inventory_snapshot AS
 SELECT location AS branch,
    products_in_txn AS product,
    max(sale_date) AS last_sold_date,
    (CURRENT_DATE - max(sale_date)) AS days_since_sold,
    sum(item_count) AS total_qty_sold_alltime,
    sum(
        CASE
            WHEN (sale_date >= (CURRENT_DATE - 30)) THEN item_count
            ELSE 0
        END) AS qty_sold_30d,
    round(((sum(
        CASE
            WHEN (sale_date >= (CURRENT_DATE - 30)) THEN item_count
            ELSE 0
        END))::numeric / 30.0), 2) AS daily_velocity
   FROM fact_sales_transactions
  WHERE (real_transaction_value > (0)::numeric)
  GROUP BY location, products_in_txn;
