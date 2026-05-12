-- ============================================================
-- VIEW: vw_sales_hourly
-- Database: portal_pharmacy
-- ============================================================

CREATE OR REPLACE VIEW vw_sales_hourly AS
 SELECT location AS branch,
    sale_date,
    sale_time,
    sale_datetime,
    (EXTRACT(hour FROM sale_datetime))::integer AS sale_hour,
    to_char(sale_datetime, 'Day'::text) AS day_of_week,
    (EXTRACT(dow FROM sale_datetime))::integer AS day_of_week_num,
    transaction_id,
    real_transaction_value,
    client_name,
    sales_rep
   FROM fact_sales_transactions
  WHERE ((real_transaction_value > (0)::numeric)
    AND (products_in_txn !~~* '%#NULL#%'::text)
    AND (products_in_txn <> ''::text)
    AND (sale_datetime IS NOT NULL)
    AND NOT EXISTS (
        SELECT 1 FROM fact_sales_lineitems fsl
        WHERE fsl.transaction_id = fact_sales_transactions.transaction_id
          AND fsl.location = fact_sales_transactions.location
          AND UPPER(fsl.department) IN ('BRANCH', 'BRANCHES')
    ));
