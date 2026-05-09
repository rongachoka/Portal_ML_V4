CREATE OR REPLACE VIEW vw_sales_hourly AS
SELECT
    location                                    AS branch,
    sale_date,
    sale_time,
    sale_datetime,
    EXTRACT(HOUR FROM sale_datetime)::INT       AS sale_hour,
    TO_CHAR(sale_datetime, 'Day')               AS day_of_week,
    EXTRACT(DOW FROM sale_datetime)::INT        AS day_of_week_num,
    transaction_id,
    real_transaction_value,
    client_name,
    sales_rep
FROM fact_sales_transactions
WHERE real_transaction_value > 0
  AND products_in_txn NOT ILIKE '%GOODS%'
  AND products_in_txn NOT ILIKE '%#NULL#%'
  AND products_in_txn <> ''
  AND sale_datetime IS NOT NULL;