-- ============================================================
-- VIEW: vw_sales_base
-- Database: portal_pharmacy
-- ============================================================

CREATE OR REPLACE VIEW vw_sales_base AS
 SELECT fsl.location AS branch,
    fsl.sale_date,
    count(DISTINCT fsl.transaction_id) AS txn_count,
    sum(fsl.total_sales_amount) AS revenue,
    count(DISTINCT NULLIF(TRIM(BOTH FROM fst.client_name), ''::text)) AS unique_clients,
    round((sum(fsl.total_sales_amount) / (NULLIF(count(DISTINCT fsl.transaction_id), 0))::numeric), 2) AS avg_basket,
    count(DISTINCT NULLIF(TRIM(BOTH FROM fst.sales_rep), ''::text)) AS active_staff
   FROM (fact_sales_lineitems fsl
     JOIN fact_sales_transactions fst
        ON (((fsl.transaction_id = fst.transaction_id) AND (fsl.location = fst.location))))
  WHERE (UPPER(fsl.department) NOT IN ('BRANCH', 'BRANCHES'))
  GROUP BY fsl.location, fsl.sale_date;
