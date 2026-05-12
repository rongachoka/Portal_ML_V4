-- ============================================================
-- VIEW: dim_departments
-- Database: portal_pharmacy
-- ============================================================

CREATE OR REPLACE VIEW dim_departments AS
 SELECT DISTINCT department
   FROM vw_sales_with_margin
  WHERE ((department IS NOT NULL) AND (TRIM(BOTH FROM department) <> ''::text))
  ORDER BY department;
