-- ============================================================
-- VIEW: vw_aligned_date
-- Database: portal_pharmacy
-- ============================================================

CREATE OR REPLACE VIEW vw_aligned_date AS
 WITH branch_dates AS (
         SELECT mv_transaction_master.location,
            max(mv_transaction_master.date) AS branch_last_date
           FROM mv_transaction_master
          GROUP BY mv_transaction_master.location
        ), totals AS (
         SELECT min(branch_dates.branch_last_date) AS aligned_date,
            max(branch_dates.branch_last_date) AS latest_branch_date,
            count(*) AS branch_count
           FROM branch_dates
        )
 SELECT t.aligned_date,
    t.latest_branch_date,
    t.branch_count,
    string_agg((((b.location || ' ('::text) || to_char((b.branch_last_date)::timestamp with time zone, 'DD Mon'::text)) || ')'::text), ', '::text ORDER BY b.location) AS lagging_branches
   FROM (totals t
     LEFT JOIN branch_dates b ON ((b.branch_last_date < t.latest_branch_date)))
  GROUP BY t.aligned_date, t.latest_branch_date, t.branch_count;
