# Session Notes — May 2026 Revenue Audit
**Date:** 2026-05-11  
**Issue:** Revenue discrepancy between DB figures and accountant figures for May 2026

---

## Root Causes Identified

### 1. Wrong column used for revenue
`vw_sales_base` computed revenue from `real_transaction_value`, which traces back to `pos_txn_sum` — the sum of `total_tax_ex` (tax-exclusive line item prices). The accountants use `Total Sales Amount` (tax-inclusive). The `total_sales_amount` column existed in `stg_sales_reports` and `fact_sales_lineitems` but was never being read or written by the ETL.

### 2. GOODS filter excluded entire transactions
The old `vw_sales_base` filtered at transaction level (`products_in_txn !~~* '%GOODS%'`), dropping the full transaction value when any single line item had "GOODS" in its name. The accountants exclude only the GOODS line items themselves, not the rest of the transaction.

---

## Code Changes Made

### `src/pipelines/pos_finance/load_to_postgres.py`
Three changes — all permanent, apply to every future ETL run:

| Location | Change |
|---|---|
| `read_sales_staging()` (line 292/303) | Added `total_sales_amount` to the SELECT in both the watermarked and full queries |
| `merge_and_transform()` (lines 647–666) | Converts `total_sales_amount` to numeric, applies `total_tax_ex` fallback for pre-format rows where `total_sales_amount == 0`, then computes `pos_txn_sum` from `total_sales_amount` instead of `total_tax_ex` |
| `load_fact_lineitems()` (lines 724–786) | Added `total_sales_amount` to INSERT column list, `ON CONFLICT DO UPDATE SET`, `IS DISTINCT FROM` guard, vectorized numeric prep, and `load_cols` |

### `vw_sales_base` (live database view — no SQL file)
Recreated via `CREATE OR REPLACE VIEW`. Key changes:
- Source changed: `mv_transaction_master` → `fact_sales_lineitems JOIN fact_sales_transactions`
- Revenue now sums `fsl.total_sales_amount` instead of `real_transaction_value`
- Filter changed: transaction-level `products_in_txn !~~* '%GOODS%'` → line-item-level description filters (see below)

Extended later in the same session to also exclude inter-branch transfer and VAT placeholder rows. Final WHERE clause:
```sql
WHERE fsl.description NOT ILIKE 'GOODS%'
  AND fsl.description NOT ILIKE 'PRODUCT VAT%'
  AND fsl.description NOT ILIKE 'PRODUCT ZERO%'
  AND NOT (fsl.description ILIKE 'GOOD %' AND fsl.description ILIKE '%NON VAT%')
```
Impact: PORTAL_CBD May 1–10 revenue dropped 684,225 → 596,748 (−87,476); 22 transactions excluded. Other branches unaffected.

---

## Backfill Scripts Created

All scripts are in `src/pipelines/maintenance_checks/`:

| Script | Purpose | Status |
|---|---|---|
| `audit_may2026_revenue.py` | Full audit — schema checks, row exclusion analysis, CSV cross-check, watermark check | Done — used for diagnosis |
| `backfill_pass1_sales_amount.py` | Copy `total_sales_amount` from stg via strict join (txn_id + description + total_tax_ex) | Ran — 14,064 rows |
| `backfill_pass1b_sales_amount.py` | Looser join (drop total_tax_ex, add qty_sold) for Apr–May rows Pass 1 missed | Ran — 3,976 rows |
| `backfill_pass1c_return_rows.py` | Return rows (qty_sold < 0) skipped by Pass 1 guard — copied negative values | Ran — 87 rows |
| `reload_prep_feb2026_plus.py` | Delete Feb 2026+ fact rows + reset watermarks for specified branches | Ran for GALLERIA + PORTAL_2R, then NGONG_MILELE + PHARMART_ABC + PORTAL_CBD; then C2R after dedup; then PHARMART_ABC again after vw_sales_base filter change |
| `backfill_pass2_sales_amount.py` | Fallback: `total_sales_amount = total_tax_ex` for pre-Feb 2026 rows (scoped to `sale_date < 2026-02-01`) | Ran — 440,499 rows across all branches |

> Note: The Pass 1/1b/1c backfill scripts were superseded by the clean reload for all branches except CENTURION_2R. They can be archived.

---

## Current Branch Status

| Branch | Feb 2026+ reload | `total_sales_amount` populated | Pre-Feb 2026 fallback | May 1–10 revenue | Notes |
|---|---|---|---|---|---|
| GALLERIA | ✅ 2026-05-11 | ✅ 100% | ✅ Pass 2 | 2,763,577 | 0.01% from accountant |
| PORTAL_2R | ✅ 2026-05-11 | ✅ 100% | ✅ Pass 2 | 3,469,670 | 0.01% from accountant |
| NGONG_MILELE | ✅ 2026-05-11 | ✅ 100% | ✅ Pass 2 | 806,235 | Pending accountant comparison |
| PHARMART_ABC | ✅ reload prep done, awaiting ETL re-run | ✅ 100% (prior reload) | ✅ Pass 2 | 2,113,863 | Re-deleted Feb 2026+ after vw_sales_base filter change; needs load_to_postgres.py |
| PORTAL_CBD | ✅ 2026-05-11 | ✅ 100% | ✅ Pass 2 | 596,749 | Revised down after PRODUCT VAT/ZERO/GOOD NON VAT filter added |
| CENTURION_2R | ✅ 2026-05-11 (after dedup) | ✅ 100% | ✅ Pass 2 | 491,755 | 181 staging duplicates removed before reload |

---

## Pending Items

### PHARMART_ABC — Needs ETL re-run
Feb 2026+ fact rows were deleted and watermark reset to 2026-01-31 at end of session. Run `load_to_postgres.py` to reload. No staging issues — this is a clean reload triggered by the `vw_sales_base` filter change.

### All branches — Accountant comparison pending
May 2026 accountant figures have only been confirmed for GALLERIA and PORTAL_2R. NGONG_MILELE, PHARMART_ABC, PORTAL_CBD, and CENTURION_2R figures should be compared once the accountant provides them.

### CENTURION_2R — Completed this session
- 181 pure duplicate rows removed from `stg_sales_reports` (April 2026 — same data loaded twice from two ingestion runs)
- Feb 2026+ fact rows deleted, watermark reset, ETL re-run completed
- May 1–10 revenue: 491,755
- Revenue against accountant figures: pending

---

## DB State Summary (end of session)

```
fact_sales_lineitems.total_sales_amount:
  All branches Feb 2026+    → 100% populated
  All branches pre-Feb 2026 → total_tax_ex fallback (Pass 2)
  PHARMART_ABC Feb 2026+    → deleted, awaiting ETL reload

vw_sales_base:
  Source   : fact_sales_lineitems JOIN fact_sales_transactions
  Revenue  : SUM(total_sales_amount)
  Filter   : description NOT ILIKE 'GOODS%'
         AND description NOT ILIKE 'PRODUCT VAT%'
         AND description NOT ILIKE 'PRODUCT ZERO%'
         AND NOT (description ILIKE 'GOOD %' AND description ILIKE '%NON VAT%')

load_to_postgres.py:
  Now reads and writes total_sales_amount on every insert/upsert
  Fallback: total_tax_ex used when total_sales_amount = 0 in staging

May 1-10 2026 network revenue (vw_sales_base, post all fixes):
  PORTAL_2R    3,469,670
  GALLERIA     2,763,577
  PHARMART_ABC 2,113,863  (pre-reload figure — will update after ETL)
  NGONG_MILELE   806,235
  CENTURION_2R   491,755
  PORTAL_CBD     596,749
  TOTAL       10,241,849
```
