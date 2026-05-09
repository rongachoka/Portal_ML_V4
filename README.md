# Portal ML V4 — Pharmacy Analytics Pipeline

End-to-end data pipeline for Portal Pharmacy. Ingests customer conversations from Respond.io, daily POS sales from SharePoint, website orders, and Meta Ads data, then enriches and attributes all of it for Power BI reporting.

---

## What This Project Does

Portal Pharmacy runs six branches across Nairobi. Customers reach out via WhatsApp, Instagram, Facebook, TikTok, and web chat. The pipeline:

1. **Ingests** Respond.io conversation exports and POS cashier/sales files daily
2. **Classifies** customer conversations into concern categories (acne, skincare, supplements, etc.) using a CrossEncoder AI model plus heuristic signal detectors (M-Pesa payment detection, brand detection, zone/location detection)
3. **Attributes** social media inquiries to POS transactions by matching phone numbers and M-Pesa codes — establishing which chats resulted in an in-store sale
4. **Enriches** attributed sales with Knowledge Base brand and category labels
5. **Outputs** Power BI-ready CSV files covering sessions, staff performance, ad ROI, social sales, website orders, and client tiers

---

## Pipeline Architecture

Two runners exist. Both cover the core 11 stages; the morning runner adds reporting stages.

### run_pipeline.py — Nightly Automated Runner (V4 Master)

Launched by `run_pipeline.bat` via Windows Task Scheduler. Implements incremental watermarks — only new data since the last run is processed.

```
Stage 1   Cleaning          Incremental load from Respond.io CSVs into
                            PostgreSQL staging tables + interim parquet.
                            Watermark per source (messages / contacts /
                            conversations / ads).

Stage 2   ML Inference      Sessionises messages (96-hour gap = new session).
                            Applies CrossEncoder AI classification + heuristics:
                            M-Pesa detection, brand detection, zone detection,
                            concern detection, price objection detection.
                            Appends new sessions to final_tagged_sessions.parquet.
          [Early-exit gate: if no new messages, stages 2, 4, 7 are skipped.
           Stages 3, 5, 6, 8–11 still run — they have their own dedup.]

Stage 3   POS ETL           Reads SharePoint downloads (sales + cashier files
                            per branch). Merges, deduplicates, date-filters to
                            Jan 2025+. Outputs all_locations_sales.csv.

Stage 4   Analytics Pass 1  Produces fact_sessions_enriched.csv for attribution
                            input. ordered_via is empty at this point — expected.

Stage 5   Attribution       Matches Respond.io sessions to POS transactions
                            (phone + M-Pesa waterfall). Stamps ordered_via back
                            onto fact_sessions_enriched.csv.

Stage 6   Enrichment        Labels products on attributed sales against the
                            Knowledge Base (brand + category).

Stage 7   Analytics Pass 2  Re-runs analytics with ordered_via populated.
                            Produces final versions of all fact tables.

Stage 8   Staff Performance Per-agent metrics: sessions handled, response times,
                            conversion rates, POS cross-referenced.

Stage 9   Concern Journey   Maps the path from customer concern (e.g. acne) to
                            purchase across sessions.

Stage 10  Recommendations   AI-assisted product recommendations per staff member
                            based on concern patterns and conversion data.

Stage 11  Website Orders    Ingests portal_order_with_prices.csv, produces
                            fact_website_orders.csv and website_fact_lineitems.csv.
```

**Single-step mode:**
```
python run_pipeline.py --step cleaning
python run_pipeline.py --step ml
python run_pipeline.py --step pos
python run_pipeline.py --step analytics
python run_pipeline.py --step attribution
python run_pipeline.py --force          # ignore watermarks, reprocess everything
```

---

### run_pipeline_copy.py — Morning Manual Runner

Run manually each morning. Calls the `_local_copy` module variants and adds reporting stages not in the nightly runner:

| Extra stage | What it does |
|---|---|
| `concern_insights` | Daily concern summary digest |
| `social_sales_direct` | Power BI social sales dashboard source (cashier `ordered_via = respond.io` path — bypasses attribution waterfall) |
| `social_sales_etl` | Gap audit report comparing cashier Amount vs sales line-item revenue |
| `staff_performance_pos` | POS-side staff performance report |
| `ad_performance` | Meta Ads ROI matrix (spend → inquiries → conversions → revenue) |
| `load_to_postgres` | Pushes POS fact table to PostgreSQL |
| `website_sales_etl` | Website sales ETL (secondary path) |

---

## Environment Setup

### Prerequisites

- Python 3.11+
- PostgreSQL client (`psql`) on PATH for `download_csv.py`
- Access to the Portal Pharmacy SharePoint drive
- Access to the PostgreSQL server at the address in `.env`

### Python dependencies

```bash
pip install -r requirements.txt
```

Key packages: `pandas`, `torch`, `sentence-transformers`, `psycopg2-binary`, `python-dotenv`, `openpyxl`, `tqdm`, `scikit-learn`.

### `.env` file

Create a `.env` file at the root of `Portal_ML_V4/` with the following variables:

```env
# PostgreSQL
DB_USER=your_db_user
DB_PASSWORD=your_db_password
DB_HOST=your_db_host
DB_PORT=5432
DB_NAME=portal_pharmacy

DB_SSLMODE=disable          # or require for SSL

# Microsoft Graph API (SharePoint access)
MS_TENANT_ID=your_tenant_id
MS_CLIENT_ID=your_client_id
MS_CLIENT_SECRET=your_client_secret

# Respond.io API
RESPOND_IO_TOKEN=your_respond_io_jwt_token

# SharePoint Drive ID
DRIVE_ID=your_sharepoint_drive_id
```

> The `.env` file is excluded from git. Never commit credentials.

---

## How to Run

### Full nightly pipeline (manual trigger)
```bash
python run_pipeline.py
```

### Morning manual runner
```bash
python run_pipeline_copy.py
```

### Nightly automated run
The pipeline is registered with Windows Task Scheduler via `setup_scheduler.bat`. It calls `run_nightly.bat` which:
1. Runs `sharepoint_downloader.py` to pull new files from SharePoint
2. Runs the pipeline

### SharePoint sync only
```bash
python sharepoint/sharepoint_downloader.py
```

### Client list export (run manually as needed)
```bash
python run_client_export.py
```
Outputs `Portal_Client_List.xlsx` with Platinum / Gold / Silver / Bronze customer segmentation.

### DB view export
```bash
python download_csv.py
```
Exports PostgreSQL views to `data/03_processed/db_exports/`.

---

## Folder Structure

```
Portal_ML_V4/
│
├── run_pipeline.py              Nightly automated runner (V4, watermark-based)
├── run_pipeline_copy.py         Morning manual runner (adds reporting stages)
├── run_client_export.py         Client list Excel export with tier segmentation
├── monthly_revenue_audit.py     QA script — reconciles cashier vs pipeline revenue
├── respond_io_tags.py           Builds Respond.io contact upload CSV (tags + tiers)
├── stack_march.py               Monthly utility — stacks branch cashier sheets
├── download_csv.py              Exports PostgreSQL views to CSV via psql
├── run_pipeline.bat             Scheduler entry point for the nightly runner
├── run_nightly.bat              Two-step: SharePoint download → pipeline
├── setup_scheduler.bat          One-time: registers run_nightly.bat in Task Scheduler
├── .env                         Credentials (not committed)
│
├── src/
│   ├── config/                  Shared constants and configuration
│   │   ├── settings.py          All file paths, DB connection string, ML thresholds
│   │   ├── constants.py         ML label lists and mappings
│   │   ├── brands.py            Brand alias map for fuzzy matching
│   │   ├── pos_aliases.py       POS term and brand aliases for KB matching
│   │   ├── tag_rules.py         Canonical category inference rules
│   │   ├── concerns.py          Concern keyword patterns
│   │   ├── department_map.py    Branch department → canonical category map
│   │   ├── zones.py             Nairobi zone / location definitions
│   │   └── ad_registry.py       Meta Ads campaign registry
│   │
│   ├── core/
│   │   ├── mpesa_engine.py      M-Pesa transaction code detection and parsing
│   │   └── signal_detectors.py  Brand, zone, concern, price signal extractors
│   │
│   ├── utils/
│   │   ├── text_cleaner.py      Message text extraction and noise filtering
│   │   ├── phone.py             Phone number normalisation (Kenya formats)
│   │   ├── name_cleaner.py      Customer name cleaning utilities
│   │   └── excel_formatters.py  Excel export styling helpers
│   │
│   └── pipelines/
│       ├── crm_chat/            Chat pipeline — the core analytics path
│       │   ├── cleaning.py           Incremental Respond.io ingest + DB staging
│       │   ├── ml_inference.py       CrossEncoder classification + session building
│       │   ├── analytics.py          fact_sessions_enriched + all analytic fact tables
│       │   ├── staff_performance.py  Per-agent metrics (chat + POS cross-reference)
│       │   ├── analyze_concern_journey.py  Concern-to-purchase journey mapping
│       │   ├── concern_insights.py   Daily concern digest
│       │   ├── clean_products.py     Product name cleaning against KB
│       │   ├── extract_customer_list.py    Customer list extraction utility
│       │   │
│       │   │   [Local-copy variants called by run_pipeline_copy.py:]
│       │   ├── cleaning_local_copy.py
│       │   ├── ml_inference_copy.py
│       │   ├── analytics_copy.py
│       │   └── analytics_copy_050526.py
│       │
│       ├── pos_finance/         POS ETL
│       │   ├── etl_local.py          SharePoint downloads ETL (V5-LOCAL, production)
│       │   ├── load_to_postgres.py   Push all_locations_sales to PostgreSQL
│       │   └── extract_customers_pos_updated.py  Customer extraction from POS data
│       │
│       ├── attribution/         Social-to-POS matching
│       │   ├── jan25_26_link_social_to_pos_new.py  Attribution waterfall V7
│       │   ├── enrich_attribution_products.py      Brand/category labelling on attributed sales
│       │   ├── social_sales_direct.py              Power BI source via ordered_via field
│       │   ├── social_sales_etl.py                 Gap audit — cashier vs sales revenue
│       │   ├── ad_performance.py                   Meta Ads ROI matrix
│       │   └── social_ad_attribution.py            Ad-to-session matching
│       │
│       ├── website/
│       │   └── website_orders.py     Website orders ETL → fact_website_orders.csv
│       │
│       └── extract_concern_recommendations.py  Staff product recommendations
│
├── sharepoint/
│   ├── sharepoint_downloader.py  Downloads new/changed files from SharePoint
│   ├── sharepoint_auth.py        Microsoft Graph API authentication
│   ├── sharepoint_paths.py       SharePoint folder path definitions per branch
│   └── db.py                     PostgreSQL connection wrapper (used by downloader)
│
├── data/
│   ├── 01_raw/                  Source inputs (Respond.io exports, KB, ads, website)
│   │   ├── Respond IO Messages History.csv
│   │   ├── Respond IO Conversations History.csv
│   │   ├── Respond IO Contacts History.csv
│   │   ├── Final_Knowledge_Base_PowerBI_New.csv   Product knowledge base
│   │   ├── ads/                 Meta Ads contact files (contacts-added/connected)
│   │   ├── meta_ads/            Meta Ads spend/performance files
│   │   └── sharepoint_downloads/{Branch}/sales_reports/ + cashier_reports/
│   │
│   ├── 02_interim/              Intermediate files (auto-regenerated each run)
│   │   ├── cleaned_messages.parquet / .csv
│   │   ├── cleaned_conversations.parquet / .csv
│   │   └── cleaned_contacts.parquet / .csv
│   │
│   └── 03_processed/            Final pipeline outputs consumed by Power BI
│       ├── final_tagged_sessions.parquet / .csv   Core ML output
│       ├── fact_sessions_enriched.csv             Analytics fact table (main)
│       ├── sales_attribution/                     Attribution waterfall outputs
│       ├── ads/                                   Ad performance tables
│       ├── website_data/                          Website order tables
│       └── db_exports/                            PostgreSQL view snapshots
│
├── api_request/                 Respond.io API import scripts (history pulls)
├── dashboards/                  Power BI .pbix files (excluded from git)
├── docs/                        Documentation and ERD
├── logs/                        Pipeline run logs (one per day)
├── models/                      Vision model files and classifiers
├── sales_reports_all/           Staff performance report generation
├── sql/                         Migration and schema scripts
├── website_orders/              Website sales ETL (secondary path)
└── archive/                     All superseded files, organised by pipeline area
```

---

## Database

**Host:** Remote PostgreSQL server (configured in `.env`)
**Database:** `portal_pharmacy`

### Staging tables (written by the pipeline)
| Table | Content |
|---|---|
| `stg_messages` | All Respond.io messages, incremental insert |
| `stg_conversations` | Respond.io conversations, full upsert |
| `stg_contacts` | Respond.io contacts, full upsert |
| `stg_ads` | Meta Ads contact events, incremental insert |
| `pipeline_watermarks` | Per-source last-loaded timestamps |

### Views and materialized views (read by Power BI / download_csv.py)
`vw_sales_base`, `vw_sales_with_margin`, `vw_dead_stock`, `mv_transaction_master`, `mv_client_list`, `dim_branch`, `dim_departments`, `vw_aligned_date`, `vw_inventory_snapshot`

See `docs/ERD.md` for the full entity-relationship diagram.

---

## Data Sources

| Source | Format | Frequency | Ingest method |
|---|---|---|---|
| Respond.io Messages | CSV export | Daily | Manual download → `cleaning.py` |
| Respond.io Contacts | CSV export | Daily | Manual download → `cleaning.py` |
| Respond.io Conversations | CSV export | Daily | Manual download → `cleaning.py` |
| Meta Ads | CSV export | Weekly | Manual download → `cleaning.py` |
| POS Sales reports | CSV / XLSX | Daily | SharePoint auto-download |
| POS Cashier reports | XLSM / XLSX | Daily | SharePoint auto-download |
| Website orders | CSV | On demand | Manual → `website_orders.py` |
| Knowledge Base | CSV | As updated | Manual → `Final_Knowledge_Base_PowerBI_New.csv` |

---

## Key Output Files for Power BI

| File | Description |
|---|---|
| `data/03_processed/fact_sessions_enriched.csv` | One row per Respond.io session — tags, tier, channel, zone, ordered_via |
| `data/03_processed/sales_attribution/social_sales_direct.csv` | Social media sales (respond.io ordered_via path) |
| `data/03_processed/sales_attribution/attributed_sales_waterfall_v7.csv` | Phone-matched attribution waterfall |
| `data/03_processed/ads/fact_ad_performance.csv` | One row per Meta Ad — spend, inquiries, revenue |
| `data/03_processed/ads/fact_ad_products.csv` | Ad × product drill-through table |
| `data/03_processed/website_data/fact_website_orders.csv` | Website orders |
| `data/03_processed/Staff_Performance_Test.csv` | Staff metrics |
| `Portal_Client_List.xlsx` | Segmented client list (run_client_export.py) |
