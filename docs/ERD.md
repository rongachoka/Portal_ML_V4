# Portal Pharmacy — Database ERD

**Database:** `portal_pharmacy` (PostgreSQL)
**Generated:** 2026-05-10

## Naming conventions

| Prefix | Layer |
|---|---|
| `stg_` | Staging — raw ingest, written by the pipeline on every run |
| `raw_` | Legacy raw tables — pre-dating the stg_ layer |
| `dim_` | Dimension tables — reference/lookup data |
| `fact_` | Fact tables — transactional grain |
| `vw_` | Views — derived/joined queries for Power BI |
| `mv_` | Materialized views — pre-aggregated, refreshed on demand |
| `ingestion_` | Pipeline run metadata |
| `*_watermarks` | Incremental load state tracking |

---

## Diagram

```mermaid
erDiagram

    %% ═══════════════════════════════════════════
    %% PIPELINE METADATA
    %% ═══════════════════════════════════════════

    ingestion_runs {
        int         id                   PK
        text        pipeline_name
        timestamptz started_at
        timestamptz finished_at
        text        status
        int         files_seen
        int         files_downloaded
        int         files_processed
        int         files_failed
        text        notes
    }

    ingestion_files {
        int         id                   PK
        int         run_id               FK
        text        branch
        text        report_type
        text        file_type
        text        filename
        text        file_extension
        text        sharepoint_item_id
        text        sharepoint_path
        timestamptz sharepoint_last_modified
        bigint      sharepoint_size_bytes
        text        local_path
        text        file_hash
        int         row_count
        boolean     is_canonical
        text        canonical_reason
        timestamptz downloaded_at
        timestamptz processed_at
        text        status
        text        error_message
        text        notes
    }

    pipeline_watermarks {
        varchar     source               PK
        timestamptz max_datetime_loaded
        timestamptz last_run_at
        int         rows_loaded_last_run
        bigint      total_rows_stored
    }

    branch_watermarks {
        text        branch               PK
        date        max_date_loaded      PK
        timestamptz last_updated_at
        int         last_run_id
        text        notes
        timestamptz max_datetime_loaded
    }

    fact_load_watermarks {
        text        branch               PK
        date        max_date_loaded      PK
        timestamptz last_updated_at
    }

    %% ═══════════════════════════════════════════
    %% RESPOND.IO STAGING (CRM)
    %% ═══════════════════════════════════════════

    stg_messages {
        bigint      id                   PK
        varchar     message_id
        timestamptz date_time
        varchar     sender_id
        varchar     sender_type
        bigint      contact_id           FK
        varchar     content_type
        varchar     message_type
        text        content
        bigint      channel_id
        varchar     type
        varchar     sub_type
        timestamptz loaded_at
        varchar     channel_source
        timestamptz message_timestamp
        bigint      staff_id
    }

    stg_contacts {
        bigint      id                   PK
        bigint      contact_id
        varchar     first_name
        varchar     last_name
        varchar     phone_number
        varchar     email
        varchar     country
        varchar     language
        text        tags
        varchar     status
        varchar     lifecycle
        varchar     assignee
        text        channels
        varchar     branch_contact_number
        timestamptz last_interaction_time
        timestamptz datetime_created
        timestamptz loaded_at
        timestamptz updated_at
    }

    stg_conversations {
        bigint      id                   PK
        bigint      conversation_id
        bigint      contact_id           FK
        timestamptz datetime_conversation_started
        timestamptz datetime_conversation_resolved
        timestamptz datetime_first_response
        timestamptz first_assignment_timestamp
        timestamptz last_assignment_timestamp
        interval    first_response_time
        interval    resolution_time
        interval    average_response_time
        interval    time_to_first_assignment
        interval    first_assignment_to_first_response_time
        interval    last_assignment_to_response_time
        interval    first_assignment_to_close_time
        interval    last_assignment_to_close_time
        varchar     assignee
        varchar     first_assignee
        varchar     last_assignee
        varchar     first_response_by
        varchar     closed_by
        varchar     closed_by_source
        varchar     closed_by_team
        varchar     opened_by_source
        bigint      opened_by_channel
        int         number_of_outgoing_messages
        int         number_of_incoming_messages
        int         number_of_responses
        int         number_of_assignments
        varchar     conversation_category
        text        closing_note_summary
        timestamptz loaded_at
        timestamptz updated_at
    }

    stg_ads {
        bigint      id                   PK
        varchar     file_source
        bigint      contact_id           FK
        timestamptz timestamp
        varchar     ad_id
        varchar     contact_name
        varchar     source
        varchar     sub_source
        varchar     ad_campaign_id
        varchar     ad_group_id
        varchar     channel
        varchar     user_name
        timestamptz loaded_at
    }

    %% ═══════════════════════════════════════════
    %% POS STAGING
    %% ═══════════════════════════════════════════

    stg_sales_reports {
        int         id                   PK
        int         source_file_id       FK
        text        source_filename
        text        branch
        text        department
        text        category
        text        item
        text        description
        decimal     on_hand
        date        last_sold
        decimal     qty_sold
        decimal     total_tax_ex
        text        transaction_id
        date        date_sold
        timestamptz loaded_at
        time        sale_time
        timestamptz sale_datetime
        decimal     unit_cost
        decimal     tax_amount
        decimal     total_sales_amount
        decimal     total_cost
        text        sales_rep_id
        text        sales_rep_name
    }

    stg_cashier_reports {
        int         id                   PK
        int         source_file_id       FK
        text        source_filename
        text        branch
        date        transaction_date
        text        receipt_txn_no
        decimal     amount
        decimal     txn_costs
        text        txn_time
        text        txn_type
        text        ordered_via
        text        client_name
        text        phone_number
        text        sales_rep
        timestamptz loaded_at
        text        respond_customer_id
    }

    stg_qty_list {
        int         id                   PK
        int         source_file_id       FK
        text        source_filename
        text        branch
        date        snapshot_date
        text        snapshot_date_source
        text        department
        text        category
        text        item_lookup_code
        text        description
        decimal     on_hand
        decimal     committed
        decimal     reorder_pt
        decimal     restock_lvl
        decimal     qty_to_order
        text        supplier
        text        reorder_no
        timestamptz loaded_at
    }

    stg_pos_sales {
        text        department
        text        category
        text        item
        text        description
        text        on_hand
        text        last_sold
        bigint      qty_sold
        float       total__tax_ex_
        bigint      transaction_id
        text        date_sold
        float       receipt_txn_no
        float       amount
        float       txn_costs
        text        time
        text        sales_rep
        text        client_name
        text        phone_number
        text        txn_type
        text        ordered_via
        text        audit_status
        text        location
        text        sale_date
        text        sale_date_str
        float       transaction_total
    }

    %% ═══════════════════════════════════════════
    %% LEGACY RAW TABLES
    %% ═══════════════════════════════════════════

    raw_sales {
        text        transaction_id       PK
        text        item_barcode
        timestamp   date_sold
        decimal     qty_sold
        decimal     total_tax_ex
        text        department
        text        category
        text        description
        decimal     on_hand
        timestamp   last_sold
        text        location             PK
        text        row_hash
        timestamp   created_at
    }

    raw_cashier {
        text        receipt_txn_no       PK
        text        location             PK
        text        sheet_date
        text        time
        decimal     amount
        decimal     txn_costs
        text        txn_type
        text        ordered_via
        text        client_name
        text        phone_number
        text        sales_rep
        text        source_sheet
        text        row_hash
        timestamp   created_at
    }

    _tmp_raw_cashier {
        varchar     receipt_txn_no
        varchar     sales_rep
        varchar     client_name
        varchar     phone_number
        varchar     txn_type
        varchar     ordered_via
        varchar     time
        decimal     amount
        decimal     txn_costs
        varchar     sheet_date
        varchar     source_sheet
        varchar     location
        varchar     row_hash
    }

    %% ═══════════════════════════════════════════
    %% DIMENSION TABLES
    %% ═══════════════════════════════════════════

    dim_branch {
        int         branch_id            PK
        varchar     branch_code
        varchar     branch_name
        varchar     location_area
        varchar     branch_type
        boolean     is_active
        date        opened_date
    }

    dim_product {
        int         product_id           PK
        varchar     item_barcode
        text        description
        varchar     department
        varchar     category
        varchar     brand
        varchar     supplier
        boolean     is_imported
        boolean     is_active
    }

    dim_products {
        text        item_barcode
        text        description
        text        department
        text        category
        text        supplier
        decimal     cost_price
        decimal     selling_price
        decimal     margin_amount
        decimal     margin_pct
        text        location
        timestamp   loaded_at
        text        department_mapped
        decimal     selling_price_incl_vat
    }

    dim_product_catalogue {
        text        item_lookup_code
        text        canonical_name
        text        department
        text        category
        text        supplier
        bigint      times_used
    }

    dim_product_map {
        text        pos_description      PK
        text        matched_name
        text        brand
        text        canonical_category
        text        sub_category
        text        concerns
        text        target_audience
        decimal     match_score
        text        match_status
        timestamptz mapped_at
        text        match_method
        text        department
        text        supplier
    }

    dim_knowledge_base {
        int         id                   PK
        text        code_1
        text        code_2
        text        item_code
        text        name
        text        brand
        text        canonical_category
        text        sub_category
        text        concerns
        text        target_audience
        decimal     price
        int         quantity
        text        product_link
        text        detailed_desc
        timestamptz loaded_at
    }

    %% ═══════════════════════════════════════════
    %% FACT TABLES
    %% ═══════════════════════════════════════════

    fact_sales {
        varchar     transaction_id
        varchar     branch_code          FK
        varchar     item_barcode         FK
        text        description
        varchar     department
        varchar     category
        decimal     qty_sold
        decimal     line_item_revenue
        date        date_key
        varchar     client_name
        varchar     phone_number
        varchar     sales_rep
        decimal     cashier_amount
        text        audit_status
    }

    fact_sales_lineitems {
        int         id                   PK
        text        location
        text        transaction_id
        text        department
        text        category
        text        item
        text        description
        decimal     qty_sold
        decimal     total_tax_ex
        text        date_sold
        date        sale_date
        text        sale_date_str
        text        client_name
        text        phone_number
        text        sales_rep
        text        txn_type
        text        ordered_via
        decimal     cashier_amount
        decimal     transaction_total
        text        audit_status
        timestamp   loaded_at
    }

    fact_sales_transactions {
        int         id                   PK
        text        location
        text        transaction_id
        date        sale_date
        text        sale_date_str
        text        client_name
        text        phone_number
        text        sales_rep
        text        txn_type
        text        ordered_via
        decimal     pos_txn_sum
        decimal     cashier_amount
        decimal     real_transaction_value
        text        products_in_txn
        int         item_count
        text        audit_status
        timestamp   loaded_at
        decimal     transaction_total
        time        sale_time
        timestamptz sale_datetime
    }

    fact_transactions {
        varchar     transaction_id
        varchar     branch_code          FK
        date        date_key
        text        client_name
        text        phone_number
        decimal     pos_txn_sum
        decimal     cashier_amount
        decimal     real_transaction_value
        text        audit_status
    }

    fact_inventory_snapshot {
        int         id                   PK
        text        branch
        date        snapshot_date
        text        snapshot_date_source
        text        department
        text        category
        text        item_lookup_code
        text        description
        decimal     on_hand
        decimal     committed
        decimal     reorder_pt
        decimal     restock_lvl
        decimal     qty_to_order
        text        supplier
        text        reorder_no
        int         source_file_id       FK
        timestamptz loaded_at
    }

    %% ═══════════════════════════════════════════
    %% MATERIALIZED VIEWS
    %% ═══════════════════════════════════════════

    mv_transaction_master {
        text        location
        text        transaction_id
        date        date
        text        client_name
        text        phone_number
        text        sales_rep
        text        txn_type
        text        ordered_via
        decimal     pos_txn_sum
        decimal     cashier_amount
        decimal     real_transaction_value
        text        products_in_txn
        int         item_count
        text        audit_status
    }

    mv_client_list {
        text        client_name
        text        phone_number
        text        preferred_location
        text        branches_visited
        text        products_bought
        date        first_interaction_date
        date        last_interaction_date
        decimal     total_lifetime_spend
        bigint      total_purchases
        text        lifetime_tier
        text        name_audit_flag
    }

    %% ═══════════════════════════════════════════
    %% VIEWS
    %% ═══════════════════════════════════════════

    vw_sales_base {
        text        branch
        date        sale_date
        bigint      txn_count
        decimal     revenue
        bigint      unique_clients
        decimal     avg_basket
        bigint      active_staff
    }

    vw_sales_with_margin {
        text        location
        text        transaction_id
        date        sale_date
        text        description
        text        client_name
        text        phone_number
        text        sales_rep
        text        txn_type
        text        ordered_via
        decimal     qty_sold
        decimal     line_revenue
        decimal     transaction_revenue
        decimal     cashier_amount
        text        audit_status
        text        canonical_name
        text        department
        text        category
        text        supplier
        decimal     cost_price
        decimal     cost_price_vat
        decimal     selling_price
        decimal     selling_price_incl_vat
        text        department_mapped
        decimal     margin_pct
        decimal     gross_profit
        text        cost_status
    }

    vw_sales_hourly {
        text        branch
        date        sale_date
        time        sale_time
        timestamptz sale_datetime
        int         sale_hour
        text        day_of_week
        int         day_of_week_num
        text        transaction_id
        decimal     real_transaction_value
        text        client_name
        text        sales_rep
    }

    vw_basket_analysis {
        text        transaction_id
        text        branch
        date        sale_date
        text        client_name
        text        phone_number
        decimal     basket_value
        bigint      departments_per_txn
        bigint      unique_products_per_txn
        decimal     total_units_per_txn
        int         has_skincare
        int         has_cosmetics
        int         has_rx
        int         has_otc
        int         has_supplements
        int         has_baby_care
        int         has_hair_care
        int         has_medicine
        int         has_antibiotics
        int         has_feminine_care
        int         has_first_aid
        int         has_lip_care
        int         has_medical_devices
        int         has_men_care
        int         has_oral_care
        int         has_perfumes
        int         has_homeopathy
        int         has_accessories
        int         has_jewellery
        int         has_cards_airtime
        int         has_vaccines
        int         has_antifungal
        int         has_contraceptives
        int         has_ppi
    }

    vw_dead_stock {
        text        branch
        text        sales_branch
        text        description
        text        department
        text        item_lookup_code
        decimal     on_hand
        text        supplier
        date        snapshot_date
        decimal     cost_price
        decimal     selling_price
        decimal     margin_pct
        decimal     stock_value
        date        last_sold
        int         days_since_sold
        text        stock_status
        text        cost_status
    }

    vw_inventory_snapshot {
        text        branch
        text        product
        date        last_sold_date
        int         days_since_sold
        bigint      total_qty_sold_alltime
        bigint      qty_sold_30d
        decimal     daily_velocity
    }

    vw_aligned_date {
        date        aligned_date
        date        latest_branch_date
        bigint      branch_count
        text        lagging_branches
    }

    dim_departments {
        text        department
    }

    %% ═══════════════════════════════════════════
    %% RELATIONSHIPS — ENFORCED FK CONSTRAINTS
    %% ═══════════════════════════════════════════

    ingestion_runs          ||--o{ ingestion_files              : "run_id"
    ingestion_files         ||--o{ stg_sales_reports            : "source_file_id"
    ingestion_files         ||--o{ stg_cashier_reports          : "source_file_id"
    ingestion_files         ||--o{ stg_qty_list                 : "source_file_id"
    ingestion_files         ||--o{ fact_inventory_snapshot      : "source_file_id"

    %% ═══════════════════════════════════════════
    %% RELATIONSHIPS — LOGICAL (no FK constraint)
    %% ═══════════════════════════════════════════

    stg_contacts            ||--o{ stg_messages                 : "contact_id"
    stg_contacts            ||--o{ stg_conversations            : "contact_id"
    stg_contacts            ||--o{ stg_ads                      : "contact_id"
    dim_branch              ||--o{ fact_sales                   : "branch_code"
    dim_branch              ||--o{ fact_transactions            : "branch_code"
    dim_products            ||--o{ fact_sales_lineitems         : "item_barcode"
    dim_products            ||--o{ vw_sales_with_margin         : "item_barcode"
    dim_product_map         ||--o{ fact_sales_lineitems         : "pos_description"
    stg_sales_reports       ||--o{ fact_sales_lineitems         : "transaction_id+branch"
    stg_cashier_reports     ||--o{ fact_sales_transactions      : "receipt_txn_no"
    fact_sales_transactions ||--o{ mv_transaction_master        : "materialized"
    mv_transaction_master   ||--o{ mv_client_list               : "phone_number"
    stg_qty_list            ||--o{ fact_inventory_snapshot      : "branch+date"
    fact_sales_transactions ||--o{ vw_sales_base                : "aggregated"
    fact_sales_lineitems    ||--o{ vw_sales_with_margin         : "transaction_id"
    fact_sales_transactions ||--o{ vw_sales_hourly              : "transaction_id"
    fact_sales_lineitems    ||--o{ vw_basket_analysis           : "transaction_id"
    stg_qty_list            ||--o{ vw_dead_stock                : "item_lookup_code"
    stg_qty_list            ||--o{ vw_inventory_snapshot        : "item_lookup_code"
    fact_sales_transactions ||--o{ vw_aligned_date              : "aggregated"
```

---

## Table Descriptions

### Pipeline Metadata

| Table | Purpose |
|---|---|
| `ingestion_runs` | One row per SharePoint sync run — counts files seen/downloaded/processed |
| `ingestion_files` | One row per file processed — tracks SharePoint metadata, hash, canonical flag |
| `pipeline_watermarks` | Last-loaded timestamp per Respond.io source (messages/contacts/conversations/ads) |
| `branch_watermarks` | Last-loaded date per branch for SharePoint POS files |
| `fact_load_watermarks` | Secondary watermark table for POS fact load tracking |

### Respond.io Staging (CRM)

| Table | Purpose |
|---|---|
| `stg_messages` | All Respond.io messages — incremental insert, dedup on message_id |
| `stg_contacts` | All Respond.io contacts — full upsert on contact_id on every run |
| `stg_conversations` | All Respond.io conversations — full upsert, captures resolve/reassign updates |
| `stg_ads` | Meta Ads contact events — incremental insert, dedup on (contact_id, timestamp, ad_id) |

### POS Staging

| Table | Purpose |
|---|---|
| `stg_sales_reports` | Sales line items from SharePoint branch sales CSVs |
| `stg_cashier_reports` | Cashier transaction header rows from branch XLSM files |
| `stg_qty_list` | Branch inventory quantity snapshots from SharePoint |
| `stg_pos_sales` | Legacy flat POS staging table (pre-stg_sales_reports split) |

### Dimension Tables

| Table | Purpose |
|---|---|
| `dim_branch` | Branch master — 6 Portal Pharmacy branches |
| `dim_product` | Product master loaded from POS system |
| `dim_products` | Extended product table with cost/price/margin data |
| `dim_product_catalogue` | Canonical product names with usage counts |
| `dim_product_map` | Fuzzy-matched POS description → KB product mapping cache |
| `dim_knowledge_base` | Full Knowledge Base — product name, brand, category, concerns, price |
| `dim_departments` | View: distinct department values from dim_products |

### Fact Tables

| Table | Purpose |
|---|---|
| `fact_sales` | Sales line items — branch-grain, linked to dim_branch and dim_products |
| `fact_sales_lineitems` | Sales line items with cashier data merged — primary ETL output of etl_local.py |
| `fact_sales_transactions` | Transaction-grain (one row per receipt) with full cashier + sales totals |
| `fact_transactions` | Legacy transaction table (pre-fact_sales_transactions) |
| `fact_inventory_snapshot` | Point-in-time inventory levels per branch from qty list files |

### Materialized Views

| View | Purpose | Refresh |
|---|---|---|
| `mv_transaction_master` | All transactions across all branches — primary Power BI source | Manual `REFRESH MATERIALIZED VIEW` |
| `mv_client_list` | Customer lifetime summary — spend, tier, branches visited | Manual |

### Views

| View | Purpose |
|---|---|
| `vw_sales_base` | Daily branch revenue summary — txn count, revenue, unique clients, avg basket |
| `vw_sales_with_margin` | Line-item view with cost/margin data joined from dim_products |
| `vw_sales_hourly` | Transaction-grain with sale hour and day-of-week for traffic pattern analysis |
| `vw_basket_analysis` | Transaction-grain with 20+ category presence flags for basket composition |
| `vw_dead_stock` | Items in inventory with no sales in >90 days, with stock value |
| `vw_inventory_snapshot` | Current inventory velocity (30-day qty sold, daily run rate) per product |
| `vw_aligned_date` | Most recent date where all branches have loaded data — used to align reports |
