-- ============================================================
-- Portal Pharmacy ML V3 — Respond.io Staging Tables
-- ============================================================
-- Run order: this file is idempotent (CREATE TABLE IF NOT EXISTS)
-- so it is safe to re-run without dropping existing data.
-- ============================================================


-- ============================================================
-- PIPELINE WATERMARKS
-- One row per source. Updated after every successful load.
-- ============================================================

CREATE TABLE IF NOT EXISTS pipeline_watermarks (
    source                  VARCHAR(50)   PRIMARY KEY,
    -- e.g. 'messages', 'conversations', 'contacts', 'ads'

    max_datetime_loaded     TIMESTAMPTZ   NOT NULL,
    -- The highest Date & Time / Timestamp / LastInteractionTime
    -- already in the staging table. Next run reads only rows
    -- strictly after this value.

    last_run_at             TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    rows_loaded_last_run    INTEGER       NOT NULL DEFAULT 0,
    total_rows_stored       BIGINT        NOT NULL DEFAULT 0
    -- Cumulative count — lets you see growth over time without
    -- counting the staging table itself every run.
);

COMMENT ON TABLE pipeline_watermarks IS
    'Tracks ingestion progress per Respond.io source file. '
    'Survives daily CSV overwrites — the DB holds the canonical watermark.';


-- ============================================================
-- STG_MESSAGES
-- Append-only. Dedup key: message_id.
-- Source: messages history export (overwritten daily).
-- ============================================================

CREATE TABLE IF NOT EXISTS stg_messages (
    id                  BIGSERIAL       PRIMARY KEY,

    -- Natural key from source
    message_id          VARCHAR(100)    NOT NULL UNIQUE,

    -- Source columns (names preserved for traceability)
    date_time           TIMESTAMPTZ     NOT NULL,
    sender_id           VARCHAR(100),
    sender_type         VARCHAR(50),
    -- e.g. 'contact', 'user', 'echo', 'workflow', 'broadcast'
    contact_id          BIGINT          NOT NULL,
    content_type        VARCHAR(50),
    message_type        VARCHAR(50),
    content             TEXT,
    channel_id          BIGINT,
    type                VARCHAR(50),
    sub_type            VARCHAR(50),

    -- Audit
    loaded_at           TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_stg_messages_contact_id
    ON stg_messages (contact_id);

CREATE INDEX IF NOT EXISTS idx_stg_messages_date_time
    ON stg_messages (date_time);

CREATE INDEX IF NOT EXISTS idx_stg_messages_contact_date
    ON stg_messages (contact_id, date_time);
-- ↑ This is the index cleaning.py uses when filtering
-- WHERE contact_id = ANY(%s) AND date_time > %s

COMMENT ON TABLE stg_messages IS
    'Append-only message history from Respond.io. '
    'Dedup key: message_id. Watermark: MAX(date_time) in pipeline_watermarks.';


-- ============================================================
-- STG_CONVERSATIONS
-- UPSERT on conversation_id.
-- A conversation opened on day 1 can be resolved/reassigned
-- on day 5 — the same row must be updated, not duplicated.
-- Source: conversations history export (overwritten daily).
-- ============================================================

CREATE TABLE IF NOT EXISTS stg_conversations (
    id                              BIGSERIAL   PRIMARY KEY,

    -- Natural key
    conversation_id                 BIGINT      NOT NULL UNIQUE,
    contact_id                      BIGINT      NOT NULL,

    -- Timing — source format is 'DD/MM/YYYY HH:MM' (e.g. '02/10/2025 21:34')
    -- Parse with dayfirst=True in loader to avoid Oct 2 → Feb 10 misread
    -- NULL where conversation is unresolved or first response hasn't happened
    datetime_conversation_started   TIMESTAMPTZ,
    datetime_conversation_resolved  TIMESTAMPTZ,
    datetime_first_response         TIMESTAMPTZ,
    first_assignment_timestamp      TIMESTAMPTZ,
    last_assignment_timestamp       TIMESTAMPTZ,

    -- Duration metrics stored as H:MM:SS in source (e.g. '0:00:29', '22:28:28')
    -- Stored as INTERVAL — supports direct arithmetic in SQL:
    --   WHERE first_response_time < INTERVAL '5 minutes'
    --   EXTRACT(EPOCH FROM first_response_time) / 60 → minutes as float
    -- NULL where Respond.io left the cell blank (e.g. unresolved conversations)
    first_response_time             INTERVAL,
    resolution_time                 INTERVAL,
    average_response_time           INTERVAL,
    time_to_first_assignment        INTERVAL,
    first_assignment_to_first_response_time INTERVAL,
    last_assignment_to_response_time        INTERVAL,
    first_assignment_to_close_time          INTERVAL,
    last_assignment_to_close_time           INTERVAL,

    -- Assignment / staff
    assignee                        VARCHAR(100),
    first_assignee                  VARCHAR(100),
    last_assignee                   VARCHAR(100),
    first_response_by               VARCHAR(100),
    closed_by                       VARCHAR(100),
    closed_by_source                VARCHAR(100),
    closed_by_team                  VARCHAR(100),

    -- Source / channel
    opened_by_source                VARCHAR(100),
    opened_by_channel               BIGINT,

    -- Volume metrics
    number_of_outgoing_messages     INTEGER,
    number_of_incoming_messages     INTEGER,
    number_of_responses             INTEGER,
    number_of_assignments           INTEGER,

    -- Classification
    conversation_category           VARCHAR(200),
    closing_note_summary            TEXT,

    -- Audit
    loaded_at                       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at                      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_stg_conversations_contact_id
    ON stg_conversations (contact_id);

CREATE INDEX IF NOT EXISTS idx_stg_conversations_started
    ON stg_conversations (datetime_conversation_started);

COMMENT ON TABLE stg_conversations IS
    'Upserted conversation records from Respond.io. '
    'Dedup key: conversation_id. '
    'Watermark: MAX(datetime_conversation_started) in pipeline_watermarks — '
    'but existing rows are always re-upserted so resolution/assignee changes '
    'are captured even for old conversations.';


-- ============================================================
-- STG_CONTACTS
-- UPSERT on contact_id.
-- Phone, tags, lifecycle change over time.
-- Watermark: LastInteractionTime (not DateTimeCreated) so
-- updates to existing contacts are caught, not just new ones.
-- Source: contacts history export (overwritten daily).
-- ============================================================

CREATE TABLE IF NOT EXISTS stg_contacts (
    id                      BIGSERIAL       PRIMARY KEY,

    -- Natural key
    contact_id              BIGINT          NOT NULL UNIQUE,

    -- Identity
    first_name              VARCHAR(200),
    last_name               VARCHAR(200),
    phone_number            VARCHAR(50),
    email                   VARCHAR(200),
    country                 VARCHAR(100),
    language                VARCHAR(50),

    -- CRM state (these change — UPSERT ensures we always have latest)
    tags                    TEXT,
    status                  VARCHAR(100),
    lifecycle               VARCHAR(100),
    assignee                VARCHAR(100),
    channels                TEXT,
    -- ↑ Respond.io stores this as a pipe/comma separated string

    -- Branch-specific field
    branch_contact_number   VARCHAR(100),

    -- Timestamps
    last_interaction_time   TIMESTAMPTZ,
    -- ↑ Watermark column — changes on every new message/event
    datetime_created        TIMESTAMPTZ,

    -- Audit
    loaded_at               TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_stg_contacts_last_interaction
    ON stg_contacts (last_interaction_time);

CREATE INDEX IF NOT EXISTS idx_stg_contacts_phone
    ON stg_contacts (phone_number);

COMMENT ON TABLE stg_contacts IS
    'Upserted contact records from Respond.io. '
    'Dedup key: contact_id. '
    'Watermark: MAX(last_interaction_time) — catches updates to existing '
    'contacts, not just newly created ones.';


-- ============================================================
-- STG_ADS
-- Append-only with dedup on (contact_id, timestamp, ad_id).
-- Merges contacts-added and contacts-connected into one table.
-- "-" and " -" values in ad ID columns are stored as NULL.
-- Source: two ad export files (overwritten daily).
-- ============================================================

CREATE TABLE IF NOT EXISTS stg_ads (
    id              BIGSERIAL       PRIMARY KEY,

    -- Source identifier — which file this row came from
    file_source     VARCHAR(20)     NOT NULL,
    -- 'added' or 'connected'

    -- Natural dedup key
    -- ad_id is nullable (some contacts-added rows have no ad)
    -- so we use a UNIQUE constraint across all three columns
    contact_id      BIGINT          NOT NULL,
    timestamp       TIMESTAMPTZ     NOT NULL,
    ad_id           VARCHAR(50),
    -- NULL when "-" in source

    UNIQUE (contact_id, timestamp, ad_id),

    -- Identity
    contact_name    VARCHAR(200),

    -- Attribution — nullable, "-" coerced to NULL at load time
    source          VARCHAR(200),
    sub_source      VARCHAR(200),
    ad_campaign_id  VARCHAR(50),
    ad_group_id     VARCHAR(50),

    -- Channel only present in contacts-connected, NULL for contacts-added
    channel         VARCHAR(100),

    -- Respond.io user who handled this
    user_name       VARCHAR(100),

    -- Audit
    loaded_at       TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_stg_ads_contact_id
    ON stg_ads (contact_id);

CREATE INDEX IF NOT EXISTS idx_stg_ads_timestamp
    ON stg_ads (timestamp);

CREATE INDEX IF NOT EXISTS idx_stg_ads_ad_id
    ON stg_ads (ad_id)
    WHERE ad_id IS NOT NULL;

COMMENT ON TABLE stg_ads IS
    'Merged ad attribution data from contacts-added and contacts-connected. '
    'Dedup key: (contact_id, timestamp, ad_id). '
    'All "-" / " -" values in ad/sub-source columns stored as NULL. '
    'channel is NULL for contacts-added rows (column does not exist in that file). '
    'Watermark: MAX(timestamp) in pipeline_watermarks.';


-- ============================================================
-- VERIFICATION QUERIES
-- Run these after migration to confirm tables exist and are empty
-- ============================================================

-- SELECT table_name, pg_size_pretty(pg_total_relation_size(quote_ident(table_name)))
-- FROM information_schema.tables
-- WHERE table_schema = 'public'
--   AND table_name IN (
--       'pipeline_watermarks', 'stg_messages', 'stg_conversations',
--       'stg_contacts', 'stg_ads'
--   )
-- ORDER BY table_name;