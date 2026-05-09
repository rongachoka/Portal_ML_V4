"""
cleaning.py
===========
V4 — Incremental loading with DB staging and watermarks.

What changed from V3:
  - All four source files now load into staging tables (stg_messages,
    stg_conversations, stg_contacts, stg_ads) in addition to interim parquet.
  - Watermarks are read from pipeline_watermarks before processing and
    updated after a successful load. Overwriting the raw CSV no longer
    loses history — the DB holds the canonical record.
  - Messages: only rows after the watermark are inserted into stg_messages.
    The parquet written for ml_inference.py contains full message history
    for active Contact IDs only (queried from stg_messages). Inactive
    contacts are skipped entirely — this is the main speed/memory saving.
  - Conversations: full upsert on every run (conversations get resolved/
    reassigned after they open, so existing rows must be updated).
  - Contacts: full upsert on every run. Watermark on LastInteractionTime
    so updates to existing contacts are caught, not just new ones.
  - Ads: incremental insert (append-only, dedup on contact_id+timestamp+ad_id).

ml_inference.py interface is unchanged — it still reads MSG_INTERIM_PARQUET.
"""

import os
import glob
import logging
from pathlib import Path
from datetime import timezone

import pandas as pd


# ── Timezone helper ───────────────────────────────────────────────────────────
# All pipeline data (CSV exports from Respond.io, POS files) is tz-naive.
# PostgreSQL returns TIMESTAMPTZ columns as UTC-aware.
# We normalise everything to tz-naive UTC at every entry point so comparisons,
# merge_asof keys, and watermark checks never hit tz-aware vs tz-naive errors.

def _to_naive(val):
    """
    Strip timezone from a scalar Timestamp, datetime, or anything
    pd.Timestamp() accepts. Returns tz-naive or None.
    """
    if val is None or (isinstance(val, float)):
        return None
    try:
        ts = pd.Timestamp(val)
        return ts.tz_localize(None) if ts.tzinfo else ts
    except Exception:
        return None


def _series_to_naive(series: pd.Series) -> pd.Series:
    """Strip timezone from a datetime Series. Safe on already-naive Series."""
    s = pd.to_datetime(series, errors='coerce')
    if s.dt.tz is not None:
        s = s.dt.tz_localize(None)
    return s

from Portal_ML_V4.src.config.settings import (
    MSG_HISTORY_RAW, CONV_HISTORY_RAW, CONTACTS_HISTORY_RAW,
    MSG_INTERIM_PARQUET, MSG_INTERIM_CSV,
    CONV_INTERIM_PARQUET, CONV_INTERIM_CSV,
    CONTACTS_INTERIM_PARQUET, CONTACTS_INTERIM_CSV,
)
from Portal_ML_V4.sharepoint.db import bulk_insert, execute, get_connection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Watermark helpers ─────────────────────────────────────────────────────────

def get_watermark(source: str) -> pd.Timestamp | None:
    """
    Read max_datetime_loaded from pipeline_watermarks for this source.
    Returns None on first run (no row exists yet).
    """
    try:
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT max_datetime_loaded FROM pipeline_watermarks "
                "WHERE source = %s",
                (source,),
            )
            row = cur.fetchone()
            cur.close()
            if row and row[0]:
                return _to_naive(row[0])
    except Exception as e:
        logger.warning(f"[Watermark] Could not read watermark for '{source}': {e}")
    return None


def save_watermark(source: str, max_dt: pd.Timestamp, rows_loaded: int) -> None:
    """
    Upsert pipeline_watermarks row for this source.
    Only advances the watermark — never moves it backwards.
    """
    try:
        execute(
            """
            INSERT INTO pipeline_watermarks
                (source, max_datetime_loaded, last_run_at, rows_loaded_last_run, total_rows_stored)
            VALUES (%s, %s, NOW(), %s, %s)
            ON CONFLICT (source) DO UPDATE SET
                max_datetime_loaded  = GREATEST(
                    pipeline_watermarks.max_datetime_loaded,
                    EXCLUDED.max_datetime_loaded
                ),
                last_run_at          = NOW(),
                rows_loaded_last_run = EXCLUDED.rows_loaded_last_run,
                total_rows_stored    = pipeline_watermarks.total_rows_stored
                                     + EXCLUDED.rows_loaded_last_run
            """,
            (source, max_dt.to_pydatetime(), rows_loaded, rows_loaded),
        )
        logger.info(f"[Watermark] '{source}' updated to {max_dt}")
    except Exception as e:
        logger.warning(f"[Watermark] Could not save watermark for '{source}': {e}")


# ── Null cleaning helper (shared by ads + any future sources) ─────────────────

def _clean_dash_nulls(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Replace '-', ' -', '' with pd.NA in the given columns."""
    for col in cols:
        if col in df.columns:
            df[col] = df[col].replace(['-', ' -', ''], pd.NA)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 1. MESSAGES
# ══════════════════════════════════════════════════════════════════════════════

def _insert_messages(conn, df: pd.DataFrame) -> int:
    """
    Bulk insert new messages into stg_messages.
    ON CONFLICT DO NOTHING — safe to re-run if a chunk partially failed.
    Returns number of rows actually inserted.
    """
    cols = [
        "message_id", "date_time", "sender_id", "sender_type",
        "contact_id", "content_type", "message_type", "content",
        "channel_id", "type", "sub_type",
    ]
    rows = []
    for _, r in df.iterrows():
        rows.append((
            r.get("Message ID"),
            r.get("Date & Time"),
            str(r["Sender ID"]) if pd.notna(r.get("Sender ID")) else None,
            r.get("Sender Type"),
            r.get("Contact ID") if pd.notna(r.get("Contact ID")) else None,
            r.get("Content Type"),
            r.get("Message Type"),
            r.get("Content"),
            r.get("Channel ID") if pd.notna(r.get("Channel ID")) else None,
            r.get("Type"),
            r.get("Sub Type"),
        ))

    with conn.cursor() as cur:
        from psycopg2.extras import execute_values
        execute_values(
            cur,
            """
            INSERT INTO stg_messages
                (message_id, date_time, sender_id, sender_type, contact_id,
                 content_type, message_type, content, channel_id, type, sub_type)
            VALUES %s
            ON CONFLICT (message_id) DO UPDATE SET
                sender_id   = COALESCE(stg_messages.sender_id, EXCLUDED.sender_id),
                sender_type = COALESCE(stg_messages.sender_type, EXCLUDED.sender_type),
                type        = COALESCE(stg_messages.type, EXCLUDED.type),
                sub_type    = COALESCE(stg_messages.sub_type, EXCLUDED.sub_type),
                channel_id  = COALESCE(stg_messages.channel_id, EXCLUDED.channel_id)
            """,
            rows,
        )
        return cur.rowcount


def process_messages(ads_dir: Path) -> pd.DataFrame:
    """
    1. Read watermark from DB
    2. Stream CSV in chunks, keep only rows after watermark
    3. Insert new rows into stg_messages
    4. Get unique Contact IDs from new rows
    5. Query stg_messages for FULL history of those contacts
       (ensures ml_inference.py has session boundary context)
    6. Merge ad attribution onto the result
    7. Return DataFrame for downstream parquet write

    Returns empty DataFrame if nothing is new.
    """
    logger.info("📖 Processing Messages (incremental)...")

    watermark = get_watermark("messages")
    if watermark:
        logger.info(f"   Watermark: {watermark} — loading only newer rows")
    else:
        logger.info("   No watermark — first run, loading all rows")

    ALL_MSG_COLS = [
        'Date & Time', 'Sender ID', 'Sender Type', 'Contact ID',
        'Message ID', 'Content Type', 'Message Type', 'Content',
        'Channel ID', 'Type', 'Sub Type',
    ]
    KEEP_SENDER_TYPES = {'contact', 'user', 'echo', 'workflow', 'broadcast'}

    new_rows = []
    total_chunks = 0

    chunks = pd.read_csv(MSG_HISTORY_RAW, chunksize=50_000, low_memory=False)
    for i, chunk in enumerate(chunks):
        total_chunks += 1
        chunk['Date & Time']  = _series_to_naive(chunk['Date & Time'])
        chunk['Contact ID']   = pd.to_numeric(chunk['Contact ID'],  errors='coerce').astype('Int64')
        chunk['Channel ID']   = pd.to_numeric(chunk['Channel ID'],  errors='coerce').astype('Int64')
        if 'Message ID' in chunk.columns:
            chunk['Message ID'] = (
                chunk['Message ID'].astype(str).str.replace(r'\.0$', '', regex=True)
            )

        # Sender type filter
        mask = chunk['Sender Type'].isin(KEEP_SENDER_TYPES)
        chunk = chunk[mask].dropna(subset=['Contact ID'])

        # Watermark filter
        if watermark is not None:
            chunk = chunk[chunk['Date & Time'] > watermark]

        # Keep only columns that exist in source
        available = [c for c in ALL_MSG_COLS if c in chunk.columns]
        new_rows.append(chunk[available].copy())
        logger.info(f"   Chunk {i+1}: {len(new_rows[-1]):,} new rows after watermark")

    if not new_rows or all(df.empty for df in new_rows):
        logger.info("   ✅ No new messages since last run.")
        return pd.DataFrame()

    df_new = pd.concat(new_rows, ignore_index=True)
    logger.info(f"   📨 {len(df_new):,} new messages to insert")

    # ── Insert into stg_messages ──────────────────────────────────────────────
    inserted = 0
    try:
        with get_connection() as conn:
            inserted = _insert_messages(conn, df_new)
        logger.info(f"   ✅ Inserted {inserted:,} rows into stg_messages")
    except Exception as e:
        logger.error(f"   ❌ stg_messages insert failed: {e}")
        # Non-fatal — continue so interim parquet is still written

    # ── Save watermark ────────────────────────────────────────────────────────
    max_dt = df_new['Date & Time'].max()
    if pd.notna(max_dt):
        save_watermark("messages", max_dt, inserted)

    # ── Get active Contact IDs ────────────────────────────────────────────────
    active_contact_ids = df_new['Contact ID'].dropna().unique().tolist()
    logger.info(
        f"   🔎 {len(active_contact_ids):,} active contacts — "
        f"fetching full message history for session context..."
    )

    # ── Query stg_messages for full history of active contacts ────────────────
    # This is what ml_inference.py needs to correctly detect session boundaries.
    # Inactive contacts are excluded entirely — their sessions were already
    # enriched in a previous run.
    try:
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT
                    date_time   AS "Date & Time",
                    sender_id   AS "Sender ID",
                    sender_type AS "Sender Type",
                    contact_id  AS "Contact ID",
                    message_id  AS "Message ID",
                    content_type AS "Content Type",
                    message_type AS "Message Type",
                    content      AS "Content",
                    channel_id   AS "Channel ID",
                    type         AS "Type",
                    sub_type     AS "Sub Type"
                FROM stg_messages
                WHERE contact_id = ANY(%s)
                ORDER BY contact_id, date_time
                """,
                (active_contact_ids,),
            )
            rows = cur.fetchall()
            col_names = [desc[0] for desc in cur.description]
            cur.close()

        df_for_ml = pd.DataFrame(rows, columns=col_names)
        df_for_ml['Date & Time'] = _series_to_naive(df_for_ml['Date & Time'])
        df_for_ml['Contact ID']  = pd.to_numeric(df_for_ml['Contact ID'],  errors='coerce').astype('Int64')
        df_for_ml['Channel ID']  = pd.to_numeric(df_for_ml['Channel ID'],  errors='coerce').astype('Int64')
        logger.info(
            f"   📦 {len(df_for_ml):,} total messages for {len(active_contact_ids):,} "
            f"active contacts → writing to parquet"
        )

    except Exception as e:
        logger.warning(
            f"   ⚠️ Could not query stg_messages for full history: {e}. "
            f"Falling back to new rows only."
        )
        df_for_ml = df_new

    # ── Merge ad attribution ──────────────────────────────────────────────────
    df_ads = load_and_prep_ads(ads_dir)
    if not df_ads.empty:
        logger.info("🔗 Merging Ad Campaign Data (1-Hour Tolerance)...")
        df_for_ml = df_for_ml.sort_values('Date & Time')

        # Normalise both keys to tz-naive before merge_asof
        df_for_ml['Date & Time'] = _series_to_naive(df_for_ml['Date & Time'])
        df_ads['Timestamp']      = _series_to_naive(df_ads['Timestamp'])

        df_for_ml = pd.merge_asof(
            df_for_ml,
            df_ads,
            left_on='Date & Time',
            right_on='Timestamp',
            by='Contact ID',
            tolerance=pd.Timedelta('1h'),
            direction='nearest',
            suffixes=('', '_ad'),
        )
        total_ad  = df_ads['Contact ID'].nunique()
        matched   = df_for_ml[df_for_ml['Ad campaign ID'].notna()]['Contact ID'].nunique()
        logger.info(f"   AD MATCHING: {total_ad:,} ad contacts | {matched:,} matched | "
                    f"{total_ad - matched:,} ghost leads")
        df_for_ml = df_for_ml.drop(columns=['Timestamp'], errors='ignore')
    else:
        logger.info("   ⚠️ Skipping Ad Merge (No Data).")

    return df_for_ml


# ══════════════════════════════════════════════════════════════════════════════
# 2. ADS (unchanged logic, now also inserts to stg_ads)
# ══════════════════════════════════════════════════════════════════════════════

def load_and_prep_ads(ads_folder_path: Path) -> pd.DataFrame:
    """
    Load and deduplicate both ad files.
    Returns merged DataFrame for the ad-merge step in process_messages().
    Also inserts new rows into stg_ads.
    """
    all_files = (
        glob.glob(str(ads_folder_path / "contacts-added*.csv")) +
        glob.glob(str(ads_folder_path / "contacts-connected*.csv"))
    )

    if not all_files:
        logger.warning("   ⚠️ No Ad files found in /ads folder. Skipping.")
        return pd.DataFrame()

    logger.info(f"   📂 Found {len(all_files)} ad file(s).")

    ad_cols_to_keep = [
        'Timestamp', 'Contact ID', 'Contact Name', 'Source', 'Sub Source',
        'Ad campaign ID', 'Ad group ID', 'Ad ID', 'Channel', 'User',
    ]
    DASH_COLS = ['Sub Source', 'Ad campaign ID', 'Ad group ID', 'Ad ID']

    watermark = get_watermark("ads")
    dfs = []

    for f in all_files:
        file_source = 'added' if 'added' in Path(f).name.lower() else 'connected'
        try:
            temp = pd.read_csv(f, dtype=str, keep_default_na=False)

            if 'Timestamp' not in temp.columns or 'Contact ID' not in temp.columns:
                logger.warning(f"   ⚠️ Missing required columns in {Path(f).name}. Skipping.")
                continue

            temp = _clean_dash_nulls(temp, DASH_COLS)

            s = temp['Timestamp'].astype(str).str.strip().str.replace('T', ' ', regex=False)
            temp['Timestamp'] = _series_to_naive(temp['Timestamp'])

            cid = temp['Contact ID'].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
            temp['Contact ID'] = pd.to_numeric(cid, errors='coerce').astype('Int64')

            before   = len(temp)
            bad_ts   = temp['Timestamp'].isna().sum()
            bad_id   = temp['Contact ID'].isna().sum()
            logger.info(
                f"   {Path(f).name}: rows={before:,} "
                f"bad_timestamp={bad_ts:,} bad_contact_id={bad_id:,}"
            )

            temp = temp.dropna(subset=['Contact ID', 'Timestamp'])
            available = [c for c in ad_cols_to_keep if c in temp.columns]
            temp = temp[available].copy()
            temp['_file_source'] = file_source

            dfs.append(temp)

        except Exception as e:
            logger.error(f"   ❌ Error reading {Path(f).name}: {e}")

    if not dfs:
        return pd.DataFrame()

    df_ads = pd.concat(dfs, ignore_index=True)

    # Dedup across both files
    dedupe_cols = ['Contact ID', 'Timestamp']
    if 'Ad ID' in df_ads.columns:
        dedupe_cols.append('Ad ID')
    elif 'Ad campaign ID' in df_ads.columns:
        dedupe_cols.append('Ad campaign ID')
    before_dedup = len(df_ads)
    df_ads = df_ads.drop_duplicates(subset=dedupe_cols)
    logger.info(f"   ✨ Ads deduplicated: {before_dedup:,} → {len(df_ads):,} rows")

    # ── Insert new rows into stg_ads ──────────────────────────────────────────
    df_to_insert = df_ads.copy()
    if watermark is not None:
        df_to_insert = df_to_insert[df_to_insert['Timestamp'] > watermark]

    if not df_to_insert.empty:
        try:
            _insert_ads(df_to_insert)
            max_ts = df_to_insert['Timestamp'].max()
            save_watermark("ads", max_ts, len(df_to_insert))
        except Exception as e:
            logger.warning(f"   ⚠️ stg_ads insert failed (non-fatal): {e}")

    return df_ads.sort_values('Timestamp')


def _insert_ads(df: pd.DataFrame) -> None:
    from psycopg2.extras import execute_values
    rows = []
    for _, r in df.iterrows():
        rows.append((
            r.get('_file_source', 'unknown'),
            r.get('Contact ID') if pd.notna(r.get('Contact ID')) else None,
            r.get('Timestamp'),
            r.get('Ad ID')    if pd.notna(r.get('Ad ID'))    else None,
            r.get('Contact Name') if pd.notna(r.get('Contact Name')) else None,
            r.get('Source')       if pd.notna(r.get('Source'))       else None,
            r.get('Sub Source')   if pd.notna(r.get('Sub Source'))   else None,
            r.get('Ad campaign ID') if pd.notna(r.get('Ad campaign ID')) else None,
            r.get('Ad group ID')    if pd.notna(r.get('Ad group ID'))    else None,
            r.get('Channel')    if pd.notna(r.get('Channel'))    else None,
            r.get('User')       if pd.notna(r.get('User'))       else None,
        ))
    with get_connection() as conn:
        with conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO stg_ads
                    (file_source, contact_id, timestamp, ad_id, contact_name,
                     source, sub_source, ad_campaign_id, ad_group_id,
                     channel, user_name)
                VALUES %s
                ON CONFLICT (contact_id, timestamp, ad_id) DO NOTHING
                """,
                rows,
            )
    logger.info(f"   ✅ {len(rows):,} rows upserted into stg_ads")


# ══════════════════════════════════════════════════════════════════════════════
# 3. CONTACTS
# ══════════════════════════════════════════════════════════════════════════════

def process_contacts() -> pd.DataFrame:
    """
    Full upsert on every run — contact details (phone, tags, lifecycle) change.
    Watermark on LastInteractionTime catches updates, not just new contacts.
    Returns cleaned DataFrame for interim parquet (unchanged for downstream).
    """
    logger.info("📖 Processing Contacts (full upsert)...")

    CONT_COLS = [
        'ContactID', 'FirstName', 'LastName', 'PhoneNumber', 'Email',
        'Country', 'Language', 'Tags', 'Status', 'Lifecycle', 'Assignee',
        'LastInteractionTime', 'DateTimeCreated', 'Channels',
        'branch_contact_number',
    ]

    df = pd.read_csv(CONTACTS_HISTORY_RAW, dtype=str, keep_default_na=False)
    df.columns = df.columns.str.strip()

    df['ContactID'] = pd.to_numeric(
        df['ContactID'].astype(str).str.replace(r'\.0$', '', regex=True),
        errors='coerce',
    ).astype('Int64')
    df['DateTimeCreated']     = _series_to_naive(df.get('DateTimeCreated'))
    df['LastInteractionTime'] = _series_to_naive(df.get('LastInteractionTime'))

    df = df.dropna(subset=['ContactID'])
    available = [c for c in CONT_COLS if c in df.columns]
    df = df[available].copy()

    # ── Upsert to stg_contacts ────────────────────────────────────────────────
    try:
        _upsert_contacts(df)
        max_interaction = df['LastInteractionTime'].dropna().max()
        if pd.notna(max_interaction):
            save_watermark("contacts", max_interaction, len(df))
    except Exception as e:
        logger.warning(f"   ⚠️ stg_contacts upsert failed (non-fatal): {e}")

    logger.info(f"   ✅ Contacts processed: {len(df):,} records")
    return df


def _upsert_contacts(df: pd.DataFrame) -> None:
    from psycopg2.extras import execute_values

    def _s(val):
        return str(val).strip() if pd.notna(val) and str(val).strip() not in ('', 'nan') else None

    def _dt(val):
        return val.to_pydatetime() if pd.notna(val) else None

    rows = []
    for _, r in df.iterrows():
        cid = r.get('ContactID')
        if pd.isna(cid):
            continue
        rows.append((
            int(cid),
            _s(r.get('FirstName')),  _s(r.get('LastName')),
            _s(r.get('PhoneNumber')), _s(r.get('Email')),
            _s(r.get('Country')),     _s(r.get('Language')),
            _s(r.get('Tags')),        _s(r.get('Status')),
            _s(r.get('Lifecycle')),   _s(r.get('Assignee')),
            _s(r.get('Channels')),    _s(r.get('branch_contact_number')),
            _dt(r.get('LastInteractionTime')),
            _dt(r.get('DateTimeCreated')),
        ))

    with get_connection() as conn:
        with conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO stg_contacts
                    (contact_id, first_name, last_name, phone_number, email,
                     country, language, tags, status, lifecycle, assignee,
                     channels, branch_contact_number,
                     last_interaction_time, datetime_created,
                     updated_at)
                VALUES %s
                ON CONFLICT (contact_id) DO UPDATE SET
                    first_name             = EXCLUDED.first_name,
                    last_name              = EXCLUDED.last_name,
                    phone_number           = EXCLUDED.phone_number,
                    email                  = EXCLUDED.email,
                    country                = EXCLUDED.country,
                    language               = EXCLUDED.language,
                    tags                   = EXCLUDED.tags,
                    status                 = EXCLUDED.status,
                    lifecycle              = EXCLUDED.lifecycle,
                    assignee               = EXCLUDED.assignee,
                    channels               = EXCLUDED.channels,
                    branch_contact_number  = EXCLUDED.branch_contact_number,
                    last_interaction_time  = EXCLUDED.last_interaction_time,
                    updated_at             = NOW()
                """,
                [(
                    *row,
                    pd.Timestamp.now().to_pydatetime(),  # updated_at placeholder
                ) for row in rows],
            )
    logger.info(f"   ✅ {len(rows):,} contacts upserted into stg_contacts")


# ══════════════════════════════════════════════════════════════════════════════
# 4. CONVERSATIONS
# ══════════════════════════════════════════════════════════════════════════════

def process_conversations(df_msg: pd.DataFrame) -> pd.DataFrame:
    """
    Full upsert on every run — conversations get resolved/reassigned after
    they open so existing rows must be updated.
    Dayfirst=True for DD/MM/YYYY date parsing.
    Channel healing logic preserved from V3.
    Returns cleaned DataFrame for interim parquet.
    """
    logger.info("📖 Processing Conversations (full upsert)...")

    CONV_COLS = [
        'Conversation ID', 'Contact ID', 'DateTime Conversation Started',
        'DateTime Conversation Resolved', 'Opened By Source', 'Opened By Channel',
        'Assignee', 'First Assignee', 'Last Assignee', 'First Response By', 'Closed By',
        'First Response Time', 'Resolution Time', 'Average Response Time',
        'Number of Incoming Messages', 'Number of Outgoing Messages', 'Number of Responses',
        'Number of Assignments', 'Conversation Category', 'Closing Note Summary',
        'DateTime First Response', 'Closed By Source', 'Closed By Team',
        'First Assignment Timestamp', 'Last Assignment Timestamp',
        'Time to First Assignment', 'First Assignment to First Response Time',
        'Last Assignment to Response Time', 'First Assignment to Close Time',
        'Last Assignment to Close Time',
    ]

    df = pd.read_csv(CONV_HISTORY_RAW, dtype=str, keep_default_na=False)
    df.columns = df.columns.str.strip()

    df['Contact ID']      = pd.to_numeric(df['Contact ID'],      errors='coerce').astype('Int64')
    df['Conversation ID'] = pd.to_numeric(df['Conversation ID'], errors='coerce').astype('Int64')
    df['Opened By Channel'] = pd.to_numeric(df.get('Opened By Channel'), errors='coerce').astype('Int64')

    # DD/MM/YYYY HH:MM — must use dayfirst=True, then strip timezone
    for col in ['DateTime Conversation Started', 'DateTime Conversation Resolved',
                'DateTime First Response', 'First Assignment Timestamp',
                'Last Assignment Timestamp']:
        if col in df.columns:
            df[col] = _series_to_naive(pd.to_datetime(df[col], dayfirst=True, errors='coerce'))

    # ── Channel healing (preserved from V3) ───────────────────────────────────
    missing_mask = df['Opened By Channel'].isna()
    if missing_mask.any() and not df_msg.empty and 'Date & Time' in df_msg.columns:
        logger.info(f"   🔍 Healing {missing_mask.sum()} blank channels from message history...")
        df_msg_tmp = df_msg.copy()
        df_msg_tmp['time_key'] = df_msg_tmp['Date & Time'].dt.floor('min')
        channel_map = (
            df_msg_tmp.drop_duplicates(['Contact ID', 'time_key'])
            .set_index(['Contact ID', 'time_key'])['Channel ID']
        )

        def _heal(row):
            key = (row['Contact ID'], row['DateTime Conversation Started'].floor('min'))
            return channel_map.get(key, row['Opened By Channel'])

        df.loc[missing_mask, 'Opened By Channel'] = df[missing_mask].apply(_heal, axis=1)

    # Filter out workflow-opened conversations (unchanged from V3)
    if 'Opened By Source' in df.columns:
        df = df[df['Opened By Source'] != 'workflow']

    available = [c for c in CONV_COLS if c in df.columns]
    df = df[available].copy()

    # ── Upsert to stg_conversations ───────────────────────────────────────────
    try:
        _upsert_conversations(df)
        max_started = df['DateTime Conversation Started'].dropna().max()
        if pd.notna(max_started):
            save_watermark("conversations", max_started, len(df))
    except Exception as e:
        logger.warning(f"   ⚠️ stg_conversations upsert failed (non-fatal): {e}")

    logger.info(f"   ✅ Conversations processed: {len(df):,} records")
    return df


def _upsert_conversations(df: pd.DataFrame) -> None:
    from psycopg2.extras import execute_values

    def _s(val):
        return str(val).strip() if pd.notna(val) and str(val).strip() not in ('', 'nan') else None

    def _dt(val):
        return val.to_pydatetime() if pd.notna(val) else None

    def _interval(val):
        """Convert H:MM:SS string to a value Postgres accepts as INTERVAL."""
        s = _s(val)
        return s  # Postgres accepts 'H:MM:SS' strings directly for INTERVAL columns

    def _int(val):
        try:
            return int(float(val)) if pd.notna(val) and str(val).strip() not in ('', 'nan') else None
        except Exception:
            return None

    rows = []
    for _, r in df.iterrows():
        cid  = r.get('Conversation ID')
        ctid = r.get('Contact ID')
        if pd.isna(cid):
            continue
        rows.append((
            int(cid), int(ctid) if pd.notna(ctid) else None,
            _dt(r.get('DateTime Conversation Started')),
            _dt(r.get('DateTime Conversation Resolved')),
            _dt(r.get('DateTime First Response')),
            _dt(r.get('First Assignment Timestamp')),
            _dt(r.get('Last Assignment Timestamp')),
            _interval(r.get('First Response Time')),
            _interval(r.get('Resolution Time')),
            _interval(r.get('Average Response Time')),
            _interval(r.get('Time to First Assignment')),
            _interval(r.get('First Assignment to First Response Time')),
            _interval(r.get('Last Assignment to Response Time')),
            _interval(r.get('First Assignment to Close Time')),
            _interval(r.get('Last Assignment to Close Time')),
            _s(r.get('Assignee')),          _s(r.get('First Assignee')),
            _s(r.get('Last Assignee')),     _s(r.get('First Response By')),
            _s(r.get('Closed By')),         _s(r.get('Closed By Source')),
            _s(r.get('Closed By Team')),    _s(r.get('Opened By Source')),
            _int(r.get('Opened By Channel')),
            _int(r.get('Number of Outgoing Messages')),
            _int(r.get('Number of Incoming Messages')),
            _int(r.get('Number of Responses')),
            _int(r.get('Number of Assignments')),
            _s(r.get('Conversation Category')),
            _s(r.get('Closing Note Summary')),
        ))

    with get_connection() as conn:
        with conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO stg_conversations (
                    conversation_id, contact_id,
                    datetime_conversation_started, datetime_conversation_resolved,
                    datetime_first_response, first_assignment_timestamp,
                    last_assignment_timestamp,
                    first_response_time, resolution_time, average_response_time,
                    time_to_first_assignment,
                    first_assignment_to_first_response_time,
                    last_assignment_to_response_time,
                    first_assignment_to_close_time,
                    last_assignment_to_close_time,
                    assignee, first_assignee, last_assignee, first_response_by,
                    closed_by, closed_by_source, closed_by_team,
                    opened_by_source, opened_by_channel,
                    number_of_outgoing_messages, number_of_incoming_messages,
                    number_of_responses, number_of_assignments,
                    conversation_category, closing_note_summary,
                    updated_at
                ) VALUES %s
                ON CONFLICT (conversation_id) DO UPDATE SET
                    datetime_conversation_resolved = EXCLUDED.datetime_conversation_resolved,
                    assignee                       = EXCLUDED.assignee,
                    last_assignee                  = EXCLUDED.last_assignee,
                    closed_by                      = EXCLUDED.closed_by,
                    closed_by_source               = EXCLUDED.closed_by_source,
                    resolution_time                = EXCLUDED.resolution_time,
                    average_response_time          = EXCLUDED.average_response_time,
                    number_of_outgoing_messages    = EXCLUDED.number_of_outgoing_messages,
                    number_of_incoming_messages    = EXCLUDED.number_of_incoming_messages,
                    number_of_responses            = EXCLUDED.number_of_responses,
                    number_of_assignments          = EXCLUDED.number_of_assignments,
                    conversation_category          = EXCLUDED.conversation_category,
                    closing_note_summary           = EXCLUDED.closing_note_summary,
                    updated_at                     = NOW()
                """,
                [(*row, pd.Timestamp.now().to_pydatetime()) for row in rows],
            )
    logger.info(f"   ✅ {len(rows):,} conversations upserted into stg_conversations")


# ══════════════════════════════════════════════════════════════════════════════
# 5. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run_production_cleaning():
    os.makedirs(MSG_INTERIM_PARQUET.parent, exist_ok=True)

    logger.info("=" * 60)
    logger.info("🚀 V4 INCREMENTAL CLEANER — DB STAGING + WATERMARKS")
    logger.info("=" * 60)

    # Log current watermarks so you can see at a glance what's already loaded
    for source in ["messages", "conversations", "contacts", "ads"]:
        wm = get_watermark(source)
        if wm:
            logger.info(f"  {source:<15} → loaded up to {wm}")
        else:
            logger.info(f"  {source:<15} → no watermark (first run)")
    logger.info("-" * 60)

    ADS_DIR = Path(MSG_HISTORY_RAW).parent / "ads"

    # ── 1. Messages ───────────────────────────────────────────────────────────
    df_msg = process_messages(ADS_DIR)

    if df_msg.empty:
        logger.info("⏭️  No new messages — skipping downstream writes.")
        logger.info("=" * 60)
        return  # Nothing new — ml_inference.py doesn't need to run either

    df_msg.to_parquet(MSG_INTERIM_PARQUET, index=False)
    df_msg.to_csv(MSG_INTERIM_CSV, index=False)
    logger.info(f"✅ Messages written: {len(df_msg):,} rows → {MSG_INTERIM_PARQUET.name}")

    # ── 2. Contacts ───────────────────────────────────────────────────────────
    df_cont = process_contacts()
    df_cont.to_parquet(CONTACTS_INTERIM_PARQUET, index=False)
    df_cont.to_csv(CONTACTS_INTERIM_CSV, index=False)
    logger.info(f"✅ Contacts written: {len(df_cont):,} records")

    # ── 3. Conversations ──────────────────────────────────────────────────────
    df_conv = process_conversations(df_msg)
    df_conv.to_parquet(CONV_INTERIM_PARQUET, index=False)
    df_conv.to_csv(CONV_INTERIM_CSV, index=False)
    logger.info(f"✅ Conversations written: {len(df_conv):,} records")

    logger.info("=" * 60)
    logger.info("🎉 CLEANING COMPLETE — all staging tables updated")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_production_cleaning()