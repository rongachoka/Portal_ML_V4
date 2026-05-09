"""
settings.py
===========
Central configuration: all file paths, ML thresholds, and database credentials.

Reads .env from the project root for DB and API secrets. Every other module
imports paths and constants from here — nothing should hardcode a file path.

Key exports:
    RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR — data layer roots
    MSG_HISTORY_RAW, CONV_HISTORY_RAW, CONTACTS_HISTORY_RAW — Respond.io source CSVs
    MSG_INTERIM_PARQUET, FINAL_TAGGED_DATA — inter-stage data files
    KB_PATH          — Knowledge Base CSV path
    META_ADS_DIR     — Latest Meta Ads CSV (auto-selected by mtime)
    DB_CONNECTION_STRING — PostgreSQL SQLAlchemy URL (built from .env)
    SESSION_GAP_HOURS, CATEGORY_CONFIDENCE_THRESHOLD — ML tuning knobs
    DRIVE_ID         — SharePoint drive ID for the downloader
"""

import os
import glob
from pathlib import Path
from dotenv import load_dotenv
from urllib.parse import quote_plus

# --- BASE DIRECTORY ANCHOR ---
# This locates the Portal_ML_V4 root folder
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# --- INPUT DATA PATHS (01_RAW) ---
RAW_DATA_DIR = BASE_DIR / "data" / "01_raw"
MSG_HISTORY_RAW = RAW_DATA_DIR / "Respond IO Messages History.csv"
CONV_HISTORY_RAW = RAW_DATA_DIR / "Respond IO Conversations History.csv"
CONTACTS_HISTORY_RAW = RAW_DATA_DIR / "Respond IO Contacts History.csv"

# --- INTERIM DATA PATHS (02_INTERIM) ---
INTERIM_DATA_DIR = BASE_DIR / "data" / "02_interim"
CLEANED_DATA_DIR = INTERIM_DATA_DIR  # <--- NEW LINE: This fixes the ImportError

MSG_INTERIM_PARQUET = INTERIM_DATA_DIR / "cleaned_messages.parquet"
MSG_INTERIM_CSV = INTERIM_DATA_DIR / "cleaned_messages.csv"
CONV_INTERIM_PARQUET = INTERIM_DATA_DIR / "cleaned_conversations.parquet"
CONV_INTERIM_CSV = INTERIM_DATA_DIR / "cleaned_conversations.csv"
CONTACTS_INTERIM_PARQUET = INTERIM_DATA_DIR / "cleaned_contacts.parquet"
CONTACTS_INTERIM_CSV = INTERIM_DATA_DIR / "cleaned_contacts.csv"

# --- PROCESSED DATA PATHS (03_PROCESSED) ---
PROCESSED_DATA_DIR = BASE_DIR / "data" / "03_processed"
FINAL_TAGGED_DATA = PROCESSED_DATA_DIR / "final_tagged_sessions.parquet"
FINAL_REPORT_CSV = PROCESSED_DATA_DIR / "final_report.csv"
POWERBI_CACHE_DIR = PROCESSED_DATA_DIR / "power_bi_cache"

# --- META ADS DATA PATHS ---
# META_ADS_DIR = RAW_DATA_DIR / "meta_ads" / "Caroline-Mwangi-Ads-Apr-5-2023-May-5-2026.csv"
_meta_files = sorted(glob.glob(str(RAW_DATA_DIR / "meta_ads" / "Caroline-Mwangi-Ads-*.csv")))
META_ADS_DIR = Path(_meta_files[-1]) if _meta_files else None

# --- KB ---
KB_PATH = RAW_DATA_DIR / "Final_Knowledge_Base_PowerBI_New.csv"


# --- ML & ANALYTICS SETTINGS ---
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CATEGORY_CONFIDENCE_THRESHOLD = 0.5
SESSION_GAP_HOURS = 96

# --- LTV & AUDIT THRESHOLDS (From Version 2) ---
HIGH_VALUE_THRESHOLD = 10000       # Example value, adjust as needed
LOYAL_PAYMENTS_THRESHOLD = 3       # Number of payments to be 'Loyal'
HIGH_LTV_SUM_THRESHOLD = 30000     # Total spend for 'High-LTV'
LOOKBACK_WINDOW_DAYS = 30          # Window for ad attribution

# --- USD KES EXCHANGE RATE ---
USD_KES_EXCHANGE_RATE = 130 


# --------------------------------------------------------- #
# ==========================================
# 🔐 DATABASE CONNECTION SETTINGS
# ==========================================

# Load the hidden .env file
load_dotenv()

PROCESSED_DATA_DIR = BASE_DIR / "data" / "03_processed"

DB_USER = os.getenv("DB_USER")
raw_password = os.getenv("DB_PASSWORD")
DB_PASSWORD = quote_plus(raw_password) if raw_password else "" # To safely encode special characteres in the password
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "portal_dw")

# This safely builds: postgresql://postgres:password@192.168.1.X:5432/portal_dw
# DB_CONNECTION_STRING = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
DB_CONNECTION_STRING = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}?sslmode=require"


# ==========================================
# 🔐 SHAREPOINT CONNECTION SETTINGS
# ==========================================

DRIVE_ID = "b!whL3rPzNh0-7qRe5yrHoftRvAUJj1gFFvAMiiq_bJDX64liSkv0CSZtdTu6bqccj"