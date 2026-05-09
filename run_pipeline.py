"""
run_pipeline.py
===============
Portal ML V4 — Master pipeline orchestrator.

Stage order (preserved from V3, incremental watermarks added in V4):
    1.  Cleaning          — incremental, loads new data into DB staging tables
    2.  ML Inference      — sessionizes + AI tags new sessions only
    3.  POS ETL           — daily sales data (always runs, has its own dedup)
    4.  Analytics Pass 1  — produces fact_sessions_enriched for attribution input
                            ordered_via is empty at this point — expected
    5.  Attribution       — matches social sessions to POS transactions,
                            stamps ordered_via back onto fact_sessions_enriched
    6.  Enrichment        — labels products on attributed sales
    7.  Analytics Pass 2  — re-runs with ordered_via populated, produces final
                            versions of all fact tables
    8.  Staff Performance
    9.  Concern Journey
    10. Recommendations
    11. Website Orders ETL

Commented-out (run manually when needed):
    - run_pos_loader       — push POS to PostgreSQL
    - run_client_export    — client list export
    - load_kb_to_postgres  — knowledge base load
    - run_build_product_map

Early-exit logic:
    If cleaning finds no new messages, ML + both Analytics passes are skipped.
    POS ETL, attribution, staff, journey, and recommendations still run —
    they have their own dedup and are not gated on new chat data.

Usage:
    python run_pipeline.py                  # normal run
    python run_pipeline.py --force          # ignore watermarks, run everything
    python run_pipeline.py --step cleaning  # single step
    python run_pipeline.py --step ml
    python run_pipeline.py --step analytics
    python run_pipeline.py --step pos
    python run_pipeline.py --step attribution
"""

import argparse
import ctypes
import gc
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Project root on path (required for Portal_ML_V4 imports on Windows)
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print(f"EXECUTION PATH: {sys.executable}\n")

from Portal_ML_V4.src.config.settings import MSG_INTERIM_PARQUET

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)


# Memory helpers

def _flush_memory():
    """
    Force Python GC then tell Windows to release freed pages back to the OS.
    Without the ctypes call, Windows keeps pages in the working set even
    after gc.collect() — causing the apparent memory leak between stages.
    """
    gc.collect()
    try:
        ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1, -1)
    except Exception:
        pass  # Non-Windows — silently skip


# New-data check

def _has_new_data() -> bool:
    """
    Returns True if cleaning.py produced output for downstream steps.
    Empty or missing MSG_INTERIM_PARQUET means nothing new to process.
    """
    path = Path(MSG_INTERIM_PARQUET)
    if not path.exists():
        return False
    try:
        import pandas as pd
        return not pd.read_parquet(path).empty
    except Exception:
        return False


# Step runner

def _run_step(name: str, fn, flush_after: bool = False) -> bool:
    """Run one pipeline step. Returns True on success, False on failure."""
    logger.info("-" * 55)
    logger.info(f"Running: {name}")
    t0 = time.time()
    try:
        fn()
        elapsed = time.time() - t0
        logger.info(f"Done: {name} ({elapsed:.1f}s)")
        if flush_after:
            _flush_memory()
            logger.info("   Memory flushed")
        return True
    except Exception as exc:
        elapsed = time.time() - t0
        logger.error(f"FAILED: {name} ({elapsed:.1f}s) — {exc}", exc_info=True)
        return False


# Lazy imports — only loaded when that step actually runs

def _run_cleaning():
    from Portal_ML_V4.src.pipelines.crm_chat.cleaning import run_production_cleaning
    run_production_cleaning()

def _run_ml():
    from Portal_ML_V4.src.pipelines.crm_chat.ml_inference import run_ml_inference
    run_ml_inference(batch_size=128)

def _run_pos_etl():
    from Portal_ML_V4.src.pipelines.pos_finance.etl import run_pos_etl_v3
    run_pos_etl_v3()

def _run_analytics():
    from Portal_ML_V4.src.pipelines.crm_chat.analytics import run_analytics_pipeline
    run_analytics_pipeline()

def _run_attribution():
    from Portal_ML_V4.src.pipelines.attribution.jan25_26_link_social_to_pos_new import run_attribution_v7
    run_attribution_v7()

def _run_enrichment():
    from Portal_ML_V4.src.pipelines.attribution.enrich_attribution_products import run_smart_enrichment
    run_smart_enrichment()

def _run_staff():
    from Portal_ML_V4.src.pipelines.crm_chat.staff_performance import run_staff_analysis
    run_staff_analysis()

def _run_journey():
    from Portal_ML_V4.src.pipelines.crm_chat.analyze_concern_journey import run_journey_analysis
    run_journey_analysis()

def _run_recommendations():
    from Portal_ML_V4.src.pipelines.extract_concern_recommendations import run_recommendation_extraction
    run_recommendation_extraction()

def _run_website():
    from Portal_ML_V4.src.pipelines.website.website_orders import run_website_orders_etl
    run_website_orders_etl()


# Main

def main():
    parser = argparse.ArgumentParser(description="Portal ML V4 Pipeline Orchestrator")
    parser.add_argument(
        "--force", action="store_true",
        help="Ignore watermarks — run all steps even if no new chat data",
    )
    parser.add_argument(
        "--step",
        choices=["cleaning", "ml", "pos", "analytics", "attribution",
                 "enrichment", "staff", "journey", "recommendations", "website"],
        default=None,
        help="Run a single step only (for debugging)",
    )
    args = parser.parse_args()

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Tee output to log file
    log_path = LOG_DIR / f"pipeline_{run_ts}.log"
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logging.getLogger().addHandler(file_handler)

    logger.info("=" * 55)
    logger.info("PORTAL PHARMACY V4 — PRODUCTION PIPELINE")
    logger.info(f"Run ID : {run_ts}")
    logger.info(f"Mode   : {'single — ' + args.step if args.step else 'full pipeline'}")
    logger.info(f"Force  : {args.force}")
    logger.info("=" * 55)

    total_start = time.time()

    # Single-step mode
    step_map = {
        "cleaning":        _run_cleaning,
        "ml":              _run_ml,
        "pos":             _run_pos_etl,
        "analytics":       _run_analytics,
        "attribution":     _run_attribution,
        "enrichment":      _run_enrichment,
        "staff":           _run_staff,
        "journey":         _run_journey,
        "recommendations": _run_recommendations,
        "website":         _run_website,
    }
    if args.step:
        _run_step(args.step, step_map[args.step])
        logger.info(f"Total: {time.time() - total_start:.1f}s")
        return

    # Full pipeline
    try:
        # Stage 1: Cleaning
        ok = _run_step("Stage 1 — Cleaning", _run_cleaning)
        if not ok:
            raise RuntimeError("Cleaning failed — pipeline aborted")

        # Chat data gate
        chat_has_new = args.force or _has_new_data()
        if not chat_has_new:
            logger.info("No new chat data — skipping ML + Analytics passes")

        # Stage 2: ML Inference
        if chat_has_new:
            ok = _run_step("Stage 2 — ML Inference", _run_ml, flush_after=True)
            if not ok:
                raise RuntimeError("ML Inference failed — pipeline aborted")

        # Stage 3: POS ETL — reads from SharePoint downloads folder, full history
        _run_step("Stage 3 — POS ETL", _run_pos_etl)

        # Stage 4: Analytics Pass 1 (pre-attribution, ordered_via empty — expected)
        if chat_has_new:
            ok = _run_step("Stage 4 — Analytics Pass 1 (pre-attribution)", _run_analytics, flush_after=True)
            if not ok:
                raise RuntimeError("Analytics Pass 1 failed — pipeline aborted")

        # Stage 5: Attribution — matches sessions to POS, stamps ordered_via back
        _run_step("Stage 5 — Attribution (social to POS match)", _run_attribution)

        # Stage 6: Product Enrichment
        _run_step("Stage 6 — Product Enrichment (brand labelling)", _run_enrichment)

        # Stage 7: Analytics Pass 2 — final, with ordered_via populated
        if chat_has_new:
            ok = _run_step("Stage 7 — Analytics Pass 2 (final, with attribution)", _run_analytics, flush_after=True)
            if not ok:
                raise RuntimeError("Analytics Pass 2 failed — pipeline aborted")

        # Stage 8: Staff Performance
        _run_step("Stage 8 — Staff Performance", _run_staff)

        # Stage 9: Concern Journey
        _run_step("Stage 9 — Concern Journey Analysis", _run_journey)

        # Stage 10: Recommendations
        _run_step("Stage 10 — Staff Recommendations", _run_recommendations)

        # Stage 11: Website Orders
        _run_step("Stage 11 — Website Orders ETL", _run_website)

        # Commented stages — uncomment and run manually when needed
        # from Portal_ML_V4.src.pipelines.pos_finance.load_to_postgres import run_pos_loader
        # _run_step("POS to PostgreSQL loader", run_pos_loader)
        #
        # from Portal_ML_V4.run_client_export import run_client_export
        # _run_step("Client list export", run_client_export)
        #
        # from Portal_ML_V4.sql.kb_to_server import load_kb_to_postgres
        # _run_step("Knowledge Base to PostgreSQL", load_kb_to_postgres)
        #
        # from Portal_ML_V4.sql.product_map_build import run_build_product_map
        # _run_step("Product map build", run_build_product_map)

        elapsed = time.time() - total_start
        logger.info("=" * 55)
        logger.info("SUCCESS — Full Pipeline Complete")
        logger.info(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
        logger.info("Audit files: data/03_processed/")
        logger.info(f"Log: {log_path}")
        logger.info("=" * 55)

    except Exception as e:
        logger.error(f"PIPELINE FAILED: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()