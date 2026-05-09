import sys
import pydantic
import os
import gc
import ctypes
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from Portal_ML_V4.src.pipelines.crm_chat.cleaning_local_copy import run_production_cleaning
from Portal_ML_V4.src.pipelines.crm_chat.ml_inference_copy import run_ml_inference
from Portal_ML_V4.src.pipelines.crm_chat.analytics_copy import run_analytics_pipeline
from Portal_ML_V4.src.pipelines.crm_chat.staff_performance import run_staff_analysis
from Portal_ML_V4.src.pipelines.crm_chat.analyze_concern_journey import run_journey_analysis
from Portal_ML_V4.src.pipelines.crm_chat.concern_insights import run_daily_analysis

from Portal_ML_V4.src.pipelines.extract_concern_recommendations import run_recommendation_extraction

from Portal_ML_V4.src.pipelines.attribution.enrich_attribution_products import run_smart_enrichment
from Portal_ML_V4.src.pipelines.attribution.jan25_26_link_social_to_pos_new_copy import run_attribution_v7
from Portal_ML_V4.src.pipelines.attribution.social_sales_direct import run_social_sales_direct
from Portal_ML_V4.src.pipelines.attribution.social_sales_etl import run_social_sales_etl
from Portal_ML_V4.src.pipelines.attribution.ad_performance import run_ad_performance


from Portal_ML_V4.src.pipelines.pos_finance.etl_local import run_pos_etl_local
# from Portal_ML_V4.src.pipelines.pos_finance.etl_full_history import run_pos_etl_full_history
from Portal_ML_V4.src.pipelines.pos_finance.load_to_postgres import run_pos_loader

from Portal_ML_V4.src.pipelines.website.website_orders import run_website_orders_etl
from Portal_ML_V4.website_orders.website_sales import run_website_sales_etl

from Portal_ML_V4.run_client_export import run_client_export


from Portal_ML_V4.sales_reports_all.staff_performance import run_staff_performance_pos


print(f"EXECUTION PATH: {sys.executable}\n")


def main():
    total_start = time.time()
    print("🚀 STARTING PORTAL PHARMACY V3 PRODUCTION PIPELINE")
    print("=" * 65)

    try:
        # STAGE 1: DATA STANDARDIZATION & HEALING
        run_production_cleaning()

        # STAGE 2: AI ML INFERENCE
        print("Running ML Inference\n")
        run_ml_inference(batch_size=128)

        gc.collect()
        ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1, -1)

        # STAGE 3: POS ETL — must run before attribution so phone numbers are available
        print("Running ETL Cleaner\n")
        run_pos_etl_local()
        # print("Running ETL Full History\n")
        # run_pos_etl_full_history()

        # STAGE 4: ANALYTICS — first pass
        # Produces fact_sessions_enriched.csv so attribution has sessions to match against
        # ordered_via will be empty at this point — that's expected
        print("Running Analytics Pipeline (Pass 1 — for attribution input)\n")
        run_analytics_pipeline()

        gc.collect()
        ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1, -1)

        # STAGE 5: ATTRIBUTION — reads fact_sessions_enriched, writes ordered_via back
        print("Running Match Maker - Finding the right sale\n")
        run_attribution_v7()

        # STAGE 6: ENRICH — labels products on attributed sales
        print("Running the Librarian - Assigning Accurate Brands\n")
        run_smart_enrichment()

        # STAGE 7: ANALYTICS — second pass
        # Now reads the ordered_via values attribution stamped back
        # This is the final version of all fact tables
        print("Running Analytics Pipeline (Pass 2 — with ordered_via attribution)\n")
        run_analytics_pipeline()

        # STAGE 7.5: DAILY CONCERN ANALYSIS
        print("Running Daily Concern Analysis\n")
        run_daily_analysis()

        # STAGE 8: STAFF PERFORMANCE
        print("Running Staff Performance\n")
        run_staff_analysis()

        # STAGE 9: CONCERN JOURNEY ANALYSIS
        print("Running Concern Journey\n")
        run_journey_analysis()

        # STAGE 10: STAFF RECOMMENDATIONS
        print("Running Staff Recommendations\n")
        run_recommendation_extraction()

        # STAGE 11: WEBSITE ORDERS ETL
        print("Running Website Orders ETL\n")
        run_website_orders_etl()

        # POS Sales
        print("Running Social Sales Direct (dashboard source)\n")
        run_social_sales_direct()

        print("Running Social Sales ETL Gap Audit (separate output)\n")
        run_social_sales_etl()


        # STAGE 11.2.5: STAFF PERFORMANCE
        print("Running Staff Performance - POS\n")
        run_staff_performance_pos()

        # STAGE 11.5 AD PERFORMANCE
        print("Running Ad Performance Analysis\n")
        run_ad_performance()

        # STAGE 12: PUSH POS DATA TO POSTGRESQL
        print("Loading POS data to PostgreSQL\n")
        run_pos_loader()

        # # STAGE 13: Website Sales
        print("Running Website Sales\n")
        run_website_sales_etl()

        
        
        # print("Exporting Client List\n")
        # run_client_export()

        

        elapsed = time.time() - total_start
        print("\n" + "="*65)
        print("🎉 SUCCESS: Full Data Pipeline Complete.")
        print(f"✅ PIPELINE COMPLETE in {elapsed:.2f} seconds.")
        print("📍 Audit files ready in: data/03_processed/")
        print("="*65)

    except Exception as e:
        print(f"\n❌ PIPELINE FAILED: {str(e)}")
        raise e
    

if __name__ == "__main__":
    main()
