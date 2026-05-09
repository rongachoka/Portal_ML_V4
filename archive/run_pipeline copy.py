import sys
import pydantic
import os
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from Portal_ML_V4.src.pipelines.crm_chat.cleaning import run_production_cleaning
from Portal_ML_V4.src.pipelines.crm_chat.ml_inference import run_ml_inference
from Portal_ML_V4.src.pipelines.crm_chat.analytics import run_analytics_pipeline
from Portal_ML_V4.src.pipelines.crm_chat.staff_performance import run_staff_analysis
from Portal_ML_V4.src.pipelines.crm_chat.analyze_concern_journey import run_journey_analysis
from Portal_ML_V4.src.pipelines.extract_concern_recommendations import run_recommendation_extraction
from Portal_ML_V4.src.pipelines.attribution.enrich_attribution_products import run_smart_enrichment
from Portal_ML_V4.src.pipelines.attribution.jan25_26_link_social_to_pos_new import run_attribution_v7
from Portal_ML_V4.src.pipelines.pos_finance.etl import run_pos_etl_v3
from Portal_ML_V4.src.pipelines.pos_finance.etl_full_history import run_pos_etl_full_history
from Portal_ML_V4.src.pipelines.pos_finance.load_to_postgres import run_pos_loader
from Portal_ML_V4.src.pipelines.website.website_orders import run_website_orders_etl
from Portal_ML_V4.run_client_export import run_client_export



print(f"EXECUTION PATH: {sys.executable}\n")


def main():
    total_start = time.time()
    print("🚀 STARTING PORTAL PHARMACY V3 PRODUCTION PIPELINE")
    print("=" * 65)
    
    try:
        # STAGE 1: DATA STANDARDIZATION & HEALING
        # Cleans raw Respond.io CSVs and handles missing channel IDs
        run_production_cleaning()

        print("")
        print("Running ML Inference")
        
        # STAGE 2: AI ML INFERENCE
        # Processes sessions, runs heuristics, and applies AI labels (uses CUDA if available)
        run_ml_inference(batch_size=128)

        print("")
        print("Running Analytics Pipeline")
        
        # STAGE 3: ANALYTICS & LOOKER STUDIO PREP
        # Applies Max-LTV logic, fixes encoding, and generates fact tables
        run_analytics_pipeline()


        # STAGE 4: STAFF PERFORMANCE
        print("Running Staff Performance\n")
        run_staff_analysis()

        # STAGE 5: CONCERN JOUNREY ANALYSIS
        print("Running Concern Journey\n")
        run_journey_analysis()

        # sTAGE 6: STAFF RECOMMENDATIONS
        print("Running Staff Recommendations\n")
        run_recommendation_extraction()
        
        # STAGE 7 ETL - THE CLEANER
        print("Running ETL Cleaner \n")
        run_pos_etl_v3()

        # STAGE 7b: ETL FULL HISTORY - THE FULLER CLEANER
        print("Running ETL Full History \n")
        run_pos_etl_full_history()

        # sTAGE 8 LINK SOCIAL TO POS - MATCH MAKER
        print("RUnning Match Maker - FInding the right sale\n")
        run_attribution_v7()

        # STAGE 9 ENRICH - LABELLING
        print("Running the Librarian - Assigning Accurate Brands\n")
        run_smart_enrichment()

        # STAGE 10: WEBSITE ORDERS ETL
        print("Running Website Orders ETL\n")
        run_website_orders_etl()

        # STAGE 11: PUSH POS DATA TO POSTGRESQL
        print("Loading POS data to PostgreSQL\n")
        run_pos_loader()

        # STAGE 12: CLIENT LIST EXPORT
        print("Exporting Client List\n")
        run_client_export()

# -------------------------------------------------------------------------
# -------------------------------------------------------------------------

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