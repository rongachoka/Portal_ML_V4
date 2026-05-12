import sys
import time
from pathlib import Path

# ✅ IMPORT YOUR MODULES
# Ensure your python path is set correctly or run this from the root folder
try:
    from Portal_ML_V4.src.pipelines.pos_finance.etl import run_pos_etl_v3
    from Portal_ML_V4.src.pipelines.crm_chat.ml_inference import run_ml_inference
    from Portal_ML_V4.src.pipelines.crm_chat.analytics import run_analytics_pipeline
    from Portal_ML_V4.src.pipelines.attribution.link_social_to_pos_fuzzy import run_attribution_pipeline
    from Portal_ML_V4.src.pipelines.crm_chat.staff_performance_old import run_staff_analysis
    from Portal_ML_V4.src.pipelines.crm_chat.analyze_brand_performance import run_brand_analysis
    # Optional: Journey/Concern scripts
    from Portal_ML_V4.src.pipelines.extract_concern_recommendations import run_recommendation_extraction
except ImportError as e:
    print(f"❌ IMPORT ERROR: {e}")
    print("   👉 Ensure you are running this from the root directory and 'Portal_ML_V4' is a package.")
    sys.exit(1)

print(f"🐍 EXECUTION PATH: {sys.executable}")
import pydantic
print(f"📦 PYDANTIC VERSION: {pydantic.VERSION}")

def print_header(step_name):
    print("\n" + "="*60)
    print(f"🚀 STEP: {step_name}")
    print("="*60 + "\n")

def main():
    total_start = time.time()
    print("🌍 STARTING MASTER DATA PIPELINE...")

    # 1. PROCESS POS DATA (The Financial Foundation)
    print_header("1. POS ETL (Finance Data)")
    try:
        run_pos_etl_v3()
    except Exception as e:
        print(f"❌ POS ETL Failed: {e}")
        # We assume we can proceed if old data exists, but ideally stop here.

    # 2. PROCESS CHAT DATA (The Context)
    print_header("2. ML INFERENCE (Chat Processing)")
    try:
        # Note: If batch_size needs tuning, change it inside the file or pass it if modified
        run_ml_inference(batch_size=128) 
    except Exception as e:
        print(f"❌ ML Inference Failed: {e}")
        return # Critical failure

    # 3. BUILD SESSIONS (The Linkage)
    print_header("3. SESSION ANALYTICS (Enrichment)")
    try:
        run_analytics_pipeline()
    except Exception as e:
        print(f"❌ Analytics Failed: {e}")
        return # Critical failure

    # 4. ATTRIBUTION (The Revenue Proof)
    print_header("4. SALES ATTRIBUTION (Chat <-> POS Match)")
    try:
        run_attribution_pipeline()
    except Exception as e:
        print(f"❌ Attribution Failed: {e}")

    # 5. GENERATE REPORTS (The Output)
    print_header("5. GENERATING REPORTS")
    
    # A. Staff Report
    print("   👤 Generating Staff Performance Report...")
    try:
        run_staff_analysis()
    except Exception as e:
        print(f"      ⚠️ Staff Report Failed: {e}")

    # B. Brand/Product Report
    print("   🏷️  Generating Brand Performance Report...")
    try:
        run_brand_analysis()
    except Exception as e:
        print(f"      ⚠️ Brand Report Failed: {e}")

    # C. Recommendation Report (Optional)
    print("   💊 Generating Recommendation Insights...")
    try:
        run_recommendation_extraction()
    except Exception as e:
        print(f"      ⚠️ Recommendation Report Failed: {e}")

    total_time = (time.time() - total_start) / 60
    print("\n" + "="*60)
    print(f"✅ PIPELINE COMPLETE in {total_time:.1f} minutes.")
    print("="*60)

if __name__ == "__main__":
    main()