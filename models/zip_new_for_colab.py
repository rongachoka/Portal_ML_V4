import shutil
import os
from pathlib import Path

# --- SETTINGS ---
BASE_DIR = Path(r"C:\\Users\\Portal Pharmacy\\Documents\\Portal ML Analys\\Ron's Work (2)\\Portal_ML\\Portal_ML_V4")
SOURCE_DIR = BASE_DIR / "data" / "sorted"
OUTPUT_NAME = BASE_DIR / "data" / "portal_dataset_v5" # Will become .zip

def zip_dataset():
    print(f"📦 PACKING DATASET V5...")
    
    if not os.path.exists(SOURCE_DIR):
        print(f"❌ Error: Could not find {SOURCE_DIR}")
        return

    # Count files
    total_files = 0
    for root, dirs, files in os.walk(SOURCE_DIR):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                total_files += 1

    print(f"   📊 Found {total_files} verified images.")
    
    # Create Zip
    print(f"   🗜️  Zipping...")
    shutil.make_archive(OUTPUT_NAME, 'zip', SOURCE_DIR)
    
    print(f"\n✅ READY: {OUTPUT_NAME}.zip")
    print("   👉 Upload this file to Google Drive to retrain!")

if __name__ == "__main__":
    zip_dataset()