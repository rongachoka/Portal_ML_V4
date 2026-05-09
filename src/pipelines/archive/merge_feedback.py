import os
import shutil
from pathlib import Path

# --- SETTINGS ---
BASE_DIR = Path(r"C:\\Users\\Portal Pharmacy\\Documents\\Portal ML Analys\\Ron's Work (2)\\Portal_ML\\Portal_ML_V4")

# Where your corrected 500 images are currently sitting
VERIFIED_DIR = BASE_DIR / "data" / "final_sorted"

# The main training tank
TRAINING_DIR = BASE_DIR / "data" / "sorted"

def merge_data():
    print(f"🚜 Harvesting data from {VERIFIED_DIR}...")
    
    if not os.path.exists(VERIFIED_DIR):
        print("❌ Final sorted folder not found.")
        return

    moved_count = 0
    
    # Iterate through every folder in final_sorted
    for folder_name in os.listdir(VERIFIED_DIR):
        source_folder = VERIFIED_DIR / folder_name
        
        # Skip if it's not a folder
        if not os.path.isdir(source_folder): continue

        # --- DETERMINE DESTINATION ---
        dest_category = None
        
        # 1. JUNK -> data/sorted/junk
        if folder_name.upper() == "JUNK":
            dest_category = "junk"
            
        # 2. CONDITIONS -> data/sorted/condition
        elif folder_name.upper() == "CONDITIONS":
            dest_category = "condition"
            
        # 3. BRANDS/PRODUCTS -> data/sorted/product
        # (Includes BRAND_CERAVE, PRODUCT_UNKNOWN, PRODUCT_UNCERTAIN, etc.)
        elif folder_name.upper().startswith("BRAND_") or folder_name.upper().startswith("PRODUCT_"):
            dest_category = "product"

        # If we found a valid destination, move the files
        if dest_category:
            dest_path = TRAINING_DIR / dest_category
            os.makedirs(dest_path, exist_ok=True)
            
            files = os.listdir(source_folder)
            for f in files:
                src_file = source_folder / f
                dst_file = dest_path / f
                
                # Handle duplicate filenames (rename if exists)
                if os.path.exists(dst_file):
                    name, ext = os.path.splitext(f)
                    import uuid
                    new_name = f"{name}_{uuid.uuid4().hex[:4]}{ext}"
                    dst_file = dest_path / new_name
                
                shutil.move(src_file, dst_file)
                moved_count += 1
                
    print("-" * 30)
    print(f"✅ Harvest Complete!")
    print(f"   🌱 Added {moved_count} new images to your training set.")
    print("   🗑️  The 'final_sorted' folders are now empty and ready for the next batch.")

if __name__ == "__main__":
    merge_data()