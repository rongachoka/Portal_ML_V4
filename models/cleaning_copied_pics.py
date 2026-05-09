import os
import shutil
from pathlib import Path

# --- SETTINGS ---
BASE_DIR = Path(r"C:\\Users\\Portal Pharmacy\\Documents\\Portal ML Analys\\Ron's Work (2)\\Portal_ML\\Portal_ML_V4")

# Where your big pile of 22k images is
SOURCE_DIR = BASE_DIR / "data" / "training_images"

# Where your finished work is
SORTED_DIR = BASE_DIR / "data" / "sorted"

# Where we will move the duplicates to get them out of the way
ARCHIVE_DIR = BASE_DIR / "data" / "processed_archive"

def clean_source_folder():
    print("🧹 STARTING CLEANUP...")
    
    if not os.path.exists(SORTED_DIR):
        print("❌ No sorted data found yet.")
        return

    # 1. Build a list of all filenames you have already sorted
    print("   📝 Listing sorted files...")
    sorted_filenames = set()
    for root, dirs, files in os.walk(SORTED_DIR):
        for f in files:
            sorted_filenames.add(f)
    
    print(f"   found {len(sorted_filenames)} images already sorted.")

    # 2. Check the source folder and move duplicates
    if not os.path.exists(ARCHIVE_DIR):
        os.makedirs(ARCHIVE_DIR)

    print("   🚚 Moving duplicates to 'processed_archive'...")
    
    moved_count = 0
    
    # Iterate through the big source folder
    # We use os.listdir to avoid deep recursion if not needed
    for filename in os.listdir(SOURCE_DIR):
        if filename in sorted_filenames:
            src_path = SOURCE_DIR / filename
            dest_path = ARCHIVE_DIR / filename
            
            try:
                shutil.move(src_path, dest_path)
                moved_count += 1
            except Exception as e:
                print(f"Error moving {filename}: {e}")

    print("-" * 30)
    print(f"✅ CLEANUP COMPLETE.")
    print(f"   - Moved {moved_count} images to: {ARCHIVE_DIR}")
    print(f"   - Your 'training_images' folder is now 100% fresh.")

if __name__ == "__main__":
    clean_source_folder()