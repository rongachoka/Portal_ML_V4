"""
delete_empty_folders.py
=======================
Utility: removes empty subdirectories from the Brand_Images_TopDown folder.

Companion to check_4_empty_folder.py — after reviewing the list, run this
to permanently remove confirmed empty brand image folders before training.

Input:  G:/My Drive/Portal_ML/.../Brand_Images_TopDown/ (Google Drive path)
Output: empty folders deleted; console log of removed paths
"""

import shutil
from pathlib import Path


# ==========================================
# CONFIGURATION
# ==========================================
TARGET_DIR = Path(
    "G:\\My Drive\\Portal_ML\\Portal_ML_V4\\data (1)\\01_raw (1)\\"
    "Brand_Images_TopDown"
)


# ==========================================
# MAIN DELETION SCRIPT
# ==========================================
def delete_empty_brand_folders():
    """Scans the target directory and deletes folders with zero JPG images."""
    print("🗑️ INITIALIZING EMPTY FOLDER CLEANUP...")
    
    if not TARGET_DIR.exists():
        print("❌ Error: Target directory does not exist.")
        return

    # Grab all subdirectories inside the main folder
    brand_folders = sorted([f for f in TARGET_DIR.iterdir() if f.is_dir()])
    deleted_count = 0

    for brand_dir in brand_folders:
        # Check specifically for JPGs
        images = list(brand_dir.glob("*.jpg"))
        
        if len(images) == 0:
            print(f"   🧹 Deleting empty folder: {brand_dir.name}")
            try:
                # Force delete the directory and any hidden system files inside
                shutil.rmtree(brand_dir)
                deleted_count += 1
            except Exception as e:
                print(f"      ❌ Failed to delete {brand_dir.name}: {e}")

    # Print the final report
    print("\n" + "=" * 40)
    print(f"✅ CLEANUP COMPLETE: Successfully deleted {deleted_count} folders.")
    print("=" * 40)


if __name__ == "__main__":
    delete_empty_brand_folders()