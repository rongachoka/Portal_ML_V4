from pathlib import Path


# ==========================================
# CONFIGURATION
# ==========================================
TARGET_DIR = Path(
    "G:\\My Drive\\Portal_ML\\Portal_ML_V4\\data (1)\\01_raw (1)\\"
    "Brand_Images_TopDown"
)


# ==========================================
# MAIN AUDITOR
# ==========================================
def find_empty_brand_folders():
    """Scans the target directory and lists all folders with zero images."""
    print("🔍 SCANNING FOR EMPTY FOLDERS...")
    
    if not TARGET_DIR.exists():
        print("❌ Error: Target directory does not exist.")
        return

    # Grab all subdirectories inside the main folder
    brand_folders = sorted([f for f in TARGET_DIR.iterdir() if f.is_dir()])
    empty_folders = []

    for brand_dir in brand_folders:
        # We explicitly count .jpg files to ensure we don't accidentally 
        # count hidden system files like .DS_Store or desktop.ini
        images = list(brand_dir.glob("*.jpg"))
        
        if len(images) == 0:
            empty_folders.append(brand_dir.name)

    # Print the final report
    print("\n" + "=" * 40)
    print(f"📊 SCAN COMPLETE: Found {len(empty_folders)} empty folders "
          f"out of {len(brand_folders)} total brands.")
    print("=" * 40)
    
    if empty_folders:
        print("\n📁 Empty Brands:")
        for folder in empty_folders:
            print(f"   - {folder}")
    else:
        print("\n🎉 Excellent! Every single brand folder has at least one image.")


if __name__ == "__main__":
    find_empty_brand_folders()