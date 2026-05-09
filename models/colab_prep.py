# import pandas as pd
# import os
# import shutil
# from pathlib import Path
# from tqdm import tqdm

# # --- SETTINGS ---
# # Adjust this if your folder structure is different
# BASE_DIR = Path(__file__).resolve().parent.parent.parent
# SOURCE_IMG_DIR = BASE_DIR / "data" / "training_images"
# CSV_FILE = SOURCE_IMG_DIR / "image_labels_manifest.csv"

# # Output settings
# OUTPUT_DIR = BASE_DIR / "data" / "colab_dataset"
# ZIP_NAME = BASE_DIR / "data" / "portal_dataset" # This creates portal_dataset.zip

# def prepare_dataset():
#     print("📦 PREPARING DATASET FOR COLAB...")
    
#     if not os.path.exists(CSV_FILE):
#         print(f"❌ Error: Manifest not found at {CSV_FILE}")
#         return

#     # 1. Load Labels
#     df = pd.read_csv(CSV_FILE, dtype={'suspected_label': str})
    
#     # Filter for valid labels (non-blank, non-null)
#     df_labeled = df[df['suspected_label'].notna() & (df['suspected_label'].str.strip() != "")].copy()
    
#     # Clean labels: lowercase, remove spaces
#     df_labeled['label'] = df_labeled['suspected_label'].str.lower().str.strip()
    
#     print(f"   📊 Found {len(df_labeled)} labeled images.")
#     print("-" * 30)
#     print(df_labeled['label'].value_counts())
#     print("-" * 30)

#     # 2. Reset Output Directory
#     if os.path.exists(OUTPUT_DIR):
#         shutil.rmtree(OUTPUT_DIR)
#     os.makedirs(OUTPUT_DIR)

#     # 3. Copy Images to Folders
#     copied_count = 0
#     missing_count = 0
    
#     for _, row in tqdm(df_labeled.iterrows(), total=len(df_labeled), desc="Sorting Images"):
#         filename = row['filename']
#         label = row['label']
        
#         # Source (Your big downloads folder)
#         src = SOURCE_IMG_DIR / filename
        
#         # Destination (The specific label folder)
#         # e.g. data/colab_dataset/product/file.jpg
#         dest_folder = OUTPUT_DIR / label
#         os.makedirs(dest_folder, exist_ok=True)
        
#         if os.path.exists(src):
#             shutil.copy2(src, dest_folder / filename)
#             copied_count += 1
#         else:
#             missing_count += 1

#     # 4. Zip the Result
#     print(f"\n   🗜️  Compressing into 'portal_dataset.zip'...")
#     # shutil.make_archive creates the zip file
#     shutil.make_archive(ZIP_NAME, 'zip', OUTPUT_DIR)
    
#     final_zip_path = str(ZIP_NAME) + ".zip"
    
#     print("\n✅ DATASET READY.")
#     print(f"   - Successfully bundled: {copied_count} images")
#     print(f"   - Missing files: {missing_count}")
#     print(f"   - ZIP Location: {final_zip_path}")
#     print("\n👉 NEXT STEP: Upload this .zip file to your Google Drive.")

# if __name__ == "__main__":
#     prepare_dataset()

import pandas as pd
from pathlib import Path

# Update path to your file
CSV_FILE = Path(r"C:\Users\Portal Pharmacy\Documents\Portal ML Analys\Ron's Work (2)\Portal_ML\Portal_ML_V4\data\training_images\image_labels_manifest.csv")

df = pd.read_csv(CSV_FILE)

# Filter for the ones you labeled
df_labeled = df[df['suspected_label'].notna() & (df['suspected_label'].str.strip() != "")]

print(f"📋 Total Labeled Rows (The 'Ghost' Count): {len(df_labeled)}")
print(f"🖼️ Unique Images (The 'Real' Count):       {df_labeled['filename'].nunique()}")