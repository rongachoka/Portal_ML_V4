import pandas as pd
import os
import requests
import time
from tqdm.auto import tqdm
from pathlib import Path

# SETTINGS
# Update these paths to match your actual project structure
BASE_DIR = Path(__file__).resolve().parent.parent.parent # Adjust based on where you save this script
INPUT_FILE = BASE_DIR / "data" / "03_processed" / "fact_sessions_enriched.csv"
OUTPUT_DIR = BASE_DIR / "data" / "training_images"
MANIFEST_FILE = OUTPUT_DIR / "image_labels_manifest.csv"

def download_images():
    print("📸 STARTING IMAGE DOWNLOAD PIPELINE")
    
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: Input file not found at {INPUT_FILE}")
        return

    # Create output folder
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load Data
    df = pd.read_csv(INPUT_FILE)
    
    # Filter: Only rows with image attachments
    # We look for rows where the column is not empty and not "No Attachment"
    df_images = df[df['image_attachments'].notna() & (df['image_attachments'] != "")].copy()
    
    print(f"🔍 Found {len(df_images)} sessions with images.")
    
    manifest_rows = []
    
    # Iterate through sessions
    for _, row in tqdm(df_images.iterrows(), total=len(df_images), desc="Downloading"):
        session_id = str(row.get('session_id', row.get('Contact ID', 'unknown')))
        image_urls = str(row['image_attachments']).split(" | ")
        
        for i, url in enumerate(image_urls):
            url = url.strip()
            if not url.startswith("http"): continue
            
            # Create a clean filename: sessionID_index.jpg
            ext = url.split('.')[-1].split('?')[0] # Grab extension (jpg, png)
            if len(ext) > 4: ext = "jpg" # Fallback if weird extension
            
            filename = f"{session_id}_{i}.{ext}"
            filepath = OUTPUT_DIR / filename
            
            # Skip if already exists (Resume capability)
            if os.path.exists(filepath):
                status = "Exists"
            else:
                try:
                    # Download
                    response = requests.get(url, timeout=10)
                    if response.status_code == 200:
                        with open(filepath, 'wb') as f:
                            f.write(response.content)
                        status = "Downloaded"
                    else:
                        status = "Error_404"
                except Exception as e:
                    status = f"Error: {str(e)}"
            
            # Add to Manifest for Labeling
            # We include the text context so you know what the image is about!
            manifest_rows.append({
                "filename": filename,
                "status": status,
                "original_url": url,
                "chat_context": str(row.get('full_context', ''))[:300], # First 300 chars of text
                "suspected_label": "" # Empty column for you to fill in later
            })
            
            # Be nice to the server
            time.sleep(0.1)

    # Save Manifest
    pd.DataFrame(manifest_rows).to_csv(MANIFEST_FILE, index=False)
    print(f"\n✅ DOWNLOAD COMPLETE.")
    print(f"📂 Images saved to: {OUTPUT_DIR}")
    print(f"📝 Labeling Manifest saved to: {MANIFEST_FILE}")
    print("   -> Open this CSV to start tagging your images for training!")

if __name__ == "__main__":
    download_images()