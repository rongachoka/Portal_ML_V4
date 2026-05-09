import pandas as pd
import os
import requests
from tqdm.auto import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- CONFIGURATION ---
BASE_DIR = Path(__file__).resolve().parent.parent.parent
INPUT_FILE = BASE_DIR / "data" / "03_processed" / "fact_sessions_enriched.csv"
OUTPUT_DIR = BASE_DIR / "data" / "training_images"
MANIFEST_FILE = OUTPUT_DIR / "image_labels_manifest.csv"

# [SPEED SETTINGS]
MAX_WORKERS = 20  # Number of parallel downloads (Try 10-50 depending on internet speed)
TIMEOUT_SECONDS = 5

def download_single_image(args):
    """
    Worker function for parallel execution.
    args: (session_id, index, url, chat_context, session_obj)
    """
    session_id, idx, url, context, session = args
    
    # Generate Filename
    ext = url.split('.')[-1].split('?')[0]
    if len(ext) > 4 or not ext: ext = "jpg"
    filename = f"{session_id}_{idx}.{ext}"
    filepath = OUTPUT_DIR / filename
    
    result = {
        "filename": filename,
        "original_url": url,
        "chat_context": str(context)[:300],
        "status": "Unknown",
        "suspected_label": ""
    }

    # Skip if exists
    if os.path.exists(filepath):
        result["status"] = "Exists"
        return result

    try:
        # Verify it's an image before downloading big files (Head Request)
        # Note: We skip this to save time, assuming URLs are valid from analytics.py
        
        # Download
        response = session.get(url, timeout=TIMEOUT_SECONDS, stream=True)
        if response.status_code == 200:
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            result["status"] = "Downloaded"
        else:
            result["status"] = f"Error_{response.status_code}"
    except Exception as e:
        result["status"] = "Failed"
    
    return result

def download_images_parallel():
    print(f"🚀 STARTING HIGH-SPEED IMAGE DOWNLOADER ({MAX_WORKERS} Threads)")
    
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: Input file not found at {INPUT_FILE}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load Data
    df = pd.read_csv(INPUT_FILE)
    # Filter for rows with images
    df_images = df[df['image_attachments'].notna() & (df['image_attachments'] != "")].copy()
    print(f"🔍 Found {len(df_images)} sessions with images.")

    # Prepare Tasks
    tasks = []
    # Create a persistent session for connection pooling
    http_session = requests.Session()
    
    for _, row in df_images.iterrows():
        sess_id = str(row.get('contact_id', row.get('Contact ID', 'unknown'))) # Use Contact ID if session missing
        urls = str(row['image_attachments']).split(" | ")
        context = row.get('full_context', '')
        
        for i, url in enumerate(urls):
            clean_url = url.strip()
            if clean_url.startswith("http"):
                tasks.append((sess_id, i, clean_url, context, http_session))

    print(f"📦 Total images to check/download: {len(tasks)}")
    
    manifest_rows = []
    
    # Parallel Execution
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        futures = [executor.submit(download_single_image, t) for t in tasks]
        
        # Monitor Progress
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
            manifest_rows.append(future.result())

    # Save Manifest
    pd.DataFrame(manifest_rows).to_csv(MANIFEST_FILE, index=False)
    print(f"\n✅ DOWNLOAD COMPLETE.")
    print(f"📂 Images saved to: {OUTPUT_DIR}")
    print(f"📝 Manifest saved to: {MANIFEST_FILE}")

if __name__ == "__main__":
    download_images_parallel()