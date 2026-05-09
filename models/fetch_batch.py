import os
import shutil
import random
from pathlib import Path

# --- SETTINGS ---
BASE_DIR = Path(r"C:\\Users\\Portal Pharmacy\\Documents\\Portal ML Analys\\Ron's Work (2)\\Portal_ML\\Portal_ML_V4")
SOURCE_DIR = BASE_DIR / "data" / "training_images"
INBOX_DIR = BASE_DIR / "data" / "incoming"

BATCH_SIZE = 5000  # How many images do you want to sort right now?

def fetch_next_batch():
    print(f"🎣 FETCHING {BATCH_SIZE} RANDOM IMAGES...")

    if not os.path.exists(INBOX_DIR):
        os.makedirs(INBOX_DIR)

    # Get all available images
    all_files = [f for f in os.listdir(SOURCE_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.webp'))]
    
    if len(all_files) == 0:
        print("⚠️ No images left in training_images! You are done!")
        return

    # Pick random ones
    selection = random.sample(all_files, min(len(all_files), BATCH_SIZE))
    
    count = 0
    for filename in selection:
        src = SOURCE_DIR / filename
        dest = INBOX_DIR / filename
        shutil.move(src, dest) # We MOVE them so they don't get picked again
        count += 1

    print(f"✅ Moved {count} images to 'data/incoming'.")
    print("   👉 Run 'sort_images.py' now!")

if __name__ == "__main__":
    fetch_next_batch()