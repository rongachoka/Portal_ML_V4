import pandas as pd
import os
from pathlib import Path

# --- SETTINGS ---
BASE_DIR = Path(__file__).resolve().parent.parent.parent
CSV_FILE = BASE_DIR / "data" / "training_images" / "image_labels_manifest.csv"

def fill_duplicates_safely():
    print("🛡️ STARTING SAFE DUPLICATE FILLER...")
    
    if not os.path.exists(CSV_FILE):
        print(f"❌ Error: File not found at {CSV_FILE}")
        return

    # 1. Load Data
    df = pd.read_csv(CSV_FILE, dtype={'suspected_label': str})
    df['suspected_label'] = df['suspected_label'].fillna('').str.strip()
    
    print(f"   📂 Loaded {len(df)} rows.")
    
    # 2. Build the "Master Map" with Conflict Detection
    url_map = {}
    conflicts = []
    
    # Iterate over rows that HAVE a label
    labeled_rows = df[df['suspected_label'] != ""]
    
    for _, row in labeled_rows.iterrows():
        url = row['original_url']
        label = row['suspected_label']
        
        if url in url_map:
            # Check if the existing label matches the new one
            if url_map[url] != label:
                conflicts.append(url)
        else:
            url_map[url] = label

    # 3. SAFETY CHECK
    if conflicts:
        print(f"\n⚠️ SAFETY STOP! Found {len(set(conflicts))} URLs with conflicting labels.")
        print("   (Example: You labeled the same image 'product' in one row and 'junk' in another)")
        print("   Please fix these manually before autofilling:")
        for i, url in enumerate(set(conflicts)):
            if i >= 5: break # Only show first 5
            print(f"   - {url}")
        return # STOP HERE

    print(f"   🧠 Safety Check Passed. Found {len(url_map)} unique labels to propagate.")

    # 4. Apply the Logic
    def apply_label(row):
        url = row['original_url']
        current_label = row['suspected_label']
        
        # If the row is already labeled, leave it alone (it matches the map anyway)
        if current_label != "":
            return current_label
            
        # If it is BLANK, look it up in our map
        if url in url_map:
            return url_map[url]
        
        return "" # Keep it blank if we don't know

    df['suspected_label'] = df.apply(apply_label, axis=1)
    
    # 5. Save
    df.to_csv(CSV_FILE, index=False)
    
    filled_count = len(df[df['suspected_label'] != ""])
    print("\n✅ DUPLICATES FILLED SUCCESSFULLY.")
    print(f"   - Total labeled rows: {filled_count}")
    print(f"   - Saved to: {CSV_FILE}")

if __name__ == "__main__":
    fill_duplicates_safely()