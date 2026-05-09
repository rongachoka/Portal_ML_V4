import os
import shutil
import re
import sys
import uuid
import cv2  # OpenCV for fast resizing
import numpy as np
import tensorflow as tf
import easyocr
from pathlib import Path
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from thefuzz import fuzz
from tqdm import tqdm

# ==========================================
# ⚙️  CONFIGURATION & SETTINGS
# ==========================================

# 1. PATHS
# Adjust this if your folder structure is different
BASE_DIR = Path(r"C:\\Users\\Portal Pharmacy\\Documents\\Portal ML Analys\\Ron's Work (2)\\Portal_ML\\Portal_ML_V4")
MODEL_PATH = BASE_DIR / "models" / "portal_vision_model_v5.h5"  # <--- ENSURE THIS IS V5

# Folders for Sorting Mode
INCOMING_DIR = BASE_DIR / "data" / "incoming"
FINAL_SORTED_DIR = BASE_DIR / "data" / "final_sorted"

# Folders for Harvesting Mode
TRAINING_DIR = BASE_DIR / "data" / "sorted"

# 2. AI PARAMETERS
CLASSES = ['Condition', 'Junk', 'Product']
CONFIDENCE_THRESHOLD = 0.60 # V5 is confident, so 60% is safe
OCR_GPU = True  # Set to False if you don't have NVIDIA CUDA

# 3. BRAND KNOWLEDGE BASE (Dynamic Import)
BRAND_LIST = []

def load_brand_config():
    """Attempts to import the authoritative BRAND_LIST from config file."""
    global BRAND_LIST
    try:
        # We append the parent directory to path so python can find 'src'
        sys.path.append(str(BASE_DIR))
        from src.config.brands import BRAND_LIST as IMPORTED_LIST
        BRAND_LIST = IMPORTED_LIST
        print(f"✅ Successfully imported {len(BRAND_LIST)} brands from src.config.brands")
    except ImportError as e:
        print(f"⚠️  WARNING: Could not import config ({e}).")

# 4. ALIAS DICTIONARY (The "Typo Fixer")
# Add new typos here as you find them in the 'debug' process
ALIASES = {
    "ceral": "CERAVE", "cera ve": "CERAVE", "cerave": "CERAVE",
    "laroche": "LA ROCHE-POSAY", "posay": "LA ROCHE-POSAY", "larocheposay": "LA ROCHE-POSAY",
    "neutr": "NEUTROGENA", "bio-oil": "BIO OIL", "e45": "E45",
    "hivea": "NIVEA", "niver": "NIVEA",
    "panado": "PANADOL", "sevenseas": "SEVEN SEAS"
}

# ==========================================
# 🧠  AI ENGINE INITIALIZATION
# ==========================================
model = None
reader = None

def load_engines():
    """Lazy load engines only when needed to save startup time."""
    global model, reader
    
    # 1. Load Brands
    if not BRAND_LIST:
        load_brand_config()

    # 2. Load Vision Model
    if model is None:
        print("\n⏳ Loading V5 Vision Model...")
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            print("   ✅ Vision Model Loaded.")
        except Exception as e:
            print(f"   ❌ Error loading model: {e}")
            print(f"      Check path: {MODEL_PATH}")
            sys.exit()

    # 3. Load OCR
    if reader is None:
        print("⏳ Loading OCR Engine...")
        reader = easyocr.Reader(['en'], gpu=OCR_GPU)
        print("   ✅ OCR Engine Loaded.")

# ==========================================
# 🛠️  HELPER FUNCTIONS
# ==========================================

def preprocess_for_ocr(img_path):
    """Resizes huge images (speed boost) and converts to gray."""
    img = cv2.imread(str(img_path))
    if img is None: return None
    
    # Resize if width > 1200px
    height, width = img.shape[:2]
    if width > 1200:
        scale = 1200 / width
        img = cv2.resize(img, None, fx=scale, fy=scale)
    
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def sanitize_brand(brand_name):
    """Standardizes brand strings for matching."""
    # We removed 'dr' from ignore list so 'Dr Organic' works
    ignore_words = ['the', 'la', 'el', 'le']
    parts = str(brand_name).lower().split()
    if len(parts) > 1 and parts[0] in ignore_words:
        clean = " ".join(parts[1:])
    else:
        clean = str(brand_name).lower()
    return re.sub(r'[^a-zA-Z0-9]', '', clean)

def get_brand_from_image(img_path):
    """
    The Core Logic: Combines OCR, Aliases, and Fuzzy Matching.
    Includes 'Sliding Scale' threshold logic.
    """
    # 1. Run Fast OCR
    try:
        processed_img = preprocess_for_ocr(img_path)
        if processed_img is None: return None
        results = reader.readtext(processed_img, detail=0)
        # Keep spaces for Whole Word matching (Regex)
        raw_text_spaced = " ".join(results).lower()
        # Remove spaces for Fuzzy matching
        clean_text = re.sub(r'[^a-zA-Z0-9]', '', raw_text_spaced)
    except:
        return None

    # 2. Check Aliases (Instant Win)
    for typo, correct_brand in ALIASES.items():
        if len(typo) < 4: # Short typos need word boundaries
            pattern = r'\b' + re.escape(typo) + r'\b'
            if re.search(pattern, raw_text_spaced): return correct_brand
        else:
            clean_typo = re.sub(r'[^a-zA-Z0-9]', '', typo)
            if clean_typo in clean_text: return correct_brand

    # 3. Fuzzy Match
    best_brand = None
    highest_score = 0
    
    for brand in BRAND_LIST:
        brand_str = str(brand).upper()
        search_term = sanitize_brand(brand_str)
        term_len = len(search_term)
        
        # Skip brands shorter than 3 chars (too risky)
        if term_len < 3: continue

        # A. Exact Match Logic
        if term_len < 5:
            # Short brands (Olay, Roc) need EXACT WHOLE WORD match
            pattern = r'\b' + re.escape(brand_str.lower()) + r'\b'
            if re.search(pattern, raw_text_spaced): return brand_str
        elif search_term in clean_text:
             # Long brands (Neutrogena) can be substrings
             return brand_str

        # B. Fuzzy Match with Sliding Scale Thresholds
        score = fuzz.partial_ratio(search_term, clean_text)
        
        if score > highest_score:
            highest_score = score
            
            # --- THE LOGIC GATES ---
            if term_len < 5: 
                threshold = 100 # Short words must match perfectly (or via Alias)
            elif term_len == 5: 
                threshold = 95  # 'Cantu', 'Nivea' (Strict to avoid 'Hivea')
            elif term_len == 6: 
                threshold = 82  # 'CeraVe' (Allows 1 typo)
            elif term_len == 7: 
                threshold = 90  # 'Eucerin' (Strict to avoid 'Glycerin')
            else: 
                threshold = 85  # Long words (Allows small errors)
            
            if score >= threshold: 
                best_brand = brand_str

    return best_brand

# ==========================================
# 🚀  TASK 1: BATCH SORTER (The Daily Tool)
# ==========================================
def run_sorter():
    print("\n" + "="*40)
    print("   🚀 STARTING BATCH SORTER")
    print("="*40)
    
    if not os.path.exists(INCOMING_DIR):
        print(f"❌ Incoming folder missing: {INCOMING_DIR}")
        return

    # Grab files
    files = [f for f in os.listdir(INCOMING_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
    if not files:
        print("⚠️  No images found in 'data/incoming'.")
        return
        
    print(f"📂 Found {len(files)} images. Loading AI...")
    load_engines() # Load now
    
    success_count = 0
    
    for filename in tqdm(files, desc="Sorting"):
        src_path = INCOMING_DIR / filename
        try:
            # A. Vision Prediction
            img = load_img(src_path, target_size=(224, 224))
            arr = img_to_array(img) / 255.0
            arr = np.expand_dims(arr, axis=0)
            
            pred = model.predict(arr, verbose=0)
            label = CLASSES[np.argmax(pred[0])]
            confidence = np.max(pred[0])
            
            # B. Routing Logic
            subfolder = "UNSORTED"
            
            if label == 'Junk':
                subfolder = "JUNK"
            elif label == 'Condition':
                subfolder = "CONDITIONS"
            elif label == 'Product':
                if confidence < CONFIDENCE_THRESHOLD:
                    subfolder = "PRODUCT_UNCERTAIN"
                else:
                    detected_brand = get_brand_from_image(src_path)
                    if detected_brand:
                        safe_brand = detected_brand.replace(" ", "_").replace("-", "_")
                        subfolder = f"BRAND_{safe_brand}"
                    else:
                        subfolder = "PRODUCT_UNKNOWN_BRAND"

            # C. Move File
            final_path = FINAL_SORTED_DIR / subfolder
            os.makedirs(final_path, exist_ok=True)
            shutil.move(src_path, final_path / filename)
            success_count += 1
            
        except Exception as e:
            print(f"❌ Error on {filename}: {e}")

    print(f"\n✅ Done! Sorted {success_count} images.")
    print(f"📂 Check results in: {FINAL_SORTED_DIR}")

# ==========================================
# 🚜  TASK 2: DATA HARVESTER (The Training Tool)
# ==========================================
def run_harvester():
    print("\n" + "="*40)
    print("   🚜 STARTING DATA HARVESTER")
    print("   Moving verified files back to training set...")
    print("="*40)
    
    if not os.path.exists(FINAL_SORTED_DIR):
        print("❌ 'final_sorted' folder is empty/missing.")
        return
        
    moved_count = 0
    
    for folder_name in os.listdir(FINAL_SORTED_DIR):
        source_folder = FINAL_SORTED_DIR / folder_name
        if not os.path.isdir(source_folder): continue
        
        # Determine Training Destination
        dest_category = None
        if folder_name.upper() == "JUNK": dest_category = "junk"
        elif folder_name.upper() == "CONDITIONS": dest_category = "condition"
        elif folder_name.upper().startswith("BRAND_") or folder_name.upper().startswith("PRODUCT_"):
            dest_category = "product"
            
        if dest_category:
            dest_path = TRAINING_DIR / dest_category
            os.makedirs(dest_path, exist_ok=True)
            
            for f in os.listdir(source_folder):
                src = source_folder / f
                dst = dest_path / f
                
                # Handle Duplicates
                if os.path.exists(dst):
                    name, ext = os.path.splitext(f)
                    dst = dest_path / f"{name}_{uuid.uuid4().hex[:4]}{ext}"
                
                shutil.move(src, dst)
                moved_count += 1
    
    print(f"\n✅ Harvest Complete!")
    print(f"🌱 Added {moved_count} new images to {TRAINING_DIR}")
    print("🗑️  'final_sorted' is now empty and ready for new batch.")

# ==========================================
# 🎮  MAIN MENU
# ==========================================
if __name__ == "__main__":
    while True:
        print("\n" + "="*30)
        print("   PORTAL AI MASTER CONTROL")
        print("="*30)
        print("1. 🚀 Sort Incoming Images")
        print("2. 🚜 Harvest Verified Data (Merge to Training)")
        print("3. ❌ Exit")
        
        choice = input("\nSelect an option [1-3]: ").strip()
        
        if choice == "1":
            run_sorter()
        elif choice == "2":
            confirm = input("⚠️  Are you sure you verified the folders in 'final_sorted'? (y/n): ")
            if confirm.lower() == 'y':
                run_harvester()
        elif choice == "3":
            print("Goodbye!")
            break
        else:
            print("Invalid option.")