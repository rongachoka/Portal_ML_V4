import os
import shutil
import re
import easyocr
import tensorflow as tf
import numpy as np
import cv2  # <--- NEW: For resizing images (Speed Boost)
from pathlib import Path
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from thefuzz import fuzz
from tqdm import tqdm

# --- CONFIGURATION ---
BASE_DIR = Path(r"C:\Users\Portal Pharmacy\Documents\Portal ML Analys\Ron's Work (2)\Portal_ML\Portal_ML_V4")

# 1. PATHS
SOURCE_DIR = BASE_DIR / "data" / "incoming" 
DEST_DIR = BASE_DIR / "data" / "final_sorted"
MODEL_PATH = BASE_DIR / "models" / "portal_vision_model_v5.h5" # 88% val accuracy
CLASSES = ['Condition', 'Junk', 'Product']
CONFIDENCE_THRESHOLD = 0.60 

# 2. BRAND LIST
try:
    from Portal_ML_V4.src.config.brands import BRAND_LIST
except ImportError:
    print("⚠️ Warning: Could not import BRAND_LIST from config.")

# 3. ALIASES (The "Memory" of past mistakes)
ALIASES = {
    "ceral": "CeraVe", 
    "cera ve": "CeraVe", 
    "cerave": "CeraVe",
    "laroche": "La Roche Posay", 
    "posay": "La Roche Posay",
    "neutr": "Neutrogena", 
    "e45": "E45",
    "hivea": "NIVEA", 
    "niver": "NIVEA", 
}

# --- LOAD ENGINES ---
print("⏳ Loading V5 Brain...")
model = tf.keras.models.load_model(MODEL_PATH)

print("📖 Loading GPU OCR...")
# gpu=True requires CUDA. If it fails, set to False.
reader = easyocr.Reader(['en'], gpu=True) 

# --- HELPERS ---

def preprocess_for_ocr(img_path):
    """Resizes huge images to speed up OCR by 5x"""
    img = cv2.imread(str(img_path))
    if img is None: return None
    
    # Resize if width > 1200px (OCR doesn't need 4000px)
    height, width = img.shape[:2]
    if width > 1200:
        scale = 1200 / width
        img = cv2.resize(img, None, fx=scale, fy=scale)
    
    # Convert to gray for slightly better text reading
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def sanitize_brand(brand_name):
    ignore_words = ['the', 'la', 'el', 'le']
    parts = str(brand_name).lower().split()
    if len(parts) > 1 and parts[0] in ignore_words:
        clean = " ".join(parts[1:])
    else:
        clean = str(brand_name).lower()
    return re.sub(r'[^a-zA-Z0-9]', '', clean)

def get_brand_from_image(img_path):
    # 1. Fast OCR Read
    try:
        processed_img = preprocess_for_ocr(img_path)
        if processed_img is None: return None
        results = reader.readtext(processed_img, detail=0)
        raw_text_spaced = " ".join(results).lower()
        clean_text = re.sub(r'[^a-zA-Z0-9]', '', raw_text_spaced)
    except:
        return None

    # 2. ALIAS CHECK
    for typo, correct_brand in ALIASES.items():
        if len(typo) < 4:
            pattern = r'\b' + re.escape(typo) + r'\b'
            if re.search(pattern, raw_text_spaced):
                return correct_brand
        else:
            clean_typo = re.sub(r'[^a-zA-Z0-9]', '', typo)
            if clean_typo in clean_text:
                return correct_brand

    # 3. FUZZY MATCH
    best_brand = None
    highest_score = 0
    
    for brand in BRAND_LIST:
        brand_str = str(brand).upper()
        search_term = sanitize_brand(brand_str)
        term_len = len(search_term)
        if term_len < 3: continue

        # Exact
        if term_len < 5:
            pattern = r'\b' + re.escape(brand_str.lower()) + r'\b'
            if re.search(pattern, raw_text_spaced):
                 return brand_str
        elif search_term in clean_text:
             return brand_str

        # Fuzzy (Strict Sliding Scale)
        score = fuzz.partial_ratio(search_term, clean_text)
        if score > highest_score:
            highest_score = score
            if term_len < 5: threshold = 100 
            elif term_len == 5: threshold = 95
            elif term_len == 6: threshold = 82 
            elif term_len == 7: threshold = 90
            else: threshold = 85
            
            if score >= threshold:
                best_brand = brand_str

    return best_brand

# --- MAIN LOOP ---
def sort_batch():
    if not os.path.exists(SOURCE_DIR):
        print("❌ Incoming folder missing.")
        return

    files = [f for f in os.listdir(SOURCE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
    print(f"🚀 V5 Sorting {len(files)} images...")
    
    for filename in tqdm(files):
        src_path = SOURCE_DIR / filename
        
        try:
            # 1. Vision Prediction (V5)
            img = load_img(src_path, target_size=(224, 224))
            arr = img_to_array(img) / 255.0
            arr = np.expand_dims(arr, axis=0)
            
            pred = model.predict(arr, verbose=0)
            label = CLASSES[np.argmax(pred[0])]
            confidence = np.max(pred[0])
            
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

            # 2. Move
            final_path = DEST_DIR / subfolder
            if not os.path.exists(final_path):
                os.makedirs(final_path)
            shutil.move(src_path, final_path / filename)

        except Exception as e:
            print(f"❌ Error {filename}: {e}")

    print("\n✅ Sort Complete!")

if __name__ == "__main__":
    sort_batch()