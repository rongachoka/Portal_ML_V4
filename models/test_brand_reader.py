import os
import re
import random
import easyocr
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from pathlib import Path
from thefuzz import fuzz

# --- IMPORT CONFIG ---
try:
    from Portal_ML_V4.src.config.brands import BRAND_LIST
except ImportError:
    print("⚠️ Warning: Could not find BRAND_LIST. Using default test list.")

# --- SETTINGS ---
BASE_DIR = Path(r"C:\\Users\\Portal Pharmacy\\Documents\\Portal ML Analys\\Ron's Work (2)\\Portal_ML\\Portal_ML_V4")
MODEL_PATH = BASE_DIR / "models" / "portal_vision_model_v3.h5" 
IMG_DIR = BASE_DIR / "data" / "sorted" / "product" / "sorted"
CLASSES = ['Condition', 'Junk', 'Product']

# LOWER CONFIDENCE GATE to catch more products
CONFIDENCE_THRESHOLD = 0.60 

print("⏳ Loading AI Brain...")
model = tf.keras.models.load_model(MODEL_PATH)
print("📖 Loading OCR Reader...")
reader = easyocr.Reader(['en'], gpu=False) 

# --- HELPERS ---
def sanitize_brand(brand_name):
    # Remove "The", "Dr", etc.
    ignore_words = ['the', 'dr', 'la', 'el', 'le']
    parts = str(brand_name).lower().split()
    if len(parts) > 1 and parts[0] in ignore_words:
        clean = " ".join(parts[1:])
    else:
        clean = str(brand_name).lower()
    return re.sub(r'[^a-zA-Z0-9]', '', clean)

def analyze_product(filename):
    path = IMG_DIR / filename
    
    # 1. VISION
    try:
        img = load_img(path, target_size=(224, 224))
        arr = img_to_array(img) / 255.0
        arr = np.expand_dims(arr, axis=0)
        
        pred = model.predict(arr, verbose=0)
        label = CLASSES[np.argmax(pred[0])]
        confidence = np.max(pred[0])
        
        if label != 'Product': return
        if confidence < CONFIDENCE_THRESHOLD: return 

    except: return

    # 2. OCR
    try:
        results = reader.readtext(str(path), detail=0)
        raw_text_spaced = " ".join(results).lower()
        clean_text = re.sub(r'[^a-zA-Z0-9]', '', raw_text_spaced)
        
        print(f"\n📸 {filename} | Product: {confidence:.1%}")
        print(f"   📖 Read: {raw_text_spaced[:60]}...") 
        
    except: return

    # 3. BRAND MATCHING (Strict Sliding Scale)
    
    found_brand = None
    best_match_name = "None"
    highest_score = 0
    
    for brand in BRAND_LIST:
        brand_str = str(brand).upper()
        search_term = sanitize_brand(brand_str)
        term_len = len(search_term)
        
        if term_len < 3: continue 

        # A. EXACT MATCH
        if term_len < 5:
            pattern = r'\b' + re.escape(brand_str.lower()) + r'\b'
            if re.search(pattern, raw_text_spaced):
                 print(f"   🎯 EXACT MATCH: {brand_str}")
                 found_brand = brand_str
                 break 
        elif search_term in clean_text:
             print(f"   🎯 EXACT MATCH: {brand_str}")
             found_brand = brand_str
             break

        # B. FUZZY MATCH
        score = fuzz.partial_ratio(search_term, clean_text)
        
        if score > highest_score:
            highest_score = score
            best_match_name = brand_str
            
            # --- FINAL TUNED THRESHOLDS ---
            if term_len < 5:
                # 3-4 chars (Olay): Must be perfect or Regex
                threshold = 100 
            elif term_len == 5:
                # 5 chars (Cantu): Very Strict (Kills 'C1antu')
                threshold = 95
            elif term_len == 6:
                # 6 chars (CeraVe): Strict-ish (Allows 'Cerale')
                threshold = 82
            elif term_len == 7:
                # 7 chars (Eucerin): VERY Strict (Kills 'Glycerin')
                threshold = 90
            else:
                # 8+ chars (Shea Moisture): High (Kills 'Moisture' alone)
                threshold = 85
            
            if score >= threshold:
                found_brand = best_match_name

    if found_brand:
        print(f"   ✅ RESULT: {found_brand}")
    else:
        print(f"   ⚠️  Unknown (Best: {best_match_name} @ {highest_score})")
    print("-" * 40)

if __name__ == "__main__":
    if not os.path.exists(IMG_DIR):
        print(f"❌ Error: Path not found: {IMG_DIR}")
    else:
        all_products = [f for f in os.listdir(IMG_DIR) if f.lower().endswith('.jpg')]
        if len(all_products) > 0:
            sample = random.sample(all_products, min(10, len(all_products)))
            for f in sample:
                analyze_product(f)
        else:
            print("⚠️ Folder is empty.")