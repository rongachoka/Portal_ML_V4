import os
import re
import cv2
import numpy as np
import tensorflow as tf
import easyocr
from pathlib import Path
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from thefuzz import fuzz


class PortalVisionAI:
    def __init__(self, base_dir):
        """
        Initialize the AI. 
        base_dir: Path to your project root (e.g. r"C:\\Users\\...\\Portal_ML_V4")
        """
        self.base_dir = Path(base_dir)
        self.model_path = self.base_dir / "models" / "portal_vision_model_v5.h5"
        self.classes = ['Condition', 'Junk', 'Product']
        self.confidence_threshold = 0.60
        
        # Load Engines
        print("⏳ Pipeline: Loading Vision Model (V5)...")
        self.model = tf.keras.models.load_model(self.model_path)
        
        print("⏳ Pipeline: Loading OCR Engine...")
        self.reader = easyocr.Reader(['en'], gpu=True) # Set False if no CUDA
        
        # Load Knowledge Base
        self.load_brands()
        self.load_aliases()
        print("✅ AI Brain Ready.")

    def load_brands(self):
        # Try importing from config, else fallback
        try:
            import sys
            sys.path.append(str(self.base_dir))
            from src.config.brands import BRAND_LIST
            self.brand_list = BRAND_LIST
        except ImportError:
            print("⚠️ Warning: Could not find BRAND_LIST.")

    def load_aliases(self):
        self.aliases = {
            "ceral": "CERAVE", "cera ve": "CERAVE", "cerave": "CERAVE",
            "laroche": "LA ROCHE-POSAY", "posay": "LA ROCHE-POSAY", 
            "larocheposay": "LA ROCHE-POSAY", "bio-oil": "BIO OIL", 
            "e45": "E45", "hivea": "NIVEA", "niver": "NIVEA", 
            "panado": "PANADOL"
        }

    def preprocess_ocr(self, img_path):
        """Resize huge images for 5x speed boost"""
        img = cv2.imread(str(img_path))
        if img is None: return None
        height, width = img.shape[:2]
        if width > 1200:
            scale = 1200 / width
            img = cv2.resize(img, None, fx=scale, fy=scale)
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def sanitize(self, text):
        ignore = ['the', 'la', 'el', 'le']
        parts = str(text).lower().split()
        clean = " ".join(parts[1:]) if len(parts) > 1 and parts[0] in ignore else str(text).lower()
        return re.sub(r'[^a-zA-Z0-9]', '', clean)

    def get_brand(self, img_path):
        """Extracts brand from image path using OCR + Fuzzy Logic"""
        processed_img = self.preprocess_ocr(img_path)
        if processed_img is None: return "UNKNOWN"

        try:
            results = self.reader.readtext(processed_img, detail=0)
            raw_text = " ".join(results).lower()
            clean_text = re.sub(r'[^a-zA-Z0-9]', '', raw_text)
        except:
            return "UNKNOWN"

        # 1. Alias Check
        for typo, correct in self.aliases.items():
            if len(typo) < 4:
                if re.search(r'\b' + re.escape(typo) + r'\b', raw_text): return correct
            elif self.sanitize(typo) in clean_text:
                return correct

        # 2. Fuzzy Match
        best_brand = None
        best_score = 0
        
        for brand in self.brand_list:
            b_str = str(brand).upper()
            search = self.sanitize(b_str)
            if len(search) < 3: continue

            if len(search) < 5:
                if re.search(r'\b' + re.escape(b_str.lower()) + r'\b', raw_text): return b_str
            elif search in clean_text:
                return b_str

            score = fuzz.partial_ratio(search, clean_text)
            if score > best_score:
                best_score = score
                # Sliding Scale Logic
                length = len(search)
                thresh = 100 if length < 5 else 95 if length == 5 else 82 if length == 6 else 90 if length == 7 else 85
                if score >= thresh: best_brand = b_str
        
        return best_brand if best_brand else "UNKNOWN"

    def analyze(self, image_path):
        """
        The Main Public Function.
        Input: Path to image
        Output: Dictionary { 'tag': '...', 'brand': '...', 'confidence': 0.99 }
        """
        result = {"type": None, "tag": "Unsorted", "brand": None, "confidence": 0.0}
        
        try:
            # Vision Prediction
            img = load_img(image_path, target_size=(224, 224))
            arr = img_to_array(img) / 255.0
            arr = np.expand_dims(arr, axis=0)
            
            pred = self.model.predict(arr, verbose=0)
            label = self.classes[np.argmax(pred[0])]
            confidence = float(np.max(pred[0]))
            
            result["type"] = label
            result["confidence"] = confidence

            # Logic
            if label == 'Junk':
                result["tag"] = "Invalid / Junk"
            elif label == 'Condition':
                result["tag"] = "Consultation / Condition"
            elif label == 'Product':
                if confidence < self.confidence_threshold:
                    result["tag"] = "Product Inquiry (Uncertain)"
                else:
                    brand = self.get_brand(image_path)
                    result["brand"] = brand
                    result["tag"] = f"Product Inquiry - {brand}"
                    
            return result

        except Exception as e:
            print(f"Error analyzing {image_path}: {e}")
            return result