import tensorflow as tf
import numpy as np
import os
import random
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from pathlib import Path

# --- SETTINGS ---
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = BASE_DIR / "models" / "portal_vision_model_v1.h5"
IMG_DIR = BASE_DIR / "data" / "training_images"

# The classes must match the ALPHABETICAL order from Colab
# Colab said: {'condition': 0, 'junk': 1, 'product': 2}
CLASSES = ['Condition', 'Junk', 'Product']

def test_model():
    print("🧠 LOADING MODEL...")
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model not found at {MODEL_PATH}")
        print("   Did you move the .h5 file there?")
        return

    # Load the trained model
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully.\n")

    # Get random images
    all_files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    if not all_files:
        print("❌ No images found to test.")
        return
        
    sample_images = random.sample(all_files, min(10, len(all_files)))

    print(f"🔍 TESTING ON {len(sample_images)} RANDOM IMAGES:\n")
    print(f"{'FILENAME':<40} | {'PREDICTION':<15} | {'CONFIDENCE':<10}")
    print("-" * 75)

    for filename in sample_images:
        path = os.path.join(IMG_DIR, filename)
        
        try:
            # 1. Preprocess Image (Must match training exactly!)
            img = load_img(path, target_size=(224, 224))
            img_array = img_to_array(img)
            img_array = img_array / 255.0  # Normalize (0-1) like we did in Colab
            img_array = np.expand_dims(img_array, axis=0) # Add batch dimension

            # 2. Predict
            predictions = model.predict(img_array, verbose=0)
            score = predictions[0]
            
            # 3. Interpret
            predicted_class_index = np.argmax(score)
            confidence = np.max(score)
            label = CLASSES[predicted_class_index]
            
            # Print Result
            print(f"{filename[:37]:<40} | {label:<15} | {confidence:.1%}")
            
        except Exception as e:
            print(f"Error reading {filename}: {e}")

if __name__ == "__main__":
    test_model()