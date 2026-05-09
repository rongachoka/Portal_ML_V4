import tensorflow as tf
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from pathlib import Path

# --- SETTINGS ---
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = BASE_DIR / "models" / "portal_vision_model_v1.h5"
IMG_DIR = BASE_DIR / "data" / "training_images"
CLASSES = ['Condition', 'Junk', 'Product']

def visualize_results():
    print("🧠 LOADING MODEL...")
    if not os.path.exists(MODEL_PATH):
        print("❌ Model not found.")
        return
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # Get random images
    all_files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith('.jpg')]
    sample_files = random.sample(all_files, 15) # Show 15 images

    # Setup the plot grid (3 rows x 5 columns)
    plt.figure(figsize=(15, 10))
    plt.subplots_adjust(hspace=0.5)
    
    print("🔍 GENERATING VISUAL REPORT...")

    for i, filename in enumerate(sample_files):
        try:
            path = os.path.join(IMG_DIR, filename)
            
            # Prepare image for AI
            original_img = load_img(path, target_size=(224, 224))
            img_array = img_to_array(original_img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Predict
            pred = model.predict(img_array, verbose=0)[0]
            idx = np.argmax(pred)
            label = CLASSES[idx]
            conf = np.max(pred)

            # Add to chart
            ax = plt.subplot(3, 5, i + 1)
            ax.imshow(original_img)
            
            # Color code the title: Green if confident (>75%), Red if unsure
            color = 'green' if conf > 0.75 else 'red'
            ax.set_title(f"{label}\n{conf:.1%}", color=color, fontsize=10, fontweight='bold')
            ax.axis('off')

        except Exception as e:
            print(f"Skipping {filename}: {e}")

    print("✅ Displaying images...")
    plt.show()

if __name__ == "__main__":
    visualize_results()