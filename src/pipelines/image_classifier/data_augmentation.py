"""
data_augmentation.py
====================
Augments the brand image training dataset using Keras ImageDataGenerator.

For each brand class with too few images, generates synthetic variations
(rotation, zoom, flip, brightness shift) to reach the target sample count,
and saves them alongside the originals.

Input:  Brand_Images_TopDown/{brand}/ — existing training images
Output: Additional augmented images written into the same brand subfolders

Run manually before model training when any class has fewer than ~50 images.
"""

import random
from pathlib import Path

from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
    img_to_array,
    load_img,
    save_img
)


# ==========================================
# CONFIGURATION
# ==========================================
# Point this to your cleaned TopDown directory
TARGET_DIR = Path(
    "G:\\My Drive\\Portal_ML\\Portal_ML_V4\\data (1)\\01_raw (1)\\"
    "Brand_Images_TopDown"
)
TARGET_MINIMUM = 10


# ==========================================
# THE "AI PHOTOGRAPHER" SETUP
# ==========================================
# This defines exactly how the AI is allowed to alter the images
datagen = ImageDataGenerator(
    rotation_range=25,          # Tilt the camera up to 25 degrees
    width_shift_range=0.15,     # Pan left/right by 15%
    height_shift_range=0.15,    # Pan up/down by 15%
    brightness_range=[0.6, 1.4],# Simulate dark shadows (0.6) to bright light (1.4)
    zoom_range=[0.8, 1.2],      # Zoom in (0.8) or zoom out (1.2)
    horizontal_flip=True,       # Mirror the image
    fill_mode='nearest'         # Fill in any blank pixels created by rotating
)


# ==========================================
# MAIN AUGMENTATION SCRIPT
# ==========================================
def augment_dataset_gaps():
    print("🚀 INITIALIZING AI DATA AUGMENTATION...")
    
    if not TARGET_DIR.exists():
        print("❌ Error: Target directory does not exist.")
        return

    brand_folders = sorted([f for f in TARGET_DIR.iterdir() if f.is_dir()])
    augmented_total = 0

    for brand_dir in brand_folders:
        existing_images = list(brand_dir.glob("*.jpg"))
        current_count = len(existing_images)
        
        # If the folder has 0 images, we can't augment anything
        if current_count == 0:
            continue
            
        images_needed = TARGET_MINIMUM - current_count
        
        if images_needed > 0:
            print(f"📂 [{brand_dir.name}] Has {current_count}/{TARGET_MINIMUM}. "
                  f"Generating {images_needed} variations...")
            
            # Load all existing images into memory to act as base templates
            base_images = []
            for img_path in existing_images:
                try:
                    img = load_img(img_path)
                    x = img_to_array(img)
                    base_images.append(x)
                except Exception as e:
                    print(f"   ⚠️ Could not read {img_path.name}: {e}")
            
            if not base_images:
                continue

            generated_for_this_brand = 0
            
            # Keep generating until we hit the minimum requirement
            while generated_for_this_brand < images_needed:
                # Randomly pick one of the base images to alter
                base_img = random.choice(base_images)
                # Reshape it to (1, width, height, channels) for the datagen
                base_img = base_img.reshape((1,) + base_img.shape)
                
                # Generate a single augmented version
                iterator = datagen.flow(base_img, batch_size=1)
                augmented_batch = next(iterator)
                augmented_img = augmented_batch[0]
                
                # Save the new image physically to the folder
                save_name = f"aug_{generated_for_this_brand + 1}.jpg"
                save_path = brand_dir / save_name
                
                try:
                    save_img(str(save_path), augmented_img)
                    generated_for_this_brand += 1
                    augmented_total += 1
                except Exception as e:
                    print(f"   ❌ Error saving augmented image: {e}")
                    break

    print("\n" + "=" * 40)
    print(f"🎉 AUGMENTATION COMPLETE! Synthesized {augmented_total} new images.")
    print("=" * 40)


if __name__ == "__main__":
    augment_dataset_gaps()