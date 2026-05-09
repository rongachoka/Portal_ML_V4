import cv2
import os
import shutil
import yaml
import gc
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from ultralytics import YOLO
import torch

# ===========================
# ⚙️ CONFIGURATION
# ===========================

# 🚨 FIX 1: Use 'r' for Raw Strings to handle Windows backslashes correctly
# TIP: It is highly recommended to move data to a local SSD (C: or D:) instead of G: Drive for speed.
SOURCE_DATA_DIR = Path(r"G:\My Drive\Portal_ML\Portal_ML_V4\data (1)\01_raw (1)\Brand_Images_TopDown")
OUTPUT_DIR = Path(r"G:\My Drive\Portal_ML\Portal_ML_V4\data (1)\01_raw (1)\yolo_dataset_v1")

VAL_SPLIT = 0.2

# ===========================
# 🛠️ AUTO-ANNOTATION ENGINE
# ===========================
def find_bounding_box(image_path):
    """
    Uses OpenCV to find the product on a white background.
    Returns YOLO format: (x_center, y_center, width, height) normalized 0-1.
    """
    try:
        # 1. Load Image
        img = cv2.imread(str(image_path))
        if img is None: return None
        h, w, _ = img.shape

        # 2. Preprocess
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

        # 3. Clean Noise
        kernel = np.ones((5,5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # 4. Find Contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None

        # 5. Get largest object
        c = max(contours, key=cv2.contourArea)
        x, y, box_w, box_h = cv2.boundingRect(c)

        # 6. Safety Check (ignore tiny specs < 5% area)
        if (box_w * box_h) < (w * h * 0.05): return None

        # 7. Convert to YOLO Format
        center_x = (x + (box_w / 2)) / w
        center_y = (y + (box_h / 2)) / h
        norm_w = box_w / w
        norm_h = box_h / h

        return (center_x, center_y, norm_w, norm_h)

    except Exception as e:
        print(f"⚠️ Error processing {image_path.name}: {e}")
        return None

def generate_dataset():
    print(f"📂 Checking Source: {SOURCE_DATA_DIR}")
    if not SOURCE_DATA_DIR.exists():
        print(f"❌ Error: Source directory not found.")
        return

    # 1. Map Classes
    classes = sorted([d.name for d in SOURCE_DATA_DIR.iterdir() if d.is_dir()])
    class_map = {name: idx for idx, name in enumerate(classes)}
    print(f"📋 Found {len(classes)} Classes")

    # 2. Setup Dirs
    for split in ['train', 'val']:
        (OUTPUT_DIR / split / 'images').mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / split / 'labels').mkdir(parents=True, exist_ok=True)

    # 3. Collect Images
    all_images = []
    for cls_name in classes:
        cls_dir = SOURCE_DATA_DIR / cls_name
        # Check multiple extensions
        imgs = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png")) + list(cls_dir.glob("*.jpeg"))
        for img in imgs:
            all_images.append((img, cls_name))

    # 4. Split
    train_imgs, val_imgs = train_test_split(all_images, test_size=VAL_SPLIT, random_state=42, stratify=[x[1] for x in all_images])

    # 5. Process
    print(f"🚀 Processing {len(all_images)} images...")
    stats = {'success': 0, 'failed': 0}

    for split_name, img_list in [('train', train_imgs), ('val', val_imgs)]:
        print(f"   Writing {split_name} set...")
        for img_path, cls_name in tqdm(img_list):
            bbox = find_bounding_box(img_path)
            if bbox:
                cls_id = class_map[cls_name]
                
                # Copy Image
                dest_img = OUTPUT_DIR / split_name / 'images' / img_path.name
                shutil.copy(img_path, dest_img)
                
                # Write Label
                label_content = f"{cls_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}"
                dest_label = OUTPUT_DIR / split_name / 'labels' / f"{img_path.stem}.txt"
                with open(dest_label, "w") as f:
                    f.write(label_content)
                stats['success'] += 1
            else:
                stats['failed'] += 1

    # 6. Generate Config
    yaml_content = {
        'path': str(OUTPUT_DIR.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'nc': len(classes),
        'names': classes
    }
    with open(OUTPUT_DIR / "data.yaml", "w") as f:
        yaml.dump(yaml_content, f, sort_keys=False)

    print(f"✅ DATASET READY: {OUTPUT_DIR}")

# ===========================
# 🚀 MAIN EXECUTION BLOCK
# ===========================
if __name__ == "__main__":
    # 1. CHECK & FIX DATASET CONFIGURATION
    print("⚙️ Verifying dataset configuration...")
    
    # Check if images exist
    images_exist = (OUTPUT_DIR / 'train' / 'images').exists() and \
                   any((OUTPUT_DIR / 'train' / 'images').iterdir())
    
    if not images_exist:
        print("❌ Dataset not found. Generating now...")
        generate_dataset()
    else:
        print("✅ Images found. Skipping copy step.")

    # 2. REGENERATE YAML (Windows Path Fix)
    try:
        if SOURCE_DATA_DIR.exists():
            classes = sorted([d.name for d in SOURCE_DATA_DIR.iterdir() if d.is_dir()])
        else:
            print("⚠️ Source dir not found, reading classes from existing yaml...")
            with open(OUTPUT_DIR / "data.yaml", 'r') as f:
                old_yaml = yaml.safe_load(f)
                classes = old_yaml['names']

        yaml_content = {
            'path': str(OUTPUT_DIR.absolute()), 
            'train': 'train/images',
            'val': 'val/images',
            'nc': len(classes),
            'names': classes
        }
        
        with open(OUTPUT_DIR / "data.yaml", "w") as f:
            yaml.dump(yaml_content, f, sort_keys=False)
            
        print(f"✅ Fixed data.yaml paths.")
        
    except Exception as e:
        print(f"❌ Error fixing YAML: {e}")
        exit()

    # ---------------------------------------------------------
    # 🧹 MEMORY CLEANUP (GC.COLLECT LOCATION)
    # ---------------------------------------------------------
    # This clears any RAM used during the file scanning/copying above
    # so your system has maximum memory available for the Training phase.
    print("🧹 Cleaning up memory before training...")
    gc.collect()
    # ---------------------------------------------------------

    # 3. Hardware Check
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"🔥 Training Device: {torch.cuda.get_device_name(0) if device == 0 else 'CPU'}")

    # 4. Train
    print("🚀 Starting YOLOv8 MEDIUM Training...")
    model = YOLO('yolov8m.pt') 

    model.train(
        data=str(OUTPUT_DIR / "data.yaml"),
        epochs=50,
        imgsz=640,
        batch=8, 
        project=str(OUTPUT_DIR / "training_runs"),
        name='yolo_v1_medium',
        exist_ok=True,
        device=device,
        patience=10,
        
        # 🚨 THE CRITICAL FIX FOR WINERROR 1455
        # Windows creates a new process for every worker. 
        # 4 workers = 4x Memory usage = Crash.
        # 0 workers = Runs in main process = Low Memory usage.
        workers=0  
    )