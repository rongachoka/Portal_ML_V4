"""
scraping_test.py
================
Test harness for web scraping logic and image download utilities.

Used to prototype and validate scraping approaches before adding them to
brand_scraper.py or competitor_scraper.py. Also tests DuckDuckGo image
search (via ddgs) and fastai image verification for training data collection.

Input:  web URLs / search queries (hardcoded inside script)
Output: downloaded images or console scraping output (for testing only)
"""

import re
import time
import warnings
import pandas as pd
import requests
import urllib.parse
from pathlib import Path
from bs4 import BeautifulSoup
from ddgs import DDGS
from fastai.vision.all import verify_images, get_image_files, download_images

# Silence Jupyter/Python deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ==========================================
# 1. CONFIGURATION
# ==========================================
# 🚨 UPDATE THESE PATHS TO YOUR LOCAL DRIVE
KB_PATH = Path("G:\\My Drive\\Portal_ML\\Portal_ML_V4\\data (1)\\01_raw (1)\\Final_Knowledge_Base_PowerBI (1).csv")
OUTPUT_DIR = Path("G:\\My Drive\\Portal_ML\\Portal_ML_V4\\data (1)\\01_raw (1)\\Brand_Image_Dataset_V1_Ultimate")

MAX_PRODUCTS_PER_BRAND = 10
IMAGES_PER_PRODUCT_FALLBACK = 3  
SEARCH_SUFFIX = " product packaging white background"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def clean_product_name(prod_name):
    """Removes weights, volumes, and special chars for clean filenames and searches."""
    clean_name = re.sub(r'\b\d+\s*(?:ml|g|kg|mg|oz|l|fl\s*oz)\b', '', str(prod_name), flags=re.IGNORECASE)
    clean_name = re.sub(r'[^a-zA-Z0-9\s]', '', clean_name)
    return clean_name.strip()

def extract_portal_image(url):
    """Method 1: Scrapes a specific Portal Pharmacy product page for the UNCROPPED image."""
    if pd.isna(url) or not str(url).startswith('http'):
        return None
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code != 200:
            return None
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 1. BEST CHOICE: The main large image link (Often wrapped in a popup/magnifier anchor)
        # OpenCart usually stores the highest-res, uncropped image in a 'thumbnails' list
        main_link = soup.select_one('ul.thumbnails li:nth-of-type(1) a.thumbnail')
        if main_link and main_link.get('href'):
            return main_link['href']

        # 2. SECOND CHOICE: The main product image tag on the page
        main_img = soup.select_one('.product-info .image img') or soup.find('img', id='zoom')
        if main_img:
            return main_img.get('data-src') or main_img.get('src')
            
        # 3. THIRD CHOICE: The general responsive image
        resp_img = soup.find('img', class_='img-responsive')
        if resp_img:
            return resp_img.get('data-src') or resp_img.get('src')

        # 4. LAST RESORT ONLY: The Open Graph Meta Image (Warning: May be cropped)
        og_image = soup.find('meta', property='og:image')
        if og_image and og_image.get('content'):
            return og_image['content']
            
    except Exception:
        pass
    return None

def search_portal_for_image(product_name):
    """Method 2: Uses Portal Pharmacy's search bar to find missing links."""
    # Clean the name to broaden the search (e.g., remove "50ml")
    clean_query = clean_product_name(product_name)
    encoded_query = urllib.parse.quote(clean_query)
    search_url = f"https://portalpharmacy.ke/index.php?route=product/search&search={encoded_query}"
    
    try:
        response = requests.get(search_url, headers=HEADERS, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            # Find the very first product in the search results
            first_product = soup.find('div', class_='product-layout') or soup.find('div', class_='product-thumb')
            if first_product:
                img_tag = first_product.find('img', class_='img-responsive')
                if img_tag:
                    return img_tag.get('data-src') or img_tag.get('src')
    except Exception:
        pass
    return None

def download_single_image(img_url, dest_path):
    """Streams a single image to the disk."""
    if dest_path.exists():
        return True
    try:
        response = requests.get(img_url, headers=HEADERS, stream=True, timeout=10)
        if response.status_code == 200:
            with open(dest_path, 'wb') as f:
                for chunk in response.iter_content(1024):
                    if chunk: f.write(chunk)
            return True
    except Exception:
        pass
    return False

def search_ddg_images(term, max_images=3):
    """Method 3: Searches DuckDuckGo for discontinued products."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.images(term, max_results=max_images))
            return [r['image'] for r in results]
    except Exception:
        return []

# ==========================================
# 3. MAIN PIPELINE
# ==========================================
def run_hybrid_pipeline():
    print("🚀 INITIALIZING ULTIMATE DATA PIPELINE")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(KB_PATH)
    df = df.dropna(subset=['Brand', 'Name'])
    link_col = 'Product_Link' if 'Product_Link' in df.columns else 'URL'
    
    tasks = []
    for brand in df['Brand'].unique():
        brand_data = df[df['Brand'] == brand].head(MAX_PRODUCTS_PER_BRAND)
        for _, row in brand_data.iterrows():
            tasks.append({
                'brand': str(row['Brand']).strip(),
                'name': str(row['Name']).strip(),
                'url': row.get(link_col, None)
            })

    print(f"📋 Loaded {len(tasks)} products to process.")
    fallback_queue = []
    
    # --- PHASE 1 & 2: PORTAL PHARMACY (Direct Link & Search) ---
    print("\n" + "="*40)
    print("🥇 PHASE 1 & 2: Scraping Portal Pharmacy")
    print("="*40)
    
    for i, task in enumerate(tasks):
        brand = task['brand']
        clean_name = clean_product_name(task['name'])
        brand_dir = OUTPUT_DIR / brand
        brand_dir.mkdir(exist_ok=True)
        
        file_prefix = clean_name.replace(' ', '_')
        if list(brand_dir.glob(f"{file_prefix}*.jpg")):
            continue
            
        print(f"[{i+1}/{len(tasks)}] Processing: {task['name']}")
        
        success = False
        img_url = extract_portal_image(task['url']) # Try direct link
        
        if not img_url:
            print("   🔍 No link/Broken. Searching Portal Pharmacy internally...")
            img_url = search_portal_for_image(task['name']) # Try internal search
            
        if img_url:
            save_path = brand_dir / f"{file_prefix}_portal.jpg"
            success = download_single_image(img_url, save_path)
            
        if success:
            print("   ✅ Image secured from Portal Pharmacy.")
            time.sleep(1.0) 
        else:
            print("   ⚠️ Not found anywhere on site. Adding to DuckDuckGo fallback.")
            fallback_queue.append(task)
            
    # --- PHASE 3: DUCKDUCKGO FALLBACK ---
    if fallback_queue:
        print("\n" + "="*40)
        print(f"🥈 PHASE 3: DuckDuckGo Fallback Activated ({len(fallback_queue)} items)")
        print("="*40)
        
        for i, task in enumerate(fallback_queue):
            brand = task['brand']
            clean_name = clean_product_name(task['name'])
            brand_dir = OUTPUT_DIR / brand
            
            search_query = f"{brand} {clean_name} {SEARCH_SUFFIX}".strip()
            print(f"[{i+1}/{len(fallback_queue)}] 🌐 Searching Web: {search_query}")
            
            urls = search_ddg_images(search_query, max_images=IMAGES_PER_PRODUCT_FALLBACK)
            if not urls:
                print("   ❌ No images found online.")
                continue
                
            try:
                download_images(brand_dir, urls=urls)
                time.sleep(2.0) 
            except Exception as e:
                print(f"   ❌ Download Error: {e}")
                time.sleep(5.0)

    # --- PHASE 4: CLEANUP ---
    print("\n" + "="*40)
    print("🧹 PHASE 4: Verifying Images")
    failed = verify_images(get_image_files(OUTPUT_DIR))
    failed.map(Path.unlink)
    print(f"🎉 Done! Deleted {len(failed)} broken images.")

if __name__ == "__main__":
    run_hybrid_pipeline()