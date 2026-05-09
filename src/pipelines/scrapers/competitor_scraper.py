"""
competitor_scraper.py
=====================
Scrapes competitor pharmacy websites for product pricing using Selenium + BeautifulSoup.

Navigates product listing pages via a headless Chrome browser, extracts product
names and prices, and saves results for price benchmarking.

Input:  none (competitor URLs hardcoded inside script)
Output: competitor price CSV saved to data/01_raw/ (path set inside script)

Run manually for quarterly competitor price benchmarking.
Requires Chrome + ChromeDriver (managed via webdriver_manager).
"""

import random
import re
import time
import urllib.parse
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager


# ==========================================
# CONFIGURATION
# ==========================================
# Pointing directly to your main dataset
TARGET_DIR = Path(
    "G:\\My Drive\\Portal_ML\\Portal_ML_V4\\data (1)\\01_raw (1)\\"
    "Brand_Images_TopDown"
)
TARGET_MINIMUM = 20

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}


# ==========================================
# HELPER FUNCTIONS
# ==========================================
def normalize_text(text):
    """Cleans text for perfect matching."""
    if not isinstance(text, str):
        return ""
    text = text.lower().replace("'", "").replace("’", "")
    text = re.sub(r'[^a-z0-9]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def get_soup_selenium(url):
    """Uses a real (invisible) Chrome browser to render JS before parsing."""
    try:
        # Polite delay before opening browser
        time.sleep(random.uniform(1.5, 3.0))
        
        options = Options()
        options.add_argument("--headless=new")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument(f"user-agent={HEADERS['User-Agent']}")
        options.add_argument("--log-level=3") 
        
        driver = webdriver.Chrome(
            service=ChromeService(ChromeDriverManager().install()), 
            options=options
        )
        
        driver.get(url)
        time.sleep(3.5) 
        
        html = driver.page_source
        driver.quit()
        return BeautifulSoup(html, 'html.parser')
    except Exception as e:
        print(f"      ❌ Browser error: {e}")
    return None


def download_image(img_url, dest_path):
    """Streams the verified image to disk."""
    if dest_path.exists():
        return True
    try:
        response = requests.get(
            img_url, headers=HEADERS, stream=True, timeout=10
        )
        if response.status_code == 200:
            with open(dest_path, "wb") as file:
                for chunk in response.iter_content(1024):
                    if chunk:
                        file.write(chunk)
            return True
    except Exception:
        pass
    return False


def get_high_res_woo_image(url):
    """Strips WooCommerce dimensions (Goodlife)."""
    if not url: 
        return ""
    return re.sub(
        r'-\d+x\d+(?=\.(jpg|jpeg|png|webp))', '', url, flags=re.IGNORECASE
    )


def format_woo_slug(text):
    """Formats brand names for WooCommerce URLs."""
    if not isinstance(text, str): 
        return ""
    text = text.lower().replace("'", "").replace("’", "")
    text = re.sub(r'[^a-z0-9]+', '-', text)
    return text.strip('-')


# ==========================================
# COMPETITOR SCRAPERS
# ==========================================
def scrape_pharmaplus(brand, needed_count, target_dir, starting_index):
    """Scrapes Pharmaplus using their Next.js structure."""
    secured_count = 0
    norm_target_brand = normalize_text(brand)
    
    brand_slug = urllib.parse.quote(brand.replace(" ", "-"))
    url = f"https://shop.pharmaplus.co.ke/products/brand/{brand_slug}"
    
    soup = get_soup_selenium(url)
    if not soup:
        return secured_count

    images = soup.find_all(
        "img", src=re.compile(r"api\.pharmaplus\.co\.ke/images/|/_next/image")
    )
    
    for img_tag in images:
        if secured_count >= needed_count: 
            break
            
        product_name = img_tag.get("alt", "")
        norm_prod_name = normalize_text(product_name)
        
        if not product_name or norm_target_brand not in norm_prod_name:
            continue
            
        img_url = img_tag.get("src")
        if img_url and img_url.startswith("/_next/image"):
            parsed_url = urllib.parse.urlparse(img_url)
            qs = urllib.parse.parse_qs(parsed_url.query)
            if 'url' in qs:
                img_url = qs['url'][0]
                if img_url.startswith("/"): 
                    img_url = "https://shop.pharmaplus.co.ke" + img_url

        if img_url:
            clean_brand = brand.replace(" ", "_").replace("/", "_")
            file_name = (
                f"{clean_brand}_pharmaplus_{starting_index + secured_count + 1}.jpg"
            )
            if download_image(img_url, target_dir / file_name):
                secured_count += 1
                
    return secured_count


def scrape_goodlife(brand, needed_count, target_dir, starting_index):
    """Scrapes Goodlife Pharmacy WooCommerce structure."""
    secured_count = 0
    norm_target_brand = normalize_text(brand)
    
    brand_slug = format_woo_slug(brand)
    url = f"https://www.goodlife.co.ke/brands/{brand_slug}/"
    
    soup = get_soup_selenium(url)
    if not soup:
        return secured_count

    products = soup.find_all("li", class_=re.compile(r"product\b"))
    
    for product in products:
        if secured_count >= needed_count: 
            break
            
        title_tag = product.find("h2", class_="woocommerce-loop-product__title")
        if not title_tag: 
            continue
        product_name = title_tag.get_text(strip=True)
        if norm_target_brand not in normalize_text(product_name):
            continue
            
        img_tag = product.find("img", class_=re.compile(r"woocommerce_thumbnail"))
        if not img_tag: 
            continue
            
        raw_img_url = img_tag.get("data-src") or img_tag.get("src")
        high_res_url = get_high_res_woo_image(raw_img_url)
        
        if high_res_url:
            clean_brand = brand.replace(" ", "_").replace("/", "_")
            file_name = (
                f"{clean_brand}_goodlife_{starting_index + secured_count + 1}.jpg"
            )
            if download_image(high_res_url, target_dir / file_name):
                secured_count += 1
                
    return secured_count


# ==========================================
# CASCADING PIPELINE MANAGER
# ==========================================
def fill_image_gaps():
    print("🚀 INITIALIZING PRODUCTION GAP-FILLER...")
    
    if not TARGET_DIR.exists():
        print("❌ Error: Target directory does not exist.")
        return

    # Iterates directly over your existing TopDown folders
    brand_folders = sorted([f for f in TARGET_DIR.iterdir() if f.is_dir()])
    
    for brand_dir in brand_folders:
        brand_name = brand_dir.name
        
        current_count = len(list(brand_dir.glob("*.jpg")))
        images_needed = TARGET_MINIMUM - current_count
        
        if images_needed > 0:
            print(
                f"\n📂 [{brand_name}] Has {current_count}/{TARGET_MINIMUM}. "
                f"Hunting for {images_needed} more..."
            )
            
            print("   ↳ 🛒 Rendering Pharmaplus (Waiting for JS)...")
            secured = scrape_pharmaplus(
                brand_name, images_needed, brand_dir, current_count
            )
            images_needed -= secured
            if secured > 0: 
                print(f"      ✅ Secured {secured} from Pharmaplus.")
            
            if images_needed > 0:
                print("   ↳ 🛒 Rendering Goodlife (Waiting for JS)...")
                secured_gl = scrape_goodlife(
                    brand_name, images_needed, brand_dir, current_count + secured
                )
                if secured_gl > 0: 
                    print(f"      ✅ Secured {secured_gl} from Goodlife.")

    print("\n🎉 PRODUCTION CASCADING PIPELINE COMPLETE.")


if __name__ == "__main__":
    fill_image_gaps()