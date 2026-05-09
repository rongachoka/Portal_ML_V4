"""
brand_scraper.py
================
Scrapes product prices and availability for brands stocked at Portal Pharmacy.

For each brand in the Knowledge Base, sends HTTP requests to the brand's
website or a pharmacy price comparison site, extracts product names and
prices, and saves results for KB enrichment.

Input:  data/01_raw/Final_Knowledge_Base_PowerBI.csv  (brand list source)
Output: scraped price CSV saved to data/01_raw/ (path set inside script)

Run manually when updating the KB with competitor or own-store pricing.
Respects rate limits with random delays between requests.
"""

import random
import re
import time
import urllib.parse
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

from Portal_ML_V4.src.config.settings import KB_PATH


# ==========================================
# CONFIGURATION
# ==========================================
# KB_PATH = Path(
#     "G:\\My Drive\\Portal_ML\\Portal_ML_V4\\data (1)\\01_raw (1)\\"
#     "Final_Knowledge_Base_PowerBI (1).csv"
# )
OUTPUT_DIR = Path(
    "G:\\My Drive\\Portal_ML\\img_classification\\Brand_Images_NEW"
)

MAX_IMAGES_PER_BRAND = 30
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}


# ==========================================
# HELPER FUNCTIONS
# ==========================================
def get_soup(url):
    """Fetches a URL and returns a BeautifulSoup object with a polite delay."""
    try:
        sleep_time = random.uniform(1.0, 2.5)
        time.sleep(sleep_time)
        response = requests.get(url, headers=HEADERS, timeout=15)
        if response.status_code == 200:
            return BeautifulSoup(response.text, 'html.parser')
    except Exception:
        pass
    return None


def normalize_text(text):
    """
    Cleans text by lowercasing, completely destroying apostrophes, 
    and converting all other special characters to spaces.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # 1. Completely remove apostrophes so "Bennett's" becomes "bennetts"
    text = text.replace("'", "").replace("’", "") 
    # 2. Replace all other non-alphanumeric characters with a single space
    text = re.sub(r'[^a-z0-9]', ' ', text)
    # 3. Strip extra whitespace
    return re.sub(r'\s+', ' ', text).strip()


def get_high_res_url(thumb_url):
    """Returns the URL of the image."""
    if not thumb_url:
        return ""
    return thumb_url


def download_image(img_url, dest_path):
    """Streams the image to disk."""
    if dest_path.exists():
        return True
    try:
        # Note: Not adding a random sleep here because we want to grab the 
        # grid images quickly once we are already on the page.
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


# ==========================================
# MAIN SCRAPER
# ==========================================
def scrape_brands_top_down():
    print("🚀 INITIALIZING VERIFIED TOP-DOWN SCRAPER")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("📂 Loading Knowledge Base...")
    df = pd.read_csv(KB_PATH)
    df = df.dropna(subset=["Brand"])
    unique_brands = sorted(df["Brand"].astype(str).unique())
    
    print(f"🧠 Found {len(unique_brands)} unique brands.")
    missed_brands = []

    for i, brand in enumerate(unique_brands):
        brand = brand.strip()
        if not brand:
            continue

        brand_dir = OUTPUT_DIR / brand
        brand_dir.mkdir(exist_ok=True)

        print(f"[{i+1}/{len(unique_brands)}] 🔍 Searching Portal: {brand}")
        # norm_target_brand = normalize_text(brand)
        
        encoded_brand = urllib.parse.quote(brand)
        search_url = (
            f"https://portalpharmacy.ke/index.php?"
            f"route=product/search&search={encoded_brand}"
        )

        # Use our new human-like soup function
        soup = get_soup(search_url)
        
        if not soup:
            print("   ❌ Error loading page or 404.")
            missed_brands.append(brand)
            continue

        products = soup.find_all("div", class_="product-layout")
        if not products:
            products = soup.find_all("div", class_="product-thumb")

        if not products:
            print("   ⚠️ No products found. Adding to missed list.")
            missed_brands.append(brand)
            continue

        download_count = 0
        for index, product in enumerate(products):
            if download_count >= MAX_IMAGES_PER_BRAND:
                break
            
            # 1. Extract the product name
            product_name = ""
            name_div = product.find("div", class_="name")
            if name_div and name_div.find("a"):
                product_name = name_div.find("a").get_text(strip=True)
            else:
                h4_tag = product.find("h4")
                if h4_tag and h4_tag.find("a"):
                    product_name = h4_tag.find("a").get_text(strip=True)

            # 2. Verify the brand is in the product name
            # norm_prod_name = normalize_text(product_name)
            
            # if norm_target_brand not in norm_prod_name:
            #     continue

            brand_pattern = r'(?<!\w)' + re.escape(brand) + r'(?!\w)'
            if not re.search(brand_pattern, product_name, re.IGNORECASE):
                continue
            
            # 3. Secure the image
            img_tag = product.find("img", class_="img-responsive")
            if not img_tag:
                continue

            img_url = img_tag.get("data-src") or img_tag.get("src")
            img_url = get_high_res_url(img_url)

            if img_url:
                clean_brand = brand.replace(" ", "_").replace("/", "_")
                file_name = f"{clean_brand}_{download_count}.jpg"
                save_path = brand_dir / file_name
                
                if download_image(img_url, save_path):
                    download_count += 1

        if download_count > 0:
            print(f"   ✅ Secured {download_count} verified images.")
        else:
            print("   ⚠️ Found items, but none matched the brand name.")
            missed_brands.append(brand)

    # 4. Handle the Missed Brands
    print("\n" + "=" * 40)
    print("📊 SCRAPE COMPLETE!")
    print(f"✅ Processed {len(unique_brands) - len(missed_brands)} brands.")
    print(f"⚠️ Missed {len(missed_brands)} brands.")
    
    if missed_brands:
        missed_df = pd.DataFrame({"Missed_Brands": missed_brands})
        missed_path = OUTPUT_DIR / "missed_brands_for_fallback.csv"
        missed_df.to_csv(missed_path, index=False)
        print(f"📁 Saved missed brands list to: {missed_path}")


if __name__ == "__main__":
    scrape_brands_top_down()