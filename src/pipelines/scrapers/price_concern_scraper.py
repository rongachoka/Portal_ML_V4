import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import re
from pathlib import Path

# --- CONFIGURATION ---
BASE_URL = "https://portalpharmacy.ke"
OUTPUT_FILE = Path("C:\\Users\\Portal Pharmacy\\Documents\\Portal ML Analys\\Ron's Work (2)\\Portal_ML\\Portal_ML_V4\\data\\01_raw\\scraped_prices_jan2026.csv")
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)


# ✅ MANUAL MASTER LIST
URLS_TO_SCRAPE = [
    "https://portalpharmacy.ke/skin-care-products",
    "https://portalpharmacy.ke/feminine-care",
    "https://portalpharmacy.ke/lipcare",
    "https://portalpharmacy.ke/medicine-treatment",
    "https://portalpharmacy.ke/men-care",
    "https://portalpharmacy.ke/oral-care",
    "https://portalpharmacy.ke/haircare-products",
    "https://portalpharmacy.ke/medical-devices-kits",
    "https://portalpharmacy.ke/supplements",
    "https://portalpharmacy.ke/first-aid",
    "https://portalpharmacy.ke/homeopathy",
    "https://portalpharmacy.ke/babycare",
]

# --- CONCERN RULES ---
CONCERN_RULES = {
    "Acne": { "all": [r"\bacne\b", r"\bacnes\b", r"\bpimple(s)?\b", r"\bbreakout(s)?\b", r"\bwhitehead(s)?\b", r"\bblackhead(s)?\b", r"\bbenzoyl peroxide?\b", r"\bsalicylic?\b", r"\beffaclar?\b"] },
    "Hyperpigmentation": { "all": [r"\bhyperpigmentation\b", r"\bdark spot(s)?\b", r"\bdark mark(s)?\b", r"\buneven (skin )?tone\b", r"\bbrightening\b", r"\bascorbic acid\b", r"\bspot corrector\b", r"\bpigment\b"] },
    "Oily Skin": { "all": [r"\boily skin\b", r"\bfor oily\b", r"\bcontrols oil\b", r"\bsebum\b", r"\blarge pores\b", r"\bpores\b", r"\boily\b", r"\bcombination to oily\b"] },
    "Dry Skin": { "all": [r"\bdry skin\b", r"\bfor dry\b", r"\bextra dry\b", r"\bvery dry\b", r"\bintense hydration\b", r"\bmy skin is dry\b"] },
    "Sensitive Skin": { "all": [r"\bsensitive skin\b", r"\bfor sensitive\b", r"\banti-irritation\b", r"\bsoothing\b"] },
    "Sleep": { "all": [r"\bsleep\b", r"\bfor sleep\b", r"\binsomnia\b", r"\brestless night(s)?\b", r"\bmagnesium glycinate\b"] },
    "Hair Loss": { "all": [r"\bhair loss\b", r"\banti-hairloss\b", r"\bthinning hair\b", r"\bminoxidil\b", r"\balopecia\b"] },
    "Weight Management": { "all": [r"\bweight loss\b", r"\bweight gain\b", r"\blose weight\b", r"\bgain weight\b", r"\bslimming\b", r"\bcut belly\b", r"\btummy trimmer\b", r"\bappetite\b", r"\badd weight\b", r"\breduce weight\b", r"\bfat burner\b"] },
    "Eczema": { "all": [r"\beczema\b", r"\batopic dermatitis\b", r"\bdermatitis\b", r"\bitchy skin\b", r"\bred patches\b"] },
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

def get_soup(url):
    try:
        sleep_time = random.uniform(1.0, 2.5)
        time.sleep(sleep_time)
        response = requests.get(url, headers=HEADERS, timeout=15)
        if response.status_code == 200:
            return BeautifulSoup(response.text, 'html.parser')
    except Exception:
        pass
    return None

def extract_concerns(full_text):
    if not isinstance(full_text, str) or not full_text: return ""
    found_concerns = set()
    text_lower = full_text.lower()
    for concern_name, rules in CONCERN_RULES.items():
        all_patterns = []
        for key in ['all', 'product', 'chat']:
            all_patterns.extend(rules.get(key, []))
        for pattern in all_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                found_concerns.add(concern_name)
                break
    return ", ".join(sorted(list(found_concerns)))

def scrape_catalog():
    print(f"🕷️  Starting Scraper (Full Text Capture)...")
    all_products = []

    for i, base_category_url in enumerate(URLS_TO_SCRAPE):
        cat_name = base_category_url.split('/')[-1].replace('-', ' ').title()
        print(f"\n[{i+1}/{len(URLS_TO_SCRAPE)}] 📂 {cat_name}")
        
        page = 1
        has_next = True
        items_in_cat = 0
        
        while has_next:
            separator = "&" if "?" in base_category_url else "?"
            url = f"{base_category_url}{separator}page={page}"
            
            print(f"   Reading Page {page}... (Total: {items_in_cat} items found)   ", end="\r", flush=True)

            soup = get_soup(url)
            if not soup: break

            items = soup.find_all('div', class_='product-layout')
            if not items: items = soup.find_all('div', class_='product-thumb')
            
            if not items: break

            count_new = 0
            for item in items:
                try:
                    # 1. NAME
                    name_tag = item.find('h4').find('a') if item.find('h4') else item.select_one('div.name a')
                    name = name_tag.get_text(strip=True) if name_tag else "Unknown"
                    link = name_tag['href'] if name_tag else ""

                    # 2. RICH DESCRIPTION (Full Capture)
                    desc_block = item.find('div', class_='block-content')
                    
                    if not desc_block:
                        desc_block = item.find('div', class_='caption')
                    
                    if desc_block:
                        # ✅ CHANGE 1: Use '\n' to keep paragraphs separate
                        desc_text = desc_block.get_text("\n", strip=True)
                    else:
                        desc_text = ""

                    # 3. CONCERNS
                    full_context = f"{name} {desc_text}"
                    concerns = extract_concerns(full_context)

                    # 4. PRICE
                    price_tag = item.find('span', class_='price-normal')
                    if not price_tag: price_tag = item.find('span', class_='price-new')
                    if not price_tag: 
                        price_div = item.find('p', class_='price')
                        raw_price = price_div.get_text(strip=True).split('\n')[0] if price_div else "0"
                    else:
                        raw_price = price_tag.get_text(strip=True)

                    clean_price = re.sub(r'[^\d.]', '', raw_price)

                    if name != "Unknown":
                        all_products.append({
                            "Website_Name": name,
                            "Website_Price": clean_price,
                            "Product_URL": link,
                            "Concerns_Identified": concerns,
                            "Source_Category": cat_name,
                            "Raw_Description": desc_text 
                        })
                        count_new += 1
                except: continue

            items_in_cat += count_new
            
            if count_new < 2: has_next = False
            else: page += 1
            
        print(f"   ✅ Finished {cat_name}: {items_in_cat} products scraped.      ")

    # SAVE
    df = pd.DataFrame(all_products)
    if not df.empty:
        df = df.drop_duplicates(subset=['Website_Name'], keep='first')
        print("\n" + "="*50)
        print(f"📊 FINAL REPORT")
        print(f"   Unique Products: {len(df)}")
        print(f"   'Acne' Products: {df['Concerns_Identified'].str.contains('Acne').sum()}")
        
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"   💾 Saved to: {OUTPUT_FILE}")
    else:
        print("\n❌ CRITICAL: No data scraped.")

if __name__ == "__main__":
    scrape_catalog()