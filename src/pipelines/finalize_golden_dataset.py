"""
finalize_golden_dataset.py
==========================
One-time utility: cleans mojibake encoding artefacts in the product Knowledge Base.

Reads the manually-verified Golden_Product_Catalog_Verified.csv, fixes
cp1252/utf-8 encoding errors (curly quotes, accented characters, em-dashes),
and writes the clean final Knowledge Base file used by all pipeline modules.

Input:  data/01_raw/Golden_Product_Catalog_Verified.csv
Output: data/01_raw/Final_Knowledge_Base_PowerBI.csv

Run manually whenever a new KB version is imported.
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path

# --- CONFIGURATION ---
BASE_DIR = Path("C:\\Users\\Portal Pharmacy\\Documents\\Portal ML Analys\\Ron's Work (2)\\Portal_ML\\Portal_ML_V4\\data\\01_raw")
INPUT_FILE = BASE_DIR / "Golden_Product_Catalog_Verified.csv"
OUTPUT_FILE = BASE_DIR / "Final_Knowledge_Base_PowerBI.csv"

# ✅ 1. MOJIBAKE CLEANER
def clean_mojibake(text):
    if not isinstance(text, str): return text
    try:
        return text.encode('cp1252').decode('utf-8')
    except:
        replacements = {
            "â€™": "'", "Ã©": "é", "Ã¨": "è", "Ã±": "ñ", "Ã¢": "â",
            "â€“": "-", "â€”": "-", "Â": "", "Ã": "í"
        }
        for bad, good in replacements.items():
            text = text.replace(bad, good)
        return text

# 2. CANONICAL CATEGORY (STRICT PRIORITY ORDER)
# ⚠️ Logic: Specific Categories (Women, Lip, Medicine) MUST check before Generic (Baby, Skin)
DEPARTMENT_TO_CANONICAL = {
    
    # --- 1. WOMEN'S HEALTH (Highest Priority) ---
    r"PREGNANCY": "Product Inquiry - Women's Health", # Catches Palmer's Pregnancy
    r"PREGNANT": "Product Inquiry - Women's Health",
    r"STRETCH MARK": "Product Inquiry - Women's Health",
    r"MATERNITY": "Product Inquiry - Women's Health",
    r"FEMININE": "Product Inquiry - Women's Health",
    r"TAMPONS": "Product Inquiry - Women's Health",
    r"INTIMATE WASH": "Product Inquiry - Women's Health",
    r"PANTY LINERS": "Product Inquiry - Women's Health",
    r"VAGINAL": "Product Inquiry - Women's Health",
    r"PADS": "Product Inquiry - Women's Health",
    r"MAXI.*PADS": "Product Inquiry - Women's Health",
    r"\bwellwoman\b": "Product Inquiry - Women's Health",
    r"\bwell woman\b": "Product Inquiry - Women's Health",
    r"\bfor women\b": "Product Inquiry - Women's Health",
    r"\bfor ladies\b": "Product Inquiry - Women's Health",
    r"\bwomen'?s\b": "Product Inquiry - Women's Health",
    r"\bwomens\b": "Product Inquiry - Women's Health",
    r"\bpant(y|ie)\s*liner(s)?\b": "Product Inquiry - Women's Health",
    r"\btampon(s)?\b": "Product Inquiry - Women's Health",
    r"\bmenstrual (cup|cups|disc|discs)\b": "Product Inquiry - Women's Health",
    r"\bperiod (pain|cramps?|cramping)\b": "Product Inquiry - Women's Health",
    r"\bpant(y|ie)\s*liner(s)?\b": "Product Inquiry - Women's Health",
    r"\btampon(s)?\b": "Product Inquiry - Women's Health",
    r"\bmenstrual (cup|cups|disc|discs)\b": "Product Inquiry - Women's Health",
    r"\bperiod (pain|cramps?|cramping)\b": "Product Inquiry - Women's Health",

    # --- 2. LIP CARE (High Priority) ---
    r"LIP BALM": "Product Inquiry - Lip Care",
    r"LIPBALM": "Product Inquiry - Lip Care",
    r"LIP CARE": "Product Inquiry - Lip Care",
    r"LIP THERAPY": "Product Inquiry - Lip Care",
    r"\bLIPS\b": "Product Inquiry - Lip Care", 
    r"VASELINE LIP": "Product Inquiry - Lip Care",
    r"CARMEX": "Product Inquiry - Lip Care",
    r"BLISTEX": "Product Inquiry - Lip Care",
    r"Lipstick": "Product Inquiry - Lip Care",

    # --- 3. MEDICINE ---
    r"ANTIBIOTIC": "Product Inquiry - Medicine",
    r"ANTIFUNGAL": "Product Inquiry - Medicine",
    r"\bPPI\b": "Product Inquiry - Medicine",
    r"PAIN RELIEF": "Product Inquiry - Medicine",
    r"ASPIRIN": "Product Inquiry - Medicine",
    r"HEADACHE": "Product Inquiry - Medicine",
    r"MIGRAINE": "Product Inquiry - Medicine",
    r"ACID REFLUX": "Product Inquiry - Medicine",
    r"HEARTBURN": "Product Inquiry - Medicine",
    r"INDIGESTION": "Product Inquiry - Medicine",
    r"LOZENGE": "Product Inquiry - Medicine",
    r"SORE THROAT": "Product Inquiry - Medicine",
    r"\bCOUGH\b": "Product Inquiry - Medicine",
    r"\bFLU\b": "Product Inquiry - Medicine",
    r"\bCOLD\b": "Product Inquiry - Medicine",
    r"JOINT CARE": "Product Inquiry - Medicine",
    r"DIGESTIVE": "Product Inquiry - Medicine",
    r"MINOXIDIL": "Product Inquiry - Medicine",
    r"PRESCRIPTION": "Product Inquiry - Medicine",
    r"HYPERTENSION": "Product Inquiry - Medicine",
    r"MEDICINE": "Product Inquiry - Medicine",
    r"\bOTC\b": "Product Inquiry - Medicine",
    r"SCABIES": "Product Inquiry - Medicine",
    r"BENZYL BENZOATE": "Product Inquiry - Medicine",
    r"ALKA SELTZER": "Product Inquiry - Medicine",
    r"STREPSILS": "Product Inquiry - Medicine",
    r"BENYLIN": "Product Inquiry - Medicine",
    r"GENERAL PRESCRIPTION": "Product Inquiry - Medicine",
    r"GENERAL PRESCRIPTIONS": "Product Inquiry - Medicine",
    r"O\.T\.C": "Product Inquiry - Medicine",
    r"PPI": "Product Inquiry - Medicine",
    r"VACCINES": "Product Inquiry - Medicine",
    r"EAR NOSE THROAT": "Product Inquiry - Medicine",

    # --- 4. ORAL CARE ---
    r"ORAL": "Product Inquiry - Oral Care",
    r"TOOTH": "Product Inquiry - Oral Care",
    r"DENTAL": "Product Inquiry - Oral Care",
    r"MOUTHWASH": "Product Inquiry - Oral Care",
    r"GUM": "Product Inquiry - Oral Care",
    r"BONJELA": "Product Inquiry - Oral Care",
    r"SENSODYNE": "Product Inquiry - Oral Care",
    r"AQUAFRESH": "Product Inquiry - Oral Care",


    # --- 5. FIRST AID ---
    r"FIRST AID": "Product Inquiry - First Aid",
    r"ANTISEPTIC": "Product Inquiry - First Aid",
    r"WOUND": "Product Inquiry - First Aid",
    r"HEALING GEL": "Product Inquiry - First Aid",
    r"BURN": "Product Inquiry - First Aid",
    r"PLASTER": "Product Inquiry - First Aid",
    r"SAVLON": "Product Inquiry - First Aid",
    r"SPIRIT": "Product Inquiry - First Aid",
    r"GAUZE": "Product Inquiry - First Aid",

    # --- 6. MEDICAL DEVICES ---
    r"MEDICAL DEVICES": "Product Inquiry - Medical Devices and Kits",
    r"BLOOD PRESSURE": "Product Inquiry - Medical Devices and Kits",
    r"THERMOMETER": "Product Inquiry - Medical Devices and Kits",
    r"BANDAGES": "Product Inquiry - Medical Devices and Kits",
    r"TEST STRIP": "Product Inquiry - Medical Devices and Kits",
    r"PREGNANCY TEST": "Product Inquiry - Medical Devices and Kits",
    r"HCG": "Product Inquiry - Medical Devices and Kits",
    r"GLUCOMETER": "Product Inquiry - Medical Devices and Kits",

    # --- 7. MEN CARE ---
    r"FOR HIM": "Product Inquiry - Men Care",
    r"SHAVING": "Product Inquiry - Men Care",
    r"GILLETTE": "Product Inquiry - Men Care",
    r"BUMP PATROL": "Product Inquiry - Men Care",
    r"\bMEN'S\b": "Product Inquiry - Men Care",

    # --- 8. BABY CARE (Specifics Only - TRAP REMOVED) ---
    r"DIAPERS": "Product Inquiry - Baby Care",
    r"PAMPERS": "Product Inquiry - Baby Care",
    r"HUGGIES": "Product Inquiry - Baby Care",
    r"WIPES": "Product Inquiry - Baby Care",
    r"INFACOL": "Product Inquiry - Baby Care",
    r"INFANT FORMULA": "Product Inquiry - Baby Care",
    r"MILK FORMULA": "Product Inquiry - Baby Care",
    r"\bBABY\b": "Product Inquiry - Baby Care", 
    r"\bINFANT\b": "Product Inquiry - Baby Care",
    r"NAPPY": "Product Inquiry - Baby Care",
    r"PACIFIER": "Product Inquiry - Baby Care",
    

    # --- 9. SUPPLEMENTS ---
    r"SUPPLEMENTS": "Product Inquiry - Supplements",
    r"VITAMIN": "Product Inquiry - Supplements",
    r"ZINC": "Product Inquiry - Supplements",
    r"MULTIVITAMIN": "Product Inquiry - Supplements",
    r"OMEGA": "Product Inquiry - Supplements",
    r"FISH OIL": "Product Inquiry - Supplements",
    r"PROBIOTICS": "Product Inquiry - Supplements",
    r"\bIRON\b": "Product Inquiry - Supplements",
    r"CALCIUM": "Product Inquiry - Supplements",
    r"SEVEN SEAS": "Product Inquiry - Supplements",
    r"BEROCCA": "Product Inquiry - Supplements",
    r"VITABIOTICS": "Product Inquiry - Supplements",
    r"IRON": "Product Inquiry - Supplements",


    # --- 11. HAIRCARE ---
    r"HAIR": "Product Inquiry - Haircare",
    r"CANTU": "Product Inquiry - Haircare",
    r"SHEA MOISTURE": "Product Inquiry - Haircare",
    r"HAIR CARE": "Product Inquiry - Haircare",
    r"DANDRUFF": "Product Inquiry - Haircare",
    r"MIELLE": "Product Inquiry - Haircare",
    r"MIZANI": "Product Inquiry - Haircare",
    r"REGAINE": "Product Inquiry - Haircare",
    r"AFRICAN PRIDE": "Product Inquiry - Haircare",
    r"SHAMPOO": "Product Inquiry - Haircare",
    r"CONDITIONER": "Product Inquiry - Haircare",
    r"HAIR TREATMENT": "Product Inquiry - Haircare",
    r"HAIR OIL": "Product Inquiry - Haircare",
    r"Detangler": "Product Inquiry - Haircare",

    # --- 12. SKINCARE (Catch All) ---
    r"SKIN": "Product Inquiry - Skincare",
    r"COSMETIC": "Product Inquiry - Skincare",
    r"TOPICAL": "Product Inquiry - Skincare",
    r"BODY SHOP": "Product Inquiry - Skincare",
    r"BATH & BODY": "Product Inquiry - Skincare",
    r"THE ORDINARY": "Product Inquiry - Skincare",
    r"KOREAN SKIN": "Product Inquiry - Skincare",
    r"La ROCHE-POSAY": "Product Inquiry - Skincare",
    r"NEUTROGENA": "Product Inquiry - Skincare",
    r"CETAPHIL": "Product Inquiry - Skincare",
    r"EUCERIN": "Product Inquiry - Skincare",
    r"AVENE": "Product Inquiry - Skincare",
    r"VICHY": "Product Inquiry - Skincare",
    r"Bioderma": "Product Inquiry - Skincare",
    r"CLINIQUE": "Product Inquiry - Skincare",
    r"OLAY": "Product Inquiry - Skincare",
    r"NIVEA": "Product Inquiry - Skincare",
    r"SEBAMED": "Product Inquiry - Skincare",
    r"CERAVE": "Product Inquiry - Skincare",
    r"ACNE": "Product Inquiry - Skincare",
    r"DR ORGANIC": "Product Inquiry - Skincare",
    r"SUNSCREEN": "Product Inquiry - Skincare",
    r"SUN BLOCK": "Product Inquiry - Skincare",
    
    # --- 10. HOMEOPATHY ---
    r"HOMEOPATHY": "Product Inquiry - Homeopathy",
    r"HOMEOPATHIC": "Product Inquiry - Homeopathy",
    r"ESSENTIAL OIL": "Product Inquiry - Homeopathy",
    r"AROMATHERAPY": "Product Inquiry - Homeopathy",
    r"TEA TREE": "Product Inquiry - Homeopathy",
    r"EUCALYPTUS": "Product Inquiry - Homeopathy",
    r"LAVENDER OIL": "Product Inquiry - Homeopathy",
    r"PEPPERMINT OIL": "Product Inquiry - Homeopathy",
    r"ROSEMARY OIL": "Product Inquiry - Homeopathy",
    r"CASTOR OIL": "Product Inquiry - Homeopathy",
    r"NEEM": "Product Inquiry - Homeopathy",
    r"RESCUE REMEDY": "Product Inquiry - Homeopathy",
    r"BACH FLOWER": "Product Inquiry - Homeopathy",

}

SUB_CAT_RULES = {
    'Styling': ['mousse', 'spray', 'wax', 'pomade', 'edge control', 'curl', 'brylcreem', 'styling'],
    'Shampoo': ['shampoo', 'wash', 'cleanser'],
    'Conditioner': ['conditioner', 'leave-in', 'rinse-out', 'detangler', 'deep conditioner', 'detangling'],
    'Hair Treatment': ['hair mask', 'hair oil', 'hair food','repair'],
    'Sunscreen': ['sunscreen', 'spf', 'sunblock', 'sun protection', 'uv'],
    'Serum & Treatment': ['serum', 'concentrate', 'ampoule', 'spot treatment', 'peel', 'acid', 'retinol'],
    'Cleanser': ['cleanser', 'face wash', 'facewash', 'micellar', 'soap', 'scrub', 'exfoliating', 'makeup remover'],
    'Mask': ['mask', 'sheet mask', 'clay', 'mud'],
    'Eye Care': ['eye cream', 'eye gel', 'eye serum', 'under eye'],
    'Toner': ['toner', 'astringent', 'essence', 'mist'],
    'Moisturizer': ['moisturizer', 'cream', 'lotion', 'balm', 'hydrator', 'emollient'],
    'Antibiotics': ['antibiotic', 'augmentin', 'amoxyl', 'cipro', 'doxy', 'azithromycin'],
    'Pain Relief': ['pain', 'panadol', 'paracetamol', 'ibuprofen', 'diclofenac', 'aspirin', 'deep heat', 'rub'],
    'Cold & Flu': ['cold', 'flu', 'cough', 'syrup', 'sinus', 'throat', 'lozenge'],
    'Digestive': ['gaviscon', 'eno', 'tums', 'acid', 'digest', 'laxative', 'antacid'],
    'Allergy': ['allergy', 'antihistamine', 'cetrizine', 'loratadine', 'piriton'],
    'First Aid': ['plaster', 'bandage', 'antiseptic', 'iodine', 'spirit', 'gauze'],
    'Diapers': ['diaper', 'pampers', 'huggies', 'pants', 'nappy'],
    'Wipes': ['wipes', 'tissue', 'wet wipe'],
    'Vitamins': ['vitamin', 'multivitamin', 'zinc', 'iron', 'calcium', 'magnesium', 'biotin'],
    'Fish Oil': ['omega', 'fish oil', 'cod liver', 'seven seas'],
}

CONCERN_RULES = {
    "Acne": { "all": [r"\bacne\b", r"\bacnes\b", r"\bpimple(s)?\b", r"\bbreakout(s)?\b", 
                      r"\bwhitehead(s)?\b", r"\bblackhead(s)?\b", r"\bbenzoyl peroxide?\b", 
                      r"\bsalicylic?\b", r"\beffaclar?\b"] },
    "Hyperpigmentation": { "all": [r"\bhyperpigmentation\b", r"\bdark spot(s)?\b", 
                                   r"\bdark mark(s)?\b", r"\buneven (skin )?tone\b",
                                     r"\bbrightening\b", r"\bascorbic acid\b", 
                                     r"\bspot corrector\b", r"\bpigment\b"] },
    "Oily Skin": { "all": [r"\boily skin\b", r"\bfor oily\b", r"\bcontrols oil\b", 
                           r"\bsebum\b", r"\blarge pores\b", r"\bpores\b", r"\boily\b", 
                           r"\bcombination to oily\b"] },
    "Dry Skin": { "all": [r"\bdry skin\b", r"\bfor dry\b", r"\bextra dry\b", 
                          r"\bvery dry\b", r"\bintense hydration\b", r"\bmy skin is dry\b",
                          r"\bdry\s+(?:and\s+)?(?:damaged\s+)?skin\b", r"\bdryness\b", 
                          r"\bdehydrated\b", r"\bcracked\b", r"\bchapped\b"] },
    "Sensitive Skin": { "all": [r"\bsensitive skin\b", r"\bfor sensitive\b", 
                                r"\banti-irritation\b", r"\bsoothing\b"] },
    "Sleep": { "all": [r"\bsleep\b", r"\bfor sleep\b", r"\binsomnia\b", r"\brestless night(s)?\b", 
                       r"\bmagnesium glycinate\b"] },
    "Hair Loss": { "all": [r"\bhair loss\b", r"\banti-hairloss\b", r"\bthinning hair\b", 
                           r"\bminoxidil\b", r"\balopecia\b"] },
    "Weight Management": { "all": [r"\bweight loss\b", r"\bweight gain\b", 
                                   r"\blose weight\b", r"\bgain weight\b", 
                                   r"\bslimming\b", r"\bcut belly\b", r"\btummy trimmer\b", 
                                   r"\bappetite\b", r"\badd weight\b", r"\breduce weight\b", r"\bfat burner\b"] },
    "Eczema": { "all": [r"\beczema\b", r"\batopic dermatitis\b", r"\bdermatitis\b", 
                        r"\bitchy skin\b", r"\bred patches\b"] },
    "Stretch Marks": { "all": [r"\bstretch mark(s)?\b", r"\bscars?\b"] },
}

ALL_WEBSITE_BRANDS = [
        "La Roche-Posay", "Advanced Clinicals", "Bath & Bodyworks", "Beauty Of Joseon", 
        "Clean And Clear", "Dark And Lovely", "Oxygen Botanicals", "Shea Moisture", 
        "Summer Fridays", "Victoria Secret", "African Pride", "Beauty Formula", 
        "Johnson's Baby", "Animal Parade", "Dr Organic", "Haliborange", 
        "Seven Seas", "The Ordinary", "Paulas Choice", "Reedle Shot", "Neutrogena", 
        "Heliocare", "Bioderma", "Aquaphor", "Vaseline", "Cetaphil", "Oatveen", 
        "Oilatum", "Sebamed", "Eucerin", "Palmers", "Regaine", "Garnier", "Blistex", 
        "Body Shop", "Burts Bees", "Laneige", "Sephora", "Topicals", "Bennets", 
        "Panoxyl", "St Ives", "Uncover", "Berocca", "Centrum", "Forever", "CeraVe", 
        "Aveeno", "Cantu", "Miadi", "Vichy", "Nivea", "Byoma", "Fenty", "Gisou", 
        "Anua", "Dove", "Fino", "Pixi", "Nyx", "E 45", "Olay", "Sinoz", "Acnes", 
        "Avent", "Radox", "Mielle", "Mizani", "Carmex", "Simple", "Cosrx", 
        "Avene", "Loreal", "Epimax", "Motions", "Biretix", "Medicube", "Naturium", 
        "Johnson", "Eos", "Health Aid", "Alka Seltzer", "Scholl", "Scabees", "Bonjela",
        "Bump Patrol", "Strepsils", "Benylin", "Sensodyne", "Halls", "Infacol", "Aquafresh", "Axe Oil"
    ]

def extract_concerns(text):
    if not isinstance(text, str): return ""
    found = set()
    text_lower = text.lower()
    for concern, rules in CONCERN_RULES.items():
        for pattern in rules['all']:
            if re.search(pattern, text_lower, re.IGNORECASE):
                found.add(concern)
                break
    return ", ".join(sorted(list(found)))

def determine_target_audience(text):
    text_lower = str(text).lower()
    if re.search(r"\bbaby\b", text_lower) or re.search(r"\binfant\b", text_lower) or re.search(r"\bnewborn\b", text_lower):
        return 'Baby & Mom'
    elif re.search(r"\bkid\b", text_lower) or re.search(r"\bchild", text_lower) or re.search(r"\bjunior\b", text_lower):
        return 'Kids'
    elif re.search(r"\bmen\b", text_lower) or re.search(r"\bman\b", text_lower) or re.search(r"\bhomme\b", text_lower) or re.search(r"\bhim\b", text_lower) or re.search(r"\bmale\b", text_lower):
        return 'Men'
    return 'Adult (General)'

def enrich_product(row):
    # Clean Encoding
    name = clean_mojibake(str(row.get('Name', '')))
    desc = clean_mojibake(str(row.get('Detailed_Desc', '')))
    
    full_text = f"{name} {desc}".upper()
    name_lower = name.lower()
    
    # 1. CATEGORY
    final_cat = "Other"
    for pattern, intent_name in DEPARTMENT_TO_CANONICAL.items():
        if re.search(pattern, full_text):
            final_cat = intent_name
            break
            
    # 2. SUB-CATEGORY
    final_sub = "General"
    for sub_name, keywords in SUB_CAT_RULES.items():
        if any(k in name_lower for k in keywords):
            final_sub = sub_name
            break

    # 3. BRAND
    found_brand = "General"
    sorted_brands = sorted(ALL_WEBSITE_BRANDS, key=len, reverse=True)
    for brand in sorted_brands:
        if brand.lower() in name_lower:
            found_brand = brand
            break

    # 4. CONCERNS
    existing = str(row.get('Concerns', ''))
    if existing in ['nan', 'General Inquiry', '', 'General', 'General Care']: existing = ""
    new_found = extract_concerns(f"{name} {desc}")
    
    combined = set()
    if existing: combined.update([c.strip() for c in existing.split(',')])
    if new_found: combined.update([c.strip() for c in new_found.split(',')])
    final_concerns = ", ".join(sorted(list(combined))) if combined else "General Care"

    # 5. AUDIENCE
    target_audience = determine_target_audience(f"{name} {row.get('Category 1','')}")

    return pd.Series([name, found_brand, final_cat, final_sub, final_concerns, target_audience])

def run_enrichment():
    print("✨ Starting Final Polish (V12 - REMOVED 'FORMULA' TRAP & DELETED EMPTY NAMES)...")
    try:
        df = pd.read_csv(INPUT_FILE)
        print(f"   📂 Loaded: {len(df)} products")
        
        # ✅ STEP 1: DELETE ITEMS WITHOUT A NAME
        initial_len = len(df)
        df = df[df['Name'].notna() & (df['Name'].str.strip() != "")]
        print(f"   🗑️ Dropped {initial_len - len(df)} rows with missing names.")
        
        # Apply enrichment
        df[['Name', 'Brand', 'Canonical_Category', 'Sub_Category', 'Concerns', 'Target_Audience']] = df.apply(enrich_product, axis=1)
        
        cols = ['ItemCode', 'Name', 'Brand', 'Canonical_Category', 'Sub_Category', 
                'Concerns', 'Target_Audience', 
                'Price', 'Quantity', 'Product_Link', 'Detailed_Desc']
        
        cols = [c for c in cols if c in df.columns]
        df[cols].to_csv(OUTPUT_FILE, index=False)
        print(f"   ✅ Saved: {OUTPUT_FILE}")
        
        # AUDIT
        print("\n   🔎 Audit 'Formula' Products (Should NOT be Baby Care):")
        audit = df[df['Detailed_Desc'].str.contains("Formula", case=False, na=False)].head(5)
        for _, row in audit.iterrows():
            print(f"      {row['Name']} -> {row['Canonical_Category']}")

    except Exception as e:
        print(f"   ❌ Error: {e}")

if __name__ == "__main__":
    run_enrichment()