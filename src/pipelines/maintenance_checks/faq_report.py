import pandas as pd
import re
import os
import spacy
from collections import Counter
from pathlib import Path

# V3 PRODUCTION IMPORTS
from Portal_ML_V4.src.config.settings import (
    FINAL_TAGGED_DATA, PROCESSED_DATA_DIR, BASE_DIR
)
from Portal_ML_V4.src.config.brands import BRAND_LIST

# Load NLP Model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    print("⚠️ Spacy model not found. Run: python -m spacy download en_core_web_sm")
    exit()

# ==========================================
# 1. CONFIGURATION
# ==========================================

FAQ_PATTERNS = {
    "Location Inquiry": [
        r"where.*located", r"location", r"branch", r"shop.*where", r"visit.*shop",
        r"physical address", r"directions", r"near.*", r"town", "cbd", "westlands"
    ],
    "Delivery Inquiry": [
        r"deliver", r"shipping", r"send.*to", r"rider", r"parcel", 
        r"how much.*transport", r"delivery fee", r"cost.*delivery", 
        r"outside nairobi", r"countrywide"
    ],
    "Payment Method": [
        r"paybill", r"account no", r"till number", r"mpesa", r"how.*pay", 
        r"payment method", r"credit card", r"cash on delivery", "pay on delivery"
    ],
    "Business Hours": [
        r"open", r"close", r"working hours", r"time.*open", r"are you open", 
        r"sunday", r"weekend", r"holiday"
    ],
    "Contact Info": [
        r"phone number", r"call me", r"whatsapp", r"contact", r"speak.*agent"
    ],
    "Medical Advice": [
        r"side effect", r"safe.*pregnancy", r"breastfeeding", r"dosage", 
        r"how to use", r"reaction", r"sensitive skin", r"routine", r"consultation"
    ],
    "Discount/Price": [
        r"discount", r"offer", r"price", r"how much", r"cost", r"sale", r"less"
    ]
}

# ✅ 1. TYPO & SLANG MAP
TYPO_MAP = {
    "spping": "shipping",
    "ts": "this",
    "wch": "which",
    "thks": "thanks",
    "avlbl": "available",
    "rmation": "information",
    "ur": "your",
    "mah": "my",
    "xmas": "christmas",
    "don t": "dont",
    "we don t": "we dont",
    "i don t": "i dont",
    "they don t": "they dont"
}

# ✅ 2. LOCATIONS / COMPETITORS / COURIERS (Pre-Scrub)
# These are stripped from the sentence BEFORE NLP processing.
ENTITIES_TO_STRIP = [
    # Competitors / Landmarks
    "goodlife pharmacy", "goodlife", "mydawa", "textbook centre", "textbook center",
    "village market", "jamie mosque", "jamia mosque", "linton", "lintons",
    "south c", "limuru road", "narok line", "bus station", "kibagare way",
    "restaurant", "resturant", "parcel office", "super metro", "kilimanjaro",
    
    # Internal Locations
    "two rivers", "ruaka", "abc place", "waiyaki way",
    "galleria mall", "galleria", "langata road",
    "milele mall", "ngong road", "cbd", "banda street", "banda st",
    "ngong", "milele", "ground floor", "town branch", "westlands", "nairobi", 
    "portal place", "centurion"
]

# ✅ 3. CONFIRMATIONS
CONFIRMATION_PHRASES = [
    "payment received", "received with thanks", "payment well received", 
    "transaction confirmed", "well received", "confirmed payment", 
    "payment confirmed", "money received", "thank you payment",
    "new m-pesa balance", "confirmed. ksh", "transaction cost", "mobile pos app",
    "equity loop", "new m", "total bill", "amount inclusive", "inclusive account"
]

# ✅ 4. PAYBILLS
PAYBILL_NUMBERS = [
    "247247", "666226", "662226", "217004", "666222", "552800", "222666"
]

# ✅ 5. MASTER BLACKLIST (Phrases to Discard)
BLACKLIST_PHRASES = [
    # System / Logs / Competitors
    "product price test", "messenger call request", "partnered online doctors", 
    "limited pdf", "limited search", "music_info", "utm_campaign", "autoplay",
    "image_url", "cdn chatapi", "ext png size", "route product search", 
    "index php", "null price", "image url", "api", "json", "undefined",
    "cart select checkout", "mydawa",
    
    # Artifacts & Fragments
    "image file", "image whatsapp", "video attachment", "image pastedimage", 
    "pastedimage", "limited jpg", "jpg", "delivery jpg", "paybill jpg",
    "and contact", "limited contact", "phone conversation", "contact details",
    "s number", "s way", "s you", "don t", "we don t", "they don t", 
    "m pesa", "m-pesa", "mpesa mpesa",
    
    # Generic / Bot Filler
    "portal pharmacy", "pharmacy limited", "contacting portal pharmacy",
    "kindly let", "let know", "make payment", "good morning", "good afternoon", "good evening",
    "customer excited", "looking forward", "payment received", "delivery fee",
    "paybill account", "equity paybill", "account number", "image attachment",
    "audio file", "details paybill", "message unavailable", "unavailable right",
    "respond soon", "right respond", "thank you", "thanks", "hello", "hi",
    "delivery details", "delivery options", "delivery location", "delivery address",
    "inclusive delivery", "total inclusive", "frequent paybills", "confirmation message",
    "same paybill", "paybill acc", "no one", "you guys", "no worries", "no problem", "this message",
    "more information", "more info", "further assistance", "any questions", "any problems",
    "what time", "total amount", "quick payment", "bill payment", "mysafaricom app",
    "mpesa app", "mpesa message", "till number", "same phone number", "phone number",
    "contact number", "whatsapp number", "delivery charges", "exact location",
    "shop located", "closest branch", "several convenient locations", "delivery services",
    "countrywide deliveries", "best delivery option", "location details", "physical shop",
    "location address", "payment method", "mpesa details", "no amount", "amount delivery",
    "delivery", "ksh", "paybill", "abc", "branch", "location", "share name", 
    "details name", "same name", "share contact", "phone call", "no response", 
    "previous messages", "last time", "tomorrow morning", "online orders", 
    "website orders", "available kes", "cash or mpesa", "cash mpesa", "yes total", 
    "perfect authenticity", "net stickers", 
    "digestive enzyme", "one tablet", "limited time offer", "colour contact",
    
    # Marketing / Holidays
    "happy holidays", "great weekend", "lovely weekend", "nice weekend", "brand day",
    "christmas bundles", "limited-time offer", "exclusive discounts",
    "free delivery", "free shipping", "cool dry place", "merry xmas",
    "active man", "blessed sunday", "happy new year",
    
    # Redundant Product/Order
    "which product", "the product", "a product", "the products", "which one",
    "your order", "an order", "the order", "make purchase", "a purchase",
    "doorstep", "reply", "riders", "different brand",
    "individual products", "best price", "how much ksh",
    "how much kshs", "lower product price", "delayed response", "last price",
    "no discount", "best results", "this one", "this link", "available",
    "paybill", "delivery", "location", "branch", "contact", "price", "discount",
    "fargo",
    

    # Store locations (Explicit Blacklist)
    "two rivers", "ruaka", "abc place", "waiyaki way",
    "galleria mall", "galleria", "langata road",
    "milele mall", "ngong road", "cbd", "banda street", "banda st",
    "ngong", "milele", "jamia mosque", "ground floor", "textbook center",
    "shopping mall", "town branch", "westlands", "nairobi", "banda st",
    "langata", "ruaka branch", "two rivers mall", "abc place mall", "mall",
    "abc branch", "town centre", "same location", "other branches", 
    "physical location", "town", "r u", "branch location", "centurion", 
    "pharmart", "portal"
]

# Stop words to strip from the START of phrases
PREFIX_STOP_WORDS = [
    "the", "a", "an", "my", "your", "our", "this", "that", "any", "all",
    "which", "what", "its", "some", "such", "wch", "ts", "these", "other",
    "their", "her", "his", "few", "ok", "okay", "s", "t", "m", "we", "they", "i"
]

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def clean_sentence(text):
    """Deep scrub of the text before NLP."""
    text = str(text).lower()
    
    # 0. STRICT ASCII
    text = text.encode('ascii', 'ignore').decode()
    
    # 1. KILL API / TECH NOISE (Regex)
    text = re.sub(r'image_url.*', '', text) # Kill JSON dumps
    text = re.sub(r'route\sproduct.*', '', text) # Kill URL fragments
    text = re.sub(r'music_info=[\w=&]+', '', text)
    text = re.sub(r'utm_[\w=&]+', '', text)
    text = re.sub(r'[\w]+\.php', '', text) # .php files
    text = re.sub(r'[\w]+\.png', '', text) # .png files
    
    # 2. FIX TYPOS
    for bad, good in TYPO_MAP.items():
        text = re.sub(rf"\b{bad}\b", good, text)

    # 3. Kill URLs
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'www\S+', '', text)
    
    # 4. Kill Locations, Paybills, Confirmations
    for lst in [PAYBILL_NUMBERS, ENTITIES_TO_STRIP, CONFIRMATION_PHRASES]:
        for item in lst:
            text = text.replace(item, "")

    # 5. Kill Blacklist Phrases (Pre-emptive)
    for phrase in BLACKLIST_PHRASES:
        text = text.replace(phrase, "")
        
    # 6. Kill Brands
    for b in BRAND_LIST:
        if b.lower() in text:
            text = text.replace(b.lower(), "")

    text = re.sub(r'[0-9]', '', text) 
    text = re.sub(r'\^im', '', text) 
    text = re.sub(r'\[.*?\]', '', text)
    
    return text.strip()

def clean_noun_chunk(chunk_text):
    # 1. STRIP PUNCTUATION & SYMBOLS
    chunk_text = chunk_text.strip(" ,|-[].:;?/_")
    
    tokens = chunk_text.split()
    if not tokens: return ""
    
    # 2. HEADLESS RULE & FRAGMENT KILLER
    while tokens and (tokens[0] in PREFIX_STOP_WORDS or len(tokens[0]) < 2):
        tokens = tokens[1:]
        
    cleaned = " ".join(tokens).strip()
    
    # 3. LENGTH CHECK (Strict 2-6 words)
    word_count = len(cleaned.split())
    if word_count < 2 or word_count > 6:
        return ""
        
    # 4. TECH DETECTOR
    if any(x in cleaned for x in ["null", "api", "cdn", "http", "www", "image_url", "php", "route"]):
        return ""
    
    # 5. BLACKLIST CHECK
    if cleaned in BLACKLIST_PHRASES:
        return ""
        
    return cleaned

def extract_key_phrases(text_list):
    """
    Extracts phrases using nlp.pipe to prevent MemoryError.
    """
    phrases = []
    
    # We disable 'ner' (Named Entity Recognition) because noun_chunks only needs the parser/tagger.
    # This reduces memory usage significantly.
    for doc in nlp.pipe(text_list, batch_size=50, disable=["ner", "lemmatizer"]):
        
        for chunk in doc.noun_chunks:
            raw_chunk = chunk.text.lower().strip()
            
            # Skip pronouns (e.g., "it", "she", "they")
            if chunk.root.pos_ == "PRON": 
                continue
            
            clean_chunk = clean_noun_chunk(raw_chunk)
            if not clean_chunk: 
                continue
            
            phrases.append(clean_chunk)
            
    return phrases

def extract_relevant_sentences(full_text, patterns):
    if not isinstance(full_text, str): return []
    sentences = re.split(r'[.!?\n]+', full_text)
    relevant = []
    for sent in sentences:
        sent_clean = sent.lower().strip()
        if any(re.search(p, sent_clean) for p in patterns):
            cleaned = clean_sentence(sent_clean)
            if len(cleaned) > 10:
                relevant.append(cleaned)
    return relevant

# ==========================================
# 3. MAIN LOGIC
# ==========================================
def run_faq_analysis():
    print("🗣️  RUNNING FAQ ANALYTICS (V17.0 - MASTER BLACKLIST)...")
    
    input_path = PROCESSED_DATA_DIR / "fact_sessions_enriched.csv"
    if not os.path.exists(input_path):
        print(f"   ❌ Enriched data not found at: {input_path}")
        return

    df = pd.read_csv(input_path)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    category_corpus = {cat: [] for cat in FAQ_PATTERNS.keys()}
    category_corpus["Uncategorized"] = []
    
    # STEP 1: SUMMARY STATS
    faq_stats_rows = []
    for _, row in df.iterrows():
        raw_text = str(row.get('full_context', '')).lower()
        matched_categories = []
        for category, patterns in FAQ_PATTERNS.items():
            relevant_sentences = extract_relevant_sentences(raw_text, patterns)
            if relevant_sentences:
                matched_categories.append(category)
                category_corpus[category].extend(relevant_sentences)
        
        if not matched_categories:
            clean_general = clean_sentence(raw_text)
            if len(clean_general) > 10:
                matched_categories.append("Uncategorized")
                category_corpus["Uncategorized"].append(clean_general)
                
        for cat in matched_categories:
            faq_stats_rows.append({"category": cat, "is_converted": row.get('is_converted', 0)})

    if faq_stats_rows:
        df_faq = pd.DataFrame(faq_stats_rows)
        df_summary = df_faq.groupby('category').agg(
            total_inquiries=('category', 'count'),
            total_conversions=('is_converted', 'sum')
        ).reset_index()
        df_summary['conversion_rate'] = (df_summary['total_conversions'] / df_summary['total_inquiries']).round(4)
        df_summary = df_summary.sort_values('total_inquiries', ascending=False)
        df_summary.to_csv(PROCESSED_DATA_DIR / "dim_faq_summary.csv", index=False)
        print("   ✅ Exported FAQ Summary.")

    # STEP 2: DEDUPLICATION & EXPORT
    print("   🧠 Extracting & Cleaning Phrases...")
    
    all_phrase_occurrences = []
    
    for category, texts in category_corpus.items():
        if len(texts) < 5: continue
        phrases = extract_key_phrases(texts)
        for p in phrases:
            all_phrase_occurrences.append({"Category": category, "Phrase": p})
            
    df_phrases = pd.DataFrame(all_phrase_occurrences)
    
    if not df_phrases.empty:
        df_counts = df_phrases.groupby(['Phrase', 'Category']).size().reset_index(name='Frequency')
        df_deduped = df_counts.sort_values('Frequency', ascending=False).drop_duplicates('Phrase', keep='first')
        
        final_rows = []
        for category in FAQ_PATTERNS.keys():
            cat_data = df_deduped[df_deduped['Category'] == category]
            top_15 = cat_data.head(15)
            for i, row in enumerate(top_15.itertuples()):
                final_rows.append({
                    "Category": row.Category,
                    "Rank": i + 1,
                    "Phrase": row.Phrase,
                    "Frequency": row.Frequency
                })
        
        # Uncategorized
        cat_data = df_deduped[df_deduped['Category'] == "Uncategorized"]
        top_15 = cat_data.head(15)
        for i, row in enumerate(top_15.itertuples()):
            final_rows.append({
                "Category": "Uncategorized",
                "Rank": i + 1,
                "Phrase": row.Phrase,
                "Frequency": row.Frequency
            })

        df_breakdown = pd.DataFrame(final_rows)
        output_path = PROCESSED_DATA_DIR / "fact_faq_breakdown.csv"
        df_breakdown.to_csv(output_path, index=False)
        print(f"   ✅ Exported Deduplicated Breakdown to {output_path}")

    print("✅ FAQ ANALYSIS COMPLETE.")

if __name__ == "__main__":
    run_faq_analysis()