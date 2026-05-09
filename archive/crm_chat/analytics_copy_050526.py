import os
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable=None, *args, **kwargs):
        return iterable

from Portal_ML_V4.src.config.settings import (
    FINAL_TAGGED_DATA,
    PROCESSED_DATA_DIR,
    CLEANED_DATA_DIR,
    BASE_DIR,
    MSG_HISTORY_RAW,
    META_ADS_DIR,
    KB_PATH,
)
from Portal_ML_V4.src.utils.phone import clean_id, normalize_phone, clean_id_excel_safe
from Portal_ML_V4.src.config.ad_name_map import AD_NAME_MAP
from Portal_ML_V4.src.config.brands import BRAND_ALIASES
from Portal_ML_V4.src.config.tag_rules import CANONICAL_CATEGORY_RULES, CONCERN_RULES
from Portal_ML_V4.src.config.concerns import CONCERN_KEYWORDS

ADS_OUTPUT_FILE = PROCESSED_DATA_DIR / "ads" / "all_ads_merged.csv"

INVALID_TEXT_VALUES = frozenset({"", "-", "nan", "None", "<NA>"})
INVALID_ASSIGNED_STAFF = {"bot", "system", "me", "you", "undefined"}
CONSULTATION_TAG_PATTERN = re.compile(r"dermatologist|skin consultation", re.IGNORECASE)
CONSULTATION_CONTEXT_PATTERN = re.compile(
    r"dermatologist|skin consultation|skin test",
    re.IGNORECASE,
)
CHANNEL_SOURCE_ORGANIC = frozenset({"incoming message", "echo message"})
SOURCE_RECOVERED_PAID = "ctc_ads"
SOURCE_RECOVERED_ORGANIC = frozenset({"contact", "user"})
TEAM_MAP = {
    "847526": "Ishmael",
    "860475": "Faith",
    "879396": "Nimmoh",
    "879430": "Rahab",
    "879438": "Brenda",
    "962460": "Katie",
    "1000558": "Sharon",
    "845968": "Joy",
    "1006108": "Jess",
    "971945": "Jeff",
}

ASSIGNED_STAFF_RE = re.compile(r"assigned to\s+([a-zA-Z0-9_]+)", re.IGNORECASE)
NEWLINE_RE = re.compile(r"\r\n|\r|\n")
WHITESPACE_RE = re.compile(r"[\xc2\xa0\t\n\r]+")
MULTISPACE_RE = re.compile(r"\s+")
NON_DIGIT_DOT_RE = re.compile(r"[^\d.]")
URL_SLUG_RE = re.compile(r"portalpharmacy\.ke/([\w-]+)", re.IGNORECASE)
HTTP_RE = re.compile(r"http\S+")
NON_ALNUM_SPACE_RE = re.compile(r"[^a-z0-9\s]")
GENERIC_STOPWORDS_RE = re.compile(
    r"\b(price|cost|how much|is|the|a|an|need|want|looking for)\b"
)
MPESA_NOISE_PATTERNS = [
    re.compile(r"(?:account|acc)\s*(?:no|number|num)?[:\.]?\s*\d+", re.IGNORECASE),
    re.compile(r"paybill\s*(?:no|number|num)?[:\.]?\s*\d+", re.IGNORECASE),
    re.compile(
        r"transaction\s*cost\s*[:.,-]?\s*(?:ksh|kes)?\.?\s*[\d,]+(?:\.\d{2})?",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:new)?\s*m-?pesa\s*balance\s*(?:is|:)?\s*(?:ksh|kes)?\.?\s*[\d,]+(?:\.\d{2})?",
        re.IGNORECASE,
    ),
    re.compile(
        r"amount\s*you\s*can\s*transact.*?(?:ksh|kes)?\.?\s*[\d,]+(?:\.\d{2})?",
        re.IGNORECASE,
    ),
]
MPESA_STD_RE = re.compile(
    r"\b([A-Z0-9]{10})\s+(?:then\s+)?Confirmed.*?(?:ksh|kes)\.?\s*([\d,\.\s]+)",
    re.IGNORECASE,
)
MPESA_BILL_RE = re.compile(
    r"Confirmed.*?Bill\s*Payment.*?(?:ksh|kes)\.?\s*([\d,\.\s]+).*?Ref\.?\s*[:\.]?\s*([A-Z0-9]{10})",
    re.IGNORECASE,
)

print(KB_PATH)
KB_DF = pd.DataFrame()
try:
    kb_df = pd.read_csv(KB_PATH)
    KB_DF = kb_df.copy()
    kb_df = kb_df.dropna(subset=["Brand"])
    DYNAMIC_BRAND_LIST = (
        kb_df["Brand"].astype(str).str.strip().str.title().unique().tolist()
    )
    BRAND_TO_CATEGORY_MAP = (
        kb_df.groupby("Brand")["Canonical_Category"]
        .agg(lambda x: x.mode()[0] if not x.mode().empty else "General Inquiry")
        .to_dict()
    )
except Exception as e:
    print(f"Warning: Could not load KB for dynamic brands: {e}")
    DYNAMIC_BRAND_LIST = []
    BRAND_TO_CATEGORY_MAP = {}
    KB_DF = pd.DataFrame()

CHANNEL_MAP = {
    389017: "WhatsApp",
    387986: "Instagram",
    388255: "Facebook",
    388267: "TikTok",
    389086: "Web Chat",
    492754: "WhatsApp",
}

MACRO_GROUP_MAP = {
    "Skincare": "Skincare",
    "Baby Care": "Baby Care",
    "Haircare": "Hair Care",
    "Oral Care": "Oral Care",
    "Supplements": "Supplements",
    "Medicine": "Medicine",
    "Medical Devices": "Medical Devices & Kits",
    "First Aid": "Medical Devices & Kits",
    "Homeopathy": "Homeopathy",
    "Men's Care": "Men Care",
    "Women's Health": "Women's Health",
    "Fragrance": "Perfumes",
    "Lip Care": "Lip Care",
    "Sexual Health": "Sexual Health",
    "Hair Care": "Hair Care",
    "Menscare": "Men Care",
    "Perfumes": "Perfumes",
}

BRAND_CONTEXT_RULES = {
    "APTAMIL": ["APTAMIL", "BABY", "MILK", "FORMULA", "INFANT", "LACTOGEN"],
    "LA ROCHE-POSAY": [
        "LA ROCHE",
        "LRP",
        "EFFACLAR",
        "CICAPLAST",
        "ANTHELIOS",
        "LIPIKAR",
        "POSAY",
    ],
    "CERAVE": ["CERAVE", "CERA VE", "SA CLEANSER", "HYDRATING", "FOAMING"],
    "THE ORDINARY": ["ORDINARY", "NIACINAMIDE", "RETINOL", "PEELING", "ACID", "TO"],
    "PANADOL": ["PANADOL", "PAIN", "HEADACHE", "FEVER"],
    "PAMPERS": ["PAMPERS", "DIAPER", "BABY", "NAPPY"],
    "HUGGIES": ["HUGGIES", "DIAPER", "BABY", "NAPPY"],
}

PSEUDO_BRANDS = ["effaclar", "zelaton", "acnes", "panadol"]

FORM_FACTORS = {
    "serum": ["cream", "wash", "cleanser", "oil", "spray", "tablet", "capsule"],
    "cream": ["serum", "wash", "cleanser", "oil", "spray", "tablet", "capsule"],
    "wash": ["serum", "cream", "oil", "spray", "tablet", "capsule"],
    "cleanser": ["serum", "cream", "oil", "spray", "tablet", "capsule"],
    "oil": ["serum", "cream", "wash", "cleanser", "tablet", "capsule"],
    "tablet": ["serum", "cream", "wash", "cleanser", "oil", "spray"],
    "capsule": ["serum", "cream", "wash", "cleanser", "oil", "spray"],
    "glycinate": ["oil", "spray", "cream", "gel"],
}

BLACKLIST_PHRASES = [
    "click on this",
    "on this link",
    "would like order",
    "would like purchase",
    "can get more",
    "more info",
    "how can assist",
    "where would like",
    "like us deliver",
    "get more info",
    "make purchase",
    "checking availability",
    "http",
    "www.",
    ".com",
    ".ke",
    "image attachment",
    "view and order",
]

JUNK_WORDS = [
    "hello",
    "hi",
    "hey",
    "how are you",
    "good morning",
    "good evening",
    "good afternoon",
    "morning",
    "evening",
    "afternoon",
    "tomorrow",
    "today",
    "tonight",
    "day",
    "night",
    "location",
    "located",
    "where",
    "branch",
    "shop",
    "visit",
    "delivery",
    "deliver",
    "shipping",
    "cost",
    "charge",
    "fee",
    "send",
    "pay",
    "payment",
    "mpesa",
    "till",
    "number",
    "code",
    "total",
    "available",
    "stock",
    "have",
    "do you have",
    "selling",
    "business",
    "learn more",
    "tell me",
    "ad",
    "advert",
    "info",
    "information",
    "thank you",
    "thanks",
    "welcome",
    "ok",
    "okay",
    "sawa",
    "fine",
    "yeah",
    "yes",
    "click",
    "link",
    "view",
    "order",
    "purchase",
    "buying",
    "help",
    "assist",
    "question",
    "inquiry",
    "product",
    "products",
    "item",
    "items",
    "naona",
    "hiyo",
    "ni",
    "kuna",
    "nataka",
    "please",
    "kindly",
    "which",
    "one",
    "seen",
    "saw",
    "from",
    "advert",
    "plus",
    "and",
    "with",
]

GENERIC_TERMS = [
    "sunscreen",
    "sun screen",
    "sunblock",
    "moisturizer",
    "cleanser",
    "wash",
    "soap",
    "lotion",
    "cream",
    "gel",
    "serum",
    "toner",
    "oil",
    "spray",
    "pills",
    "tablets",
    "medicine",
    "drugs",
    "capsules",
    "vitamins",
    "shampoo",
    "conditioner",
    "hair food",
    "treatment",
    "scrub",
    "mask",
    "spf",
    "spf50",
    "spf30",
]

GENERIC_TERMS_SET = set(GENERIC_TERMS)
PSEUDO_BRANDS_SET = set(PSEUDO_BRANDS)
BRAND_ALIAS_LOWER = {str(k).lower(): str(v).lower() for k, v in BRAND_ALIASES.items()}
JUNK_PATTERNS = [re.compile(rf"\b{re.escape(word)}\b") for word in JUNK_WORDS]
DYNAMIC_BRAND_LOOKUPS = [(brand, brand.lower()) for brand in DYNAMIC_BRAND_LIST]
CONCERN_PATTERNS = {
    concern: [re.compile(pat) for pat in patterns.get("all", []) + patterns.get("chat", [])]
    for concern, patterns in CONCERN_RULES.items()
}


def extract_assigned_staff(text):
    if pd.isna(text):
        return None
    matches = ASSIGNED_STAFF_RE.findall(str(text))
    if matches:
        found = matches[-1].title()
        if found.lower() in INVALID_ASSIGNED_STAFF:
            return None
        return found
    return None


def sanitize_text_for_csv_export(df: pd.DataFrame) -> pd.DataFrame:
    export_df = df.copy()
    text_cols = export_df.select_dtypes(include=["object", "string"]).columns
    for col in text_cols:
        mask = export_df[col].notna()
        export_df.loc[mask, col] = (
            export_df.loc[mask, col].astype(str).str.replace(NEWLINE_RE, " ", regex=True)
        )
    return export_df


def clean_scientific_id_series(series: pd.Series) -> pd.Series:
    series = series.astype("string").str.strip()
    series = series.where(~series.isin(["", "-", "nan", "None"]), pd.NA)
    numeric = pd.to_numeric(series, errors="coerce")
    cleaned = series.copy()
    numeric_mask = numeric.notna()
    if numeric_mask.any():
        cleaned.loc[numeric_mask] = numeric.loc[numeric_mask].astype("Int64").astype("string")
    cleaned = cleaned.str.replace(r"\.0$", "", regex=True)
    return cleaned


def recalculate_mpesa_amount(full_context, current_amt=0):
    raw_text = str(full_context).lower()
    clean_text = WHITESPACE_RE.sub(" ", raw_text)
    clean_text = MULTISPACE_RE.sub(" ", clean_text)

    try:
        current_amt = float(current_amt)
    except (TypeError, ValueError):
        current_amt = 0.0

    for pattern in MPESA_NOISE_PATTERNS:
        clean_text = pattern.sub(" [NOISE] ", clean_text)

    found_txns = {}

    for match in MPESA_STD_RE.finditer(clean_text):
        try:
            code = match.group(1).upper()
            amt_str = NON_DIGIT_DOT_RE.sub("", match.group(2))
            found_txns[code] = float(amt_str)
        except Exception:
            pass

    for match in MPESA_BILL_RE.finditer(clean_text):
        try:
            code = match.group(2).upper()
            amt_str = NON_DIGIT_DOT_RE.sub("", match.group(1))
            found_txns[code] = float(amt_str)
        except Exception:
            pass

    if found_txns:
        return sum(found_txns.values())
    return current_amt


def recalculate_mpesa_smart(row):
    return recalculate_mpesa_amount(
        row.get("full_context", ""),
        row.get("mpesa_amount", 0),
    )


class ProductMatcher:
    def __init__(self):
        path = KB_PATH
        if not os.path.exists(path):
            self.active = False
            return

        self.df = (KB_DF.copy() if not KB_DF.empty else pd.read_csv(path)).fillna("")
        self.df["search_text"] = (
            self.df["Brand"]
            + " "
            + self.df["Name"]
            + " "
            + self.df["Name"]
            + " "
            + self.df["Name"]
            + " "
            + self.df["Sub_Category"]
            + " "
            + self.df["Sub_Category"]
            + " "
            + self.df["Concerns"]
        ).str.lower()

        self.brand_names = self.df["Brand"].astype(str).str.strip().tolist()
        self.brand_names_clean = [brand.lower().replace("the ", "").strip() for brand in self.brand_names]
        self.product_names = self.df["Name"].astype(str).str.lower().tolist()
        self.match_sub_categories = self.df["Sub_Category"].astype(str).str.lower().tolist()
        self.product_word_sets = [set(name.split()) for name in self.product_names]
        self.has_baby_product = [
            any(token in name for token in ("baby", "kid", "pediatric"))
            for name in self.product_names
        ]
        self._clean_cache = {}
        self._match_cache = {}

        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 4),
            min_df=1,
            strip_accents="unicode",
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["search_text"])
        self.knn = NearestNeighbors(n_neighbors=50, metric="cosine").fit(self.tfidf_matrix)
        self.active = True
        print(f"Knowledge Base Loaded: {len(self.df)} products")

    def _extract_url_slugs(self, text):
        slugs = []
        matches = URL_SLUG_RE.findall(str(text))
        for match in matches:
            if "index.php" not in match and "search" not in match:
                slugs.append(match.replace("-", " "))
        return " ".join(slugs)

    def _clean_context_smart(self, text):
        raw_text = str(text)
        cached = self._clean_cache.get(raw_text)
        if cached is not None:
            return cached

        cleaned_text = raw_text.lower()
        url_context = self._extract_url_slugs(cleaned_text)
        cleaned_text = HTTP_RE.sub("", cleaned_text)
        for phrase in BLACKLIST_PHRASES:
            cleaned_text = cleaned_text.replace(phrase, "")
        cleaned_text = cleaned_text + " " + url_context
        for junk_pattern in JUNK_PATTERNS:
            cleaned_text = junk_pattern.sub("", cleaned_text)
        for bad, good in BRAND_ALIAS_LOWER.items():
            if bad in cleaned_text:
                cleaned_text = cleaned_text.replace(bad, good)
        cleaned_text = NON_ALNUM_SPACE_RE.sub("", cleaned_text).strip()
        self._clean_cache[raw_text] = cleaned_text
        return cleaned_text

    def get_best_match(self, context, mpesa_amount=0, brand_hint=None):
        if not self.active:
            return None

        try:
            cleaned_text = self._clean_context_smart(context)
            cache_key = (cleaned_text, brand_hint or "")
            cached_idx = self._match_cache.get(cache_key)
            if cached_idx is not None:
                return None if cached_idx < 0 else self.df.iloc[cached_idx]

            has_official_brand = bool(brand_hint and brand_hint != "Unknown")
            has_pseudo_brand = any(pseudo_brand in cleaned_text for pseudo_brand in PSEUDO_BRANDS_SET)

            if not has_official_brand and not has_pseudo_brand:
                check_text = GENERIC_STOPWORDS_RE.sub("", cleaned_text).strip()
                words = [word for word in check_text.split() if len(word) > 2]
                if words and all(word in GENERIC_TERMS_SET for word in words):
                    self._match_cache[cache_key] = -1
                    return None
                if len(check_text) < 4:
                    self._match_cache[cache_key] = -1
                    return None

            if not has_official_brand and not has_pseudo_brand:
                self._match_cache[cache_key] = -1
                return None

            query_text = cleaned_text
            if has_official_brand:
                query_text = f"{brand_hint} {brand_hint} {cleaned_text}"
            elif has_pseudo_brand:
                for pseudo_brand in PSEUDO_BRANDS:
                    if pseudo_brand in cleaned_text:
                        query_text = f"{pseudo_brand} {pseudo_brand} {cleaned_text}"
                        break

            if len(query_text) < 3:
                self._match_cache[cache_key] = -1
                return None

            query_vec = self.vectorizer.transform([query_text])
            distances, indexes = self.knn.kneighbors(query_vec)

            best_match_idx = None
            highest_score = 0.0
            cleaned_words = set(cleaned_text.split())
            brand_hint_clean = brand_hint.lower().replace("the ", "").strip() if has_official_brand else None
            has_child_context = any(token in cleaned_text for token in ["baby", "kid", "child", "pediatric", "born"])
            context_upper = cleaned_text.upper()

            for position in range(len(indexes[0])):
                match_idx = indexes[0][position]
                similarity = 1 - distances[0][position]
                brand_name_clean = self.brand_names_clean[match_idx]
                match_sub = self.match_sub_categories[match_idx]
                product_name = self.product_names[match_idx]

                if has_official_brand:
                    if brand_hint_clean not in brand_name_clean and brand_name_clean not in brand_hint_clean:
                        continue

                if self.has_baby_product[match_idx] and not has_child_context:
                    continue

                for anchor, bad_forms in FORM_FACTORS.items():
                    if anchor in cleaned_text and any(bad_form in product_name or bad_form in match_sub for bad_form in bad_forms):
                        similarity -= 0.35

                if cleaned_words & self.product_word_sets[match_idx]:
                    similarity += 0.10

                if similarity > highest_score:
                    highest_score = similarity
                    best_match_idx = match_idx

            threshold = 0.55 if has_official_brand else 0.65
            if best_match_idx is not None and highest_score > threshold:
                predicted_brand_upper = self.brand_names[best_match_idx].upper()
                if predicted_brand_upper in BRAND_CONTEXT_RULES:
                    required_keywords = BRAND_CONTEXT_RULES[predicted_brand_upper]
                    if not any(keyword in context_upper for keyword in required_keywords):
                        self._match_cache[cache_key] = -1
                        return None

                self._match_cache[cache_key] = best_match_idx
                return self.df.iloc[best_match_idx]

            self._match_cache[cache_key] = -1
            return None
        except Exception:
            return None


def find_id_col(df):
    candidates = ["Contact ID", "ContactID", "ID", "id", "contact_id"]
    return next((candidate for candidate in candidates if candidate in df.columns), None)


def split_tags_logic(ts):
    if not isinstance(ts, str):
        return "", "", "", ""
    raw = [tag.strip() for tag in ts.split("|") if tag.strip()]
    primary_zone, secondary_zone, funnel, other = [], [], [], []
    for tag in raw:
        tag_lower = tag.lower()
        if "secondary zone:" in tag_lower:
            secondary_zone.append(re.sub(r"secondary zone:\s*", "", tag, flags=re.I).strip())
        elif "zone:" in tag_lower:
            primary_zone.append(re.sub(r"zone:\s*", "", tag, flags=re.I).strip())
        elif any(term in tag_lower for term in ["price", "payment", "converted"]):
            funnel.append(re.sub(r"(funnel|concern):\s*", "", tag, flags=re.I).strip())
        else:
            other.append(tag.strip())
    return " | ".join(primary_zone), " | ".join(secondary_zone), " | ".join(funnel), " | ".join(other)


def extract_meta(row):
    tags = str(row.get("final_tags", "")).lower()
    context = str(row.get("full_context", "")).lower()
    stock = tags.count("stock")
    recommendation = tags.count("recommendation")
    is_consult = int(
        "dermatologist" in tags
        or "skin consultation" in tags
        or "dermatologist" in context
        or "skin consultation" in context
        or "skin test" in context
    )
    found = [brand for brand in DYNAMIC_BRAND_LIST if brand.lower() in tags]
    return stock, is_consult, recommendation, " | ".join(found) if found else ""


def normalize_brands_with_intent(brand_str, full_text):
    found_brands = set()
    if brand_str and brand_str != "Unknown":
        raw_list = [brand.strip().lower() for brand in str(brand_str).split("|") if brand.strip()]
        for raw in raw_list:
            matched = False
            for typo, clean in BRAND_ALIASES.items():
                if typo in raw:
                    found_brands.add(clean)
                    matched = True
                    break
            if not matched:
                found_brands.add(raw.title())

    text_lower = str(full_text).lower()
    for typo, clean in BRAND_ALIASES.items():
        if re.search(r"\b" + re.escape(typo) + r"\b", text_lower):
            found_brands.add(clean)

    scored_brands = []
    buy_words = ["buy", "order", "price", "cost", "much", "link", "recommend", "need"]
    context_words = ["using", "use", "used", "currently", "have", "routine"]

    for brand in found_brands:
        score = 0
        brand_clean = brand.lower()
        if brand_clean.replace(" ", "-") in text_lower:
            score += 5
        matches = [match.start() for match in re.finditer(re.escape(brand_clean), text_lower)]
        for match_pos in matches:
            start = max(0, match_pos - 50)
            end = min(len(text_lower), match_pos + 50)
            window = text_lower[start:end]
            if any(word in window for word in buy_words):
                score += 3
            if any(word in window for word in context_words):
                score -= 2
        scored_brands.append((brand, score))

    scored_brands.sort(key=lambda item: item[1], reverse=True)
    return [brand for brand, _ in scored_brands]


def extract_dynamic_brands_from_tags(tags: str) -> str:
    tags = str(tags)
    found = [brand for brand, brand_lower in DYNAMIC_BRAND_LOOKUPS if brand_lower in tags]
    return " | ".join(found) if found else ""


def clean_scientific_id(val):
    if pd.isna(val) or str(val).strip() in ("", "-", "nan", "None"):
        return pd.NA
    try:
        return str(int(float(str(val).strip())))
    except ValueError:
        return str(val).strip().replace(".0", "")


@lru_cache(maxsize=1)
def load_meta_ad_name_map() -> dict:
    if not Path(META_ADS_DIR).exists():
        print(f"Meta ads file not found at {META_ADS_DIR}")
        return {}

    try:
        df_meta = pd.read_csv(
            META_ADS_DIR,
            dtype=str,
            usecols=lambda column: column.strip() in {"Ad ID", "Ad name"},
        )
        df_meta.columns = df_meta.columns.str.strip()
        df_meta["Ad ID"] = clean_scientific_id_series(df_meta["Ad ID"])
        df_meta = df_meta.dropna(subset=["Ad ID"])
        name_map = df_meta.set_index("Ad ID")["Ad name"].to_dict()
        print(f"Meta ad name map built: {len(name_map):,} ads")
        return name_map
    except Exception as e:
        print(f"Could not build Meta ad name map: {e}")
        return {}


def load_ads_for_analytics():
    ads_dir = Path(MSG_HISTORY_RAW).parent / "ads"
    all_files = sorted(ads_dir.glob("contacts-*.csv"))
    if not all_files:
        return pd.DataFrame(), set(), set()

    dataframes = []
    columns = ["Timestamp", "Contact ID", "Source", "Ad campaign ID", "Ad group ID", "Ad ID"]
    for file_path in all_files:
        try:
            frame = pd.read_csv(
                file_path,
                dtype=str,
                keep_default_na=False,
                usecols=lambda column: column in columns,
            )
            frame["Timestamp"] = pd.to_datetime(frame["Timestamp"], errors="coerce", format="mixed")
            frame["Contact ID"] = pd.to_numeric(
                frame["Contact ID"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True),
                errors="coerce",
            ).astype("Int64")
            dataframes.append(frame.dropna(subset=["Contact ID", "Timestamp"]))
        except Exception as e:
            print(f"Could not read {Path(file_path).name}: {e}")

    if not dataframes:
        return pd.DataFrame(), set(), set()

    df = pd.concat(dataframes, ignore_index=True).sort_values("Timestamp", ascending=True)

    for column in ["Ad campaign ID", "Ad group ID", "Ad ID"]:
        if column in df.columns:
            df[column] = clean_scientific_id_series(df[column])

    df = df.drop_duplicates(subset=["Contact ID", "Timestamp", "Ad campaign ID", "Ad ID"])

    ad_name_map = load_meta_ad_name_map()
    if "Ad ID" in df.columns:
        df["Ad Name"] = df["Ad ID"].map(ad_name_map)
        all_cols = list(df.columns)
        all_cols.remove("Ad Name")
        all_cols.insert(all_cols.index("Ad ID") + 1, "Ad Name")
        df = df[all_cols]

    ADS_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values("Timestamp", ascending=False).to_csv(ADS_OUTPUT_FILE, index=False)
    print(f"Merged ads saved: {len(df):,} rows -> {ADS_OUTPUT_FILE}")

    if "Source" in df.columns:
        source_clean = df["Source"].fillna("").str.lower().str.strip()
        paid_contact_ids = set(
            df.loc[source_clean == "paid ads", "Contact ID"].dropna().astype(int).tolist()
        )
        organic_contact_ids = set(
            df.loc[source_clean.isin(CHANNEL_SOURCE_ORGANIC), "Contact ID"]
            .dropna()
            .astype(int)
            .tolist()
        )
    else:
        paid_contact_ids = set(
            df.loc[df["Ad campaign ID"].notna(), "Contact ID"].dropna().astype(int).tolist()
        )
        organic_contact_ids = set()

    print(
        f"Ads loaded: {len(df):,} records | "
        f"{len(paid_contact_ids):,} paid | {len(organic_contact_ids):,} organic"
    )
    return df, paid_contact_ids, organic_contact_ids


def run_analytics_pipeline():
    print("V55.0 MASTER ANALYTICS: Intelligent Revenue + Character Fix + Aggregation")
    if not os.path.exists(FINAL_TAGGED_DATA):
        return

    def build_text_series(frame: pd.DataFrame, column: str) -> pd.Series:
        if column in frame.columns:
            return frame[column].astype("string").fillna("").str.strip()
        return pd.Series("", index=frame.index, dtype="string")

    def map_staff_value(val):
        if pd.isna(val) or str(val).strip() == "":
            return None
        clean_val = str(val).replace(".0", "").strip()
        return TEAM_MAP.get(clean_val, f"Other ({clean_val})")

    def resolve_staff_temp_values(last_assignee, assignee, first_response_by, first_assignee):
        for candidate in (last_assignee, assignee, first_response_by, first_assignee):
            clean = str(candidate).replace(".0", "").strip()
            if clean in TEAM_MAP:
                return TEAM_MAP[clean]
            if clean not in ["nan", "None", "System", "Bot", "Auto Assign"] and len(clean) > 2:
                return clean
        return "Unassigned"

    def process_ai_record(full_context, mpesa_amount, detected_brands):
        match_row = None
        if detected_brands:
            for brand in detected_brands:
                match_row = matcher.get_best_match(full_context, mpesa_amount, brand_hint=brand)
                if match_row is not None:
                    break
        if match_row is None:
            match_row = matcher.get_best_match(full_context, mpesa_amount, brand_hint=None)

        if match_row is not None:
            product_name = match_row["Name"]
            product_brand = match_row["Brand"]
            product_category = match_row["Canonical_Category"]
            product_sub = match_row["Sub_Category"]
            product_concern = match_row["Concerns"]
            product_audience = match_row["Target_Audience"]
        else:
            if detected_brands:
                product_brand = detected_brands[0]
                if mpesa_amount > 0:
                    product_name = "Unmatched Paid Product - Manual Review"
                    product_brand = "Unknown"
                    product_category = "General Inquiry"
                else:
                    product_name = f"General {product_brand} Inquiry"
                    product_category = BRAND_TO_CATEGORY_MAP.get(product_brand, "General Inquiry")
            else:
                product_name, product_brand, product_category = "Unknown", "Unknown", "General Inquiry"
            product_concern, product_audience, product_sub = "General", "General", "General"

        return (
            MACRO_GROUP_MAP.get(product_category, "General Inquiry"),
            product_sub,
            product_name,
            product_brand,
            product_concern,
            product_audience,
        )

    def parse_concern_string(raw: str) -> list:
        if not raw or str(raw).strip().lower() in {"nan", "none", ""}:
            return []
        raw = re.sub(r"[,.]", ",", str(raw))
        parts = [
            " ".join(part.strip().split()).title()
            for part in raw.split(",")
            if part.strip()
        ]
        exclusions = {"general", "general care", "", "nan", "none"}
        return [part for part in parts if part.lower() not in exclusions]

    df_sess = pd.read_parquet(FINAL_TAGGED_DATA)
    matcher = ProductMatcher()

    print("Deduplicating revenue...")
    initial_context = df_sess.get("full_context", pd.Series("", index=df_sess.index)).fillna("")
    initial_amount = pd.to_numeric(df_sess.get("mpesa_amount", 0), errors="coerce").fillna(0)
    df_sess["mpesa_amount"] = [
        recalculate_mpesa_amount(context, amount)
        for context, amount in zip(initial_context.tolist(), initial_amount.tolist())
    ]

    print("Extracting staff from assignment messages...")
    df_sess["extracted_staff_name"] = initial_context.map(extract_assigned_staff)

    print("Linking ad campaigns to sessions...")
    df_ads, paid_ad_contact_ids, organic_contact_ids = load_ads_for_analytics()
    meta_ad_name_map = load_meta_ad_name_map() or AD_NAME_MAP

    df_sess["session_start"] = pd.to_datetime(df_sess["session_start"], errors="coerce")
    if not df_ads.empty:
        df_sess = df_sess.sort_values("session_start").reset_index(drop=True)
        df_sess["Contact ID"] = pd.to_numeric(df_sess["Contact ID"], errors="coerce").astype("Int64")
        df_ads["Contact ID"] = pd.to_numeric(df_ads["Contact ID"], errors="coerce").astype("Int64")

        df_sess = pd.merge_asof(
            df_sess,
            df_ads,
            left_on="session_start",
            right_on="Timestamp",
            by="Contact ID",
            tolerance=pd.Timedelta("6h"),
            direction="backward",
            suffixes=("", "_new"),
        )
        for column in ["Ad campaign ID", "Ad group ID", "Ad ID", "Ad Name", "Source"]:
            new_column = f"{column}_new"
            if new_column in df_sess.columns:
                if column in df_sess.columns:
                    df_sess[column] = df_sess[column].fillna(df_sess[new_column])
                else:
                    df_sess[column] = df_sess[new_column]
                df_sess.drop(columns=[new_column], inplace=True)
        print(f"Ads linked: {df_sess['Ad campaign ID'].notna().sum():,} sessions matched")
        df_sess.drop(columns=["Timestamp"], errors="ignore", inplace=True)

    print("Cleaning IDs...")
    df_sess["Contact ID"] = df_sess["Contact ID"].apply(clean_id)

    cleaned_conv_path = CLEANED_DATA_DIR / "cleaned_conversations.csv"
    cleaned_cont_path = CLEANED_DATA_DIR / "cleaned_contacts.csv"
    potential_conv_cols = [
        "Contact ID",
        "conv_start",
        "Conversation ID",
        "Opened By Source",
        "Assignee",
        "First Assignee",
        "Last Assignee",
        "Closed By",
        "First Response By",
        "First Response Time",
        "Average Response Time",
        "Resolution Time",
        "Number of Responses",
        "Number of Outgoing Messages",
        "Number of Incoming Messages",
        "Conversation Category",
        "Closing Note Summary",
    ]

    if os.path.exists(cleaned_conv_path):
        conv_header = pd.read_csv(cleaned_conv_path, nrows=0).columns.tolist()
        conv_id_col = next((c for c in ["Contact ID", "ContactID", "ID", "id", "contact_id"] if c in conv_header), None)
        conv_usecols = [
            col
            for col in (["DateTime Conversation Started", conv_id_col] + potential_conv_cols)
            if col and col in conv_header
        ]
        df_c = pd.read_csv(cleaned_conv_path, usecols=list(dict.fromkeys(conv_usecols)))
        if conv_id_col:
            df_c["Contact ID"] = df_c[conv_id_col].apply(clean_id)
        df_c["conv_start"] = pd.to_datetime(df_c["DateTime Conversation Started"], errors="coerce")
        df_c = df_c.sort_values("conv_start")
        for column in [
            "Average Response Time",
            "Number of Responses",
            "Number of Outgoing Messages",
            "Number of Incoming Messages",
            "Resolution Time",
            "First Response Time",
        ]:
            if column in df_c.columns:
                df_c[column] = pd.to_numeric(df_c[column], errors="coerce").fillna(0)
        cols_to_merge = [col for col in potential_conv_cols if col in df_c.columns]
        df_c_subset = df_c[cols_to_merge].copy()
    else:
        df_c_subset = pd.DataFrame()

    name_map = {}
    phone_map = {}
    if os.path.exists(cleaned_cont_path):
        cont_header = pd.read_csv(cleaned_cont_path, nrows=0).columns.tolist()
        id_n = next((c for c in ["Contact ID", "ContactID", "ID", "id", "contact_id"] if c in cont_header), None)
        name_c = next((c for c in cont_header if "name" in c.lower() and "phone" not in c.lower()), None)
        phone_c = next((c for c in cont_header if "phone" in c.lower() or "number" in c.lower()), None)
        contact_usecols = [col for col in [id_n, name_c, phone_c] if col]
        if contact_usecols:
            df_n = pd.read_csv(cleaned_cont_path, usecols=list(dict.fromkeys(contact_usecols)))
            df_n[id_n] = df_n[id_n].apply(clean_id)
            if name_c:
                name_map = df_n.set_index(id_n)[name_c].to_dict()
            if phone_c:
                df_n[phone_c] = df_n[phone_c].apply(normalize_phone)
                phone_map = df_n.set_index(id_n)[phone_c].to_dict()

    df_sess = df_sess.sort_values("session_start").reset_index(drop=True)
    df_sess["Conversation ID"] = pd.NA
    df_sess["active_staff"] = pd.NA

    if not df_c_subset.empty:
        print("Linking sessions...")
        grace_period = pd.Timedelta(hours=24)
        lookback_period = pd.Timedelta(days=14)

        df_c_subset["Temp_Staff_Name"] = [
            resolve_staff_temp_values(last_assignee, assignee, first_response_by, first_assignee)
            for last_assignee, assignee, first_response_by, first_assignee in zip(
                df_c_subset.get("Last Assignee", pd.Series("", index=df_c_subset.index)).tolist(),
                df_c_subset.get("Assignee", pd.Series("", index=df_c_subset.index)).tolist(),
                df_c_subset.get("First Response By", pd.Series("", index=df_c_subset.index)).tolist(),
                df_c_subset.get("First Assignee", pd.Series("", index=df_c_subset.index)).tolist(),
            )
        ]

        df_sess["_session_row"] = np.arange(len(df_sess))
        session_link = df_sess[["_session_row", "Contact ID", "session_start"]].copy()
        session_link["session_link_upper"] = session_link["session_start"] + grace_period
        session_link = session_link.dropna(subset=["Contact ID", "session_link_upper"])

        conv_link = df_c_subset[["Contact ID", "conv_start", "Conversation ID", "Temp_Staff_Name"]].copy()
        conv_link = conv_link.dropna(subset=["Contact ID", "conv_start"])

        # merge_asof is strict: both frames must be sorted by the asof key first.
        session_link = session_link.sort_values(["session_link_upper", "Contact ID"]).reset_index(drop=True)
        conv_link = conv_link.sort_values(["conv_start", "Contact ID"]).reset_index(drop=True)

        linked = pd.merge_asof(
            session_link,
            conv_link,
            left_on="session_link_upper",
            right_on="conv_start",
            by="Contact ID",
            direction="backward",
            tolerance=lookback_period + grace_period,
        )
        valid_match = linked["conv_start"].notna() & (
            linked["conv_start"] >= (linked["session_start"] - lookback_period)
        )
        linked = linked.loc[valid_match, ["_session_row", "Conversation ID", "Temp_Staff_Name"]]
        print(f"Linked {linked['Conversation ID'].notna().sum():,} sessions")

        df_sess = df_sess.merge(linked, on="_session_row", how="left")
        df_sess["Conversation ID"] = df_sess["Conversation ID_y"].combine_first(df_sess["Conversation ID_x"])
        df_sess["active_staff"] = df_sess["Temp_Staff_Name"].combine_first(df_sess["active_staff"])
        df_sess.drop(
            columns=["Conversation ID_x", "Conversation ID_y", "Temp_Staff_Name", "_session_row"],
            inplace=True,
            errors="ignore",
        )

        cols_needed = [
            "Conversation ID",
            "First Response By",
            "Assignee",
            "Last Assignee",
            "Closed By",
            "First Assignee",
            "Opened By Source",
        ]
        cols_present = [col for col in cols_needed if col in df_c_subset.columns]
        meta_df = df_c_subset[cols_present].drop_duplicates(subset=["Conversation ID"])
        cols_to_drop = [col for col in cols_present if col in df_sess.columns and col != "Conversation ID"]
        if cols_to_drop:
            df_sess.drop(columns=cols_to_drop, inplace=True)

        df_sess["Conversation ID"] = pd.to_numeric(df_sess["Conversation ID"], errors="coerce").astype("Int64")
        meta_df["Conversation ID"] = pd.to_numeric(meta_df["Conversation ID"], errors="coerce").astype("Int64")
        df_sess = pd.merge(df_sess, meta_df, on="Conversation ID", how="left")
    else:
        print("No conversation data found")

    if "Opened By Source" in df_sess.columns:
        df_sess.rename(columns={"Opened By Source": "recovered_source"}, inplace=True)
    elif "recovered_source" not in df_sess.columns:
        df_sess["recovered_source"] = pd.NA

    df_sess = df_sess.sort_values(by=["Contact ID", "session_start"]).reset_index(drop=True)
    df_sess["session_id"] = (
        df_sess["Contact ID"].astype(str) + "_" + df_sess["session_start"].dt.strftime("%Y-%m-%d %H:%M:%S")
    )
    df_sess["contact_name"] = df_sess["Contact ID"].map(name_map).fillna("Unknown")
    df_sess["mpesa_amount"] = pd.to_numeric(df_sess["mpesa_amount"], errors="coerce").fillna(0)

    final_tags = df_sess.get("final_tags", pd.Series("", index=df_sess.index)).fillna("").astype(str)
    full_context = df_sess.get("full_context", pd.Series("", index=df_sess.index)).fillna("").astype(str)
    full_context_lower = full_context.str.lower()
    final_tags_lower = final_tags.str.lower()

    df_sess["is_converted"] = (
        final_tags.str.contains("Converted", na=False) | (df_sess["mpesa_amount"] > 0)
    ).astype(int)

    channel_ids = pd.to_numeric(df_sess.get("Channel ID", pd.Series(np.nan, index=df_sess.index)), errors="coerce")
    df_sess["channel_name"] = channel_ids.map(CHANNEL_MAP).fillna("Unknown")

    contact_id_numeric = pd.to_numeric(df_sess["Contact ID"], errors="coerce").astype("Int64")
    ad_campaign_clean = build_text_series(df_sess, "Ad campaign ID")
    ad_id_clean = build_text_series(df_sess, "Ad ID")
    recovered_source_clean = build_text_series(df_sess, "recovered_source").str.lower()

    in_paid = contact_id_numeric.isin(paid_ad_contact_ids)
    in_organic = contact_id_numeric.isin(organic_contact_ids)
    has_paid_id = ~ad_campaign_clean.isin(INVALID_TEXT_VALUES) | ~ad_id_clean.isin(INVALID_TEXT_VALUES)

    df_sess["acquisition_source"] = np.select(
        [
            in_paid,
            ~in_paid & in_organic,
            ~(in_paid | in_organic) & has_paid_id,
            recovered_source_clean.eq(SOURCE_RECOVERED_PAID),
            recovered_source_clean.isin(SOURCE_RECOVERED_ORGANIC),
        ],
        [
            "Paid Ads",
            "Organic / Direct",
            "Paid Ads",
            "Paid Ads",
            "Organic / Direct",
        ],
        default="Pending Classification",
    )

    df_sess["clean_ad_id"] = clean_scientific_id_series(
        df_sess["Ad ID"] if "Ad ID" in df_sess.columns else pd.Series(pd.NA, index=df_sess.index, dtype="string")
    )
    existing_ad_names = df_sess.get(
        "Ad Name",
        pd.Series(pd.NA, index=df_sess.index, dtype="object"),
    ).replace("", pd.NA)
    df_sess["Ad Name"] = existing_ad_names.fillna(df_sess["clean_ad_id"].map(meta_ad_name_map))

    df_sess["visit_rank"] = df_sess.groupby("Contact ID").cumcount() + 1
    df_sess["prior_orders"] = df_sess.groupby("Contact ID")["is_converted"].cumsum().shift(1).fillna(0)
    df_sess["customer_status"] = np.where(df_sess["prior_orders"] == 0, "New", "Returning")

    print("Running AI enrichment...")
    split_results = pd.DataFrame(
        [split_tags_logic(value) for value in final_tags.tolist()],
        columns=["zone_name", "secondary_zones", "funnel_history", "_other_tags"],
        index=df_sess.index,
    )
    df_sess["zone_name"] = split_results["zone_name"]
    df_sess["secondary_zones"] = split_results["secondary_zones"]
    df_sess["funnel_history"] = split_results["funnel_history"]

    df_sess["is_stock_inquiry"] = final_tags_lower.str.count("stock")
    df_sess["is_recommendation"] = final_tags_lower.str.count("recommendation")
    df_sess["is_consultation"] = (
        final_tags_lower.str.contains(CONSULTATION_TAG_PATTERN, regex=True)
        | full_context_lower.str.contains(CONSULTATION_CONTEXT_PATTERN, regex=True)
    ).astype(int)
    df_sess["temp_brands"] = final_tags_lower.map(extract_dynamic_brands_from_tags)

    detected_brands_by_session = [
        normalize_brands_with_intent(temp_brands, context)
        for temp_brands, context in zip(df_sess["temp_brands"].tolist(), full_context.tolist())
    ]
    ai_results = [
        process_ai_record(context, amount, detected_brands)
        for context, amount, detected_brands in tqdm(
            zip(
                full_context.tolist(),
                df_sess["mpesa_amount"].tolist(),
                detected_brands_by_session,
            ),
            total=len(df_sess),
            desc="AI Enrichment",
            unit="session",
        )
    ]
    df_sess[
        [
            "primary_category",
            "sub_category",
            "matched_product",
            "matched_brand",
            "matched_concern",
            "target_audience",
        ]
    ] = pd.DataFrame(ai_results, index=df_sess.index)

    print("Mapping staff...")
    for source_col, target_col in [
        ("First Response By", "first_response_name"),
        ("Assignee", "assignee_name"),
        ("Closed By", "closed_by_name"),
        ("Last Assignee", "last_assignee_name"),
    ]:
        source_values = df_sess[source_col].tolist() if source_col in df_sess.columns else [None] * len(df_sess)
        df_sess[target_col] = [map_staff_value(value) for value in source_values]

    if "active_staff" not in df_sess.columns:
        df_sess["active_staff"] = None

    df_sess["active_staff"] = df_sess["active_staff"].fillna(
        df_sess["closed_by_name"].fillna(
            df_sess["last_assignee_name"].fillna(
                df_sess["assignee_name"].fillna(
                    df_sess["first_response_name"].fillna(
                        df_sess["extracted_staff_name"].fillna("Unmapped")
                    )
                )
            )
        )
    )
    df_sess["sales_owner"] = df_sess["closed_by_name"].fillna(
        df_sess["last_assignee_name"].fillna(
            df_sess["assignee_name"].fillna(
                df_sess["first_response_name"].fillna(
                    df_sess["extracted_staff_name"].fillna("Unmapped")
                )
            )
        )
    )
    df_sess["activity_date"] = df_sess["session_start"].dt.date

    stats = df_sess.groupby("Contact ID", sort=False).agg(
        frequency=("session_id", "count"),
        monetary_value=("mpesa_amount", "sum"),
        total_orders=("is_converted", "sum"),
        last_seen=("session_start", "max"),
        first_seen=("session_start", "min"),
    )
    stats["first_seen"] = pd.to_datetime(stats["first_seen"], errors="coerce")
    stats["last_seen"] = pd.to_datetime(stats["last_seen"], errors="coerce")
    stats["days_to_convert"] = (stats["last_seen"] - stats["first_seen"]).dt.days

    broad_bracket_sort = {"No Spend": 0, "0 - 7k": 1, "7k - 13k": 2, "13k - 20k": 3, "20k+": 4}
    lifetime_tier_sort = {"Bronze": 0, "Silver": 1, "Gold": 2, "Platinum": 3}
    stats_monetary = stats["monetary_value"]
    stats["lifetime_tier_history"] = np.select(
        [stats_monetary > 20000, stats_monetary > 13000, stats_monetary > 7000],
        ["Platinum", "Gold", "Silver"],
        default="Bronze",
    )
    stats["lifetime_tier_sort"] = pd.Series(stats["lifetime_tier_history"], index=stats.index).map(
        lifetime_tier_sort
    )
    stats["lifetime_bracket"] = np.select(
        [stats_monetary <= 0, stats_monetary <= 7000, stats_monetary <= 13000, stats_monetary <= 20000],
        ["No Spend", "0 - 7k", "7k - 13k", "13k - 20k"],
        default="20k+",
    )
    stats["lifetime_bracket_sort"] = pd.Series(stats["lifetime_bracket"], index=stats.index).map(
        broad_bracket_sort
    )

    df_sess = df_sess.merge(
        stats[["lifetime_tier_history", "lifetime_bracket", "days_to_convert"]],
        on="Contact ID",
        how="left",
    )

    session_amount = df_sess["mpesa_amount"]
    df_sess["session_bracket_broad"] = np.select(
        [session_amount <= 0, session_amount <= 7000, session_amount <= 13000, session_amount <= 20000],
        ["No Spend", "0 - 7k", "7k - 13k", "13k - 20k"],
        default="20k+",
    )
    df_sess["broad_sort"] = pd.Series(df_sess["session_bracket_broad"], index=df_sess.index).map(
        broad_bracket_sort
    )
    df_sess["customer_tier"] = np.select(
        [session_amount > 20000, session_amount > 13000, session_amount > 7000, session_amount > 0],
        ["Platinum", "Gold", "Silver", "Bronze"],
        default="No Spend",
    )
    df_sess["session_tier"] = df_sess["customer_tier"]

    granular_brackets = pd.cut(
        session_amount,
        bins=[-np.inf, 0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, np.inf],
        labels=["0. No Spend", "0 - 1", "1-2k", "2-3k", "3-4k", "4-5k", "5-6k", "6-7k", "7-8k", "8-9k", "9-10k", "10k+"],
        include_lowest=True,
    )
    df_sess["bracket_granular"] = granular_brackets.astype("string")
    df_sess["bracket_sort"] = granular_brackets.cat.codes

    brackets_10_20 = pd.cut(
        session_amount,
        bins=[-np.inf, 10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 20000, np.inf],
        labels=["0-10k", "10-11k", "11-12k", "12-13k", "13-14k", "14-15k", "15-16k", "16-17k", "17-18k", "18-19k", "19-20k", "20k+"],
        include_lowest=True,
    )
    df_sess["bracket_10_20k"] = brackets_10_20.astype("string")
    df_sess["bracket_10_20k_sort"] = brackets_10_20.cat.codes

    print("Calculating session-based conversion speed...")
    df_sess = df_sess.sort_values(["Contact ID", "session_start"]).reset_index(drop=True)
    df_sess["prev_session"] = df_sess.groupby("Contact ID")["session_start"].shift(1)
    df_sess["days_since_last"] = (df_sess["session_start"] - df_sess["prev_session"]).dt.days.fillna(9999)
    df_sess["is_new_journey"] = (df_sess["days_since_last"] > 30).astype(int)
    df_sess["journey_id"] = df_sess.groupby("Contact ID")["is_new_journey"].cumsum()
    journey_starts = df_sess.groupby(["Contact ID", "journey_id"])["session_start"].transform("min")
    df_sess["session_days_to_convert"] = (df_sess["session_start"] - journey_starts).dt.days
    df_sess["conversion_speed"] = np.select(
        [
            df_sess["is_converted"].eq(0),
            df_sess["session_days_to_convert"].le(3),
            df_sess["session_days_to_convert"].le(7),
        ],
        [
            "Not Converted",
            "Within 3 Days",
            "Research (1 Week)",
        ],
        default="Consultative (8+ Days)",
    )

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    df_sess.drop(columns=["prev_session", "days_since_last", "is_new_journey", "journey_id"], inplace=True, errors="ignore")
    df_sess["phone_number"] = df_sess["Contact ID"].map(phone_map)

    ad_cols = ["Ad campaign ID", "Ad group ID", "Ad ID"]
    for column in ad_cols:
        if column not in df_sess.columns:
            df_sess[column] = None

    print("Finalizing column order...")
    cols = list(df_sess.columns)
    target_block = ["acquisition_source", "recovered_source", "channel_name", "clean_ad_id", "Ad Name"] + ad_cols
    for col in target_block:
        if col in cols:
            cols.remove(col)
    anchor_idx = cols.index("session_id") + 1 if "session_id" in cols else 2
    for offset, col_name in enumerate(target_block):
        if col_name in df_sess.columns:
            cols.insert(anchor_idx + offset, col_name)
    df_sess = df_sess[cols]

    df_sess_export = sanitize_text_for_csv_export(df_sess)
    df_sess_export.to_csv(PROCESSED_DATA_DIR / "fact_sessions_enriched.csv", index=False)
    stats.reset_index().to_csv(PROCESSED_DATA_DIR / "dim_customers_rfv.csv", index=False)

    p_rows = []
    for session_id, contact_id, context, revenue, category, is_converted, date_value, detected_brands in zip(
        df_sess["session_id"].tolist(),
        df_sess["Contact ID"].tolist(),
        df_sess["full_context"].fillna("").astype(str).tolist(),
        df_sess["mpesa_amount"].tolist(),
        df_sess["primary_category"].tolist(),
        df_sess["is_converted"].tolist(),
        df_sess["session_start"].tolist(),
        detected_brands_by_session,
    ):
        session_brands = detected_brands or [None]
        found_products_in_session = set()
        for brand_hint in session_brands:
            match_row = matcher.get_best_match(context, revenue, brand_hint=brand_hint)
            if match_row is None:
                continue
            product_name = str(match_row["Name"]).strip()
            brand_name = str(match_row["Brand"]).strip()
            unique_key = f"{product_name}_{brand_name}"
            if unique_key in found_products_in_session:
                continue
            found_products_in_session.add(unique_key)
            clean_product_name = product_name
            if len(brand_name) > 2 and product_name.lower().startswith(brand_name.lower()):
                clean_product_name = product_name[len(brand_name):].strip(" -:")
            p_rows.append(
                {
                    "session_id": session_id,
                    "Contact ID": contact_id,
                    "brand_name": brand_name,
                    "product_name": clean_product_name,
                    "original_product_name": product_name,
                    "category": category,
                    "sub_category": match_row["Sub_Category"],
                    "concern": match_row["Concerns"],
                    "revenue": revenue,
                    "is_converted": is_converted,
                    "date": date_value,
                }
            )
    if p_rows:
        pd.DataFrame(p_rows).to_csv(PROCESSED_DATA_DIR / "fact_product_mentions.csv", index=False)

    b_rows = []
    for session_id, contact_id, matched_brand, temp_brands, revenue, is_converted, date_value in zip(
        df_sess["session_id"].tolist(),
        df_sess["Contact ID"].tolist(),
        df_sess["matched_brand"].tolist(),
        df_sess["temp_brands"].tolist(),
        df_sess["mpesa_amount"].tolist(),
        df_sess["is_converted"].tolist(),
        df_sess["session_start"].tolist(),
    ):
        if matched_brand != "Unknown":
            brands_to_use = [matched_brand]
        else:
            raw_brands = re.split(r"\s*[,|]\s*", str(temp_brands))
            brands_to_use = [brand.strip() for brand in raw_brands if brand.strip()]
        if not brands_to_use:
            brands_to_use = ["General Inquiry"]
        for brand in brands_to_use:
            b_rows.append(
                {
                    "session_id": session_id,
                    "Contact ID": contact_id,
                    "brand_name": brand,
                    "revenue": revenue,
                    "is_converted": is_converted,
                    "date": date_value,
                }
            )
    if b_rows:
        pd.DataFrame(b_rows).to_csv(PROCESSED_DATA_DIR / "fact_brand_mentions.csv", index=False)

    funnel_rows = []
    for session_id, contact_id, contact_name, date_value, category, source, funnel_history, is_converted, customer_tier in zip(
        df_sess["session_id"].tolist(),
        df_sess["Contact ID"].tolist(),
        df_sess["contact_name"].tolist(),
        df_sess["session_start"].tolist(),
        df_sess["primary_category"].tolist(),
        df_sess["acquisition_source"].tolist(),
        df_sess["funnel_history"].tolist(),
        df_sess["is_converted"].tolist(),
        df_sess["customer_tier"].tolist(),
    ):
        base_row = {
            "session_id": session_id,
            "id": contact_id,
            "name": contact_name,
            "date": date_value,
            "cat": category,
            "src": source,
        }
        history = str(funnel_history)
        funnel_rows.append({**base_row, "stage": "Inquiry", "sort_order": 1, "val": 1})
        if "Price Quoted" in history:
            funnel_rows.append({**base_row, "stage": "Price Quoted", "sort_order": 2, "val": 1})
        if "Price Objection" in history:
            funnel_rows.append({**base_row, "stage": "Price Objection", "sort_order": 3, "val": 1})
        if is_converted:
            funnel_rows.append({**base_row, "stage": "Converted", "sort_order": 4, "val": 1})
        if customer_tier == "Platinum":
            funnel_rows.append({**base_row, "stage": "High Val Cust", "sort_order": 5, "val": 1})
    pd.DataFrame(funnel_rows).to_csv(PROCESSED_DATA_DIR / "fact_funnel_analytics.csv", index=False)

    zone_rows = []
    for session_id, zone_name, secondary_zones in zip(
        df_sess["session_id"].tolist(),
        df_sess["zone_name"].tolist(),
        df_sess["secondary_zones"].tolist(),
    ):
        if zone_name:
            zone_rows.append({"session_id": session_id, "type": "Primary", "loc": zone_name})
        if secondary_zones:
            for zone in str(secondary_zones).split("|"):
                if zone.strip():
                    zone_rows.append({"session_id": session_id, "type": "Secondary", "loc": zone.strip()})
    if zone_rows:
        pd.DataFrame(zone_rows).to_csv(PROCESSED_DATA_DIR / "fact_session_zones.csv", index=False)

    concern_rows = []
    for session_id, contact_id, matched_concern, context, is_converted, revenue, matched_brand, matched_product, primary_category, sub_category, date_value in zip(
        df_sess["session_id"].tolist(),
        df_sess["Contact ID"].tolist(),
        df_sess["matched_concern"].tolist(),
        df_sess["full_context"].fillna("").astype(str).str.lower().tolist(),
        df_sess["is_converted"].tolist(),
        df_sess["mpesa_amount"].tolist(),
        df_sess["matched_brand"].tolist(),
        df_sess["matched_product"].tolist(),
        df_sess["primary_category"].tolist(),
        df_sess["sub_category"].tolist(),
        df_sess["session_start"].tolist(),
    ):
        concerns_found = set()
        source = "chat_pattern"

        parsed_kb = parse_concern_string(str(matched_concern))
        if parsed_kb:
            concerns_found.update(parsed_kb)
            source = "kb_match"

        if not concerns_found:
            for concern, patterns in CONCERN_PATTERNS.items():
                if any(pattern.search(context) for pattern in patterns):
                    concerns_found.add(concern)

        for concern in concerns_found:
            concern_rows.append(
                {
                    "session_id": session_id,
                    "Contact ID": contact_id,
                    "concern": concern,
                    "date": date_value,
                    "is_converted": is_converted,
                    "revenue": revenue,
                    "source": source,
                    "brand": matched_brand,
                    "product": matched_product,
                    "primary_category": primary_category,
                    "sub_category": sub_category,
                }
            )

    if concern_rows:
        pd.DataFrame(concern_rows).to_csv(PROCESSED_DATA_DIR / "fact_session_concerns.csv", index=False)
        print(
            f"Concern export: {len(concern_rows):,} rows across "
            f"{len(set(row['concern'] for row in concern_rows))} unique concerns"
        )

    print("ANALYTICS COMPLETE.")


if __name__ == "__main__":
    run_analytics_pipeline()
