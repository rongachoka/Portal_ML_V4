"""
brand_crawler.py
================
Collects product-packaging images per brand from the Knowledge Base.

Sources   : Bing Images (primary) -> Google Images (fallback)
Browser   : Selenium + Chrome
Filter    : CLIP keeps product-packaging images only
Branding  : OCR plus URL/title/category checks reduce wrong-brand contamination
Resume    : Skips any brand folder already containing >= TARGET_IMAGES images
Dedup     : Perceptual hash rejects near-duplicate downloads

Install dependencies first:
    pip install selenium webdriver-manager transformers torch torchvision
                Pillow requests pandas imagehash pytesseract

Optional but recommended for stronger brand checks:
    Install Tesseract OCR and make sure pytesseract can find it.

Run from the project root:
    python -m img_classification.brand_crawler
or:
    python img_classification/brand_crawler.py
"""

from __future__ import annotations

import gc
import json
import logging
import random
import re
import signal
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional

import imagehash
import pandas as pd
import requests
import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from selenium import webdriver
from selenium.common.exceptions import InvalidSessionIdException, WebDriverException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from transformers import CLIPModel, CLIPProcessor
from webdriver_manager.chrome import ChromeDriverManager

try:
    import pytesseract
except ImportError:
    pytesseract = None

try:
    from Portal_ML_V4.src.config.settings import KB_PATH
except ModuleNotFoundError:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT.parent) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT.parent))
    from Portal_ML_V4.src.config.settings import KB_PATH


def _handle_sigint(sig, frame):
    raise KeyboardInterrupt


signal.signal(signal.SIGINT, _handle_sigint)


OUTPUT_DIR = Path(r"G:\My Drive\Portal_ML\img_classification\Brand_Images_NEW")

# Targeting / filtering
TARGET_IMAGES = 30
DOWNLOAD_TARGET = 60
CLIP_THRESHOLD = 0.38
MIN_SIZE = (80, 80)
REQUEST_TIMEOUT = 12
PHASH_MAX_DISTANCE = 4
MIN_BRAND_TOKEN_LENGTH = 3
MIN_CATEGORY_TOKEN_LENGTH = 4
AMBIGUOUS_BRAND_MAX_LEN = 4
MAX_SEARCH_CATEGORIES = 2

# Browser
HEADLESS = False

# KB columns
BRAND_COL = "Brand"
CATEGORY_COL = "Canonical_Category"

# Brands to skip even if present in KB
SKIP_BRANDS = {
    "unknown",
    "nan",
    "general",
    "",
    "logistics",
    "other social sale",
    "prescription",
    "antibiotics",
    "1st+",
}

CATEGORY_STOPWORDS = {
    "and",
    "with",
    "for",
    "the",
}

CATEGORY_QUERY_TEMPLATES: List[str] = [
    '"{brand}" {category} product packaging',
    '"{brand}" {category} product',
    '"{brand}" {category} bottle',
    '"{brand}" official {category} product',
]
GENERIC_QUERY_TEMPLATES: List[str] = [
    '"{brand}" product packaging',
    '"{brand}" official product',
]

POSITIVE_PROMPTS: List[str] = [
    "product packaging on a white background",
    "cosmetic bottle or tube",
    "skincare cream serum or lotion",
    "medicine or pharmaceutical bottle",
    "personal care product container",
]
NEGATIVE_PROMPTS: List[str] = [
    "person model or face",
    "landscape nature or outdoor scene",
    "brand logo or icon without product",
    "website screenshot or text document",
    "food drink or grocery",
    "store shelf interior photo",
]

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def _setup_logging() -> logging.Logger:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_file = OUTPUT_DIR / "crawler.log"
    logger = logging.getLogger("brand_crawler")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


log = _setup_logging()


@dataclass(frozen=True)
class BrandRecord:
    brand: str
    categories: tuple[str, ...]
    primary_category: str


@dataclass(frozen=True)
class ImageCandidate:
    image_url: str
    query: str
    engine: str
    source_url: str = ""
    source_title: str = ""


def safe_folder_name(brand: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', "_", brand).strip()


def slugify(value: str, max_len: int = 50) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug[:max_len] or "brand"


def normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def collapse_spaces(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def split_category_values(value: str) -> List[str]:
    if not value:
        return []
    parts = re.split(r"[,;|]+", str(value))
    cleaned = [collapse_spaces(part) for part in parts if collapse_spaces(part)]
    return list(dict.fromkeys(cleaned))


def normalize_category_for_query(category: str) -> str:
    return normalize_text(category)


def count_images(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(
        1
        for file_path in folder.iterdir()
        if file_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )


def existing_hashes(folder: Path) -> List[imagehash.ImageHash]:
    hashes: List[imagehash.ImageHash] = []
    if not folder.exists():
        return hashes

    for file_path in folder.iterdir():
        if file_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        try:
            with Image.open(file_path) as img:
                hashes.append(imagehash.phash(img.convert("RGB")))
        except Exception:
            pass
    return hashes


def is_near_duplicate(
    candidate_hash: imagehash.ImageHash,
    known_hashes: List[imagehash.ImageHash],
    max_distance: int = PHASH_MAX_DISTANCE,
) -> bool:
    return any(candidate_hash - saved_hash <= max_distance for saved_hash in known_hashes)


class CLIPFilter:
    def __init__(self, threshold: float = CLIP_THRESHOLD):
        self.threshold = threshold
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._n_pos = len(POSITIVE_PROMPTS)
        self._prompts = POSITIVE_PROMPTS + NEGATIVE_PROMPTS
        log.info(
            "Loading CLIP filter (device=%s). First run may download weights.",
            self.device,
        )
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(
            self.device
        )
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model.eval()
        log.info("CLIP filter ready.")

    def score(self, image: Image.Image) -> float:
        try:
            inputs = self.processor(
                text=self._prompts,
                images=image.convert("RGB"),
                return_tensors="pt",
                padding=True,
            ).to(self.device)
            with torch.no_grad():
                output = self.model(**inputs)
                probs = output.logits_per_image.softmax(dim=-1)[0]
            return float(probs[: self._n_pos].sum())
        except Exception as exc:
            log.debug("CLIP score error: %s", exc)
            return 0.0


class BrandVerifier:
    def __init__(self) -> None:
        self.ocr_enabled = pytesseract is not None
        if self.ocr_enabled:
            log.info("OCR brand verifier enabled via pytesseract.")
        else:
            log.warning(
                "pytesseract is not installed. Brand verification will fall back to URL/title hints only."
            )

    def _brand_parts(self, brand: str) -> Dict[str, object]:
        normalized = normalize_text(brand)
        compact = normalized.replace(" ", "")
        tokens = [token for token in normalized.split() if len(token) >= MIN_BRAND_TOKEN_LENGTH]
        return {
            "normalized": normalized,
            "compact": compact,
            "tokens": tokens,
        }

    def _text_matches_brand(self, text: str, brand: str) -> bool:
        if not text:
            return False

        brand_parts = self._brand_parts(brand)
        normalized_text = normalize_text(text)
        compact_text = normalized_text.replace(" ", "")
        words = set(normalized_text.split())

        if brand_parts["compact"] and brand_parts["compact"] in compact_text:
            return True

        tokens = brand_parts["tokens"]
        if not tokens:
            return False

        matched_tokens = sum(1 for token in tokens if token in words)
        if len(tokens) == 1:
            return matched_tokens == 1
        return matched_tokens >= min(len(tokens), 2)

    def _category_tokens(self, categories: tuple[str, ...]) -> List[str]:
        tokens: List[str] = []
        for category in categories:
            normalized = normalize_text(category)
            for token in normalized.split():
                if len(token) >= MIN_CATEGORY_TOKEN_LENGTH and token not in CATEGORY_STOPWORDS:
                    tokens.append(token)
        return list(dict.fromkeys(tokens))

    def _text_matches_category(self, text: str, categories: tuple[str, ...]) -> bool:
        if not text or not categories:
            return False

        tokens = self._category_tokens(categories)
        if not tokens:
            return False

        normalized_text = normalize_text(text)
        words = set(normalized_text.split())
        return any(token in words for token in tokens)

    def _is_ambiguous_brand(self, brand: str) -> bool:
        compact = normalize_text(brand).replace(" ", "")
        return len(compact) <= AMBIGUOUS_BRAND_MAX_LEN

    def _ocr_variants(self, image: Image.Image) -> List[Image.Image]:
        grayscale = image.convert("L")
        high_contrast = ImageOps.autocontrast(grayscale)
        binary = high_contrast.point(lambda pixel: 255 if pixel > 170 else 0)
        return [image.convert("RGB"), high_contrast.convert("RGB"), binary.convert("RGB")]

    def _extract_ocr_text(self, image: Image.Image) -> str:
        if not self.ocr_enabled:
            return ""

        chunks: List[str] = []
        for variant in self._ocr_variants(image):
            try:
                text = pytesseract.image_to_string(variant, config="--psm 6")
            except Exception as exc:
                log.debug("OCR error: %s", exc)
                return ""
            text = text.strip()
            if text and text not in chunks:
                chunks.append(text)
        return " ".join(chunks)

    def verify(
        self,
        image: Image.Image,
        brand_record: BrandRecord,
        candidate: ImageCandidate,
    ) -> Dict[str, str]:
        ocr_text = self._extract_ocr_text(image)
        ocr_brand_match = self._text_matches_brand(ocr_text, brand_record.brand)

        metadata_blob = " ".join(
            [
                candidate.image_url or "",
                candidate.source_url or "",
                candidate.source_title or "",
            ]
        )
        metadata_brand_match = self._text_matches_brand(metadata_blob, brand_record.brand)
        category_match = self._text_matches_category(
            " ".join([ocr_text, metadata_blob]),
            brand_record.categories,
        )
        ambiguous_brand = self._is_ambiguous_brand(brand_record.brand)

        if ocr_brand_match:
            reason = "ocr_brand+category" if category_match else "ocr_brand"
            return {
                "matched": "true",
                "reason": reason,
                "ocr_text": ocr_text[:250],
                "category_match": str(category_match).lower(),
            }

        if metadata_brand_match and category_match:
            return {
                "matched": "true",
                "reason": "metadata_brand+category",
                "ocr_text": ocr_text[:250],
                "category_match": "true",
            }

        if metadata_brand_match and not ambiguous_brand:
            return {
                "matched": "true",
                "reason": "metadata_brand",
                "ocr_text": ocr_text[:250],
                "category_match": str(category_match).lower(),
            }

        if metadata_brand_match and ambiguous_brand:
            return {
                "matched": "false",
                "reason": "ambiguous_brand_without_category",
                "ocr_text": ocr_text[:250],
                "category_match": str(category_match).lower(),
            }

        return {
            "matched": "false",
            "reason": "brand_not_detected",
            "ocr_text": ocr_text[:250],
            "category_match": str(category_match).lower(),
        }


def download_image(url: str) -> Optional[Image.Image]:
    try:
        response = requests.get(
            url,
            headers=_HEADERS,
            timeout=REQUEST_TIMEOUT,
            stream=True,
        )
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "")
        if "image" not in content_type and not url.lower().endswith(
            (".jpg", ".jpeg", ".png", ".webp")
        ):
            return None

        raw = response.content
        img = Image.open(BytesIO(raw))
        img.verify()
        img = Image.open(BytesIO(raw)).convert("RGB")

        if img.size[0] < MIN_SIZE[0] or img.size[1] < MIN_SIZE[1]:
            return None
        return img
    except (UnidentifiedImageError, requests.RequestException, OSError):
        return None
    except Exception as exc:
        log.debug("Unexpected download error for %s: %s", url, exc)
        return None


class ImageSearcher:
    def __init__(self) -> None:
        self.driver = None
        self.wait = None
        self._start_driver()

    def _start_driver(self) -> None:
        opts = Options()
        if HEADLESS:
            opts.add_argument("--headless=new")
        opts.add_argument("--disable-gpu")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--window-size=1366,900")
        opts.add_argument("--disable-dev-shm-usage")
        opts.add_argument("--lang=en-US,en;q=0.9")
        opts.add_argument(
            "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
        opts.add_experimental_option("excludeSwitches", ["enable-automation"])
        opts.add_experimental_option("useAutomationExtension", False)

        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=opts)
        self.wait = WebDriverWait(self.driver, 10)
        log.info("Chrome WebDriver started.")

    def _restart_driver(self, reason: str) -> bool:
        log.warning("Browser session lost (%s). Restarting Chrome WebDriver...", reason)
        try:
            self.quit()
        except Exception:
            pass

        try:
            self._start_driver()
            return True
        except Exception as exc:
            log.error("Failed to restart Chrome WebDriver: %s", exc, exc_info=True)
            return False

    def _is_session_error(self, exc: Exception) -> bool:
        message = str(exc).lower()
        if isinstance(exc, InvalidSessionIdException):
            return True
        if isinstance(exc, WebDriverException):
            return (
                "invalid session id" in message
                or "session deleted" in message
                or "disconnected" in message
                or "chrome not reachable" in message
            )
        return False

    def _scroll(self, times: int = 4, pause: float = 1.8) -> None:
        for _ in range(times):
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(pause + random.uniform(0.0, 0.8))

    def _dismiss_overlays(self) -> None:
        for xpath_text in ["Accept all", "Accept", "I agree", "Agree"]:
            try:
                button = self.driver.find_element(
                    By.XPATH,
                    f"//button[contains(., '{xpath_text}')]",
                )
                button.click()
                time.sleep(0.8)
                return
            except Exception:
                pass

    def _bing_urls(self, query: str, n: int, allow_retry: bool = True) -> List[ImageCandidate]:
        try:
            search_url = (
                "https://www.bing.com/images/search"
                f"?q={requests.utils.quote(query)}&form=HDRSC2&first=1"
            )
            self.driver.get(search_url)
            time.sleep(random.uniform(2.0, 3.0))
            self._dismiss_overlays()
            self._scroll(times=4, pause=1.5)

            try:
                see_more = self.driver.find_element(By.CLASS_NAME, "btn_seemore")
                self.driver.execute_script("arguments[0].click();", see_more)
                time.sleep(1.5)
                self._scroll(times=2, pause=1.2)
            except Exception:
                pass

            results: List[ImageCandidate] = []
            for elem in self.driver.find_elements(By.CLASS_NAME, "iusc"):
                try:
                    metadata = elem.get_attribute("m")
                    if not metadata:
                        continue
                    data = json.loads(metadata)
                    image_url = data.get("murl", "")
                    if not image_url.startswith("http"):
                        continue
                    results.append(
                        ImageCandidate(
                            image_url=image_url,
                            query=query,
                            engine="bing",
                            source_url=data.get("purl", ""),
                            source_title=data.get("t", "") or elem.get_attribute("aria-label") or "",
                        )
                    )
                except Exception:
                    pass
                if len(results) >= n:
                    break

            log.debug("      Bing -> %s URLs", len(results))
            return results
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            if allow_retry and self._is_session_error(exc) and self._restart_driver("Bing search session expired"):
                return self._bing_urls(query, n, allow_retry=False)
            log.warning("      Bing search failed: %s", exc)
            return []

    def _google_urls(self, query: str, n: int, allow_retry: bool = True) -> List[ImageCandidate]:
        try:
            search_url = (
                "https://www.google.com/search"
                f"?q={requests.utils.quote(query)}&tbm=isch&hl=en&gl=us"
            )
            self.driver.get(search_url)
            time.sleep(random.uniform(2.0, 3.0))
            self._dismiss_overlays()
            self._scroll(times=6, pause=1.5)

            results: List[ImageCandidate] = []
            for selector in ["img.Q4LuWd", "img.YQ4gaf", "img.rg_i", "div.isv-r img"]:
                for img_elem in self.driver.find_elements(By.CSS_SELECTOR, selector):
                    src = img_elem.get_attribute("src") or img_elem.get_attribute("data-src")
                    if (
                        src
                        and src.startswith("http")
                        and "gstatic.com/images" not in src
                        and "encrypted-tbn" not in src
                    ):
                        results.append(
                            ImageCandidate(
                                image_url=src,
                                query=query,
                                engine="google",
                                source_title=img_elem.get_attribute("alt") or "",
                            )
                        )
                if results:
                    break

            deduped: List[ImageCandidate] = []
            seen_urls = set()
            for candidate in results:
                if candidate.image_url not in seen_urls:
                    seen_urls.add(candidate.image_url)
                    deduped.append(candidate)
                if len(deduped) >= n:
                    break

            log.debug("      Google -> %s URLs", len(deduped))
            return deduped
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            if allow_retry and self._is_session_error(exc) and self._restart_driver("Google search session expired"):
                return self._google_urls(query, n, allow_retry=False)
            log.warning("      Google search failed: %s", exc)
            return []

    def _build_queries(self, brand_record: BrandRecord) -> List[str]:
        queries: List[str] = []
        categories = list(brand_record.categories[:MAX_SEARCH_CATEGORIES])

        for category in categories:
            category_query = normalize_category_for_query(category)
            if not category_query:
                continue
            for template in CATEGORY_QUERY_TEMPLATES:
                queries.append(
                    collapse_spaces(template.format(brand=brand_record.brand, category=category_query))
                )

        for template in GENERIC_QUERY_TEMPLATES:
            queries.append(collapse_spaces(template.format(brand=brand_record.brand)))

        return list(dict.fromkeys(query for query in queries if query))

    def get_urls(self, brand_record: BrandRecord, n: int = DOWNLOAD_TARGET) -> List[ImageCandidate]:
        all_candidates: List[ImageCandidate] = []
        seen_urls = set()
        queries = self._build_queries(brand_record)

        for query in queries:
            if len(all_candidates) >= n:
                break

            remaining = n - len(all_candidates)
            log.debug("   Query: '%s' (need %s more)", query, remaining)

            batch = self._bing_urls(query, n=remaining)
            if len(batch) < 10:
                batch.extend(self._google_urls(query, n=remaining - len(batch)))

            for candidate in batch:
                if candidate.image_url in seen_urls:
                    continue
                seen_urls.add(candidate.image_url)
                all_candidates.append(candidate)

            time.sleep(random.uniform(1.5, 2.5))

        return all_candidates[:n]

    def quit(self) -> None:
        try:
            if self.driver is not None:
                self.driver.quit()
        except Exception:
            pass
        finally:
            self.driver = None
            self.wait = None


class BrandImageCrawler:
    def __init__(self) -> None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.clip = CLIPFilter()
        self.verifier = BrandVerifier()
        self.browser = ImageSearcher()
        self.manifest_rows: List[Dict[str, object]] = []

    def _save_progress(self, results: List[Dict[str, object]]) -> None:
        if results:
            summary_path = OUTPUT_DIR / "crawl_summary.csv"
            pd.DataFrame(results).to_csv(summary_path, index=False)
            log.info("Summary -> %s", summary_path)

        if self.manifest_rows:
            manifest_path = OUTPUT_DIR / "crawl_manifest.csv"
            pd.DataFrame(self.manifest_rows).to_csv(manifest_path, index=False)
            log.info("Manifest -> %s", manifest_path)

    def _load_brands(self) -> List[BrandRecord]:
        dataframe = pd.read_csv(KB_PATH, usecols=[BRAND_COL, CATEGORY_COL], low_memory=False)
        dataframe[BRAND_COL] = dataframe[BRAND_COL].fillna("").astype(str).str.strip()
        dataframe[CATEGORY_COL] = dataframe[CATEGORY_COL].fillna("").astype(str).str.strip()

        brand_category_counts: Dict[str, Counter] = defaultdict(Counter)
        for _, row in dataframe.iterrows():
            raw_brand = row.get(BRAND_COL, "")
            brand = "" if pd.isna(raw_brand) else str(raw_brand).strip()
            if not brand or brand.lower() in SKIP_BRANDS or brand.lower() == "nan":
                continue

            raw_category = row.get(CATEGORY_COL, "")
            category_value = "" if pd.isna(raw_category) else str(raw_category).strip()
            categories = split_category_values(category_value)
            if not categories:
                brand_category_counts[brand][""] += 1
                continue

            for category in categories:
                brand_category_counts[brand][category] += 1

        records: List[BrandRecord] = []
        for brand, category_counts in brand_category_counts.items():
            ordered_categories = [
                category
                for category, _ in sorted(
                    category_counts.items(),
                    key=lambda item: (-item[1], item[0]),
                )
                if category
            ]
            primary_category = ordered_categories[0] if ordered_categories else ""
            records.append(
                BrandRecord(
                    brand=brand,
                    categories=tuple(ordered_categories),
                    primary_category=primary_category,
                )
            )

        records.sort(key=lambda record: record.brand.lower())
        log.info("Loaded %s brands from KB: %s", len(records), KB_PATH.name)
        return records

    def _process_brand(self, brand_record: BrandRecord) -> Dict[str, object]:
        brand = brand_record.brand
        folder_name = safe_folder_name(brand)
        folder = OUTPUT_DIR / folder_name
        existing = count_images(folder)
        category_label = brand_record.primary_category or "Unspecified"

        if existing >= TARGET_IMAGES:
            log.info("   SKIP '%s' - %s images already present", brand, existing)
            return {
                "brand": brand,
                "primary_category": category_label,
                "status": "skipped",
                "saved_total": existing,
                "saved_new": 0,
                "download_attempted": 0,
                "clip_rejected": 0,
                "brand_rejected": 0,
                "hash_rejected": 0,
            }

        needed = TARGET_IMAGES - existing
        folder.mkdir(parents=True, exist_ok=True)
        seen_hashes = existing_hashes(folder)

        log.info(
            "   Folder: %s (%s existing, need %s more) | Category: %s",
            folder_name,
            existing,
            needed,
            category_label,
        )
        candidates = self.browser.get_urls(brand_record, n=DOWNLOAD_TARGET)
        log.info("   %s candidate URLs collected", len(candidates))

        saved_new = 0
        attempted = 0
        clip_rejected = 0
        brand_rejected = 0
        hash_rejected = 0

        for candidate in candidates:
            if saved_new >= needed:
                break

            attempted += 1
            image = download_image(candidate.image_url)
            if image is None:
                continue

            image_hash = imagehash.phash(image)
            if is_near_duplicate(image_hash, seen_hashes):
                hash_rejected += 1
                continue

            clip_score = self.clip.score(image)
            if clip_score < CLIP_THRESHOLD:
                clip_rejected += 1
                log.debug(
                    "      CLIP reject (score=%.3f): %s",
                    clip_score,
                    candidate.image_url[:120],
                )
                continue

            verification = self.verifier.verify(image, brand_record, candidate)
            if verification["matched"] != "true":
                brand_rejected += 1
                log.debug(
                    "      Brand reject (%s): %s",
                    verification["reason"],
                    candidate.image_url[:120],
                )
                continue

            seen_hashes.append(image_hash)
            file_index = existing + saved_new + 1
            filename = f"{slugify(folder_name)}_{file_index:03d}.jpg"
            save_path = folder / filename
            image.save(save_path, "JPEG", quality=92)
            saved_new += 1

            self.manifest_rows.append(
                {
                    "brand": brand,
                    "primary_category": brand_record.primary_category,
                    "all_categories": " | ".join(brand_record.categories),
                    "file_name": filename,
                    "saved_path": str(save_path),
                    "image_url": candidate.image_url,
                    "source_engine": candidate.engine,
                    "source_url": candidate.source_url,
                    "source_title": candidate.source_title,
                    "query": candidate.query,
                    "clip_score": round(clip_score, 4),
                    "brand_signal": verification["reason"],
                    "category_match": verification["category_match"],
                    "ocr_text": verification["ocr_text"],
                }
            )
            log.debug(
                "      Saved %s (CLIP=%.3f, brand=%s)",
                filename,
                clip_score,
                verification["reason"],
            )

        total_now = existing + saved_new
        log.info(
            "   Saved %s new images | Total: %s | Attempted: %s | CLIP rejected: %s | Brand rejected: %s | Hash dedup: %s",
            saved_new,
            total_now,
            attempted,
            clip_rejected,
            brand_rejected,
            hash_rejected,
        )

        if total_now < TARGET_IMAGES * 0.5:
            log.warning(
                "   '%s' only reached %s/%s images. Consider adding stronger brand/category-specific queries.",
                brand,
                total_now,
                TARGET_IMAGES,
            )

        return {
            "brand": brand,
            "primary_category": category_label,
            "status": "done",
            "saved_total": total_now,
            "saved_new": saved_new,
            "download_attempted": attempted,
            "clip_rejected": clip_rejected,
            "brand_rejected": brand_rejected,
            "hash_rejected": hash_rejected,
        }

    def run(self) -> None:
        results: List[Dict[str, object]] = []
        interrupted = False
        try:
            brand_records = self._load_brands()
            total = len(brand_records)

            log.info("%s", "=" * 65)
            log.info("BRAND IMAGE CRAWLER - %s brands to process", total)
            log.info("Target  : %s images/brand", TARGET_IMAGES)
            log.info("CLIP    : threshold = %s", CLIP_THRESHOLD)
            log.info("Output  : %s", OUTPUT_DIR)
            log.info("%s", "=" * 65)

            for index, brand_record in enumerate(brand_records, start=1):
                log.info("")
                log.info(
                    "[%3s/%s] %s | Category: %s",
                    index,
                    total,
                    brand_record.brand,
                    brand_record.primary_category or "Unspecified",
                )
                try:
                    results.append(self._process_brand(brand_record))
                except KeyboardInterrupt:
                    interrupted = True
                    log.info("\nInterrupted at brand '%s' - saving progress...", brand_record.brand)
                    break
                except Exception as exc:
                    log.error("   Error on '%s': %s", brand_record.brand, exc, exc_info=True)
                    results.append(
                        {
                            "brand": brand_record.brand,
                            "primary_category": brand_record.primary_category or "Unspecified",
                            "status": "error",
                            "saved_total": 0,
                            "saved_new": 0,
                            "download_attempted": 0,
                            "clip_rejected": 0,
                            "brand_rejected": 0,
                            "hash_rejected": 0,
                        }
                    )

                if index < total:
                    time.sleep(random.uniform(3.0, 6.0))

                if index % 25 == 0:
                    gc.collect()
                    log.debug("   GC collect done.")

            if not interrupted:
                df_out = pd.DataFrame(results)
                done = int((df_out["status"] == "done").sum()) if not df_out.empty else 0
                skipped = int((df_out["status"] == "skipped").sum()) if not df_out.empty else 0
                errors = int((df_out["status"] == "error").sum()) if not df_out.empty else 0

                log.info("")
                log.info("%s", "=" * 65)
                log.info("CRAWLER COMPLETE")
                log.info("Done    : %s", done)
                log.info("Skipped : %s", skipped)
                log.info("Errors  : %s", errors)
                if errors > 0 and not df_out.empty:
                    failed = df_out.loc[df_out["status"] == "error", "brand"].tolist()
                    log.info("Failed brands: %s", ", ".join(failed))
                log.info("%s", "=" * 65)
        except KeyboardInterrupt:
            interrupted = True
            log.info("Interrupted before first brand completed.")
        finally:
            self._save_progress(results)
            self.browser.quit()
            log.info("Browser closed.")


if __name__ == "__main__":
    crawler = BrandImageCrawler()
    crawler.run()
