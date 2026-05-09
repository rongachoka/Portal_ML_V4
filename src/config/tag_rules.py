
import re
from typing import Dict, List, Set
import pandas as pd
from pathlib import Path

from Portal_ML_V4.src.config.settings import RAW_DATA_DIR, KB_PATH

# ======================================================
# 1. CONCERN RULES (Acne, Hyperpigmentation, Sleep, etc.)
# ======================================================

# We distinguish between where the text comes from:
# - "product"  -> product name/description on the website
# - "chat"     -> customer chat text (sessions)
#
# "all" rules apply to both if present.

CONCERN_RULES: Dict[str, Dict[str, List[str]]] = {
    "Acne": {
        "all": [
            r"\bacne\b",
            r"\bacnes\b",
            r"\bpimple(s)?\b",
            r"\bbreakout(s)?\b",
            r"\bwhitehead(s)?\b",
            r"\bblackhead(s)?\b",
            r"\bbenzoyl peroxide?\b",
            r"\bsalicylic?\b",
            r"\beffaclar?\b"
        ],
    },
    "Hyperpigmentation": {
        "all": [
            r"\bhyperpigmentation\b",
            r"\bdark spot(s)?\b",
            r"\bdark mark(s)?\b",
            r"\buneven (skin )?tone\b",
            r"\bbrightening\b",
            r"\bascorbic acid\b",
            r"\bspot corrector\b",
            r"\bpigment\b",
        ],
    },
    "Oily Skin": {
        "product": [
            r"\boily skin\b",
            r"\bfor oily\b",
            r"\bcontrols oil\b",
            r"\bsebum\b",
            r"\blarge pores\b",
            r"\bpores\b",

        ],
        "chat": [
            r"\boily skin\b",
            r"\bskin is oily\b",
            r"\boily\b",
            r"\bnormal to oily\b"
            r"\bcombination to oily\b"
            r"\blarge pores\b",
            r"\bpores\b",
        ],
    },
    "Dry Skin": {
        "product": [
            r"\bdry skin\b",
            r"\bfor dry\b",
            r"\bextra dry\b",
            r"\bvery dry\b",
            r"\bintense hydration\b",
        ],
        "chat": [
            r"\bdry skin\b",
            r"\bmy skin is dry\b",
            r"\bfor dry\b",
            r"\bextra dry\b",
            r"\bvery dry\b"
        ],
    },
    "Sensitive Skin": {
        "product": [
            r"\bsensitive skin\b",
            r"\bfor sensitive\b",
            r"\banti-irritation\b",
            r"\bsoothing\b",
        ],
        "chat": [
            r"\bsensitive skin\b",
            r"\bmy skin is sensitive\b",
        ],
    },
    "Sleep": {
        "all": [
            r"\bsleep\b",
            r"\bfor sleep\b",
            r"\binsomnia\b",
            r"\brestless night(s)?\b",
            r"\bcan.?t sleep\b",
            r"\bmagnesium glycinate\b",
        ],
    },
    "Hair Loss": {
        "all": [
            r"\bhair loss\b",
            r"\banti-hairloss\b",
            r"\bthinning hair\b",
            r"\bminoxidil\b",
            r"\balopecia\b",
        ]
    },
    "Weight Management": {
        "chat": [
            r"\bweight loss\b",
            r"\bweight gain\b",
            r"\blose weight\b",
            r"\bgain weight\b",
            r"\bslimming\b",
            r"\bcut belly\b",
            r"\btummy trimmer\b",
            r"\bappetite\b",
            r"\badd weight\b",
            r"\breduce weight\b",
            r"\bfat burner\b",
        ],
    },
    "Eczema": {
        "all": [
            r"\beczema\b",
            r"\batopic dermatitis\b",
            r"\bdermatitis\b",
            r"\bitchy skin\b",
            r"\bred patches\b",
        ],
    },
}


import re
from typing import Set, List, Dict, Any

# ==============================================================================
# 🚀 OPTIMIZATION START: PRE-COMPILE REGEX PATTERNS
# ==============================================================================
# We mirror the structure of CONCERN_RULES but store compiled Regex Objects instead of strings.
print("   ⚙️ Pre-compiling Concern Rules to prevent freezing...")

COMPILED_CONCERN_RULES: Dict[str, Dict[str, List[Any]]] = {}

for concern, src_map in CONCERN_RULES.items():
    COMPILED_CONCERN_RULES[concern] = {}
    
    # Iterate through sources like 'all', 'chat', 'product'
    for source_key, pattern_list in src_map.items():
        compiled_list = []
        for pat in pattern_list:
            try:
                # Compile once, case-insensitive
                compiled_list.append(re.compile(pat, re.IGNORECASE))
            except re.error:
                print(f"   ⚠️ WARN: Skipping invalid regex in {concern}: {pat}")
        
        COMPILED_CONCERN_RULES[concern][source_key] = compiled_list

# ==============================================================================
# 🚀 OPTIMIZATION END
# ==============================================================================

def extract_concerns_from_text(text: str, source: str) -> Set[str]:
    """
    Optimized generic concern extractor using pre-compiled patterns.
    Includes safety truncation to prevent CPU hangs.

    source: "product" for product name/description,
            "chat"    for customer messages/session context.
    """
    if not isinstance(text, str) or not text:
        return set()

    # 🛑 SAFETY TRUNCATION: Prevents "Catastrophic Backtracking" hangs
    # If a log file or massive string accidentally gets here, we chop it.
    if len(text) > 3000:
        t = text[:3000].lower()
    else:
        t = text.lower()

    # Normalize whitespace (replace tabs/newlines with space)
    t = re.sub(r"\s+", " ", t)

    out: Set[str] = set()

    # Loop through the PRE-COMPILED dictionary, not the raw strings
    for concern, src_map in COMPILED_CONCERN_RULES.items():
        patterns: List[Any] = []
        
        # 1. Get patterns that apply everywhere ('all')
        patterns.extend(src_map.get("all", []))
        
        # 2. Get patterns specific to this source ('chat' or 'product')
        patterns.extend(src_map.get(source, []))

        if not patterns:
            continue

        for pat_obj in patterns:
            # Use the compiled object's .search() method (Much faster)
            if pat_obj.search(t):
                out.add(concern)
                break  # Found this concern, move to the next concern

    return out


# ======================================================
# 2. CATEGORY RULES (Product Inquiry - Women's Health, etc.)
# ======================================================

# Canonical category rules keyed by our "Product Inquiry - X" labels.
# Again, we distinguish by text source:
#  - "category" -> site/category strings (for category_functions)
#  - "product"  -> product name/description
#  - "chat"     -> customer chat text (if we ever need that)
CANONICAL_CATEGORY_RULES: Dict[str, Dict[str, List[str]]] = {
    # Supplements
    "Product Inquiry - Supplements": {
        "category": [
            r"\bsupplement(s)?\b",
            r"\bvitamin(s)?\b",
            r"\bminerals?\b",
            r"\bnutritional\b",
        ],
        "product": [
            r"\bsupplement(s)?\b",
            r"\bmultivitamin\b",
            r"\bomega\b",
            r"\bmagnesium\b",
            r"\bzinc\b",
            r"\bvit(amin)? (c|d|b12)\b",
            r"\bcreatine\b",
            r"\bcalcium\b",
        ],
        "chat": [
            r"\bsupplement(s)?\b",
            r"\bmultivitamin\b",
            r"\bomega\b",
            r"\bmagnesium\b",
            r"\bzinc\b",
            r"\bvit(amin)? (c|d|b12)\b",
            r"\bcreatine\b",
            r"\bcalcium\b",
            r"\bbiotin\b",
        ],
    },
    # Skincare
    "Product Inquiry - Skincare": {
        "category": [
            r"\bskin ?care\b",
            r"\bskincare\b",
            r"\bdermocosmetic(s)?\b",
        ],
        "product": [
            r"\bcleanser(s)?\b",
            r"\bface wash\b",
            r"\bfoam(ing)? cleanser\b",
            r"\bmicellar water\b",
            r"\btoner(s)?\b",
            r"\bpeptide(s)?\b",
            r"\bniacinamide\b",
            r"\bretinol\b",
            r"\bserum(s)?\b",
            r"\blotion(s)?\b",
            r"\bcream(s)?\b",
            r"\bmoisturi[sz]er(s)?\b",
            r"\bmoisturi[sz]ing\b",
            r"\bsunscreen(s)?\b",
            r"\bspf\b",
            r"\bspot treatment\b",
        ],
        "chat": [
            r"\bskin ?care\b",
            r"\bskincare\b",
            r"\bskin care routine\b",
            r"\bskin routine\b",
            r"\bface wash\b",
            r"\bcleanser\b",
            r"\bmoisturi[sz]er\b",
            r"\bsunscreen\b",
            r"\bsun cream\b",
            r"\bsunburn\b",
            r"\bspf\b",
            r"\bacne\b",
            r"\bhyperpigmentation\b",
            r"\bpeptide(s)?\b",
            r"\bniacinamide\b",
            r"\bretinol\b",
        ],
    },

    # Women's health
    "Product Inquiry - Women's Health": {
        "category": [
            r"\bwomen'?s health\b",
            r"\bfeminine care\b",
            r"\bintimate (care|hygiene)\b",
            r"\blad(y|ies) care\b",
        ],
        "product": [
            # general “for women”
            r"\bwellwoman\b",
            r"\bwell woman\b",
            r"\bfor women\b",
            r"\bfor ladies\b",
            r"\bwomen'?s\b",
            r"\bwomens\b",
            r"\blad(y|ies)\b",
            r"\bfemale\b",
            r"\bfeminine\b",

            # menstrual products
            r"\bpad(s)?\b",
            r"\bpant(y|ie)\s*liner(s)?\b",
            r"\btampon(s)?\b",
            r"\bmenstrual (cup|cups|disc|discs)\b",
            r"\bperiod (pain|cramps?|cramping)\b",

            # intimate / vaginal care
            r"\b(vaginal|intimate)\s+(wash|gel|cream|moisturizer|moisturiser)\b",
            r"\bfeminine wash\b",
            r"\bfeminine hygiene\b",
            r"\bintimate wash\b",
            r"\bintimate hygiene\b",
            r"\bph\b.*\b(vaginal|intimate)\b",

            # infections / dryness
            r"\byeast infection\b",
            r"\bthrush\b",
            r"\bcandida\b",
            r"\bvaginal dryness\b",
            r"\bvaginal (itch|itching|burning|irritation)\b",
            r"\bvaginal (odou?r|smell|discharge)\b",

            # menopause support etc. (optional but useful)
            r"\bmenopause\b",
            r"\bmenopausal\b",
            r"\bperimenopause\b",
        ],
        "chat": [
            # general women-focus
            r"\bfor women\b",
            r"\bfor ladies\b",
            r"\bfeminine\b",
            r"\bwomen'?s\b",
            r"\bwomens\b",

            # menstruation
            r"\bperiod(s)?\b",
            r"\bthat time of the month\b",
            r"\bmenstruation\b",
            r"\bperiod (pain|cramps?|cramping)\b",
            r"\bsevere cramps\b",
            r"\bheavy (flow|bleeding)\b",
            r"\birregular period(s)?\b",

            # products: pads / tampons
            r"\bpad(s)?\b",
            r"\bpant(y|ie)\s*liner(s)?\b",
            r"\btampon(s)?\b",
            r"\bmenstrual (cup|cups|disc|discs)\b",

            # vaginal / intimate symptoms
            r"\bvaginal dryness\b",
            r"\bdryness during sex\b",
            r"\bvaginal (itch|itching|burning|irritation)\b",
            r"\bvaginal (odou?r|smell|discharge)\b",
            r"\bfishy smell\b",
            r"\bsmell down there\b",
            r"\byeast infection\b",
            r"\bthrush\b",
            r"\bcandida\b",
             r"\bboric acid\b",

            # UTIs (if you want them under women's health)
            r"\buti\b",
            r"\burinary tract infection\b",
            r"\bcystitis\b",
            r"\bburning (when|while)?\s*pee(ing)?\b",
            r"\bpain(ful)? urination\b",

            # pregnancy-related (only if you don’t have a separate Pregnancy category)
            r"\bpregnancy test\b",
            r"\bmissed period\b",
            r"\blate period\b",
        ],
    },

    # Men care
    "Product Inquiry - Men Care": {
        "category": [
            r"\bmen'?s health\b",
            r"\bmen care\b",
            r"\bmale health\b",
            r"\bgents'?\s+care\b",
        ],
        "product": [
            # General men
            r"\bfor men\b",
            r"\bmale\b",
            r"\bgents\b",
            r"\bfor him\b",
            r"\bmen'?s (wash|cleanser|lotion|cream|shampoo|conditioner)\b",

            # Beard / shaving
            r"\bbeard (oil|balm|wash|conditioner|growth)\b",
            r"\baftershave\b",
            r"\brazor (bump|burn)s?\b",
            r"\bingrown hair\b",
            r"\bshaving (cream|gel|foam)\b",

            # Hair loss
            r"\breceding hairline\b",
            r"\bbald(ing|ness)\b",
            r"\bdht blocker\b",

            # Vitality / testosterone
            r"\btestosterone\b",
            r"\btest booster\b",
            r"\bt-?booster\b",
            r"\bmale vitality\b",
            r"\bstamina for men\b",

            # Prostate
            r"\bprostate\b",
            r"\bbph\b",
            r"\bsaw palmetto\b",
            r"\bprostate support\b",

            # Sexual health (clinical)
            r"\blow libido\b",
            r"\bmen'?s libido\b",
            r"\bperformance support\b",
            r"\ber(ection)?\s+difficult(y|ies)\b",
            r"\bpremature ejaculation\b",
            r"\bdelay spray\b",

            # Fitness (men-marketed)
            r"\bmen'?s protein\b",
            r"\bmen'?s creatine\b",
            r"\bfor men gym\b",
        ],
        "chat": [
            # General men
            r"\bfor men\b",
            r"\bfor him\b",
            r"\bgents\b",
            r"\bmen'?s\b",
            r"\bmale\b",

            # Hair
            r"\breceding hairline\b",
            r"\bbald spot\b",

            # Grooming
            r"\bbeard\b",
            r"\baftershave\b",
            r"\bingrown hair\b",


            # Prostate
            r"\bprostate\b",
            r"\bbph\b",

            # Testosterone
            r"\blow testosterone\b",
            r"\btestosterone\b",
        ],
    },

    # Baby Care
    "Product Inquiry - Baby Care": {
        "product": [
            r"\bbaby\b",
            r"\bnewborn\b",
            r"\binfant\b",
            r"\btoddler\b",
            r"\bdiaper\b",
            r"\bnappy\b",
            r"\bbaby milk\b",
            r"\binfant formula\b",
            r"\bbaby lotion\b",
            r"\baptamil\b",
            r"\bpampers\b",
            r"\bmolfix\b",
        ],
        "chat": [
            r"\bmy baby\b",
            r"\bbaby\b",
            r"\bnewborn\b",
            r"\binfant\b",
            r"\btoddler\b",
            r"\baptamil\b",
            r"\bpampers\b",
            r"\bmolfix\b",
            r"\bdiapers\b",
        ],
        "category": [
            r"\bbaby care\b",
            r"\bbaby products\b",
            r"\binfant care\b",
        ],
    },

    # Haircare
    "Product Inquiry - Haircare": {
        "product": [
            r"\bshampoo\b",
            r"\bconditioner\b",
            r"\bhair (oil|mask|serum|spray|cream)\b",
            r"\banti-dandruff\b",
            r"\bdandruff\b",
            r"\bminoxidil\b",
        ],
        "chat": [
            r"\bshampoo\b",
            r"\bconditioner\b",
            r"\bhair (oil|mask|serum|spray|cream)\b",
            r"\banti-dandruff\b",
            r"\bdandruff\b",
            r"\bminoxidil\b",
        ]

    },

    # Perfumes / fragrance
    "Product Inquiry - Perfumes": {
        "product": [
            r"\bperfume\b",
            r"\bfragrance\b",
            r"\bparfum\b",
            r"\body mist\b",
            r"\bdeodorant\b",
        ],
        "chat": [
            r"\bperfume\b",
            r"\bfragrance\b",
            r"\bparfum\b",
            r"\body mist\b",
            r"\bdeodorant\b",
        ],
    },

    # HOmeopathy / Natural Remedies
    "Product Inquiry - Homeopathy": {
        "category": [
            r"\bhomeopathy\b", r"\bnatural products\b", r"\bflower remed(y|ies)\b", 
            r"\brescue remedy\b", r"\bach flower\b", r"\bboiron\b", r"\barnica\b",
            r"\bremed(y|ies)\b"
        ],
        "product": [
            r"\bhomeopathy\b", r"\bnatural products\b", r"\brescue remedy\b", 
            r"\bach flower\b", r"\barnica\b", r"\baessentioal oil\b",
            r"\bremed(y|ies)\b"
        ],
        "chat": [
            r"\bhomeopathy\b", r"\bnatural medicine\b", r"\bherbal\b", 
            r"\brescue remedy\b", r"\barnica\b", r"\baessentioal oil\b",
            r"\bremed(y|ies)\b"
        ]
    },

    # Medicine / Prescription Drugs
    "Product Inquiry - Medicine": {
        "chat": [
            # High-Signal Medical Terms ONLY
            r"\bprescription\b",
            r"\bprescriptions?\b",
            r"\bmedicine\b",
            r"\bdrugs?\b",
            r"\bmedication\b",
            # Specific Formats that usually imply medicine (not supplements)
            r"\binjection\b",
            r"\binhaler\b",
            r"\bsyrup\b",        # Cough syrups (Supplements rarely use syrup)
            r"\bantibiotic\b",
            r"\bantifungal\b",
            r"\bvaccine\b",
            r"\bpain killer\b",
            r"\bpain relief\b",
            r"\bhistamine\b",
            r"\binhaler\b",
        ],
        "product": [
            r"\bprescription\b",
            r"\bmedicine\b",
            r"\binjection\b",
            r"\bantibiotic\b",
            r"\bpharmacy only\b", # POM
        ],
        "category": [
            r"\bmedicine\b",
            r"\bprescription\b",
            r"\bdrugs\b",
            r"\bpharmacy\b",
        ]
    },

    # Location/ HOurs
    "Inquiry - Location and Hours": {
        "chat": [
            # --- OPENING HOURS (Time-based questions) ---
            r"\bopening hours\b",
            r"\bworking hours\b",
            r"\bwhen do you open\b",
            r"\bwhen do you close\b",
            r"\bopen on (sunday|saturday|weekend)s?\b",
            
            # --- GENERAL SHOP LOCATION QUESTIONS (Must imply "YOUR" shop) ---
            r"\bwhere are you\b",
            r"\bwhere are you located\b",
            r"\bwhere is (the|your) shop\b",
            r"\bshop location\b",
            r"\bvisit your shop\b",
            r"\bphysical shop\b",
            r"\bphysical location\b",
            
            # --- SPECIFIC BRANCH MENTIONS (Your 4 Main Hubs) ---
            # We match these specifically to catch "Are you at Two Rivers?"
            r"\btwo rivers\b",
            r"\babc place\b",
            r"\babc branch\b",
            r"\bngong milele\b",
            r"\bmilele mall\b",
            r"\bbanda street\b",
            r"\bbanda st\b",
            r"\bgalleria mall\b",
            
            # (Optional: Add other known store names if you have them, e.g. "Galleria")
        ],
    },

    # Stock Availability
    "Inquiry - Stock Availability": {
        "chat": [
            r"\bdo you have (this|it)\b",
            r"\bdo you stock\b",
            r"\bis (it|this) in stock\b",
            r"\bhave (it|this) in stock\b",
            r"\bis (it|this) available\b",
            r"\bavailability\b",
            # "Available" can be noisy (e.g. "delivery available"), so we are stricter:
            r"\bproduct available\b",
        ],
    },

    # Insurance Questions
    "Inquiry - Insurance": {
        "chat": [
            r"\binsurance\b",
            r"\binsurance cover\b",
            r"\bpay via insurance\b",
            r"\bmedical cover\b",
            
            # Common Kenyan Insurers (High signal)
            r"\bjubilee\b",
            r"\baar\b",
            r"\bapa\b",
            r"\bminet\b",
            r"\bold mutual\b",
            r"\britam\b",
            r"\bcigna\b",
            r"\bfirst assurance\b",
            r"\bheritage\b",
            r"\bmadison\b",
        ],
    },

    # STANLEY CUPS (Stand-alone Tag)
    "Stanley Cups": {
        "chat": [
            r"\bstanley\b",
            r"\bstanley cup\b",
        ],
    },


    "Inquiry - Product Recommendation": {
        "chat": [
            r"\brecommend\b",
            r"\bsuggest\b",
            r"\bwhat is good for\b",
            r"\bsomething for\b",
            r"\badvice on\b",
            r"\bhelp me with\b",
            r"\bbest product for\b",
            r"\broutine for\b",
        ],
    },

    "Inquiry - Skin Consultation": {
        "chat": [
            r"\bskin analysis\b",
            r"\bconsultation\b",
            r"\bdermatologist\b",
            r"\bexamine my skin\b",
        ],
    },

    # ========================================================
    # RENAMED: OTHERS (Catch-all for Overalls, Jobs, Packaging)
    # ========================================================
    "Others": {
        "chat": [
            # Non-Pharmacy Items
            r"\boverall(s)?\b",
            r"\bdust\s?coat\b",
            r"\buniform\b",
            r"\bppe\b",
            r"\bgloves\b", 
            
            # Business/Admin
            r"\bpackaging\b",
            r"\bjob\b",
            r"\bvacancy\b",
            r"\bhiring\b",
            r"\bsupplier\b",
        ],
    },
    # You can add more canonical categories here over time
}


def enrich_canonical_categories_from_text(
    text: str,
    existing: Set[str] | None,
    source: str,
) -> Set[str]:
    """
    Given some text (category string, product text, or chat text) and a
    set of existing canonical categories, add any extra categories that
    match CANONICAL_CATEGORY_RULES for the given source.

    source: "category", "product", or "chat".
    """
    if not isinstance(text, str):
        text = ""

    t = text.lower()
    t = re.sub(r"\s+", " ", t)

    cats: Set[str] = set(existing or set())
    for canon_label, src_map in CANONICAL_CATEGORY_RULES.items():
        patterns: List[str] = []
        patterns.extend(src_map.get(source, []))
        patterns.extend(src_map.get("all", []))  # optional shared
        if not patterns:
            continue

        for pat in patterns:
            if re.search(pat, t):
                cats.add(canon_label)
                break

    return cats


# ======================================================
# DYNAMIC KB CONCERN EXTENSION
# Loads all unique concerns from the Knowledge Base and
# adds them to CONCERN_RULES so they're detectable in chat.
# ======================================================



def _normalise_concern(c: str) -> str:
    """Title-case and strip whitespace from a concern string."""
    return " ".join(c.strip().split()).title()

CONCERN_EXCLUSIONS = {
    "General", "General Care", "", "Nan", "None"
}

try:

    _KB_PATH = KB_PATH

    if _KB_PATH.exists():
        _kb_df = pd.read_csv(_KB_PATH)
        
        # Split on both comma and period, explode, normalise
        _kb_concerns = (
            _kb_df['Concerns']
            .dropna()
            .str.replace(r'[,.]', ',', regex=True)  # normalise separators
            .str.split(',')
            .explode()
            .str.strip()
            .apply(_normalise_concern)
            .unique()
            .tolist()
        )

        _added = 0
        for _concern in _kb_concerns:
            if _concern in CONCERN_EXCLUSIONS:
                continue
            if _concern not in CONCERN_RULES:
                # Build word boundary pattern from the concern name
                _pattern = r'\b' + re.escape(_concern.lower()) + r'\b'
                CONCERN_RULES[_concern] = {"all": [_pattern]}
                _added += 1

        print(f"   ✅ Extended CONCERN_RULES with {_added} KB concerns "
              f"({len(CONCERN_RULES)} total)")
    else:
        print(f"   ⚠️ KB not found at {_KB_PATH} — using hardcoded concerns only")

except Exception as _e:
    print(f"   ⚠️ Could not extend concerns from KB: {_e}")